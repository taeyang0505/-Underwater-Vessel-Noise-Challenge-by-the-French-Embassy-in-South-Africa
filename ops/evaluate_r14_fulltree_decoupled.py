"""Fit-only finite regression/classifier coupling of frozen R14 learners.

No model is fitted here. fit opens validation_fit only, after both full
trees and the anchor model freeze. evaluate reads aligned stored17 panels
only after selected_policy.json is durable. Exposed panels never select.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/r14_fulltree_decoupled_v1'
FULL=ROOT/'models/r14_tree_full_r1';ANCHOR=ROOT/'models/r14_anchor_cpu_r1'
STRICT=ROOT/'models/r12_mask_matched_control_r2/tree'
FIT=ROOT/'data/synthetic/r12_cpu_ablation_v3/validation_fit.npz'
REG=(0.,.25,.5,.75,1.);CLASS=('full5252','strict673','average');LAM=(0.,.25,.5,.75,1.)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def token(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def npz(p):
    import numpy as np
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def auc(y,p):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y)>0,p)) if len(np.unique(np.asarray(y)>0))==2 else None
def blend(cnn,tree,a):
    import numpy as np
    return (a*np.asarray(cnn,float)+(1-a)*np.asarray(tree,float)).astype(np.float32)
def score(p,c,l):
    import numpy as np
    return ((1-l)*np.asarray(p,float)+l*np.asarray(c,float)).astype(np.float32)
def metrics(y,p,s):
    import numpy as np
    y,p,s=map(lambda x:np.asarray(x,float),(y,p,s));e=p-y;hi=y>.2
    if p.shape!=y.shape or s.shape!=y.shape or not np.isfinite(p).all() or not np.isfinite(s).all() or ((p<0)|(p>1)|(s<0)|(s>1)).any():raise ValueError('Invalid prediction alignment/range')
    return {'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(abs(e).mean()),'bias':float(e.mean()),
        'auc_share':auc(y,p),'auc_score':auc(y,s),'prediction_mean':float(p.mean()),'target_mean':float(y.mean()),
        'high_y_n':int(hi.sum()),'high_y_bias':float(e[hi].mean()) if hi.any() else None}
def csv_write(p,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def plan():
    if OUT.exists():raise FileExistsError('Fresh predeclared output required')
    separation=FULL/'R12_VALIDATION_FIT_SOURCE_SEPARATION.json';audit=read(separation)
    if (audit['status']!='passed' or audit['primary_ID_intersection'] or audit['background_group_intersection']
        or audit['minimum_900second_support_gap_hours']<24):raise ValueError('TRAIN/fit source separation failed; stop')
    paths=[Path(__file__),separation,FULL/'PRE_FIT_FREEZE.json',ANCHOR/'ALL_MODELS_FROZEN.json',
        ANCHOR/'anchor_c0/summary.json',ANCHOR/'anchor_c0/best.pt',ANCHOR/'anchor_c0/validation_fit_predictions.npz',
        STRICT/'summary.json',STRICT/'FROZEN_BEFORE_HELDOUT.json',STRICT/'presence.txt',STRICT/'validation_fit_predictions.npz',FIT,
        ROOT/'data/synthetic/r12_cpu_ablation_v3/validation_fit_metadata.parquet',
        ROOT/'pipeline/trees_ensemble_r13.py',ROOT/'ops/train_r12_lightgbm.py',ROOT/'pipeline/features_r12.py',
        ROOT/'ops/train_r14_anchor_cpu.py',ROOT/'pipeline/features_r14_anchor.py']
    p={'format':'r14_fulltree_decoupled_plan_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'fit_role':'R12 validation_fit1000 only','regression':'a*anchor_c0_CNN+(1-a)*baseline5252Tree',
        'CNN_weight_candidates':list(REG),'classifier_candidates':list(CLASS),'classifier_weight_candidates':list(LAM),
        'regression_selection':'minimum fit RMSE, first canonical tie','score_selection':'maximum fit AUC after frozen regression choice, class order then lambda order',
        'score_formula':'(1-lambda)*selected_regression_share+lambda*independent_classifier',
        'required_fullTree_global_freeze':'models/r14_tree_full_r1/ALL_TREES_FROZEN.json',
        'required_anchor_global_freeze':'models/r14_anchor_cpu_r1/ALL_MODELS_FROZEN.json',
        'panel_scope':'exact common17 saved predictions; no new panel inference, no parameter choice from panels',
        'postfreeze_all75_finite_combinations_descriptive_only':True,
        'evaluation_targets':{'high_snr_rmse_lt':.145,'unseen_rmse_le':.0527,'unseen_score_auc_ge':.92,'full15_sf_score_auc_ge':.82,'native15_sf_score_auc_ge':.8},
        'source_metadata_audit':audit,'own_fullTree_ES_selection_vs_fit_metadata':{'background_ID_overlap':105,'group_overlap':12},
        'limitations':['anchor checkpoint already selected on same validation_fit; fullTree ES source overlaps105 fit background IDs despite different physical mixtures',
            'Validation_fit is fitting/selection, not independent risk estimation; all17 are development-exposed',
            'Full and compact models change source law,TRAIN rows,PCA and ES observations; no feature-only causal comparison',
            '509 reference recordings are shared, no reference-independent OOF or official Test claim'],
        'source_sha256':{str(n.relative_to(ROOT)):sha(n) for n in paths},'model_fit_calls':0,
        'new_MPS_GPU_Test_network_or_production_approval':False,'production_approved':False}
    p['plan_sha256']=token(p);OUT.mkdir(parents=True);write(OUT/'plan.json',p);return p

def load_plan():
    p=read(OUT/'plan.json')
    if token({k:v for k,v in p.items() if k!='plan_sha256'})!=p['plan_sha256']:raise ValueError('Plan changed')
    for n,h in p['source_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Frozen source changed:'+n)
    return p
def verify_model_freeze():
    f=read(FULL/'ALL_TREES_FROZEN.json')
    if f.get('status')!='completed' or f.get('all_models_frozen') is not True or set(f['models'])!={'baselineP1','highshareP1'}:raise ValueError('Both completed full trees required')
    if token({k:v for k,v in f.items() if k!='freeze_sha256'})!=f['freeze_sha256']:raise ValueError('Full freeze content changed')
    if (FULL/'FAILED.json').exists():raise ValueError('Failed full tree run')
    for v in f['models'].values():
        if sha(ROOT/v['completion_path'])!=v['completion_sha256']:raise ValueError('Completion changed')
        for name,h in v['artifact_sha256'].items():
            if sha(FULL/name)!=h:raise ValueError('Full model changed')
    a=read(ANCHOR/'ALL_MODELS_FROZEN.json')
    if a['status']!='completed' or sha(ANCHOR/'anchor_c0/summary.json')!=a['models']['anchor_c0']['summary_sha256'] or sha(ANCHOR/'anchor_c0/best.pt')!=a['models']['anchor_c0']['checkpoint_sha256']:raise ValueError('Anchor not frozen')
    return {'full_global_freeze_sha256':sha(FULL/'ALL_TREES_FROZEN.json'),'anchor_global_freeze_sha256':sha(ANCHOR/'ALL_MODELS_FROZEN.json')}

def fit():
    import numpy as np
    p=load_plan();frozen=verify_model_freeze()
    if (OUT/'selected_policy.json').exists():raise FileExistsError('Policy is immutable; no fitting repeat')
    # This barrier is durable before the first validation_fit numerical read.
    write(OUT/'FROZEN_BEFORE_FIT.json',dict(frozen,plan_sha256=p['plan_sha256'],no_panel_opened=True))
    from pipeline.trees_ensemble_r13 import PhysicsLGBM
    s=read(FULL/'baselineP1/model/summary.json');tree=PhysicsLGBM.load(FULL/'baselineP1/model',expected_binding=s['training_binding'])
    data=npz(FIT);y=data['share'];ids=data['sample_id'].astype(str)
    if len(y)!=1000 or len(set(ids))!=1000:raise ValueError('Exact unique validation_fit1000 required')
    full=tree.predict(data['db'],batch_size=128)
    anchor=npz(ANCHOR/'anchor_c0/validation_fit_predictions.npz');strict=npz(STRICT/'validation_fit_predictions.npz')
    if not np.array_equal(anchor['target_share'],y) or not np.array_equal(strict['target_share'],y.astype(float)) or not np.array_equal(strict['sample_id'].astype(str),ids):raise ValueError('Fit target/ID order differs')
    # Anchor file contains no IDs. Its source-pinned training code predicts the
    # entire pinned validation NPZ in original order; target parity is additional
    # evidence, not a claim that repeated zero labels prove unique row order.
    cls={'full5252':full['score_classifier'],'strict673':strict['score_classifier']}
    cls['average']=(.5*cls['full5252'].astype(float)+.5*cls['strict673'].astype(float)).astype(np.float32)
    regression=[{'CNN_weight':a,**metrics(y,blend(anchor['share'],full['share'],a),blend(anchor['share'],full['share'],a))} for a in REG]
    best=min(regression,key=lambda r:r['rmse']);reg=blend(anchor['share'],full['share'],best['CNN_weight'])
    choices=[{'classifier':name,'lambda':l,'AUC':auc(y,score(reg,cls[name],l))} for name in CLASS for l in LAM]
    chosen=max(choices,key=lambda r:r['AUC'])
    policy={'format':'r14_fulltree_decoupled_frozen_policy_v1','fit_role':'validation_fit1000_only','CNN_weight':best['CNN_weight'],
        'classifier':chosen['classifier'],'classifier_weight':chosen['lambda'],'regression_fit_rmse':best['rmse'],'score_fit_AUC':chosen['AUC'],
        'plan_sha256':p['plan_sha256'],'freeze_sha256':frozen,'production_approved':False,'panel_values_used_for_selection':False,
        'anchor_fit_order_evidence':'same SHA-pinned validation NPZ, original source-pinned predictor order and exact target vector; anchor NPZ has no per-row IDs',
        'full_baseline_summary_sha256':sha(FULL/'baselineP1/model/summary.json'),'strict673_presence_sha256':sha(STRICT/'presence.txt')}
    policy['policy_sha256']=token(policy)
    csv_write(OUT/'fit_regression_candidates.csv',regression);csv_write(OUT/'fit_score_candidates.csv',choices)
    np.savez_compressed(OUT/'fit_inputs.npz',sample_id=ids,target_share=y,CNN=anchor['share'],fullTree=full['share'],
        full_classifier=cls['full5252'],strict_classifier=cls['strict673'],selected_share=reg,selected_score=score(reg,cls[chosen['classifier']],chosen['lambda']))
    write(OUT/'selected_policy.json',policy);return policy

def evaluate():
    import numpy as np
    p=load_plan();frozen=verify_model_freeze();policy=read(OUT/'selected_policy.json');policy_sha=sha(OUT/'selected_policy.json')
    if token({k:v for k,v in policy.items() if k!='policy_sha256'})!=policy['policy_sha256'] or policy['freeze_sha256']!=frozen:raise ValueError('Frozen policy changed')
    if (OUT/'RESULTS.json').exists():raise FileExistsError('One-shot evaluation; no repeat or panel fitting')
    sources={'full':ROOT/'reports/r14_tree_full_r1','anchor':ROOT/'reports/r14_anchor_cpu_r1','strict':ROOT/'reports/r12_mask_matched_evaluation_r2'}
    receipts={k:read(v/'RESULTS.json') for k,v in sources.items()}
    if receipts['full']['status']!='completed' or receipts['full']['global_freeze_file_sha256']!=frozen['full_global_freeze_sha256']:raise ValueError('Full saved17 evaluation is incomplete')
    files=sorted((sources['full']/'predictions').glob('*.npz'))
    if len(files)!=17:raise ValueError('Exactly common17 required')
    write(OUT/'FROZEN_BEFORE_PANELS.json',{'selected_policy_file_sha256':policy_sha,'source_result_sha256':{k:sha(v/'RESULTS.json') for k,v in sources.items()},'parameters_refit':False})
    selected=[];allrows=[];align=[];(OUT/'predictions').mkdir(exist_ok=False)
    for path in files:
        if sha(OUT/'selected_policy.json')!=policy_sha:raise ValueError('Policy changed during evaluation')
        arrays={}
        for origin,folder in sources.items():
            file=folder/'predictions'/path.name;key='outputs' if origin=='anchor' else 'output_sha256'
            if sha(file)!=receipts[origin][key][str(file.relative_to(folder))]:raise ValueError('Saved panel artifact changed')
            arrays[origin]=npz(file)
        d=arrays['full'];y=d['target_share'];ids=d['sample_id'].astype(str)
        for v in arrays.values():
            if not np.array_equal(v['target_share'],y) or not np.array_equal(v['sample_id'].astype(str),ids):raise ValueError('Panel ID/target alignment failed')
        tree=d['share'][:,d['model_order'].astype(str).tolist().index('FullTree_baselineP1')]
        a=arrays['anchor'];cnn=a['share'][:,a['model_order'].astype(str).tolist().index('CNN_c0')]
        cls={'full5252':d['baselineP1_classifier'],'strict673':arrays['strict']['extra_tree_classifier']}
        cls['average']=(.5*cls['full5252'].astype(float)+.5*cls['strict673'].astype(float)).astype(np.float32)
        reg=blend(cnn,tree,policy['CNN_weight']);sc=score(reg,cls[policy['classifier']],policy['classifier_weight'])
        selected.append({'panel':path.stem,'model':'fit_selected',**metrics(y,reg,sc)})
        for weight in REG:
            pp=blend(cnn,tree,weight)
            for name in CLASS:
                for l in LAM:allrows.append({'panel':path.stem,'CNN_weight':weight,'classifier':name,'lambda':l,**metrics(y,pp,score(pp,cls[name],l))})
        np.savez_compressed(OUT/'predictions'/path.name,sample_id=ids,target_share=y,share=reg,score=sc,CNN=cnn,fullTree=tree,
            full_classifier=cls['full5252'],strict_classifier=cls['strict673'])
        align.append({'panel':path.stem,'n':len(y),'ID_and_targets_exact_match':True,'input_sha256':{k:sha(v/'predictions'/path.name) for k,v in sources.items()}})
    def requirements(rows):
        r={x['panel']:x for x in rows};t=p['evaluation_targets']
        return {'high_snr':r['R12_high_snr_long']['rmse']<t['high_snr_rmse_lt'],
            'unseen_rmse':r['R10_unseen__r9_complete']['rmse']<=t['unseen_rmse_le'],
            'unseen_auc':r['R10_unseen__r9_complete']['auc_score']>=t['unseen_score_auc_ge'],
            'full15_sf':r['R11_full_smooth15_VS_st_francis_dense']['auc_score']>=t['full15_sf_score_auc_ge'],
            'native15_sf':r['R11_native_long15_VS_st_francis_dense']['auc_score']>=t['native15_sf_score_auc_ge']}
    candidate_pass=[]
    for weight in REG:
        for name in CLASS:
            for l in LAM:
                rr=[v for v in allrows if v['CNN_weight']==weight and v['classifier']==name and v['lambda']==l]
                checks=requirements(rr);candidate_pass.append({'CNN_weight':weight,'classifier':name,'lambda':l,'all_pass':all(checks.values()),**checks})
    checks=requirements(selected);csv_write(OUT/'selected_results.csv',selected);csv_write(OUT/'all75_descriptive_results.csv',allrows);csv_write(OUT/'all75_feasibility.csv',candidate_pass)
    result={'status':'completed','selected_policy':policy,'selected_policy_file_sha256':policy_sha,'selected_policy_requirements':checks,
        'selected_policy_all5_pass':all(checks.values()),'descriptive_all75_pass_count':sum(v['all_pass'] for v in candidate_pass),
        'no_policy_reselection_from_descriptive75_or17panels':True,'panel_alignment':align,'plan_sha256':p['plan_sha256'],
        'production_approved':False,'new_model_fit_calls':0,'new_panel_inference_calls':0,'validation_fit_fullTree_inference_rows':1000,
        'limitations':p['limitations'],'original_models_policy_sources_unchanged':True,'Test_MPS_GPU_network':False}
    load_plan();verify_model_freeze();result['output_sha256']={str(v.relative_to(OUT)):sha(v) for v in OUT.rglob('*') if v.is_file()}
    write(OUT/'RESULTS.json',result);return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['plan','fit','evaluate']);a=ap.parse_args()
    from ops.train_r12_lightgbm import environment,guard
    environment(2);sys.addaudithook(guard)
    result=globals()[a.action]();print(json.dumps(result,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
