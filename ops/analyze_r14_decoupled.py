"""R14 saved-prediction-only decomposition; no model inference or Test I/O.

Fit all new gating/score choices on the original validation_fit1000 only.
Freeze the policy before reading numeric prediction panels.  The historical
panels are development-exposed; a passing diagnostic is not a blind test.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
R13=ROOT/'reports/r13_cpu_correction_r1'
R12=ROOT/'reports/r12_cpu_ablation_r1'
MATCH=ROOT/'reports/r12_mask_matched_evaluation_r2'
TREE=ROOT/'models/r13_compact_trees_v1'
CNN_BASE=ROOT/'models/r12_mask_matched_control_r2/cnn'
CNN_HIGH=ROOT/'models/r13_cpu_cnn_r1/highshareP1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def token(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def rows_csv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def csv_write(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def auc(y,p):
    z=np.asarray(y)>0
    return float(roc_auc_score(z,p)) if len(np.unique(z))==2 else None
def metrics(y,p,score):
    y=np.asarray(y,np.float64);p=np.asarray(p,np.float64);score=np.asarray(score,np.float64)
    assert y.shape==p.shape==score.shape and np.isfinite(p).all() and np.isfinite(score).all()
    assert np.all((p>=0)&(p<=1)) and np.all((score>=0)&(score<=1))
    e=p-y;hi=y>.2
    return dict(n=len(y),rmse=float(np.sqrt(np.mean(e*e))),mae=float(abs(e).mean()),bias=float(e.mean()),
        auc_share=auc(y,p),auc_score=auc(y,score),target_mean=float(y.mean()),prediction_mean=float(p.mean()),
        high_y_n=int(hi.sum()),high_y_bias=float(e[hi].mean()) if hi.any() else None,
        high_y_rmse=float(np.sqrt(np.mean(e[hi]**2))) if hi.any() else None,
        score_ge099=float((score>=.99).mean()),share_ge099=float((p>=.99).mean()))

def regression_candidates():
    c=[dict(name='linear_high'+str(a),kind='linear',high_weight=a) for a in (0.,.25,.5,.75,1.)]
    for low in (0.,.25):
        for high in (.5,.75,1.):
            for width in (.03,.05,.1):
                c.append(dict(name=f'smooth_low{low}_high{high}_width{width}',kind='smooth',
                    threshold=.2,width=width,low=low,high=high))
    return c
def blend(base,high,c):
    base=np.asarray(base,np.float64);high=np.asarray(high,np.float64)
    if c['kind']=='linear':g=np.full(len(base),c['high_weight'])
    else:
        # Monotone gate in base prediction; the final prediction itself need
        # not be globally monotone when the two regressors disagree.
        g=c['low']+(c['high']-c['low'])/(1+np.exp(-np.clip((base-c['threshold'])/c['width'],-40,40)))
    p=((1-g)*base+g*high).astype(np.float32)
    assert np.all(p>=np.minimum(base,high)-1e-7) and np.all(p<=np.maximum(base,high)+1e-7)
    return p
def score_mix(p,c,lam):return ((1-lam)*np.asarray(p,np.float64)+lam*np.asarray(c,np.float64)).astype(np.float32)

def source_files():
    paths=[Path(__file__),R13/'RESULTS.json',R13/'results.csv',R13/'selected_policy.json',R13/'FROZEN_BEFORE_GATE.json',
        R12/'RESULTS.json',R12/'results.csv',MATCH/'RESULTS.json',MATCH/'results.csv',
        ROOT/'reports/R12_TRAIN_MASK_PROVENANCE_AUDIT.json',
        ROOT/'data/synthetic/r12_mask_matched_control_v2/audit.json',
        ROOT/'data/synthetic/r13_cpu_correction_v1/manifest.json',
        ROOT/'models/r13_cpu_cnn_r1/PRE_FIT_FREEZE.json',ROOT/'models/r13_cpu_cnn_r1/ALL_CNN_FROZEN.json',
        TREE/'ALL_TREES_FROZEN.json',ROOT/'models/r12_mask_matched_control_r2/ALL_CONTROL_FROZEN.json',
        ROOT/'ops/train_r12_lightgbm.py',ROOT/'pipeline/features_r12.py',ROOT/'ops/evaluate_r13_cpu.py']
    paths+=list((R13/'predictions').glob('*.npz'))
    for p in (CNN_BASE,CNN_HIGH,TREE/'baselineP1',TREE/'highshareP1'):
        paths += [p/'summary.json',p/'validation_fit_predictions.npz']
    for name in ('L1H0P1','L1H1P1'):
        paths += [ROOT/'models/r12_compact_trees_v1'/name/'summary.json',
            ROOT/'models/r12_compact_trees_v1'/name/'fit_plan.json']
    paths += [ROOT/'models/r12_mask_matched_control_r2/tree/summary.json',
              ROOT/'models/r12_mask_matched_control_r2/tree/fit_plan.json']
    paths += [TREE/name/'fit_plan.json' for name in ('baselineP1','highshareP1')]
    return list(dict.fromkeys(paths))

def fit_policy(out,plan):
    t={name:npz(TREE/name/'validation_fit_predictions.npz') for name in ('baselineP1','highshareP1')}
    cb=npz(CNN_BASE/'validation_fit_predictions.npz');ch=npz(CNN_HIGH/'validation_fit_predictions.npz')
    y=t['baselineP1']['target_share'];ids=t['baselineP1']['sample_id']
    assert len(y)==1000 and len(np.unique(ids))==1000
    assert np.array_equal(y,t['highshareP1']['target_share']) and np.array_equal(ids,t['highshareP1']['sample_id'])
    cf=read(ROOT/'models/r13_cpu_cnn_r1/ALL_CNN_FROZEN.json')
    assert cf['all_models_frozen'] and cf['models']['baselineP1']['training_arrays_exact_same']
    assert sha(CNN_BASE/'summary.json')==cf['models']['baselineP1']['reused_summary_sha256']
    for name in ('baselineP1','highshareP1'):
        s=read(TREE/name/'summary.json');p=TREE/name/'validation_fit_predictions.npz'
        assert s['status']=='completed' and not s['fixture_only']
        assert s['artifact_sha256'][p.name]==sha(p)
        assert s['validation_fit_rows']==1000
        assert s['training_binding']['validation_fit_npz_sha256']=='d25446f1e088a29e9e4a26649813b5be33ee4ca5f1231be9ace554454d2e2562'
    assert read(CNN_BASE/'summary.json')['epochs_run']==read(CNN_HIGH/'summary.json')['epochs_run']==100
    for a in (cb,ch):assert len(a['share'])==1000
    base=(.5*cb['share'].astype(float)+.5*t['baselineP1']['share'].astype(float)).astype(np.float32)
    high=(.5*ch['share'].astype(float)+.5*t['highshareP1']['share'].astype(float)).astype(np.float32)
    cls={'base':t['baselineP1']['score_classifier'],'high':t['highshareP1']['score_classifier']}
    cls['avg']=(.5*cls['base'].astype(float)+.5*cls['high'].astype(float)).astype(np.float32)
    candidates=regression_candidates();fitrows=[]
    for c in candidates:
        p=blend(base,high,c);fitrows.append(dict(candidate=c['name'],**metrics(y,p,p)))
    chosen=min(range(len(candidates)),key=lambda i:fitrows[i]['rmse'])
    p=blend(base,high,candidates[chosen]);score_choices=[]
    for name in ('base','high','avg'):
        for lam in (0.,.25,.5,1.):
            score_choices.append(dict(classifier=name,lam=lam,auc=auc(y,score_mix(p,cls[name],lam))))
    # First canonical tie, not a panel-label-derived choice.
    sc=max(score_choices,key=lambda v:v['auc'])
    policy=dict(format='r14_decoupled_saved_predictions_policy_v1',fit_role='validation_fit1000_only',
        regression=candidates[chosen],classifier=sc['classifier'],classifier_score_weight=sc['lam'],
        regression_base_definition='0.5 CNN baselineP1 + 0.5 Tree baselineP1, stored FP32',
        regression_high_definition='0.5 CNN highshareP1 + 0.5 Tree highshareP1, stored FP32',
        selection='min validation_fit RMSE then max validation_fit score AUC; first canonical tie',
        gate_panel_not_used_to_select=True,panels17_not_used_to_fit=True,plan_sha256=plan['plan_sha256'],
        validation_fit_prediction_sha256={str(p.relative_to(ROOT)):sha(p) for p in (CNN_BASE/'validation_fit_predictions.npz',CNN_HIGH/'validation_fit_predictions.npz',TREE/'baselineP1/validation_fit_predictions.npz',TREE/'highshareP1/validation_fit_predictions.npz')},
        validation_fit_CNN_order_evidence='Same immutable validation_fit NPZ SHA and original predictor order; CNN NPZ has no IDs, tree NPZ provides IDs/targets. No new replay inferred.',
        development_not_blind=True,Test_used=False,production_approved=False)
    policy['policy_sha256']=token(policy);write(out/'selected_policy.json',policy)
    csv_write(out/'validation_fit_regression_candidates.csv',fitrows)
    csv_write(out/'validation_fit_score_candidates.csv',score_choices)
    np.savez_compressed(out/'validation_fit_saved_inputs.npz',sample_id=ids,target_share=y,base_share=base,high_share=high,
        classifier_base=cls['base'],classifier_high=cls['high'],selected_share=p,
        selected_score=score_mix(p,cls[policy['classifier']],policy['classifier_score_weight']))
    return policy

def provenance(out):
    relevant=[]
    for origin,folder in [('R12_original',R12),('R12_strict_matched',MATCH),('R13',R13)]:
        for r in rows_csv(folder/'results.csv'):
            if r['panel'] not in ('R11_full_smooth15_VS_st_francis_dense','R11_native_long15_VS_st_francis_dense') or not r['model'].startswith('Tree_'):continue
            relevant.append(dict(origin=origin,source=str((folder/'results.csv').relative_to(ROOT)),**r))
    csv_write(out/'classifier_provenance_rows.csv',relevant)
    quotes={}
    for value in (.944494,.950814,.760763,.770705):
        choices=[r for r in relevant if r.get('auc_classifier') not in ('',None)]
        r=min(choices,key=lambda x:abs(float(x['auc_classifier'])-value))
        quotes[str(value)]=dict(model=r['model'],panel=r['panel'],origin=r['origin'],output='independent Tree classifier probability',
            classifier_auc=float(r['auc_classifier']),share_auc=float(r['auc_share']),hybrid_score_auc=float(r['auc_score']),source=r['source'],
            absolute_difference_from_rounded_quote=abs(float(r['auc_classifier'])-value))
    summaries={}
    for label,path in [('R12_L1H1P1',ROOT/'models/r12_compact_trees_v1/L1H1P1'),
          ('R12_L1H0P1',ROOT/'models/r12_compact_trees_v1/L1H0P1'),
          ('R12_strict',ROOT/'models/r12_mask_matched_control_r2/tree'),
          ('R13_baselineP1',TREE/'baselineP1'),('R13_highshareP1',TREE/'highshareP1')]:
        s=read(path/'summary.json');f=read(path/'fit_plan.json')
        summaries[label]=dict(summary_path=str((path/'summary.json').relative_to(ROOT)),summary_sha256=sha(path/'summary.json'),
            feature_count=s['feature_count'],train_rows=s['train_rows'],validation_fit_rows=s['validation_fit_rows'],
            best_regression_iteration=s['best_iteration'],best_classifier_iteration=s['classifier_best_iteration'],
            settings=s['settings'],training_binding=s['training_binding'],
            TRAIN_typed_share_sha256=f['input_identity']['train']['share'],
            TRAIN_typed_db_sha256=f['input_identity']['train']['db'],
            validation_fit_typed_sha256=f['input_identity']['validation_fit'])
    a=read(ROOT/'reports/R12_TRAIN_MASK_PROVENANCE_AUDIT.json');m=read(ROOT/'data/synthetic/r12_mask_matched_control_v2/audit.json')
    confound={name:{k:r[k] for k in ('AUC_negative_Nobs','site_x_Nobs_by_presence')} for name,r in a['records'].items() if name in ('L1_H0_TRAIN12000','L1_H1_TRAIN12000')}
    result=dict(rounded_quote_matches=quotes,model_provenance=summaries,original_H1_mask_site_confound=confound,
        strict_matching_preserved_site_effective_mask_target=m['all_rows_site_and_exact_mask_and_targets_unchanged'],
        strict_matching_preserved_sensor_recorder=m['all_rows_sensor_and_recorder_unchanged'],
        strict_replacements=m['actual_replacements'],remaining_limit=m['remaining_limit'],
        causal_limit='Same-panel same-output comparisons are descriptive. Highshare manipulation, feature response, random fit variation and known mask/site confounding are not separately identified causes of official errors.')
    write(out/'classifier_provenance.json',result)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',default='reports/r14_decoupled_v1');a=ap.parse_args()
    out=(ROOT/a.output).resolve()
    if not out.is_relative_to(ROOT) or out.exists():raise FileExistsError('Fresh project-contained diagnostic output only')
    paths=source_files();pins={str(p.relative_to(ROOT)):sha(p) for p in paths};out.mkdir();(out/'predictions').mkdir()
    plan=dict(format='r14_decoupled_predeclared_plan_v1',created_unix=time.time(),source_file_sha256=pins,
        regression_candidates=regression_candidates(),classifier_candidates=['base','high','avg'],score_classifier_weights=[0.,.25,.5,1.],
        fitting_role='original validation_fit1000 only; no gate or 17-panel parameter fitting',
        regression_selection='minimum fit RMSE; first canonical tie',score_selection='maximum fit AUC; classifier base/high/avg then lambda0/.25/.5/1 first tie',
        panels='all existing R13 fixed17 prediction panels, without new inference or panel selection',
        targets=dict(high_snr_rmse_lt=.145,unseen_rmse_le=.0527,unseen_score_auc_ge=.92,full15_sf_score_auc_ge=.82,native15_sf_score_auc_ge=.80),
        prior_inspection='Schema/dtype/model_order inspected before this plan; no numeric panel summary or labels used to choose candidates.',
        inference=0,model_fit=0,Test=False,network=False,production_approved=False,development_not_blind=True)
    plan['plan_sha256']=token(plan);write(out/'predeclared_plan.json',plan)
    policy=fit_policy(out,plan);policyfile=sha(out/'selected_policy.json')
    # First numeric panel read follows policy serialization.
    rows=[];primary=[];aligned=[];candidates=regression_candidates()
    original=read(R13/'RESULTS.json')
    for path in sorted((R13/'predictions').glob('*.npz')):
        assert sha(out/'selected_policy.json')==policyfile
        assert sha(path)==original['output_sha256'][str(path.relative_to(R13))]
        d=npz(path);names=d['model_order'].astype(str).tolist();y=d['target_share'];ids=d['sample_id'];panel=path.stem
        assert len(np.unique(ids))==len(ids)
        base=d['share'][:,names.index('Blend50_baselineP1_lambda0.0')]
        high=d['share'][:,names.index('Blend50_highshareP1_lambda0.0')]
        cl={'base':d['baselineP1_classifier'],'high':d['highshareP1_classifier']}
        cl['avg']=(.5*cl['base'].astype(float)+.5*cl['high'].astype(float)).astype(np.float32)
        reg=np.column_stack([blend(base,high,c) for c in candidates]);classifier=np.column_stack([cl[k] for k in ('base','high','avg')])
        for j,c in enumerate(candidates):
            for cn in ('base','high','avg'):
                for lam in (0.,.25,.5,1.):
                    rows.append(dict(panel=panel,model=c['name']+'__'+cn+'__lambda'+str(lam),regression=c['name'],classifier=cn,lam=lam,
                        **metrics(y,reg[:,j],score_mix(reg[:,j],cl[cn],lam))))
        j=next(i for i,c in enumerate(candidates) if c==policy['regression']);p=reg[:,j];score=score_mix(p,cl[policy['classifier']],policy['classifier_score_weight'])
        primary.append(dict(panel=panel,model='R14_fit_only_selected',**metrics(y,p,score)))
        for name in ('R13_selected','R12_matched_share_only','R9_frozen_all5'):
            if name in names:rows.append(dict(panel=panel,model=name,**metrics(y,d['share'][:,names.index(name)],d['score'][:,names.index(name)])))
        np.savez_compressed(out/'predictions'/path.name,sample_id=ids,target_share=y,regression_order=np.asarray([c['name'] for c in candidates]),
            regression_share=reg,classifier_order=np.asarray(['base','high','avg']),classifier_probability=classifier,
            selected_share=p,selected_score=score)
        aligned.append(dict(panel=panel,n=len(ids),source_sha256=sha(path)))
    assert len(primary)==17
    prov=provenance(out);csv_write(out/'results.csv',rows);csv_write(out/'selected_results.csv',primary)
    bypanel={r['panel']:r for r in primary}
    checks=dict(high_snr_rmse_lt145=bypanel['R12_high_snr_long']['rmse']<.145,
        unseen_rmse_le0527=bypanel['R10_unseen__r9_complete']['rmse']<=.0527,
        unseen_score_auc_ge92=bypanel['R10_unseen__r9_complete']['auc_score']>=.92,
        full15_sf_score_auc_ge82=bypanel['R11_full_smooth15_VS_st_francis_dense']['auc_score']>=.82,
        native15_sf_score_auc_ge80=bypanel['R11_native_long15_VS_st_francis_dense']['auc_score']>=.80)
    for n,h in pins.items():assert sha(ROOT/n)==h,n
    assert sha(out/'selected_policy.json')==policyfile
    result=dict(format='r14_decoupled_saved_prediction_results_v1',status='completed',plan_sha256=plan['plan_sha256'],
        selected_policy=policy,selected_policy_file_sha256=policyfile,selected_results=primary,target_checks=checks,
        simultaneous_target_checks_passed=all(checks.values()),candidate_count=len(candidates)*3*4,metric_rows=len(rows),
        source_file_sha256=pins,panels=aligned,original_inputs_unchanged=True,inference=0,model_fit=0,
        limitations=['All panels were exposed by historical development. New parameters use fit1000 only but this is not pristine blind validation.',
            'The sigmoid high-weight is monotone in base prediction; the final output is not mathematically guaranteed monotone when two models disagree.',
            'No claim that highshare is physically corrupted, PCA removes between-band magnitude, or a local diagnostic identifies official failure causes.',
            'Score classifier probability concerns controlled added-recording presence, not naturally ship-free ground truth.',
            'The chosen policy is one predeclared fit-only selection, and is not changed to whichever recipe passes a development panel.'])
    result['output_sha256']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    write(out/'RESULTS.json',result)
    print(json.dumps(dict(status='completed',output=str(out),policy=policy['regression'],classifier=policy['classifier'],lam=policy['classifier_score_weight'],checks=checks),ensure_ascii=False))

if __name__=='__main__':main()
