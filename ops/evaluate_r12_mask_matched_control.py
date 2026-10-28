"""Separate two-model matched-HF follow-up; original8recipe stays unchanged.

Only after BOTH the original17-model evaluation and the strict-v2 matched
control finish may evaluation run. No new fitting, Test, external actions,
original policy edits or retrospective inclusion in the original selector.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTROL_VERSION='r12_site_sensor_recorder_mask_matched_HF_control_v2'
CONTROL_MANIFEST_SHA='ef8a08e159f3abab0fb2b7ff66f01cddaf4b5e2a88984b33a5722f9d2157143e'
EXTRA_NAMES=('CNN_L1HM1P1','Tree_L1HM1P1','Blend50_L1HM1P1')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def token(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def local(p):
    p=Path(p);p=ROOT/p if not p.is_absolute() else p
    if p.is_symlink():raise ValueError('Symlinkinputnotallowed')
    p=p.resolve()
    if ROOT not in p.parents:raise ValueError('Project-containedpathrequired')
    return p
def rel(p):return str(local(p).relative_to(ROOT))
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def verify_pins(pins):
    for n,h in pins.items():
        if sha(local(n))!=h:raise AssertionError('Changedpinnedfile:'+n)
def array_sha(value):
    import numpy as np
    a=np.ascontiguousarray(value);h=hashlib.sha256(json.dumps({'dtype':str(a.dtype),'shape':list(a.shape)},sort_keys=True).encode());h.update(a.tobytes());return h.hexdigest()
def complete(path):
    try:return read(path).get('status')=='completed'
    except (FileNotFoundError,json.JSONDecodeError):return False
def csvfile(path,rows):
    names=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)

def prepare(a):
    """Metadata/SHA only; neither model nor evaluation arrays are opened."""
    path=local(a.plan)
    if path.exists():raise FileExistsError('Preserveexistingfollowupplan')
    ctrl=local(a.control_data);m=read(ctrl/'manifest.json');ready=read(ctrl/'readiness.json')
    if sha(ctrl/'manifest.json')!=CONTROL_MANIFEST_SHA or m['version']!=CONTROL_VERSION or ready.get('ready_to_train') is not True:
        raise ValueError('Onlystrictsite/sensor/recorder/maskv2controlapproved')
    names=[__file__,'ops/evaluate_r12_ablation.py','ops/verify_r12_evaluation.py','ops/train_r12_mask_matched_control_v2.py',
        'ops/build_r12_mask_matched_control_v2.py','ops/train_r12_lightgbm.py','pipeline/training_r12.py','pipeline/features_r12.py',
        ctrl/'manifest.json',ctrl/'readiness.json','reports/R12_EVALUATION_PREDECLARED_PLAN.json']
    pins={rel(n):sha(local(n)) for n in names}
    value={'format':'r12_matched_control_followup_eval_plan_v1','created_unix':time.time(),
        'control_data':rel(ctrl),'control_root':rel(a.control_root),'main_results':rel(a.main_results),'global_freeze':rel(a.global_freeze),
        'source_pins':pins,'required_control_manifest_sha256':CONTROL_MANIFEST_SHA,
        'execution_barrier':'completed original17evaluation AND complete strictv2 CNN+tree receipt',
        'models':list(EXTRA_NAMES),'fixed_blend_weights':{'CNN':.5,'tree':.5},'extra_CNN_and_blend_score':'share',
        'comparators':['R9_frozen_all5','Blend50_L1H0P1','Blend50_L1H1P1','original_selected_Blend50'],
        'panel_protocol':'exact17originalevaluationpanels; verify target/ID/order against original stored predictions',
        'policy':'followup report only; excluded from original8candidate gate selection; no reweight/retrain',
        'metrics':['RMSE','MAE','bias','mean_RMS','share_and_score_and_classifier_AUC','zero_bias','contrast_ge099','high_y_SSE'],
        'CI':{'repetitions':500,'unit':'same original known background_group/source_id fallback','method':'independent weighted cluster bootstrap'},
        'no_Test_inference_fitting_original_artifact_mutation':True,'development_not_blind':True}
    value['plan_sha256']=token(value);write(path,value);return value

def check_extra(plan,original):
    """Source/checkpoint binding checks only; no TRAIN arrays are materialized."""
    import numpy as np,torch
    ctrl=local(plan['control_data']);root=local(plan['control_root']);m=read(ctrl/'manifest.json');ready=read(ctrl/'readiness.json')
    if (sha(ctrl/'manifest.json')!=plan['required_control_manifest_sha256'] or m['version']!=CONTROL_VERSION
        or ready.get('ready_to_train') is not True or ready['manifest_sha256']!=sha(ctrl/'manifest.json')
        or token({k:v for k,v in m.items() if k!='data_hash'})!=m['data_hash'] or ready['data_hash']!=m['data_hash']):raise AssertionError('Controlreadiness/sourceSHAfailed')
    verify_pins(m['input_sha256'])
    for n,h in m['file_sha256'].items():
        if Path(n).is_absolute() or '..' in Path(n).parts or sha(ctrl/n)!=h:raise AssertionError('Controlartifactchanged')
    audit=m['audit']
    for k in ['positive_common_field_bytes_unchanged','all_rows_site_and_exact_mask_and_targets_unchanged','all_rows_sensor_and_recorder_unchanged','known_zero_energy_q_and_share_exact']:
        if audit[k] is not True:raise AssertionError('Matchedsourceauditfailed:'+k)
    if (audit['actual_replacements']!=1199 or audit['actual_CNN_replacements']!=599 or audit['shortfall']!=1
        or any(v['AUC_delta']!=0 for v in audit['before_after'].values())):raise AssertionError('Approvedreplacement/maskconditiondiffers')
    if (root/'FAILED.json').exists():raise AssertionError('Failedextraorbitcannotbeevaluated')
    allf=read(root/'ALL_CONTROL_FROZEN.json');binding=read(root/'PRE_FIT_FREEZE.json')
    if allf.get('status')!='completed' or allf.get('epochs_run')!=100 or allf.get('original8policy_changed') is not False or binding!=allf['binding']:
        raise AssertionError('Additionalcontrolcompletionbarrierfailed')
    dm=read(local(original['data_dir'])/'manifest.json');basis=sha(local(original['basis_path']))
    expected={'control_manifest_sha256':sha(ctrl/'manifest.json'),'original_data_hash':dm['data_hash'],
        'original_data_manifest_sha256':sha(local(original['data_dir'])/'manifest.json'),'basis_sha256':basis,
        'script_sha256':sha(ROOT/'ops/train_r12_mask_matched_control_v2.py')}
    if any(binding.get(k)!=v for k,v in expected.items()) or m['parent_data_hash']!=dm['data_hash'] or m['basis_training_binding']!=dm['basis_training_binding']:
        raise AssertionError('Extraoriginalsource/PCA/trainerbindingdiffers')
    expectedpaths={rel(root/k/'summary.json') for k in ('cnn','tree')}
    if set(allf['model_summary_sha256'])!=expectedpaths:raise AssertionError('Extra2summarymapdiffers')
    verify_pins(allf['model_summary_sha256']);nn=read(root/'cnn/summary.json');tree=read(root/'tree/summary.json')
    if any(s.get('status')!='completed' for s in (nn,tree)):raise AssertionError('Twoextramodelsmustcomplete')
    hist=[json.loads(v) for v in (root/'cnn/epochs.jsonl').read_text().splitlines()]
    if [v['epoch'] for v in hist]!=list(range(1,101)):raise AssertionError('ExtraCNN100epochsnotcomplete')
    values=np.asarray([v['validation_fit_rmse'] for v in hist],float)
    if not np.isfinite(values).all():raise AssertionError('Nonfiniteextratraininghistory')
    best=int(np.argmin(values))+1
    wanted={'name':'L1HM1P1','long':True,'HF':True,'shape':True,'legacy_control':False}
    if (nn['epochs_run']!=100 or nn['parameters']!=26953 or nn['early_stopping'] is not False or nn['optimizer']!='Adam'
        or nn['arm']!=wanted or nn['best_epoch']!=best or nn['score']!='share'):raise AssertionError('ExtraCNNarchitecture/policydiffers')
    if sha(root/'cnn/best.pt')!=nn['checkpoint_sha256'] or allf['CNN_checkpoint_sha256']!=nn['checkpoint_sha256']:raise AssertionError('ExtraCNNcheckpointchanged')
    ck=torch.load(root/'cnn/best.pt',map_location='cpu',weights_only=True)
    if ck['arm']!=wanted or ck['epoch']!=best or ck['parameters']!=26953 or sum(v.numel() for v in ck['state_dict'].values())!=26953:raise AssertionError('ExtraCNNsavedstatecountdiffers')
    if (tree.get('fixture_only') is not False or tree['training_binding']!=binding or tree['include_shape'] is not True
        or tree['feature_count']!=673 or tree['train_rows']!=12000 or tree['validation_fit_rows']!=1000 or tree['basis_training_binding']!=dm['basis_training_binding']):raise AssertionError('ExtraTreebinding/featurecountdiffers')
    for k,h in allf['tree_booster_sha256'].items():
        if sha(root/'tree'/k)!=h or tree['artifact_sha256'][k]!=h:raise AssertionError('ExtraTreeboosterchanged')
    pins={rel(root/'ALL_CONTROL_FROZEN.json'):sha(root/'ALL_CONTROL_FROZEN.json'),rel(root/'PRE_FIT_FREEZE.json'):sha(root/'PRE_FIT_FREEZE.json'),
        rel(root/'cnn/best.pt'):sha(root/'cnn/best.pt'),rel(root/'cnn/epochs.jsonl'):sha(root/'cnn/epochs.jsonl'),**allf['model_summary_sha256']}
    return allf,binding,nn,tree,ck,pins

class Predictor:
    """db-only inference; labels/metadata never enter the model call."""
    def __init__(self,original,plan,binding,checkpoint,threads,batch):
        import torch
        from pipeline.training_r12 import R12MaskCNN
        from pipeline.features_r12 import load_spectral_basis
        from ops.train_r12_lightgbm import CompactPhysicsLGBM
        torch.set_num_threads(threads);self.batch=batch
        dm=read(local(original['data_dir'])/'manifest.json')
        self.basis=load_spectral_basis(local(original['basis_path']),expected_sha256=binding['basis_sha256'],expected_training_binding=dm['basis_training_binding'])
        self.nn=R12MaskCNN();self.nn.load_state_dict(checkpoint['state_dict'],strict=True);self.nn.eval()
        if not all(torch.isfinite(v).all() for v in self.nn.parameters()):raise FloatingPointError('Nonfiniteadditionalweights')
        self.tree=CompactPhysicsLGBM.load(local(plan['control_root'])/'tree',expected_binding=binding)
    def __call__(self,db):
        import numpy as np,torch
        from collections.abc import Mapping
        from pipeline.training_r12 import channels,predict
        if isinstance(db,Mapping):raise TypeError('Predictoracceptsdbonly;notlabels/recipes')
        x,w=channels(db,self.basis,True);r=predict(self.nn,torch.from_numpy(x),torch.from_numpy(w),self.batch,legacy_score=False)
        gap=float(np.max(abs((w*r['cell_mask']).sum((1,2))-r['share'])))
        if gap>3e-6 or np.any(r['cell_mask'][w==0]!=0):raise AssertionError('ExtraCNNphysicalpool/maskfailed')
        t=self.tree.predict(db,batch_size=self.batch);blend=(.5*r['share'].astype(float)+.5*t['share'].astype(float)).astype(np.float32)
        return {'CNN_L1HM1P1':{k:r[k] for k in ('share','score','contrast','cell_mask')},'Tree_L1HM1P1':t,
            'Blend50_L1HM1P1':{'share':blend,'score':blend.copy()}},{'physical_pool_max_error':gap,'input_fields':['db'],
            'input_db_typed_sha256':array_sha(db),'input_db_shape':list(np.asarray(db).shape),'label_fields_passed_to_model':[]}

def enriched_metrics(panel,pred):
    import numpy as np
    from ops.evaluate_r12_ablation import metrics
    r=metrics(panel,pred);y=np.asarray(panel['y'],float);p=np.asarray(pred['share'],float);zero=y==0
    r.update(zero_prediction_mean=float(p[zero].mean()) if zero.any() else None,
        zero_bias=float((p[zero]-y[zero]).mean()) if zero.any() else None,
        zero_rmse=float(np.sqrt(np.mean((p[zero]-y[zero])**2))) if zero.any() else None,
        share_saturation_le001=float(np.mean(p<=.01)),share_saturation_ge099=float(np.mean(p>=.99)))
    return r

def match_original(panel,original_path,policy):
    import numpy as np
    with np.load(original_path,allow_pickle=False) as z:
        if (not np.array_equal(z['sample_id'].astype(str),np.asarray(panel['sample_id']).astype(str))
            or not np.array_equal(z['target_share'],np.asarray(panel['y']))):raise AssertionError('Followuppaneltarget/IDorderdiffersfrommain17')
        order=z['model_order'].astype(str).tolist();share=z['share'];score=z['score']
        wanted={'original_Blend50_L1H0P1':'Blend50_L1H0P1','original_Blend50_L1H1P1':'Blend50_L1H1P1',
                'original_selected_Blend50':'Blend50_'+policy['selected_arm']}
        if 'R9_frozen_all5' in order:wanted['R9_frozen_all5']='R9_frozen_all5'
        result={label:{'share':share[:,order.index(name)].copy(),'score':score[:,order.index(name)].copy(),
                        'mixed_R9_component':name=='R9_frozen_all5'} for label,name in wanted.items()}
    return result

def execute(a):
    from ops import evaluate_r12_ablation as master
    master.environment(a.threads)
    import numpy as np
    from ops.verify_r12_evaluation import cluster_ci
    plan=read(local(a.plan))
    if token({k:v for k,v in plan.items() if k!='plan_sha256'})!=plan['plan_sha256']:raise AssertionError('Followupplancontentschanged')
    verify_pins(plan['source_pins']);mainpath=local(plan['main_results']);extra=local(plan['control_root'])
    if not complete(mainpath) or not complete(extra/'ALL_CONTROL_FROZEN.json'):raise RuntimeError('Bothcompletionconditionsrequired;noevaluationstarted')
    main=read(mainpath);mainout=mainpath.parent;gp=local(plan['global_freeze']);original,f=master.check_freeze(gp)
    if main.get('status')!='completed' or (mainout/'FAILED.json').exists() or main['global_freeze_file_sha256']!=sha(gp):raise AssertionError('Originalevaluationfailed/foreign')
    for n,h in main['output_sha256'].items():
        if Path(n).is_absolute() or '..' in Path(n).parts or sha(mainout/n)!=h:raise AssertionError('Mainoutputchanged')
    policy_path=mainout/'selected_policy.json';policy=master.check_policy(policy_path,f['freeze_sha256'])
    if policy!=main['selected_policy']:raise AssertionError('Originalselectedpolicydiffers')
    allf,binding,nn,tree,checkpoint,controlpins=check_extra(plan,original)
    barriers={rel(mainpath):sha(mainpath),rel(gp):sha(gp),rel(policy_path):sha(policy_path),**controlpins}
    out=local(a.output)
    if out.exists():raise FileExistsError('Freshfollowupevaluationonly;nopartialreuse')
    out.mkdir(parents=True);(out/'predictions').mkdir();started=time.perf_counter()
    write(out/'FROZEN_BEFORE_EXTRA_PREDICTIONS.json',{'plan_file_sha256':sha(local(a.plan)),'barrier_sha256':barriers,
        'source_pins':plan['source_pins'],'original_policy_unchanged':True,'extra_in_original8selector':False,
        'control_manifest_sha256':sha(local(plan['control_data'])/'manifest.json'),'binding':binding})
    predictor=Predictor(original,plan,binding,checkpoint,a.threads,a.batch)
    panels=[master.load_R12(original,'validation_gate')]
    panels.extend(v for v in [master.load_R12(original,role,policy_path,f['freeze_sha256']) for role in ('eval','high_snr_long')] if v is not None)
    dev,r11=master.load_development();panels.extend(dev)
    for positive in master.DEV_R11_POS:
        for negative in master.DEV_R11_NEG:
            x,y=r11[positive],r11[negative]
            panels.append({'name':x['name']+'_VS_'+negative,'stage':'exposed_development','y':np.r_[x['y'],y['y']],
                'db':np.concatenate([x['db'],y['db']]),'sample_id':np.r_[x['sample_id'],y['sample_id']],
                'groups':np.r_[x['groups'],y['groups']],'cluster_key':x['cluster_key'],'site':np.r_[x['site'],y['site']]})
    if len(panels)!=17:raise AssertionError('Same17panelcontractrequired')
    rows=[];cis=[];proofs={}
    for panel in panels:
        name=panel['name'];saved=mainout/'predictions'/(name+'.npz');comparators=match_original(panel,saved,policy)
        pred,proof=predictor(panel['db']);allpred={**pred,**comparators}
        for model,p in allpred.items():rows.append({'panel':name,'model':model,'stage':panel['stage'],'followup_not_original_selector':True,**enriched_metrics(panel,p)})
        selected=pred['Blend50_L1HM1P1']
        for other,values in comparators.items():
            calculated=cluster_ci(name,panel['y'],panel['groups'],panel['cluster_key'],(selected['share'],selected['score']),(values['share'],values['score']))
            cis.extend({'panel':name,'comparison':'matched50_minus_'+other,'metric':m,**v} for m,v in calculated.items())
        arrays={'sample_id':panel['sample_id'],'target_share':panel['y'],'model_order':np.asarray(list(allpred)),
            'share':np.column_stack([p['share'] for p in allpred.values()]),'score':np.column_stack([p['score'] for p in allpred.values()]),
            'extra_cnn_contrast':pred['CNN_L1HM1P1']['contrast'],'extra_cnn_cell_mask':pred['CNN_L1HM1P1']['cell_mask'],
            'extra_tree_classifier':pred['Tree_L1HM1P1']['score_classifier']}
        np.savez_compressed(out/'predictions'/(name+'.npz'),**arrays)
        proofs[name]={**proof,'original_saved_prediction_sha256':sha(saved),'n':len(panel['y']),
            'sample_ID_order_sha256':array_sha(np.asarray(panel['sample_id']).astype(str)),'target_share_sha256':array_sha(np.asarray(panel['y'])),
            'target_ID_order_exact':True,'baseline_predictions_reused_not_reinferred':True}
        print(json.dumps({'panel_completed':name,'matched50_rmse':rows[-len(allpred)+2]['rmse'],'elapsed_seconds':time.perf_counter()-started}),flush=True)
    if len(rows)!=118 or len(cis)!=201:raise AssertionError('Expected118metricrowsand201CIrows')
    csvfile(out/'results.csv',rows);csvfile(out/'paired_cluster_CI.csv',cis);verify_pins(barriers);verify_pins(plan['source_pins']);master.check_freeze(gp)
    result={'format':'r12_mask_matched_followup_evaluation_v2','status':'completed','plan_file_sha256':sha(local(a.plan)),
        'original_selected_arm_unchanged':policy['selected_arm'],'original_selected_policy_sha256':sha(policy_path),
        'additional_control_not_in_original8_selector':True,'no_refit_reweight_or_Test':True,'development_not_blind':True,
        'panels':17,'metric_rows':len(rows),'CI_rows':len(cis),'metrics_csv':'results.csv','CI_csv':'paired_cluster_CI.csv',
        'source_pins':plan['source_pins'],'barrier_sha256':barriers,'per_panel_input_proofs':proofs,
        'control_source_audit':read(local(plan['control_data'])/'manifest.json')['audit'],
        'extra_model_summaries':{'CNN':nn,'tree':tree},'elapsed_seconds':time.perf_counter()-started,
        'limitations':['Follow-upafterdiscoveringmask/classassociation;notanoriginalpreregisteredeighthcandidate.',
            'Exactsite/sensor/recorder/maskmarginalsheldfixed,butbackgroundtextures/sourcegroupdistributionscanstillchange.',
            'Sharedreferencesanddevelopment-exposedpanels;notblindOOForanofficialTestscore.',
            'ConditionalclusterCIsdonotadjustmultiplecomparisonsorfollowupselection.',
            'R9derivedscoreandnewshare-onlyscorearenotthesamescoredefinition.']}
    result['output_sha256']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()};write(out/'RESULTS.json',result)
    lines=['# R12 엄격 matched HF 추가 대조','',
        '원래 17개 모델·8개 recipe 선택 규칙은 그대로 보존했습니다. 원 평가와 추가 CNN/tree가 모두 완료된 뒤 같은 17패널을 평가했습니다. 추가 모델은 원래 gate 후보에 합치지 않았습니다.',
        '추가 TRAIN은 site·sensor·recorder·정확한 관측 minute-mask·양성 q/y와 전체 class 표를 보존했습니다. 1,200개 요청 중1,199개, CNN subset에서는600개 중599개를 바꿨고 부족한 한 행은 원본을 유지했습니다.',
        '', '|패널|모델|RMSE|share AUC|score AUC|bias|zero bias|예측RMS|contrast≥.99|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'undefined' if x is None else f'{x:.9f}'
    for r in rows:lines.append('|'+ '|'.join([r['panel'],r['model']]+[fmt(r[k]) for k in ['rmse','auc_share','auc_score','bias','zero_bias','prediction_rms','contrast_ge099_fraction']])+'|')
    lines+=['', '같은 nuisance 집계량을 보존해도 배경 texture와 source group의 분포까지 완전히 같다는 뜻은 아닙니다. 개발용 후속 대조이며 Test·blind OOF 성능이나 인과 원인의 완전한 증명이 아닙니다.',
        'R9는 원래 혼합 score를 유지하며 추가 CNN/50:50은 share를 score로 사용했습니다. score 정의 차이와 실제 회귀 개선을 구분해야 합니다.',
        '', '[전체 지표·보호 한계·SHA](RESULTS.json) / [paired cluster CI](paired_cluster_CI.csv)']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'status':'completed','output':rel(out),'rows':len(rows),'CIs':len(cis)}));return result

def fixture():
    import numpy as np,tempfile
    from ops.verify_r12_evaluation import pair_auc,cluster_ci
    y=np.asarray([0,0,.1,.2,.4,.6],np.float32);p=np.clip(y+.02,0,1);g=np.asarray(['a','a','b','b','c','c'])
    panel={'name':'fixture','y':y,'sample_id':np.asarray(list('abcdef')),'groups':g,'cluster_key':'numeric','site':np.asarray(['s']*6)}
    r=enriched_metrics(panel,{'share':p,'score':p,'contrast':np.zeros(6)})
    if abs(r['rmse']-np.sqrt(np.mean((p.astype(float)-y)**2)))>1e-12 or r['zero_bias']!=float(p[:2].astype(float).mean()):raise AssertionError('Fixturemetricsfailed')
    c=cluster_ci('fixture',y,g,'numeric',(p,p),(p,p))
    if any(v['point_difference']!=0 or v['lo95']!=0 or v['hi95']!=0 for v in c.values()):raise AssertionError('FixtureidentityCIfailed')
    with tempfile.TemporaryDirectory(prefix='r12_followup_fixture_') as temp:
        file=Path(temp)/'original.npz';order=['Blend50_L1H0P1','Blend50_L1H1P1','Blend50_L0H0P0'];a=np.column_stack([p,p,p])
        np.savez(file,sample_id=panel['sample_id'],target_share=y,model_order=np.asarray(order),share=a,score=a)
        q=match_original(panel,file,{'selected_arm':'L0H0P0'})
        if not np.array_equal(q['original_selected_Blend50']['share'],p):raise AssertionError('Fixturestoredbaselinefailed')
        changed=dict(panel,sample_id=panel['sample_id'][::-1])
        try:match_original(changed,file,{'selected_arm':'L0H0P0'})
        except AssertionError:pass
        else:raise AssertionError('IDordermismatchaccepted')
    return {'status':'passed','numeric_only':True,'metrics_zero_bias_oracle':True,'paired500_identity_CI_zero':True,
        'stored_baseline_ID_exact_and_mismatch_rejected':True,'actual_models_acoustics_Test_read':False,'script_sha256':sha(__file__)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['plan','fixture','evaluate'])
    p.add_argument('--plan',default='reports/R12_MASK_MATCHED_EVALUATION_PLAN.json');p.add_argument('--control-data',default='data/synthetic/r12_mask_matched_control_v2')
    p.add_argument('--control-root',default='models/r12_mask_matched_control_r2');p.add_argument('--main-results',default='reports/r12_cpu_ablation_r1/RESULTS.json')
    p.add_argument('--global-freeze',default='reports/R12_ALL17_GLOBAL_FREEZE.json');p.add_argument('--output',default='reports/r12_mask_matched_evaluation_r2')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--batch',type=int,default=128);p.add_argument('--watch',action='store_true');p.add_argument('--timeout',type=int,default=7200)
    a=p.parse_args()
    from ops.evaluate_r12_ablation import guard
    sys.addaudithook(guard)
    if a.mode=='plan':print(json.dumps(prepare(a)));return
    if a.mode=='fixture':print(json.dumps(fixture()));return
    started=time.monotonic()
    while not (complete(local(a.main_results)) and complete(local(a.control_root)/'ALL_CONTROL_FROZEN.json')):
        if (local(a.main_results).parent/'FAILED.json').exists() or (local(a.control_root)/'FAILED.json').exists():raise RuntimeError('Failedmainorcontrol;preservepartials')
        if not a.watch:print(json.dumps({'status':'waiting_both_completion_barriers','no_inference':True}));raise SystemExit(3)
        if time.monotonic()-started>a.timeout:raise TimeoutError('Boundedwaitexpired;norestart')
        time.sleep(30)
    try:execute(a)
    except BaseException as e:
        out=local(a.output)
        if out.exists() and not (out/'FAILED.json').exists():write(out/'FAILED.json',{'error':repr(e),'no_retraining_or_reweighting':True,'partial_reuse':False})
        raise
if __name__=='__main__':main()
