"""R18 production evaluation, fixed B1/B2 and limited generation mixing.

This module never trains. Phases run in separate processes so Torch and the
macOS OpenMP LightGBM runtime are not loaded into one inference process.
Query bytes are opened once only, after selection is frozen and report done.
"""
from __future__ import annotations
import csv,datetime,gc,hashlib,io,json,os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REL='reports/r18_production_r1/finalization'
FAMILIES=('lightgbm','dinov2_small','dasheng_base')
REGIMES=('D0','H0')
TASKS=('regression','classification')
FIXED=('local_gate','domain_holdout','native12_VS_hf_high_negative','native12_VS_st_francis_dense_negative')
BLENDS={'B1':(.60,.25,.15),'B2':(.50,.30,.20)}
OPERATORS=('control_eq6','mfhf_shelf6','mfhf_shelf12','mfhf_shelf12_lfminus6','alpha_preEQ_inserted','independent_subband_snr')
CODE=('pipeline/r18_production_finalization.py','ops/finalize_r18_production.py')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item()if isinstance(x,np.generic)else str(x))+'\n')
    t.replace(p)

def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    with t.open('wb')as f:np.savez_compressed(f,**d)
    t.replace(p)

def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def path(root,rel):
    p=Path(rel)
    if '..'in p.parts:raise ValueError('No path traversal')
    return p if p.is_absolute()else Path(root)/p

def out(root):return Path(root)/REL

def immutable_json(p,v):
    if Path(p).exists():
        if read(p)!=v:raise ValueError('Immutable artifact differs: '+str(p))
    else:write(p,v)

def plan(root=ROOT):
    root=Path(root);folder=out(root);p=folder/'PLAN.json'
    if p.exists():
        verify_plan(root);return read(p)
    manifest=root/'data/synthetic/r18_production_v1/manifest.json'
    m=read(manifest)
    if m['status']!='completed' or m['Test_used'] or m['KFold']:raise ValueError('Completed no-Test single-split data required')
    fixed=root/'reports/r16_support_temporal_r1/data/manifest.json'
    if read(fixed)['status']!='completed':raise ValueError('Fixed raw panels incomplete')
    legacy_pins=read(root/'reports/r17_eda_audit_20261030/task1/protocol.json')['source_sha256']
    legacy_pins.update(read(root/'reports/r17_eda_audit_20261030/shortcut/tree_sources_sha256.json'))
    preserved={str(p.relative_to(root)):sha(p)for p in(root/'submissions/r17_fixed_blends_20261030').glob('*.csv')}
    protocol={'format':'r18_fixed_blends_finalization_v1','created_utc':utc(),'status':'registered_before_evaluation',
        'member_order':list(FAMILIES),'regimes':list(REGIMES),'D0_H0_weights':[.5,.5],
        'B1':list(BLENDS['B1']),'B2':list(BLENDS['B2']),'score_rule':'Same fixed blend weights over independent classifier probability; share weight exactly 0',
        'candidate_generation_fractions':[0.,.25,.50,.75,1.],
        'candidate_count':10,'fresh_fit_rule':'Average each input geometry equally; choose ONE of ten candidates by fresh_fit RMSE only; deterministic ties prefer old B1 then B1 then smaller new fraction',
        'fresh_fit_dependence':'D0 and H0 input panels have different db/targets although IDs are paired. Score each separately; 400 shared background pairs, not800 independent backgrounds.',
        'gate_rule':{'reference':'old_B1','axes':list(FIXED),'rmse_non_degradation_tolerance':1e-6,
            'require_local_gate_strict_improvement':1e-6,'fallback':'old_B1',
            'score':'Independent probability weights locked to selected regression recipe; AUC is reported, not silently retuned'},
        'postprocessing':'Only elementwise clip[0,1]; no q05/q20 upper bound, average shift, Test matching or new calibration search',
        'report':'Read after DECISION_FREEZE; never changes decision. These fixed panels were repeatedly used historically and are not an untouched new test.',
        'references':'TRAIN465 excludes B19/Cprime25 bytes; shared hull/day and historical legacy exposure remain. clean_ES/fresh_fit share465references but have background-source purge.',
        'legacy_exposure':'Old LGB5 includes folds that may have used fresh-fit background source groups; its comparison is auxiliary. Fixed gate/report had repeated historical development exposure.',
        'query':'One raw CSV byte read after fixed decision and report; all inference receives compact db-only query NPZ; no Test fitting, cohort transforms or label inference',
        'official_upload':False,'new_training':False,'new_KFold':False,
        'data_manifest_path':str(manifest.relative_to(root)),'data_manifest_sha256':sha(manifest),
        'fixed_panel_manifest_path':str(fixed.relative_to(root)),'fixed_panel_manifest_sha256':sha(fixed),
        'legacy_source_sha256':legacy_pins,'original_R17_candidate_sha256':preserved,
        'source_sha256':{n:sha(ROOT/n)for n in CODE}}
    write(p,protocol);return protocol

def verify_plan(root):
    p=read(out(root)/'PLAN.json')
    for n,h in p['source_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Finalizer code changed after registration: '+n)
    for key in ('data_manifest','fixed_panel_manifest'):
        if sha(path(root,p[key+'_path']))!=p[key+'_sha256']:raise ValueError('Data manifest changed')
    return p

def _join(parts):
    common=set.intersection(*(set(p)for p in parts))
    return{k:np.concatenate([p[k]for p in parts])for k in common}

def _compact(d,weights=None):
    wanted=('db','share','presence','sample_id','site','recorder','base_window_index','background_group','operator','active_minutes')
    x={k:d[k]for k in wanted if k in d}
    if 'active_minutes'not in x and 'cell_share'in d:x['active_minutes']=np.any(d['cell_share']>0,2).sum(1)
    if weights is not None:x['weights']=np.asarray(weights,float)
    n=len(x['db'])
    if x['db'].shape!=(n,15,25) or not np.array_equal(x['share']>0,x['presence']>0):raise ValueError('Raw panel labels/schema')
    if len(set(x['sample_id'].astype(str)))!=n:raise ValueError('Repeated sample ID in evaluation panel')
    return x

def raw_fixed(root,name):
    """Pinned immutable NPZ inputs; own phase locks replace legacy run receipt."""
    mp=Path(root)/'reports/r16_support_temporal_r1/data/manifest.json';m=read(mp)
    def get(role,k):
        e=m['long_validation'][k]if role=='long_validation'else m['panels'][role][k]
        p=path(root,e['path'])if e.get('scope')=='project'else path(mp.parent,e['path'])
        if sha(p)!=e['sha256']:raise ValueError('Fixed source panel changed')
        # Avoid E/A/B arrays: only raw observations, targets and scalar metadata.
        with np.load(p,allow_pickle=False)as z:
            keys=('db','cell_share','share','presence','sample_id','site','recorder','base_window_index','background_group','operator','active_minutes')
            return{k:z[k]for k in keys if k in z}
    if name in ('local_gate','local_report','domain_holdout'):
        return _compact(_join([get(name,op)for op in OPERATORS]))
    if name.startswith('native12_VS_'):
        pos=_join([get('long_validation','native12_positive__'+op)for op in OPERATORS])
        neg=get('long_validation',name.split('_VS_',1)[1])
        w=np.r_[np.full(len(pos['share']),.5/len(pos['share'])),np.full(len(neg['share']),.5/len(neg['share']))]
        return _compact(_join([pos,neg]),w)
    raise ValueError(name)

def names(stage):
    if stage=='development':return('fresh_fit_D0','fresh_fit_H0',*FIXED)
    if stage=='report':return('local_report',)
    if stage=='query':return('query',)
    raise ValueError(stage)

def stage_guard(root,stage):
    verify_plan(root)
    if stage in ('report','query'):
        freeze=out(root)/'DECISION_FREEZE.json'
        f=read(freeze)
        if f['PLAN_sha256']!=sha(out(root)/'PLAN.json')or f['policy_sha256']!=sha(out(root)/'selected_policy.json'):raise ValueError('Decision changed')
    if stage=='query':
        report=read(out(root)/'REPORT_DONE.json')
        if report['decision_sha256']!=sha(out(root)/'DECISION_FREEZE.json'):raise ValueError('Report not bound to frozen policy')

def prepare_panels(root=ROOT,stage='development'):
    stage_guard(root,stage)
    if stage=='query':raise ValueError('Use prepare-query after policy/report')
    manifest=read(Path(root)/'data/synthetic/r18_production_v1/manifest.json');dest=out(root)/'inputs';receipts={}
    for name in names(stage):
        if name.startswith('fresh_fit_'):
            r=name.rsplit('_',1)[1];e=manifest['roles']['fresh_fit'][r];p=path(root,e['path'])
            if sha(p)!=e['sha256']:raise ValueError('Fresh fit changed')
            d=_compact(load(p))
        else:d=raw_fixed(root,name)
        target=dest/(name+'.npz')
        if target.exists():
            old=load(target)
            for k,v in d.items():
                same=np.array_equal(old[k],v,equal_nan=True)if v.dtype.kind in 'fc'else np.array_equal(old[k],v)
                if not same:raise ValueError('Previously prepared input differs')
        else:save(target,d)
        receipts[name]={'path':str(target.relative_to(root)),'sha256':sha(target),'n':len(d['db']),'bytes':target.stat().st_size}
    write(out(root)/f'INPUTS_{stage}.json',{'status':'completed','PLAN_sha256':sha(out(root)/'PLAN.json'),'panels':receipts,'Test_used':False})
    return receipts

def dataset(root,name):
    p=out(root)/'inputs'/(name+'.npz');stage='query'if name=='query'else'report'if name=='local_report'else'development'
    stage_guard(root,stage);receipt=read(out(root)/f'INPUTS_{stage}.json')['panels'][name]
    if sha(p)!=receipt['sha256']:raise ValueError('Prepared input changed')
    return load(p)

def pred_path(root,generation,family,name):return out(root)/'predictions'/generation/family/(name+'.npz')

def put_predictions(root,generation,family,name,d,values,model_pins):
    p=pred_path(root,generation,family,name)
    a={'sample_id':d['sample_id'].astype(str),**{k:np.asarray(v,float)for k,v in values.items()}}
    for k,v in values.items():
        if np.asarray(v).shape!=(len(d['db']),)or not np.isfinite(v).all()or np.any((v<0)|(v>1)):raise ValueError('Prediction range/shape')
    if p.exists():
        old=load(p)
        for k,v in a.items():
            if not np.array_equal(old[k],v):raise ValueError('No partial prediction overwrite: '+str(p))
    else:save(p,a)
    write(p.with_suffix('.json'),{'status':'completed','generation':generation,'family':family,'dataset':name,'n':len(d['db']),
        'input_sha256':sha(out(root)/'inputs'/(name+'.npz')),'prediction_sha256':sha(p),'model_sha256':model_pins,
        'feature_masking_at_inference':False,'optimizer_steps':0,'fitting_performed':False,'created_utc':utc()})

def completed_models(root,tree_root=None,neural_root=None):
    tree=Path(tree_root)if tree_root else Path(root)/'models/r18_production_r1/trees'
    neural=Path(neural_root)if neural_root else Path(root)/'models/r18_production_r1/neural'
    pins={};manifest_sha=read(out(root)/'PLAN.json')['data_manifest_sha256']
    for r in REGIMES:
        p=tree/r/'summary.json';s=read(p)
        if s['status']!='completed':raise RuntimeError('Tree incomplete')
        pins[str(p)]=sha(p)
        for n,h in s['artifact_sha256'].items():
            if sha(p.parent/n)!=h:raise ValueError('Tree artifact changed')
            pins[str(p.parent/n)]=h
        for family in FAMILIES[1:]:
            for task in TASKS:
                p=neural/family/r/task/'summary.json';s=read(p)
                if s['status']!='completed'or s['data_manifest_sha256']!=manifest_sha:raise RuntimeError('Neural incomplete/wrong data')
                cp=p.parent/'best.pt';h=s['artifacts']['best.pt']
                if sha(cp)!=h:raise ValueError('Neural best changed')
                pins[str(p)]=sha(p);pins[str(cp)]=h
    return pins

def collect_tree(root=ROOT,stage='development',tree_root=None):
    from pipeline.trees_r18_production import predict
    stage_guard(root,stage);model_root=Path(tree_root)if tree_root else Path(root)/'models/r18_production_r1/trees'
    for name in names(stage):
        d=dataset(root,name);v={};pins={}
        for r in REGIMES:
            folder=model_root/r;s=read(folder/'summary.json')
            if s['status']!='completed':raise RuntimeError('Tree not complete')
            for n,h in s['artifact_sha256'].items():pins[str(folder/n)]=h
            a=predict(folder,d['db'],branches=('selected',))
            v['share_'+r]=a['share_selected'];v['prob_'+r]=a['classifier_prob']
        put_predictions(root,'new','lightgbm',name,d,v,pins)

def _neural_branch_predict(model,d,task,device,batch,*,old=False):
    import torch
    from pipeline.training_r18_production_neural import feature17,autocast
    from pipeline.training_r16_pretrained_cuda_bf16 import CLASSIFIER_CHANNELS
    result=[];model.eval()
    with torch.inference_mode():
        for begin in range(0,len(d['db']),batch):
            x,w=feature17(d['db'][begin:begin+batch])
            if old and task=='classification':x=x[:,CLASSIFIER_CHANNELS]
            x=torch.as_tensor(x.copy(),device=device,dtype=torch.float32)
            with autocast(device):logits=model(x)
            p=logits.float().sigmoid()
            if task=='regression':p=(p*torch.as_tensor(w.copy(),device=device,dtype=torch.float32)).sum((1,2))
            result.append(p.cpu().numpy())
    return np.concatenate(result)

def collect_neural(root=ROOT,stage='development',neural_root=None,device='cuda',batch=16,old=False):
    import torch
    from pipeline import training_r18_production_neural as newapi
    from pipeline import training_r16_pretrained_cuda_bf16 as oldapi
    stage_guard(root,stage);newapi.device_setup(device)
    base=Path(neural_root)if neural_root else Path(root)/('models/r16_pretrained_cuda_bf16_r1/both'if old else'models/r18_production_r1/neural')
    nms=names(stage)
    if old:nms=tuple(n for n in nms if n.startswith('fresh_fit_'))
    if not nms:return
    for family in FAMILIES[1:]:
        values={n:{}for n in nms};pins={}
        for r in REGIMES:
            for task in TASKS:
                folder=base/family/r/task;s=read(folder/'summary.json');cp=folder/'best.pt'
                if s['status']!='completed':raise RuntimeError('Completed frozen neural required')
                if not old and s['data_manifest_sha256']!=read(out(root)/'PLAN.json')['data_manifest_sha256']:raise ValueError('Foreign data checkpoint')
                expected_sha=s['artifact_sha256']['best.pt']if old else s['artifacts']['best.pt']
                if sha(cp)!=expected_sha:raise ValueError('Frozen best checkpoint changed')
                ck=torch.load(cp,map_location='cpu',weights_only=True)
                if (ck['family'],ck['regime'],ck['task'])!=(family,r,task):raise ValueError('Checkpoint job identity')
                if old:
                    model,_=oldapi.branch(family,task,device)
                    fingerprint=oldapi.base_fingerprint;expected=ck['permanent_base_fingerprint']
                else:
                    model,_=newapi.branch(family,task,device);fingerprint=newapi.fingerprint;expected=ck['frozen_base_sha256']
                if fingerprint(model)!=expected:raise ValueError('Public backbone differs')
                model.load_adapter_state(ck['adapter_state']);pins[str(cp)]=sha(cp);pins[str(folder/'summary.json')]=sha(folder/'summary.json')
                for name in nms:
                    d=dataset(root,name);values[name][('share_'if task=='regression'else'prob_')+r]=_neural_branch_predict(model,d,task,device,batch,old=old)
                if fingerprint(model)!=expected:raise ValueError('Frozen base changed during inference')
                del model,ck;gc.collect()
                if device=='cuda':torch.cuda.empty_cache()
        for name in nms:put_predictions(root,'old'if old else'new',family,name,dataset(root,name),values[name],pins)

def collect_legacy_tree(root=ROOT,stage='development',legacy_root=None):
    import lightgbm as lgb
    from pipeline.training_r16_multisupport import tree_features
    stage_guard(root,stage);legacy=Path(legacy_root)if legacy_root else Path(root)
    for name in names(stage):
        if not name.startswith('fresh_fit_'):continue
        d=dataset(root,name);values={};pins={}
        for task in TASKS:
            x=tree_features(d['db'],task)
            for r in REGIMES:
                pp=[]
                for fold in range(5):
                    p=legacy/f'models/r16_multisupport_full_r1/trees/{r}/fold_{fold:02d}/{task}.txt'
                    rel=str(p.relative_to(legacy));expected=read(out(root)/'PLAN.json')['legacy_source_sha256'][rel]
                    if sha(p)!=expected:raise ValueError('Legacy tree changed')
                    model=lgb.Booster(model_file=str(p));pp.append(np.clip(model.predict(x,num_threads=2),0,1));pins[str(p)]=expected
                values[('share_'if task=='regression'else'prob_')+r]=np.mean(pp,axis=0)
        put_predictions(root,'old','lightgbm',name,d,values,pins)

def collect_legacy_saved(root=ROOT,stage='development',legacy_root=None):
    stage_guard(root,stage);legacy=Path(legacy_root)if legacy_root else Path(root)
    indices={'lightgbm':0,'dasheng_base':2,'dinov2_small':3}
    for name in names(stage):
        if name.startswith('fresh_fit_'):continue
        source=legacy/('models/r16_v1_fourway_r2/query_predictions.npz'if name=='query'else f'models/r16_v1_fourway_r2/predictions/{name}.npz')
        expected=read(out(root)/'PLAN.json')['legacy_source_sha256'][str(source.relative_to(legacy))]
        if sha(source)!=expected:raise ValueError('Pinned legacy saved predictions changed')
        old=load(source);d=dataset(root,name)
        if not np.array_equal(old['sample_id'].astype(str),d['sample_id'].astype(str)):raise ValueError('Legacy/raw input ID order differs')
        if 'share'in d and not np.array_equal(old['target_share'],d['share']):raise ValueError('Legacy target differs')
        for family,j in indices.items():
            values={}
            for r in REGIMES:values['share_'+r]=old['share_'+r][:,j];values['prob_'+r]=old['classifier_'+r][:,j]
            put_predictions(root,'old',family,name,d,values,{str(source):sha(source)})

def prediction(root,generation,family,name):
    p=pred_path(root,generation,family,name);receipt=read(p.with_suffix('.json'))
    if receipt['prediction_sha256']!=sha(p)or receipt['input_sha256']!=sha(out(root)/'inputs'/(name+'.npz')):raise ValueError('Compact prediction identity differs')
    a=load(p);d=dataset(root,name)
    if not np.array_equal(a['sample_id'].astype(str),d['sample_id'].astype(str)):raise ValueError('Prediction ID order differs')
    return a

def blend(root,name,which,new_fraction):
    result=[]
    for target in ('share','prob'):
        v=np.zeros(len(dataset(root,name)['db']))
        for generation,gw in [('old',1-new_fraction),('new',new_fraction)]:
            if gw==0:continue
            for family,fw in zip(FAMILIES,BLENDS[which]):
                a=prediction(root,generation,family,name)
                v+=gw*fw*.5*(a[target+'_D0']+a[target+'_H0'])
        result.append(np.clip(v,0,1))
    return tuple(result)

def metric(d,s,c):
    from sklearn.metrics import roc_auc_score
    y=d['share'];w=np.asarray(d.get('weights',np.ones(len(y))),float);w=w/w.sum();error=s-y;z=y>0
    value={'n':len(y),'rmse':float(np.sqrt(w@(error*error))),'bias':float(w@error),'mean':float(w@s),
        'rms':float(np.sqrt(w@(s*s))),'classifier_auc':float(roc_auc_score(z,c,sample_weight=w))if z.any()and(~z).any()else None,
        'share_auc':float(roc_auc_score(z,s,sample_weight=w))if z.any()and(~z).any()else None,
        'score_mean':float(w@c),'score_ge090':float(w@(c>=.9))}
    if 'site'in d:
        sites=d['site'].astype(str);value['site_macro_rmse']=float(np.mean([np.sqrt(np.mean(error[sites==a]**2))for a in np.unique(sites)]))
    for label,ix in [('weak_positive',(y>0)&(y<=.03)),('long_positive',(y>0)&(d.get('active_minutes',np.zeros(len(y)))>=12)),('zero',y==0)]:
        value[label+'_n']=int(ix.sum())
        if ix.any():value[label+'_rmse']=float(np.sqrt(np.mean(error[ix]**2)));value[label+'_bias']=float(error[ix].mean())
    return value

def candidates():
    return[(('old_'if a==0 else'new_'if a==1 else f'multigen_new{int(a*100):02d}_')+b,b,a)for a in (0.,.25,.5,.75,1.)for b in BLENDS]

def compare(root,stage):
    rows=[]
    for name in names(stage):
        d=dataset(root,name)
        for generation in ('old','new'):
            for family in FAMILIES:
                a=prediction(root,generation,family,name)
                for regime in(*REGIMES,'D0H0_50_50'):
                    s=a['share_'+regime]if regime in REGIMES else .5*(a['share_D0']+a['share_H0'])
                    c=a['prob_'+regime]if regime in REGIMES else .5*(a['prob_D0']+a['prob_H0'])
                    rows.append({'panel':name,'generation':generation,'model':family,'regime':regime,**metric(d,s,c)})
    import pandas as pd
    pd.DataFrame(rows).to_csv(out(root)/f'INDIVIDUAL_METRICS_{stage}.csv',index=False)
    mixtures=[]
    for name in names(stage):
        for label,b,a in candidates():mixtures.append({'panel':name,'candidate':label,'blend':b,'new_fraction':a,**metric(dataset(root,name),*blend(root,name,b,a))})
    pd.DataFrame(mixtures).to_csv(out(root)/f'BLEND_METRICS_{stage}.csv',index=False)
    return rows,mixtures

def select(root=ROOT,tree_root=None,neural_root=None):
    verify_plan(root);folder=out(root)
    if (folder/'DECISION_FREEZE.json').exists():raise FileExistsError('Decision already frozen; report may not select again')
    modelpins=completed_models(root,tree_root,neural_root)
    individual,mixtures=compare(root,'development')
    scores=[]
    for label,b,a in candidates():
        m=[r for r in mixtures if r['candidate']==label and r['panel'].startswith('fresh_fit_')]
        assert len(m)==2
        rmse=float(np.sqrt(np.mean([r['rmse']**2 for r in m])))
        scores.append({'candidate':label,'blend':b,'new_fraction':a,'fit_rmse':rmse})
    winner=min(scores,key=lambda r:(round(r['fit_rmse'],12),r['blend']!='B1',r['new_fraction']))
    get=lambda label,axis:next(r for r in mixtures if r['candidate']==label and r['panel']==axis)
    passed=all(get(winner['candidate'],a)['rmse']<=get('old_B1',a)['rmse']+1e-6 for a in FIXED)
    passed=passed and get(winner['candidate'],'local_gate')['rmse']<get('old_B1','local_gate')['rmse']-1e-6
    chosen=winner if passed else next(r for r in scores if r['candidate']=='old_B1')
    policy={'status':'frozen','selected':chosen,'fit_winner':winner,'gate_adopted':bool(passed),'fallback':'old_B1',
        'D0_H0_weights':[.5,.5],'family_order':list(FAMILIES),'family_weights':list(BLENDS[chosen['blend']]),'score_share_weight':0,
        'report_used_for_selection':False,'Test_used_for_selection':False,'fit_candidates':scores,'created_utc':utc()}
    write(folder/'selected_policy.json',policy)
    prediction_pins={str(p.relative_to(folder)):sha(p)for p in(folder/'predictions').rglob('*.npz')if p.stem in names('development')}
    write(folder/'DECISION_FREEZE.json',{'status':'completed','PLAN_sha256':sha(folder/'PLAN.json'),'policy_sha256':sha(folder/'selected_policy.json'),
        'new_models_frozen_sha256':modelpins,'development_prediction_sha256':prediction_pins,'all_models_frozen':True,
        'ensemble_policy_frozen':True,'created_utc':utc()})
    return policy

def report(root=ROOT):
    stage_guard(root,'report');folder=out(root)
    if(folder/'REPORT_DONE.json').exists():raise FileExistsError('Report already finalized')
    individual,mixtures=compare(root,'report')
    policy=read(folder/'selected_policy.json');selected=policy['selected']['candidate']
    summary={'status':'completed','decision_sha256':sha(folder/'DECISION_FREEZE.json'),'selected':selected,
        'fixed_policy_report_metrics':next(r for r in mixtures if r['candidate']==selected),'policy_changed':False,'created_utc':utc()}
    write(folder/'REPORT_DONE.json',summary);return summary

def prepare_query(root=ROOT,test_path=None):
    """A single file open captures both exact Test bytes and SHA, after freeze."""
    stage_guard(root,'query');folder=out(root);receipt=folder/'INPUTS_query.json'
    if receipt.exists():
        r=read(receipt);p=folder/'inputs/query.npz'
        if sha(p)!=r['panels']['query']['sha256']:raise ValueError('Frozen query changed')
        return r
    from ops.predict_r16_production_nn import FEATURE_COLUMNS,raw_db
    p=Path(test_path)if test_path else Path(root)/'data/raw/Test.csv'
    b=p.read_bytes()  # The only raw Test open in this pipeline.
    reader=csv.reader(io.StringIO(b.decode('utf-8-sig')))
    if next(reader,None)!=['ID',*FEATURE_COLUMNS]:raise ValueError('Test schema/order differs')
    ids=[];values=[]
    for r in reader:
        if len(r)!=376 or not r[0]or r[0].strip()!=r[0]:raise ValueError('Malformed query')
        ids.append(r[0]);values.append([float(v)if v.strip()else np.nan for v in r[1:]])
    if len(ids)!=1500 or len(set(ids))!=1500:raise ValueError('Expected1500 unique IDs')
    db=raw_db(np.asarray(values,np.float32).reshape(-1,15,25));target=folder/'inputs/query.npz'
    save(target,{'db':db,'sample_id':np.asarray(ids)})
    result={'status':'completed','decision_sha256':sha(folder/'DECISION_FREEZE.json'),'raw_Test_sha256':hashlib.sha256(b).hexdigest(),
        'raw_Test_open_count':1,'query_labels_used':False,'Test_used_for_fit':False,
        'panels':{'query':{'path':str(target.relative_to(root)),'sha256':sha(target),'n':1500}}}
    write(receipt,result);return result

def stats(a):
    a=np.asarray(a,float);q=np.quantile(a,[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1])
    return{'mean':float(a.mean()),'std':float(a.std()),'rms':float(np.sqrt(np.mean(a*a))),
        **dict(zip(('min','p01','p05','p10','p25','median','p75','p90','p95','p99','max'),map(float,q))),
        'ge090_fraction':float(np.mean(a>=.9))}

def export(root=ROOT):
    stage_guard(root,'query');folder=out(root);policy=read(folder/'selected_policy.json');d=dataset(root,'query')
    dest=Path(root)/'submissions/r18_production_r1';dest.mkdir(parents=True,exist_ok=True)
    specs=[('B1','B1',1.),('B2','B2',1.),('selected',policy['selected']['blend'],policy['selected']['new_fraction'])]
    results=[]
    for label,b,a in specs:
        s,c=blend(root,'query',b,a);p=dest/f'submission_r18_{label}.csv'
        stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n');writer.writerow(['ID','share_added_vessel','score_added_vessel'])
        writer.writerows((sid,format(ss,'.17g'),format(cc,'.17g'))for sid,ss,cc in zip(d['sample_id'],s,c))
        payload=stream.getvalue().encode()
        if p.exists()and p.read_bytes()!=payload:raise FileExistsError('Do not overwrite existing candidate')
        if not p.exists():p.write_bytes(payload)
        with p.open(newline='')as f:
            reader=csv.reader(f);assert next(reader)==['ID','share_added_vessel','score_added_vessel'];rows=list(reader)
        assert[r[0]for r in rows]==d['sample_id'].tolist()
        values=np.asarray([[float(r[1]),float(r[2])]for r in rows])
        assert np.array_equal(values,np.column_stack([s,c]))and np.isfinite(values).all()and np.all((values>=0)&(values<=1))
        results.append({'candidate':label,'blend':b,'new_generation_fraction':a,'path':str(p.relative_to(root)),'sha256':sha(p),'rows':1500,
            'unique_ID_order_verified':True,'finite_and_range_verified':True,'exact_roundtrip_verified':True,
            'share_statistics':stats(s),'score_statistics':stats(c)})
    result={'status':'completed','created_utc':utc(),'PLAN_sha256':sha(folder/'PLAN.json'),'decision_sha256':sha(folder/'DECISION_FREEZE.json'),
        'report_done_sha256':sha(folder/'REPORT_DONE.json'),'candidates':results,'official_upload':False,'new_Public_score':'unmeasured',
        'reference_independence_limitation':read(folder/'PLAN.json')['references'],'historical_validation_exposure':True}
    for name,h in read(folder/'PLAN.json')['original_R17_candidate_sha256'].items():
        original=Path(root)/name
        if original.exists()and sha(original)!=h:raise ValueError('Original R17 CSV changed')
    write(dest/'candidate_manifest.json',result);write(folder/'COMPLETED.json',result)
    import pandas as pd
    dev=pd.read_csv(folder/'BLEND_METRICS_development.csv');rep=pd.read_csv(folder/'BLEND_METRICS_report.csv')
    selected=('old_B1','old_B2','new_B1','new_B2',policy['selected']['candidate'])
    metric_table=pd.concat([dev[dev.candidate.isin(selected)&~dev.panel.str.startswith('fresh_fit_')],rep[rep.candidate.isin(selected)]])
    text='# R18 본체 비교 및 제출 후보\n\n학습 없이 동결 모델을 추론하고 사전 고정된 B1/B2 및 세대 결합만 비교했다. 공식 제출은 수행하지 않았다.\n\n'
    text+='B1: LGB60% + DINO25% + Dasheng15%. B2: LGB50% + DINO30% + Dasheng20%. 각 계열 D0:H0는50:50이다. 독립 분류 확률에도 같은 비율을 쓰며 share 혼합비는0이다.\n\n'
    text+=f"fresh_fit 승자: **{policy['fit_winner']['candidate']}**. gate 채택: **{policy['gate_adopted']}**. 최종 동결 정책: **{policy['selected']['candidate']}**.\n\n"
    text+=metric_table[['panel','candidate','rmse','bias','classifier_auc']].to_markdown(index=False,floatfmt='.6f')+'\n\n'
    text+='local_report는 동결 후 최종 확인에만 사용했으며 정책을 바꾸지 않았다. 다만 과거 여러 차례 개발에 사용된 고정 패널이므로 새롭고 독립적인 테스트라고 부르지 않는다. 구세대5-Fold LGB의 원천 노출과 확대465개 참조의 hull/day 공유 한계도 남는다.\n\n'
    for r in results:text+=f"- `{r['path']}`: SHA-256 `{r['sha256']}`, share 평균 {r['share_statistics']['mean']:.8f}, RMS {r['share_statistics']['rms']:.8f}, 중앙값 {r['share_statistics']['median']:.8f}.\n"
    (folder/'R18_PRODUCTION_COMPARISON.md').write_text(text)
    return result
