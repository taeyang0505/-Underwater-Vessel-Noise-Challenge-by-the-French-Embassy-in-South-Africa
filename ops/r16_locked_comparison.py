"""R16 fit-only policy freeze and subsequent held-out small-model audit.

No challenge Test IO. Gate/report/domain never select weights/checkpoints.
All candidate policies remain reported; a cap is a hypothesis, not a promise.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.r16_forward_data import OPERATORS, load_panel, sha, guard
from ops.r16_clean_models import write, metrics, predict_arm

MANIFEST=ROOT/'reports/r16_cross_operator_r1/data/manifest.json'
OUT=ROOT/'reports/r16_locked_comparison_r1'
SMALL=ROOT/'models/r16_clean_pilot_r1'
FROZEN=ROOT/'reports/r16_frozen_evaluation_r1'
KINDS=('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer')
ROLES=('local_fit','local_gate','local_report','domain_holdout')

def read(p):return json.loads(Path(p).read_text())

def load_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def get_saved(folder,role):
    ds=[load_npz(folder/'predictions'/f'{role}__{op}.npz') for op in OPERATORS]
    for d in ds[1:]:
        assert np.array_equal(d['model_order'],ds[0]['model_order'])
        assert np.array_equal(d['classifier_model_order'],ds[0]['classifier_model_order'])
    keys=('target_share','target_presence','site','share','own_score','raw_classifier','sample_id')
    joined={k:np.concatenate([d[k] for d in ds]) for k in keys}
    joined['model_order']=ds[0]['model_order'];joined['classifier_model_order']=ds[0]['classifier_model_order']
    return joined

def simplex_fit(x,y,bounds):
    x=np.asarray(x,float);y=np.asarray(y,float)
    start=np.array([.5,.125,.125,.125,.125])
    r=minimize(lambda w:np.mean((x@w-y)**2),start,jac=lambda w:2*x.T@(x@w-y)/len(y),
       method='SLSQP',bounds=bounds,constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(5)}],
       options={'ftol':1e-12,'maxiter':1000})
    if not r.success or abs(r.x.sum()-1)>1e-7:raise ValueError('Simplex fit failed: '+r.message)
    return r.x

def freeze():
    if OUT.exists():raise FileExistsError('Fresh policy folder required')
    OUT.mkdir()
    # Write constraints before reading any development prediction metrics.
    plan={'status':'predeclared_before_policy_fit','created_unix':time.time(),
       'selection_role':'local_fit only, six operators equal row weight',
       'old_families':['R9','R14_baseline'],
       'old_candidates':['fixed50tree_4x12.5NN','unconstrained_fit','capped_fit'],
       'capped_bounds':{'tree5_family':[.4,.8],'each_of_four_NN':[.05,.4]},
       'tree5_exception':'family is aggregate, each constituent fold has family_weight/5 <=.16',
       'highshare_models':'comparison only, excluded from all newly fitted policies',
       'new_candidates':['nn','tree','fixed60tree40nn'],
       'classifier_mix':'new small NN/Tree probabilities equal average, separate from regressor',
       'new_score_share_lambdas':[0.,.25],'heavy_share_lambda':.75,
       'no_selection_on_gate_report_domain':True,'no_Test_read_or_statistics':True,
       'TF37':'paused partial-budget diagnostic, no optimizer restart'}
    write(OUT/'PRE_FIT_PLAN.json',plan)
    d=get_saved(FROZEN/'development','local_fit');names=list(d['model_order']);clnames=list(d['classifier_model_order'])
    y=d['target_share'];policies={};fit_rows=[]
    for family in ('R9','R14_baseline'):
        tree='R9_tree5' if family=='R9' else 'R14_tree5_baseline'
        prefix='R9_' if family=='R9' else 'R14_'
        members=[tree]+[prefix+k for k in KINDS]
        x=d['share'][:,[names.index(k) for k in members]]
        classifier=d['raw_classifier'][:,clnames.index(tree)]
        weights={'fixed50tree_4x12.5NN':np.array([.5,.125,.125,.125,.125]),
                 'unconstrained_fit':simplex_fit(x,y,[(0,1)]*5),
                 'capped_fit':simplex_fit(x,y,[(.4,.8)]+[(.05,.4)]*4)}
        for method,w in weights.items():
            name=family+'__'+method;p=x@w
            aucs={str(lam):metrics(y,p,(1-lam)*classifier+lam*p)['auc_classifier'] for lam in (0.,.25)}
            lam=max((0.,.25),key=lambda a:(aucs[str(a)],-a))
            policies[name]={'members':members,'weights':w.tolist(),'classifier_member':tree,
                            'score_share_lambda':lam,'fit_score_auc_candidates':aucs}
            fit_rows.append({'name':name,**metrics(y,p,classifier)})
    # New models select score share contribution using local_fit only.
    for arm in ('control','mixed'):
        a=load_npz(SMALL/arm/'fit_nn_predictions.npz');b=load_npz(SMALL/arm/'fit_tree_predictions.npz')
        for method in ('nn','tree','fixed60tree40nn'):
            p=a['share'] if method=='nn' else b['share'] if method=='tree' else .4*a['share']+.6*b['share']
            c=a['classifier_prob'] if method=='nn' else b['classifier_prob'] if method=='tree' else .5*a['classifier_prob']+.5*b['classifier_prob']
            vals={str(l):metrics(y,p,(1-l)*c+l*p)['auc_classifier'] for l in (0.,.25)}
            lam=max((0.,.25),key=lambda l:(vals[str(l)],-l))
            policies[arm+'__'+method]={'score_share_lambda':lam,'fit_score_auc_candidates':vals,
                                      'regression_weights':[0,1] if method=='nn' else [1,0] if method=='tree' else [.6,.4]}
            fit_rows.append({'name':arm+'__'+method,**metrics(y,p,c)})
    write(OUT/'POLICIES.json',{'status':'frozen_fit_only','policies':policies,'fit_rows':fit_rows,
                              'manifest_sha256':sha(MANIFEST),'fit_predictions_sha256':{
            str(p.relative_to(ROOT)):sha(p) for p in list((FROZEN/'development/predictions').glob('*.npz'))+list(SMALL.glob('*/fit_*predictions.npz'))}})
    pins={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'ops/r16_clean_models.py',ROOT/'ops/r16_forward_data.py',ROOT/'ops/r16_evaluate_frozen.py',OUT/'POLICIES.json',OUT/'PRE_FIT_PLAN.json',SMALL/'PLAN.json',SMALL/'ALL_COMPLETED.json']}
    for arm in ('control','mixed'):
        r=read(SMALL/arm/'COMPLETED.json')
        assert r['status']=='completed' and r['epochs']==100 and r['early_stopping'] is False
        for kind in ('checkpoint_sha256','tree_sha256'):
            for name,h in r[kind].items():
                p=SMALL/arm/name
                if sha(p)!=h:raise ValueError('New model artifact mismatch')
                pins[str(p.relative_to(ROOT))]=h
        pins[str((SMALL/arm/'COMPLETED.json').relative_to(ROOT))]=sha(SMALL/arm/'COMPLETED.json')
    old=read(FROZEN/'development/RESULTS.json')
    for n,h in old['source_artifact_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Old frozen artifact mismatch: '+n)
        pins[n]=h
    receipt={'status':'completed','all_models_frozen':True,'ensemble_policy_frozen':True,
       'data_manifest_sha256':sha(MANIFEST),'artifact_sha256':pins,'created_unix':time.time(),
       'scopes':'R16 pilot control/mixed and old frozen diagnostic models; not a new production model',
       'fit_role':'local_fit only','gate_report_domain_opened_for_policy_selection':False,'Test_used':False}
    write(OUT/'GLOBAL_FREEZE.json',receipt)
    print(json.dumps({'status':'globally_frozen','policies':len(policies),'pinned_artifacts':len(pins),'receipt':str(OUT/'GLOBAL_FREEZE.json')}),flush=True)

def verify():
    r=read(OUT/'GLOBAL_FREEZE.json')
    for name,h in r['artifact_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Global frozen artifact changed: '+name)

def evaluate_small():
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    out=OUT/'small_locked';out.mkdir(exist_ok=False)
    verify();policy=read(OUT/'POLICIES.json')['policies'];rows=[];began=time.time();pins={}
    for role in ROLES:
        for op in OPERATORS:
            d=load_panel(MANIFEST,role,op,freeze_receipt=OUT/'GLOBAL_FREEZE.json')
            for arm in ('control','mixed'):
                verify();pr=predict_arm(SMALL/arm,d)
                p=out/f'{role}__{op}__{arm}.npz'
                np.savez_compressed(p,**pr,target_share=d['share'],target_presence=d['presence'],site=d['site'],sample_id=d['sample_id'])
                pins[p.name]=sha(p)
                for name,pk,ck in [('nn','nn_share','nn_classifier'),('tree','tree_share','tree_classifier'),('fixed60tree40nn','balanced_share','balanced_classifier')]:
                    m=metrics(d['share'],pr[pk],pr[ck],d['site'])
                    lam=policy[arm+'__'+name]['score_share_lambda']
                    m['auc_frozen_score']=metrics(d['share'],pr[pk],lam*pr[pk]+(1-lam)*pr[ck])['auc_classifier']
                    rows.append({'role':role,'operator':op,'arm':arm,'model':name,'score_share_lambda':lam,**m})
            write(out/'progress.json',{'status':'evaluating','last_role':role,'last_operator':op,'elapsed_seconds':time.time()-began})
            print(json.dumps({'event':'small_panel_complete','role':role,'operator':op,'seconds':time.time()-began}),flush=True)
    verify();write(out/'RESULTS.json',{'status':'completed','rows':rows,'predictions_sha256':pins,'elapsed_seconds':time.time()-began,'Test_used':False,'policies_changed':False})
    keys=[k for k in rows[0] if k!='site_slices']
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows([{k:r[k] for k in keys}for r in rows])
    print(json.dumps({'status':'small_locked_completed','seconds':time.time()-began}),flush=True)

def combine_old():
    verify();out=OUT/'old_policy_audit';out.mkdir(exist_ok=False);policies=read(OUT/'POLICIES.json')['policies'];rows=[]
    for role in ROLES:
        data=get_saved(FROZEN/('development'if role=='local_fit'else'locked'),role)
        names=list(data['model_order']);cn=list(data['classifier_model_order'])
        for opi,op in enumerate(OPERATORS):
            sl=slice(opi*400,(opi+1)*400);y=data['target_share'][sl];site=data['site'][sl]
            # Existing model and exact unchanged aggregate policies.
            for i,name in enumerate(names):
                p=data['share'][sl,i];score=data['own_score'][sl,i]
                r={'role':role,'operator':op,'model':name,'policy':'old_unchanged',**metrics(y,p,score,site)}
                if name in cn:r['raw_classifier_auc']=metrics(y,p,data['raw_classifier'][sl,cn.index(name)])['auc_classifier']
                rows.append(r)
            for name,p in policies.items():
                if 'members'not in p:continue
                pred=data['share'][sl][:,[names.index(n)for n in p['members']]]@np.asarray(p['weights'])
                clf=data['raw_classifier'][sl,cn.index(p['classifier_member'])];lam=p['score_share_lambda']
                m=metrics(y,pred,clf,site);m['auc_frozen_score']=metrics(y,pred,lam*pred+(1-lam)*clf)['auc_classifier']
                rows.append({'role':role,'operator':op,'model':name,'policy':'new_fit_only_frozen','score_share_lambda':lam,**m})
    write(out/'RESULTS.json',{'status':'completed','rows':rows,'Test_used':False,'no_post_gate_fit':True})
    print(json.dumps({'status':'old_policy_audit_completed','rows':len(rows)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','evaluate_small','combine_old']);a=p.parse_args()
    sys.addaudithook(guard)
    {'freeze':freeze,'evaluate_small':evaluate_small,'combine_old':combine_old}[a.command]()
