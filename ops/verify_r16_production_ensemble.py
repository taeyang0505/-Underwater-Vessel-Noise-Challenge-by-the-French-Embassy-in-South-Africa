"""R16 ensemble numeric/safety audit: fixed fit IDs/labels, no signal/query IO."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def blocked(event,args):
    if event=='open'and args and isinstance(args[0],(str,bytes)):
        p=str(args[0]).lower()
        if Path(p).name in ('test.csv','sample_submission.csv')or'/eda_deep/'in p:raise PermissionError('No challenge inputs in ensemble QA')
def reject(fn):
    try:fn()
    except (ValueError,TypeError,KeyError,AssertionError):return True
    return False

def audit():
    from pipeline.ensemble_r16_production import MEMBERS,PRIOR,BOUNDS,RIDGE,fit_policy,apply_policy,validate_policy,token,fixture,local_fit_contract,make_local_fit_binding
    began=time.perf_counter();rng=np.random.default_rng(2026102719);n=600
    y=rng.uniform(0,.9,n);y[::5]=0
    shares={k:np.clip(y+rng.normal(0,.1,n),0,1)for k in MEMBERS};shares['bg_cnn']=y.copy()
    clf={k:np.clip((y>0)*.72+rng.uniform(0,.25,n),0,1)for k in MEMBERS}
    binding={'role':'local_fit','Test_used':False,'fixture':True}
    p=fit_policy(shares,clf,y,sample_ids=np.arange(n),binding=binding,allow_fixture=True)
    w=np.asarray(p['regression_weights']);c=np.asarray(p['classifier_weights']);x=np.column_stack([shares[k]for k in MEMBERS]);cl=np.column_stack([clf[k]for k in MEMBERS])
    if w[0]<.5-1e-8 or (w[1:]>.25+1e-8).any()or abs(w.sum()-1)>1e-8:raise AssertionError('Regression cap failed')
    # Convex quadratic: positive ridge Hessian plus exact KKT gives global
    # optimality for this constrained policy fit (not for neural training).
    hess=2*x.T@x/n+2*RIDGE*np.eye(5);grad=2*x.T@(x@w-y)/n+2*RIDGE*(w-PRIOR)
    lo=np.asarray([a for a,b in BOUNDS]);hi=np.asarray([b for a,b in BOUNDS]);free=(w>lo+1e-6)&(w<hi-1e-6)
    if not free.any():raise AssertionError('Expected free coordinate in numeric fixture')
    mult=-float(grad[free].mean());g=grad+mult
    kkt=max(float(abs(g[free]).max()),float(np.maximum(-g[w<=lo+1e-6],0).max(initial=0)),float(np.maximum(g[w>=hi-1e-6],0).max(initial=0)))
    if kkt>1e-6 or np.linalg.eigvalsh(hess).min()<=0:raise AssertionError('Convex optimum KKT failed')
    out=apply_policy(p,shares,clf,allow_fixture=True);changed={k:1-shares[k]for k in MEMBERS};out2=apply_policy(p,changed,clf,allow_fixture=True)
    if not np.array_equal(out['score'],out2['score']):raise AssertionError('Score depends on regression')
    out3=apply_policy(p,shares,{k:1-clf[k]for k in MEMBERS},allow_fixture=True)
    if not np.array_equal(out['share'],out3['share']):raise AssertionError('Regression depends on classification')
    if np.max(abs(out['share']-(x@w).astype(np.float32)))!=0 or np.max(abs(out['score']-(cl@c).astype(np.float32)))!=0:raise AssertionError('Frozen numerical policy differs')
    negatives={}
    for role in ('train','ensemble_gate','local_report','domain_holdout','Test'):
        negatives['fit_role_'+role]=reject(lambda role=role:fit_policy(shares,clf,y,sample_ids=np.arange(n),binding={'role':role,'Test_used':False,'fixture':True},allow_fixture=True))
    negatives['fit_query_used']=reject(lambda:fit_policy(shares,clf,y,sample_ids=np.arange(n),binding={'role':'local_fit','Test_used':True,'fixture':True},allow_fixture=True))
    negatives['duplicate_IDs']=reject(lambda:fit_policy(shares,clf,y,sample_ids=np.zeros(n),binding=binding,allow_fixture=True))
    negatives['missing_member']=reject(lambda:apply_policy(p,{k:v for k,v in shares.items()if k!='fdy'},clf,allow_fixture=True))
    negatives['legacy_highshare_member']=reject(lambda:apply_policy(p,{**shares,'highshare':np.ones(n)},clf,allow_fixture=True))
    for field,val in [('regression_weights',[.49,.25,.2,.03,.03]),('regression_weights',[.5,.26,.24,0,0]),('score_share_weight',.25),('classifier_weights',[1.1,-.1,0,0,0])]:
        q={**p,field:val};q['policy_sha256']=token({k:v for k,v in q.items()if k!='policy_sha256'})
        negatives[field+'__'+str(val)]=reject(lambda q=q:validate_policy(q,allow_fixture=True))
    negatives['bytes_tamper']=reject(lambda:validate_policy({**p,'fit_rmse':1.},allow_fixture=True))
    hardening={}
    q={**p,'member_order':list(reversed(MEMBERS))};q['policy_sha256']=token({k:v for k,v in q.items()if k!='policy_sha256'})
    hardening['rehash_wrong_member_order_rejected']=reject(lambda:validate_policy(q,allow_fixture=True))
    hardening['unbound_nonfixture_fit_rejected']=reject(lambda:fit_policy(shares,clf,y,sample_ids=np.arange(n),binding={'role':'local_fit','Test_used':False}))
    hardening['fixture_policy_forbidden_in_production']=reject(lambda:apply_policy(p,shares,clf))
    hardening['fixture_fit_requires_explicit_opt_in']=reject(lambda:fit_policy(shares,clf,y,sample_ids=np.arange(n),binding=binding))
    for field,value in [('tree5_family_minimum',.4),('each_NN_architecture_maximum',.4),('ridge_to_prior',0.),('regression_prior',[1.,0,0,0,0]),('classifier_weights',[.2,.2,.2,.2,.2])]:
        q={**p,field:value};q['policy_sha256']=token({k:v for k,v in q.items()if k!='policy_sha256'})
        hardening['rehash_invalid_'+field]=reject(lambda q=q:validate_policy(q,allow_fixture=True))
    for field,value in [('regression_weights',[float('nan'),0,0,0,1]),('classifier_weights',[float('inf'),0,0,0,0])]:
        q={**p,field:value}
        hardening['nonfinite_'+field]=reject(lambda q=q:validate_policy(q,allow_fixture=True))
    contract,real_ids,real_y=local_fit_contract()
    if make_local_fit_binding(real_ids,real_y)!=contract:raise AssertionError('Exact fixed source contract did not validate')
    hardening['fixed_ID_order_change_rejected']=reject(lambda:make_local_fit_binding(real_ids[::-1],real_y))
    changed_y=real_y.copy();changed_y[0]=1-changed_y[0]
    hardening['fixed_labels_change_rejected']=reject(lambda:make_local_fit_binding(real_ids,changed_y))
    real_d={k:np.clip(real_y+rng.normal(0,.1,len(real_y)),0,1)for k in MEMBERS}
    real_c={k:np.clip((real_y>0)*.72+rng.uniform(0,.25,len(real_y)),0,1)for k in MEMBERS}
    bound=fit_policy(real_d,real_c,real_y,sample_ids=real_ids,binding=contract)
    apply_policy(bound,real_d,real_c)
    for field,value in [('data_manifest_sha256','0'*64),('target_share_sha256','0'*64),('sample_ids_sha256','0'*64),('fit_source_verified',False)]:
        foreign={**contract,field:value}
        hardening['foreign_binding_'+field]=reject(lambda foreign=foreign:fit_policy(real_d,real_c,real_y,sample_ids=real_ids,binding=foreign))
        q={**bound,'binding':foreign};q['policy_sha256']=token({k:v for k,v in q.items()if k!='policy_sha256'})
        hardening['rehash_foreign_policy_'+field]=reject(lambda q=q:validate_policy(q))
    if not all(negatives.values())or not all(hardening.values()):raise AssertionError('Mandatory policy rejection contract failed')
    return {'status':'passed_hardened_numeric_and_fit_source_contract','source_code_sha256':sha(ROOT/'pipeline/ensemble_r16_production.py'),
        'fixture':fixture(),'independent_fixture_weights':w.tolist(),'independent_classifier_weights':c.tolist(),
        'Tree5_family_weight':float(w[0]),'maximum_NN_architecture_weight':float(w[1:].max()),
        'score_share_weight':p['score_share_weight'],'score_regression_intervention_gap':float(abs(out['score']-out2['score']).max()),
        'regression_classifier_intervention_gap':float(abs(out['share']-out3['share']).max()),
        'QP_Hessian_min_eigenvalue':float(np.linalg.eigvalsh(hess).min()),'QP_KKT_max_violation':kkt,
        'required_negative_tests':negatives,'all_required_negative_tests_passed':all(negatives.values()),
        'additional_hardening_tests':hardening,'fixed_fit_source_contract':contract,
        'fixed_fit_labels_IDs_verified':True,'actual_fit_predictions_reviewed':False,'Test_or_signal_matrices_accessed':False,
        'scope':'Synthetic model outputs for numeric QA; actual fixed local_fit IDs/labels and panel bytes verified; no trained prediction claim',
        'verifier_code_sha256':sha(__file__),'elapsed_seconds':time.perf_counter()-began}

if __name__=='__main__':
    sys.addaudithook(blocked);r=audit();out=ROOT/'reports/R16_PRODUCTION_ENSEMBLE_INDEPENDENT_QA.json';out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(r,ensure_ascii=False))
