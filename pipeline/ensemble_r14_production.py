"""Approved R14 fit/gate-only convex regression and independent score policy.

No files, data sources, models or public query labels are read here. Caller
must establish same-fold source isolation before passing the two roles.
"""
from __future__ import annotations
import hashlib,json
from collections.abc import Mapping
import numpy as np

VERSION='r14_production_fit_gate_ensemble_v1'
LAMBDAS=(0.,.25,.5,.75,.85,1.)
CLASSIFIERS=('baseline','highshare','mean')
EPS=1e-4
RIDGE=1e-6

def token(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def auc(y,p):
    z=np.asarray(y)>0;p=np.asarray(p,float)
    if p.shape!=z.shape or not np.isfinite(p).all():raise ValueError('Aligned finite score required')
    n=int(z.sum());m=len(z)-n
    if not n or not m:raise ValueError('Both addition classes required for fit/gate AUC')
    _,inv,cnt=np.unique(p,return_inverse=True,return_counts=True)
    ranks=np.cumsum(cnt)-.5*(cnt-1)
    return float((ranks[inv][z].sum()-n*(n+1)/2)/(n*m))

def _targets(y,n):
    y=np.asarray(y,float)
    if y.shape!=(n,) or not np.isfinite(y).all() or ((y<0)|(y>1)).any():raise ValueError('Known physical share targets required')
    if not (y==0).any() or not (y>0).any():raise ValueError('Both known addition classes required')
    return y

def _matrix(values,names=None,n=None):
    if not isinstance(values,Mapping) or not values:raise TypeError('Named prediction map required')
    names=list(values) if names is None else list(names)
    if set(values)!=set(names) or len(names)!=len(set(names)) or any(not isinstance(k,str) for k in names):raise ValueError('Exact declared member set required')
    arrays=[np.asarray(values[k],float) for k in names]
    if any(a.ndim!=1 for a in arrays) or len({len(a) for a in arrays})!=1:raise ValueError('Aligned prediction vectors required')
    if n is not None and len(arrays[0])!=n:raise ValueError('Prediction role row count differs')
    matrix=np.column_stack(arrays)
    if not np.isfinite(matrix).all() or ((matrix<0)|(matrix>1)).any():raise ValueError('Finite bounded predictions required')
    return names,matrix

def _classifiers(values,n):
    if set(values)!=set(CLASSIFIERS):raise ValueError('baseline/highshare/mean classifier vectors required')
    _,matrix=_matrix(values,CLASSIFIERS,n)
    if not np.allclose(matrix[:,2],.5*(matrix[:,0]+matrix[:,1]),atol=1e-7,rtol=0):raise ValueError('Classifier mean is not the arithmetic mean')
    return {k:matrix[:,i] for i,k in enumerate(CLASSIFIERS)}

def fit_policy(fit_y,fit_shares,fit_classifiers,gate_y,gate_shares,gate_classifiers,*,binding,fixture=False):
    from scipy.optimize import minimize
    if not isinstance(binding,dict) or not binding:raise ValueError('Explicit same-fold source/role binding required')
    token(binding)
    names,x=_matrix(fit_shares);_,g=_matrix(gate_shares,names)
    if not fixture and (len(x)!=500 or len(g)!=500):raise ValueError('Production ensemble_fit/gate500 required')
    y=_targets(fit_y,len(x));gy=_targets(gate_y,len(g));fc=_classifiers(fit_classifiers,len(x));gc=_classifiers(gate_classifiers,len(g))
    m=x.shape[1];uniform=np.full(m,1/m)
    fun=lambda w:float(np.mean((x@w-y)**2)+RIDGE*np.sum((w-uniform)**2))
    jac=lambda w:2*x.T@(x@w-y)/len(y)+2*RIDGE*(w-uniform)
    opt=minimize(fun,uniform,jac=jac,bounds=[(0.,1.)]*m,
        constraints=[{'type':'eq','fun':lambda w:float(w.sum()-1),'jac':lambda w:np.ones_like(w)}],
        method='SLSQP',options={'maxiter':500,'ftol':1e-12})
    if not opt.success or not np.isfinite(opt.x).all() or np.min(opt.x)<-1e-7 or abs(opt.x.sum()-1)>1e-6:
        raise RuntimeError('Nonnegative simplex optimizer failed:'+str(opt.message))
    proposed=np.maximum(opt.x,0);proposed/=proposed.sum()
    fit_single=[float(np.sqrt(np.mean((x[:,i]-y)**2))) for i in range(m)]
    best=int(np.argmin(fit_single));single=np.eye(m)[best]
    rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
    rg=rmse(g@proposed,gy);sg=rmse(g[:,best],gy)
    accepted=rg<=sg-EPS;weights=proposed if accepted else single
    fp=np.clip(x@weights,0,1);gp=np.clip(g@weights,0,1)
    candidates=[]
    for source in CLASSIFIERS:
        for lam in LAMBDAS:
            candidates.append({'classifier':source,'lambda':lam,'fit_AUC':auc(y,(1-lam)*fp+lam*fc[source])})
    selected=max(candidates,key=lambda v:v['fit_AUC'])
    share_auc=auc(gy,gp);candidate_auc=auc(gy,(1-selected['lambda'])*gp+selected['lambda']*gc[selected['classifier']])
    score_accepted=candidate_auc>=share_auc-EPS
    score={'classifier':selected['classifier'],'lambda':selected['lambda'] if score_accepted else 0.,
           'fit_selected_candidate':selected,'gate_candidate_AUC':candidate_auc,'gate_share_AUC':share_auc,
           'candidate_accepted_on_gate':score_accepted}
    policy={'version':VERSION,'binding':binding,'member_order':names,'regression_weights':weights.tolist(),
      'proposed_regression_weights':proposed.tolist(),'fit_single_RMSE':dict(zip(names,fit_single)),
      'fit_best_single':names[best],'gate_proposed_RMSE':rg,'gate_single_RMSE':sg,
      'regression_accepted_on_gate':accepted,'ridge_to_uniform':RIDGE,'safety_epsilon':EPS,
      'score_policy':score,'score_fit_candidates':candidates,'fit_role':'same_fold_baseline_ensemble_fit',
      'gate_role':'same_fold_baseline_ensemble_gate','report_Test_for_fitting':False,'fixture_only':bool(fixture),
      'postprocessing':'numericalclip[0,1] only'}
    policy['policy_sha256']=token(policy)
    return policy

def apply_policy(policy,shares,classifiers,*,allow_fixture=False):
    if policy.get('version')!=VERSION or policy.get('fixture_only') and not allow_fixture:raise ValueError('Bound production policy required')
    if token({k:v for k,v in policy.items() if k!='policy_sha256'})!=policy.get('policy_sha256'):raise ValueError('Policy content changed')
    _,x=_matrix(shares,policy['member_order']);c=_classifiers(classifiers,len(x))
    w=np.asarray(policy['regression_weights'],float)
    if w.shape!=(x.shape[1],) or not np.isfinite(w).all() or (w<0).any() or abs(w.sum()-1)>1e-7:raise ValueError('Invalid frozen simplex weights')
    share=np.clip(x@w,0,1);s=policy['score_policy'];lam=s['lambda']
    if lam not in LAMBDAS or s['classifier'] not in CLASSIFIERS:raise ValueError('Unknown frozen score candidate')
    score=np.clip((1-lam)*share+lam*c[s['classifier']],0,1)
    return {'share':share.astype(np.float32),'score':score.astype(np.float32)}
