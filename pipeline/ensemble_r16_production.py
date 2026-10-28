"""R16 cap-constrained regression and independent-only presence scoring.

No input observations or site metadata accepted. Fit on fixed local_fit
predictions only; Test inference applies a frozen numerical policy unchanged.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

MEMBERS=('tree5','bg_cnn','fdy','itransformer','tfgrid_fdy_conformer')
PRIOR=np.array([.5,.125,.125,.125,.125])
BOUNDS=((.5,1.),(0.,.25),(0.,.25),(0.,.25),(0.,.25))
RIDGE=.0001
ROOT=Path(__file__).resolve().parents[1]
LOCAL_FIT_MANIFEST='reports/r16_cross_operator_r1/data/manifest.json'
LOCAL_FIT_MANIFEST_SHA256='509a6fa2d1820eb25c551cef0bd343b16b531cae97b19ff16959081f1845b3c1'
OPERATORS=('control_eq6','mfhf_shelf6','mfhf_shelf12','mfhf_shelf12_lfminus6',
           'alpha_preEQ_inserted','independent_subband_snr')
CLASSIFIER_CANDIDATES={'tree_only':[1.,0,0,0,0],'tree75_nn25':[.75,.0625,.0625,.0625,.0625],
                'tree60_nn40':[.6,.1,.1,.1,.1],'tree50_nn50':PRIOR.tolist()}

def token(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def local_fit_contract():
    """Verify only fixed development labels/IDs, never gate/query observations.

    Fixed manifest byte identity prevents a replacement manifest (even one
    declaring itself local_fit) from authorizing another role. NPZ access reads
    only sample_id/share; signal matrices are not extracted by this function.
    """
    path=ROOT/LOCAL_FIT_MANIFEST
    if _sha(path)!=LOCAL_FIT_MANIFEST_SHA256:raise ValueError('Fixed R16 local_fit manifest changed')
    manifest=json.loads(path.read_text());entries=manifest['panels']['local_fit']
    if tuple(entries)!=OPERATORS:raise ValueError('Fixed local_fit operator order changed')
    ids=[];targets=[];pins={}
    for op in OPERATORS:
        e=entries[op];p=(path.parent/e['path']).resolve()
        if not p.is_relative_to(path.parent.resolve())or e['n']!=400 or _sha(p)!=e['sha256']:
            raise ValueError('Fixed local_fit panel identity changed: '+op)
        with np.load(p,allow_pickle=False)as z:
            a=np.asarray(z['sample_id']).astype(str);y=np.asarray(z['share'],np.float64)
        if a.shape!=(400,)or y.shape!=(400,):raise ValueError('Fixed local_fit label/ID shape changed')
        ids.append(a);targets.append(y);pins[str(p.relative_to(ROOT))]=e['sha256']
    ids=np.concatenate(ids);y=np.concatenate(targets)
    return {'role':'local_fit','Test_used':False,'fixture':False,'data_manifest_path':LOCAL_FIT_MANIFEST,
            'data_manifest_sha256':LOCAL_FIT_MANIFEST_SHA256,'operator_order':list(OPERATORS),
            'panel_sha256':pins,'n':2400,'sample_ids_sha256':token(ids.tolist()),
            'target_share_sha256':token(y.tolist()),'fit_source_verified':True},ids,y
def make_local_fit_binding(sample_ids,y):
    contract,expected_ids,expected_y=local_fit_contract()
    if not np.array_equal(np.asarray(sample_ids).astype(str),expected_ids)or not np.array_equal(np.asarray(y,np.float64),expected_y):
        raise ValueError('Predictions must match exact fixed local_fit order and unmodified labels')
    return contract
def matrix(d):
    if set(d)!=set(MEMBERS):raise ValueError('All completed R16 model families required')
    if any(np.asarray(d[k]).ndim!=1 for k in MEMBERS):raise ValueError('Each model output must be a 1D vector')
    a=np.column_stack([np.asarray(d[k],float)for k in MEMBERS])
    if a.ndim!=2 or not len(a)or not np.isfinite(a).all()or ((a<0)|(a>1)).any():raise ValueError('Invalid db-free model probabilities/shares')
    return a
def validate_policy(p,*,allow_fixture=False):
    q={k:v for k,v in p.items()if k!='policy_sha256'}
    if p.get('policy_sha256')!=token(q)or p.get('format')!='r16_capped_regression_independent_score_v1':raise ValueError('Frozen policy hash/format mismatch')
    if p.get('member_order')!=list(MEMBERS):raise ValueError('Exact fixed member order required')
    if p.get('tree5_family_minimum')!=.5 or p.get('each_NN_architecture_maximum')!=.25 or p.get('ridge_to_prior')!=RIDGE or p.get('regression_prior')!=PRIOR.tolist():raise ValueError('Frozen cap/prior metadata differs from contract')
    if p.get('Test_used')is not False or p.get('gate_report_domain_for_weight_fit')is not False:raise ValueError('Non-development policy fitting forbidden')
    binding=p['binding']
    if binding.get('role')!='local_fit'or binding.get('Test_used')is not False:raise ValueError('Explicit local_fit binding required')
    if binding.get('fixture')is True:
        if not allow_fixture:raise ValueError('Fixture policy forbidden in production')
    else:
        expected,_,_=local_fit_contract()
        if any(binding.get(k)!=v for k,v in expected.items()):raise ValueError('Production policy local_fit source identity differs')
        if p.get('fit_sample_ids_sha256')!=expected['sample_ids_sha256']or p.get('fit_target_share_sha256')!=expected['target_share_sha256']:raise ValueError('Policy local_fit labels/IDs differ')
    w=np.asarray(p['regression_weights'],float);c=np.asarray(p['classifier_weights'],float)
    if not np.isfinite(w).all()or not np.isfinite(c).all():raise ValueError('Non-finite policy weights')
    if w.shape!=(5,)or c.shape!=(5,)or abs(w.sum()-1)>1e-7 or abs(c.sum()-1)>1e-7:raise ValueError('Simplex policy invalid')
    if w[0]<.5-1e-7 or (w[1:]>.25+1e-7).any()or (w<0).any()or(c<0).any()or p['score_share_weight']!=0.:raise ValueError('User cap or independent score contract breached')
    name=p.get('classifier_candidate_selected')
    if name not in CLASSIFIER_CANDIDATES or c.tolist()!=CLASSIFIER_CANDIDATES[name]:raise ValueError('Independent score candidate differs')
    aucs=p.get('classifier_fit_AUC_candidates',{})
    if set(aucs)!=set(CLASSIFIER_CANDIDATES)or any(not np.isfinite(v)or not 0<=v<=1 for v in aucs.values()):raise ValueError('Classifier fit metrics invalid')
    if not np.isfinite(p['fit_rmse'])or not 0<=p['fit_rmse']<=1:raise ValueError('Invalid fit metric')
    return p
def fit_policy(shares,classifiers,y,*,sample_ids,binding,allow_fixture=False):
    x=matrix(shares);cl=matrix(classifiers);y=np.asarray(y,float);ids=np.asarray(sample_ids).astype(str)
    if y.shape!=(len(x),)or ids.ndim!=1 or cl.shape!=x.shape or len(set(ids))!=len(ids)or len(ids)!=len(x):raise ValueError('Aligned unique local_fit IDs required')
    if not np.isfinite(y).all()or ((y<0)|(y>1)).any()or len(np.unique(y>0))!=2:raise ValueError('Finite physical labels and both presence classes required')
    if binding.get('role')!='local_fit'or binding.get('Test_used')is not False:raise ValueError('Explicit local_fit-only source binding')
    if binding.get('fixture')is True:
        if not allow_fixture:raise ValueError('Explicit fixture opt-in required')
    else:
        verified=make_local_fit_binding(ids,y)
        if any(binding.get(k)!=v for k,v in verified.items()):raise ValueError('Use make_local_fit_binding for production fitting')
    result=minimize(lambda w:np.mean((x@w-y)**2)+RIDGE*np.sum((w-PRIOR)**2),PRIOR,
        jac=lambda w:2*x.T@(x@w-y)/len(x)+2*RIDGE*(w-PRIOR),method='SLSQP',bounds=BOUNDS,
        constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(5)}],options={'ftol':1e-12,'maxiter':1000})
    if not result.success:raise ValueError('Constrained fit failed: '+result.message)
    # All candidates contain only independent presence probabilities, no share.
    candidates=CLASSIFIER_CANDIDATES
    au={k:float(roc_auc_score(y>0,cl@np.asarray(w)))for k,w in candidates.items()}
    name=max(candidates,key=lambda n:(au[n],candidates[n][0]))
    p={'format':'r16_capped_regression_independent_score_v1','member_order':list(MEMBERS),
       'regression_weights':result.x.tolist(),'tree5_family_minimum':.5,'each_NN_architecture_maximum':.25,
       'ridge_to_prior':RIDGE,'regression_prior':PRIOR.tolist(),'classifier_weights':candidates[name],
       'classifier_candidate_selected':name,'classifier_fit_AUC_candidates':au,'score_share_weight':0.,
       'fit_rmse':float(np.sqrt(np.mean((x@result.x-y)**2))),'binding':binding,'fit_sample_ids_sha256':token(ids.tolist()),'fit_target_share_sha256':token(y.tolist()),
       'gate_report_domain_for_weight_fit':False,'Test_used':False}
    p['policy_sha256']=token(p);return validate_policy(p,allow_fixture=allow_fixture)
def apply_policy(p,shares,classifiers,*,allow_fixture=False):
    validate_policy(p,allow_fixture=allow_fixture);x=matrix(shares);c=matrix(classifiers)
    if c.shape!=x.shape:raise ValueError('Prediction alignment differs')
    return {'share':np.clip(x@np.asarray(p['regression_weights']),0,1).astype(np.float32),
            'score':np.clip(c@np.asarray(p['classifier_weights']),0,1).astype(np.float32)}
def fixture():
    rng=np.random.default_rng(2026102718);y=rng.uniform(0,.9,500);y[::5]=0
    # CNN intentionally looks perfect on development: cap must still hold.
    d={k:np.clip(y+rng.normal(0,.12,len(y)),0,1)for k in MEMBERS};d['bg_cnn']=y.copy()
    c={k:np.clip((y>0)*.7+rng.uniform(0,.2,len(y)),0,1)for k in MEMBERS}
    p=fit_policy(d,c,y,sample_ids=np.arange(500),binding={'role':'local_fit','Test_used':False,'fixture':True},allow_fixture=True)
    assert p['regression_weights'][1]<=.25+1e-8 and p['regression_weights'][0]>=.5-1e-8
    a=apply_policy(p,d,c,allow_fixture=True);changed={k:1-np.asarray(v)for k,v in d.items()};b=apply_policy(p,changed,c,allow_fixture=True)
    assert np.array_equal(a['score'],b['score'])
    return {'status':'passed','weights_when_CNN_perfect':p['regression_weights'],'score_independent_of_regression':True,
       'Tree_family_floor':.5,'NN_architecture_cap':.25,'ridge':RIDGE}
