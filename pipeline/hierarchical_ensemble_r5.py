"""INNER-only R5 stacking. No datasets, fixed Ultra keys, Test fit or cohort ranks.

Source leave-one-out crossfits the META learner; it does not retrain base models.
Default convex share fusion, optional strictly improved neural-logit mask fusion.
"""
from itertools import combinations
import hashlib,json
import numpy as np

BASE_CANDIDATES=('timemoe_public','dasheng','tfgrid_fdy_conformer','bg_cnn','lgbm_physics')
COMPARISON_CANDIDATES=('fdy','itransformer')
ALL_CANDIDATES=BASE_CANDIDATES+COMPARISON_CANDIDATES
LARGE_CANDIDATES=('timemoe_public','dasheng')
LIGHT_CANDIDATES=tuple(k for k in ALL_CANDIDATES if k not in LARGE_CANDIDATES)
NEURAL_KINDS=tuple(k for k in ALL_CANDIDATES if k!='lgbm_physics')
RIDGE=.05
EPSILON=1e-4
LOGIT_CLIP=1e-6
VERSION='r5_inner_source_loo_convex_policy_v1'
NO_REFERENCE='__no_added_reference__'

def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _binding(value):
    keys=('train_npz_sha256','data_package_hash','source_hash','seed','fold')
    if not isinstance(value,dict) or any(k not in value for k in keys):raise ValueError('Explicit TRAIN/source/package/seed/fold binding required')
    b=dict(value)
    for k in keys[:3]:
        if not isinstance(b[k],str) or len(b[k])!=64 or any(c not in '0123456789abcdef' for c in b[k]):raise ValueError('Invalid SHA binding: '+k)
    for k in keys[3:]:
        if isinstance(b[k],(bool,np.bool_)) or not isinstance(b[k],(int,np.integer)):raise ValueError('Integer binding required: '+k)
        b[k]=int(b[k])
    if b['seed']<0 or not 0<=b['fold']<5:raise ValueError('Invalid seed/fold')
    return json.loads(json.dumps(b,sort_keys=True,allow_nan=False))

def _np(x):
    if hasattr(x,'detach'):x=x.detach().cpu().numpy()
    return np.asarray(x,dtype=np.float64)

def _vector(x,n=None,name='prediction'):
    a=_np(x)
    if a.ndim!=1 or not len(a) or n is not None and len(a)!=n:raise ValueError('Unaligned/empty vector: '+name)
    if not np.isfinite(a).all() or (a<0).any() or (a>1).any():raise ValueError('Finite [0,1] vector required: '+name)
    return a

def _rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))

def _groups(values,n):
    if len(values)!=n:raise ValueError('Reference groups do not align')
    result=[]
    for value in values:
        if isinstance(value,str) and value.lstrip().startswith('['):value=json.loads(value)
        if isinstance(value,(list,tuple)):
            if not value:value=NO_REFERENCE
            elif len(value)==1:value=value[0]
            else:raise ValueError('Single-reference R5 expected; overlapping multi-reference combinations are not independent groups')
        if not isinstance(value,str) or not value.strip():raise ValueError('Explicit reference group/no-addition sentinel required')
        result.append(value)
    return np.asarray(result)

def _admit(y,arrays):
    scores={k:_rmse(y,a) for k,a in arrays.items()}
    light=[k for k in LIGHT_CANDIDATES if k in arrays]
    best=min(light,key=lambda k:(scores[k],ALL_CANDIDATES.index(k)))
    keys=[k for k in ALL_CANDIDATES if k in arrays and (k not in LARGE_CANDIDATES or scores[k]+EPSILON<scores[best])]
    return keys,best,scores

def _fit_weights(P,y,groups):
    """Exact small convex-simplex QP, max127 active faces; no HPO/search seeds.

    Equal-reference-group MSE + .05*||w-uniform||², w>=0,sum(w)=1.
    """
    n,k=P.shape
    if not n or not 1<=k<=7:raise ValueError('One to seven aligned candidates required')
    if k==1:return np.ones(1)
    _,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
    row_w=1./(len(counts)*counts[inverse]);anchor=np.full(k,1/k)
    H=P.T@(row_w[:,None]*P)+RIDGE*np.eye(k);b=P.T@(row_w*y)+RIDGE*anchor
    best=None
    for size in range(1,k+1):
        for active in combinations(range(k),size):
            ids=np.asarray(active);h=H[np.ix_(ids,ids)]
            hb=np.linalg.solve(h,b[ids]);h1=np.linalg.solve(h,np.ones(size))
            lam=(hb.sum()-1)/h1.sum();a=hb-lam*h1
            if (a < -1e-10).any():continue
            w=np.zeros(k);w[ids]=np.maximum(a,0);w/=w.sum()
            objective=float(np.dot(row_w,(P@w-y)**2)+RIDGE*np.sum((w-anchor)**2))
            if best is None or objective<best[0]-1e-14:best=(objective,w)
    if best is None:raise RuntimeError('Fixed convex INNER ridge fit failed')
    return best[1]

def _grids(neural_masks,energy_weights,arrays,n):
    if not isinstance(neural_masks,dict) or set(neural_masks)-set(NEURAL_KINDS):raise ValueError('Known neural mask mapping required')
    w=_np(energy_weights)
    if w.shape!=(n,15,25) or not np.isfinite(w).all() or (w<0).any() or not np.allclose(w.sum((1,2)),1,atol=2e-5,rtol=0):raise ValueError('Original normalized energy weights required')
    masks={}
    for k in arrays:
        if k not in NEURAL_KINDS:continue
        if k not in neural_masks:raise ValueError('Missing required neural mask: '+k)
        q=_np(neural_masks[k])
        if q.shape!=w.shape or not np.isfinite(q).all() or (q<0).any() or (q>1).any():raise ValueError('Invalid fraction mask: '+k)
        if not np.allclose((q*w).sum((1,2)),arrays[k],atol=3e-6,rtol=0):raise ValueError('Masks/shares must use SAME original energy weights: '+k)
        masks[k]=q
    return masks,w

def _predict(keys,weights,arrays,mode,masks=None,energy=None):
    if mode=='convex_share_average':
        # Fixed elementwise order, independent of BLAS batch/chunk kernel choice.
        result=np.zeros(len(next(iter(arrays.values()))))
        for k,w in zip(keys,weights):result+=w*arrays[k]
        return result
    if mode!='logit_neural_mask_pool_with_tree_share':raise ValueError('Unknown frozen mode')
    result=np.zeros(len(next(iter(arrays.values()))));neural=[]
    for k,w in zip(keys,weights):
        if k in NEURAL_KINDS and w>0:neural.append((k,float(w)))
        elif k not in NEURAL_KINDS:result+=w*arrays[k]
    total=sum(w for _,w in neural)
    if total:
        logits=np.zeros_like(energy)
        for k,w in neural:
            q=np.clip(masks[k],LOGIT_CLIP,1-LOGIT_CLIP);logits+=(w/total)*(np.log(q)-np.log1p(-q))
        cell=np.empty_like(logits);positive=logits>=0
        cell[positive]=1/(1+np.exp(-logits[positive]));e=np.exp(logits[~positive]);cell[~positive]=e/(1+e)
        result+=total*(cell*energy).sum((1,2))
    return result

def _seal(policy):
    p=json.loads(json.dumps(policy,sort_keys=True,allow_nan=False));p['policy_sha256']=canonical_sha256(p);return p

def select_inner_policy(y,branch_shares,reference_groups,*,fit_split='inner',binding,
                        explicit_exclusions=None,enable_logit_mask_candidate=False,
                        neural_masks=None,energy_weights=None,ridge=.05,epsilon=1e-4):
    """Select/freeze on INNER only. Missing base candidates need explicit reasons.

    Optional fdy/iTransformer comparisons belong to the predeclared light pool.
    Large individual admission is recomputed on every meta-training partition.
    Fusion must beat best light in BOTH full INNER and source-meta-LOO RMSE.
    """
    if fit_split!='inner':raise ValueError('Fitting/selection/calibration on OUTER/Test is forbidden; use INNER only')
    if ridge!=RIDGE or epsilon!=EPSILON or type(enable_logit_mask_candidate) is not bool:raise ValueError('Predeclare fixed ridge=.05, epsilon=1e-4 and boolean logit option')
    y=_vector(y,name='INNER target');n=len(y);bind=_binding(binding)
    exclusions={} if explicit_exclusions is None else explicit_exclusions
    if not isinstance(exclusions,dict) or set(exclusions)-set(ALL_CANDIDATES) or any(not isinstance(v,str) or not v.strip() for v in exclusions.values()):raise ValueError('Known explicit exclusions with reason strings required')
    if not isinstance(branch_shares,dict) or set(branch_shares)-set(ALL_CANDIDATES):raise ValueError('Unknown branch keys')
    missing=[k for k in BASE_CANDIDATES if k not in branch_shares and k not in exclusions]
    if missing:raise ValueError('Missing base candidates need explicit exclusions; no silent renormalization: '+','.join(missing))
    arrays={k:_vector(branch_shares[k],n,k) for k in ALL_CANDIDATES if k in branch_shares and k not in exclusions}
    if not set(arrays).intersection(LIGHT_CANDIDATES):raise ValueError('At least one predeclared light fallback candidate required')
    groups=_groups(reference_groups,n);keys,best_light,scores=_admit(y,arrays)
    weights=_fit_weights(np.column_stack([arrays[k] for k in keys]),y,groups)
    modes=['convex_share_average'];masks=energy=None
    if enable_logit_mask_candidate:
        masks,energy=_grids(neural_masks,energy_weights,arrays,n);modes.append('logit_neural_mask_pool_with_tree_share')
    full={m:_predict(keys,weights,arrays,m,masks,energy) for m in modes}
    unique=sorted(set(groups));positive_groups=sorted(set(groups)-{NO_REFERENCE})
    can_crossfit=len(positive_groups)>=3 and len(unique)>=3
    loo=[];cross={m:np.empty(n) for m in modes}
    if can_crossfit:
        for group in unique:
            held=groups==group;train=~held;aa={k:v[train] for k,v in arrays.items()}
            kk,bb,ss=_admit(y[train],aa);ww=_fit_weights(np.column_stack([aa[k] for k in kk]),y[train],groups[train])
            query={k:v[held] for k,v in arrays.items()}
            row={'heldout_reference_group':group,'heldout_examples':int(held.sum()),'meta_train_examples':int(train.sum()),
                 'meta_train_group_ids':sorted(set(groups[train])),'best_meta_train_light':bb,
                 'meta_train_branch_rmse':ss,'admitted_on_meta_train_only':kk,'weights_fitted_on_meta_train':dict(zip(kk,ww.tolist())),'heldout_rmse':{}}
            for mode in modes:
                mm=None if masks is None else {k:v[held] for k,v in masks.items()}
                p=_predict(kk,ww,query,mode,mm,None if energy is None else energy[held]);cross[mode][held]=p;row['heldout_rmse'][mode]=_rmse(y[held],p)
            loo.append(row)
    full_rmse={m:_rmse(y,p) for m,p in full.items()};loo_rmse={m:_rmse(y,p) for m,p in cross.items()} if can_crossfit else {m:None for m in modes}
    passed={m:bool(can_crossfit and full_rmse[m]+EPSILON<scores[best_light] and loo_rmse[m]+EPSILON<scores[best_light]) for m in modes}
    mode='single_light_fallback';chosen={best_light:1.};reason='Strict INNER/meta-LOO improvement gate failed; best-INNER-light fallback.'
    if not can_crossfit:reason='Fewer than three positive reference groups; use best-INNER-light fallback.'
    if passed['convex_share_average']:mode='convex_share_average';chosen={k:float(w) for k,w in zip(keys,weights) if w>0};reason='Both full INNER and reference-group meta-LOO improvement gates passed.'
    logit_pass=False
    if enable_logit_mask_candidate:
        m='logit_neural_mask_pool_with_tree_share'
        logit_pass=bool(passed[m] and full_rmse[m]+EPSILON<full_rmse['convex_share_average'] and loo_rmse[m]+EPSILON<loo_rmse['convex_share_average'])
        if logit_pass:mode=m;chosen={k:float(w) for k,w in zip(keys,weights) if w>0};reason='Secondary logit mode strictly beat best light and convex-share mode at both INNER gates.'
    statuses={}
    for k in ALL_CANDIDATES:
        if k in exclusions:state={'status':'explicit_exclusion','reason':exclusions[k],'prediction_supplied':k in branch_shares}
        elif k not in arrays:state={'status':'not_requested_comparison','reason':'Optional comparison not supplied.'}
        elif k in LARGE_CANDIDATES and k not in keys:state={'status':'excluded_large_individual_gate','reason':'Individual INNER RMSE did not beat best light by epsilon.'}
        elif chosen.get(k,0)>0:state={'status':'selected','reason':reason}
        else:state={'status':'excluded_by_frozen_final_policy','reason':'Final weight zero or single-light fallback selected.'}
        statuses[k]={**state,'inner_rmse':scores.get(k),'final_weight':float(chosen.get(k,0))}
    return _seal({'version':VERSION,'fit_split':'inner','frozen_before_outer':True,'training_binding':bind,
        'selected_mode':mode,'weights':chosen,'required_prediction_keys':[k for k in ALL_CANDIDATES if chosen.get(k,0)>0],
        'all_predeclared_candidates':list(ALL_CANDIDATES),'available_prediction_keys':list(arrays),'candidate_status':statuses,
        'explicit_exclusions':dict(exclusions),'inner_examples':n,'inner_branch_rmse':scores,'best_inner_light':best_light,
        'predeclared_fallback_light_candidates':[k for k in LIGHT_CANDIDATES if k in arrays],
        'fallback':{'kind':'best_INNER_light_single','model':best_light,'weights':{best_light:1.}},
        'large_individual_admission':{k:{'available':k in arrays,'admitted':k in keys,'individual_inner_rmse':scores.get(k),
            'must_beat_best_light':scores[best_light],'epsilon_rmse':EPSILON} for k in LARGE_CANDIDATES},
        'ridge_lambda':RIDGE,'ridge_anchor':'uniform over admitted candidates','epsilon_rmse':EPSILON,
        'fit_objective':'equal-reference-group MSE + .05*||w-uniform||^2 on convex simplex',
        'group_balance_for_weight_fit':True,'source_loo_is_meta_level_only':True,'positive_reference_group_count':len(positive_groups),
        'reference_group_count':len(unique),'source_meta_loo_available':can_crossfit,'source_meta_loo_results':loo,
        'candidate_full_inner_rmse':full_rmse,'candidate_meta_loo_rmse':loo_rmse,'candidate_gates_passed':passed,
        'full_INNER_admitted_ridge_weights_including_exact_zeros':dict(zip(keys,weights.tolist())),
        'secondary_logit_candidate_requested':enable_logit_mask_candidate,'secondary_logit_candidate_strict_gate_passed':logit_pass,
        'logit_clip':LOGIT_CLIP,'logit_definition':'Neural total weight s: sigmoid(sum(w_k/s*logit(clip(q_k)))); pool SAME original w; multiply by s; add frozen weighted tree share.',
        'selection_reason':reason,'outer_used_for_fit':False,'Test_used_for_fit':False,
        'outer_failure_action':'Report/preserve failure; never refit or switch policy from OUTER values.',
        'silent_missing_branch_renormalization':False})

def validate_policy(policy,*,binding=None):
    if not isinstance(policy,dict) or policy.get('version')!=VERSION or policy.get('fit_split')!='inner' or policy.get('frozen_before_outer') is not True:raise ValueError('INNER-frozen R5 policy required')
    content={k:v for k,v in policy.items() if k!='policy_sha256'}
    if policy.get('policy_sha256')!=canonical_sha256(content):raise ValueError('Frozen policy SHA mismatch')
    if policy.get('ridge_lambda')!=RIDGE or policy.get('epsilon_rmse')!=EPSILON or policy.get('logit_clip')!=LOGIT_CLIP:raise ValueError('Unapproved fixed policy constants')
    bb=_binding(policy.get('training_binding'))
    if binding is not None and bb!=_binding(binding):raise ValueError('Policy belongs to another TRAIN/fold/package')
    weights=policy.get('weights',{})
    if not isinstance(weights,dict) or not weights or set(weights)-set(ALL_CANDIDATES):raise ValueError('Invalid frozen branch weights')
    ww=np.asarray(list(weights.values()),float)
    if not np.isfinite(ww).all() or (ww<0).any() or abs(ww.sum()-1)>1e-10:raise ValueError('Frozen weights must be convex')
    required=[k for k in ALL_CANDIDATES if weights.get(k,0)>0]
    if policy.get('required_prediction_keys')!=required:raise ValueError('Frozen required branches changed')
    mode=policy.get('selected_mode')
    if mode not in {'single_light_fallback','convex_share_average','logit_neural_mask_pool_with_tree_share'}:raise ValueError('Unknown frozen mode')
    if mode=='single_light_fallback' and (len(required)!=1 or required[0] not in LIGHT_CANDIDATES or weights[required[0]]!=1):raise ValueError('Invalid light fallback')
    return policy

def apply_policy(policy,branch_shares,*,evaluation_split='inference',binding=None,
                 neural_masks=None,energy_weights=None,labels=None,return_details=False):
    """Fixed row-independent query inference. Never takes labels or fits anything."""
    if labels is not None:raise ValueError('Query labels are forbidden in frozen inference, including OUTER/Test')
    if evaluation_split not in {'inner','outer','test','inference'}:raise ValueError('Explicit inference role required')
    validate_policy(policy,binding=binding)
    if not isinstance(branch_shares,dict) or set(branch_shares)-set(ALL_CANDIDATES):raise ValueError('Unknown query branch keys')
    keys=policy['required_prediction_keys'];missing=[k for k in keys if k not in branch_shares]
    if missing:raise ValueError('Missing frozen required branches; no renormalization/fallback: '+','.join(missing))
    first=_vector(branch_shares[keys[0]],name=keys[0]);n=len(first)
    arrays={k:_vector(branch_shares[k],n,k) for k in keys};ww=np.asarray([policy['weights'][k] for k in keys])
    mode=policy['selected_mode'];masks=energy=None
    if mode=='single_light_fallback':result=arrays[keys[0]].copy()
    else:
        if mode=='logit_neural_mask_pool_with_tree_share':masks,energy=_grids(neural_masks,energy_weights,arrays,n)
        result=_predict(keys,ww,arrays,mode,masks,energy)
    if not np.isfinite(result).all() or (result < -2e-12).any() or (result>1+2e-5).any():raise RuntimeError('Invalid frozen ensemble prediction')
    result=np.clip(result,0,1)
    return {'share':result,'selected_mode':mode,'weights':dict(policy['weights']),'policy_sha256':policy['policy_sha256'],'fit_executed':False} if return_details else result
