"""Predeclared INNER-source ensemble gate. Never accepts OUTER inputs."""
import json
import numpy as np
from scipy.optimize import minimize


def rmse(y,p):return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def source_groups(values):
    result=[]
    for value in values:
        refs=json.loads(value) if isinstance(value,str) else value
        if not isinstance(refs,list):raise ValueError('Reference-group metadata must contain JSON lists')
        result.append('|'.join(sorted(map(str,refs))) if refs else '__no_added_reference__')
    return np.asarray(result)


def fit_weights(P,y,groups,anchor,ridge):
    """One convex, group-balanced fit with a fixed ridge coefficient."""
    _,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
    row_weight=1./counts[inverse];row_weight/=row_weight.mean()
    def objective(w):
        error=P@w-y
        return float(np.mean(row_weight*error**2)+ridge*np.sum((w-anchor)**2))
    def gradient(w):
        return 2*(P.T@(row_weight*(P@w-y)))/len(y)+2*ridge*(w-anchor)
    fit=minimize(objective,anchor.copy(),jac=gradient,method='SLSQP',bounds=[(0.,1.)]*len(anchor),
        constraints={'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones_like(w)},
        options={'ftol':1e-12,'maxiter':300})
    if not fit.success or not np.isfinite(fit.x).all():raise RuntimeError('INNER safety fit failed: '+fit.message)
    weights=np.maximum(fit.x,0.);weights/=weights.sum()
    return weights


def select_inner_policy(P,y,models,reference_groups,settings):
    """Fit on INNER only, gate against CNN and fixed three-mask mean.

    Source-LOO checks meta-learner transfer across INNER components; it does
    not claim the base learners were retrained for this meta-level crossfit.
    The user-specified fallback is fixed before any OUTER values are opened.
    """
    P=np.asarray(P,dtype=float);y=np.asarray(y,dtype=float)
    if models!=['lgbm_retrieval','cnn','fdy','itransformer']:
        raise ValueError('R4 requires exactly the four predeclared candidates in order')
    if P.shape!=(len(y),4) or not np.isfinite(P).all() or not np.isfinite(y).all():
        raise ValueError('Invalid INNER prediction matrix')
    if (P<0).any() or (P>1).any():raise ValueError('INNER predictions outside [0,1]')
    ridge=float(settings.get('ridge_lambda',.05));epsilon=float(settings.get('epsilon_rmse',1e-4))
    if ridge!=.05 or epsilon!=1e-4 or settings.get('outer_gate_changes_predictions') is not False:
        raise ValueError('R4 safety policy must be predeclared, not tuned against OOF')
    anchor=np.array([0.,1/3,1/3,1/3]);groups=source_groups(reference_groups)
    if len(groups)!=len(y):raise ValueError('INNER groups do not align with predictions')
    positive_groups=sorted(set(groups)-{'__no_added_reference__'})
    base={'cnn':rmse(y,P[:,1]),'mask_mean':rmse(y,P@anchor)}
    detail={'version':'inner_source_crossfit_safety_r4_v1','fit_split':'inner validation only',
        'outer_used_for_fit':False,'ridge_lambda':ridge,'epsilon_rmse':epsilon,
        'fallback_weights':anchor.tolist(),'fallback':'predeclared equal three-mask mean',
        'inner_baselines':base,'positive_reference_groups':positive_groups,
        'group_balance_for_weight_fit':True,'source_loo_is_meta_level_only':True,
        'outer_gate_changes_predictions':False}
    if len(positive_groups)<3:
        detail.update({'candidate_accepted':False,'reason':'fewer than three positive INNER reference components'})
        return anchor,detail
    try:candidate=fit_weights(P,y,groups,anchor,ridge)
    except RuntimeError as exc:
        detail.update({'candidate_accepted':False,'reason':'fixed fallback: full INNER optimization failed','optimization_error':str(exc)})
        return anchor,detail
    crossfit=np.empty(len(y));heldout=[]
    for group in sorted(set(groups)):
        validation=groups==group;training=~validation
        try:w=fit_weights(P[training],y[training],groups[training],anchor,ridge)
        except RuntimeError as exc:
            detail.update({'candidate_accepted':False,'reason':'fixed fallback: INNER source crossfit failed','optimization_error':str(exc)})
            return anchor,detail
        crossfit[validation]=P[validation]@w
        heldout.append({'heldout_group':group,'n':int(validation.sum()),'weights':w.tolist(),
            'candidate_rmse':rmse(y[validation],crossfit[validation]),
            'cnn_rmse':rmse(y[validation],P[validation,1]),
            'mask_mean_rmse':rmse(y[validation],P[validation]@anchor)})
    full=rmse(y,P@candidate);cross=rmse(y,crossfit)
    full_pass=all(full+epsilon<value for value in base.values())
    cross_pass=all(cross+epsilon<value for value in base.values())
    accepted=bool(full_pass and cross_pass)
    detail.update({'candidate_weights':candidate.tolist(),'fullfit_rmse':full,
        'source_crossfit_rmse':cross,'fullfit_gate_passed':bool(full_pass),
        'crossfit_gate_passed':bool(cross_pass),'candidate_accepted':accepted,
        'source_loo_results':heldout,'reason':'both INNER gates passed' if accepted else 'INNER improvement gate failed; use fixed three-mask mean'})
    return (candidate if accepted else anchor),detail


def outer_safety_report(pooled_metrics,epsilon=1e-4):
    """Report after freezing predictions; never switch or refit an OOF policy."""
    value=pooled_metrics['ensemble']['rmse'];cnn=pooled_metrics['cnn']['rmse']
    fallback=pooled_metrics['mask_mean']['rmse']
    return {'ensemble_rmse':value,'cnn_rmse':cnn,'mask_mean_rmse':fallback,
        'improves_cnn':bool(value+epsilon<cnn),'improves_mask_mean':bool(value+epsilon<fallback),
        'four_way_release_gate_passed':bool(value+epsilon<min(cnn,fallback)),
        'outer_changed_weights_or_predictions':False,
        'interpretation':'Post-fit development acceptance evidence only; no leaderboard superiority claim.'}
