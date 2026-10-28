"""Inference-safe feature quantization and TRAIN-only observation censoring.

Raw observation energy defines the pooling weights and controlled-addition
labels. Rounding db to 0.1 is a feature representation, not a new physical
measurement or permission to copy labels onto rounded-energy weights.

The endpoint policy is a predeclared generic candidate (.2, up to 2 minutes,
minimum 3 observed minutes), unrelated to Test frequencies/site identities.
Random proposals depend only on seed/epoch/sample index. Missing endpoints
are skipped; insufficient-support proposals are rejected without redrawing.
Known q/y remain supervision, never query inputs or teacher p_ship aliases.
"""
from dataclasses import dataclass,asdict
import numpy as np


@dataclass(frozen=True)
class EndpointDropoutPolicy:
    probability: float = .2
    max_drop: int = 2
    min_observed_minutes: int = 3

    def __post_init__(self):
        if not np.isfinite(self.probability) or not 0<=self.probability<=1:raise ValueError('Invalid endpoint probability')
        for name in ['max_drop','min_observed_minutes']:
            v=getattr(self,name)
            if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)):raise ValueError('Integer policy value required: '+name)
        if not 1<=self.max_drop<15 or not 3<=self.min_observed_minutes<=15:raise ValueError('Unsupported endpoint/support policy')


def _db(db,min_observed=3):
    if isinstance(db,dict):raise TypeError('Pass db only, never a labelled dictionary')
    x=np.asarray(db)
    if x.ndim!=3 or x.shape[1:]!=(15,25) or not len(x) or not np.issubdtype(x.dtype,np.floating):raise ValueError('Expected nonempty floating [N,15,25] db')
    observed=np.isfinite(x)
    if np.isinf(x).any() or not (observed.any(2).sum(1)>=min_observed).all():raise ValueError('Invalid/inadequate observed support')
    return x,observed


def _energy_weights(db):
    observed=np.isfinite(db);x=db.astype('float64');peak=np.nanmax(x,(1,2),keepdims=True)
    e=np.where(observed,10**((x-peak)/10),0)
    if not (e[observed]>0).all() or not np.isfinite(e).all():raise ValueError('Unsupported observed energy underflow/overflow')
    mass=e.sum((1,2),keepdims=True)
    if not (mass>0).all():raise ValueError('Zero observed energy')
    return e,e/mass


def prepare_query_observations(db,*,quantization_decimals=1):
    """Accept db ONLY. Return owned arrays; physical_db is a read-only copy.

    energy_weights always comes from raw db. feature_energy_weights is an
    explicitly named diagnostic for the rounded representation. Defaults do
    not substitute it into physical pooling. None disables feature rounding.
    Normalization is computed in float64; returned weights use float32.
    """
    x,observed=_db(db)
    if quantization_decimals is not None:
        if isinstance(quantization_decimals,(bool,np.bool_)) or not isinstance(quantization_decimals,(int,np.integer)) or not 0<=quantization_decimals<=6:raise ValueError('Quantization decimals must be None or an integer 0..6')
        feature=np.round(x.astype('float64'),int(quantization_decimals)).astype(x.dtype)
    else:feature=x.copy()
    raw=x.copy();raw.setflags(write=False)
    _,physical_weights=_energy_weights(raw);_,feature_weights=_energy_weights(feature)
    return {'feature_db':feature,'physical_db':raw,'energy_weights':physical_weights.astype('float32'),
        'feature_energy_weights':feature_weights.astype('float32'),'observed_mask':observed.copy()}


def _integer(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<0:raise ValueError('Nonnegative integer '+name+' required')
    return int(value)


def _proposal(seed,epoch,index,policy):
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,index,0x4F425335]))
    # Draw all choices BEFORE consulting observed support or any labels.
    requested=bool(rng.random()<policy.probability);last=bool(rng.integers(2));count=int(rng.integers(1,policy.max_drop+1))
    positions=list(range(15-count,15)) if last else list(range(count))
    return requested,'last' if last else 'first',count,positions


def augment_training_observations(db,cell_share,*,epoch,sample_indices,seed=42,
                                   policy=EndpointDropoutPolicy(),split='train',quantization_decimals=1):
    """Censor drawn endpoint minutes and recompute q/w/y/presence exactly.

    For O' after censoring, E'=O' * 10^(raw_db/10), w'=E'/sum(E'),
    q'=q on O' and zero elsewhere, y'=sum(q'E')/sum(E'). Presence is y'>0.
    Erasing all known added support can legitimately yield y'=presence'=0;
    that transition is audited rather than replaced with the original label.
    Raw energy, not rounded feature_db, defines this target. No retries.
    """
    if split!='train':raise ValueError('Observation dropout is TRAIN-only')
    if not isinstance(policy,EndpointDropoutPolicy):raise TypeError('Explicit EndpointDropoutPolicy required')
    seed=_integer(seed,'seed');epoch=_integer(epoch,'epoch');x,observed=_db(db,policy.min_observed_minutes)
    q=np.asarray(cell_share)
    if q.shape!=x.shape or not np.issubdtype(q.dtype,np.floating) or not np.isfinite(q).all() or ((q<0)|(q>1)).any():raise ValueError('Known q must be finite floating [N,15,25] values in [0,1]')
    if (q[~observed]!=0).any():raise ValueError('Known q must be zero in originally unobserved cells')
    ids=np.asarray(sample_indices)
    if ids.shape!=(len(x),) or not np.issubdtype(ids.dtype,np.integer) or np.issubdtype(ids.dtype,np.bool_) or (ids<0).any():raise ValueError('Stable nonnegative integer sample indices are required')
    physical=x.copy();new_q=q.copy();records=[]
    _,old_w=_energy_weights(x);old_y=(old_w*q.astype('float64')).sum((1,2))
    for row,index in enumerate(ids):
        requested,side,count,positions=_proposal(seed,epoch,int(index),policy)
        before=int(observed[row].any(1).sum());visible=[j for j in positions if observed[row,j].any()]
        removed=[];reason='not_requested'
        if requested:
            if not visible:reason='drawn_endpoint_already_unobserved'
            elif before-len(visible)<policy.min_observed_minutes:reason='minimum_observed_support_protected'
            else:
                physical[row,positions,:]=np.nan;new_q[row,positions,:]=0;removed=visible;reason='applied'
        records.append({'sample_index':int(index),'requested':requested,'endpoint':side,'requested_count':count,
            'proposed_positions':positions,'observed_proposed_positions':visible,'removed_observed_positions':removed,
            'reason':reason,'observed_minutes_before':before,'observed_minutes_after':before-len(removed)})
    result=prepare_query_observations(physical,quantization_decimals=quantization_decimals)
    new_q[~result['observed_mask']]=0
    _,new_w64=_energy_weights(result['physical_db']);target=(new_q.astype('float64')*new_w64).sum((1,2))
    share=target.astype('float32')
    if ((target>0)&(share==0)).any():raise ValueError('Positive observed target would underflow float32')
    presence=(share>0).astype('float32')
    result.update({'cell_share':new_q,'share':share,'presence':presence})
    # The alternate feature-derived aggregate is DIAGNOSTIC, not a target.
    feature_aggregate=(new_q.astype('float64')*result['feature_energy_weights'].astype('float64')).sum((1,2))
    stored_aggregate=(new_q.astype('float64')*result['energy_weights'].astype('float64')).sum((1,2))
    audit={'scope':'TRAIN-only observation augmentation, no Test-derived policy','policy':asdict(policy),
        'seed':seed,'epoch':epoch,'quantization_decimals':quantization_decimals,'input_windows':len(x),
        'proposal_depends_on_observations_or_labels':False,'resampled_or_retried_proposals':0,
        'requested_events':sum(r['requested'] for r in records),'applied_events':sum(r['reason']=='applied' for r in records),
        'already_unobserved_skips':sum(r['reason']=='drawn_endpoint_already_unobserved' for r in records),
        'minimum_support_protection_skips':sum(r['reason']=='minimum_observed_support_protected' for r in records),
        'removed_observed_minutes':sum(len(r['removed_observed_positions']) for r in records),
        'positive_to_zero_after_removing_known_support':int(((old_y>0)&(target==0)).sum()),
        'maximum_float32_raw_weight_target_residual':float(np.max(np.abs(stored_aggregate-share))),
        'maximum_feature_weight_vs_raw_target_difference':float(np.max(np.abs(feature_aggregate-target))),
        'physical_weights_use_raw_observations':True,'physical_db_is_readonly_owned_copy':True,
        'known_labels_are_not_p_ship':True,'records':records}
    return result,audit
