"""Synthetic CPU fixtures for observation masking/rounding; no datasets/models."""
from pathlib import Path
import hashlib,json
import numpy as np
from pipeline.observation_augmentation_r5 import (EndpointDropoutPolicy,
    prepare_query_observations,augment_training_observations)

ROOT=Path(__file__).resolve().parents[1]

def oracle(db,q):
    o=np.isfinite(db);x=db.astype(float);peak=np.nanmax(x,(1,2),keepdims=True)
    e=np.zeros_like(x);np.power(10,(x-peak)/10,out=e,where=o)
    e[~o]=0;w=e/e.sum((1,2),keepdims=True)
    return w,(np.where(o,q,0)*w).sum((1,2))

def verify():
    rng=np.random.default_rng(19);n=256
    raw=rng.normal(65,18,(n,15,25)).astype('float32');raw[:,5]=np.nan;raw[::3,2,10:]=np.nan
    q=rng.uniform(0,1,raw.shape).astype('float32');q[~np.isfinite(raw)]=0
    raw_before=raw.copy();q_before=q.copy();ids=np.arange(1000,1000+n,dtype=np.int64)
    policy=EndpointDropoutPolicy();args={'epoch':3,'sample_indices':ids,'seed':42,'policy':policy}
    result,audit=augment_training_observations(raw,q,**args)
    w,y=oracle(result['physical_db'],result['cell_share'])
    np.testing.assert_allclose(result['energy_weights'],w,atol=3e-8,rtol=1e-6)
    np.testing.assert_allclose(result['share'],y,atol=1e-7,rtol=0)
    assert np.array_equal(result['presence'],(y>0).astype('float32'))
    assert (result['cell_share'][~result['observed_mask']]==0).all()
    assert (result['observed_mask'].any(2).sum(1)>=3).all()
    assert np.array_equal(raw,raw_before,equal_nan=True) and np.array_equal(q,q_before)
    assert not result['physical_db'].flags.writeable and not np.shares_memory(result['physical_db'],raw)
    assert not np.shares_memory(result['feature_db'],result['physical_db'])
    replay,replay_audit=augment_training_observations(raw,q,**args)
    chunks=[];chunk_audits=[]
    for begin in range(0,n,13):
        out,a=augment_training_observations(raw[begin:begin+13],q[begin:begin+13],epoch=3,
            sample_indices=ids[begin:begin+13],seed=42,policy=policy);chunks.append(out);chunk_audits.append(a)
    for key in result:
        assert np.array_equal(result[key],replay[key],equal_nan=True),key
        assert np.array_equal(result[key],np.concatenate([c[key] for c in chunks]),equal_nan=True),key
    assert audit==replay_audit
    assert audit['records']==[row for a in chunk_audits for row in a['records']]
    for key in ['input_windows','requested_events','applied_events','already_unobserved_skips','minimum_support_protection_skips','removed_observed_minutes','positive_to_zero_after_removing_known_support']:
        assert audit[key]==sum(a[key] for a in chunk_audits),key
    different,different_audit=augment_training_observations(raw,q,epoch=4,sample_indices=ids,seed=42)
    assert not np.array_equal(result['observed_mask'],different['observed_mask'])
    # Proposals cannot depend on the mask or known labels. Only application
    # protection/skip decisions may change when observation support changes.
    dense=np.zeros((n,15,25),dtype='float32');last_q=np.zeros_like(dense);last_q[:,14]=1
    erased,erased_audit=augment_training_observations(dense,last_q,**args)
    no_q=np.zeros_like(dense);_,zero_audit=augment_training_observations(dense,no_q,**args)
    missing=dense.copy();missing[:,:2]=np.nan;missing[:,13:]=np.nan
    missing_q=np.where(np.isfinite(missing),last_q,0)
    missing_out,missing_audit=augment_training_observations(missing,missing_q,**args)
    columns=['sample_index','requested','endpoint','requested_count','proposed_positions']
    proposal=lambda a:[{k:r[k] for k in columns} for r in a['records']]
    assert proposal(audit)==proposal(erased_audit)==proposal(zero_audit)==proposal(missing_audit)
    assert erased_audit['positive_to_zero_after_removing_known_support']>0
    for i,row in enumerate(erased_audit['records']):
        if row['reason']=='applied' and row['endpoint']=='last':
            assert erased['share'][i]==0 and erased['presence'][i]==0 and not erased['cell_share'][i].any()
    assert missing_audit['requested_events']>0 and missing_audit['already_unobserved_skips']==missing_audit['requested_events']
    assert np.array_equal(missing_out['physical_db'],missing,equal_nan=True)
    sparse=np.full_like(dense,np.nan);sparse[:,[0,7,14]]=0;sparse_q=np.where(np.isfinite(sparse),.2,0).astype('float32')
    protected,protected_audit=augment_training_observations(sparse,sparse_q,**args)
    assert protected_audit['minimum_support_protection_skips']==protected_audit['requested_events']>0
    assert np.array_equal(protected['physical_db'],sparse,equal_nan=True)
    assert np.array_equal(protected['cell_share'],sparse_q)
    # A small, independently calculable counterexample: rounded features
    # have different energy weights, although raw physical targets do not.
    golden=np.zeros((1,15,25),dtype='float64');golden[0,0,0]=.049
    golden_q=np.zeros_like(golden);golden_q[0,0,0]=1
    prepared=prepare_query_observations(golden);control=prepare_query_observations(golden,quantization_decimals=None)
    raw_w,raw_y=oracle(golden,golden_q);rounded_w,rounded_y=oracle(prepared['feature_db'],golden_q)
    assert np.array_equal(prepared['energy_weights'],control['energy_weights'])
    assert np.array_equal(control['feature_db'],golden,equal_nan=True)
    assert np.array_equal(control['feature_energy_weights'],control['energy_weights'])
    assert abs(float(raw_y[0]-rounded_y[0]))>1e-6
    assert np.max(np.abs(raw_w-rounded_w))>1e-6
    np.testing.assert_allclose(prepared['energy_weights'],raw_w,atol=1e-9,rtol=1e-6)
    rejected={}
    def rejects(name,call):
        try:call()
        except (ValueError,TypeError):rejected[name]=True
        else:raise AssertionError('Expected rejection: '+name)
    rejects('non_train_split',lambda:augment_training_observations(raw,q,**args,split='validation'))
    rejects('query_oracle_argument',lambda:prepare_query_observations(raw,cell_share=q))
    rejects('query_labeled_dict',lambda:prepare_query_observations({'db':raw,'share':q}))
    broken=q.copy();broken[0,5,0]=.1
    rejects('q_in_missing_cell',lambda:augment_training_observations(raw,broken,**args))
    invalid=q.copy();invalid[0,0,0]=float('nan')
    rejects('nonfinite_q',lambda:augment_training_observations(raw,invalid,**args))
    invalid=q.copy();invalid[0,0,0]=1.01
    rejects('q_out_of_range',lambda:augment_training_observations(raw,invalid,**args))
    rejects('all_observations_missing',lambda:prepare_query_observations(np.full((1,15,25),np.nan)))
    insufficient=np.full((1,15,25),np.nan);insufficient[:,0]=0
    rejects('insufficient_initial_observations',lambda:prepare_query_observations(insufficient))
    rejects('noninteger_sample_indices',lambda:augment_training_observations(raw,q,**{**args,'sample_indices':ids+.5}))
    rejects('write_physical_copy',lambda:prepared['physical_db'].__setitem__((0,0,0),10))
    report={'status':'passed observation augmentation CPU fixtures','scope':'Synthetic CPU observations/known-q fixtures only',
        'production_candidate_policy':{'probability':.2,'max_drop':2,'min_observed_minutes':3},
        'policy_uses_Test_frequencies_or_site_weights':False,'raw_inputs_and_known_q_unchanged':True,
        'physical_db_readonly_owned_copy':True,'same_epoch_index_seed_and_chunk_bitwise':True,
        'different_epoch_changes_proposals':True,'proposals_independent_of_observations_and_labels':True,
        'all_original_and_new_missing_q_zero':True,'raw_energy_weighted_y_and_presence_recomputed':True,
        'minimum_support_protection_verified':True,'hidden_known_addition_can_become_zero':True,
        'default_audit_counts':{k:v for k,v in audit.items() if k!='records'},
        'erased_known_addition_count':erased_audit['positive_to_zero_after_removing_known_support'],
        'already_unobserved_skip_count':missing_audit['already_unobserved_skips'],
        'minimum_support_skip_count':protected_audit['minimum_support_protection_skips'],
        'quantization_counterexample':{'raw_first_cell_db':.049,'rounded_first_cell_db':float(prepared['feature_db'][0,0,0]),
            'physical_raw_y':float(raw_y[0]),'rounded_feature_energy_aggregate':float(rounded_y[0]),
            'difference_if_wrongly_pooling_rounded_energy':float(rounded_y[0]-raw_y[0]),
            'maximum_physical_vs_feature_weight_difference':float(np.max(np.abs(raw_w-rounded_w))),
            'returned_default_physical_weights_remain_raw':True,'quantization_None_control_passed':True},
        'guards_rejected':rejected,'R4_or_grid_inputs_modified':False,
        'actual_training_Test_outer_or_GPU_used_in_these_fixtures':False,
        'note':'Separate parent-requested R4 metadata alignment check is not an augmentation input or experiment.',
        'code_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['pipeline/observation_augmentation_r5.py','ops/verify_observation_augmentation_r5.py']}}
    (ROOT/'reports/R5_OBSERVATION_AUGMENTATION_FIXTURE.json').write_text(json.dumps(report,indent=2,allow_nan=False));return report

if __name__=='__main__':
    d=verify();print(json.dumps({'status':d['status'],'quantization_counterexample':d['quantization_counterexample'],
        'default_applied_events':d['default_audit_counts']['applied_events'],
        'legitimate_positive_to_zero':d['erased_known_addition_count'],
        'report':'reports/R5_OBSERVATION_AUGMENTATION_FIXTURE.json'},indent=2))
