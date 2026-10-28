"""Synthetic NumPy-only tests for the R5 frozen INNER ensemble; no data/model IO."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys
import numpy as np
from pipeline.hierarchical_ensemble_r5 import (
    ALL_CANDIDATES,NEURAL_KINDS,select_inner_policy,apply_policy,validate_policy,
    canonical_sha256,_predict,_fit_weights,
)

ROOT=Path(__file__).resolve().parents[1]
BINDING={'train_npz_sha256':'a'*64,'data_package_hash':'b'*64,'source_hash':'c'*64,'seed':42,'fold':0,'train_examples':100}

def blocked(call):
    try:call()
    except (ValueError,TypeError):return True
    raise AssertionError('A forbidden/invalid operation was accepted')

def assertions(policy,shares,n):
    before=deepcopy(policy)
    full=apply_policy(policy,shares,evaluation_split='outer',binding=BINDING)
    chunks=np.concatenate([apply_policy(policy,{k:v[a:b] for k,v in shares.items()},evaluation_split='test',binding=BINDING) for a,b in [(0,1),(1,37),(37,n)]])
    assert np.array_equal(full,chunks)
    permutation=np.random.default_rng(447).permutation(n)
    permuted=apply_policy(policy,{k:v[permutation] for k,v in shares.items()})
    assert np.array_equal(full[permutation],permuted)
    assert policy==before and np.isfinite(full).all() and ((0<=full)&(full<=1)).all()
    reload=json.loads(json.dumps(policy));assert np.array_equal(full,apply_policy(reload,shares))
    tampered=deepcopy(policy);tampered['weights'][next(iter(tampered['weights']))]+=.001
    blocked(lambda:apply_policy(tampered,shares))
    blocked(lambda:apply_policy(policy,shares,binding={**BINDING,'fold':1}))
    blocked(lambda:apply_policy(policy,shares,evaluation_split='outer',labels=np.zeros(n)))
    missing=dict(shares);missing.pop(policy['required_prediction_keys'][0])
    blocked(lambda:apply_policy(policy,missing))
    return {'query_chunk_and_row_permutation_bitwise':True,'JSON_policy_roundtrip_bitwise':True,
            'policy_not_modified_by_outer_Test_inference':True,'missing_required_branch_rejected':True,
            'policy_SHA_and_cross_fold_binding_tampering_rejected':True,'query_labels_rejected':True}

def main():
    def no_arrays(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            p=Path(args[0]).resolve()
            if str(p).startswith(str(ROOT)) and p.suffix.lower() in {'.npy','.npz','.csv','.parquet','.wav','.flac'}:
                raise RuntimeError('Fixture forbids project data/arrays: '+p.name)
    sys.addaudithook(no_arrays)
    n=240;u=np.tile(np.linspace(-1,1,48),5);y=.5+.25*u;groups=np.repeat([f'G{i}' for i in range(5)],48)
    perturb=.08*np.sin(np.arange(n)%48*2*np.pi/48)
    shares={'tfgrid_fdy_conformer':y+perturb,'bg_cnn':y-perturb,'lgbm_physics':y+.08*np.cos(np.arange(n)%48*2*np.pi/48),
            'timemoe_public':y+.13,'dasheng':y-.13,'fdy':y+.09,'itransformer':y-.09}
    policy=select_inner_policy(y,shares,groups,binding=BINDING)
    assert policy['selected_mode']=='convex_share_average' and len(policy['candidate_status'])==7
    assert all(w>0 for w in policy['weights'].values())
    assert all(policy['candidate_status'][k]['status']=='excluded_large_individual_gate' for k in ['timemoe_public','dasheng'])
    simple_checks=assertions(policy,shares,n)
    # The held-out group's labels cannot affect that meta-training partition.
    ychanged=y.copy();ychanged[groups=='G0']=np.clip(ychanged[groups=='G0']+.1,0,1)
    pchanged=select_inner_policy(ychanged,shares,groups,binding=BINDING)
    row=lambda p:next(r for r in p['source_meta_loo_results'] if r['heldout_reference_group']=='G0')
    assert row(policy)['weights_fitted_on_meta_train']==row(pchanged)['weights_fitted_on_meta_train']
    assert 'G0' not in row(policy)['meta_train_group_ids']
    for role in ['outer','test','outer_validation','train']:
        blocked(lambda role=role:select_inner_policy(y,shares,groups,fit_split=role,binding=BINDING))
    blocked(lambda:select_inner_policy(y,{k:v for k,v in shares.items() if k!='dasheng'},groups,binding=BINDING))
    blocked(lambda:select_inner_policy(y,shares,groups,binding=BINDING,ridge=.01))
    # A genuinely better large individual is eligible, without forcing weight.
    improved={**shares,'timemoe_public':y+.005*np.sin(np.arange(n)%48)}
    pp=select_inner_policy(y,improved,groups,binding=BINDING)
    assert pp['large_individual_admission']['timemoe_public']['admitted']
    assert not pp['large_individual_admission']['dasheng']['admitted']
    # Perfect light cannot be beaten by epsilon: deterministic single fallback.
    exact={k:np.full(n,.4) for k in ALL_CANDIDATES};exact['tfgrid_fdy_conformer']+=.05
    yy=np.full(n,.4);fallback=select_inner_policy(yy,exact,groups,binding=BINDING)
    assert fallback['selected_mode']=='single_light_fallback' and fallback['weights']=={'bg_cnn':1.}
    assert np.array_equal(apply_policy(fallback,exact),yy)
    # Missing large branches and other omitted candidates require stated reasons.
    omitted={k:'fixture deliberately omitted; not an implicit substitute' for k in ['timemoe_public','dasheng','lgbm_physics']}
    logits=np.tile(np.linspace(-2,2,48),5)
    sigmoid=lambda a:1/(1+np.exp(-a))
    q1=sigmoid(logits-1.4);q2=sigmoid(logits+1.4);truth=sigmoid(logits)
    ww=np.random.default_rng(2).random((n,15,25));ww[:,2]=0;ww/=ww.sum((1,2),keepdims=True)
    masks={'tfgrid_fdy_conformer':np.broadcast_to(q1[:,None,None],ww.shape).copy(),
           'bg_cnn':np.broadcast_to(q2[:,None,None],ww.shape).copy()}
    ss={k:(q*ww).sum((1,2)) for k,q in masks.items()}
    logit_policy=select_inner_policy(truth,ss,groups,binding=BINDING,explicit_exclusions=omitted,
        enable_logit_mask_candidate=True,neural_masks=masks,energy_weights=ww)
    assert logit_policy['selected_mode']=='logit_neural_mask_pool_with_tree_share'
    assert logit_policy['secondary_logit_candidate_strict_gate_passed']
    logit_full=apply_policy(logit_policy,ss,neural_masks=masks,energy_weights=ww)
    logit_chunk=np.concatenate([apply_policy(logit_policy,{k:v[a:b] for k,v in ss.items()},neural_masks={k:v[a:b] for k,v in masks.items()},energy_weights=ww[a:b]) for a,b in [(0,31),(31,n)]])
    assert np.array_equal(logit_full,logit_chunk) and np.max(np.abs(logit_full-truth))<1e-12
    blocked(lambda:apply_policy(logit_policy,ss,neural_masks=masks,energy_weights=ww*2))
    blocked(lambda:select_inner_policy(truth,ss,groups,binding=BINDING,explicit_exclusions=omitted,
        enable_logit_mask_candidate=True,neural_masks=masks,energy_weights=np.roll(ww,1,axis=0)*2))
    # Linear mask averaging identity, separate from nonlinear logit averaging.
    wmask=.3*masks['tfgrid_fdy_conformer']+.7*masks['bg_cnn']
    linear_error=float(np.max(np.abs((wmask*ww).sum((1,2))-(.3*ss['tfgrid_fdy_conformer']+.7*ss['bg_cnn']))))
    assert linear_error<1e-14
    tree=np.full(n,.2);keys=['tfgrid_fdy_conformer','bg_cnn','lgbm_physics'];weights=np.array([.3,.4,.3])
    blended=_predict(keys,weights,{**ss,'lgbm_physics':tree},'logit_neural_mask_pool_with_tree_share',masks,ww)
    expected=.7*sigmoid((.3*(logits-1.4)+.4*(logits+1.4))/.7)+.3*tree
    np.testing.assert_allclose(blended,expected,rtol=0,atol=2e-15)
    # Zero-only/single-row has no positive reference LOO: explicit light fallback.
    zz={k:np.zeros(1) for k in ALL_CANDIDATES}
    zero=select_inner_policy(np.zeros(1),zz,[[]],binding=BINDING)
    assert zero['selected_mode']=='single_light_fallback' and not zero['source_meta_loo_available']
    assert apply_policy(zero,zz).item()==0
    # QP objective beats every simplex vertex; no negative or nonunit weights.
    P=np.column_stack([shares[k] for k in ['tfgrid_fdy_conformer','bg_cnn','lgbm_physics']]);fit=_fit_weights(P,y,groups)
    objective=lambda a:float(np.mean((P@a-y)**2)+.05*np.sum((a-np.full(3,1/3))**2))
    assert (fit>=0).all() and abs(fit.sum()-1)<1e-12 and all(objective(fit)<=objective(v)+1e-12 for v in np.eye(3))
    before=deepcopy(BINDING);assert policy['training_binding']==before
    validate_policy(policy,binding=BINDING)
    report={'passed':True,'scope':'Synthetic NumPy-only ensemble fixture; no actual model/data/GPU work or RMSE improvement claim',
        'code_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['pipeline/hierarchical_ensemble_r5.py','ops/verify_ensemble_r5.py']},
        'fixture_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'all_seven_keys_reported':True,'outer_Test_fit_rejected':True,'outer_Test_query_labels_rejected':True,
        'unreported_missing_large_rejected':True,'explicit_missing_models_reported':True,
        'reference_meta_LOO_heldout_labels_do_not_change_heldout_fold_fit':True,
        'fixed_ridge_simplex_objective_check':True,'linear_mask_share_identity_max_abs_error':linear_error,
        'strict_secondary_logit_mode_selected_on_purpose_built_fixture':True,
        'logit_tree_formula_and_chunk_bitwise_passed':True,'common_energy_weight_mismatch_rejected':True,
        'zero_single_row_light_fallback_passed':True,'query_checks':simple_checks,
        'simple_fixture_policy':policy,'logit_fixture_policy':logit_policy,
        'zero_fixture_policy':zero,'actual_Fold_training_or_Test_arrays_accessed':False}
    out=ROOT/'reports/R5_ENSEMBLE_FIXTURE.json';out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'report':str(out),'code_sha256':report['code_sha256'],'simple_mode':policy['selected_mode'],
        'logit_mode':logit_policy['selected_mode'],'linear_identity_error':linear_error},indent=2))

if __name__=='__main__':main()
