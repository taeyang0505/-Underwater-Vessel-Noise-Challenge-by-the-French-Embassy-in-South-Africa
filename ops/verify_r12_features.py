"""Numeric-only R12 feature/PCA/state verification; no acoustic datasets.

The basis produced here is explicitly a disposable numeric fixture, never
production training state. Old R9 source/code/data remain untouched.
"""
from __future__ import annotations
import copy,hashlib,inspect,json,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline.features_r12 import (fit_spectral_basis,transform_channels,compact_relative_features,
    compact_feature_names,save_spectral_basis,load_spectral_basis,validate_spectral_basis,FEATURE_NAMES)
from pipeline.relative_grid_r6_v2 import relative_grid_v2
from pipeline.relative_grid_r6 import quantized_db,quantization_pooling_bound


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ahash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def raises(fn):
    try:fn()
    except (TypeError,ValueError,FileExistsError):return True
    raise AssertionError('Invalid input/state was silently accepted')


def main():
    folder=ROOT/'reports/R12_FEATURES_FIXTURE';folder.mkdir(exist_ok=False)
    old=('pipeline/relative_grid_r6.py','pipeline/relative_grid_r6_v2.py')
    pins={n:sha(ROOT/n) for n in old}
    rng=np.random.default_rng(20261027);axis=np.linspace(-1,1,23)
    candidate=np.stack([axis,axis*axis-np.mean(axis*axis),np.sin(np.pi*axis)],1)
    axes=np.linalg.qr(candidate)[0].T
    profiles=(rng.normal(size=(32,15,3))*4)@axes
    profile_base=3*axis+2*(axis*axis-np.mean(axis*axis))
    train=np.empty((32,15,25),float);train[:,:,:23]=75+profile_base+profiles
    train[:,:,23:]=65+rng.normal(0,4,(32,15,2))
    train[0,:2,:]=np.nan;train[1,0,17]=np.nan
    ids=np.array(['numeric_train_background_'+str(i) for i in range(32)])
    binding={'fixture_only':True,'source_role':'TRAIN','background_kind':'pure_SA_no_added_component_numeric_analogue','source_count':32}
    before=ahash(train)
    basis=fit_spectral_basis(train,ids,training_binding=binding,max_fit_spectra=256)
    assert ahash(train)==before
    assert basis==fit_spectral_basis(train,ids,training_binding=binding,max_fit_spectra=256)
    validate_spectral_basis(basis,expected_training_binding=binding)
    assert np.allclose(np.asarray(basis['components'])@np.asarray(basis['components']).T,np.eye(3),atol=1e-12)
    query=train[:8].copy();query[2,:,7]=np.nan;query[3,3,:]=np.nan;query[4,5,:23]=np.nan
    query[5,4,9]=np.nan
    x8,w=transform_channels(query);frozen,frozenw=relative_grid_v2(query)
    x11,w11=transform_channels(query,basis,True)
    assert np.array_equal(x8,frozen) and np.array_equal(w,frozenw)
    assert np.array_equal(x11[:,:8],frozen) and np.array_equal(w11,frozenw)
    assert x11.shape==(8,11,15,25) and w.shape==(8,15,25)
    observed=np.isfinite(query);assert np.isfinite(x11).all() and np.isfinite(w11).all()
    assert not x11.transpose(0,2,3,1)[~observed].any() and not w11[~observed].any()
    assert np.allclose(w11.sum((1,2)),1,atol=1e-7) and np.array_equal(x11[:,5],(w11>0).astype(np.float32))
    assert np.array_equal(x11[:,9],np.maximum(x11[:,8],0)) and not x11[:,8:, :,23:].any()
    assert not x11[5,10,4,10] # missing neighbor9 forbids f10 curvature
    chunks=np.concatenate([transform_channels(query[i:i+1],basis,True)[0] for i in range(len(query))])
    chunk_error=float(np.max(abs(chunks-x11)));assert chunk_error<1e-6
    # Added maps must not use unsafe bands23/24 at all. Original8/w intentionally do.
    unsafe=query.copy();unsafe[:,:,23:]+=40
    other,otherw=transform_channels(unsafe,basis,True)
    unsafe_error=float(np.max(abs(other[:,8:]-x11[:,8:])));assert unsafe_error==0
    assert not np.array_equal(otherw,w11)
    # Common gain on a 0.1 grid: finite floating roundoff only. Arbitrary gain
    # can cross different quantization bins and is NOT claimed exact.
    grid=quantized_db(query);base,bw=transform_channels(grid,basis,True)
    gains=[-.7,7.4,80.];gain_errors=[];werrors=[]
    for gain in gains:
        a,b=transform_channels(grid+gain,basis,True);gain_errors.append(float(np.max(abs(a-base))));werrors.append(float(np.max(abs(b-bw))))
        assert np.allclose(a,base,atol=2e-6,rtol=0) and np.allclose(b,bw,atol=1e-7,rtol=0)
    arbitrary,_=transform_channels(query+.037,basis,True)
    arbitrary_error=float(np.max(abs(arbitrary-base)))
    assert np.isfinite(arbitrary_error)
    altered=grid.copy();altered[:,:,14]+=4
    changed,_=transform_channels(altered,basis,True)
    static_offset_effect=float(np.max(abs(changed[:,8:]-base[:,8:])));assert static_offset_effect>.01
    gain_fit=fit_spectral_basis(quantized_db(train)+7.4,ids,training_binding=binding,max_fit_spectra=256)
    # Differences in raw input SHA are correct; feature subspace is invariant.
    c=np.asarray(basis['components']);cg=np.asarray(gain_fit['components'])
    projector_error=float(np.max(abs(c.T@c-cg.T@cg)));assert projector_error<1e-10
    # Out-of-subspace, persistent added log-spectral structure is detectable
    # as a residual. It is NOT an estimated ship-energy fraction.
    probe=75+profile_base[None,None,:]+profiles[10:11]
    pure=np.concatenate([probe,np.full((1,15,2),65.)],2)
    residual,_=transform_channels(pure,basis,True)
    peaked=pure.copy();peaked[:,:,14]+=6
    residual_peak,_=transform_channels(peaked,basis,True)
    natural_residual=float(np.mean(abs(residual[:,8])));novel_residual=float(np.mean(abs(residual_peak[:,8])))
    assert novel_residual>natural_residual+.01
    # Known components keep their raw target; these transformations never
    # accept/rewrite it. Rounded weights obey the declared quantization bound.
    b=10**((pure-pure.max())/10);a=np.zeros_like(b);a[:,4:10,10:20]=.7*b[:,4:10,10:20]
    e=b+a;rawdb=10*np.log10(e)+pure.max();q=a/e;truth=float(a.sum()/e.sum())
    _,rw=transform_channels(rawdb,basis,True);rounded_gap=abs(float((rw*q).sum())-truth)
    assert rounded_gap<quantization_pooling_bound(.05)+1e-6
    compact=compact_relative_features(query);shapecompact=compact_relative_features(query,basis,True)
    assert compact.shape==(8,466) and shapecompact.shape==(8,673)
    assert len(compact_feature_names())==466 and len(compact_feature_names(True))==673
    cchunks=np.concatenate([compact_relative_features(query[i:i+1],basis,True) for i in range(8)])
    compact_error=float(np.max(abs(cchunks-shapecompact)));assert compact_error<1e-6
    state=folder/'numeric_fixture_basis_DO_NOT_USE_FOR_TRAINING.npz'
    receipt=save_spectral_basis(basis,state)
    restored=load_spectral_basis(state,expected_sha256=receipt['file_sha256'],expected_training_binding=binding)
    assert basis==restored
    restoredx,restoredw=transform_channels(query,restored,True)
    assert np.array_equal(restoredx,x11) and np.array_equal(restoredw,w11)
    errors={}
    errors['validation_fit_rejected']=raises(lambda:fit_spectral_basis(train,ids,split='validation'))
    errors['Test_fit_rejected']=raises(lambda:fit_spectral_basis(train,ids,split='Test'))
    errors['synthetic_fit_rejected']=raises(lambda:fit_spectral_basis(train,ids,source_kind='synthetic_mixture'))
    errors['duplicate_source_ids_rejected']=raises(lambda:fit_spectral_basis(train,np.repeat('same',32)))
    errors['wrong_component_count_rejected']=raises(lambda:fit_spectral_basis(train,ids,n_components=2))
    errors['missing_basis_rejected']=raises(lambda:transform_channels(query,include_shape=True))
    errors['query_labels_rejected']=raises(lambda:transform_channels({'db':query,'share':np.zeros(8)}))
    errors['query_extra_q_argument_rejected']=raises(lambda:transform_channels(query,q=np.zeros_like(query)))
    tampered=copy.deepcopy(basis);tampered['components'][0][0]+=.1
    errors['basis_tamper_rejected']=raises(lambda:transform_channels(query,tampered,True))
    errors['foreign_TRAIN_binding_rejected']=raises(lambda:load_spectral_basis(state,expected_training_binding={'other':True}))
    errors['file_SHA_mismatch_rejected']=raises(lambda:load_spectral_basis(state,expected_sha256='0'*64))
    errors['basis_overwrite_rejected']=raises(lambda:save_spectral_basis(basis,state))
    errors['infinite_query_rejected']=raises(lambda:transform_channels(np.full((1,15,25),np.inf),basis,True))
    errors['entirely_missing_query_rejected']=raises(lambda:transform_channels(np.full((1,15,25),np.nan),basis,True))
    assert all(errors.values()) and all(sha(ROOT/n)==v for n,v in pins.items())
    report={'status':'passed','fixture_only':True,'new_acoustic_data_or_Test_or_val_fit_read':False,'new_model_training_inference':False,
        'script_sha256':sha(__file__),'module_sha256':sha(ROOT/'pipeline/features_r12.py'),'unchanged_frozen_source_sha256':pins,
        'schema':{'channels_base':8,'channels_shape':11,'compact_base':466,'compact_shape':673,'channels':FEATURE_NAMES},
        'PCA_numeric_fixture':{'TRAIN_windows':32,'eligible_complete_safe_spectra':basis['eligible_complete_spectra'],'fitted_spectra':basis['fitted_spectra'],
            'source_inputs_unchanged':True,'repeatfit_state_exact':True,'basis_content_sha256':basis['basis_sha256'],
            'explained_variance_ratio':basis['explained_variance_ratio'],'fixture_basis_receipt':receipt},
        'numeric_checks':{'legacy8_and_w_bitwise_equal':True,'new_missing_cells_all_zero':True,'positive_is_relu_signed_exact':True,
            'safe_extra_channels_unsafe_band_perturbation_error':unsafe_error,'chunk_transform_max_error':chunk_error,'chunk_compact_max_error':compact_error,
            'grid_common_gain_max_errors':dict(zip(map(str,gains),gain_errors)),'grid_common_gain_w_max_errors':dict(zip(map(str,gains),werrors)),
            'arbitrary_prequant_gain_.037_max_feature_change_not_exactly_invariant':arbitrary_error,'static_band_offset_extra_feature_change_expected':static_offset_effect,
            'fit_common_gain_PCA_projector_error':projector_error,'TRAIN_manifold_baseline_absolute_residual':natural_residual,
            'unseen_spectral_peak_absolute_residual':novel_residual,'known_raw_label_rounded_pool_gap':rounded_gap,
            'known_raw_label_rounding_theoretical_bound':quantization_pooling_bound(.05),'NPZ_reload_channels_weights_bitwise_equal':True},
        'guards':errors,'pure_NumPy_no_torch_import':'torch' not in sys.modules,
        'limits':['Numerical input alone cannot certify TRAIN/A0 provenance: caller must verify source IDs and binding.',
            'A0 means no controlled addition, not natural-vessel absence.',
            'New residuals are not isolated-vessel energy, certified background or presence labels.',
            'Added maps safe23 protection does not remove unsafe bands from unchanged base8/full25 physical weights.',
            'Before rounding, arbitrary common gain has quantization error; static band offsets can change new shape maps.',
            'Numeric fixture basis is never production training state.']}
    (folder/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (ROOT/'reports/R12_FEATURES_FIXTURE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    md='''# R12 특징 수학·물리 검증\n\n**PASS — 숫자 fixture만 실행했습니다. 실제 음향 데이터 fit·모델 학습·추론·Test/validation 읽기는 0회입니다.**\n\n기존 relative8과 rounded physical w는 bitwise 동일합니다. 옵션 shape는 signed PCA 잔차/10, positive 잔차/10, `(2db[f]−db[f−1]−db[f+1])/10` 곡률의 3채널을 추가하여11채널입니다. PCA는 TRAIN 순수SA A=0의 완전 관측 safe0..22 minute spectra만 사용하고, query는 저장 basis로 관측 band만 projection합니다. 누락 분을 채우지 않습니다. 추가 residual/curvature는 band23/24의 값에 영향을 받지 않았습니다. **기존8채널과 full25-band w는 그대로이므로 그 값까지 upper-band 독립이라는 뜻은 아닙니다.**\n\nTRAIN source IDs는 중복 불가이며 input db·IDs·binding SHA가 기록됩니다. 숫자만으로 실제 TRAIN/A0 provenance를 증명할 수 없어 builder의 원천검증이 필요합니다. A=0은 추가성분 없음이며 자연 선박 부재 GT가 아닙니다. residual도 ship energy나 참 배경의 추정 정답이 아닙니다.\n\n0.1dB-grid common gain 불변성, 임의 pre-quant gain의 수치차이, per-band static offset에 대한 비불변성, PCA repeat fit·basis reload·chunk 일관성·NaN mask·양성잔차·곡률 이웃조건·알려진 raw target의 rounded pooling bound를 확인했습니다. TRAIN 외 fit, 합성 source-kind, 중복 ID, state/file 변조, binding 불일치, 덮어쓰기를 거부합니다. 상세 수치는 [검증 JSON](R12_FEATURES_FIXTURE.json)에 있습니다. fixture NPZ는 명시적으로 DO_NOT_USE_FOR_TRAINING입니다.\n\nAPI: `fit_spectral_basis(db_background, source_ids, split='train', training_binding=...)`; `transform_channels(db,basis=None,include_shape=False)` → x8/11,w; `compact_relative_features(...)` →466/673열; `compact_feature_names(include_shape)`; `save_spectral_basis` / `load_spectral_basis(expected_sha256=...,expected_training_binding=...)`.\n'''
    (ROOT/'reports/R12_FEATURES_FIXTURE.md').write_text(md)
    print(json.dumps({'status':'passed','module_sha256':report['module_sha256'],'report':str(ROOT/'reports/R12_FEATURES_FIXTURE.json'),'report_sha256':sha(ROOT/'reports/R12_FEATURES_FIXTURE.json'),'checks':report['numeric_checks']},ensure_ascii=False))

if __name__=='__main__':main()
