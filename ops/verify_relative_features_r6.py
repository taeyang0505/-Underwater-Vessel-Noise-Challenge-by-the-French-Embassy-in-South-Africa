"""Numeric-only R6 fixture. Never opens competition data or runs a formal fit."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from pipeline.relative_features_r6 import (
    FEATURE_NAMES, TinyMaskCNN, quantization_pooling_bound, quantized_db, relative_grid,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights(db):
    mask=np.isfinite(db);peak=np.nanmax(db,axis=(1,2),keepdims=True)
    e=np.where(mask,10.**((db-peak)/10.),0.)
    return e/e.sum(axis=(1,2),keepdims=True)


def run():
    started=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(42)
    # exec starts a fresh interpreter: it must not inherit this fixture's Torch imports.
    pure_probe=subprocess.run([sys.executable,'-c',
        "import sys,json,numpy as np; from pipeline.relative_grid_r6 import relative_grid; "
        "x,w=relative_grid(np.full((1,15,25),100.)); "
        "assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules); "
        "print(json.dumps({'torch_loaded':False,'shape':list(x.shape),'weight_sum':float(w.sum())}))"],
        check=True,capture_output=True,text=True,timeout=20)
    pure_probe_result=json.loads(pure_probe.stdout)
    from pipeline.relative_grid_r6 import relative_grid as pure_relative_grid
    assert relative_grid is pure_relative_grid
    rng=np.random.default_rng(42)
    db=rng.normal(100.,7.,size=(7,15,25)).astype('float64')
    db[0,0]=np.nan;db[1,4:7]=np.nan;db[2,:,3]=np.nan;db[3,3,8]=np.nan
    before=db.copy();x,w=relative_grid(db);mask=np.isfinite(db)
    assert x.shape==(7,6,15,25) and w.shape==db.shape
    assert x.dtype==w.dtype==np.float32
    assert np.array_equal(db,before,equal_nan=True)
    assert np.isfinite(x).all() and np.isfinite(w).all()
    assert np.array_equal(x[:,5],mask.astype('float32'))
    assert (x[:,:5]*~mask[:,None]).sum()==0 and (w[~mask]==0).all()
    assert np.allclose(w.sum((1,2)),1,atol=2e-7)
    assert np.max(np.abs(w.astype('float64')-weights(quantized_db(db))))<1e-8
    pairs=mask[:,1:]&mask[:,:-1]
    assert (x[:,2,0]==0).all() and (x[:,2,1:][~pairs]==0).all()
    # Independent hand-computed median/q20, adjacent delta, and co-rising bands.
    ramp=np.broadcast_to(100.+np.arange(15)[None,:,None],(1,15,25)).copy()
    ramp[0,1,5:9]=99.;xr,wr=relative_grid(ramp)
    assert abs(float(xr[0,0,0,0])+0.7)<1e-7
    assert abs(float(xr[0,1,0,0])+2.8/15)<1e-7
    assert abs(float(xr[0,2,1,0])-.1)<1e-7
    assert abs(float(xr[0,3,1,10])-.1*math.tanh((5/9)/3))<1e-7
    assert (xr[:,3,:,:10]==0).all()
    gain_x,gain_w=relative_grid(db+12.)
    common_gain={'feature_max_abs':float(np.max(abs(gain_x-x))),
                 'weight_max_abs':float(np.max(abs(gain_w-w)))}
    assert np.allclose(gain_x,x,atol=2e-7) and np.allclose(gain_w,w,atol=2e-7)
    band_offset=np.linspace(-6.,6.,25)[None,None,:]
    offset_x,offset_w=relative_grid(db+band_offset)
    band_offset_audit={'channels_0_to_3_max_abs':float(np.max(abs(offset_x[:,:4]-x[:,:4]))),
                       'profile_channel_max_abs':float(np.max(abs(offset_x[:,4]-x[:,4]))),
                       'weight_max_abs':float(np.max(abs(offset_w-w)))}
    assert np.allclose(offset_x[:,:4],x[:,:4],atol=2e-7)
    assert band_offset_audit['profile_channel_max_abs']>.01
    assert band_offset_audit['weight_max_abs']>1e-4
    # A non-grid gain can change rounded features: do not claim exact invariance.
    shifted_x,shifted_w=relative_grid(db+.037)
    nongrid={'feature_max_abs':float(np.max(abs(shifted_x-x))),
             'smooth_co_rise_channel_max_abs':float(np.max(abs(shifted_x[:,3]-x[:,3]))),
             'weight_max_abs':float(np.max(abs(shifted_w-w)))}
    assert nongrid['feature_max_abs']>0 and nongrid['weight_max_abs']>0
    # Compare to the initial hard-sign formula on the identical numeric input.
    def old_coherence(raw):
        z=quantized_db(raw);valid=np.isfinite(z);pair=valid[:,1:]&valid[:,:-1]
        d=np.zeros_like(z);d[:,1:]=np.where(pair,z[:,1:]-z[:,:-1],0.)
        low=pair[:,:,:9];counts=low.sum(2,keepdims=True)
        gate=np.zeros(z.shape[:2]+(1,));gate[:,1:]=np.divide(
            ((d[:,1:,:9]>0)&low).sum(2,keepdims=True),counts,
            out=np.zeros_like(counts,dtype='float64'),where=counts>0)
        c=np.maximum(d,0.)/10.*gate;c[:,:,:10]=0.;return c
    old_jump=float(np.max(abs(old_coherence(db+.037)-old_coherence(db))))
    assert nongrid['smooth_co_rise_channel_max_abs']<old_jump
    # Continuity before quantization: tiny low-band rises cannot flip a large gate.
    tiny=np.full((1,15,25),100.);tiny[:,1:,10:]=130.
    tiny[:,1:,:9]+=1e-8
    low_mean=1e-8;continuous_gate=math.tanh(low_mean/3.)
    assert continuous_gate<4e-9 and (30./10.*continuous_gate)<=1.01e-8
    tiny_x,_=relative_grid(tiny)
    assert (tiny_x[:,3]==0).all()  # sub-quantization low rise rounds to exactly zero
    no_low=ramp.copy();no_low[:,:,:9]=np.nan
    no_low_x,_=relative_grid(no_low)
    assert (no_low_x[:,3]==0).all()
    chunkx=[];chunkw=[]
    for ids in [slice(0,1),slice(1,4),slice(4,7)]:
        a,b=relative_grid(db[ids]);chunkx.append(a);chunkw.append(b)
    assert np.array_equal(np.concatenate(chunkx),x)
    assert np.array_equal(np.concatenate(chunkw),w)
    perm=np.array([5,1,6,0,3,4,2]);px,pw=relative_grid(db[perm])
    assert np.array_equal(px,x[perm]) and np.array_equal(pw,w[perm])
    # Real-valued reweighting TV bound: independent two-mass numerical oracle.
    delta=.05;h=math.log(10.)*delta/10.;bound=quantization_pooling_bound(delta)
    p=np.linspace(0.,1.,100001)
    changed=p*np.exp(h)/(p*np.exp(h)+(1-p)*np.exp(-h))
    numerical_bound=float(np.max(changed-p))
    assert abs(numerical_bound-bound)<1e-10
    # Actual rounding example nearly attains the bound without modifying q/y.
    extreme=np.full((1,15,25),np.nan)
    extreme[0,0,0]=100.049999;extreme[0,0,1]=99.950001
    q=np.zeros_like(extreme);q[0,0,0]=1.
    original_q=q.copy();raw_y=float((np.nan_to_num(q)*weights(extreme)).sum())
    _,ew=relative_grid(extreme);rounded_y=float((np.nan_to_num(q)*ew).sum())
    observed_gap=abs(raw_y-rounded_y)
    assert observed_gap<bound and observed_gap>bound*.999
    assert np.array_equal(q,original_q,equal_nan=True)
    raw_w=weights(db);known_q=rng.uniform(size=db.shape);known_q[~mask]=0.
    random_gap=np.abs((known_q*raw_w).sum((1,2))-(known_q*w).sum((1,2)))
    assert random_gap.max()<bound+1e-7
    assert quantization_pooling_bound(0)==0
    invalid=0
    for v in [{'db':db,'share':np.zeros(7)},np.zeros((15,25)),np.full((1,15,25),np.nan),np.full((1,15,25),np.inf)]:
        try: relative_grid(v)
        except (TypeError,ValueError):invalid+=1
    assert invalid==4
    # Numeric fixture gradients only: two Adam steps, no dataset/model artefacts.
    models=[]
    for channels in [6,8]:
        model=TinyMaskCNN(input_channels=channels)
        inp=torch.from_numpy(x if channels==6 else np.concatenate([x,x[:,0:2]],1))
        tw=torch.from_numpy(w);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        losses=[]
        for _ in range(2):
            model.train();optimizer.zero_grad();share,pres,cell=model(inp,tw)
            target=torch.linspace(.05,.8,len(inp))
            loss=(share-target).square().mean()+.05*torch.nn.functional.binary_cross_entropy_with_logits(pres,(target>.1).float())
            loss.backward()
            assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
            assert model.mask_head.weight.grad.abs().max()>0
            optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            result=model(inp,tw)
            assert torch.max(abs(result[0]-(result[2]*tw).sum((1,2))))==0
            assert ((result[0]>=0)&(result[0]<=1+1e-7)).all()
            assert (result[2][tw==0]==0).all()
            split=[model(inp[i:i+1],tw[i:i+1]) for i in range(len(inp))]
            batch_error=max(float(torch.max(abs(torch.cat([v[j] for v in split])-result[j]))) for j in range(3))
            assert batch_error<1e-5
        blob=io.BytesIO();torch.save(model.state_dict(),blob);blob.seek(0)
        restored=TinyMaskCNN(input_channels=channels).eval()
        restored.load_state_dict(torch.load(blob,weights_only=True,map_location='cpu'))
        with torch.no_grad():
            assert all(torch.equal(a,b) for a,b in zip(result,restored(inp,tw)))
        models.append({'input_channels':channels,'parameters':sum(v.numel() for v in model.parameters()),
                       'numeric_Adam_steps':2,'fixture_losses':losses,
                       'max_singleton_vs_batch_abs':batch_error,'reload_bitwise':True})
    return {'format':'r6_relative_features_numeric_fixture_v3','status':'passed','fixture_only':True,
            'code_sha256':{'pipeline/relative_grid_r6.py':sha('pipeline/relative_grid_r6.py'),
                           'pipeline/relative_features_r6.py':sha('pipeline/relative_features_r6.py'),
                           'ops/verify_relative_features_r6.py':sha(__file__)},
            'source_data_opened':False,'Test_OUTER_INNER_or_TRAIN_datasets_used':False,
            'GPU_SSH_external_submission_used':False,'formal_training_performed':False,
            'feature_names':list(FEATURE_NAMES),'co_rise_bands_hz':'31.6--199.5 inclusive (indices0--8)',
            'co_rise_output_bands_hz':'316.2--7943.3 inclusive (indices10--24)',
            'co_rise_gate_formula':'tanh(mean_observed_low_band(max(adjacent_delta_db,0))/3.0)',
            'pre_fit_numerical_correction':{
                'old_formula':'fraction_observed_low_band(adjacent_delta_db>0)',
                'new_formula':'tanh(mean_observed_low_band(max(adjacent_delta_db,0))/3.0)',
                'reason':'initial numeric fixture showed a hard-sign jump under common +0.037dB gain',
                'scale_db':3.0,'model_or_heldout_performance_used':False,
                'old_numeric_hard_sign_max_channel_change':old_jump,
                'new_numeric_smooth_max_channel_change':nongrid['smooth_co_rise_channel_max_abs'],
                'initial_fixture_preserved':'R6_RELATIVE_FEATURES_FIXTURE_hard_sign_prefit.json',
                'CNN_architecture_unchanged':True},
            'pre_fit_engine_correction':{
                'reason':'parent numeric CPU fixture found Torch/LightGBM OpenMP engine coexistence deadlock',
                'pure_feature_module':'pipeline.relative_grid_r6',
                'CNN_and_compatibility_module':'pipeline.relative_features_r6',
                'fresh_process_pure_module_probe':pure_probe_result,
                'backwards_public_reexport_same_function_object':True,
                'previous_smooth_fixture_preserved':'R6_RELATIVE_FEATURES_FIXTURE_smooth_prefit.json',
                'feature_algorithm_or_CNN_architecture_changed':False,
                'actual_LightGBM_engine_retest_performed_here':False,
                'dataset_or_heldout_model_performance_used':False},
            'query_api':'relative_grid(db) -> float32 x[N,6,15,25],w[N,15,25]',
            'weights_from_same_0p1dB_quantized_input':True,'original_input_unchanged':True,
            'mask_and_adjacent_pair_checks':True,'chunk_and_permutation_bitwise':True,
            'hand_computed_feature_oracle':True,'common_gain_12dB':common_gain,
            'per_band_static_offset':band_offset_audit,'nongrid_gain_0p037dB':nongrid,
            'quantization':{'max_real_valued_db_error':delta,'pooling_error_bound':bound,
                            'bound_formula':'tanh(ln(10)*0.05/20)',
                            'independent_2mass_numerical_bound':numerical_bound,
                            'actual_near_worst_rounding_raw_y':raw_y,
                            'actual_near_worst_rounding_rounded_weight_y':rounded_y,
                            'actual_near_worst_rounding_gap':observed_gap,
                            'random_known_q_max_pooling_gap':float(random_gap.max()),
                            'supervision_unchanged':True,'float32_weight_roundoff_additional':True},
            'CNN_numeric_fixtures':models,'environment':{'numpy':np.__version__,'torch':torch.__version__,
                                                        'CPU_threads':torch.get_num_threads()},
            'elapsed_seconds':time.perf_counter()-started}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='reports/R6_RELATIVE_FEATURES_FIXTURE.json')
    args=parser.parse_args();path=Path(args.output)
    if path.exists():raise FileExistsError('Fresh fixture output required: '+str(path))
    result=run();path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    md=path.with_suffix('.md')
    md.write_text('# R6 relative input numeric fixture\n\n'
        'Status: **PASS, numeric fixture only**. No competition TRAIN/INNER/OUTER/Test files, GPU or SSH were used. '
        'The two Adam updates per model check gradients and reload compatibility; these weights are not saved or used for training.\n\n'
        'The six db-only channels and their formulas are documented in [relative_grid_r6.py](../pipeline/relative_grid_r6.py). '
        'The pooling weights use the same 0.1 dB rounded observations. Known raw-precision q/y labels remain unchanged.\n\n'
        f'For |dB error| ≤0.05, fixed-q pooling changes by at most **{result["quantization"]["pooling_error_bound"]:.10f}** '
        'in real arithmetic: tanh(ln(10)×0.05/20). Float32 output rounding is additional. '
        f'The two-cell near-worst example changes {result["quantization"]["actual_near_worst_rounding_raw_y"]:.10f} '
        f'to {result["quantization"]["actual_near_worst_rounding_rounded_weight_y"]:.10f}; this is an acknowledged mismatch, not a relabeling.\n\n'
        'Common gain invariance is exact after quantization, and for pre-quantization grid-aligned gains when rounding commutes. '
        'Arbitrary gain can cross rounding boundaries. Channels 0–3 are also invariant to static band offsets under that caveat; '
        'the spectral-profile channel and energy weights are not.\n\n'
        '## Pre-fit numerical correction\n\n'
        'Before dataset training or held-out model evaluation, the initial hard-sign low-band fraction was replaced by '
        '`tanh(mean_observed_low(max(adjacent_delta_db, 0))/3.0)`. The fixed 3-dB divisor makes the gate continuous '
        'in its unquantized deltas; 0.1-dB input quantization still creates finite steps. '
        'The gate multiplies positive mid/high-band delta/10 only (indices 10–24); the low-band mean uses indices 0–8. '
        'Missing adjacent pairs do not enter the denominator, and no low-band pairs means zero gate. '
        'The CNN architecture, target labels and other five channels are unchanged.\n\n'
        f'The same +0.037-dB numeric stress changes the old coherence channel by '
        f'{result["pre_fit_numerical_correction"]["old_numeric_hard_sign_max_channel_change"]:.10f}, '
        f'and the corrected channel by {result["pre_fit_numerical_correction"]["new_numeric_smooth_max_channel_change"]:.10f}. '
        '[Initial fixture JSON](R6_RELATIVE_FEATURES_FIXTURE_hard_sign_prefit.json) / '
        '[initial fixture MD](R6_RELATIVE_FEATURES_FIXTURE_hard_sign_prefit.md) remain archived.\n\n'
        '## Pre-fit engine correction\n\n'
        'After a separate parent-owned numeric CPU fixture exposed Torch/LightGBM OpenMP coexistence deadlock, '
        'the feature functions were moved unchanged to `pipeline.relative_grid_r6`, which imports NumPy only '
        '(plus standard-library math/warnings). `pipeline.relative_features_r6` retains the CNN and re-exports '
        'the previous public feature API. Function and CNN ASTs were checked unchanged at extraction; '
        'the full numeric fixture passes again. A fresh Python process imports and computes the pure grid '
        'without loading any Torch module. This does not itself claim that a LightGBM fit was re-tested; '
        'the parent isolates tree execution in a fresh process. No data/model result informed this engine change.\n\n'
        '[Previous smooth-gate fixture JSON](R6_RELATIVE_FEATURES_FIXTURE_smooth_prefit.json) / '
        '[previous smooth-gate fixture MD](R6_RELATIVE_FEATURES_FIXTURE_smooth_prefit.md) remain archived.\n\n'
        'Fixtures cover missing bands/minutes, no differencing across missing minutes, hand-calculated co-rise, '
        'chunk/permutation replay, independent numerical bound, 6/8-channel gradients, energy pooling and bitwise checkpoint reload.\n\n'
        f'[Detailed JSON]({path.name})\n')
    print(json.dumps({'status':result['status'],'report':str(path),'report_sha256':sha(path),
                      'models':result['CNN_numeric_fixtures'],'bound':result['quantization']['pooling_error_bound']}))


if __name__=='__main__':main()
