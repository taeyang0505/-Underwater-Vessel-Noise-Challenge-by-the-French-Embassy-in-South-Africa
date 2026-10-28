"""Independent saved-component oracle for the TRAIN-only R15 sensitivity data.

Does not import the generator, a neural/tree model, a feature/PCA factory,
or inference code. Reads admitted TRAIN background slices, admitted native
reference slices, source metadata and the completed prototype only.
"""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/r15_validation_checks_r1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as z:
        result={k:z[k] for k in z.files}
    assert not any(x.dtype.kind=='O' for x in result.values())
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prototype',default='reports/r15_forward_prototype_r1')
    args=parser.parse_args();folder=ROOT/args.prototype
    OUT.mkdir(exist_ok=True)
    target=OUT/'PROTOTYPE_INDEPENDENT_QA.json'
    if target.exists():raise FileExistsError('Preserve previous QA')
    manifest_path=folder/'manifest.json';manifest=json.loads(manifest_path.read_text())
    assert manifest['status']=='completed_TRAIN_physics_sensitivity_only'
    assert not manifest['competition_training_ready'] and not manifest['candidate_selection_or_Test_distance_computed']
    assert not manifest['Test_opened'] and not manifest['model_fit_or_inference_performed']
    plan_path=folder/'plan.json';plan=json.loads(plan_path.read_text())
    assert sha(plan_path)==manifest['plan_sha256']
    assert sha(ROOT/'ops/r15_forward_prototype.py')==plan['code_sha256']==manifest['code_sha256']
    assert plan['rows_per_candidate']==4000 and plan['total_candidate_rows']==16000
    assert plan['no_source_count']==800 and plan['observed_positive_count']==3200
    assert plan['PCA_anchor_feature_transform'] is False and plan['highshare_forcing'] is False
    assert not plan['Test_read_comparison_distance_optimization_and_candidate_selection']
    pins={str((folder/'manifest.json').relative_to(ROOT)):sha(manifest_path)}
    for name,h in manifest['artifact_sha256'].items():
        p=folder/name;assert sha(p)==h;pins[str(p.relative_to(ROOT))]=h
    for name,h in manifest['source_pins'].items():assert sha(ROOT/name)==h

    config=json.loads((ROOT/plan['source_config']).read_text())
    bgdir=ROOT/config['background_version'];refdir=ROOT/config['reference_version']
    bgmeta=json.loads((bgdir/'manifest.json').read_text());refmeta=json.loads((refdir/'manifest.json').read_text())
    bgpath=ROOT/bgmeta['background_bank']['path'];refpath=ROOT/refmeta['bank']['path']
    bgsha=sha(bgpath);refsha=sha(refpath)
    assert bgsha==bgmeta['background_bank']['sha256']==manifest['source_identity']['background_bank_sha256']
    assert refsha==refmeta['bank']['sha256']==manifest['source_identity']['reference_bank_sha256']
    chosen=pd.read_parquet(folder/'background_metadata.parquet');train=pd.read_parquet(bgdir/'fold_00/train_manifest.parquet')
    assert len(chosen)==chosen.base_window_index.nunique()==4000
    assert set(chosen.base_window_index)<=set(train.base_window_index)
    held=pd.concat([pd.read_parquet(bgdir/'fold_00'/(r+'_manifest.parquet')) for r in ['selection','ensemble_fit','ensemble_gate','report']],ignore_index=True)
    assert not(set(chosen.base_window_index)&set(held.base_window_index))
    assert not(set(chosen.background_group)&set(held.background_group))
    gap=[]
    for sensor,part in chosen.groupby('sensor_group'):
        other=held[held.sensor_group==sensor]
        if not len(other):continue
        a=pd.to_datetime(part.start_export_clock).astype('int64').to_numpy()/1e9
        b=np.sort(pd.to_datetime(other.start_export_clock).astype('int64').to_numpy()/1e9)
        pos=np.searchsorted(b,a)
        delta=np.minimum(np.abs(a-b[np.clip(pos-1,0,len(b)-1)]),np.abs(a-b[np.clip(pos,0,len(b)-1)]))
        gap.extend(np.maximum(delta-900,0)/3600)
    assert min(gap)>=24-1e-8

    common=load(folder/'common_backgrounds.npz');recipes=json.loads((folder/'recipes.json').read_text())
    assert len(recipes)==4000
    indices=chosen.base_window_index.to_numpy(np.int64)
    assert np.array_equal(indices,common['base_window_index'])
    assert np.array_equal(chosen.site.astype(str).to_numpy(),common['site'])
    assert np.array_equal(chosen.background_group.astype(str).to_numpy(),common['background_group'])
    bank=np.load(bgpath,mmap_mode='r',allow_pickle=False)
    original=np.asarray(bank[indices]).astype(np.float64)
    bits=chosen.boundary_mask_bits.to_numpy(np.int64)
    original[((bits[:,None]>>np.arange(15))&1).astype(bool)]=np.nan
    assert np.array_equal(original,common['background_db'],equal_nan=True)
    observed=np.isfinite(original)
    assert np.array_equal(observed,common['observed_mask'])
    assert not (observed.any(2)!=observed.all(2)).any() and observed.any(2).sum(1).min()>=3
    offset=np.nanmax(original,axis=(1,2))
    background=np.where(observed,10.**((np.where(observed,original,offset[:,None,None])-offset[:,None,None])/10.),0.)
    assert np.array_equal(background,common['background_observed_power'])
    assert np.array_equal(offset,common['power_db_offset'])
    zero=common['explicit_no_source'].astype(bool)
    assert int(zero.sum())==800 and (common['reference_index'][zero]==-1).all()
    assert len(np.unique(common['pair_id']))==4000

    refs=pd.read_parquet(refdir/'reference_manifest.parquet').sort_values('reference_row_index').reset_index(drop=True)
    assert len(refs)==509 and set(refs.archive_role)=={'vessel_passage'}
    refbank=np.load(refpath,mmap_mode='r',allow_pickle=False);native=[]
    for row in refs.itertuples(index=False):
        a=np.asarray(refbank[int(row.segment_start):int(row.segment_start)+int(row.segment_count)])
        assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==row.expected_segment_slice_sha256
        n=len(a)//30;assert n==int(row.native_complete_minute_count) and n>=1
        native.append(a[:n*30].astype(np.float64).reshape(n,30,25).mean(1))
    refcounts=np.bincount(common['reference_index'][~zero],minlength=509)
    assert (refcounts>0).all() and refcounts.min()==6 and refcounts.max()==7

    # Reconstruct the prescribed smooth shelf independently, not by importing
    # the generator. Treat the channels as fixed sensitivity hypotheses only.
    frequencies=np.round(10.**(1.5+.1*np.arange(25)),1)
    z=np.clip((np.log(frequencies)-np.log(200.))/(np.log(500.)-np.log(200.)),0,1)
    shelf=.5-.5*np.cos(np.pi*z)
    prescribed={'control_eq6':(0.,0.),'mfhf_shelf6':(6.,0.),'mfhf_shelf12':(12.,0.),'mfhf_shelf12_lfminus6':(12.,-6.)}
    assert set(plan['channel_hypotheses'])==set(prescribed)
    candidate_results={};zero_db=None;max_native_replay=0.;max_alpha_error=0.
    expected_keys={'db','unrounded_db','cell_share','share','presence','added_full_power','added_observed_power','mixture_observed_power','pair_id','sample_id'}
    for name,(boost,low) in prescribed.items():
        data=load(folder/(name+'.npz'));assert set(data)==expected_keys
        assert data['db'].shape==(4000,15,25) and data['share'].shape==(4000,)
        assert np.array_equal(data['pair_id'],common['pair_id'])
        assert np.array_equal(data['sample_id'],np.asarray([s+'__'+name for s in common['pair_id']]))
        assert np.array_equal(np.isfinite(data['db']),observed) and np.array_equal(np.isfinite(data['unrounded_db']),observed)
        assert data['cell_share'].dtype==np.float64 and data['share'].dtype==np.float64
        assert np.isfinite(data['cell_share']).all() and ((data['cell_share']>=0)&(data['cell_share']<=1)).all()
        assert np.array_equal(data['presence'].astype(bool),~zero)
        added=data['added_observed_power'];energy=data['mixture_observed_power']
        assert np.array_equal(added,np.where(observed,data['added_full_power'],0.))
        assert np.array_equal(energy,background+added)
        q=np.divide(added,energy,out=np.zeros_like(energy),where=observed)
        y=added.sum((1,2))/energy.sum((1,2))
        qerr=float(np.max(np.abs(q-data['cell_share'])));yerr=float(np.max(np.abs(y-data['share'])))
        assert qerr<=1e-12 and yerr<=1e-12
        assert (data['share'][zero]==0).all() and (data['share'][~zero]>0).all()
        assert not data['added_full_power'][zero].any() and not data['cell_share'][zero].any()
        assert np.array_equal(np.round(data['unrounded_db'],1).astype(np.float32),data['db'],equal_nan=True)
        if zero_db is None:zero_db=data['db'][zero].copy()
        else:assert np.array_equal(zero_db,data['db'][zero],equal_nan=True)
        for i,recipe in enumerate(recipes):
            assert recipe['pair_id']==str(common['pair_id'][i]) and recipe['base_window_index']==int(indices[i])
            assert recipe['boundary_mask_bits']==int(bits[i]) and recipe['explicit_no_source']==bool(zero[i])
            if zero[i]:continue
            rid=int(common['reference_index'][i]);r=native[rid];row=refs.iloc[rid]
            assert recipe['reference_index']==rid and recipe['reference_clip_id']==str(row.clip_id) and recipe['reference_SHA']==str(row.sha256)
            trajectory=recipe['trajectory'];start=trajectory['window_start_minute'];length=trajectory['selected_minutes'];begin=trajectory['crop_begin_minute']
            assert 1<=length<=min(6,len(r)) and 0<=begin<=len(r)-length and 0<=start<=15-length
            peak=int(np.argmax(r.sum(1)));assert begin<=peak<begin+length and peak==trajectory['measured_full_source_peak_index']
            assert trajectory['time_warp'] is False and trajectory['source_repetition'] is False and trajectory['added_bell_or_fade'] is False and trajectory['fixed_full15_component'] is False
            common_eq=np.asarray(recipe['common_random_eq_db']);assert np.max(np.abs(common_eq))<=6+1e-12
            curve=common_eq+low*(1-shelf)+boost*shelf
            stored=np.asarray(recipe['candidates'][name]['total_reference_channel_db']);assert np.max(np.abs(curve-stored))<1e-12
            transformed=r*10.**(curve[None,:]/10.)
            bmedian=np.median(background[i].sum(1)[observed[i].all(1)])
            snr=recipe['same_sampled_SNR_db'];assert -10<=snr<=10
            alpha=10.**(snr/10.)*bmedian/np.max(transformed.sum(1))
            savedalpha=recipe['candidates'][name]['alpha'];ae=abs(alpha-savedalpha)/max(abs(alpha),np.finfo(float).tiny)
            max_alpha_error=max(max_alpha_error,ae);assert ae<1e-12
            expected=np.zeros((15,25));expected[start:start+length]=alpha*transformed[begin:begin+length]
            replay=float(np.max(np.abs(expected-data['added_full_power'][i])));max_native_replay=max(max_native_replay,replay)
            assert replay<=1e-12
            legal=sum(np.where(observed[i][s:s+length],r[begin:begin+length],0.).sum()>0 for s in range(16-length))
            assert legal==trajectory['legal_observed_placements']
        max_raw=max_round=0.;max_bound_violation=0.;max_Edb=0.
        for start in range(0,4000,128):
            stop=start+128;o=observed[start:stop];raw=data['unrounded_db'][start:stop];rounded=data['db'][start:stop].astype(float)
            e=energy[start:stop];cell=q[start:stop]
            eraw=np.where(o,10.**((np.where(o,raw,offset[start:stop,None,None])-offset[start:stop,None,None])/10.),0.)
            max_Edb=max(max_Edb,float(np.max(np.abs(eraw-e))))
            wr=eraw/eraw.sum((1,2),keepdims=True)
            eq=np.where(o,10.**((np.where(o,rounded,offset[start:stop,None,None])-offset[start:stop,None,None])/10.),0.)
            wq=eq/eq.sum((1,2),keepdims=True)
            rawgap=np.abs((wr*cell).sum((1,2))-y[start:stop]);roundgap=np.abs((wq*cell).sum((1,2))-y[start:stop])
            delta=np.max(np.where(o,np.abs(np.where(o,rounded,0)-np.where(o,raw,0)),0),(1,2))
            bound=np.tanh(np.log(10)*delta/20)
            max_raw=max(max_raw,float(rawgap.max()));max_round=max(max_round,float(roundgap.max()))
            max_bound_violation=max(max_bound_violation,float(np.max(roundgap-bound)))
        assert max_raw<1e-10 and max_Edb<1e-12 and max_bound_violation<1e-10
        candidate_results[name]={'rows':4000,'zero_count':800,'observed_positive_count':3200,'q_max_error':qerr,'y_max_error':yerr,
            'E_BplusA_bitwise_exact':True,'raw_db_E_reconstruction_max_error':max_Edb,'raw_pool_max_error':max_raw,
            'rounded_pool_max_gap':max_round,'maximum_actual_quantization_bound_violation':max_bound_violation,
            'all_observed_masks_preserved':True,'raw_labels_not_replaced_with_rounded_pool':True}
    assert pins=={n:sha(ROOT/n) for n in pins}
    for n,h in manifest['source_pins'].items():assert sha(ROOT/n)==h
    report={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Independent TRAIN-only saved-component and admitted source-slice replay; no model inference/training, PCA or Test input access',
        'prototype_manifest_sha256':sha(manifest_path),'prototype_plan_sha256':sha(plan_path),
        'physical_oracle':'Independent NumPy equations/native30x2s minute mean, no generator import',
        'unique_TRAIN_backgrounds':4000,'candidate_count':4,'candidate_rows':16000,'all509_measured_reference_slices_SHA_verified':True,
        'source_bank_global_integrity_SHA256':{'background':bgsha,'reference':refsha},
        'TRAIN_heldout_base_index_intersection':0,'TRAIN_heldout_group_intersection':0,
        'minimum_same_sensor_900s_support_gap_hours':float(min(gap)),'absolute_UTC_not_asserted':True,
        'PCA_anchor_model_feature_columns_not_present':True,'metadata_identity_not_a_model_input':True,
        'positive_support_and_zero_schedule_preserved':True,'zero_inputs_paired_exactly':True,
        'native_A_replay_max_absolute_error':max_native_replay,'alpha_replay_max_relative_error':max_alpha_error,
        'candidate_checks':candidate_results,'source_and_prototype_SHA_unchanged':True,'input_artifact_sha256':pins,
        'limitations':['Diagnostic four fixed TRAIN sensitivities only; not competition-ready or evidence of performance.',
            'Authors previously saw Test descriptive results, so predeclared means before this generation rather than blind.',
            'No comparison to Test or best-candidate acceptance/selection performed.',
            'All509 references are shared; this is not reference-independent validation.',
            'Placement is conditioned on known TRAIN observation support and differs from original R9 mask-independent geometry.',
            'Added whole-recording energy is known; naturally isolated vessel-only energy is not.']}
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','report':str(target.relative_to(ROOT)),'sha256':sha(target),'max_native_replay':max_native_replay}))


if __name__=='__main__':main()
