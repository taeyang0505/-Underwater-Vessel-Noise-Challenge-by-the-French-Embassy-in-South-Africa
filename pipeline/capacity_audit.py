"""Audit effective source diversity and exact frozen synthetic targets.

Counts do not turn augmented examples into independent observations. This
audit reads training/development artefacts only, never challenge Test inputs
or the sealed NOAA evaluation source.
"""
from pathlib import Path
import json,collections
import numpy as np
import pandas as pd
from pipeline.mixtures import MixtureSampler
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=ROOT/'data/processed';data=ROOT/'data/synthetic/fullsources_cpu_a6d7f3a5_seed42'
    bmeta=pd.read_parquet(p/'sa_window_manifest.parquet');emeta=pd.read_parquet(p/'external_clip_manifest.parquet')
    output={'challenge_test_accessed':False,'sealed_NOAA_accessed':False,'partitions':{},
        'interpretation':['Do not use a context model to predict the same labelled rows already in its context as a train-error measure.',
            'For frozen foundation models, use fresh train-source mixtures or leave-context-out evaluation to measure fitting on the training domain.',
            'Random per-sample calibration crossfit does not make a new source-independent calibration sample.',
            'The effective sample counts below are descriptive; unknown physical hull recurrence remains possible.']}
    for split in ['train','validation','calibration','domain_holdout']:
        z=dict(np.load(data/f'{split}.npz'));rec=json.loads((data/f'{split}_recipes.json').read_text())
        batches=rec['batches'];bid=np.concatenate([np.array(q['background_index']) for q in batches])
        refs=sum([q['reference_indices'] for q in batches],[])
        modes=sum([q['reference_modes'] for q in batches],[])
        desired=np.concatenate([q['drawn_share'] for q in batches])
        alpha=np.concatenate([q['addition_scale'] for q in batches])
        assert len(bid)==len(refs)==len(modes)==len(desired)==len(z['share'])
        assert (bmeta.iloc[bid].split==split).all()
        flat_refs=np.concatenate([np.asarray(q,dtype=int) for q in refs])
        assert (emeta.iloc[flat_refs].split==split).all()
        assert np.array_equal(z['presence'],(z['share']>0).astype('float32'))
        # Recover normalized observed energy solely from the saved input, and
        # compare the independently stored cell target with the window target.
        x=np.asarray(z['db'],dtype='float64');obs=np.isfinite(x)
        peak=np.nanmax(x,axis=(1,2),keepdims=True)
        energy=np.where(obs,10**((x-peak)/10),0)
        recovered=(energy*z['cell_share']).sum((1,2))/energy.sum((1,2))
        error=float(np.max(np.abs(recovered-z['share'])))
        assert error<3e-6
        nonzero=z['share']>0
        assert np.max(np.abs(z['share'][nonzero]-desired[nonzero]))<1e-6
        lost=(desired>0)&~nonzero
        coverage=obs.all(2).sum(1)
        bg=bmeta.iloc[bid].copy();bg['day']=pd.to_datetime(bg.start_export_clock).dt.strftime('%Y-%m-%d')
        eref=emeta.iloc[flat_refs]
        sources=eref.source.value_counts().to_dict()
        mode_counts=collections.Counter(mode for group in modes for mode in group)
        ambig=eref.weak_class.astype(str).str.lower().isin(['background','unknown','nan'])
        zero_by_coverage=[]
        for low,high in [(3,5),(6,11),(12,15)]:
            selected=(coverage>=low)&(coverage<=high)
            if selected.any():zero_by_coverage.append({'observed_minutes':f'{low}-{high}','n':int(selected.sum()),
                'zero_target_fraction':float((~nonzero[selected]).mean()),
                'positive_draw_but_unobserved_reference_fraction':float(lost[selected].mean())})
        result={'examples':len(bid),'unique_background_windows':int(np.unique(bid).size),
            'background_sites':int(bg.site.nunique()),'background_sensor_groups':int(bg.sensor_group.nunique()),
            'background_site_recorder_days':int(bg.groupby(['site','recorder','day']).ngroups),
            'unique_external_clips':int(np.unique(flat_refs).size),'external_source_group_count':int(eref.group_key.nunique()),
            'external_sources_by_reference_usage':sources,'external_recording_days_by_source':eref.groupby('source').recording_day.nunique().to_dict(),
            'reference_modes':dict(mode_counts),'ambiguous_AIS_class_reference_usages':int(ambig.sum()),
            'zero_target_examples':int((~nonzero).sum()),'positive_draw_becoming_zero_target':int(lost.sum()),
            'zero_by_coverage':zero_by_coverage,'max_saved_cell_energy_target_error':error,
            'all_recipe_sources_in_partition':True}
        # Exact replay checks the immutable bank, RNG recipe, target and NaN mask.
        replay=MixtureSampler(split,seed=rec['seed']).draw(rec['batch_size'])
        assert np.allclose(replay[0],z['db'][:rec['batch_size']],equal_nan=True,atol=0,rtol=0)
        assert np.array_equal(replay[1],z['share'][:rec['batch_size']])
        result['first_frozen_batch_exactly_replayed']=True
        output['partitions'][split]=result
    curves={}
    for ctx in [1024,16000]:
        path=ROOT/f'models/tabicl_cpu_context{ctx}'
        if not (path/'manifest.json').exists():continue
        metrics={}
        for split in ['validation','calibration']:
            pred=np.load(path/f'{split}_predictions.npz');n=min(256,len(pred['share_true']))
            metrics[split]={'paired_prefix_n':n,'rmse':float(np.sqrt(np.mean((pred['tabicl_share'][:n]-pred['share_true'][:n])**2)))}
        curves[str(ctx)]=metrics
    output['paired_prefix_foundation_context_learning_curve']=curves
    (ROOT/'reports/data_quality_capacity_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
