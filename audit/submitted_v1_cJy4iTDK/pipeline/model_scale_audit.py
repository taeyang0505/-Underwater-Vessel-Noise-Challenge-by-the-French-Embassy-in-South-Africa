"""Storage/coverage forecast and effective-source audit. Does not read Test."""
from pathlib import Path
import json,math,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    base='fullsources_cpu_a6d7f3a5_seed42';data=ROOT/'data/synthetic'/base
    wm=pd.read_parquet(ROOT/'data/processed/sa_window_manifest.parquet')
    em=pd.read_parquet(ROOT/'data/processed/external_clip_manifest.parquet')
    rows=[]
    for split in ['train','validation','calibration','domain_holdout']:
        d=dict(np.load(data/f'{split}.npz'));recipes=json.loads((data/f'{split}_recipes.json').read_text())
        bg=[];refs=[];modes={}
        for b in recipes['batches']:
            bg.extend(b['background_index'])
            for i,rr in enumerate(b['reference_indices']):
                if b['addition_scale'][i]>0:
                    refs.extend(rr)
                    for mode in b['reference_modes'][i]:modes[mode]=modes.get(mode,0)+1
        bgm=wm.iloc[np.unique(bg)];refm=em.iloc[np.unique(refs)]
        days=pd.to_datetime(bgm.start_export_clock).dt.strftime('%Y-%m-%d')
        rows.append({'split':split,'synthetic_examples':len(bg),'array_payload_bytes':sum(a.nbytes for a in d.values()),
            'compressed_npz_bytes':(data/f'{split}.npz').stat().st_size,
            'unique_background_windows_sampled':len(bgm),'background_sites_sampled':bgm.site.nunique(),
            'background_sensor_groups_sampled':bgm.sensor_group.nunique(),
            'background_site_days_sampled':len(set(zip(bgm.site,days))),
            'unique_added_reference_clips_sampled':len(refm),'reference_source_groups_sampled':refm.group_key.nunique(),
            'reference_source_group_counts':refm.groupby('source').group_key.nunique().to_dict(),
            'reference_modes':modes})
    ratio=sum(x['compressed_npz_bytes'] for x in rows)/sum(x['array_payload_bytes'] for x in rows)
    pool=int((wm.split=='train').sum())
    forecasts=[]
    for n in [25000,100000,200000,500000,1000000,5000000]:
        forecasts.append({'synthetic_examples':n,'input_db_float32_bytes':n*15*25*4,
            'cell_target_float32_bytes':n*15*25*4,'all_four_arrays_bytes':n*(2*15*25*4+8),
            'compressed_bytes_if_current_ratio_persists':round(n*(2*15*25*4+8)*ratio),
            'tabm_1254_feature_CPU_bytes':n*1254*4,
            'cnn_channels_plus_weights_CPU_bytes':n*(4*15*25+15*25)*4,
            'foundation_96_feature_CPU_bytes':n*96*4,
            'expected_unique_backgrounds_under_uniform_sampling':pool*(-math.expm1(n*math.log1p(-1/pool)))})
    plan={'status':'data/model planning; GPU not benchmarked; no challenge Test or sealed NOAA used',
        'current_synthetic_source_audit':rows,'compression_ratio_measured':ratio,
        'forecasts':forecasts,'forecast_warning':'Compressed bytes depend on source entropy and masks. Repeated mixtures are not independent recordings. Derived feature arrays are usually CPU resident, not all GPU resident.',
        'native_recording_extension_examples':[{'added_hours':h,'maximum_nonoverlapping_15min_windows':int(h*4),
            'uncompressed_48k_mono_PCM16_bytes':int(h*3600*48000*2),
            '2s_25band_float32_bank_bytes':int(h*3600/2*25*4)} for h in [10,50,100]],
        'GPU_recommendation':{'initial':'1 x RTX 4090 / 24GB, estimator_batch_size=1, frozen foundation weights',
            'CPU_RAM':'64GB preferred; 32GB minimum for sequential preparation and model runs',
            'CPU':'8+ cores','free_SSD':'100GB preferred, 50GB for initial bounded pilot',
            'CUDA_VRAM_measured':False,
            'first_server_action':'Benchmark 4k then 8k/16k training context with own validation queries; record torch.cuda.max_memory_allocated/reserved and synchronized runtime before larger jobs',
            '48GB_condition':'Context/caching/batching expansion exceeds the measured 24GB budget, or gradient fine-tuning with a larger context is justified by independent validation',
            '80GB_condition':'Large-context/full-fine-tuning experiments require it after profiling; not justified by the current dataset file size alone'},
        'GPU_training_prep_ETA':{'current_v1_candidate':'Model/data artifacts already exist; no new GPU training required for one frozen local candidate',
            'v2_expansion_and_quality_freeze':'Planning range 1-3 hours after bounded source acquisition begins; depends on external source availability, quality exclusions and integration',
            'confidence':'Estimate, not a measured completion deadline; external-data and quality agents own remaining critical-path evidence'},
        'primary_sources':[
            {'title':'RTX4090 official hardware specification','url':'https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/'},
            {'title':'TabICLv2 official code','url':'https://github.com/soda-inria/tabicl'},
            {'title':'Kumo Tabular official release','url':'https://huggingface.co/blog/nvidia/kumo-tabular'},
            {'title':'Kumo Tabular official weights/license','url':'https://huggingface.co/nvidia/Kumo-Tabular'}]}
    (out/'model_scale_plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps({'splits':rows,'compressed_ratio':ratio,'forecast_200k':forecasts[2]},indent=2))
if __name__=='__main__':main()
