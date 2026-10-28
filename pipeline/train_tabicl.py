"""Pinned TabICLv2 comparison on training context only, no challenge Test access."""
from pathlib import Path
import argparse,json,time,hashlib,resource,os
import numpy as np
import torch
from tabicl import TabICLRegressor
from pipeline.mixtures import foundation_features as features
ROOT=Path(__file__).resolve().parents[1]

def main(a):
    torch.set_num_threads(a.threads);torch.manual_seed(42)
    a.output.mkdir(parents=True,exist_ok=True)
    context=dict(np.load(a.data/'train.npz'));rng=np.random.default_rng(42)
    ids=rng.choice(len(context['share']),min(a.context,len(context['share'])),replace=False)
    X=features(context['db'][ids]);y=context['share'][ids]
    model=TabICLRegressor(n_estimators=a.members,batch_size=1,kv_cache='repr',
        norm_methods=['none','power'],model_path=ROOT/'models/pretrained/tabicl-regressor-v2-20260212.ckpt',
        allow_auto_download=False,device=a.device,use_amp='auto',use_fa3=False,
        offload_mode='auto',n_jobs=a.threads,random_state=42)
    start=time.perf_counter();model.fit(X,y);fit_seconds=time.perf_counter()-start
    params=sum(p.numel() for p in model.model_.parameters())
    print('Loaded pretrained regressor',params,'parameters; context',len(y),'features',X.shape[1],flush=True)
    info={'context_rows':len(y),'features':X.shape[1],'parameters':params,'members':a.members,
        'device':a.device,'fit_seconds':fit_seconds,'results':{},'checkpoint':json.loads((ROOT/'models/pretrained/tabicl_provenance.json').read_text()),
        'test_accessed':False,'data':str(a.data),'context_indices':ids.tolist(),
        'comparison_limitations':'Context subset vs full-data CPU baselines; no claim of competition SOTA. No re-use of sealed NOAA assessment.'}
    for split in ['validation','calibration']+(['domain_holdout'] if a.evaluate_domain else []):
        d=dict(np.load(a.data/f'{split}.npz'));x=features(d['db'][:a.n_eval]);truth=d['share'][:a.n_eval]
        start=time.perf_counter();p=np.clip(model.predict(x),0,1);seconds=time.perf_counter()-start
        # The official model's training-only column attention must preserve query independence.
        if split=='validation':
            singles=np.concatenate([model.predict(x[i:i+1]) for i in range(min(3,len(x)))])
            if not np.allclose(p[:len(singles)],np.clip(singles,0,1),atol=2e-5):raise ValueError('Query batch dependence requires further audit; do not use challenge Test')
            info['per_query_independence_checked']=True
        info['results'][split]={'n':len(truth),'rmse':float(np.sqrt(np.mean((p-truth)**2))),
            'seconds':seconds,'zero_addition_mean':float(p[truth==0].mean())}
        np.savez_compressed(a.output/f'{split}_predictions.npz',share_true=truth,tabicl_share=p)
        print(split,info['results'][split],flush=True)
    info['peak_CPU_resident_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30
    model.save(a.output/'fitted.pkl',save_model_weights=False,save_training_data=True,save_kv_cache=False)
    (a.output/'manifest.json').write_text(json.dumps(info,indent=2));print(json.dumps({k:v for k,v in info.items() if k!='context_indices'},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/synthetic/fullsources_cpu_a6d7f3a5_seed42')
    p.add_argument('--output',type=Path,default=ROOT/'models/tabicl_cpu_context1024');p.add_argument('--context',type=int,default=1024)
    p.add_argument('--n-eval',type=int,default=256);p.add_argument('--members',type=int,default=2)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--evaluate-domain',action='store_true');main(p.parse_args())
