"""NVIDIA Kumo Large regression with a fixed, training-only context."""
from pathlib import Path
import argparse,json,time,resource
import numpy as np
import torch
import sdm
from pipeline.mixtures import foundation_features
ROOT=Path(__file__).resolve().parents[1]
def main(a):
    torch.set_num_threads(a.threads);torch.manual_seed(42);a.output.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/synthetic/fullsources_cpu_a6d7f3a5_seed42'
    tr=dict(np.load(data/'train.npz'));ids=np.random.default_rng(42).choice(len(tr['share']),min(a.context,len(tr['share'])),replace=False)
    X=torch.from_numpy(foundation_features(tr['db'][ids])).to(a.device)
    y=torch.from_numpy(tr['share'][ids,None]).to(a.device)
    # Load a checksum-verified pinned local file; disable runtime remote downloads.
    m=sdm.models.KumoTabular(task='regression',size='large',pretrained=False,device='cpu')
    state=torch.load(ROOT/'models/pretrained/kumo_large_regressor.pt',map_location='cpu',weights_only=True)
    m.models[sdm.Task.regression].load_state_dict(state,assign=True);del state
    m.to(a.device);m.eval();params=sum(p.numel() for p in m.parameters())
    print('Kumo Large loaded',params,'parameters',flush=True)
    start=time.perf_counter()
    with torch.no_grad(),torch.autocast(a.device,dtype=torch.float16,enabled=a.device.startswith('cuda')):
        m.fit(X,y,num_estimators=a.members,estimator_batch_size=1,generator=torch.Generator(device=a.device).manual_seed(42))
    info={'model':'NVIDIA Kumo Tabular Large','parameters':params,'context_rows':len(ids),'features':X.shape[1],
          'members':a.members,'device':a.device,'context_fit_seconds':time.perf_counter()-start,
          'test_accessed':False,'results':{},'checkpoint':json.loads((ROOT/'models/pretrained/kumo_provenance.json').read_text()),
          'status':'CPU/GPU candidate comparison, not a claim of competition SOTA; sealed NOAA assessment is not reused'}
    for split in ['validation','calibration']+(['domain_holdout'] if a.evaluate_domain else []):
        d=dict(np.load(data/f'{split}.npz'));q=torch.from_numpy(foundation_features(d['db'][:a.n_eval])).to(a.device)
        start=time.perf_counter()
        with torch.no_grad(),torch.autocast(a.device,dtype=torch.float16,enabled=a.device.startswith('cuda')):
            out=m.predict(q).numerical.mean(-1).cpu().numpy()
            if split=='validation':
                one=np.concatenate([m.predict(q[i:i+1]).numerical.mean(-1).cpu().numpy() for i in range(min(2,len(q)))])
                assert np.allclose(out[:len(one)],one,atol=3e-5),'Query dependence: do not use challenge Test'
        pred=np.clip(out,0,1);truth=d['share'][:a.n_eval]
        info['results'][split]={'n':len(truth),'rmse':float(np.sqrt(np.mean((truth-pred)**2))),'seconds':time.perf_counter()-start}
        np.savez_compressed(a.output/f'{split}_predictions.npz',share_true=truth,kumo_share=pred)
        print(split,info['results'][split],flush=True)
    info['peak_CPU_resident_GiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30
    info['query_independence_checked']=True
    (a.output/'manifest.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=int,default=1024);p.add_argument('--n-eval',type=int,default=256)
    p.add_argument('--members',type=int,default=1);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',type=Path,default=ROOT/'models/kumo_large_cpu_context1024')
    p.add_argument('--evaluate-domain',action='store_true');main(p.parse_args())
