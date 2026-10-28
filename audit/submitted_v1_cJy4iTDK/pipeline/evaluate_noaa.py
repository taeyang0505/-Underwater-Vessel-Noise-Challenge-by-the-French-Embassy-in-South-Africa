"""Independent source transfer test, evaluated only after model/policy freezing."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import torch
from pipeline.infer import predict
ROOT=Path(__file__).resolve().parents[1]
def main(a):
    torch.set_num_threads(4);rng=np.random.default_rng(7401)
    p=ROOT/'data/processed';meta=pd.read_parquet(p/'sa_window_manifest.parquet')
    bank=np.load(p/'sa_windows_db.npy',mmap_mode='r')
    selected=np.flatnonzero(meta.split.to_numpy()=='domain_holdout')
    noaa=ROOT/'external/noaa_sanctsound_holdout'
    refs=[np.load(noaa/(r['crop'].split('_15min')[0]+'_minutes_power.npy')) for r in json.loads((noaa/'manifest.json').read_text())]
    n=a.n;ids=rng.choice(selected,n);db=np.array(bank[ids],dtype=np.float64);valid=np.isfinite(db)
    peak=np.nanmax(db,axis=(1,2),keepdims=True);b=np.where(valid,10**((db-peak)/10),0)
    r=np.stack([refs[i%len(refs)] for i in range(n)]).astype('float64')
    r/=r.sum((1,2),keepdims=True)
    u=rng.uniform(.005,.98,n);u[rng.random(n)<.2]=0
    alpha=u*b.sum((1,2))/np.maximum((1-u)*(r*valid).sum((1,2)),1e-30)
    added=alpha[:,None,None]*r;energy=b+added
    y=(added*valid).sum((1,2))/(energy*valid).sum((1,2))
    obs=10*np.log10(np.maximum(energy,1e-30))+peak;obs[~valid]=np.nan
    pb=(a.run/'postprocess.json').read_bytes();policy=json.loads(pb)
    pred,score=predict(obs.astype('float32'),a.run,policy,torch.device(a.device))
    info={'n':n,'rmse':float(np.sqrt(np.mean((y-pred)**2))),
          'zero_addition_mean':float(pred[y==0].mean()),'seed':7401,
          'references':'3 actual contiguous NOAA 15-minute intervals, 1 deployment, 3 dates; unknown hull independence',
          'policy_sha256':hashlib.sha256(pb).hexdigest(),'used_for_model_selection':False,'test_accessed':False}
    (a.run/'noaa_independent_assessment.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--n',type=int,default=1200)
    p.add_argument('--device',default='cpu');main(p.parse_args())
