"""Frozen tree inference in a process without Torch's OpenMP runtime."""
from pathlib import Path
import argparse,json
import numpy as np
import lightgbm as lgb
from pipeline.mixtures import fixed_features,physics_features,retrieval_features

def feature_version_for(run,kind):
    summary=run/(kind+'_training_summary.json')
    if summary.exists():
        info=json.loads(summary.read_text())
        if 'feature_version' in info:return int(info['feature_version'])
    manifest=run/'run_manifest.json'
    if manifest.exists():
        info=json.loads(manifest.read_text());return int(info.get('feature_version',info.get('arguments',{}).get('feature_version',1)))
    return 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('input',type=Path)
    p.add_argument('output',type=Path);p.add_argument('--kinds',nargs='+',required=True);a=p.parse_args()
    db=np.load(a.input);out={}
    for k in a.kinds:
        version=feature_version_for(a.run,k);parts=[fixed_features(db,version=version)]
        if k!='lgbm':parts.append(physics_features(db))
        if k=='lgbm_retrieval':parts.append(retrieval_features(db,a.run/'prototypes.npz',version=version))
        x=np.concatenate(parts,1)
        reg=lgb.Booster(model_file=str(a.run/(k+'_share.txt')))
        if reg.num_feature()!=x.shape[1]:raise ValueError('Saved tree feature count differs from frozen preprocessing')
        out[k+'_share']=np.clip(reg.predict(x,num_threads=1),0,1)
        out[k+'_score']=lgb.Booster(model_file=str(a.run/(k+'_presence.txt'))).predict(x,num_threads=1)
    np.savez_compressed(a.output,**out)
