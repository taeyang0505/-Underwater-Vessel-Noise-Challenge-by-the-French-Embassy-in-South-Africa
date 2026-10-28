"""Frozen tree inference in a process without Torch's OpenMP runtime."""
from pathlib import Path
import argparse
import numpy as np
import lightgbm as lgb
from pipeline.mixtures import fixed_features,physics_features
from pipeline.prototypes import features as prototype_features
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('input',type=Path)
    p.add_argument('output',type=Path);p.add_argument('--kinds',nargs='+',required=True);a=p.parse_args()
    db=np.load(a.input);out={}
    for k in a.kinds:
        parts=[fixed_features(db)]
        if k!='lgbm':parts.append(physics_features(db))
        if k=='lgbm_retrieval':parts.append(prototype_features(db,a.run/'prototypes.npz'))
        x=np.concatenate(parts,1)
        out[k+'_share']=np.clip(lgb.Booster(model_file=str(a.run/(k+'_share.txt'))).predict(x,num_threads=1),0,1)
        out[k+'_score']=lgb.Booster(model_file=str(a.run/(k+'_presence.txt'))).predict(x,num_threads=1)
    np.savez_compressed(a.output,**out)
