"""Aggregated training-group spectral prototypes, never a clip-identity index."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def normalize_shape(power):
    x=10*np.log10(np.maximum(power,1e-30));x-=x.mean(-1,keepdims=True)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def build(output):
    p=ROOT/'data/processed';m=pd.read_parquet(p/'external_clip_manifest.parquet')
    bank=np.load(p/'external_segments_power.npy',mmap_mode='r');groups={}
    for r in m[m.split.eq('train')].itertuples():
        s=bank[int(r.segment_start):int(r.segment_start+r.segment_count)].mean(0,dtype=np.float64)
        groups.setdefault(r.group_key,[]).append(normalize_shape(s))
    X=np.stack([np.mean(v,axis=0) for v in groups.values()]);X/=np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
    rng=np.random.default_rng(42);k=min(8,max(1,len(X)//3));C=X[rng.choice(len(X),k,replace=False)].copy()
    for _ in range(30):
        ids=np.argmax(X@C.T,1)
        C=np.stack([X[ids==j].mean(0) if (ids==j).any() else C[j] for j in range(k)])
        C/=np.maximum(np.linalg.norm(C,axis=1,keepdims=True),1e-12)
    counts=np.bincount(ids,minlength=k);C=C[counts>=3];counts=counts[counts>=3]
    if not len(C):C=X.mean(0,keepdims=True);counts=np.array([len(X)])
    np.savez_compressed(output,centres=C.astype('float32'),group_counts=counts,
        manifest_hash=json.loads((p/'external_preparation_manifest.json').read_text())['source_hash'])
    return {'prototypes':len(C),'training_source_groups':len(X),'minimum_groups_per_prototype':int(counts.min()),
            'no_individual_clip_retrieval':True,'test_accessed':False}

def features(db,path):
    with np.load(path) as z:C=z['centres']
    x=np.asarray(db,dtype=np.float64);peak=np.nanmax(x,axis=(1,2),keepdims=True)
    energy=np.where(np.isfinite(x),10**((x-peak)/10),0).sum(1)
    shape=normalize_shape(energy)
    return (shape@C.T).astype('float32')
