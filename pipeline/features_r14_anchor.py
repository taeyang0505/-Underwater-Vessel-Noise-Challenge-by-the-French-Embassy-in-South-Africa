"""R14 explicit low-anchor representations; no new acoustic information.

These maps expose contrasts already encoded by the relative spectral profile.
They are not a proof that a low band is pure background, nor a gain calibration.
Frozen TRAIN PCA mean is used as a nuisance spectral template; no query fit.
All added maps use safe bands 0..22. Raw energy labels are never accepted.
"""
import warnings
import numpy as np
from pipeline.features_r12 import transform_channels, compact_relative_features, validate_spectral_basis
from pipeline.relative_grid_r6 import quantized_db

ANCHOR_NAMES=('mid250_3162_minus_low31_160_div20',
              'train_mean_profile_low_anchored_residual_div20',
              'train_mean_profile_low_anchored_positive_fraction')
LOW=slice(0,8)  # 31.6--158.5Hz
MID=slice(9,21) # 251.2--3162.3Hz

def anchor_maps(db,basis):
    validate_spectral_basis(basis)
    d=quantized_db(db);o=np.isfinite(d)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        low=np.nanmean(d[:,:,LOW],2,keepdims=True)
        mid=np.nanmean(d[:,:,MID],2,keepdims=True)
    mean=np.asarray(basis['mean_db'],float)
    offset=mean-mean[LOW].mean()
    valid=o&np.isfinite(low);lift=np.where(np.isfinite(mid-low),(mid-low)/20,0)
    a=np.where(valid,np.broadcast_to(lift,d.shape),0)
    b=np.zeros_like(d);c=np.zeros_like(d)
    residual=d[:,:,:23]-(low+offset[None,None,:])
    b[:,:,:23]=np.where(valid[:,:,:23],residual/20,0)
    c[:,:,:23]=np.where(valid[:,:,:23],np.maximum(0,1-np.power(10.,np.clip(-residual/10,-30,30))),0)
    a[:,:,23:]=0
    result=np.stack((a,b,c),1).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite anchor maps')
    return result

def channels(db,basis):
    x,w=transform_channels(db,basis,True)
    return np.concatenate((x,anchor_maps(db,basis)),1),w

def tree_features(db,basis,batch=256):
    pieces=[]
    for s in range(0,len(db),batch):
        d=db[s:s+batch];base=compact_relative_features(d,basis,True)
        maps=anchor_maps(d,basis)[:,:,:,:23].astype(float)
        obs=np.isfinite(d[:,:,:23]);n=obs.sum(1)
        cols=[]
        for v in maps.transpose(1,0,2,3):
            m=np.divide(v.sum(1),n,out=np.zeros_like(n,dtype=float),where=n>0)
            var=np.divide(np.where(obs,(v-m[:,None,:])**2,0).sum(1),n,out=np.zeros_like(m),where=n>0)
            mx=np.max(np.where(obs,v,-np.inf),1);mx=np.where(n>0,mx,0)
            cols.append(np.stack((m,np.sqrt(np.maximum(var,0)),mx),2).reshape(len(d),-1))
        result=np.concatenate((base,*cols),1).astype(np.float32)
        if result.shape[1]!=880 or not np.isfinite(result).all():raise ValueError('Expected 880 finite features')
        pieces.append(result)
    return np.concatenate(pieces)
