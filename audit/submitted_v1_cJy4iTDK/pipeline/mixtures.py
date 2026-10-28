"""Energy-correct simulator; every choice comes from training sources or fixed priors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

class MixtureSampler:
    def __init__(self,split='train',prior='uniform',seed=20261024):
        p=ROOT/'data/processed'
        self.background=np.load(p/'sa_windows_db.npy',mmap_mode='r')
        self.bmeta=pd.read_parquet(p/'sa_window_manifest.parquet')
        self.emeta=pd.read_parquet(p/'external_clip_manifest.parquet')
        self.segments=np.load(p/'external_segments_power.npy',mmap_mode='r')
        self.bi=np.flatnonzero(self.bmeta.split.to_numpy()==split)
        self.ei=np.flatnonzero((self.emeta.split.to_numpy()==split)&(self.emeta.archive_role.to_numpy()=='vessel_passage'))
        if not len(self.bi) or not len(self.ei):raise ValueError(f'No source-disjoint {split} samples ready')
        self.rng=np.random.default_rng(seed);self.prior=prior;self.split=split
        self.site_indices=[ix.to_numpy() for _,ix in self.bmeta.iloc[self.bi].groupby('site').groups.items()]
    def draw(self,n):
        rng=self.rng
        # Mixture prior is declared a priori; it is never estimated from Test.csv.
        if self.prior=='site_balanced':
            bid=np.array([rng.choice(self.site_indices[rng.integers(len(self.site_indices))]) for _ in range(n)])
        else:bid=rng.choice(self.bi,n)
        db=np.array(self.background[bid],dtype=np.float64)
        valid=np.isfinite(db)
        maximum=np.nanmax(db,axis=(1,2),keepdims=True)
        b=np.where(valid,10**((db-maximum)/10),0)
        v=np.zeros_like(b);refids=[];modes=[]
        freq_axis=np.linspace(-1,1,25)
        desired=rng.uniform(.005,.98,n)
        if self.prior=='low_share':desired=rng.beta(.6,2.0,n)*.98
        elif self.prior=='bimodal':desired=rng.beta(.6,.6,n)*.98
        desired[rng.random(n)<.20]=0
        for i in range(n):
            references=[];mode_names=[]
            for _ in range(1 if rng.random()<.8 else int(rng.integers(2,4))):
                j=int(rng.choice(self.ei));r=self.emeta.iloc[j]
                s=np.asarray(self.segments[int(r.segment_start):int(r.segment_start+r.segment_count)],dtype=np.float64)
                s=s/max(float(np.mean(s.sum(1))),1e-30)
                if r.source=='HearMyShip_V1' and rng.random()<.65:
                    # A declared surrogate envelope; not a measured 15-minute transit.
                    centre=rng.uniform(0,14);width=rng.uniform(.35,5)
                    env=1/(1+((np.arange(450)/30-centre)/width)**2)
                    w=env[:,None]*s.mean(0)[None,:]
                    mode='short_CPA_spectral_profile_with_surrogate_envelope'
                else:
                    scale=rng.uniform(.75,1.5)
                    length=max(1,int(len(s)*scale))
                    start=int(rng.integers(-max(1,length//2),450))
                    q=(np.arange(450)-start)/scale
                    w=np.stack([np.interp(q,np.arange(len(s)),s[:,k],left=0,right=0) for k in range(25)],axis=1)
                    mode='measured_2s_reference_placed_and_time_rescaled'
                tilt=rng.uniform(-6,6)*freq_axis
                smooth=np.interp(freq_axis,[-1,-.33,.33,1],rng.normal(0,1.5,4))
                w*=10**((tilt+smooth)/10)[None,:]
                v[i]+=w.reshape(15,30,25).mean(1)*10**rng.uniform(-.5,.5)
                references.append(j);mode_names.append(mode)
            refids.append(references);modes.append(mode_names)
        ve=(v*valid).sum((1,2));be=b.sum((1,2))
        absent=(ve<=0)|(desired==0)
        alpha=np.divide(desired*be,(1-desired)*ve,out=np.zeros(n),where=~absent)
        v*=alpha[:,None,None]
        mixture=b+v
        target=(v*valid).sum((1,2))/mixture.sum((1,2),where=valid)
        cell=np.divide(v,mixture,out=np.zeros_like(v),where=valid&(mixture>0))
        gain=rng.uniform(-8,8,(n,1,1))
        out=10*np.log10(np.maximum(mixture,1e-30))+maximum+gain
        out[~valid]=np.nan
        assert np.isfinite(target).all() and ((target>=0)&(target<=1)).all()
        assert np.allclose((cell*np.where(valid,mixture,0)).sum((1,2))/mixture.sum((1,2),where=valid),target,atol=1e-7)
        recipes={'background_index':bid,'reference_indices':refids,'reference_modes':modes,
                 'split':self.split,'prior':self.prior,'addition_scale':alpha.tolist(),
                 'drawn_share':desired.tolist(),'common_gain_db':gain[:,0,0].tolist()}
        return out.astype('float32'),target.astype('float32'),(target>0).astype('float32'),cell.astype('float32'),recipes

def fixed_features(db):
    """No fitted/test population statistics: fixed, per-window transformations only."""
    x=np.asarray(db,dtype=np.float32);valid=np.isfinite(x)
    with np.errstate(all='ignore'):
        mean=np.nanmean(x,axis=(1,2),keepdims=True)
        rel=np.where(valid,(x-mean)/20,0)
        absolute=np.where(valid,x/100,0)
        band_mean=np.nanmean(x,axis=1)
        band_std=np.nanstd(x,axis=1)
        band_min=np.nanmin(x,axis=1)
        band_max=np.nanmax(x,axis=1)
        minute_mean=np.divide(np.where(valid,x,0).sum(2),valid.sum(2),out=np.zeros(x.shape[:2],dtype=np.float32),where=valid.sum(2)>0)
    # Fixed zero filling has an explicit mask and does not enter any energy sum.
    stats=np.concatenate([band_mean/100,band_std/20,(band_max-band_min)/20,
                          np.diff(band_mean,axis=1)/20,minute_mean/100,valid.sum(2)/25],axis=1)
    return np.nan_to_num(np.concatenate([rel.reshape(len(x),-1),absolute.reshape(len(x),-1),
                                        valid.reshape(len(x),-1),stats],axis=1),nan=0,posinf=0,neginf=0).astype('float32')

def cnn_channels(db):
    x=np.asarray(db,dtype=np.float32);valid=np.isfinite(x)
    mean=np.nanmean(x,axis=(1,2),keepdims=True)
    peak=np.nanmax(x,axis=(1,2),keepdims=True)
    power=np.where(valid,10**((x-peak)/10),0)
    weights=power/np.maximum(power.sum((1,2),keepdims=True),1e-20)
    channels=np.stack([np.where(valid,(x-mean)/20,0),np.where(valid,x/100,0),
                       np.sqrt(weights),valid.astype(np.float32)],axis=1)
    return channels.astype('float32'),weights.astype('float32')

def physics_features(db):
    """Per-window, gain-invariant energy statistics; no population fitting.

    A quantile floor is a feature, not a certified background estimate.
    """
    x=np.asarray(db,dtype=np.float64);valid=np.isfinite(x)
    peak=np.nanmax(x,axis=(1,2),keepdims=True)
    e=np.where(valid,10**((x-peak)/10),np.nan)
    total=np.nansum(e,axis=(1,2));band=np.nansum(e,axis=1)
    band=band/np.maximum(total[:,None],1e-30)
    floor=np.nanquantile(e,[.10,.20,.50],axis=1)
    excess=np.stack([np.nansum(np.maximum(e-floor[j,:,None,:],0),axis=(1,2))/total for j in range(3)],1)
    minute=np.nansum(e,axis=2)/total[:,None]
    tilt=np.log10((band[:,:8].sum(1)+1e-12)/(band[:,14:].sum(1)+1e-12))[:,None]
    entropy=-(band*np.log(np.maximum(band,1e-30))).sum(1,keepdims=True)
    return np.nan_to_num(np.concatenate([band,minute,excess,tilt,entropy],1)).astype('float32')

def row_features(db):
    """25-band gain-invariant features for the external weak-label teacher."""
    x=np.asarray(db,dtype=np.float64);rel=(x-x.mean(1,keepdims=True))/20
    energy=10**((x-x.max(1,keepdims=True))/10);energy/=energy.sum(1,keepdims=True)
    entropy=-(energy*np.log(np.maximum(energy,1e-30))).sum(1,keepdims=True)
    return np.concatenate([rel,np.diff(rel,axis=1),energy,entropy],1).astype('float32')

def foundation_features(db):
    """96 fixed window features, retaining the <=100-feature foundation regime."""
    x=np.asarray(db,dtype=np.float32)
    return np.nan_to_num(np.concatenate([physics_features(x),np.nanmean(x,1)/100,
        np.nanstd(x,1)/20,np.isfinite(x).sum((1,2))[:,None]/375],1)).astype('float32')
