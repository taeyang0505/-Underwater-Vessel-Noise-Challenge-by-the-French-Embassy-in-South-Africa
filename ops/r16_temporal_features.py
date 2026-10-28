"""Stateless R16 support/temporal ablation features. No learned transform.

Every added channel/statistic is invariant to static per-band dB offsets.
No absolute dB, frequency profile, energy denominator, PCA or LF anchor enters
these features. Energy weights remain external to the regression pooling.
"""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
GRID_NAMES=('temporal_zscore','rolling_mean2_relative_div10','rolling_mean4_relative_div10',
 'rolling_mean8_relative_div10','observed_contiguous_lag1_div10','observed_contiguous_lag2_div10',
 'observed_contiguous_lag3_div10','rolling4_persistence_above_q20_plus3dB','local3_arch_curvature_div10')

def check(db):
 if isinstance(db,dict):raise TypeError('Features accept only db arrays')
 d=np.asarray(db,np.float64)
 if d.ndim!=3 or d.shape[1:]!=(15,25)or not len(d)or np.isinf(d).any()or not np.isfinite(d).any((1,2)).all():raise ValueError('N15x25 finite/NaN dB arrays required')
 return d

def _rolling(a,obs,width):
 out=np.zeros_like(a);support=np.zeros_like(a,dtype=bool)
 for t in range(15):
  lo=max(0,t-(width-1)//2);hi=min(15,t+width//2+1);m=obs[:,lo:hi];n=m.sum(1)
  out[:,t]=np.divide(np.where(m,a[:,lo:hi],0).sum(1),n,out=np.zeros((len(a),25)),where=n>0)
  support[:,t]=n>0
 return out,support

def _quantile(d,obs,q):
 # A fully missing band receives baseline0 and remains masked in all outputs.
 a=np.where(obs,d,np.nan);empty=~obs.any(1)
 if empty.any():
  a=a.copy();a[:,0]=np.where(empty,0,a[:,0])
 return np.nanquantile(a,q,axis=1)

def temporal_grid(db):
 d=check(db);obs=np.isfinite(d);n=obs.sum(1)
 mean=np.divide(np.where(obs,d,0).sum(1),n,out=np.zeros((len(d),25)),where=n>0)
 centered=np.where(obs,d-mean[:,None],0)
 sd=np.sqrt(np.divide((centered**2).sum(1),n,out=np.zeros_like(mean),where=n>0))
 z=np.divide(centered,sd[:,None],out=np.zeros_like(d),where=sd[:,None]>1e-8)
 median=_quantile(d,obs,.5);relative=np.where(obs,(d-median[:,None])/10.,0)
 channels=[z]
 for width in(2,4,8):
  v,_=_rolling(relative,obs,width);channels.append(np.where(obs,v,0))
 for lag in(1,2,3):
  delta=np.zeros_like(d)
  for t in range(lag,15):
   valid=obs[:,t-lag:t+1].all(1)
   delta[:,t]=np.where(valid,(np.where(obs[:,t],d[:,t],0)-np.where(obs[:,t-lag],d[:,t-lag],0))/10.,0)
  channels.append(delta)
 q20=_quantile(d,obs,.2);event=obs&(d>q20[:,None]+3.)
 persistence,_=_rolling(event.astype(float),obs,4);channels.append(np.where(obs,persistence,0))
 arch=np.zeros_like(d)
 for t in range(1,14):
  left=obs[:,max(0,t-3):t];right=obs[:,t+1:min(15,t+4)]
  ln=left.sum(1);rn=right.sum(1)
  lm=np.divide(np.where(left,relative[:,max(0,t-3):t],0).sum(1),ln,out=np.zeros((len(d),25)),where=ln>0)
  rm=np.divide(np.where(right,relative[:,t+1:min(15,t+4)],0).sum(1),rn,out=np.zeros((len(d),25)),where=rn>0)
  arch[:,t]=np.where(obs[:,t]&(ln>0)&(rn>0),2*relative[:,t]-lm-rm,0)
 channels.append(arch)
 a=np.stack(channels,axis=1).astype(np.float32)
 if a.shape!=(len(d),len(GRID_NAMES),15,25)or not np.isfinite(a).all():raise ValueError('Temporal grid invalid')
 return a,obs

def _corr(a,b,obs):
 n=obs.sum(1);ma=np.divide(np.where(obs,a,0).sum(1),n,out=np.zeros(len(a)),where=n>0)
 mb=np.divide(np.where(obs,b,0).sum(1),n,out=np.zeros(len(a)),where=n>0)
 aa=np.where(obs,a-ma[:,None],0);bb=np.where(obs,b-mb[:,None],0)
 den=np.sqrt((aa*aa).sum(1)*(bb*bb).sum(1))
 return np.divide((aa*bb).sum(1),den,out=np.zeros(len(a)),where=den>1e-10)

def temporal_statistics(db):
 """Per-band multiscale summaries and within-window eigenspectrum.

 Lag statistics pair values at their original clock indices. Both endpoints
 must be observed; intervening missing minutes are not compressed or filled.
 This differs intentionally from the contiguous-support grid deltas, which
 require every minute between the two endpoints to be observed.
 """
 d=check(db);a,obs=temporal_grid(d);n=obs.sum(1);cols=[];names=[]
 for c,name in enumerate(GRID_NAMES):
  v=a[:,c].astype(float);mean=np.divide(v.sum(1),n,out=np.zeros_like(n,dtype=float),where=n>0)
  sd=np.sqrt(np.divide(np.where(obs,(v-mean[:,None])**2,0).sum(1),n,out=np.zeros_like(mean),where=n>0))
  mx=np.where(n>0,np.max(np.where(obs,v,-np.inf),1),0);mn=np.where(n>0,np.min(np.where(obs,v,np.inf),1),0)
  for stat,b in [('mean',mean),('std',sd),('max',mx),('min',mn)]:
   cols.append(b);names.extend([f'{name}/{stat}/band{f:02d}'for f in range(25)])
 z=a[:,0].astype(float)
 for lag in(1,2,3):
  for f in range(25):
   m=obs[:,lag:,f]&obs[:,:-lag,f]
   cols.append(_corr(z[:,lag:,f],z[:,:-lag,f],m)[:,None]);names.append(f'temporal_z_autocorr_lag{lag}/band{f:02d}')
 # Cross-band zero-lag correlations and normalized eigenspectrum use only the
 # current window's time-centered z values. This is not fitted PCA.
 covariance=np.einsum('ntf,ntg->nfg',z,z)/15.
 eigen=np.linalg.eigvalsh(covariance)[:,::-1];trace=np.maximum(eigen.sum(1),1e-12)
 norm=eigen/trace[:,None];cols.extend([eigen[:,:5],norm[:,:5]])
 names.extend([f'window_temporal_covariance_eigenvalue_{i+1}'for i in range(5)]+[f'window_temporal_covariance_eigenshare_{i+1}'for i in range(5)])
 for f in range(25):
  for g in range(f+1,25):
   cols.append(_corr(z[:,:,f],z[:,:,g],obs[:,:,f]&obs[:,:,g])[:,None]);names.append(f'window_z_band_corr/{f:02d}/{g:02d}')
 # True order-sensitive cross-band lagged correlations for safe LF/mid/HF
 # groups. Group means are temporal z scores, never spectral energy ratios.
 groups=((0,8),(10,20),(15,23),(20,25));group_z=[];group_obs=[]
 for lo,hi in groups:
  m=obs[:,:,lo:hi];count=m.sum(2);group_z.append(np.divide(np.where(m,z[:,:,lo:hi],0).sum(2),count,out=np.zeros((len(d),15)),where=count>0));group_obs.append(count>0)
 for lag in(1,2,3):
  for f in range(len(groups)):
   for g in range(len(groups)):
    m=group_obs[f][:,lag:]&group_obs[g][:,:-lag]
    cols.append(_corr(group_z[f][:,lag:],group_z[g][:,:-lag],m)[:,None]);names.append(f'group_z_crosscorr_lag{lag}/{f}/{g}')
 result=np.concatenate(cols,1).astype(np.float32)
 if result.shape[1]!=len(names)or not np.isfinite(result).all():raise ValueError('Temporal statistics invalid')
 return result,names

def fixture():
 began=time.perf_counter();rng=np.random.default_rng(2026102817)
 d=np.round(rng.normal(80,6,(10,15,25)),1);d[0,0]=np.nan;d[1,4:7]=np.nan;d[2,:,24]=np.nan;d[3]=77.
 offset=np.round(rng.normal(0,15,(10,1,25)),1)
 x,m=temporal_grid(d);a,names=temporal_statistics(d);b,_=temporal_statistics(d+offset);xx,_=temporal_grid(d+offset)
 gap=float(abs(a-b).max());gridgap=float(abs(x-xx).max())
 if gap>3e-5 or gridgap>3e-5:raise AssertionError('Static band offset invariance differs')
 aa,_=temporal_statistics(d)
 if not np.array_equal(a,aa):raise AssertionError('Feature non-determinism')
 # Shuffling must change ordered features for an asymmetric nonconstant trace.
 permutation=np.asarray([0,4,8,12,1,5,9,13,2,6,10,14,3,7,11]);s,_=temporal_statistics(d[:,permutation]);shuffle_delta=float(abs(a-s).max())
 if shuffle_delta<=.01:raise AssertionError('Added features do not encode time order')
 return{'status':'passed','grid_channels':list(GRID_NAMES),'statistics_columns':len(names),'static_per_band_offset_max_abs':gap,
  'grid_static_per_band_offset_max_abs':gridgap,'missing_and_constant_windows_finite':True,'deterministic_exact':True,
  'time_shuffle_changes_features_max_abs':shuffle_delta,'learned_transform':False,'PCA_or_profile_or_energy_denominator':False,
  'lag_pair_validity':'Actual clock-index lag; both observed endpoints; no compression/interpolation',
  'grid_delta_validity':'Every intermediate minute observed, including both endpoints',
  'Test_used':False,'elapsed_seconds':time.perf_counter()-began,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

if __name__=='__main__':print(json.dumps(fixture(),ensure_ascii=False))
