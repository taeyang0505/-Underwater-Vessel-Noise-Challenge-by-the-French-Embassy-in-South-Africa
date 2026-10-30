"""Read-only, mask-aware R17 audit definitions; no fitted features."""
from pathlib import Path
import json, hashlib, warnings
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r17_eda_audit_20261030'
FREQ=10.**np.arange(1.5,4.,.1)
MEMBERS=['lightgbm5','catboost5','dasheng_base','dinov2_small','timemoe']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def jsave(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n')
def csv(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(x).to_csv(p,index=False,encoding='utf-8-sig')
def band_curve(db,lo,hi):
 a=np.asarray(db,float)[:,:,lo:hi];o=np.isfinite(a);base=np.nanmax(a,axis=(1,2),keepdims=True)
 e=np.where(o,10**((a-base)/10),0).sum(2)
 return np.where(o.any(2),10*np.log10(np.maximum(e,1e-300))+base[:,0,0,None],np.nan)
def corr(a,b,lag=0):
 if lag:a,b=a[:,:-lag],b[:,lag:]
 o=np.isfinite(a)&np.isfinite(b);n=o.sum(1);aa=np.where(o,a,0);bb=np.where(o,b,0)
 aa=np.where(o,aa-aa.sum(1)[:,None]/np.maximum(n[:,None],1),0);bb=np.where(o,bb-bb.sum(1)[:,None]/np.maximum(n[:,None],1),0)
 den=np.sqrt((aa*aa).sum(1)*(bb*bb).sum(1));return np.divide((aa*bb).sum(1),den,out=np.full(len(a),np.nan),where=(den>1e-10)&(n>=3))
def physical_features(db):
 d=np.asarray(db,float);o=np.isfinite(d);mo=o.any(2);n=len(d)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  med=np.nanmedian(d,1);bump=np.nanmax(d,1)-med;low=band_curve(d,0,13);mid=band_curve(d,10,20);high=band_curve(d,20,25);safe=band_curve(d,15,23);broad=band_curve(d,0,25)
  f={'n_obs':mo.sum(1),'missing_first':~mo[:,0],'missing_last':~mo[:,-1],
     'bump_low_db':np.nanmax(low,1)-np.nanmedian(low,1),'bump_mid_db':np.nanmax(mid,1)-np.nanmedian(mid,1),
     'bump_high_db':np.nanmax(high,1)-np.nanmedian(high,1),'bump_safe_db':np.nanmax(safe,1)-np.nanmedian(safe,1),
     'std_broadband_db':np.nanstd(broad,1),'corr_low_high':corr(low,high),'corr_mid_high':corr(mid,high),
     'lag1_mid_high':corr(mid,high,1),'lag2_mid_high':corr(mid,high,2),'tilt_db':np.nanmedian(high-low,1)}
  run=np.zeros(n,int);maximum=np.zeros(n,int)
  for t in range(15):run=np.where(mo[:,t],run+1,0);maximum=np.maximum(maximum,run)
  f['longest_observed_run']=maximum
  for name,curve in [('HF',high),('MF',mid),('LF',low)]:
   event=np.isfinite(curve)&(curve>np.nanquantile(curve,.2,axis=1)[:,None]+3)
   f[name+'_above_q20_plus3_count']=event.sum(1);run=np.zeros(n,int);maximum=np.zeros(n,int)
   for t in range(15):run=np.where(event[:,t],run+1,0);maximum=np.maximum(maximum,run)
   f[name+'_longest_bump_run']=maximum
  off=np.nanmax(d,axis=(1,2),keepdims=True);e=np.where(o,10**((d-off)/10),0);den=e.sum((1,2))
  for name,lo,hi in [('LF',0,9),('MF',9,20),('HF',20,25)]:f[name+'_energy_fraction']=e[:,:,lo:hi].sum((1,2))/den
  for i in range(25):f[f'band{i:02d}_max_minus_median']=bump[:,i]
  for i in range(25):f[f'band{i:02d}_relative_median']=med[:,i]-np.nanmedian(med,1)
  f['quantization_error_max']=np.nanmax(abs(d*10-np.rint(d*10)),axis=(1,2))
 return pd.DataFrame(f)
def batches_features(db):return pd.concat([physical_features(db[i:i+512]) for i in range(0,len(db),512)],ignore_index=True)
def panel_data(name):
 from ops.run_r16_postprocess import panel_data as get
 return get(name,ROOT/'models/r16_pretrained_cuda_bf16_r1/both/final/GLOBAL_FREEZE.json')[0]
def table(rows,cols):
 return pd.DataFrame(rows)[cols].to_markdown(index=False,floatfmt='.6f')
