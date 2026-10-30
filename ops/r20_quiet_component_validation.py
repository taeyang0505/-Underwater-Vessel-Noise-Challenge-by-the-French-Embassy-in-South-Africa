from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from ops.r20_quiet_background_features import quiet
out=root/'reports/r20_integrated_cpu_diagnostics_r1/quiet_features'
freeze=json.loads((out/'COEFFICIENT_FREEZE.json').read_text());coeff={int(k):np.array(v) for k,v in freeze['coefficients'].items()}
rows=[]
for name,path in [('vanilla_addition',root/'reports/r19_multislice_law_transfer_r1/law_panels/vanilla_addition.npz'),('hardneg3_full15',root/'reports/r20_integrated_cpu_diagnostics_r1/hard_panels/panels/hardneg3_full15.npz')]:
 with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in ['db','share','background_observed_power','added_observed_power','active_minutes']}
 x=d['db'];b=d['background_observed_power'].astype(float);a=d['added_observed_power'].astype(float);y=d['share'];o=np.isfinite(x)
 # Input powers normalized independently of component truths; components are
 # rescaled by a COMMON per-window scalar matching total rounded observation
 # energy to raw component E=B+A. Component caches already use an arbitrary
 # per-window source unit; they are NOT absolute 10**(db/10) powers.
 e_obs=np.where(o,10**((x-np.nanmax(x,(1,2))[:,None,None])/10),0)
 scale=e_obs.sum((1,2))/np.maximum((b+a).sum((1,2)),1e-300)
 bb=b*scale[:,None,None];aa=a*scale[:,None,None]
 for k,c in coeff.items():
  qb,e,obs,ix,_=quiet(x,k);bh=qb*c
  addedquiet=aa[np.arange(len(x))[:,None],ix].sum((1,2));eq=(aa+bb)[np.arange(len(x))[:,None],ix].sum((1,2))
  frac=np.divide(addedquiet,eq,out=np.zeros_like(eq),where=eq>0)
  bt=bb.sum((1,2));estbt=(bh*obs.sum(1)).sum(1)
  err=np.divide(estbt-bt,bt,out=np.full_like(bt,np.nan),where=bt>0)
  l1=np.divide(np.where(obs,abs(bh[:,None,:]-bb),0).sum((1,2)),bt,out=np.full_like(bt,np.nan),where=bt>0)
  yh=np.clip(1-estbt/e.sum((1,2)),0,1)
  for sl,m in [('all',np.ones(len(y),bool)),('y=0',y==0),('positive',y>0),('positive_active>=12',(y>0)&(d['active_minutes']>=12))]:
   valid=m&np.isfinite(err)&np.isfinite(yh)
   if not valid.any():continue
   rows.append({'panel':name,'K':k,'slice':sl,'n':int(valid.sum()),'quiet_known_added_fraction_mean':float(frac[valid].mean()),'quiet_A_exact_zero_fraction':float((addedquiet[valid]==0).mean()),'Btotal_relative_bias_mean':float(err[valid].mean()),'Btotal_relative_error_median':float(np.median(err[valid])),'B_timeconstant_normalized_L1_median':float(np.median(l1[valid])),'yhat_RMSE':float(np.sqrt(np.mean((yh[valid]-y[valid])**2))),'yhat_bias':float(np.mean(yh[valid]-y[valid]))})
pd.DataFrame(rows).to_csv(out/'TRUE_COMPONENT_QUIET_CONTAMINATION.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
