"""Known synthetic component proxies from q and rounded E, never Test labels."""
import sys,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.r17_audit_common import *
from pipeline.r16_multisupport_data import role

def main():
 out=OUT/'shortcut';quant=[];mask=[];jumps=[];conds=[]
 for name in ['TRAIN_D0','TRAIN_H0','selection','local_gate','local_report','domain_holdout']:
  d=role(0,name[-2:],'train')if name.startswith('TRAIN_')else panel_data(name);db=d['db'];q=d['cell_share'];y=d['share'];obs=np.isfinite(db);off=np.nanmax(db,axis=(1,2),keepdims=True);E=np.where(obs,10**((db-off)/10),0);B=E*(1-q);A=E*q
  LFbg=B[:,:,:9].sum((1,2))/np.maximum(B.sum((1,2)),1e-30);LFbgE=B[:,:,:9].sum((1,2))/E.sum((1,2));nobs=obs.any(2).sum(1)
  for z in [False,True]:
   ix=(y>0)==z;delta=abs(db[ix]*10-np.rint(db[ix]*10))
   pool_error=A.sum((1,2))/E.sum((1,2))-y
   quant.append(dict(dataset=name,positive=z,n=int(ix.sum()),quantization_error_max=float(np.nanmax(delta)),raw_q_rounded_E_pool_error_max=float(np.max(abs(pool_error[ix])))))
  for n in np.unique(nobs):
   ix=nobs==n;mask.append(dict(dataset=name,n_obs=int(n),n=int(ix.sum()),actual_positive_fraction=float((y[ix]>0).mean()),target_mean=float(y[ix].mean()),active_minutes_mean=float((q[ix]>0).any(2).sum(1).mean())))
  active=(q>0).any(2);curve=band_curve(db,10,25)
  for kind in ['support_start','support_end','other_adjacent']:
   valid=obs.any(2)[:,1:]&obs.any(2)[:,:-1];changed=(~active[:,:-1]&active[:,1:])if kind=='support_start'else(active[:,:-1]&~active[:,1:])if kind=='support_end'else(active[:,:-1]==active[:,1:])
   delta=(curve[:,1:]-curve[:,:-1])[valid&changed]
   if len(delta):jumps.append(dict(dataset=name,condition=kind,n_pairs=len(delta),signed_delta_median=np.median(delta),absolute_delta_median=np.median(abs(delta)),absolute_delta_p90=np.quantile(abs(delta),.9)))
  if name in ['selection','local_gate','local_report','domain_holdout']:
   f=pd.read_parquet(OUT/'eda'/f'{name}.parquet');f['LF_background_fraction']=LFbg;f['LF_background_over_totalE']=LFbgE
   if name=='selection':
    ids=load(ROOT/'models/r16_v1_fourway_r2/predictions/selection.npz')['ensemble_fit_row_indices'];f=f.iloc[ids].copy();name='ensemble_fit'
   f=f[f.share.between(.05,.15)].copy()
   for c in ['LF_background_fraction','LF_background_over_totalE']:
    f['condition_bin']=pd.cut(f[c],[0,.25,.5,.75,.9,1.00001],include_lowest=True).astype(str)
    for k,z in f.groupby('condition_bin'):
     for m in ['lightgbm5','catboost5','dinov2_small','dasheng_base']:
      err=z['error_'+m];conds.append(dict(panel=name,y_interval='0.05..0.15',condition=c,value=k,n=len(z),model=m,target_mean=z.share.mean(),prediction_mean=z['pred_'+m].mean(),rmse=np.sqrt((err*err).mean()),bias=err.mean()))
  print('COMPONENT',name,flush=True)
 csv(out/'quantization_by_true_class.csv',quant);csv(out/'support_mask_by_observation_density.csv',mask);csv(out/'insertion_boundary_jumps.csv',jumps);csv(out/'same_y_background_LF_error.csv',conds)
 jsave(out/'component_audit_definition.json',{'raw_components_stored':False,'relative_components':'E=10^((rounded_db-window_max)/10); A_proxy=E*known_q; B_proxy=E*(1-known_q)','quantization_limitation':'q/y were computed from unrounded components; reconstructed proxies are not original exact A/B. Pool-error maxima reported; labels unchanged. Ratios common-gain invariant.','boundary_analysis':'observed adjacent clock minutes only; temporal added-support starts/ends from known q>0; no waveform causality inferred','Test_used':False,'training':False})
if __name__=='__main__':main()
