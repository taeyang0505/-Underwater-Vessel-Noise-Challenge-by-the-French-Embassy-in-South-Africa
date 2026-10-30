"""R20 hard-panel frozen CPU inference using unchanged R19 model loaders.

No retraining, policy fitting or Test access. Torch/LightGBM processes are
separate. Labels are physical added-recording energy; a separately named
counterfactual vessel label is reported only for the ambient sensitivity arm.
"""
from pathlib import Path
import argparse, datetime as dt, hashlib, json, os, subprocess, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops import r19_frozen_law_inference as legacy
BASE=ROOT/'reports/r20_integrated_cpu_diagnostics_r1/hard_panels'
OUT=BASE/'inference';PANELS=BASE/'panels'
NAMES=('hardneg1_natural','hardneg2_ambient','hardneg3_full15','scale_laws')
sha=legacy.sha;read=legacy.read;write=legacy.write;save=legacy.save;load=legacy.load
original_source_pins=legacy.source_pins
def source_pins():
 return {**original_source_pins(),'ops/r20_frozen_hard_inference.py':sha(__file__),
  'ops/r20_hard_panels.py':sha(ROOT/'ops/r20_hard_panels.py')}
def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 frozen=read(BASE/'PANELS_FROZEN.json');parts=[];meta=[];pins={}
 for name in NAMES:
  p=PANELS/(name+'.npz');d=load(p);n=len(d['db'])
  expected=next(m['sha256']for m in frozen['panels']if m['panel']==name)
  if sha(p)!=expected:raise ValueError('Panel pin changed')
  if d['db'].shape!=(n,15,25) or np.isinf(d['db']).any():raise ValueError('Bad panel shape')
  if not np.array_equal(d['share']>0,d['presence']>0):raise ValueError('Physical energy label changed')
  pins[str(p.relative_to(ROOT))]=expected;parts.append(d['db'].astype(np.float32));meta.append({'law':name,'n':n,'path':str(p.relative_to(ROOT))})
 db=np.concatenate(parts);seen={};unique=[];inverse=[]
 for row in db:
  key=np.nan_to_num(row,nan=np.float32(-1e20)).tobytes();j=seen.get(key)
  if j is None:j=len(unique);seen[key]=j;unique.append(row)
  inverse.append(j)
 u=np.stack(unique);inverse=np.array(inverse,np.int64)
 assert np.array_equal(u[inverse],db,equal_nan=True)
 p={'status':'registered_before_frozen_model_predictions','created_KST':legacy.now(),
  'laws':meta,'input_sha256':pins,'source_sha256':source_pins(),'panels_frozen_sha256':sha(BASE/'PANELS_FROZEN.json'),
  'rows_total':len(db),'exact_unique_db_rows':len(u),'exact_repeats_reused':len(db)-len(u),
  'device':'cpu','precision':'FP32 no autocast','new_training':False,'optimizer_steps':0,'Test_or_query_read':False,
  'old_B1_weights':{'lightgbm':.60,'dinov2_small':.25,'dasheng_base':.15},
  'new_B2_weights':{'lightgbm':.50,'dinov2_small':.30,'dasheng_base':.20},
  'D0_H0_weights':[.5,.5],'classification':'independent classifier probabilities, same predetermined ensemble weights, no share mixing',
  'no_policy_or_threshold_fitting':True,'counterfactual_labels':'ambient-vessel analysis only, not official energy task ground truth'}
 pp=OUT/'PROTOCOL.json'
 if pp.exists():
  old=read(pp)
  for k in ['input_sha256','source_sha256','rows_total','exact_unique_db_rows']:
   if old[k]!=p[k]:raise ValueError('Immutable protocol changed '+k)
 else:write(pp,p)
 save(OUT/'input_index.npz',{'inverse':inverse})
 return u,inverse,meta

def ece(z,p):
 z=np.asarray(z);p=np.asarray(p);b=np.minimum((p*10).astype(int),9);out=0.
 for i in range(10):
  m=b==i
  if m.any():out+=m.mean()*abs(z[m].mean()-p[m].mean())
 return float(out)
def metrics(y,z,p,c):
 from sklearn.metrics import roc_auc_score,average_precision_score,r2_score
 y=np.asarray(y,float);z=np.asarray(z,bool);p=np.asarray(p,float);c=np.asarray(c,float);e=p-y
 two=np.unique(z).size==2
 return {'n':len(y),'positive_n':int(z.sum()),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(abs(e).mean()),
  'bias':float(e.mean()),'r2':float(r2_score(y,p))if len(y)>1 and np.var(y)>0 else None,
  'target_mean':float(y.mean()),'target_rms':float(np.sqrt(np.mean(y*y))),'share_mean':float(p.mean()),'share_rms':float(np.sqrt(np.mean(p*p))),
  'score_mean':float(c.mean()),'score_ge090_fraction':float((c>=.9).mean()),'score_ge050_fraction':float((c>=.5).mean()),
  'score_AUC':float(roc_auc_score(z,c))if two else None,'share_AUC':float(roc_auc_score(z,p))if two else None,
  'score_PR_AUC':float(average_precision_score(z,c))if two else None,'brier':float(np.mean((c-z)**2)),'ece10':ece(z,c)}
def finalize():
 import pandas as pd
 prepare();legacy.verify_protocol();pred={};modelpins={}
 for generation in ['old','new']:
  for family in legacy.FAMILIES:
   p=OUT/f'{generation}_{family}.npz';r=read(p.with_suffix('.json'))
   if sha(p)!=r['prediction_sha256']:raise ValueError('Prediction hash changed')
   modelpins.update(r['model_sha256']);d=load(p)
   for regime in ['D0','H0']:pred[f'{generation}_{family}_{regime}']=(d['share_'+regime],d['prob_'+regime])
   pred[f'{generation}_{family}']=(.5*(d['share_D0']+d['share_H0']),.5*(d['prob_D0']+d['prob_H0']))
 for name,g,w in [('old_B1','old',[.6,.25,.15]),('new_B2','new',[.5,.3,.2])]:
  pred[name]=tuple(sum(a*pred[f'{g}_{fam}'][i]for a,fam in zip(w,legacy.FAMILIES))for i in [0,1])
 offset=0;allmetrics=[];rowparts=[];cross=[]
 for name in NAMES:
  d=load(PANELS/(name+'.npz'));n=len(d['db']);y=d['share'];z=y>0
  row=pd.DataFrame({k:d[k]for k in ['sample_id','base_window_index','site','recorder','reference_index','kind','law','active_minutes','counterfactual_vessel_presence','counterfactual_share']})
  row['panel']=name;row['target_share']=y;row['physical_presence']=z.astype(int)
  for model,(pp,cc) in pred.items():
   p,c=pp[offset:offset+n],cc[offset:offset+n];row[model+'_share']=p;row[model+'_score']=c
   slices={'all':np.ones(n,bool),'y=0':y==0,'0<y<=0.05':z&(y<=.05),'0<y<=0.10':z&(y<=.10),
    '0.05<y<=0.15':(y>.05)&(y<=.15),'y>0.15':y>.15,'active>=12 & y>0':z&(d['active_minutes']>=12)}
   slices.update({f'kind:{k}':d['kind']==k for k in np.unique(d['kind'])})
   if name=='scale_laws':slices.update({f'law:{l}+zero':(d['law']==l)|(y==0)for l in ['native_peak','integrated_target','HF_SNR_scalar']})
   for label,ix in slices.items():
    if ix.any():allmetrics.append({'panel':name,'model':model,'label_semantics':'physical_added_recording','slice':label,**metrics(y[ix],z[ix],p[ix],c[ix])})
   if name=='hardneg2_ambient':
    cf=d['counterfactual_vessel_presence']>0;cy=d['counterfactual_share']
    allmetrics.append({'panel':name,'model':model,'label_semantics':'UNVERIFIED_counterfactual_vessel_only_NOT_official','slice':'all',**metrics(cy,cf,p,c)})
   # Pairwise inversion is exactly related to AUC; expose direction and ties.
   if z.any() and (~z).any():
    for target,v in [('share',p),('score',c)]:
     delta=v[~z,None]-v[z][None,:]
     cross.append({'panel':name,'model':model,'target':target,'negative_gt_positive_fraction':float((delta>0).mean()),'tie_fraction':float((delta==0).mean())})
  rowparts.append(row);offset+=n
 rows=pd.concat(rowparts,ignore_index=True);df=pd.DataFrame(allmetrics)
 rows.to_csv(OUT/'ALL_INDIVIDUAL_AND_BLEND_ROW_PREDICTIONS.csv',index=False)
 df.to_csv(OUT/'ALL_INDIVIDUAL_AND_BLEND_METRICS.csv',index=False)
 df[(df['slice']=='all')&df.model.isin(['old_B1','new_B2'])].to_csv(OUT/'FOUR_GROUP_BLEND_METRICS.csv',index=False)
 pd.DataFrame(cross).to_csv(OUT/'PAIRWISE_RANK_INVERSION.csv',index=False)
 # Matched ship / dedicated ambient / weak-background effect: same B,target,geometry.
 effects=[];q=rows[rows.panel.eq('hardneg2_ambient')]
 for model in ['old_B1','new_B2']+[f'{g}_{f}'for g in ['old','new']for f in legacy.FAMILIES]:
  ship=q[q.kind.eq('ship_added')].reset_index(drop=True)
  for kind in ['ambient16_added','weak_background22_added']:
   v=q[q.kind.eq(kind)].reset_index(drop=True)
   assert np.allclose(v.target_share,ship.target_share) and np.array_equal(v.base_window_index,ship.base_window_index)
   effects.append({'model':model,'comparison':kind+' minus matched ship','n':len(v),
    'share_delta_mean':float((v[model+'_share']-ship[model+'_share']).mean()),'score_delta_mean':float((v[model+'_score']-ship[model+'_score']).mean()),
    'physical_target_mean':float(v.target_share.mean()),'interpretation':'Equal responses can be correct for whole-added-recording target; not proof of equipment shortcut.'})
 pd.DataFrame(effects).to_csv(OUT/'MATCHED_AMBIENT_MINUS_SHIP.csv',index=False)
 files=[p for p in OUT.iterdir()if p.is_file()and p.name not in ['OUTPUT_SHA256.json','COMPLETED.json','progress.json']]
 write(OUT/'OUTPUT_SHA256.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size}for p in files})
 write(OUT/'COMPLETED.json',{'status':'completed','created_KST':legacy.now(),'rows':len(rows),'device':'cpu','optimizer_steps':0,'new_training':False,
  'Test_read':False,'model_sha256':modelpins,'all_metrics_sha256':sha(OUT/'ALL_INDIVIDUAL_AND_BLEND_METRICS.csv')})
 print(df[(df['slice']=='all')&df.model.isin(['old_B1','new_B2'])].to_string(index=False),flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','neural','trees','finalize','all'],default='all');args=parser.parse_args()
 sys.addaudithook(legacy.forbid_test)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false')
 legacy.OUT=OUT;legacy.PANELS=PANELS;legacy.LAWS=NAMES
 legacy.prepare=prepare;legacy.source_pins=source_pins
 if args.phase=='prepare':prepare()
 elif args.phase=='neural':legacy.neural()
 elif args.phase=='trees':legacy.trees()
 elif args.phase=='finalize':finalize()
 else:
  prepare()
  for phase in ['neural','trees','finalize']:
   subprocess.run([sys.executable,str(Path(__file__).resolve()),'--phase',phase],cwd=ROOT,env=os.environ.copy(),check=True)
if __name__=='__main__':main()
