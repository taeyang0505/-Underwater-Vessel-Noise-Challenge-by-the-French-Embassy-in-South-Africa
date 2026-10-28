"""Paired D0/H0 full-fold data; versioned separately from frozen R16 runs."""
from pathlib import Path
import json, hashlib, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/synthetic/r16_multisupport_v1'
PARENT=ROOT/'reports/r16_support_temporal_r1/data/manifest.json'
SEED=202610280518
REGIMES=('D0','H0')
KINDS=('bg_cnn','fdy','itransformer','tfgrid_fdy_conformer')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 with t.open('wb')as f:np.savez_compressed(f,**d)
 t.replace(p)
def load(p):
 with np.load(p,allow_pickle=False)as f:return{k:f[k]for k in f.files}
def guard(event,args):
 if event=='open'and args and isinstance(args[0],(str,bytes)):
  s=str(args[0]).lower()
  if Path(s).name in('test.csv','sample_submission.csv')or'/eda_deep/'in s or'/r15_band_comparison_'in s:raise PermissionError('No query access in data/training')
def schedule(refs,n,fold,role):
 rng=np.random.default_rng([SEED,fold,0 if role=='train'else 1]);zero=np.zeros(n,bool);zero[rng.choice(n,n//5,replace=False)]=True
 pos=np.flatnonzero(~zero);groups=[refs[refs.minute_count.between(1,4)],refs[refs.minute_count.between(5,10)],refs[refs.minute_count.ge(11)]]
 probabilities=np.array([.35,.40,.25]);probabilities[[not len(g)for g in groups]]=0;probabilities/=probabilities.sum()
 counts=np.floor(probabilities*len(pos)).astype(int)
 for j in np.argsort(-(probabilities*len(pos)-counts))[:len(pos)-counts.sum()]:counts[j]+=1
 strata=np.repeat(np.arange(3),counts);rng.shuffle(strata);ri=np.full(n,-1,int)
 for j,g in enumerate(groups):
  rows=pos[strata==j];sequence=[];available=g.reference_row_index.to_numpy(int)
  while len(sequence)<len(rows):sequence.extend(rng.permutation(available).tolist())
  ri[rows]=sequence[:len(rows)]
 from ops.r16_forward_data import OPERATORS
 ops=np.full(n,'control_eq6',dtype='<U32');a=np.resize(np.asarray(OPERATORS),len(pos));rng.shuffle(a);ops[pos]=a
 return zero,ri,ops,dict(zip(('short','medium','long'),map(int,counts)))
def materialize(source,frame,refs,fold,role,regime,sch,path):
 from ops.r16_forward_data import choose_geometry,generate_operator,channel_zero_mean
 from ops.r16_support_temporal_data import natural_geometry
 from pipeline.sampler_r9 import smooth_eq
 n=len(frame);zero,ri,op,bins=sch;ref=refs.set_index('reference_row_index');began=time.perf_counter()
 a={'db':np.empty((n,15,25),np.float32),'cell_share':np.empty((n,15,25),np.float32),'share':np.empty(n,np.float64),
    'presence':(~zero).astype(np.int64),'reference_index':ri,'operator':op,'selected_minutes':np.zeros(n,np.int64),
    'reference_component':np.full(n,'',dtype='<U64'),'family':np.full(n,'no_source',dtype='<U40'),
    'crop_begin':np.zeros(n,np.int64),'window_start':np.zeros(n,np.int64),'active_minutes':np.zeros(n,np.int64),
    'sample_id':np.asarray([f'R16MS_{fold}_{role}_{i:06d}'for i in range(n)]),
    'site':frame.site.to_numpy(dtype=str),'base_window_index':frame.base_window_index.to_numpy(np.int64),
    'background_group':frame.background_group.to_numpy(dtype=str),'explicit_no_source':zero}
 maxraw=maxround=0.
 for i,row in frame.iterrows():
  channel=np.random.default_rng([SEED,fold,0 if role=='train'else 1,i,0x4348]);geomrng=np.random.default_rng([SEED,fold,0 if role=='train'else 1,i,0x4745])
  bg=np.asarray(source.background(row),float);native=None if zero[i]else source.reference_minutes[int(ri[i])]
  g=None if zero[i]else(natural_geometry(bg,native,geomrng)if regime=='H0'else choose_geometry(bg,native,geomrng))
  eq=np.zeros(25)if zero[i]else smooth_eq(channel,6.);post=np.zeros(25)if zero[i]else channel_zero_mean(channel,6. if role=='train'else 12.)
  snr=float(channel.uniform(-10,10));sub=channel.uniform(-10,10,2)
  d,receipt=generate_operator(bg,native,g,eq,post,snr,sub,str(op[i]),zero=bool(zero[i]))
  E,A,B=d['mixture_observed_power'],d['added_observed_power'],d['background_observed_power']
  if not np.array_equal(E,B+A)or abs(A.sum()/E.sum()-d['share'])>1e-12:raise ValueError('Per-row energy identity failed')
  if not np.array_equal(np.isfinite(d['db']),d['observed_mask'])or not np.array_equal(d['presence']>0,~zero[i]):raise ValueError('Mask or physical presence differs')
  for k in('db','cell_share','share'):a[k][i]=d[k]
  a['active_minutes'][i]=np.any(d['cell_share']>0,1).sum()
  maxraw=max(maxraw,receipt['raw_pool_identity_error']);maxround=max(maxround,receipt['rounded_pool_gap'])
  if g:
   a['selected_minutes'][i]=g['selected_minutes'];a['crop_begin'][i]=g['crop_begin_minute'];a['window_start'][i]=g['window_start_minute'];a['family'][i]=g['family'];a['reference_component'][i]=str(ref.loc[int(ri[i])].reference_component)
  if i and i%10000==0:print(json.dumps({'stage':'data','fold':fold,'role':role,'regime':regime,'rows':i,'total':n,'seconds':time.perf_counter()-began}),flush=True)
 if maxround>.0058 or np.any(a['cell_share'][~np.isfinite(a['db'])]!=0)or not np.array_equal(a['share']>0,a['presence']>0):raise ValueError('Output label/quantization bound failed')
 save(path,a);check=load(path)
 if any(not np.array_equal(a[k],check[k],equal_nan=True)if a[k].dtype.kind in'fc'else not np.array_equal(a[k],check[k])for k in a):raise ValueError('NPZ round-trip differs')
 active=a['active_minutes'][~zero]
 return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'n':n,'zero_count':int(zero.sum()),'positive_support_bins':bins,
   'share_mean':float(a['share'].mean()),'share_rms':float(np.sqrt(np.mean(a['share']**2))),'positive_active_median':float(np.median(active)),
   'positive_active_ge12':int((active>=12).sum()),'max_raw_identity_error':float(maxraw),'max_rounded_pool_gap':float(maxround),
   'all_rows_energy_checked':True,'size_bytes':path.stat().st_size,'seconds':time.perf_counter()-began}
def build():
 import pandas as pd
 from pipeline.dataset_r9 import R9Sources
 from pipeline.r16_production_data import validate_production_manifest,_bucket_groups,_crossrole
 from ops.r16_support_temporal_data import new_reference_split,load_panel,LONG_TRAIN
 from ops.r16_forward_data import OPERATORS
 sys.addaudithook(guard)
 if OUT.exists():raise FileExistsError('Fresh full data version required')
 OUT.mkdir(parents=True);source=R9Sources();pins={str(PARENT.relative_to(ROOT)):sha(PARENT)}
 refs,refqa=new_reference_split(source);ra=refs[refs.r16_reference_partition.eq('A')].copy();assign,_=_bucket_groups(ra,'reference_component')
 # A metadata-only fold-label permutation puts the sole >=15-minute group in
 # fold4, so the requested neural Fold0 has genuine long TRAIN sources.
 old=assign[LONG_TRAIN];assign={k:(4 if v==old else old if v==4 else v)for k,v in assign.items()};ra['full_cv_fold']=ra.reference_component.map(assign)
 ra.to_parquet(OUT/'reference_A_folds.parquet',index=False)
 parts=[load_panel(PARENT,'local_fit',op)for op in OPERATORS];common=set.intersection(*(set(d)for d in parts));selection={k:np.concatenate([d[k]for d in parts])for k in common}
 save(OUT/'local_fit.npz',selection);fit_entry={'path':str((OUT/'local_fit.npz').relative_to(ROOT)),'sha256':sha(OUT/'local_fit.npz'),'n':len(selection['share'])}
 frozen=pd.read_parquet(PARENT.parent/'local_fit_backgrounds.parquet')
 qa=[];folds={}
 for f in range(5):
  oldmanifest=ROOT/f'data/synthetic/r16_production_v1/fold_{f:02d}/manifest.json';validate_production_manifest(oldmanifest,f)
  pins[str(oldmanifest.relative_to(ROOT))]=sha(oldmanifest);folder=OUT/f'fold_{f:02d}';folder.mkdir()
  trainrefs=ra[ra.full_cv_fold.ne(f)].copy();heldrefs=ra[ra.full_cv_fold.eq(f)].copy()
  if set(trainrefs.reference_component)&set(heldrefs.reference_component):raise ValueError('Reference split overlap')
  trainrefs.to_parquet(folder/'train_references.parquet',index=False);heldrefs.to_parquet(folder/'cv_references.parquet',index=False)
  roles={}
  for role,oldrole,rr in [('train','train',trainrefs),('cv','cv_validation',heldrefs)]:
   meta=oldmanifest.parent/(oldrole+'_backgrounds.parquet');frame=pd.read_parquet(meta).reset_index(drop=True);pins[str(meta.relative_to(ROOT))]=sha(meta)
   if role=='train':qa+=_crossrole(frame,{'shared_fit':frozen})
   frame.to_parquet(folder/(role+'_backgrounds.parquet'),index=False);sch=schedule(rr,len(frame),f,role)
   roles[role]={reg:materialize(source,frame,rr,f,role,reg,sch,folder/(role+'_'+reg+'.npz'))for reg in REGIMES}
   d0=load(ROOT/roles[role]['D0']['path']);h0=load(ROOT/roles[role]['H0']['path']);z=d0['share']==0
   for k in ('sample_id','base_window_index','reference_index','operator','explicit_no_source'):assert np.array_equal(d0[k],h0[k]),k
   for k in ('db','cell_share','share'):assert np.array_equal(d0[k][z],h0[k][z],equal_nan=True),k
  folds[str(f)]={'roles':roles,'train_reference_clips':len(trainrefs),'train_reference_components':trainrefs.reference_component.nunique(),
   'cv_reference_clips':len(heldrefs),'train_long_component_available':LONG_TRAIN in set(trainrefs.reference_component)}
  write(OUT/'progress.json',{'completed_folds':list(folds),'folds':folds})
 manifest={'status':'completed','format':'r16_multisupport_full_data_v1','seed':SEED,'source_identity':source.identity,
  'source_code_sha256':{n:sha(ROOT/n)for n in ('pipeline/r16_multisupport_data.py','ops/r16_support_temporal_data.py','ops/r16_forward_data.py','pipeline/r16_production_data.py')},
  'source_artifact_sha256':pins,'folds':folds,'selection':fit_entry,'reference_partition':refqa,'shared_fit_purge_checks':qa,
  'fivefold_scope':'nested inside original R9 Fold0 TRAIN 78869 backgrounds; A-prime75 refs only; B19/C-prime25 held globally',
  'native_ge15_component_heldout_in_fold':4,'fold4_no_native_ge11_TRAIN_support':True,'no_HF_negative_replacement':True,
  'paired_negative_arrays_exact':True,'Test_used':False,'training_oracle_target_forcing':False}
 write(OUT/'manifest.json',manifest);return manifest
def role(fold,regime,name):
 m=read(OUT/'manifest.json');e=m['selection']if name=='selection'else m['folds'][str(fold)]['roles'][name][regime]
 if sha(ROOT/e['path'])!=e['sha256']:raise ValueError('Data SHA changed')
 return load(ROOT/e['path'])
