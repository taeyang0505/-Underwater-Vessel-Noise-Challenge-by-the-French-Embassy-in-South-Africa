from pathlib import Path
import json,hashlib,datetime,itertools
import numpy as np,pandas as pd
from pipeline.dataset_r12 import (ROOT,validate_r12_bundle,R9Sources,ARMS,VAL_ROLES,
    numeric_fixture,sha,r11_used_sources,synthesize_r12)
OUT=ROOT/'data/synthetic/r12_cpu_ablation_v3'; M=validate_r12_bundle(OUT)
S=R9Sources(); checks=[]; profiles={}; stored={}
def check(name,ok,**details):
 checks.append({'name':name,'passed':bool(ok),**details}); assert ok,name
for arm in ARMS:
 with np.load(OUT/M['arms'][arm]['train']['path'],allow_pickle=False) as z: stored[arm]={k:z[k] for k in z.files}
subset=np.load(OUT/M['cnn_subset']['path'],allow_pickle=False)
zero=stored['L0_H0']['explicit_no_source'];check('same 6000 row indices/no duplicates',len(subset)==6000 and len(set(subset))==6000)
for arm,a in stored.items():
 check(arm+' exact zero schedule',np.array_equal(a['explicit_no_source'],zero))
 check(arm+' 6000 subset composition',a['explicit_no_source'][subset].sum()==1200 and (a['share'][subset]>0).sum()==4800)
 check(arm+' all TRAIN base IDs',set(a['base_window_index'])<=set(S.role_metadata(0,'train').base_window_index))
for l in (0,1):
 a,b=stored[f'L{l}_H0'],stored[f'L{l}_H1'];pos=~zero
 for k in ('db','unrounded_db','cell_share','share','presence','base_window_index','reference_index','family'):
  check(f'L{l} HF preserves positive '+k,np.array_equal(a[k][pos],b[k][pos],equal_nan=a[k].dtype.kind=='f'))
 check(f'L{l} HF modifies only1200zero sourceIDs',np.count_nonzero(a['base_window_index']!=b['base_window_index'])==1200)
for h in (0,1):
 a,b=stored[f'L0_H{h}'],stored[f'L1_H{h}'];long=np.char.startswith(b['family'],'long_')
 check(f'H{h} 2880 long-only modifications',long.sum()==2880 and long[subset].sum()==1440)
 for k in ('db','cell_share','share','reference_index','base_window_index'):
  check(f'H{h} unaffected rows '+k,np.array_equal(a[k][~long],b[k][~long],equal_nan=a[k].dtype.kind=='f'))
 check(f'H{h} long originals shared background draws',np.array_equal(a['base_window_index'],b['base_window_index']))
check('R12 v2-v3 numeric artifacts unchanged',all(sha(OUT/p)==sha(ROOT/'data/synthetic/r12_cpu_ablation_v2'/p) for p in M['file_sha256'] if p.endswith(('.npz','.npy','.parquet'))))
source_sets={};group_sets={};used,_,_=r11_used_sources(ROOT)
paths={arm:M['arms'][arm]['train'] for arm in ARMS}|M['common_validation']|{'high_snr_long':M['high_snr_long']}
for role,entry in paths.items():
 with np.load(OUT/entry['path'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
 n=len(a['share']);o=np.isfinite(a['db']);q=a['cell_share'].astype(np.float64);y=a['share'].astype(np.float64)
 check(role+' numeric schema',a['db'].shape==(n,15,25) and not any(v.dtype.kind=='O' for v in a.values()) and np.isfinite(q).all() and np.all((q>=0)&(q<=1)) and np.all(q[~o]==0))
 check(role+' exact20pct known zeros/observed positives',np.count_nonzero(y==0)*5==n and np.array_equal(y>0,~a['explicit_no_source']) and len(set(a['base_window_index']))==n)
 def powers(db):
  top=np.nanmax(db,axis=(1,2));e=np.where(o,10**((db-top[:,None,None])/10),0);return e/e.sum((1,2))[:,None,None]
 raw=(powers(a['unrounded_db'])*q).sum((1,2));rounded=(powers(a['db'].astype(float))*q).sum((1,2))
 rawerr=float(np.max(abs(raw-y)));roundgap=float(np.max(abs(rounded-y)))
 check(role+' independent saved-q/raw-energy target oracle',rawerr<1e-7,max_abs_error=rawerr)
 rec=json.loads((OUT/entry['recipes_path']).read_text());bound=np.array([r['rounding_bound'] for r in rec])
 check(role+' rounded-weight theoretical bound',np.all(abs(rounded-y)<=bound+1e-7),max_gap=roundgap)
 if role in VAL_ROLES or role=='high_snr_long':
  source_sets[role]=set(map(int,a['base_window_index']));group_sets[role]=set(a['background_group']);check(role+' excludes all R11 used IDs',not(source_sets[role]&used))
 ix=np.unique(np.linspace(0,n-1,8).astype(int));meta=pd.read_parquet(OUT/entry['metadata_path']);err=0
 for i in ix:
  r=rec[int(i)];row=meta.iloc[int(i)].to_dict();bg=S.background(row);ref=int(r['reference_index']);out,_=synthesize_r12(bg,None if ref<0 else S.reference_minutes[ref],r['seed'],r['family'],no_source=r['no_source'],snr_range=(10.,20.) if role=='high_snr_long' else (-10.,10.))
  for k in ('db','unrounded_db','cell_share','share','presence'):
   check(role+f' exact replay {int(i)}/{k}',np.array_equal(np.asarray(out[k]),np.asarray(a[k][i]),equal_nan=np.asarray(out[k]).dtype.kind=='f'))
 profiles[role]={'rows':n,'raw_target_error':rawerr,'rounded_gap':roundgap,'replayed_indices':ix.tolist()}
for x,y in itertools.combinations(VAL_ROLES,2):
 check(x+'/'+y+' unique IDs and whole groups',not(source_sets[x]&source_sets[y]) and not(group_sets[x]&group_sets[y]))
check('high-SNR new eval IDs',not(source_sets['eval']&source_sets['high_snr_long']))
with np.load(OUT/'background_basis.npz',allow_pickle=False) as z:b={k:z[k] for k in z.files}
tr=S.role_metadata(0,'train');bm=pd.read_parquet(OUT/'background_basis_metadata.parquet')
check('PCA background TRAIN-only unique2000',len(b['db'])==2000 and len(set(b['base_window_index']))==2000 and set(b['base_window_index'])<=set(tr.base_window_index))
for start in range(0,len(bm),128):
 rows=bm.iloc[start:start+128];actual=np.stack([S.background(row) for _,row in rows.iterrows()]);check('PCA raw background byte parity '+str(start),np.array_equal(actual,b['db'][start:start+len(rows)],equal_nan=True))
S.assert_unchanged();validate_r12_bundle(OUT)
output={'format':'r12_pretraining_dataset_verification_v1','status':'passed','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'data_path':OUT.relative_to(ROOT).as_posix(),'manifest_sha256':sha(OUT/'manifest.json'),'data_hash':M['data_hash'],'code_sha256':M['code_sha256'],'numeric_fixture':numeric_fixture(),'checks':checks,'profiles':profiles,'all_source_code_model_policy_changes':False,'new_model_fit_or_inference_Test_or_unused_report_values':False,'validation_limitations':M['validation_semantics'],'prefit_failed_r1_preserved':True,'v2_v3_numeric_data_unchanged':True,'verification_script_sha256':sha(__file__)}
p=ROOT/'reports/R12_DATASET_VERIFICATION.json';assert not p.exists();p.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':output['status'],'checks':len(checks),'path':str(p),'sha256':sha(p),'manifest_sha256':output['manifest_sha256']}))
