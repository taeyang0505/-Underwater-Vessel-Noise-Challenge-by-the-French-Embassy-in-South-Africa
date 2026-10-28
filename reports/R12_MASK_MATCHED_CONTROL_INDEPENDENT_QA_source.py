import json,datetime,numpy as np,pandas as pd
from ops.build_r12_mask_matched_control_v2 import load_control,mask_bits,sha,fixed_auc,ROOT
from pipeline.dataset_r12 import R9Sources
from pipeline.source_split_r6 import apply_minute_qc
D=ROOT/'data/synthetic/r12_mask_matched_control_v2';V=ROOT/'data/synthetic/r12_cpu_ablation_v3';a,m=load_control(D);pm=json.loads((V/'manifest.json').read_text());old=pm['arms']['L1_H0']['train']
with np.load(V/old['path'],allow_pickle=False) as z:b={k:z[k] for k in z.files}
meta=pd.read_parquet(D/'train_metadata.parquet');oldmeta=pd.read_parquet(V/old['metadata_path']);full=pd.read_parquet(V/'TRAIN_source_features.parquet');ix=meta.index[meta.special_zero].to_numpy();lookup=full.set_index('base_window_index',drop=False);rec=json.loads((D/'replacement_receipts.json').read_text());sub=np.load(D/'cnn_subset_indices.npy',allow_pickle=False);checks=[]
def ck(name,ok,**kw):
 checks.append(dict(check=name,passed=bool(ok),**kw));assert ok,name
ck('1200 requested slots/CNN600',len(rec)==1200 and sum(r['row_index'] in set(sub) for r in rec)==600)
ck('shortfalls retained',len(ix)+m['audit']['shortfall']==1200)
for k in ['site','sensor_group','recorder']:
 ck('all row source '+k,np.array_equal(lookup.loc[a['base_window_index'],k].to_numpy(),lookup.loc[b['base_window_index'],k].to_numpy()))
ck('all observation bits preserved',np.array_equal(np.isfinite(a['db']),np.isfinite(b['db'])))
for k in ['share','presence','cell_share','explicit_no_source','sample_id']:
 ck('all row targets and identity '+k,np.array_equal(a[k],b[k]))
pos=b['presence']>0
for k in b:ck('positive bytes '+k,np.ascontiguousarray(a[k][pos]).tobytes()==np.ascontiguousarray(b[k][pos]).tobytes())
ck('TRAIN only and unique sources',set(a['base_window_index'])<=set(full.base_window_index) and len(set(a['base_window_index']))==12000)
ck('replacements disjoint original12000',not(set(a['base_window_index'][ix])&set(b['base_window_index'])))
ck('sourceID exact',a['source_id'].tolist()==['SA:'+str(int(i)) for i in a['base_window_index']])
S=R9Sources();nf=lookup.loc[a['base_window_index'][ix]];raw=apply_minute_qc(np.asarray(S.background_bank()[nf.base_window_index.to_numpy(int)]),nf.boundary_mask_bits.to_numpy(np.int64));o=np.isfinite(raw).all(2);v=raw[:,:,18:25].astype(float);top=np.max(np.where(o[:,:,None],v,0),2);broad=top+10*np.log10(np.power(10,np.where(o[:,:,None],v-top[:,:,None],0)/10).sum(2));broad[~o]=np.nan;hf=np.nanmax(broad,1)-np.nanmedian(broad,1);known={r['row_index']:r for r in rec if r['replaced']};saved=np.array([known[int(i)]['new_HF_bump_db'] for i in ix]);thr=np.array([known[int(i)]['stratum_HF_q90_db'] for i in ix]);err=float(np.max(abs(hf-saved)))
ck('manual raw HF score and strict threshold',err<1e-9 and np.all(hf>thr),error=err,min_excess_db=float(np.min(hf-thr)))
ck('zero rounded db actual source parity',np.array_equal(a['db'][ix],np.round(raw.astype(float),1).astype(np.float32),equal_nan=True))
ck('all known zero q share presence exact',np.all(a['share'][~pos]==0) and np.all(a['cell_share'][~pos]==0) and np.all(a['presence'][~pos]==0))
bits=[]
for start in range(0,len(full),512):
 f=full.iloc[start:start+512];x=apply_minute_qc(np.asarray(S.background_bank()[f.base_window_index.to_numpy(int)]),f.boundary_mask_bits.to_numpy(np.int64));bits.extend(mask_bits(x).tolist())
full['effective_mask_bits']=bits;thresholds={k:float(np.quantile(g.hf_bump_2_8_db.to_numpy(float),.9,method='linear')) for k,g in full.groupby(['site','sensor_group','recorder','effective_mask_bits'])};stored=json.loads((D/'stratum_HF_thresholds.json').read_text());terr=[abs(float(thresholds[(r['site'],r['sensor_group'],r['recorder'],r['effective_mask_bits'])])-r['q90_db']) for r in stored];ck('all full-stratum TRAIN q90 reproduced',max(terr)<1e-12,strata=len(stored),error=max(terr),tolerance=1e-12)
aucs={}
for mode,ids in [('TRAIN12000',np.arange(12000)),('CNN6000',sub)]:
 n0=np.isfinite(b['db'][ids]).all(2).sum(1);n1=np.isfinite(a['db'][ids]).all(2).sum(1);v0=fixed_auc(b['presence'][ids],-n0);v1=fixed_auc(a['presence'][ids],-n1);ck(mode+' maskAUC exact0delta',v0==v1);aucs[mode]={'before':v0,'after':v1,'delta':v1-v0}
 of=lookup.loc[b['base_window_index'][ids]].copy();nf=lookup.loc[a['base_window_index'][ids]].copy();of['bits']=mask_bits(b['db'][ids]);nf['bits']=mask_bits(a['db'][ids]);of['presence']=b['presence'][ids];nf['presence']=a['presence'][ids];cols=['site','sensor_group','recorder','bits','presence'];ck(mode+' full provenance x label table exact',of.groupby(cols).size().equals(nf.groupby(cols).size()))
S.assert_unchanged();load_control(D);ck('legacy v1 builder exact restored',sha(ROOT/'ops/build_r12_mask_matched_control.py')==sha(ROOT/'data/synthetic/r12_mask_matched_control_v1/builder_before_recorder_refinement.py'))
out={'format':'r12_matched_control_independent_verification_v2','status':'passed','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'data_hash':m['data_hash'],'control_manifest_sha256':sha(D/'manifest.json'),'all_checks':checks,'actual_replacements':len(ix),'actual_CNN_replacements':int(np.isin(sub,ix).sum()),'shortfall_receipts':[r for r in rec if not r['replaced']],'all_rows_sensor_recorder_site_effective_mask_and_targets_unchanged':True,'mask_only_AUC':aucs,'original_data_model_policy_modified':False,'model_fit_inference_Test_validation_locked_or_network':False,'legacy_v1_script_exact_restored_SHA':sha(ROOT/'ops/build_r12_mask_matched_control.py'),'independent_QA_script_sha256':sha(__file__)}
p=D/'independent_verification.json';assert not p.exists();p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'passed','checks':len(checks),'manifestSHA':sha(D/'manifest.json'),'QA_SHA':sha(p),'scriptSHA':sha(ROOT/'ops/build_r12_mask_matched_control_v2.py'),'TRAIN_SHA':sha(D/'train.npz'),'data_hash':m['data_hash'],'replacements':len(ix),'CNN_replacements':int(np.isin(sub,ix).sum()),'shortfall':out['shortfall_receipts']},ensure_ascii=False))
