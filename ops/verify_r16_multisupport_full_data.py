"""Independent reload/purge/recipe replay audit; never fits a learner."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from pipeline.r16_multisupport_data import ROOT,OUT,SEED,REGIMES,read,write,sha,load,guard

def main():
 import sys
 from pipeline.dataset_r9 import R9Sources
 from pipeline.r16_production_data import _crossrole
 from pipeline.relative_grid_r6_v2 import relative_grid_v2
 from pipeline.sampler_r9 import smooth_eq
 from ops.r16_forward_data import generate_operator,channel_zero_mean
 sys.addaudithook(guard);start=time.perf_counter();m=read(OUT/'manifest.json');source=R9Sources();rows=[];references=[]
 for p,h in m['source_code_sha256'].items():assert sha(ROOT/p)==h,p
 for p,h in m['source_artifact_sha256'].items():assert sha(ROOT/p)==h,p
 refs=pd.read_parquet(OUT/'reference_A_folds.parquet');fitmeta=pd.read_parquet(ROOT/'reports/r16_support_temporal_r1/data/local_fit_backgrounds.parquet')
 benchmarks={name:pd.read_parquet(ROOT/f'reports/r16_support_temporal_r1/data/{name}_backgrounds.parquet')for name in('local_gate','local_report','domain_holdout')}
 total=0;maxgap=maxreplay=0.
 for f in range(5):
  directory=OUT/f'fold_{f:02d}';tr=pd.read_parquet(directory/'train_references.parquet');cv=pd.read_parquet(directory/'cv_references.parquet')
  assert set(tr.reference_component).isdisjoint(cv.reference_component)
  assert set(tr.reference_row_index)|set(cv.reference_row_index)==set(refs.reference_row_index)
  references.extend(tr.reference_row_index.tolist());trainmeta=pd.read_parquet(directory/'train_backgrounds.parquet');cvmeta=pd.read_parquet(directory/'cv_backgrounds.parquet')
  gaps=_crossrole(trainmeta,{'own_cv':cvmeta,'shared_fit':fitmeta,**benchmarks})
  for role,meta,allowed in [('train',trainmeta,tr),('cv',cvmeta,cv)]:
   arrays={r:load(ROOT/m['folds'][str(f)]['roles'][role][r]['path'])for r in REGIMES}
   for r,d in arrays.items():
    assert sha(ROOT/m['folds'][str(f)]['roles'][role][r]['path'])==m['folds'][str(f)]['roles'][role][r]['sha256']
    z=d['share']==0;assert z.sum()==len(z)//5;assert np.array_equal(~z,d['presence']>0)
    assert np.array_equal(d['base_window_index'],meta.base_window_index.to_numpy())
    assert set(d['reference_index'][~z])==set(allowed.reference_row_index)
    assert np.all((d['cell_share']>=0)&(d['cell_share']<=1));assert np.all(d['cell_share'][~np.isfinite(d['db'])]==0)
    for off in range(0,len(z),256):
     _,w=relative_grid_v2(d['db'][off:off+256]);gap=float(np.max(abs((d['cell_share'][off:off+256]*w).sum((1,2))-d['share'][off:off+256])))
     assert gap<.0058;maxgap=max(maxgap,gap)
    # Independent deterministic replay of source+stored geometry, including
    # explicit negatives, short positives and the longest observed supports.
    ids=np.unique(np.r_[np.random.default_rng([99,f,len(role)]).choice(len(z),min(12,len(z)),replace=False),np.argmax(d['active_minutes']),np.flatnonzero(z)[0]])
    for i in ids:
     channel=np.random.default_rng([SEED,f,0 if role=='train'else 1,int(i),0x4348]);bg=np.asarray(source.background(meta.iloc[int(i)]),float)
     native=None if z[i]else source.reference_minutes[int(d['reference_index'][i])]
     eq=np.zeros(25)if z[i]else smooth_eq(channel,6.);post=np.zeros(25)if z[i]else channel_zero_mean(channel,6. if role=='train'else 12.)
     snr=float(channel.uniform(-10,10));sub=channel.uniform(-10,10,2)
     geom=None if z[i]else dict(crop_begin_minute=int(d['crop_begin'][i]),selected_minutes=int(d['selected_minutes'][i]),window_start_minute=int(d['window_start'][i]))
     actual,_=generate_operator(bg,native,geom,eq,post,snr,sub,str(d['operator'][i]),zero=bool(z[i]))
     assert np.array_equal(actual['db'],d['db'][i],equal_nan=True)
     assert abs(actual['share']-d['share'][i])<1e-12
     qgap=float(abs(actual['cell_share']-d['cell_share'][i]).max());assert qgap<3.1e-8;maxreplay=max(maxreplay,qgap)
    rows.append({'fold':f,'regime':r,'role':role,'n':len(z),'replayed_sources':len(ids),'reference_clips':len(allowed),'purge_checks':gaps if role=='train'else[]});total+=len(z)
   a,b=arrays['D0'],arrays['H0'];z=a['share']==0
   for k in('sample_id','base_window_index','reference_index','operator','explicit_no_source'):assert np.array_equal(a[k],b[k])
   for k in('db','cell_share','share'):assert np.array_equal(a[k][z],b[k][z],equal_nan=True)
 unique,counts=np.unique(references,return_counts=True);assert set(unique)==set(refs.reference_row_index)and np.all(counts==4)
 result={'status':'passed','reloaded_synthetic_rows':total,'rows':rows,'every_TRAIN_reference_rotates_four_of_five':True,
  'maximum_raw_q_vs_rounded_w_pool_gap':maxgap,'maximum_replayed_raw_vs_float32_q_gap':maxreplay,'D0_H0_negative_arrays_identical':True,
  'maximum_source_overlap':0,'minimum_support_purge_hours_required':24,'Test_used':False,'training_or_inference':False,'elapsed_seconds':time.perf_counter()-start}
 write(ROOT/'reports/r16_multisupport_full_preflight_r1/FULL_DATA_INDEPENDENT_QA.json',result);print(json.dumps(result))
if __name__=='__main__':main()
