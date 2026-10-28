from pathlib import Path
import argparse,ctypes,hashlib,json,sys,time
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--rows',type=int,default=2048);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
# Deliberately mirror full CLI import closure before native tree fitting.
import pipeline.runner_r9,pipeline.dataset_r9
import numpy as np,torch,lightgbm,sklearn
from pipeline.trees_ensemble_r9 import PhysicsLGBM
from threadpoolctl import threadpool_info
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def images():
 d=ctypes.CDLL(None);d._dyld_image_count.restype=ctypes.c_uint32;d._dyld_get_image_name.argtypes=[ctypes.c_uint32];d._dyld_get_image_name.restype=ctypes.c_char_p
 return sorted({str(Path(d._dyld_get_image_name(i).decode()).resolve()) for i in range(d._dyld_image_count()) if 'libomp' in d._dyld_get_image_name(i).decode().lower()})
pre={'rows':a.rows,'features':3827,'threads':4,'package_versions':{m.__name__:m.__version__ for m in [np,torch,lightgbm,sklearn]},'libomp_images':images(),'threadpools':threadpool_info(),'source_dataset_or_Test_read':False}
(out/'pre_fit.json').write_text(json.dumps(pre,indent=2)+'\n');print(json.dumps({'before_fit':pre}),flush=True)
rng=np.random.default_rng(91);b=rng.uniform(.5,1.5,(a.rows+256,15,25));mask=np.arange(len(b))%5!=0
shape=np.exp(-np.linspace(-1,1,15)**2/.2)[None,:,None]*np.exp(-np.linspace(-1,1,25))[None,None,:]
added=mask[:,None,None]*shape*rng.uniform(.01,3.,(len(b),1,1));e=b+added
raw={'db':np.round(100+10*np.log10(e),1).astype(np.float32),'share':(added.sum((1,2))/e.sum((1,2))).astype(np.float32),'presence':mask.astype(np.float32)}
train={k:v[:a.rows] for k,v in raw.items()};inner={k:v[a.rows:] for k,v in raw.items()};binding={'fixture_only':True,'rows':a.rows,'numeric_seed':91,'threads':4}
settings={'maximum_trees':8,'early_stopping_patience':4,'threads':4,'minimum_leaf_rows':50}
t=time.perf_counter();tree=PhysicsLGBM(settings,seed=91);tree.fit(train,inner,out/'model',binding,fixture=True)
pred=tree.predict(inner['db']);reload=PhysicsLGBM.load(out/'model',expected_binding=binding,allow_fixture=True);delta={}
for batch in [1,7,256]:
 replay=reload.predict(inner['db'],batch_size=batch)
 for k in ['share','score']:
  diff=float(np.max(np.abs(replay[k]-pred[k])));assert diff<=1e-6;delta[k+'_batch'+str(batch)]=diff
assert all(np.isfinite(v).all() for v in pred.values())
result={**pre,'status':'passed','elapsed_seconds':time.perf_counter()-t,'libomp_images_after':images(),'reload_batch_replay_max_delta':delta,'model_summary_sha256':sha(out/'model/summary.json'),'fixture_code_sha256':sha(__file__),'production_fit':False}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','seconds':result['elapsed_seconds'],'libomp_images':result['libomp_images_after']}),flush=True)
