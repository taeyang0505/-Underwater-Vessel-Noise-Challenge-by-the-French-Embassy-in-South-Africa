"""CPU-only frozen old-B1/B2 predictions plus classifier-input ablations.

LightGBM and Torch deliberately run in different processes. No optimizer,
training, backbone mutation, Test access or new feature fitting is allowed.
"""
from __future__ import annotations
import argparse,datetime as dt,gc,hashlib,json,os,subprocess,sys,time
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false')
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/r21_clean_val_zero_training_r1/inference'
FAMILIES=('lightgbm','dinov2_small','dasheng_base')
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  name=Path(os.fsdecode(args[0])).name.lower()
  if name in ('test.csv','query.npz','query_predictions.npz','sample_submission.csv') or 'submission' in name:raise PermissionError('No Test/submission reads in R21 inference')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def save(p,d):
 with Path(p).open('wb') as f:np.savez_compressed(f,**d)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def prepare(path):
 OUT.mkdir(parents=True,exist_ok=True);d=load(path);n=len(d['db'])
 if d['db'].shape!=(n,15,25) or np.isinf(d['db']).any() or len(set(d['sample_id'].astype(str)))!=n:raise ValueError('Malformed input/IDs')
 receipt={'input':str(path.relative_to(ROOT)),'input_sha256':sha(path),'rows':n,'device':'cpu','optimizer_steps':0,'model_selection':False,'Test_read':False,
  'old_B1_weights':[.6,.25,.15],'old_B2_weights':[.5,.3,.2],'D0_H0_weights':[.5,.5],
  'NN_classifier_ablation':'Stem forward-pre-hook hides input feature channel5 on a clone AFTER forward has read real O for validity/head pooling. Raw observation mask, attention-pooling validity, derived acoustics, regression, all weights unchanged. Diagnostic OOD intervention, not retraining.',
  'script_sha256':sha(__file__)}
 p=OUT/'PROTOCOL.json'
 if p.exists():
  r=json.loads(p.read_text())
  if r['input_sha256']!=receipt['input_sha256'] or r['script_sha256']!=receipt['script_sha256']:raise ValueError('Inference protocol changed')
 else:write(p,receipt)
 return d
def neural(path):
 import torch
 from pipeline import training_r16_pretrained_cuda_bf16 as api
 from pipeline.training_r18_production_neural import feature17
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 d=prepare(path);x,w=feature17(d['db']);x=torch.from_numpy(x.copy());w=torch.from_numpy(w.copy())
 results=[];start=time.perf_counter()
 for family in FAMILIES[1:]:
  parts={'sample_id':d['sample_id']};pins={}
  for regime in ('D0','H0'):
   for task in ('regression','classification'):
    folder=ROOT/f'models/r16_pretrained_cuda_bf16_r1/both/{family}/{regime}/{task}';cp=folder/'best.pt';summary=json.loads((folder/'summary.json').read_text())
    if sha(cp)!=summary['artifact_sha256']['best.pt']:raise ValueError('Checkpoint SHA differs')
    pins[str(cp.relative_to(ROOT))]=sha(cp)
    ck=torch.load(cp,map_location='cpu',weights_only=True);model,_=api.branch(family,task,'cpu')
    if api.base_fingerprint(model)!=ck['permanent_base_fingerprint']:raise ValueError('Backbone differs')
    model.load_adapter_state(ck['adapter_state']);model.eval()
    if any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('Non-CPU model')
    xx=x[:,api.CLASSIFIER_CHANNELS].contiguous() if task=='classification' else x
    predictions=[];masked=[]
    with torch.inference_mode():
     for off in range(0,len(xx),32):
      batch=xx[off:off+32];p=model(batch).float().sigmoid()
      if task=='regression':p=(p*w[off:off+32]).sum((1,2))
      predictions.append(p.numpy())
      if task=='classification':
       # The original O channel is ALSO the head's validity contract. Blinding
       # x itself would declare every window unobserved. Hide only the stem's
       # feature copy while forward/head retain the real observation mask.
       def hide_feature(_module,inputs):
        blinded=inputs[0].clone();blinded[:,5]=0.
        return (blinded,)+inputs[1:]
       hook=model.stem.register_forward_pre_hook(hide_feature)
       try:masked.append(model(batch).float().sigmoid().numpy())
       finally:hook.remove()
    key=('share_' if task=='regression' else 'prob_')+regime;parts[key]=np.concatenate(predictions)
    if masked:parts['prob_O_hidden_'+regime]=np.concatenate(masked)
    if api.base_fingerprint(model)!=ck['permanent_base_fingerprint']:raise ValueError('Backbone changed')
    del model,ck;gc.collect()
    print(json.dumps({'family':family,'regime':regime,'task':task,'elapsed_sec':time.perf_counter()-start}),flush=True)
  save(OUT/(family+'.npz'),parts);write(OUT/(family+'.json'),{'prediction_sha256':sha(OUT/(family+'.npz')),'models_sha256':pins,'optimizer_steps':0,'precision':'CPU FP32'})
  results.append(family)
 write(OUT/'NEURAL_COMPLETED.json',{'status':'completed','seconds':time.perf_counter()-start,'outputs':results})
def trees(path):
 if 'torch' in sys.modules:raise RuntimeError('Torch/LightGBM must run in separate processes')
 import lightgbm as lgb
 from pipeline.training_r16_multisupport import tree_features
 from ops.r21_mask_proxy_audit import neutralize_old_classifier_features
 d=prepare(path);parts={'sample_id':d['sample_id']};pins={};start=time.perf_counter()
 expected=json.loads((ROOT/'reports/r18_production_r1/finalization/PLAN.json').read_text())['legacy_source_sha256']
 for task in ('regression','classification'):
  xx=tree_features(d['db'],task)
  for regime in ('D0','H0'):
   orig=[];mod=[]
   for fold in range(5):
    cp=ROOT/f'models/r16_multisupport_full_r1/trees/{regime}/fold_{fold:02d}/{task}.txt';name=str(cp.relative_to(ROOT))
    if sha(cp)!=expected[name]:raise ValueError('Tree model SHA differs')
    pins[name]=sha(cp);model=lgb.Booster(model_file=str(cp))
    orig.append(np.clip(model.predict(xx,num_threads=2),0,1))
    if task=='classification':
     neutral=neutralize_old_classifier_features(xx,fold=fold,regime=regime)
     mod.append(np.clip(model.predict(neutral,num_threads=2),0,1))
   key=('share_' if task=='regression' else 'prob_')+regime;parts[key]=np.mean(orig,axis=0)
   if mod:parts['prob_proxy_neutral_'+regime]=np.mean(mod,axis=0)
 save(OUT/'lightgbm.npz',parts);write(OUT/'lightgbm.json',{'prediction_sha256':sha(OUT/'lightgbm.npz'),'models_sha256':pins,'optimizer_steps':0,'seconds':time.perf_counter()-start})
def main():
 sys.addaudithook(guard)
 a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--phase',choices=['prepare','neural','trees','all'],default='all');args=a.parse_args();path=(ROOT/args.input).resolve()
 if not path.is_relative_to(ROOT):raise ValueError('Project input only')
 if args.phase=='prepare':prepare(path)
 elif args.phase=='neural':neural(path)
 elif args.phase=='trees':trees(path)
 else:
  prepare(path)
  for phase in ('neural','trees'):subprocess.run([sys.executable,str(Path(__file__).resolve()),'--input',str(path),'--phase',phase],cwd=ROOT,check=True)
  write(OUT/'COMPLETED.json',{'status':'completed','input_sha256':sha(path),'n':len(load(path)['db']),'new_training':False,'optimizer_steps':0,'device':'cpu','Test_read':False})
if __name__=='__main__':main()
