"""Small frozen CPU forward ONLY to recover unsaved q_hat for EDA figures.

Task1 metrics retain authoritative saved scalar predictions. No gradients,
optimizer, training, Test input, policy fitting or submission generation.
"""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['DO_NOT_TRACK']='1'
import sys,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.r17_audit_common import *

def main():
 import torch
 from pipeline.training_r16_pretrained_cuda_bf16 import branch,base_fingerprint
 from pipeline.training_r16_multisupport import features
 torch.set_num_threads(2);out=OUT/'explorer';a=load(out/'representative_windows.npz');x,w=features(a['db']);results={};receipts=[]
 cases=json.loads((out/'cases.json').read_text());saved={n:load(ROOT/f'models/r16_v1_fourway_r2/predictions/{n}.npz') for n in ['selection','domain_holdout']}
 freeze=json.loads((ROOT/'models/r16_pretrained_cuda_bf16_r1/both/final/GLOBAL_FREEZE.json').read_text())
 for family in ['dinov2_small','dasheng_base']:
  for regime in ['D0','H0']:
   path=ROOT/f'models/r16_pretrained_cuda_bf16_r1/both/{family}/{regime}/regression/best.pt';cp=torch.load(path,map_location='cpu',weights_only=True)
   assert cp['family']==family and cp['regime']==regime and cp['task']=='regression'
   m,prov=branch(family,'regression','cpu');assert base_fingerprint(m)==cp['permanent_base_fingerprint'];m.load_adapter_state(cp['adapter_state']);m.eval();qq=[]
   with torch.inference_mode():
    for off in range(0,len(x),4):qq.append(m(torch.from_numpy(x[off:off+4].copy())).float().sigmoid().numpy())
   q=np.concatenate(qq);pooled=(q*w).sum((1,2));results[family+'_'+regime]=q.astype(np.float32);differences=[]
   for i,c in enumerate(cases):
    label=c['label'];key='selection'if label=='ensemble_fit'else'domain_holdout'if label=='domain_holdout'else None
    if key:expected=saved[key]['share_'+regime][c['source_index'],MEMBERS.index(family)];differences.append(float(abs(pooled[i]-expected)))
   receipts.append(dict(family=family,regime=regime,windows=len(q),checkpoint_sha256=sha(path),frozen_base_fingerprint_verified=True,torch=torch.__version__,dtype='CPU FP32 for visualization; original CUDA BF16 scalar outputs remain authoritative',max_saved_scalar_drift=max(differences),mean_saved_scalar_drift=float(np.mean(differences)),training_steps=0,Test_used=False))
   np.savez_compressed(out/'frozen_neural_masks.npz',**results);jsave(out/'frozen_mask_receipt.json',receipts)
   print('MASKS',receipts[-1],flush=True);del m,cp;gc.collect()
 np.savez_compressed(out/'frozen_neural_masks.npz',**results);jsave(out/'frozen_mask_receipt.json',receipts)
 import xarray as xr
 ds=xr.open_dataset(out/'representative_windows.xarray.nc').load()
 for k,v in results.items():ds[k+'_q_hat']=(('case','minute','frequency_hz'),np.where(ds.O.values,v,np.nan))
 ds.to_netcdf(out/'representative_windows_with_qhat.xarray.nc',engine='scipy')
if __name__=='__main__':main()
