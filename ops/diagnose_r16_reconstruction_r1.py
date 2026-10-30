"""Read-only frozen H0 TimeMoE numerical reconstruction audit; no labels/Test."""
from pathlib import Path
import json, sys, time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.r16_multisupport_data import read, write, load, role
from pipeline.training_r16_multisupport import features, cache_read, cache_role
from pipeline.training_r16_pretrained_cuda_bf16 import branch, base_fingerprint, predict, OPS
from pipeline.r16_cuda_precision import require_cuda_bf16, autocast_context
import ops.finalize_r16_pretrained_cuda_bf16 as original

def main():
    started=time.perf_counter(); torch.set_num_threads(2)
    original.configure_variant('both')
    d=role(0,'D0','selection');x,w=features(d['db'])
    cached=cache_read(cache_role('D0','selection'))
    before=dict(tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
      tf32_cudnn=torch.backends.cudnn.allow_tf32,benchmark=torch.backends.cudnn.benchmark)
    result={'original_sources_unchanged':True,'no_training':True,'Test_or_report_used':False,
      'feature_max_delta':float(abs(x-cached['x']).max()),
      'weight_max_delta':float(abs(w-cached['w']).max()),'initial_backend_flags':before,'cases':[]}
    folder=original.RUN/'timemoe/H0/regression'
    cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    original.check_checkpoint(cp,read(folder/'summary.json'),'timemoe','H0','regression',read(OPS/'RUN_PLAN.json'))
    model,prov=branch('timemoe','regression','cuda')
    assert base_fingerprint(model)==cp['permanent_base_fingerprint']
    model.load_adapter_state(cp['adapter_state']);model.eval()
    saved=load(folder/'selection.npz')['prediction'].astype(float)
    final=load(original.RUN/'final/selection_predictions.npz')['share_H0'][:,4].astype(float)
    assert np.array_equal(saved,final)
    values={}
    for settings,strict,batch,mode in [('default32_no_grad',False,32,'no_grad'),
      ('strict64_no_grad',True,64,'no_grad'),('strict32_no_grad',True,32,'no_grad'),
      ('strict32_inference',True,32,'inference')]:
        if strict:require_cuda_bf16()
        outs=[]
        context=torch.inference_mode if mode=='inference' else torch.no_grad
        with context():
            for off in range(0,len(x),batch):
                xx=torch.as_tensor(x[off:off+batch].copy(),device='cuda')
                ww=torch.as_tensor(w[off:off+batch].copy(),device='cuda')
                with autocast_context('cuda'):logits=model(xx)
                outs.append((logits.float().sigmoid()*ww).sum((1,2)).cpu().numpy())
        p=np.concatenate(outs).astype(float);delta=abs(p-saved);values[settings]=p
        result['cases'].append({'case':settings,'max_delta':float(delta.max()),
          'p99_delta':float(np.quantile(delta,.99)),'mean_delta':float(delta.mean()),
          'count_gt_002':int((delta>.002).sum()),'max_row_index':int(delta.argmax())})
        print(json.dumps(result['cases'][-1]),flush=True)
    result['pairwise_max_delta']={a+'__'+b:float(abs(values[a]-values[b]).max())
      for a in values for b in values if a<b}
    result['elapsed_seconds']=time.perf_counter()-started
    write(ROOT/'ops/r16_numerical_reconstruction_r1/H0_TIMEMOE.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
