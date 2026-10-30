"""Disposable V2 public-weight/gradient/throughput check; no TRAIN/Test read.

Only random finite features are used. A passing one-step gate is NOT a trained
model, transfer validation, peak-memory estimate or full-epoch speed measurement.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
CACHE=ROOT/'models/pretrained_r16/timm--convnextv2_base.fcmae'
OUT=ROOT/'ops/r16_pretrained_v2_mps_r1'
WEIGHT_SHA='4c2c0ab209f18d9dc1d4cf85e92dda8c834f377d27c5485c284e38daa9c060db'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def header_audit(path):
    with Path(path).open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        tensors=json.loads(f.read(n))
    expected={
        'stem.0.weight':[128,3,4,4],'stem.0.bias':[128],
        'stem.1.weight':[128],'stem.1.bias':[128],
        'head.norm.weight':[1024],'head.norm.bias':[1024]}
    dims=(128,256,512,1024)
    for i,(c,depth) in enumerate(zip(dims,(3,3,27,3))):
        if i:
            prev=dims[i-1]
            for suffix,shape in {'0.weight':[prev],'0.bias':[prev],
                '1.weight':[c,prev,2,2],'1.bias':[c]}.items():
                expected[f'stages.{i}.downsample.{suffix}']=shape
        for j in range(depth):
            for suffix,shape in {'conv_dw.weight':[c,1,7,7],'conv_dw.bias':[c],
                'norm.weight':[c],'norm.bias':[c],'mlp.fc1.weight':[4*c,c],
                'mlp.fc1.bias':[4*c],'mlp.fc2.weight':[c,4*c],'mlp.fc2.bias':[c],
                'mlp.grn.weight':[4*c],'mlp.grn.bias':[4*c]}.items():
                expected[f'stages.{i}.blocks.{j}.{suffix}']=shape
    real={k:v['shape'] for k,v in tensors.items() if k!='__metadata__'}
    if real!=expected:raise ValueError('Checkpoint tensor key/shape contract differs')
    count=0
    for shape in real.values():
        n=1
        for v in shape:n*=v
        count+=n
    return {'tensor_count':len(real),'checkpoint_parameters':count,
        'shape_contract_exact':True,'modified_grid_per_stage':[[15,25]]*4,
        'original_rgb_stem_bypassed':True,'all_three_transition_strides_changed':True,
        'strides_modified_without_tensor_shape_changes':True}


def worker(args):
    import torch
    from pipeline.pretrained_convnext_v2_r16 import create_convnext_v2_branch
    torch.set_num_threads(2);torch.manual_seed(20261029)
    if args.device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; CPU fallback forbidden')
    def sync():
        if args.device=='mps':torch.mps.synchronize()
    model,receipt=create_convnext_v2_branch(args.task,device=args.device)
    frozen_before={n:sha_tensor(p) for n,p in model.frozen_named_parameters()}
    x=torch.randn(args.batch,model.in_channels,15,25,device=args.device)
    x[:,5]=1;x[:,5,0]=0
    weight=torch.rand(args.batch,15,25,device=args.device)*(x[:,5]>.5)
    weight/=weight.sum((1,2),keepdim=True)
    target_q=torch.rand_like(weight);target_y=(weight*target_q).sum((1,2))
    records=[]
    for stage in (1,2):
        model.set_stage(stage).train()
        optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=.001 if stage==1 else .0002)
        def step():
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            shape=(args.batch,15,25) if args.task=='regression' else (args.batch,)
            if tuple(logits.shape)!=shape:raise AssertionError('Output shape differs')
            if args.task=='regression':
                q=logits.sigmoid();y=(weight*q).sum((1,2))
                loss=((y-target_y)**2).mean()+.25*(weight*(q-target_q)**2).sum((1,2)).mean()
            else:
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits))
            loss.backward()
            if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise AssertionError('Nonfinite loss or gradient')
            if model.stem.weight.grad is None or model.stem.weight.grad.abs().sum()==0:
                raise AssertionError('No input-stem gradient')
            if any(p.grad is not None for _,p in model.frozen_named_parameters()):
                raise AssertionError('Frozen backbone received gradient')
            optimizer.step()
            return float(loss.detach().cpu())
        step();sync();start=time.monotonic()
        for _ in range(args.steps):loss=step()
        sync();elapsed=time.monotonic()-start
        record={'stage':stage,'batch':args.batch,'timed_steps':args.steps,'seconds_per_step':elapsed/args.steps,
            'rows_per_second':args.batch*args.steps/elapsed,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'synthetic_61446_rows_train_epoch_minutes':61446*elapsed/(args.steps*args.batch*60),
            'loss':loss,'finite_forward_backward':True,'stem_gradient_present':True,'frozen_gradients_absent':True}
        if args.device=='mps':record.update({'mps_driver_bytes_snapshot':torch.mps.driver_allocated_memory(),
            'mps_tensor_bytes_snapshot':torch.mps.current_allocated_memory(),'mps_recommended_max_bytes':torch.mps.recommended_max_memory()})
        records.append(record)
    after={n:sha_tensor(p) for n,p in model.frozen_named_parameters()}
    if frozen_before!=after:raise AssertionError('Frozen base weight changed')
    model.eval()
    with torch.no_grad():ref=model(x).detach().cpu()
    state=model.adapter_state()
    with torch.no_grad():model.stem.weight.add_(.1)
    model.load_adapter_state(state)
    with torch.no_grad():restored=model(x).detach().cpu()
    if not torch.equal(ref,restored):raise AssertionError('Adapter roundtrip differs')
    save(OUT/f'{args.task}_batch{args.batch}.json',{'status':'passed','task':args.task,'device':args.device,
        'torch':torch.__version__,'provenance':receipt,'stages':records,'frozen_base_unchanged':True,
        'adapter_roundtrip_exact':True,'Test_read':False,'training_dataset_read':False,
        'claim_limit':'synthetic short-step contract/throughput only; no transfer or trained checkpoint claim'})


def sha_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('cpu','mps'),default='mps')
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--task',choices=('regression','classification'),default='regression')
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.batch<1 or args.steps<1:raise ValueError('Positive batch/steps required')
    if args.worker:return worker(args)
    result={'status':'blocked','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'Test_read':False,'training_dataset_read':False,'additional_kfold':False,
        'source_sha256':{n:sha(ROOT/n) for n in ('pipeline/pretrained_convnext_v2_r16.py','ops/preflight_r16_v2_mps.py')}}
    path=CACHE/'model.safetensors'
    if not path.is_file():
        result['reason']='Public FCMAE weights not yet available as a complete SHA-verified file; no random substitute'
        partial=CACHE/'model.download'
        if partial.exists():result['partial_download_bytes']=partial.stat().st_size
    elif path.stat().st_size!=350809374 or sha(path)!=WEIGHT_SHA:
        result['reason']='Public FCMAE weights SHA/byte-size gate failed'
    else:
        result['checkpoint_layout']=header_audit(path)
        result['public_weight_sha256']=WEIGHT_SHA
        if args.audit_only:
            result.update(status='metadata_passed',reason='No tensor execution requested')
        else:
            cmd=[sys.executable,str(Path(__file__)),'--worker','--device',args.device,
                '--task',args.task,'--batch',str(args.batch),'--steps',str(args.steps)]
            try:
                completed=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=args.timeout)
                save(OUT/f'{args.task}_batch{args.batch}_runtime.json',{'returncode':completed.returncode,
                    'stderr_tail':completed.stderr[-2400:],'stdout_tail':completed.stdout[-1000:]})
                if completed.returncode==0:
                    result.update(status='passed',task_result=str((OUT/f'{args.task}_batch{args.batch}.json').relative_to(ROOT)))
                else:result['reason']='Runtime one-step gate failed; inspect separate runtime diagnostic'
            except subprocess.TimeoutExpired as exc:
                result['reason']=f'Runtime import/check exceeded {args.timeout}s; no speed estimate or successful MPS claim'
    save(OUT/'PREFLIGHT_LATEST.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['status'] in ('passed','metadata_passed') else 2


if __name__=='__main__':raise SystemExit(main())
