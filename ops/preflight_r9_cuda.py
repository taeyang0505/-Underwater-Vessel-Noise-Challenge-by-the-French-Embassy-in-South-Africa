#!/usr/bin/env python3
"""Disposable full-size CUDA connection probe, never production checkpoints."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pipeline.dataset_r9 import R9Sources, DynamicR9Dataset, sha
from pipeline.models_r9 import make_r9_model
from pipeline.training_r9 import physical_multitask_loss
from pipeline.runner_r9 import CODE_NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/r9_full_training_v1.json')
    p.add_argument('--output',default='reports/R9_CUDA_FULL_MODEL_PREFLIGHT.json')
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError('Fresh CUDA probe output required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA not available')
    torch.set_num_threads(4);torch.manual_seed(20261026)
    root=Path(__file__).resolve().parents[1]
    pins={n:sha(root/n) for n in CODE_NAMES}
    pins['ops/preflight_r9_cuda.py']=sha(__file__)
    start=time.monotonic();sources=R9Sources(a.config)
    ds=DynamicR9Dataset(a.config,0,sources=sources)
    # Same production shape/workers; only 128 predeclared TRAIN indices.
    loader=DataLoader(Subset(ds,range(128)),batch_size=128,num_workers=4,
                      multiprocessing_context='spawn',shuffle=False,pin_memory=True)
    batch=next(iter(loader));device=torch.device('cuda')
    x=batch['x'].to(device,dtype=torch.float32);w=batch['w'].to(device,dtype=torch.float32)
    labels=tuple(batch[k].to(device,dtype=torch.float32) for k in ['share','presence','cell_share'])
    weights=batch['loss_weight'].to(device,dtype=torch.float32)
    enabled=torch.cuda.is_bf16_supported();cases=[]
    try:
        for kind in ('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer'):
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            t=time.monotonic();model=make_r9_model(kind,dropout=.1).to(device)
            opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
            model.train();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled):out=model(x,w)
            per,terms=physical_multitask_loss(out,labels,w)
            loss=(per*weights).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite CUDA loss')
            loss.backward()
            gradients=[v.grad for v in model.parameters() if v.requires_grad]
            if any(v is None or not torch.isfinite(v).all() for v in gradients):raise FloatingPointError('Nonfinite CUDA gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();torch.cuda.synchronize()
            if any(not torch.isfinite(v).all() for v in model.parameters()):raise FloatingPointError('Nonfinite CUDA parameter')
            model.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled):pred=model(x,w)
            physical_multitask_loss(pred,labels,w)
            pool=float((pred[0].float()-(pred[2].float()*w).sum((1,2))).abs().max())
            if pool>3e-6 or pred[0].dtype!=torch.float32 or pred[2].dtype!=torch.float32:
                raise AssertionError('FP32 physical pool contract failed')
            cases.append({'kind':kind,'architecture':model.architecture(),
                          'batch_size':128,'Adam_steps':1,'BF16_enabled':enabled,
                          'FP32_pool':True,'pool_max_abs_error':pool,
                          'raw_label_vs_rounded_pool_gap':float(terms['raw_label_vs_rounded_pool_gap'].max()),
                          'all_gradients_parameters_finite':True,
                          'maximum_allocated_bytes':torch.cuda.max_memory_allocated(),
                          'maximum_reserved_bytes':torch.cuda.max_memory_reserved(),
                          'elapsed_seconds':time.monotonic()-t})
            del model,opt,out,pred,loss,per,terms,gradients
        sources.assert_unchanged()
        if any(sha(root/n)!=h for n,h in pins.items()):raise ValueError('Frozen source changed')
        status='passed';error=None
    except Exception as e:
        status='failed';error={'type':type(e).__name__,'message':str(e)}
    output.parent.mkdir(parents=True,exist_ok=True)
    report={'format':'r9_full_size_CUDA_connection_preflight_v1','status':status,
            'ready_for_production':status=='passed' and len(cases)==4,
            'disposable_connection_fixture_only':True,'production_checkpoints_created':False,
            'source_dataset_identity':ds.identity,'code_sha256':pins,
            'TRAIN_examples':128,'workers':4,'selection_report_Test_or_WAV_accessed':False,
            'environment':{'python':platform.python_version(),'torch':torch.__version__,
                           'numpy':np.__version__,'CUDA':torch.version.cuda,
                           'GPU':torch.cuda.get_device_name(0),'BF16_supported':enabled},
            'cases':cases,'error':error,'elapsed_seconds':time.monotonic()-start}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':status,'output':str(output),'sha256':sha(output),
                      'cases':len(cases),'elapsed_seconds':report['elapsed_seconds']}),flush=True)
    if status!='passed':raise SystemExit(1)


if __name__=='__main__':main()
