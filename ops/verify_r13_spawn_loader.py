"""Actual R13 TRAIN item parity through four spawned prefetched workers.

This measures only a 64-item transport smoke test, not parallel epoch speedup.
"""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from pipeline.full_data_r13 import R13Sources,DynamicR13Dataset,sha
from pipeline.training_r13 import _seed_worker


def main():
    out=ROOT/'reports/R13_SPAWN_LOADER_CONNECTION.json'
    if out.exists():raise FileExistsError('Fresh transport verification required')
    torch.set_num_threads(1);s=R13Sources(bundle=ROOT/'data/synthetic/group_cv_physics_r13_variants_v2')
    ds=DynamicR13Dataset(s,0);indices=list(range(64));sub=Subset(ds,indices)
    def read(workers):
        gen=torch.Generator().manual_seed(2026102713);begin=time.perf_counter()
        loader=DataLoader(sub,batch_size=16,shuffle=False,num_workers=workers,generator=gen,
            worker_init_fn=_seed_worker,persistent_workers=False,
            **({'multiprocessing_context':'spawn','prefetch_factor':2} if workers else {}))
        outrows=[]
        for batch in loader:outrows.append(batch)
        return {k:torch.cat([b[k] for b in outrows]).numpy() for k in ('x','w','share','cell_share','presence')},time.perf_counter()-begin
    baseline,t0=read(0);parallel,t4=read(4)
    assert all(np.array_equal(baseline[k],parallel[k]) for k in baseline)
    assert all(np.isfinite(a).all() for a in parallel.values());s.assert_unchanged()
    value={'format':'r13_real_TRAIN_spawn_prefetch_smoke_v1','status':'passed','model_fit':False,
        'actual_TRAIN_rows':64,'fold':0,'epoch':1,'workers':4,'prefetch_factor':2,'worker_threads':1,
        'zero_vs_four_workers_bitwise_all_numeric_keys':True,'parent_loader_seconds':t0,'spawn_loader_seconds':t4,
        'not_a_parallel_full_epoch_benchmark':True,'cannot_extrapolate_linear_speedup':True,
        'code_sha256':{n:sha(ROOT/n) for n in ('pipeline/full_data_r13.py','pipeline/training_r13.py','ops/verify_r13_spawn_loader.py')},
        'source_unchanged':True,'Test_report_arrays_GPU_SSH':False}
    out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'status':'passed','seconds':t4,'receipt':str(out)}))

if __name__=='__main__':main()
