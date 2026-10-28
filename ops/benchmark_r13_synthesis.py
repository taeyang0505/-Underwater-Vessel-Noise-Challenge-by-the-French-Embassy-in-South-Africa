"""One actual TRAIN epoch of synthesis+11ch conversion, no optimizer/model."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,platform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pipeline.full_data_r13 import R13Sources,DynamicR13Dataset,validate_full_bundle,DATA_CODE,sha

def main():
    out=ROOT/'reports/R13_SYNTHESIS_FULL_EPOCH_BENCHMARK.json'
    if out.exists():raise FileExistsError('Fresh benchmark receipt required')
    start=time.perf_counter();source=R13Sources(bundle=ROOT/'data/synthetic/group_cv_physics_r13_variants_v2');dataset=DynamicR13Dataset(source,0,epoch=1)
    manifest=validate_full_bundle(source.bundle,verify_all=False);initial={n:sha(ROOT/n) for n in DATA_CODE}
    startup=time.perf_counter()-start;begin=time.perf_counter();fingerprint=hashlib.sha256();zero=0;gap=0
    for i in range(len(dataset)):
        row=dataset[i];zero+=int(row['share']==0)
        gap=max(gap,abs(float((row['cell_share']*row['w']).sum())-float(row['share'])))
        for k in ('x','w','share','cell_share'):
            a=np.ascontiguousarray(row[k]);fingerprint.update(a.tobytes())
        if (i+1)%10000==0:print(json.dumps({'TRAIN_rows':i+1,'elapsed_seconds':time.perf_counter()-begin}),flush=True)
    elapsed=time.perf_counter()-begin
    if zero!=dataset.epoch_audit()['zero_count'] or gap>.0058:raise AssertionError('Actual epoch coverage/labels failed')
    source.original.assert_unchanged()
    for n,h in initial.items():
        if sha(ROOT/n)!=h:raise ValueError('Source changed during benchmark:'+n)
    value={'format':'r13_actual_TRAIN_full_epoch_synthesis_benchmark_v1','status':'completed',
        'rows':len(dataset),'fold':0,'epoch':1,'threads':1,'startup_seconds':startup,
        'synthesis_and11channel_seconds':elapsed,'rows_per_second':len(dataset)/elapsed,
        'max_raw_label_rounded_pool_gap':gap,'observed_zero_count':zero,
        'schedule_audit':{k:v for k,v in dataset.epoch_audit().items() if k!='quota_rows'},
        'actual_stream_sha256':fingerprint.hexdigest(),'data_hash':manifest['data_hash'],'code_sha256':initial,
        'platform':platform.platform(),'CPU_count':os.cpu_count(),
        'includes':['native reference synthesis','optional observed-mean highshare solve','11channel TRAIN-PCA projection','numeric hash'],
        'does_not_measure':['GPUmodel100epoch training','CUDA throughput/VRAM','4worker scaling'],
        'new_model_or_optimizer_or_validation_or_Test':False,'source_unchanged':True}
    out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'completed','seconds':elapsed,'receipt':str(out)}))
if __name__=='__main__':main()
