"""Actual raw TRAIN epoch + batched CPU frontend, no backbone or optimizer."""
from pathlib import Path
import json,sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from pipeline.full_data_r13 import R13Sources,DynamicR13Dataset
from pipeline.features_r14_torch import TorchFeatureFrontend
from pipeline.raw_frontend_r14 import RawR14Dataset

def main():
    out=ROOT/'reports/R14_TORCH_FRONTEND/FULL_EPOCH.json'
    if out.exists():raise FileExistsError('Fresh benchmark receipt required')
    torch.set_num_threads(2);bundle=ROOT/'data/synthetic/group_cv_physics_r13_variants_v2'
    sources=R13Sources(bundle=bundle,runtime_policy=json.loads((ROOT/'configs/r13_runtime_candidate_highshare_P1_v2.json').read_text()))
    dataset=RawR14Dataset(DynamicR13Dataset(sources,0));frontend=TorchFeatureFrontend(sources.basis(0))
    total_start=time.perf_counter();synthesis=features=0.;count=zeros=0;digest=hashlib.sha256()
    with torch.inference_mode():
        for start in range(0,len(dataset),128):
            begin=time.perf_counter();rows=[dataset[i] for i in range(start,min(start+128,len(dataset)))];db=np.stack([r['db'] for r in rows]);synthesis+=time.perf_counter()-begin
            begin=time.perf_counter();x,w=frontend(torch.from_numpy(db));features+=time.perf_counter()-begin
            digest.update(x.numpy().tobytes());digest.update(w.numpy().tobytes());count+=len(rows);zeros+=sum(float(r['share'])==0 for r in rows)
    elapsed=time.perf_counter()-total_start;sources.assert_unchanged();assert count==len(dataset)
    result={'status':'completed','fold':0,'epoch':1,'actual_primary_rows':count,'observed_zero_rows':zeros,
        'CPU_threads':2,'batch_size':128,'raw_synthesis_collation_seconds':synthesis,
        'batched_Torch_features_seconds':features,'total_seconds_including_stream_hash':elapsed,
        'feature_stream_sha256':digest.hexdigest(),'data_hash':json.loads((bundle/'manifest.json').read_text())['data_hash'],
        'code_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('pipeline/features_r14_torch.py','pipeline/raw_frontend_r14.py','ops/benchmark_r14_full_epoch.py')},
        'DataLoader_multiprocessing_backbone_optimizer_GPU_steps':False,
        'concurrent_MPS_diagnostic_CPU_load_present':True,'not_a_GPU_epoch_or_fulltraining_estimate':True,
        'original_sources_unchanged':True,'Test_report_newfit_or_network':False}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
