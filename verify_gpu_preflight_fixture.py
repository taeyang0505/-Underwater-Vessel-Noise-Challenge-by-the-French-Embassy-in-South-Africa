"""CPU-only schema/guard fixture; no real source arrays or checkpoints opened."""
from pathlib import Path
import hashlib,json,tempfile
from unittest import mock
import numpy as np
import torch
from torch import nn
from pipeline import gpu_preflight as probe
ROOT=Path(__file__).resolve().parent
class TinyTab(nn.Module):
    def __init__(self):super().__init__();self.fc=nn.Linear(1254,2)
    def forward(self,x):return self.fc(x)[:,None,:]

def main():
    torch.set_num_threads(1);checks={}
    with tempfile.TemporaryDirectory(prefix='uvn-gpu-schema-fixture-') as tmp:
        tmp=Path(tmp);output=tmp/'no_cuda.json'
        a=probe.build_parser().parse_args(['--data-dir',str(tmp/'nonexistent_data'),'--output',str(output)])
        with mock.patch('torch.cuda.is_available',return_value=False),mock.patch('pipeline.cv_foundations.verify_cv_package',side_effect=AssertionError('No-CUDA guard opened real inputs')):
            assert probe.run(a)==2
        no_cuda=json.loads(output.read_text());assert no_cuda['CUDA_measured'] is False and no_cuda['data_package_verified'] is False and no_cuda['models']==[]
        assert no_cuda['requested_settings']['batch']==128 and no_cuda['requested_settings']['warmup']==3 and no_cuda['requested_settings']['measured_steps']==20
        checks['no_CUDA_nonzero_and_no_data_open']='passed'
        def data(n,seed):
            rng=np.random.default_rng(seed);db=rng.normal(90,3,(n,15,25)).astype('float32');db[::2,-1]=np.nan
            y=np.resize(np.array([0.,.25,.75,.9],dtype='float32'),n)
            return {'db':db,'share':y,'presence':(y>0).astype('float32'),'cell_share':np.where(np.isfinite(db),y[:,None,None],0).astype('float32')}
        train=data(8,1);inner=data(4,2);original={k:v.copy() for k,v in train.items()}
        a.batch=4;a.warmup=1;a.steps=2;a.threads=1
        with mock.patch('pipeline.gpu_preflight.make_model',side_effect=lambda *x,**kw:TinyTab()):
            result=probe.benchmark_neural('tabm',train,inner,a,torch.device('cpu'),development_fixture=True)
        assert result['CUDA_measured'] is False and result['peak_allocated_bytes'] is None and result['peak_reserved_bytes'] is None
        assert result['development_fixture_estimate_not_for_GPU_rental'] is True and result['model_saved'] is False
        assert result['examples_per_second']>0 and result['estimated_100epoch_per_fold_allowance_seconds']>0
        assert result['estimated_epoch_train_plus_inner_prediction_allowance_seconds']>0 and result['estimate_is_guaranteed_upper_bound'] is False
        assert all(np.array_equal(v,original[k],equal_nan=True) for k,v in train.items())
        assert not list(tmp.rglob('*.pt')) and not list(tmp.rglob('*.npz'))
        checks['tiny_CPU_forward_backward_eval_schema_only']='passed'
        checks['input_arrays_unchanged_and_no_model_saved']='passed'
        checks['CPU_timing_explicitly_not_CUDA_or_rental_estimate']='passed'
    report={'status':'passed CPU guard/schema fixtures only','real_CUDA_executed':False,'real_source_data_or_checkpoints_opened':False,'test_outer_calibration_sealed_opened':False,'checks':checks,'code_sha256':hashlib.sha256((ROOT/'pipeline/gpu_preflight.py').read_bytes()).hexdigest(),'limits':'This fixture validates schema and control flow. It supplies no usable GPU speed or VRAM measurement.'}
    (ROOT/'reports/gpu_preflight_fixture_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
