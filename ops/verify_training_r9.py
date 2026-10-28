"""R9 engine CPU numeric fixtures only; no competition data or GPU operations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import sys

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from pipeline.relative_grid_r6_v2 import relative_grid_v2
from pipeline.models_r9 import make_r9_model
from pipeline.training_r9 import train_neural,predict_loader,physical_multitask_loss,CODE_NAMES


ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class NumericDataset(Dataset):
    def __init__(self,role,n=4):
        self.role=role;self.n=n;self.epoch=0;self.epoch_calls=[]
        self.identity={'fixture_only':True,'role':role,'rows':n,'numeric_seed':91}
    def __len__(self):return self.n
    def set_epoch(self,epoch):
        if self.role!='train':raise AssertionError('Fixed validation must never receive set_epoch')
        self.epoch=epoch;self.epoch_calls.append(epoch)
    def __getitem__(self,index):
        r=np.random.default_rng(91+index);b=r.uniform(.5,1.5,(15,25))
        a=np.exp(np.linspace(.5,-.5,25))[None,:]*np.maximum(0,np.sin(np.linspace(-.2,3.4,15)))[:,None]
        a=a*(.3+.03*self.epoch) if index%2 else np.zeros_like(a)
        mask=np.ones((15,25),bool)
        if index==0:mask[0:3]=False
        b=np.where(mask,b,0.);a=np.where(mask,a,0.);e=a+b
        raw=np.full((15,25),np.nan);raw[mask]=100+10*np.log10(e[mask]);q=np.divide(a,e,out=np.zeros_like(e),where=mask)
        x,w=relative_grid_v2(raw[None]);y=float(a.sum()/e.sum())
        return {'x':x[0],'w':w[0],'share':np.float32(y),'presence':np.float32(y>0),'cell_share':q.astype('float32'),
                'site':'numeric_B' if index==3 else 'numeric_A','sample_id':self.role+'_'+str(index),
                'loss_weight':np.float32(2. if index==3 else 2/3)}


class InvalidDataset(NumericDataset):
    def __init__(self,mode):super().__init__('train');self.mode=mode
    def __getitem__(self,index):
        if self.mode=='OOM':raise torch.cuda.OutOfMemoryError('explicit numeric fixture exception; no CUDA allocation')
        v=super().__getitem__(index)
        if self.mode=='NaN':v['x'][0,5,5]=np.nan
        else:v['loss_weight']=np.float32(-1)
        return v


def snapshot(dataset):
    h=hashlib.sha256()
    for i in range(len(dataset)):
        for k,v in dataset[i].items():
            h.update(k.encode());h.update(np.ascontiguousarray(v).tobytes() if isinstance(v,np.ndarray) else str(v).encode())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/R9_TRAINING_ENGINE_VERIFICATION.json')
    args=p.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError('Fresh verification report required')
    source={n:sha(ROOT/n) for n in CODE_NAMES};started=time.perf_counter();torch.set_num_threads(1)
    roots=[(ROOT/'data').resolve(),(ROOT/'data.nosync').resolve()]
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            if any(path==q or q in path.parents for q in roots):raise PermissionError('Source dataset I/O forbidden in numeric engine verification')
    sys.addaudithook(audit)
    temp=Path(tempfile.mkdtemp(prefix='r9-engine-numeric-',dir='/private/tmp'))
    models={'bg_cnn':{'width':8,'blocks':1},'tfgrid_fdy_conformer':{'embed_dim':8,'hidden_size':16,'grid_blocks':1,'conformer_blocks':1},
            'fdy':{'width':8,'blocks':1},'itransformer':{'width':8,'blocks':1}}
    cases=[]
    for kind,model_settings in models.items():
        train=NumericDataset('train');val=NumericDataset('validation');vbefore=snapshot(val)
        s={'fixture_only':True,'epochs':2,'num_workers':0,'threads':1,'batch_size':4,'evaluation_batch_size':4,
           'model_settings':model_settings,'seed':42,'monitor':'window_rmse'}
        summary=train_neural(kind,train,val,s,temp/kind,device='cpu')
        require=lambda b:(_ for _ in ()).throw(AssertionError(kind)) if not b else None
        require(train.epoch_calls==[1,2] and val.epoch_calls==[] and vbefore==snapshot(val))
        require(summary['epochs_run']==2 and summary['optimizer_steps']==2 and summary['fixture_only'])
        hist=[json.loads(v) for v in (temp/kind/'epochs.jsonl').read_text().splitlines()]
        w=min(range(2),key=lambda i:hist[i]['validation']['rmse'])+1;m=min(range(2),key=lambda i:hist[i]['validation']['site_macro_rmse'])+1
        require(summary['best_window_epoch']==w and summary['best_macro_epoch']==m)
        require(all(v['TRAIN_set_epoch_called']==v['epoch'] for v in hist))
        require(len({v['validation_input_sha256'] for v in hist})==1)
        require(all(abs(v['TRAIN_loss_weight_mean']-1)<1e-6 and v['TRAIN_loss_weight_max']==2. for v in hist))
        require(any(abs(v['train_balanced_multitask_loss']-v['train_unweighted_multitask_loss'])>1e-7 for v in hist))
        best=torch.load(temp/kind/'best_window.pt',map_location='cpu',weights_only=True)
        last=torch.load(temp/kind/'last.pt',map_location='cpu',weights_only=True)
        require(best['epoch']==w and last['epoch']==2 and last['optimizer_steps']==2)
        require(all(float(v['step'])==2 for v in last['optimizer_state_dict']['state'].values()))
        require(last['scheduler_state_dict']['last_epoch']==2)
        require(best['input_fingerprint']==last['input_fingerprint']==summary['input_fingerprint'])
        require(best['code_sha256']==source)
        for v in last['state_dict'].values():require(torch.isfinite(v).all())
        restored=make_r9_model(kind,**best['architecture']['model_settings']).eval();restored.load_state_dict(best['state_dict'],strict=True)
        require(restored.architecture()==best['architecture'])
        pred=predict_loader(restored,DataLoader(val,batch_size=4,shuffle=False),'cpu',collect_targets=True)
        with np.load(temp/kind/'best_window_validation_predictions.npz',allow_pickle=False) as z:
            require(all(np.array_equal(z[k],pred[k]) for k in ['share','score','cell_mask']))
        # Real strict model checks, distinct from the checkpoint roundtrip.
        with torch.no_grad():
            batch=next(iter(DataLoader(val,batch_size=4)));result=restored(batch['x'],batch['w'])
            require(torch.max(abs(result[0]-(result[2]*batch['w']).sum((1,2))))<3e-6)
            require((result[2][batch['w']==0]==0).all())
        try:train_neural(kind,train,val,s,temp/kind,device='cpu')
        except FileExistsError:resume_rejected=True
        else:raise AssertionError('Output reuse accepted')
        cases.append({'kind':kind,'status':'passed','fixture_only':True,'actual_Adam_steps':2,'epochs':2,
                      'TRAIN_set_epoch_calls':train.epoch_calls,'validation_set_epoch_calls':val.epoch_calls,
                      'validation_unchanged':True,'validation_byte_fingerprint_fixed':True,'best_window_epoch':w,'best_macro_epoch':m,
                      'checkpoint_reload_prediction_bitwise':True,'optimizer_scheduler_last_state_verified':True,
                      'resume_rejected':resume_rejected,'weighted_loss_distinct_from_unweighted_metric':True,
                      'parameters':summary['architecture']['total_parameters']})
    # Direct objective oracle and raw-label/rounded-w quantization compatibility.
    w=torch.zeros((1,15,25));w[0,0,:2]=.5;q=torch.zeros_like(w);q[0,0,0]=1
    pred=torch.tensor([.5],requires_grad=True);logit=torch.tensor([.2],requires_grad=True)
    y=torch.tensor([.5057561]);z=torch.ones(1);ycopy=y.clone()
    losses,parts=physical_multitask_loss((pred,logit,q),(y,z,q),w)
    expected=(pred-y).square()+.05*torch.nn.functional.binary_cross_entropy_with_logits(logit,z,reduction='none')
    assert torch.equal(losses,expected) and torch.equal(y,ycopy)
    assert .0057<float(parts['raw_label_vs_rounded_pool_gap'])<.00577
    try:physical_multitask_loss((pred,logit,q),(torch.tensor([.8]),z,q),w)
    except ValueError:bad_target=True
    else:raise AssertionError('Inconsistent known target accepted')
    failures=[]
    for name in ['NaN','OOM','negative_weight']:
        target=temp/('failure_'+name);tr=InvalidDataset(name);val=NumericDataset('validation')
        try:train_neural('bg_cnn',tr,val,{'fixture_only':True,'epochs':2,'num_workers':0,'threads':1,'batch_size':4,
                                       'model_settings':{'width':8,'blocks':1}},target,'cpu')
        except (ValueError,FloatingPointError,torch.cuda.OutOfMemoryError):
            fail=json.loads((target/'FAILED.json').read_text());assert not fail['resume_or_partial_reuse_allowed']
            failures.append({'case':name,'failure_recorded':True,'resume_forbidden':True,'CUDA_OOM_simulated':name=='OOM'})
        else:raise AssertionError('Bad numeric input did not stop')
    for epochs in [2,99,101]:
        try:train_neural('bg_cnn',NumericDataset('train'),NumericDataset('validation'),{'epochs':epochs},temp/f'bad_prod_{epochs}','cpu')
        except ValueError:pass
        else:raise AssertionError('Non100 production accepted')
    assert source=={n:sha(ROOT/n) for n in CODE_NAMES}
    result={'format':'r9_training_engine_numeric_fixture_v1','status':'passed','fixture_only':True,
            'real_source_dataset_or_Test_OUTER_cross_accessed':False,'formal_training_or_GPU_SSH_external_transmission':False,
            'code_sha256':source,'fixture_code_sha256':sha(__file__),'cases':cases,'failure_guards':failures,
            'loss_formula_oracle_exact':True,'raw_label_rounded_weight_gap_validated_not_relabelled':True,
            'invalid_known_target_rejected':bad_target,'production_non100_rejected':True,
            'production100_CUDA_BF16_workers4_not_executed_here':True,
            'temporary_numeric_outputs_must_not_be_reused':str(temp),'elapsed_seconds':time.perf_counter()-started}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    output.with_suffix('.md').write_text('# R9 training engine numeric verification\n\n'
        '**PASS, CPU numeric fixtures only.** Four small fresh-weight R9 adapters each completed two real Adam steps. '
        'No source dataset, Test, OUTER, cross panel, GPU, SSH or external transmission was used. Numeric weights are temporary and must never be reused.\n\n'
        'Checks: TRAIN set_epoch(1,2), fixed validation identities and actual input bytes, exact weighted objective with unweighted online RMSE, '
        'strict best-window/best-macro selection, saved last optimizer/scheduler steps, architecture/input/code identities, masked pooling, '
        'bitwise restored validation predictions and output reuse rejection. NaN, simulated OOM and negative loss weights stop with failed receipts. '
        'The original label is retained when rounded-weight pooling differs within its quantization bound; a larger target mismatch is rejected.\n\n'
        'Production requires exactly100epochs with no early stopping and uses Adam1e-3/wd3e-4/dropout.1/clip1. The scheduler monitors validation '
        'window RMSE by default; site-macro is an explicit alternative. Optional TRAIN loss_weight changes the objective without duplicating backgrounds, '
        'and does not imply improved leaderboard performance.\n\n'
        'This fixture did not run production100epochs, CUDA BF16, four spawned workers or actual source throughput. Those remain execution checks, '
        'not claims supported by this CPU fixture. Validation selection remains a development estimate, not an independent generalization guarantee.\n\n'
        f'[Detailed JSON]({output.name})\n')
    print(json.dumps({'status':'passed','report':str(output),'report_sha256':sha(output),'cases':len(cases),'elapsed_seconds':result['elapsed_seconds']}))


if __name__=='__main__':main()
