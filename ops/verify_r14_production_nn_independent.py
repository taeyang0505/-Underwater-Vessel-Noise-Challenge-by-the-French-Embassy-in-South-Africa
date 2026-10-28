"""Independent bounded CPU contract checks, never production fitting/MPS.

One synthetic Adam step per default-size architecture exercises portable
optimizer checkpoints. All fixture weights remain under /private/tmp and
are explicitly forbidden to the production loader by default.
"""
from pathlib import Path
import copy,gc,hashlib,json,random,tempfile,time
import numpy as np
import torch
from pipeline.features_r14_production import ProductionFrontend
from pipeline.models_r14_production import make_production_model
from pipeline.training_r14_production import (source_codes,validate_config,atomic_pt,
    restore_checkpoint,training)
from pipeline.training_r13 import physical_multitask_loss
from pipeline.features_r12 import load_spectral_basis

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def finite_cpu(v):
    if isinstance(v,torch.Tensor):return v.device.type=='cpu' and bool(torch.isfinite(v).all())
    if isinstance(v,dict):return all(finite_cpu(a) for a in v.values())
    if isinstance(v,(list,tuple)):return all(finite_cpu(a) for a in v)
    return True

def main():
    out=ROOT/'reports/R14_PRODUCTION_NN_INDEPENDENT_QA.json'
    if out.exists():raise FileExistsError('Fresh final QA output required')
    config=ROOT/'configs/r14_production_mac_v1.json';c=validate_config(config);code=source_codes()
    assert sha(config)=='da33239b31a55e7232ccb518273d7777deab4c7d0c7ca1329300dfc0a919df13'
    approval=ROOT/c['approval_path'];assert sha(approval)=='b89cc22dac97db24b91a0d0e0edcddd554ddd884f5fafbac3df1b21cbb2e79c3'
    a=read(approval);assert sha(a['user_instruction_path'])==a['user_instruction_sha256']=='54bae81c2c0aa8b5e5fa299cf6aebbbe0af98b69d68f41add3c3b1802a212ba6'
    features=ROOT/'reports/R14_PRODUCTION_14_FEATURE_INDEPENDENT_QA.json';fq=read(features)
    assert fq['status']=='passed' and fq['config_sha256']==sha(config)
    for p,h in fq['source_sha256'].items():assert sha(ROOT/p)==h,p
    manifest=read(ROOT/c['bundle_path']/'manifest.json');f=manifest['folds']['0'];basisfile=ROOT/c['bundle_path']/f['basis_path']
    basis=load_spectral_basis(basisfile,expected_sha256=f['basis_sha256'],expected_training_binding=f['basis_training_binding'])
    torch.set_num_threads(2);rng=np.random.default_rng(2026102714)
    raw=np.round(80+rng.normal(0,4,(2,15,25)),1);raw[:,[0,5],:]=np.nan
    front=ProductionFrontend(basis);x,w=front(torch.from_numpy(raw))
    q=torch.full_like(w,.2);q[1]=0;q=torch.where(w>0,q,torch.zeros_like(q));y=(w*q).sum((1,2));z=(y>0).float()
    temp=Path(tempfile.mkdtemp(prefix='r14-production-CPU-contract-fixture-',dir='/private/tmp'));models=[];started=time.perf_counter()
    expected=dict(expected_config_sha256=sha(config),expected_fold=0,expected_data_hash=manifest['data_hash'],
        expected_basis_file_sha256=f['basis_sha256'],expected_approval_sha256=sha(approval))
    for kind in ('bg_cnn','fdy','itransformer','tfgrid_fdy_conformer'):
        torch.manual_seed(2026102714);model=make_production_model(kind);model.train()
        opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=c['models'][kind]['epochs'],eta_min=1e-5)
        pred=model(x,w);loss,parts=physical_multitask_loss(pred,(y,z,q),w);loss.mean().backward()
        assert torch.isfinite(loss).all() and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();sched.step();model.eval()
        with torch.no_grad():before=model(x,w)
        assert torch.equal(before[0],before[1]) and torch.all(before[2][w==0]==0)
        npstate=np.random.get_state()
        ck=dict(format='r14_production14_checkpoint_v1',fixture_only=True,kind=kind,epoch=1,
            state_dict=model.state_dict(),architecture=model.architecture(),feature_basis=basis,frontend_metadata=front.metadata(),
            config_sha256=sha(config),source_code_sha256=code,data_hash=manifest['data_hash'],fold=0,
            basis_file_sha256=f['basis_sha256'],approval_sha256=sha(approval),user_instruction_sha256=a['user_instruction_sha256'],
            device_used='cpu_fixture_only',optimizer_steps=1,optimizer_state_dict=opt.state_dict(),scheduler_state_dict=sched.state_dict(),
            torch_rng_state=torch.get_rng_state(),MPS_rng_state=None,
            numpy_rng_state=dict(algorithm=npstate[0],state_uint32=npstate[1].tolist(),position=npstate[2],has_gauss=npstate[3],cached_gaussian=npstate[4]),
            python_random_state=random.getstate(),raw_epoch_receipt_path='fixture_only_not_production',raw_epoch_receipt_sha256='fixture_only')
        folder=temp/kind;folder.mkdir();path=folder/'best.pt';atomic_pt(path,ck);atomic_pt(folder/'latest.pt',ck)
        saved=torch.load(path,map_location='cpu',weights_only=True);assert finite_cpu(saved)
        restored,newfront,reloaded=restore_checkpoint(path,**expected,device='cpu',allow_fixture=True)
        with torch.no_grad():after=restored(x,w)
        assert all(torch.equal(a,b) for a,b in zip(before,after))
        assert all(torch.equal(a,b) for a,b in zip(front(torch.from_numpy(raw)),newfront(torch.from_numpy(raw))))
        rejection={}
        try:restore_checkpoint(path,**expected)
        except ValueError:rejection['fixture_rejected_by_default']=True
        else:raise AssertionError('Fixture accepted for production')
        for name,key,bad in [('config','expected_config_sha256','0'*64),('fold','expected_fold',1),('data','expected_data_hash','0'*64),
                ('basis','expected_basis_file_sha256','0'*64),('approval','expected_approval_sha256','0'*64)]:
            kw=dict(expected);kw[key]=bad
            try:restore_checkpoint(path,**kw,allow_fixture=True)
            except ValueError:rejection[name]=True
            else:raise AssertionError('Foreign context accepted:'+name)
        models.append(dict(kind=kind,parameters=sum(p.numel() for p in model.parameters()),synthetic_Adam_steps=1,
            finite_loss_and_gradients=True,checkpoint_CPU_tensor_portability=True,optimizer_state_CPU_finite=True,
            strict_reload_predictions_bitwise=True,frontend_reload_bitwise=True,default_fixture_rejected=True,
            context_rejection_cases=rejection,fixture_checkpoint_sha256=sha(path)))
        del model,restored,opt,sched,ck,saved,reloaded,pred,before,after;gc.collect()
    # No MPS call: an existing output must be refused before any train step.
    preflight=temp/'unused_preflight.json';preflight.write_text(json.dumps({'status':'passed','config_sha256':sha(config),'source_code_sha256':code,'fold':0}))
    exists=temp/'failed_or_completed_output';exists.mkdir()
    try:training(config,exists,preflight,0)
    except FileExistsError:pass
    else:raise AssertionError('Existing output accepted')
    assert source_codes()==code
    for p,h in fq['source_sha256'].items():assert sha(ROOT/p)==h,p
    report=dict(format='r14_production_neural_independent_CPU_contract_QA_v1',status='passed',config_sha256=sha(config),source_code_sha256=code,
        approval_file_sha256=sha(approval),user_instruction_sha256=a['user_instruction_sha256'],data_hash=manifest['data_hash'],
        feature_mask_pattern_QA_path=str(features.relative_to(ROOT)),feature_mask_pattern_QA_sha256=sha(features),
        models=models,existing_output_rejected_before_training=True,fixture_weights_directory=str(temp),
        production_model_weights_created=False,actual_MPS_run_by_this_QA=False,
        actual_MPS_capability_preflight_separate_and_required=True,Test_or_heldout_arrays_read=False,
        elapsed_seconds=time.perf_counter()-started,
        limits=['CPU checkpoint fixture only; actual MPS-to-CPU checkpoint roundtrip must be reported by capability preflight.',
            'For production inference use restore_completed_model plus trusted global-catalog/runidentity SHA pins; primitive restore only checks the supplied expected context fields.',
            'Fixed fast3 budget100 and TF budget50 reflect explicit user permission; TF50 must not be reported as100.',
            'All new synthetic fixture checkpoints remain fixture_only=True and are forbidden to production reuse.'])
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'passed','report_sha256':sha(out),'models':[(r['kind'],r['parameters']) for r in models],'elapsed_seconds':report['elapsed_seconds']}))
if __name__=='__main__':main()
