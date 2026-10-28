"""R16 neural capability checks and fresh fixed100-epoch training.

`smoke` uses synthetic only. `preflight` uses actual TRAIN rows only, never
locked panels or Test. Training is an explicit separate command.
"""
from pathlib import Path
import argparse,gc,json,os,sys,tempfile,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def resume_fixture(x,w,raw,config,device):
    """One diagnostic optimizer-step resume, not reuse of a production run."""
    import numpy as np,torch
    from pipeline.models_r16_production import make_r16_model,decoupled_loss,clip_decoupled_gradients
    from pipeline.training_r16_production import source_codes,atomic_pt,restore_checkpoint,rng_state,restore_rng,cpu_state
    torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    import random
    random.seed(config['seed'])
    model=make_r16_model('bg_cnn').to(device).train()
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=1e-5)
    xx=torch.from_numpy(x[:8]).to(device);ww=torch.from_numpy(w[:8]).to(device)
    labels=tuple(torch.from_numpy(np.asarray(raw[k][:8],np.float32)).to(device)for k in('share','presence','cell_share'))
    def step(m,o,s):
        m.train();o.zero_grad(set_to_none=True);pr=m(xx,ww);loss,_=decoupled_loss(pr,labels,ww);loss.backward();clip_decoupled_gradients(m);o.step();s.step()
        if device=='mps':torch.mps.synchronize()
        return float(loss.detach().cpu())
    step(model,opt,scheduler)
    p=Path(tempfile.mkdtemp(prefix='r16-resume-fixture-',dir='/private/tmp'))/'epoch1.pt'
    ck={'format':'r16_production8_checkpoint_v1','kind':'bg_cnn','epoch':1,'state_dict':model.state_dict(),'architecture':model.architecture(),
       'fixture_only':True,'source_code_sha256':source_codes(),'identity_sha256':'resume_fixture_only',
       'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'rng_state':rng_state(device)}
    atomic_pt(p,ck)
    expected_loss=step(model,opt,scheduler);expected=cpu_state(model.state_dict());expected_lr=opt.param_groups[0]['lr']
    restored,loaded=restore_checkpoint(p,device=device,allow_fixture=True)
    ro=torch.optim.Adam(restored.parameters(),lr=.001,weight_decay=.0003);ro.load_state_dict(loaded['optimizer_state_dict'])
    for state in ro.state.values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor)and key!='step':state[key]=value.to(device)
    rs=torch.optim.lr_scheduler.CosineAnnealingLR(ro,T_max=100,eta_min=1e-5);rs.load_state_dict(loaded['scheduler_state_dict'])
    restore_rng(loaded['rng_state'],device);actual_loss=step(restored,ro,rs)
    err=max(float((expected[k]-v.detach().cpu()).abs().max())for k,v in restored.state_dict().items())
    tol=3e-4 if device=='mps'else 2e-7
    if err>tol or abs(actual_loss-expected_loss)>tol or ro.param_groups[0]['lr']!=expected_lr:raise ValueError('Complete optimizer/scheduler/RNG resume replay differs')
    del model,opt,scheduler,restored,ro,rs;gc.collect()
    if device=='mps':torch.mps.empty_cache()
    return{'status':'passed','device':device,'checkpoint_epoch':1,'replayed_epoch2_step_max_state_abs':err,
           'loss_abs_difference':abs(actual_loss-expected_loss),'next_learning_rate_exact':True,
           'optimizer_scheduler_CPU_torch_MPS_numpy_python_RNG_restored':True,'production_steps':0}

def capability(config,output,manifest=None,fold=0):
    import numpy as np,torch
    from pipeline.models_r16_production import make_r16_model,decoupled_loss,clip_decoupled_gradients
    from pipeline.relative_grid_r6_v2 import relative_grid_v2
    from pipeline.training_r16_production import validate_config,source_codes,sha,atomic_json,atomic_pt,restore_checkpoint
    c=validate_config(config);out=Path(output)
    if out.exists():raise FileExistsError('Fresh R16 capability report required')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if any(v['device']=='mps'for v in c['models'].values())and(not torch.backends.mps.is_available()or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0'):raise RuntimeError('MPS available and explicit fallback0 required')
    if manifest is not None:
        from pipeline.r16_production_data import load_production_role,validate_production_manifest
        data_qa=validate_production_manifest(manifest,fold,verify_sources=True)
        d=load_production_role(manifest,fold,'train');raw={k:np.asarray(d[k])[:128].copy()for k in('db','share','presence','cell_share')}
    else:
        rng=np.random.default_rng(c['seed']);db=np.round(rng.normal(80,5,(128,15,25)),1).astype(np.float32)
        db[:16,0]=np.nan;db[16:24,:,23:]=np.nan;db[24:28,:,4]=np.nan
        x,w=relative_grid_v2(db);q=rng.uniform(0,1,w.shape).astype(np.float32);q[w==0]=0
        raw={'db':db,'share':(q*w).sum((1,2)),'presence':np.ones(128,np.float32),'cell_share':q}
    x,w=relative_grid_v2(raw['db']);source=source_codes();results={};began=time.perf_counter()
    plan={'status':'prespecified','fold':fold,'config_sha256':sha(config),'data_manifest_sha256':sha(manifest)if manifest else None,'source_code_sha256':source,
        'synthetic_only':manifest is None,'batch_candidates':c['training']['batch_candidates'],'warmup_steps':1,'timed_steps':3,
        'devices':{k:v['device']for k,v in c['models'].items()},'maximum_sampled_MPS_driver_GiB':8,
        'CPU_device_eval_atol':3e-4,'checkpoint_portability_atol':3e-4,'model_performance_not_used_for_device_batch_selection':True,
        'global_minimum_guarantee':False,'production_training_started':False,'Test_used':False}
    atomic_json(out.with_suffix('.plan.json'),plan)
    try:
        for kind,spec in c['models'].items():
            candidates=[]
            for bs in c['training']['batch_candidates']:
                torch.manual_seed(c['seed']);model=make_r16_model(kind).eval();device=spec['device']
                xx=torch.from_numpy(x[:bs]);ww=torch.from_numpy(w[:bs])
                with torch.no_grad():cp=tuple(t.clone()for t in model(xx[:8],ww[:8]))
                # Independent loss gradients checked on CPU, zero optimizer steps.
                model.train();p,l,q=model(xx[:8],ww[:8]);model.zero_grad(set_to_none=True)
                l.sum().backward()
                if any(v.grad is not None for v in model.regressor.parameters()):raise ValueError('Classifier gradient reached mask regressor')
                model.zero_grad(set_to_none=True);p,l,q=model(xx[:8],ww[:8]);p.sum().backward()
                if any(v.grad is not None for v in model.classifier.parameters()):raise ValueError('Regression gradient reached classifier')
                model.zero_grad(set_to_none=True);model=model.to(device).eval();xd=xx.to(device);wd=ww.to(device)
                with torch.no_grad():dp=tuple(t.cpu()for t in model(xd[:8],wd[:8]))
                error=max(float(abs(a-b).max())for a,b in zip(cp,dp))
                if error>3e-4:raise ValueError('CPU/device parity outside fixed tolerance')
                poolerr=float(abs(dp[0]-(dp[2]*ww[:8]).sum((1,2))).max())
                if poolerr>3e-6 or np.max(np.abs(dp[2].numpy()[w[:8]==0]),initial=0)!=0:raise ValueError('Physical pooling/missing mask changed')
                # Classification must ignore static profile and energy weights.
                with torch.no_grad():
                    changed=xd[:8].clone();changed[:,4]+=2*changed[:,5];changed[:,7]=changed[:,5]*.1
                    a=model.classifier(xd[:8]);b=model.classifier(changed)
                if float(abs(a-b).max())!=0:raise ValueError('Independent classifier reads profile or sqrt(w)')
                labels=tuple(torch.from_numpy(np.asarray(raw[k][:bs],np.float32)).to(device)for k in('share','presence','cell_share'))
                model.train();opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003);times=[];finite=True
                for i in range(4):
                    if device=='mps':torch.mps.synchronize()
                    t=time.perf_counter();opt.zero_grad(set_to_none=True);pr=model(xd,wd);loss,parts=decoupled_loss(pr,labels,wd);loss.backward();clip_decoupled_gradients(model);opt.step()
                    if device=='mps':torch.mps.synchronize()
                    if i:times.append(time.perf_counter()-t)
                    finite=finite and bool(torch.isfinite(loss))
                path=Path(tempfile.mkdtemp(prefix='r16-neural-portability-',dir='/private/tmp'))/'fixture.pt'
                ck={'format':'r16_production8_checkpoint_v1','kind':kind,'epoch':0,'state_dict':model.state_dict(),'architecture':model.architecture(),
                    'fixture_only':True,'source_code_sha256':source,'identity_sha256':'synthetic_capability_only','optimizer_state_dict':opt.state_dict()}
                atomic_pt(path,ck);restored,loaded=restore_checkpoint(path,device='cpu',allow_fixture=True)
                with torch.no_grad():
                    model.eval();saved=tuple(t.cpu()for t in model(xd[:8],wd[:8]));replay=restored(xx[:8],ww[:8])
                portability=max(float(abs(a-b).max())for a,b in zip(saved,replay))
                if portability>3e-4:raise ValueError('Checkpoint CPU round-trip outside fixed tolerance')
                driver=int(torch.mps.driver_allocated_memory())if device=='mps'else 0
                result={'batch_size':bs,'device':device,'parameters':model.architecture(),'median_step_seconds':float(np.median(times)),
                    'step_seconds':times,'examples_per_second':bs/float(np.median(times)),'CPU_device_max_abs':error,'pool_max_abs':poolerr,
                    'classifier_profile_energy_invariance_max_abs':0.,'cross_parameter_gradients':False,
                    'checkpoint_CPU_portability_max_abs':portability,'finite_loss_grad_optimizer':finite,
                    'sampled_MPS_driver_bytes':driver,'memory_measurement_is_true_peak':False,'within_memory_limit':driver<=8*2**30}
                candidates.append(result)
                del model,opt,restored,loaded;gc.collect()
                if device=='mps':torch.mps.empty_cache()
            okay=[v for v in candidates if v['within_memory_limit']]
            if not okay:raise RuntimeError('No memory-safe batch; do not silently change config')
            chosen=max(okay,key=lambda v:v['examples_per_second'])
            results[kind]={'selected_batch_size':chosen['batch_size'],'device':spec['device'],'options':candidates}
            print(json.dumps({'event':'capability_model_done','kind':kind,'batch':chosen['batch_size'],'examples_per_second':chosen['examples_per_second']}),flush=True)
        if source_codes()!=source:raise ValueError('Neural capability code changed')
        resume_checks=[resume_fixture(x,w,raw,c,'cpu')]
        if any(v['device']=='mps'for v in c['models'].values()):resume_checks.append(resume_fixture(x,w,raw,c,'mps'))
        result={'status':'passed','format':'r16_neural_capability_v1','fold':fold,'config_sha256':sha(config),
             'data_manifest_sha256':sha(manifest)if manifest else None,'source_code_sha256':source,'models':results,
             'synthetic_only':manifest is None,'data_source_validation':data_qa if manifest is not None else None,
             'elapsed_seconds':time.perf_counter()-began,'benchmark_weights_discarded':True,
             'production_training_started':False,'Test_used':False,'resume_fixture':resume_checks,
             'runtime':{'Python':sys.version,'torch':str(torch.__version__)}}
        atomic_json(out,result);return result
    except BaseException as e:
        atomic_json(out.with_suffix('.FAILED.json'),{'status':'failed','error':repr(e),'production_started':False});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['smoke','preflight','train','cache']);p.add_argument('--config',default='configs/r16_neural_mac_v1.json');p.add_argument('--manifest');p.add_argument('--output',required=True);p.add_argument('--fold',type=int,default=0);p.add_argument('--preflight');p.add_argument('--resume',action='store_true');p.add_argument('--cache-root');a=p.parse_args()
    from pipeline.training_r16_production import guard,train,materialize_features
    sys.addaudithook(guard)
    if a.command in('preflight','train','cache')and not a.manifest:p.error('--manifest is required')
    if a.command=='smoke':result=capability(a.config,a.output,fold=a.fold)
    elif a.command=='preflight':result=capability(a.config,a.output,a.manifest,a.fold)
    elif a.command=='cache':
        result={role:materialize_features(a.manifest,a.fold,role,Path(a.output)/role)for role in('train','selection')}
    else:
        if not a.preflight:p.error('--preflight required for production')
        import fcntl
        handle=(ROOT/'models/.r16_production_MPS_writer.lock').open('a+')
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another R16 MPS training writer is active')
        result=str(train(a.config,a.manifest,a.output,a.preflight,a.fold,a.resume,a.cache_root))
    print(json.dumps({'status':'completed','result':result},default=str))
if __name__=='__main__':main()
