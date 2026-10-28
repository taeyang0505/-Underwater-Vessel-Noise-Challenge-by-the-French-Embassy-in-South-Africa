"""Explicit M4 MPS capability smoke: fresh actual R13-size models, not a fit run.

The FP64-parity frontend stays on CPU; only its FP32 x/w and targets enter
MPS. No automatic CPU operator fallback, Test reader or production launcher.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,time,gc,resource,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from pipeline.features_r12 import load_spectral_basis
from pipeline.features_r14_torch import TorchFeatureFrontend
from pipeline.models_r13 import make_r13_model
from pipeline.training_r13 import physical_multitask_loss

OUT=ROOT/'reports/R14_MPS_BENCHMARK_r2'
def write(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if OUT.exists():raise FileExistsError('Fresh MPS diagnostic output required')
    OUT.mkdir(parents=True)
    plan={'version':'r14_MPS_capability_smoke_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'model_kinds':['bg_cnn','tfgrid_fdy_conformer','fdy','itransformer'],'batches':[8,16,32],
        'CPU_threads':2,'precision':'FP32 backbone, CPU FP64 frontend','warmup_Adam_steps':1,'timed_Adam_steps':3,
        'CPU_MPS_prediction_atol':3e-4,'CPU_MPS_prediction_rtol':2e-4,
        'fresh_random_weights_not_production':True,'no_Test_report_or_training_dataset_fit':True,
        'MPS_operator_CPU_fallback_allowed':False,'full_epoch_or_full_training':False,
        'CPU_synthesis_excluded_from_step_timing_and_reported_separately':True}
    write(OUT/'plan.json',plan)
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise RuntimeError('Set PYTORCH_ENABLE_MPS_FALLBACK=0 explicitly')
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():raise RuntimeError('Actual MPS device is unavailable, no CPU substitution')
    fixture=ROOT/'reports/R14_TORCH_FRONTEND/actual_TRAIN_fixture.npz'
    with np.load(fixture,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    bundle=ROOT/'data/synthetic/group_cv_physics_r13_variants_v2';manifest=json.loads((bundle/'manifest.json').read_text());f=manifest['folds']['0']
    basis=load_spectral_basis(bundle/f['basis_path'],expected_sha256=f['basis_sha256'],expected_training_binding=f['basis_training_binding'])
    frontend=TorchFeatureFrontend(basis);source_names=['pipeline/models_r13.py','pipeline/training_r13.py','pipeline/features_r14_torch.py','ops/benchmark_r14_mps.py']
    before={n:sha(ROOT/n) for n in source_names}
    try:torch.ones(1,device='mps',dtype=torch.float64)
    except (TypeError,RuntimeError) as e:float64={'supported':False,'error':str(e)}
    else:float64={'supported':True}
    results=[];failures=[]
    for size in plan['batches']:
        raw=torch.from_numpy(data['db'][:size]);labels=tuple(torch.from_numpy(data[k][:size]).float() for k in ('share','presence','cell_share'))
        feature_times=[]
        for repeat in range(3):
            start=time.perf_counter();x,w=frontend(raw);feature_times.append(time.perf_counter()-start)
        for kind in plan['model_kinds']:
            write(OUT/'progress.json',{'status':'running','kind':kind,'batch_size':size,'completed_cases':len(results),'failed_cases':len(failures)})
            cpu=mps=opt_cpu=opt_mps=None
            try:
                torch.manual_seed(2026102714);cpu=make_r13_model(kind).eval()
                mps=make_r13_model(kind);mps.load_state_dict(cpu.state_dict(),strict=True);mps=mps.to('mps').eval()
                mx,mw=x.to('mps'),w.to('mps');mlabels=tuple(a.to('mps') for a in labels)
                with torch.no_grad():
                    cp=cpu(x,w);mp=tuple(a.cpu() for a in mps(mx,mw))
                errors=[float(abs(a-b).max()) for a,b in zip(cp,mp)]
                parity=all(torch.allclose(a,b,atol=plan['CPU_MPS_prediction_atol'],rtol=plan['CPU_MPS_prediction_rtol']) for a,b in zip(cp,mp))
                if not parity:raise AssertionError('Prespecified CPU/MPS prediction tolerance failed: '+str(errors))
                cpu.train();mps.train()
                opt_cpu=torch.optim.Adam(cpu.parameters(),lr=.001,weight_decay=.0003)
                opt_mps=torch.optim.Adam(mps.parameters(),lr=.001,weight_decay=.0003)
                memory=[]
                def step(model,opt,a,b,targets,is_mps):
                    opt.zero_grad(set_to_none=True);pred=model(a,b);loss,_=physical_multitask_loss(pred,targets,b)
                    if not bool(torch.isfinite(loss).all()):raise FloatingPointError('Nonfinite loss')
                    loss.mean().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                    if is_mps:
                        torch.mps.synchronize();memory.append({'current_tensor_bytes':torch.mps.current_allocated_memory(),
                            'driver_bytes':torch.mps.driver_allocated_memory()})
                    return float(loss.detach().mean().cpu())
                step(cpu,opt_cpu,x,w,labels,False);step(mps,opt_mps,mx,mw,mlabels,True)
                times={};lastloss={}
                for device,model,opt,a,b,target in [('CPU',cpu,opt_cpu,x,w,labels),('MPS',mps,opt_mps,mx,mw,mlabels)]:
                    samples=[]
                    for repeat in range(3):
                        if device=='MPS':torch.mps.synchronize()
                        start=time.perf_counter();lastloss[device]=step(model,opt,a,b,target,device=='MPS');samples.append(time.perf_counter()-start)
                    times[device]={'step_seconds':samples,'median_step_seconds':float(np.median(samples)),
                        'examples_per_second':size/float(np.median(samples))}
                integrated=[]
                for repeat in range(3):
                    torch.mps.synchronize();start=time.perf_counter();xx,ww=frontend(raw)
                    lastloss['MPS_integrated']=step(mps,opt_mps,xx.to('mps'),ww.to('mps'),tuple(a.to('mps') for a in labels),True)
                    integrated.append(time.perf_counter()-start)
                result={'kind':kind,'batch_size':size,'parameters':sum(p.numel() for p in cpu.parameters()),
                    'CPU_MPS_eval_max_abs_errors_share_score_mask':errors,'parity_passed':parity,
                    'all_Adam_losses_and_gradient_norms_finite':True,'last_fixture_loss':lastloss,
                    'CPU_frontend_seconds':feature_times,'backbone_timing':times,
                    'MPS_CPUfrontend_transfer_Adam_seconds':integrated,
                    'observed_max_current_tensor_bytes':max(p['current_tensor_bytes'] for p in memory),
                    'observed_max_driver_bytes':max(p['driver_bytes'] for p in memory),
                    'memory_scope':'sampled after synchronized Adam steps, not a true activation peak',
                    'process_lifetime_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                results.append(result);print(json.dumps({'kind':kind,'batch':size,'status':'passed','CPU_s':times['CPU']['median_step_seconds'],
                    'MPS_s':times['MPS']['median_step_seconds'],'integrated_MPS_s':float(np.median(integrated))}),flush=True)
            except BaseException as e:
                failures.append({'kind':kind,'batch_size':size,'error_type':type(e).__name__,'error':str(e),'parameters_changed_or_retried':False})
                print(json.dumps({'kind':kind,'batch':size,'status':'failed','error':str(e)}),flush=True)
            finally:
                cpu=mps=opt_cpu=opt_mps=None;gc.collect();torch.mps.empty_cache();torch.mps.synchronize()
            write(OUT/'partial_results.json',{'completed':results,'failures':failures})
    assert all(sha(ROOT/n)==h for n,h in before.items())
    result={'status':'passed' if not failures else 'partial_capability_failures','torch':torch.__version__,'python':sys.version,
        'MPS_available':True,'recommended_max_memory_bytes':torch.mps.recommended_max_memory(),'MPS_float64_probe':float64,
        'CPU_fp64_frontend_plus_MPS_fp32_backbone':True,'MPS_operator_CPU_fallback':False,
        'data_hash':manifest['data_hash'],'actual_TRAIN_fixture_sha256':sha(fixture),'source_code_sha256':before,
        'plan_sha256':sha(OUT/'plan.json'),'results':results,'failures':failures,'full_training_or_production_launch':False,
        'GPU_epoch_or_100epoch_total_training_time_estimated':False,'fresh_weights_discarded':True,
        'cannot_claim_CUDA_parity_or_same_throughput':True,'concurrent_CPU_load_uncontrolled':True}
    write(OUT/'RESULTS.json',result);write(OUT/'progress.json',{'status':'completed','cases':len(results),'failures':len(failures)})
    print(json.dumps({'status':result['status'],'cases':len(results),'failures':len(failures),'result':str(OUT/'RESULTS.json')}))

if __name__=='__main__':main()
