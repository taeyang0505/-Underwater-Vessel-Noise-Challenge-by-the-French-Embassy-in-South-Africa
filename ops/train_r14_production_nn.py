"""Explicit raw-cache, capability and fresh Mac production14 NN commands."""
from pathlib import Path
import argparse,json,os,sys,time,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def preflight(config_path,output):
    import numpy as np,torch
    from pipeline.training_r14_production import validate_config,feature_basis,source_codes,sha,atomic_json,atomic_pt,restore_checkpoint
    from pipeline.features_r14_production import ProductionFrontend
    from pipeline.features_r14_anchor import channels
    from pipeline.models_r14_production import make_production_model
    from pipeline.training_r13 import physical_multitask_loss
    out=Path(output)
    if out.exists():raise FileExistsError('Fresh capability report required')
    c=validate_config(config_path);code=source_codes();torch.set_num_threads(2)
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise RuntimeError('Actual MPS with fallback0 required')
    basis,val,bundle=feature_basis(c,0);frontend=ProductionFrontend(basis)
    # No public queries: first128 stored full TRAIN rows from the exact frozen
    # baseline epoch1, with its materialization checksum validated first.
    source=ROOT/'models/r14_tree_full_r1/baselineP1/epoch1_train.npz';manifest=json.loads((source.parent/'epoch1_manifest.json').read_text())
    if sha(source)!=manifest['NPZ_sha256']:raise ValueError('Frozen source epoch differs')
    with np.load(source,allow_pickle=False) as z:raw={k:z[k][:128].copy() for k in ('db','share','presence','cell_share')}
    expected,w=channels(raw['db'],basis);actual,ww=frontend(torch.from_numpy(raw['db']))
    err=float(abs(expected-actual.numpy()).max());we=float(abs(w-ww.numpy()).max())
    if err>3e-6 or we>3e-8:raise ValueError('Actual14 NumPy/Torch parity failed')
    probes=raw['db'][:8].astype(float).copy();probes[0,:10]=np.nan;probes[1,:,23:]=np.nan;probes[2,:,3]=np.nan;probes[3,:,0:8]=np.nan
    oracle,ow=channels(probes,basis);px,pw=frontend(torch.from_numpy(probes))
    if abs(oracle-px.numpy()).max()>3e-6 or abs(ow-pw.numpy()).max()>3e-8:raise ValueError('Missing/partial14 parity failed')
    plan={'format':'r14_production14_capability_plan_v1','config_sha256':sha(config_path),'source_code_sha256':code,
        'batch_candidates':c['training']['batch_candidates'],'device_routing':{k:v['device'] for k,v in c['models'].items()},
        'CPU_threads':2,'warmup_steps':1,'timed_steps':3,'MPS_driver_memory_safety_limit_GiB':8,
        'batch_selection':'highest measured examples/sec among passing64/128, canonical smaller batch on tie',
        'model_performance_or_Test_not_used':True,'no_production_weights_reused_from_this_benchmark':True}
    atomic_json(out.with_suffix('.plan.json'),plan);models={};started=time.perf_counter()
    try:
        for kind,spec in c['models'].items():
            options=[]
            for batch in c['training']['batch_candidates']:
                torch.manual_seed(c['source_seed']);model=make_production_model(kind);device=spec['device'];cpu=model.eval()
                x,weight=actual[:batch],ww[:batch]
                with torch.no_grad():cp=tuple(t.clone() for t in cpu(x[:8],weight[:8]))
                model=model.to(device).eval();xx=x.to(device);wdev=weight.to(device)
                with torch.no_grad():mp=tuple(t.cpu() for t in model(xx[:8],wdev[:8]))
                delta=max(float(abs(a-b).max()) for a,b in zip(cp,mp))
                if delta>3e-4:raise ValueError('Actual14 CPU/device parity outside prespecified tolerance')
                labels=tuple(torch.from_numpy(raw[k][:batch]).to(device) for k in ('share','presence','cell_share'))
                opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003);model.train()
                times=[]
                for repeat in range(4):
                    if device=='mps':torch.mps.synchronize()
                    begin=time.perf_counter();opt.zero_grad(set_to_none=True);pred=model(xx,wdev);loss,_=physical_multitask_loss(pred,labels,wdev)
                    loss.mean().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                    if device=='mps':torch.mps.synchronize()
                    if repeat:times.append(time.perf_counter()-begin)
                import tempfile
                checkpoint_path=Path(tempfile.mkdtemp(prefix='r14-production14-portability-',dir='/private/tmp'))/'fixture.pt'
                payload={'format':'r14_production14_checkpoint_v1','fixture_only':True,'kind':kind,'epoch':0,'state_dict':model.state_dict(),
                    'architecture':model.architecture(),'feature_basis':basis,'frontend_metadata':frontend.metadata(),
                    'basis_file_sha256':bundle['folds']['0']['basis_sha256'],'config_sha256':sha(config_path),'source_code_sha256':code,
                    'approval_sha256':sha(ROOT/c['approval_path']),'data_hash':bundle['data_hash'],'fold':0,'optimizer_state_dict':opt.state_dict()}
                atomic_pt(checkpoint_path,payload)
                restored,rf,loaded=restore_checkpoint(checkpoint_path,allow_fixture=True,device='cpu')
                model.eval()
                with torch.no_grad():stored=tuple(t.cpu() for t in model(xx[:8],wdev[:8]));replay=restored(x[:8],weight[:8])
                portable_error=max(float(abs(a-b).max()) for a,b in zip(stored,replay))
                if portable_error>3e-4:raise ValueError('Actual device checkpoint CPU round-trip failed')
                if any(v.device.type!='cpu' or not torch.isfinite(v).all() for state in loaded['optimizer_state_dict']['state'].values() for v in state.values() if isinstance(v,torch.Tensor)):raise ValueError('Nonportable/nonfinite optimizer state')
                driver=torch.mps.driver_allocated_memory();rate=batch/float(np.median(times));safe=device=='cpu' or driver<=8*2**30
                options.append({'batch_size':batch,'device':device,'step_seconds':times,'median_step_seconds':float(np.median(times)),
                    'examples_per_second':rate,'driver_bytes_sampled':driver,'memory_is_true_peak':False,'within_predeclared_memory_limit':safe,
                    'CPU_device_eval_max_abs_error':delta,'CPU_checkpoint_roundtrip_max_abs_error':portable_error,
                    'optimizer_state_finite_CPU':True,'parameters':sum(p.numel() for p in model.parameters())})
                del model,opt,cpu,restored,loaded;gc.collect();torch.mps.empty_cache()
            passing=[a for a in options if a['within_predeclared_memory_limit']]
            if not passing:raise RuntimeError('No passing batch; stop, do not silently change settings')
            best=max(passing,key=lambda a:a['examples_per_second']);models[kind]={'selected_batch_size':best['batch_size'],'device':spec['device'],'options':options}
            print(json.dumps({'kind':kind,'selected_batch_size':best['batch_size'],'examples_per_second':best['examples_per_second']}),flush=True)
        if source_codes()!=code:raise ValueError('Preflight source changed')
        result={'format':'r14_production14_capability_v1','status':'passed','fold':0,'config_sha256':sha(config_path),
            'source_code_sha256':code,'models':models,'actual14_feature_max_error':err,'weights_max_error':we,
            'source_train_npz_sha256':sha(source),'preflight_plan_sha256':sha(out.with_suffix('.plan.json')),
            'elapsed_seconds':time.perf_counter()-started,'benchmark_weights_discarded':True,'full_actual_epoch_still_required_before_auto_continue':True}
        atomic_json(out,result);return result
    except BaseException as e:atomic_json(out.with_suffix('.FAILED.json'),{'error':repr(e),'source_code_sha256':code,'no_production_started':True});raise

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
    for action in ('materialize','preflight','train'):
        p=sub.add_parser(action);p.add_argument('--config',default='configs/r14_production_mac_v1.json');p.add_argument('--fold',type=int,default=0)
        if action=='materialize':p.add_argument('--epoch',type=int,required=True);p.add_argument('--cache',required=True)
        else:p.add_argument('--output',required=True)
        if action=='train':p.add_argument('--preflight',required=True);p.add_argument('--qa',required=True)
    a=ap.parse_args()
    from pipeline.training_r14_production import materialize,training,sha
    if a.action=='materialize':r=materialize(a.config,a.fold,a.epoch,a.cache)
    elif a.action=='preflight':
        if a.fold!=0:ap.error('Stage1 capability profile is explicitly Fold0; later-fold profiles require their own source-bound entry point')
        r=preflight(a.config,a.output)
    else:
        import fcntl
        qa=json.loads(Path(a.qa).read_text())
        if qa.get('status')!='passed' or qa['config_sha256']!=sha(a.config):raise ValueError('Independent production14 QA required')
        from pipeline.training_r14_production import source_codes
        if qa['source_code_sha256']!=source_codes():raise ValueError('QA source closure differs')
        handle=(ROOT/'models/.r14_production_MPS_writer.lock').open('a+')
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another production MPS process is active')
        r=training(a.config,a.output,a.preflight,a.fold)
    print(json.dumps({'status':'completed','result':str(r)},default=str))
if __name__=='__main__':main()
