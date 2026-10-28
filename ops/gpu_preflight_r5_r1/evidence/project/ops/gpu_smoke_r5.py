"""One actual R5 neural object, two synthetic CUDA microbatch Adam updates.

Run one --kind in each fresh process. No real TRAIN/INNER/OUTER/Test arrays,
source banks, labels, CSVs, downloads or production model directories are read.
This is a dtype/gradient/memory/checkpoint smoke, not full model training or
an effective-batch production experiment. OOM is reported and raised without
automatic batch reduction, retry, precision fallback or configuration edits.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
KINDS = ('bg_cnn', 'tfgrid_fdy_conformer', 'fdy', 'itransformer', 'timemoe_public', 'dasheng')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def _protect_real_data():
    forbidden = [(ROOT/name).resolve() for name in ('data', 'data.nosync', 'external', 'external.nosync')]
    def protect(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            if any(p == folder or folder in p.parents for folder in forbidden):
                raise RuntimeError('Real project data/source IO forbidden during synthetic CUDA smoke')
    sys.addaudithook(protect)


def generated_training(n, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.arange(15)[None, :, None]
    f = np.arange(25)[None, None, :]
    b = 10**((rng.normal(0, 3, (n, 1, 25)) + rng.normal(0, .8, (n, 15, 1)))/10)
    centre = rng.uniform(3, 12, (n, 1, 1))
    width = rng.uniform(1.5, 4, (n, 1, 1))
    r = np.exp(-.5*((t-centre)/width)**2)*np.exp(-f/rng.uniform(7, 15, (n, 1, 1)))
    r *= rng.uniform(.8, 1.2, (n, 1, 25))
    observed = rng.random((n, 15, 25)) > .025
    observed[::3, 0] = False
    observed[1::5, -1] = False
    b = np.where(observed, b, 0.)
    r = np.where(observed, r, 0.)
    wanted = rng.uniform(.05, .9, n)
    wanted[::4] = 0.
    alpha = wanted*b.sum((1, 2))/((1-wanted)*r.sum((1, 2)))
    a = alpha[:, None, None]*r
    e = b+a
    q = np.divide(a, e, out=np.zeros_like(e), where=observed).astype('float32')
    db = np.where(observed, 100+10*np.log10(np.maximum(e, 1e-300)), np.nan).astype('float32')
    peak = np.nanmax(db.astype(float), (1, 2), keepdims=True)
    saved_e = np.where(observed, 10**((db.astype(float)-peak)/10), 0.)
    share = ((q.astype(float)*saved_e).sum((1, 2))/saved_e.sum((1, 2))).astype('float32')
    return {'db':db, 'cell_share':q, 'share':share, 'presence':(share>0).astype('float32')}


def array_sha(data):
    import numpy as np
    h = hashlib.sha256()
    for name in sorted(data):
        a = data[name]
        h.update(json.dumps({'name':name,'shape':list(a.shape),'dtype':str(a.dtype)},sort_keys=True).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def parameter_digest(model, *, frozen=False, sample=False):
    """Full TRAINABLE bytes, or explicitly sampled FROZEN bytes (not full base)."""
    import torch
    h = hashlib.sha256(); elements = 0; hashed_bytes = 0; tensors = 0
    for name, p in model.named_parameters():
        if p.requires_grad == frozen:
            continue
        h.update(name.encode()+b'\0'+str((tuple(p.shape),str(p.dtype))).encode())
        x = p.detach().reshape(-1)
        if sample and len(x)>64:
            x = torch.cat((x[:32],x[-32:]))
        raw = x.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        h.update(raw); elements += p.numel(); hashed_bytes += len(raw); tensors += 1
    return {'sha256':h.hexdigest(),'parameter_elements':elements,'hashed_bytes':hashed_bytes,
            'parameter_tensors':tensors,'sample_only':sample}


def cpu_public_evidence(kind):
    """Link existing public-object CPU evidence; do not run it or read weights."""
    if kind not in {'timemoe_public','dasheng'}:
        return []
    records = []
    for p in sorted((ROOT/'reports').glob('R5_ACTUAL_'+kind.upper()+'_CPU*.json')):
        row = json.loads(p.read_text())
        before = row.get('frozen_base_sha256_before')
        records.append({'report':str(p.relative_to(ROOT)),'file_sha256':file_sha(p),
            'status':row.get('status'),'public_weight_sha256':row.get('weight_sha256'),
            'CPU_full_frozen_base_digest_equal':bool(before and before==row.get('frozen_base_sha256_after')),
            'CPU_only_not_this_CUDA_full_base_hash':True})
    return records


def cuda_memory(torch, device):
    return {'allocated_bytes':int(torch.cuda.memory_allocated(device)),
        'reserved_bytes':int(torch.cuda.memory_reserved(device)),
        'peak_allocated_bytes':int(torch.cuda.max_memory_allocated(device)),
        'peak_reserved_bytes':int(torch.cuda.max_memory_reserved(device))}


def run(args, report):
    import numpy as np
    import torch
    # Fail before any model, synthetic prior fit, public-weight read or CUDA job.
    device = torch.device(args.device)
    report.update({'torch':torch.__version__,'CUDA_available':torch.cuda.is_available()})
    if device.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('An available real CUDA device is required; no CPU smoke fallback')
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This configured smoke requires CUDA BF16 support; no dtype fallback')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42); np.random.seed(42)
    sys.path.insert(0,str(ROOT))
    from pipeline.runner_r5 import validate_config,CODE_NAMES
    from pipeline.grid_inputs_r5 import canonical_sha256,fit_shared_prior
    from pipeline.neural_r5 import make_neural,grid_inputs,physical_multitask_loss,_checkpoint,restore_into
    config = json.loads(args.config.read_text())
    config_sha = validate_config(config)
    code_hashes = {name:file_sha(ROOT/name) for name in CODE_NAMES}
    settings = dict(config['training'])
    if args.kind in {'timemoe_public','dasheng'}:
        settings.update(config['foundation_training_overrides'])
    microbatch = settings['microbatch'] if args.batch_size is None else args.batch_size
    if isinstance(microbatch,bool) or not isinstance(microbatch,int) or microbatch<1:
        raise ValueError('A positive integer microbatch is required')
    n = max(128,microbatch)
    policy = config['observation_candidates'][args.observation]
    prop = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    report.update({'phase':'synthetic_input','GPU_used':True,'config_sha256':config_sha,'config_file_sha256':file_sha(args.config),
        'code_sha256':code_hashes,
        'GPU_name':prop.name,'GPU_total_bytes':int(prop.total_memory),'GPU_free_before_bytes':int(free),
        'CUDA_runtime':torch.version.cuda,'cuDNN':torch.backends.cudnn.version(),
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'configured_microbatch':settings['microbatch'],'smoke_microbatch':microbatch,
        'explicit_microbatch_override':args.batch_size is not None,'production_effective_batch':settings['effective_batch'],
        'smoke_gradient_accumulation_microbatches':1,'smoke_optimizer_updates':2,
        'production_effective_batch_exercised':microbatch==settings['effective_batch'],
        'observation':args.observation,'observation_policy':policy,'threads':args.threads})
    begin = time.perf_counter()
    train = generated_training(n,42)
    train_sha = array_sha(train)
    binding = {'train_npz_sha256':train_sha,
        'data_package_hash':canonical_sha256({'synthetic_only':True,'arrays_sha256':train_sha}),
        'source_hash':canonical_sha256({'synthetic_generator':'gpu_smoke_r5.generated_training','seed':42,'rows':n}),
        'seed':42,'fold':0,'train_examples':n}
    prior = fit_shared_prior(train,binding,split='train')
    db = train['db'][:microbatch]
    grid, w = grid_inputs(db,prior,binding,policy)
    report.update({'synthetic_TRAIN_rows':n,'synthetic_array_sha256':train_sha,
        'synthetic_prior_payload_sha256':prior['shared_prior_payload_sha256'],
        'grid_shape':list(grid.shape),'input_preparation_seconds':time.perf_counter()-begin,
        'maximum_raw_weight_target_error':float(np.max(np.abs((train['cell_share'][:microbatch]*w).sum((1,2))-train['share'][:microbatch]))),
        'CPU_public_frozen_base_evidence':cpu_public_evidence(args.kind)})
    report['phase']='model_load'
    torch.cuda.reset_peak_memory_stats(device)
    loaded = time.perf_counter()
    model, receipt = make_neural(args.kind,config['model_settings'][args.kind],cache_dir=args.cache_dir)
    frozen_before = parameter_digest(model,frozen=True,sample=True)
    report.update({'model_receipt':receipt,
        'parameters':{'total':sum(p.numel() for p in model.parameters()),
            'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'frozen':sum(p.numel() for p in model.parameters() if not p.requires_grad)}})
    model = model.to(device)
    torch.cuda.synchronize(device)
    report['model_load_seconds']=time.perf_counter()-loaded
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError('No trainable parameters')
    transfer = time.perf_counter()
    x = torch.from_numpy(grid).to(device)
    weights = torch.from_numpy(w).to(device)
    targets = tuple(torch.from_numpy(train[k][:microbatch]).to(device) for k in ('share','presence','cell_share'))
    torch.cuda.synchronize(device)
    report['host_to_device_seconds']=time.perf_counter()-transfer
    optimizer = torch.optim.Adam(params,lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    steps=[]
    report['steps']=steps
    report['phase']='BF16_Adam_microbatch_steps'
    for step in range(1,3):
        model.train();optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device);started=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            outputs=model(x,weights)
            losses,parts=physical_multitask_loss(outputs,targets,weights,
                cell_weight=settings['cell_loss_weight'],presence_weight=settings['presence_loss_weight'])
        losses.mean().backward()
        missing=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]
        nonfinite=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
        frozen_grad=[name for name,p in model.named_parameters() if not p.requires_grad and p.grad is not None]
        if missing or nonfinite or frozen_grad:
            raise RuntimeError('Invalid gradients: '+json.dumps({'missing':missing[:8],'nonfinite':nonfinite[:8],'frozen':frozen_grad[:8]}))
        pool_error=float((outputs[0].float()-(outputs[2].float()*weights.float()).sum((1,2))).abs().max().detach())
        norm=torch.nn.utils.clip_grad_norm_(params,settings['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-started
        row={'step':step,'microbatch_examples':microbatch,'seconds':elapsed,'examples_per_second':microbatch/elapsed,
            'loss_smoke_only':float(losses.mean().detach()),
            **{name:float(value.mean().detach()) for name,value in parts.items()},
            'maximum_pool_reconstruction_error':pool_error,'unclipped_gradient_norm':float(norm),
            'all_trainable_gradients_finite_and_present':True,'frozen_gradients_absent':True,
            **cuda_memory(torch,device)}
        steps.append(row);print(json.dumps({'kind':args.kind,**row}),flush=True)
    optimizer.zero_grad(set_to_none=True)
    del optimizer,outputs,losses,parts
    gc.collect();torch.cuda.synchronize(device)
    report['phase']='checkpoint_reload'
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        expected=tuple(a.detach().clone() for a in model(x,weights))
    trainable_before = parameter_digest(model)
    checkpoint = _checkpoint(model,args.kind,receipt,binding,prior['shared_prior_payload_sha256'],policy,settings,2)
    checkpoint['synthetic_CUDA_smoke_only']=True
    checkpoint['production_training_executed']=False
    checkpoint_path=args.output.with_suffix('.checkpoint.pt')
    with checkpoint_path.open('xb') as handle:
        torch.save(checkpoint,handle)
    del checkpoint
    restored=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    with torch.no_grad():
        params[0].reshape(-1)[0].add_(1.)
    restore_into(model,restored,kind=args.kind,binding=binding,
        prior_sha=prior['shared_prior_payload_sha256'],policy=policy)
    del restored
    trainable_after = parameter_digest(model)
    if trainable_before!=trainable_after:
        raise RuntimeError('Saved trainable parameter bytes differ after restore')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        actual=model(x,weights)
    replay_error=[float((a.float()-b.float()).abs().max()) for a,b in zip(actual,expected)]
    if max(replay_error)>3e-6:
        raise RuntimeError('Same-input checkpoint replay exceeds3e-6: '+str(replay_error))
    frozen_after=parameter_digest(model,frozen=True,sample=True)
    if frozen_before!=frozen_after:
        raise RuntimeError('Sampled frozen parameter bytes changed')
    if array_sha(train)!=train_sha:
        raise RuntimeError('Synthetic raw inputs/labels were mutated')
    if file_sha(args.config)!=report['config_file_sha256'] or any(file_sha(ROOT/name)!=h for name,h in code_hashes.items()):
        raise RuntimeError('Frozen config/core source changed during CUDA smoke')
    torch.cuda.synchronize(device)
    memory=cuda_memory(torch,device)
    report.update({'status':'PASS','phase':'completed','checkpoint_file':checkpoint_path.name,
        'checkpoint_file_sha256':file_sha(checkpoint_path),'checkpoint_adapter_only':args.kind in {'timemoe_public','dasheng'},
        'checkpoint_torch_weights_only_load':True,'trainable_parameter_bytes_equal_after_restore':True,
        'checkpoint_prediction_max_abs_error':dict(zip(('share','presence_logit','cell_mask'),replay_error)),
        'sampled_frozen_parameter_digest_before':frozen_before,'sampled_frozen_parameter_digest_after':frozen_after,
        'CUDA_full_frozen_base_byte_hash_performed':False,'synthetic_inputs_unchanged':True,
        'total_seconds':time.perf_counter()-begin,'memory':memory,
        'peak_allocated_fraction_GPU_total':memory['peak_allocated_bytes']/total,
        'peak_reserved_fraction_GPU_total':memory['peak_reserved_bytes']/total,
        'reserved_peak_at_most_85_percent_total':memory['peak_reserved_bytes']/total<=.85,
        'maximum_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)),
        'timing_scope':'Two resident synthetic microbatch forward/backward/Adam updates; excludes CPU production source rendering and per-batch H2D pipeline costs.',
        'small_sample_timing_not_production_epoch_estimate':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',required=True,choices=KINDS)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/r5_training_integration.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--observation',default='resolution_01')
    parser.add_argument('--batch-size',type=int,help='Explicit smoke microbatch override; original config remains unchanged')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.threads<1 or args.batch_size is not None and args.batch_size<1:
        parser.error('Positive threads and optional batch size required')
    args.output=args.output.resolve()
    checkpoint_path=args.output.with_suffix('.checkpoint.pt')
    if args.output.exists() or args.output.is_symlink() or checkpoint_path.exists() or checkpoint_path.is_symlink():
        raise FileExistsError('Fresh JSON and checkpoint paths required; no overwrite or retry')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'running','kind':args.kind,'scope':'Synthetic two-microbatch-update CUDA smoke only',
        'production_training_executed':False,'actual_TRAIN_INNER_OUTER_Test_calibration_sealed_opened':False,
        'synthetic_known_q_y_used_as_supervision_only':True,'one_live_neural_object':True,
        'automatic_resize_retry_or_precision_fallback':False,'GPU_execution_attempted':True,'GPU_used':False,
        'script_sha256':file_sha(Path(__file__)),'platform':platform.platform()}
    with args.output.open('x') as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
    os.environ['OMP_NUM_THREADS']=str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS']=str(args.threads)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    _protect_real_data()
    try:
        run(args,report)
    except BaseException as exc:
        report.update({'status':'FAIL','error_type':type(exc).__name__,'error':str(exc),
            'OOM':type(exc).__name__ in {'OutOfMemoryError','CudaOutOfMemoryError'} or 'out of memory' in str(exc).lower()})
        if 'torch' in sys.modules:
            torch=sys.modules['torch']
            if torch.cuda.is_available():
                try:report['memory_at_failure']=cuda_memory(torch,torch.device(args.device))
                except Exception:pass
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        raise
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'kind':args.kind,'report':str(args.output),
        'smoke_microbatch':report['smoke_microbatch'],'memory':report['memory']}),flush=True)


if __name__=='__main__':
    main()
