"""Actual R5 factories through fit_neural, synthetic CPU fixtures only.

Each model runs in its own process. Public foundations are an explicit optional
extension using already cached pinned checkpoints, never downloads/random bases.
These losses are numerical test outputs, not validation performance evidence.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,tempfile,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SMALL=('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer')
PUBLIC=('timemoe_public','dasheng')
CHECKPOINT_CLOSURE={
    'pipeline/'+name for name in (
        'neural_r5.py','hybrid_models_r5.py','grid_models_r5.py','models.py','foundation_masks_r5.py',
        'grid_inputs_r5.py','mixtures.py','background_prior_features.py','observation_augmentation_r5.py',
        'load_foundation_r5.py','dynamic_passage_r5.py')}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def _numeric_arrays(n,seed):
    rng=np.random.default_rng(seed);db=rng.normal(65,10,(n,15,25)).astype('float32')
    db[0,[2,7]]=np.nan;q=rng.uniform(.05,.8,db.shape).astype('float32');q[~np.isfinite(db)]=0
    if n>=5:q[0]=0
    e=np.where(np.isfinite(db),10**((db.astype(float)-np.nanmax(db,(1,2),keepdims=True))/10),0)
    y=((e*q).sum((1,2))/e.sum((1,2))).astype('float32')
    return {'db':db,'share':y,'presence':(y>0).astype('float32'),'cell_share':q}

def create_fixture(folder):
    from pipeline.grid_inputs_r5 import fit_shared_prior,save_shared_prior,canonical_sha256
    train=_numeric_arrays(5,20261101);inner=_numeric_arrays(3,20261102)
    np.savez_compressed(folder/'train.npz',**train);np.savez_compressed(folder/'inner.npz',**inner)
    binding={'train_npz_sha256':sha(folder/'train.npz'),
        'data_package_hash':canonical_sha256({'TRAIN':sha(folder/'train.npz'),'INNER':sha(folder/'inner.npz'),'fixture_only':True}),
        'source_hash':canonical_sha256({'source':'generic independent numeric TRAIN/INNER','shape':[15,25]}),
        'seed':42,'fold':0,'train_examples':5}
    prior=fit_shared_prior(train,binding);saved=save_shared_prior(folder/'shared_prior.json',prior,expected_binding=binding)
    manifest={'binding':binding,'prior':saved,'prior_file':'shared_prior.json','train_file':'train.npz','inner_file':'inner.npz',
        'training_background_indices':[1000,1001,1002,1003,1004],
        'input_sha256':{'train.npz':sha(folder/'train.npz'),'inner.npz':sha(folder/'inner.npz')},
        'fixture_only':True,'actual_dataset_used':False}
    (folder/'fixture.json').write_text(json.dumps(manifest,indent=2));return manifest

def _parameter_digest(model,trainable):
    h=hashlib.sha256();elements=0
    for name,p in model.named_parameters():
        if bool(p.requires_grad)!=trainable:continue
        h.update(name.encode());h.update(str((tuple(p.shape),str(p.dtype))).encode())
        a=p.detach().cpu().contiguous().view(__import__('torch').uint8).numpy();h.update(a);elements+=p.numel()
    return h.hexdigest(),elements

def worker(kind,fixture,work,cache_dir):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    def no_network(event,args):
        if event in ['socket.connect','socket.getaddrinfo']:raise RuntimeError('Network disabled in numerical runner fixture')
    sys.addaudithook(no_network)
    import torch
    from pipeline import neural_r5
    from pipeline.grid_inputs_r5 import load_shared_prior
    from pipeline import observation_augmentation_r5 as observation
    torch.set_num_threads(2);torch.manual_seed(42);begin=time.monotonic()
    manifest=json.loads((fixture/'fixture.json').read_text());binding=manifest['binding']
    for name,h in manifest['input_sha256'].items():assert sha(fixture/name)==h
    with np.load(fixture/'train.npz',allow_pickle=False) as z:train={k:z[k] for k in ['db','share','presence','cell_share']}
    with np.load(fixture/'inner.npz',allow_pickle=False) as z:inner={k:z[k] for k in ['db','share','presence','cell_share']}
    prior=load_shared_prior(fixture/manifest['prior_file'],expected_binding=binding,
        expected_file_sha256=manifest['prior']['file_sha256'],expected_prior_sha256=manifest['prior']['payload_sha256'])
    config=json.loads((ROOT/'configs/r5_training_integration.json').read_text())
    policy=dict(config['observation_candidates']['resolution_01_endpoint']);settings=dict(config['training'])
    public=kind in PUBLIC;settings.update(epochs=1,patience=7,effective_batch=1 if public else 4,evaluation_batch=1 if public else 2)
    if public:settings['learning_rate']=config['foundation_training_overrides']['learning_rate']
    model,receipt=neural_r5.make_neural(kind,config['model_settings'][kind],cache_dir=cache_dir)
    before,trainable_elements=_parameter_digest(model,True)
    frozen_before,frozen_elements=_parameter_digest(model,False) if public else (None,0)
    canonical_grid,canonical_w=neural_r5.grid_inputs(train['db'],prior,binding,policy)
    first_forward={};gradient_stats={'hook_calls':0,'nonzero_hook_calls':0,'maximum_norm':0.};hooks=[]
    def before_forward(module,args):
        x,w=args[:2]
        if not first_forward:first_forward.update(shape=list(x.shape),first_example_grid_sha256=array_sha(x[:1].detach().numpy()),first_example_weight_sha256=array_sha(w[:1].detach().numpy()))
    forward_hook=model.register_forward_pre_hook(before_forward)
    def gradient_hook(gradient):
        if not torch.isfinite(gradient).all():raise FloatingPointError('Nonfinite parameter gradient')
        norm=float(gradient.detach().float().norm());gradient_stats['hook_calls']+=1;gradient_stats['nonzero_hook_calls']+=int(norm>0)
        gradient_stats['maximum_norm']=max(gradient_stats['maximum_norm'],norm);return gradient
    for p in model.parameters():
        if p.requires_grad:hooks.append(p.register_hook(gradient_hook))
    augmentation_counts=[];original_augment=observation.augment_training_observations
    def tracked_augment(*args,**kwargs):
        result,audit=original_augment(*args,**kwargs);augmentation_counts.append({k:audit[k] for k in ['input_windows','requested_events','applied_events','minimum_support_protection_skips','maximum_float32_raw_weight_target_residual']});return result,audit
    observation.augment_training_observations=tracked_augment
    fitting_rows=1 if public else 5;indices=np.asarray(manifest['training_background_indices'])
    def batches(epoch):
        for start in range(0,fitting_rows,1 if public else 2):
            end=min(fitting_rows,start+(1 if public else 2))
            yield {**{k:v[start:end] for k,v in train.items()},'background_index':indices[start:end]}
    fixed_inner={k:v[:1] if public else v for k,v in inner.items()}
    try:
        summary,pred=neural_r5.fit_neural(model,kind,receipt,batches,fixed_inner,prior,binding,policy,settings,work/'fit',device='cpu',fixture=True)
    finally:
        observation.augment_training_observations=original_augment;forward_hook.remove()
        for hook in hooks:hook.remove()
    history=[json.loads(s) for s in (work/'fit/epochs.jsonl').read_text().splitlines()]
    assert len(history)==1 and history[0]['optimizer_updates']==(1 if public else 2)
    assert history[0]['train_examples']==fitting_rows and np.isfinite(history[0]['maximum_unclipped_gradient_norm'])
    assert gradient_stats['hook_calls']>0 and gradient_stats['nonzero_hook_calls']>0
    after,_=_parameter_digest(model,True);assert before!=after,'Adam did not change trainable parameters'
    if public:
        frozen_after,_=_parameter_digest(model,False);assert frozen_before==frozen_after
    else:frozen_after=None
    checkpoint=torch.load(work/'fit/best.pt',map_location='cpu',weights_only=True)
    assert set(checkpoint['model_code_sha256'])==CHECKPOINT_CLOSURE
    for name,h in checkpoint['model_code_sha256'].items():assert sha(ROOT/name)==h
    # Alter only a private checkpoint mapping, never the actual source files.
    for name in sorted(CHECKPOINT_CLOSURE):
        changed={**checkpoint,'model_code_sha256':{**checkpoint['model_code_sha256'],name:'0'*64}}
        try:neural_r5.restore_into(model,changed,kind=kind,binding=binding,prior_sha=manifest['prior']['payload_sha256'],policy=policy)
        except ValueError:pass
        else:raise AssertionError('Changed preprocessing/model closure SHA accepted: '+name)
    for code_map in ({},{k:v for k,v in checkpoint['model_code_sha256'].items() if k!='pipeline/observation_augmentation_r5.py'}):
        changed={**checkpoint,'model_code_sha256':code_map}
        try:neural_r5.restore_into(model,changed,kind=kind,binding=binding,prior_sha=manifest['prior']['payload_sha256'],policy=policy)
        except ValueError:pass
        else:raise AssertionError('Missing preprocessing/model closure was accepted')
    if public:
        # One large vendor object per process: wipe adapters and reload them
        # onto that same verified frozen backbone, rather than claiming a
        # second large-object reload test.
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:p.zero_()
        restored=model
    else:restored,_=neural_r5.make_neural(kind,config['model_settings'][kind])
    neural_r5.restore_into(restored,checkpoint,kind=kind,binding=binding,prior_sha=manifest['prior']['payload_sha256'],policy=policy)
    replay=neural_r5.predict_neural(restored,fixed_inner['db'],prior,binding,policy,device='cpu',batch_size=settings['evaluation_batch'])
    for key in pred:assert np.array_equal(pred[key],replay[key]),('reload bitwise',kind,key)
    bad_bindings=[{'binding':{**binding,'seed':43},'prior_sha':manifest['prior']['payload_sha256'],'policy':policy},
        {'binding':binding,'prior_sha':'0'*64,'policy':policy},
        {'binding':binding,'prior_sha':manifest['prior']['payload_sha256'],'policy':{**policy,'quantization_decimals':None}}]
    for args in bad_bindings:
        try:neural_r5.restore_into(restored,checkpoint,kind=kind,**args)
        except ValueError:pass
        else:raise AssertionError('Checkpoint binding guard accepted mismatch')
    # Exercise scalar-head bypass rejection separately from any architecture.
    w=torch.from_numpy(canonical_w[:1]);qq=torch.full_like(w,.4);p=(qq*w).sum((1,2));s=torch.zeros(1)
    try:neural_r5.physical_multitask_loss((p+.1,s,qq),(p,torch.ones(1),qq),w)
    except ValueError:pass
    else:raise AssertionError('A nonphysical scalar prediction bypassed pooling guard')
    for name,h in manifest['input_sha256'].items():assert sha(fixture/name)==h
    for key,value in pred.items():assert np.isfinite(value).all()
    report={'status':'passed actual-factory fit_neural CPU fixture','kind':kind,'fixture_only':True,
        'actual_dataset_Test_OUTER_or_GPU_used':False,'no_network_during_load_or_compute':True,
        'public_checkpoint_used':public,'receipt':receipt,'parameters':sum(p.numel() for p in model.parameters()),
        'trainable_elements':trainable_elements,'frozen_elements':frozen_elements,
        'TRAIN_fixture_pool_rows':5,'TRAIN_rows_fitted':fitting_rows,'INNER_rows':len(fixed_inner['db']),
        'Adam_updates':history[0]['optimizer_updates'],'epochs':1,'finite_parameter_gradients':gradient_stats,
        'trainable_sha_changed':True,'frozen_parameter_sha_before':frozen_before,'frozen_parameter_sha_after':frozen_after,
        'first_forward':first_forward,'shared_prior_payload_sha256':manifest['prior']['payload_sha256'],
        'canonical_five_window_grid_sha256':array_sha(canonical_grid),'canonical_raw_physical_weight_sha256':array_sha(canonical_w),
        'observation_augmentation_calls':augmentation_counts,'best_checkpoint_restored':True,
        'checkpoint_reload_bitwise':True,'fresh_second_model_for_reload':not public,
        'same_verified_backbone_adapter_reload':public,'wrong_data_prior_or_policy_rejected':True,
        'checkpoint_code_closure_sha256':checkpoint['model_code_sha256'],
        'each_of_eleven_code_closure_mismatches_rejected':True,
        'empty_and_partial_code_closure_rejected':True,
        'physical_pooling_guard_rejected_scalar_bypass':True,'saved_summary':summary,
        'fixture_prediction_sha256':{k:array_sha(v) for k,v in pred.items()},
        'training_input_files_unchanged':True,'elapsed_seconds':time.monotonic()-begin}
    (work/'worker_report.json').write_text(json.dumps(report,indent=2,allow_nan=False));return report

def main(args):
    if args.worker:
        args.workdir.mkdir(parents=True,exist_ok=False)
        worker(args.worker,args.fixture,args.workdir,args.cache_dir);return
    start=time.monotonic();work=Path(tempfile.mkdtemp(prefix='r5_neural_runner_fixture_'));fixture=work/'numeric_data';fixture.mkdir();create_fixture(fixture)
    models=list(SMALL)+(list(PUBLIC) if args.include_public_foundations else []);results={};failures={}
    for kind in models:
        target=work/kind
        command=[sys.executable,'-m','ops.verify_neural_runner_r5','--worker',kind,'--fixture',str(fixture),'--workdir',str(target),'--cache-dir',str(args.cache_dir)]
        log=work/(kind+'.log')
        with log.open('w') as f:run=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if run.returncode:
            failures[kind]={'exit_code':run.returncode,'log':str(log),'tail':log.read_text()[-3500:]};break
        results[kind]=json.loads((target/'worker_report.json').read_text())
        print('PASS',kind,'Adam',results[kind]['Adam_updates'],'elapsed',round(results[kind]['elapsed_seconds'],2),flush=True)
    if not failures:
        assert len({r['shared_prior_payload_sha256'] for r in results.values()})==1
        assert len({r['canonical_five_window_grid_sha256'] for r in results.values()})==1
        assert len({r['canonical_raw_physical_weight_sha256'] for r in results.values()})==1
        assert len({r['first_forward']['first_example_grid_sha256'] for r in results.values()})==1
        assert len({r['first_forward']['first_example_weight_sha256'] for r in results.values()})==1
    report={'status':'passed synthetic CPU neural runner integration' if not failures else 'failed CPU fixture; no production run',
        'fixture_root':str(work),'models':results,'failures':failures,
        'same_shared_prior_grid8_and_raw_weight_inputs_across_factories':not failures,
        'actual_public_objects_separate_processes':args.include_public_foundations,
        'R4_R5_real_datasets_Test_OUTER_or_GPU_accessed':False,'full_TRAIN_or_production_training_performed':False,
        'synthetic_prior_for_production_reuse':False,'official_or_validation_performance_claimed':False,
        'existing_public_CPU_reports_overwritten':False,'elapsed_seconds':time.monotonic()-start,
        'code_sha256':{name:sha(ROOT/name) for name in ['ops/verify_neural_runner_r5.py','pipeline/neural_r5.py','pipeline/grid_inputs_r5.py','pipeline/observation_augmentation_r5.py']}}
    path=ROOT/'reports/R5_NEURAL_RUNNER_FIXTURE.json';path.write_text(json.dumps(report,indent=2,allow_nan=False))
    if failures:raise RuntimeError('CPU neural fixture failed; see '+str(path))
    print(json.dumps({'status':report['status'],'models':list(results),'report':str(path)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--include-public-foundations',action='store_true')
    p.add_argument('--worker',choices=SMALL+PUBLIC);p.add_argument('--fixture',type=Path);p.add_argument('--workdir',type=Path)
    p.add_argument('--cache-dir',type=Path,default=ROOT/'models/pretrained_r5');main(p.parse_args())
