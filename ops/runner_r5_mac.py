"""Fresh R5 controller with an EXCLUSIVE Mac tree executor.

The frozen core17/tree7 remain unchanged. This controller NEVER fits a tree
on the server. It imports only explicitly certified completed legacy NN
children, trains other NN children, and waits for a hash-bound Mac spool job.
No automatic restart, partial-model resume, Pod control, Test or submission.
"""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, os, shutil, subprocess, sys, time, zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pipeline.inference_r5 import _safe, _json, file_sha256, REQUIRED_CODE, FrozenInferenceR5
from pipeline.hierarchical_ensemble_r5 import canonical_sha256, select_inner_policy

NEURAL={'bg_cnn','tfgrid_fdy_conformer','timemoe_public','dasheng','fdy','itransformer'}
TREE_EXTRA={'pipeline/train_lgbm.py','pipeline/prototypes.py'}
TREE_FILES={'summary.json','share.txt','presence.txt','prototypes.npz','physics_pca.json','inner_predictions.npz','producer_summary.json'}
FORMAT='r5_strict_mac_controller_v1'

def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')

def atomic(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def verify_mapping(root,mapping):
    if not isinstance(mapping,dict) or not mapping:raise ValueError('Explicit immutable SHA mapping required')
    for name,h in mapping.items():
        if file_sha256(_safe(root,name))!=h:raise ValueError('Frozen artifact/code changed: '+name)

def process_alive(pid,start_ticks=None,*,_proc_root=Path('/proc')):
    try:
        tail=(_proc_root/str(int(pid))/'stat').read_text().rsplit(')',1)[1].split();state,actual=tail[0],tail[19]
        if start_ticks is not None and actual!=str(start_ticks):raise ValueError('Legacy PID was reused; identity mismatch')
        if state in {'Z','X','x'}:return False
    except (OSError,TypeError,IndexError):pass
    try:os.kill(int(pid),0)
    except (ProcessLookupError,TypeError,ValueError):return False
    except PermissionError:return True
    return True

class NativeOperations:
    fixture_backend=False
    def fold(self,path,seed):
        from pipeline.fold_data_r5 import FoldDataR5
        return FoldDataR5(path,seed)
    def prior(self,train,binding,path):
        from pipeline.grid_inputs_r5 import fit_shared_prior,save_shared_prior
        return save_shared_prior(path,fit_shared_prior(train,binding,split='train'),expected_binding=binding)
    def checkpoint(self,path):
        import torch
        return torch.load(path,map_location='cpu',weights_only=True)
    def nn_fit(self,context,kind,observation,output):
        from pipeline.runner_r5 import _call
        _call('pipeline.neural_r5',['fit','--context',context,'--kind',kind,'--observation',observation,'--output',output])
    def predict(self,context,kind,observation,model,policy,output):
        from pipeline.runner_r5 import _call
        module='pipeline.lgbm_r5' if kind=='lgbm_physics' else 'pipeline.neural_r5'
        args=['predict','--context',context,'--observation',observation,'--model-dir',model,'--role','outer','--frozen-policy',policy,'--output',output]
        if kind!='lgbm_physics':args+=['--kind',kind]
        _call(module,args)
    def read_predictions(self,path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    def weights(self,db):
        from pipeline.observation_augmentation_r5 import prepare_query_observations
        return prepare_query_observations(db,quantization_decimals=None)['energy_weights']
    def metrics(self,arrays,meta,prediction):
        from pipeline.runner_r5 import _metrics
        return _metrics(arrays,meta,prediction)
    def ensemble(self,policy,branches,w,binding,role):
        from pipeline.runner_r5 import _ensemble_prediction
        return _ensemble_prediction(policy,branches,w,binding,role)
    def verify_mac(self,context,selection,model,expected_config_sha,root):
        # The producer installs irreversible TRAIN/INNER-only audit hooks.
        # Isolate that verifier so later NN subprocesses/OUTER remain legal.
        env=os.environ.copy();env['PYTHONPATH']=str(root)+os.pathsep+env.get('PYTHONPATH','');env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        command=[sys.executable,str(Path(root)/'ops/mac_lgbm_r5.py'),'verify','--context',str(context),'--selection',str(selection),'--output',str(model),'--receipt',str(model/'producer_summary.json'),'--expected-config-sha',expected_config_sha]
        result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,check=True)
        return json.loads(result.stdout)
    def aggregate(self,parts):
        from pipeline.metrics import evaluate_regression
        return evaluate_regression(np.concatenate([p['truth'] for p in parts]),np.concatenate([p['share'] for p in parts]),np.concatenate([p['presence'] for p in parts]),np.concatenate([p['score'] for p in parts]),groups={'site':np.concatenate([p['site'] for p in parts]),'background_source':np.concatenate([p['background_group'] for p in parts]),'reference_combination':np.concatenate([p['reference_group'] for p in parts])},quality_flags={k:np.concatenate([p[k] for p in parts]) for k in ['boundary_artifact_candidate','urn_8321_nominal_warning']})

def validate_plan(path,config,project):
    plan=_json(path)
    if plan.get('format')!='r5_mac_executor_plan_v1' or plan.get('tree_executor')!='mac_spool_v1' or plan.get('user_approved') is not True or plan.get('server_tree_fit_allowed') is not False or plan.get('config_sha256')!=canonical_sha256(config):raise ValueError('Explicit approved strict Mac-only executor plan required')
    codes=plan.get('code_sha256',{})
    if not {'ops/runner_r5_mac.py','ops/mac_lgbm_r5.py'}.issubset(codes):raise ValueError('Controller/producer code pins required')
    verify_mapping(project,codes);return plan

def validate_handoff(path,legacy_run,new_run,config_hash,*,alive=process_alive):
    r=_json(path)
    if r.get('format')!='r5_controller_handoff_v1' or r.get('config_sha256')!=config_hash or r.get('old_driver_stopped_or_dead') is not True or r.get('old_driver_dead_confirmed') is not True or r.get('newcontroller_start_allowed') is not True or r.get('child_finished_exit0') is not True:raise ValueError('A finalized completed-child, explicitly stopped-controller handoff is required')
    if Path(r.get('legacy_run_path','')).resolve()!=Path(legacy_run).resolve() or Path(r.get('new_run_path','')).resolve()!=Path(new_run).resolve():raise ValueError('Handoff RUN paths differ')
    for name in ['legacy_driver_pid','current_child_pid']:
        if type(r.get(name)) is not int or r[name]<1:raise ValueError('Recorded positive legacy process identities required')
    if alive(r.get('legacy_driver_pid'),r.get('legacy_driver_start_ticks')):raise ValueError('Legacy controller is still alive; duplicate writers forbidden')
    if alive(r.get('current_child_pid'),r.get('current_child_start_ticks')):raise ValueError('Legacy child is still alive; no partial import/resume')
    return r

def completed_nn(folder,context,kind,observation,config,operations,project,*,fixture=False):
    summary=_json(_safe(folder,'summary.json'));binding=context['training_binding']
    if summary.get('status')!='completed' or summary.get('kind')!=kind or summary.get('fixture_only') is not fixture or summary.get('outer_or_Test_accessed') is not False or summary.get('training_binding')!=binding or summary.get('observation_policy')!=observation:raise ValueError('Only healthy completed matching NN children may be imported')
    if any(Path(folder).rglob('FAILED.json')):raise ValueError('Failed child cannot be imported')
    settings=dict(config['training'])
    if kind in {'timemoe_public','dasheng'}:settings.update(config['foundation_training_overrides'])
    if summary.get('settings')!=settings:raise ValueError('Legacy NN training settings differ')
    for name,key in [('best.pt','checkpoint_sha256'),('inner_predictions.npz','prediction_sha256')]:
        if file_sha256(_safe(folder,name))!=summary[key]:raise ValueError('Completed NN checkpoint/prediction SHA changed')
    for name in ['epochs.jsonl','dynamic_source_receipts.json']:_safe(folder,name)
    engine=object.__new__(FrozenInferenceR5);engine.root=Path(project);engine.snapshot={};engine.fixture_mode=fixture
    checkpoint=operations.checkpoint(Path(folder)/'best.pt')
    engine._inspect_checkpoint(checkpoint,kind,binding,context['prior']['payload_sha256'],observation,summary)
    engine._dynamic_receipt(checkpoint,summary,Path(folder),binding)
    hashes={p.name:file_sha256(p) for p in Path(folder).iterdir() if p.is_file() and p.suffix in {'.pt','.json','.jsonl','.npz'}}
    if any(p.is_symlink() for p in Path(folder).iterdir()):raise ValueError('Symlink in completed NN child')
    return summary,hashes

def import_legacy(source_fold,target,relative,context,kind,observation,config,handoff,operations,project,*,fixture=False):
    source=Path(source_fold)/relative
    if not source.exists():return False
    if not (source/'summary.json').exists():raise ValueError('Matching legacy child exists but is incomplete; cannot retrain or resume it silently')
    summary_sha=file_sha256(_safe(source,'summary.json'));cert=handoff.get('completed_models',{}).get(relative)
    certified=cert and cert.get('exit_code')==0 and cert.get('summary_sha256')==summary_sha
    if not certified and summary_sha!=handoff.get('complete_summary_sha256'):raise ValueError('Completed legacy child lacks explicit exit0 import certification')
    summary,files=completed_nn(source,context,kind,observation,config,operations,project,fixture=fixture)
    if target.exists():raise FileExistsError('Legacy import target must be fresh')
    target.mkdir(parents=True)
    for name,h in files.items():
        shutil.copyfile(_safe(source,name),target/name)
        if file_sha256(target/name)!=h:raise ValueError('Legacy import byte mismatch')
    verify_mapping(source,files)
    write_new(target/'legacy_import_receipt.json',{'format':'r5_completed_NN_import_v1','source_path':str(source),'source_summary_sha256':summary_sha,'training_binding':context['training_binding'],'prior_sha256':context['prior']['payload_sha256'],'observation_policy':observation,'artifact_sha256':files,'optimizer_scheduler_RNG_state_resumed':False,'completed_weights_only_cold_import':True,'source_parent_run_claimed_completed':False})
    return True

def unpack_mac(job,request,stage):
    completion=_json(_safe(job,'completion.json'))
    if completion.get('format')!='r5_mac_tree_completion_v1' or completion.get('job_id')!=request['job_id'] or completion.get('request_sha256')!=request['request_sha256']:raise ValueError('Mac completion job/request binding mismatch')
    archive=_safe(job,'staging/artifacts.zip')
    if file_sha256(archive)!=completion.get('archive_sha256'):raise ValueError('Mac archive SHA mismatch')
    files=completion.get('artifact_sha256',{})
    if set(files)!=TREE_FILES:raise ValueError('Exactly core six files plus producer receipt required')
    stage.mkdir()
    try:
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None or set(z.namelist())!=TREE_FILES or len(z.namelist())!=len(TREE_FILES):raise ValueError('Mac ZIP entries/CRC invalid')
            for name in TREE_FILES:
                info=z.getinfo(name)
                if PurePosixPath(name).name!=name or ((info.external_attr>>16)&0o170000)==0o120000:raise ValueError('Mac ZIP path/symlink forbidden')
                with z.open(name) as src,(stage/name).open('xb') as dst:shutil.copyfileobj(src,dst)
                if file_sha256(stage/name)!=files[name]:raise ValueError('Mac artifact SHA mismatch')
        if file_sha256(stage/'producer_summary.json')!=completion.get('producer_receipt_sha256'):raise ValueError('Mac receipt SHA mismatch')
        if _json(stage/'producer_summary.json').get('producer_environment')!=completion.get('producer_environment'):raise ValueError('Completion/producer environment receipt differs')
    except BaseException:shutil.rmtree(stage);raise
    return completion

def wait_mac(job,request,context,selection,target,operations,project,*,wait_seconds=15):
    job.mkdir(parents=True,exist_ok=False);write_new(job/'request.json',request);request_file_sha=file_sha256(job/'request.json')
    if not getattr(operations,'fixture_backend',False):print(json.dumps({'event':'waiting_for_Mac_CPU_tree','job_id':request['job_id'],'request':str(job/'request.json'),'server_tree_fit':False}),flush=True)
    while not (job/'completion.json').exists():
        if file_sha256(job/'request.json')!=request_file_sha:raise ValueError('Immutable Mac request changed while waiting')
        if (job/'FAILED.json').exists():raise RuntimeError('Mac worker failure: '+json.dumps(_json(job/'FAILED.json')))
        time.sleep(wait_seconds)
    if (job/'FAILED.json').exists():raise RuntimeError('A failed Mac job cannot later be reused')
    completion_file_sha=file_sha256(_safe(job,'completion.json'))
    if target.exists():raise FileExistsError('Tree output must be fresh; server never fits/reuses partial outputs')
    stage=target.with_name(target.name+'.verified_stage')
    if stage.exists():raise FileExistsError('Fresh tree verification stage required')
    target.parent.mkdir(parents=True,exist_ok=True);completion=unpack_mac(job,request,stage)
    verified=operations.verify_mac(context,selection,stage,request['config_sha256'],project)
    if verified.get('status')!='verified' or verified.get('fit_performed') is not False or verified.get('OUTER_or_Test_opened') is not False:raise ValueError('Mac native INNER replay verification failed')
    if file_sha256(job/'request.json')!=request_file_sha:raise ValueError('Mac request changed during verification')
    if file_sha256(job/'completion.json')!=completion_file_sha:raise ValueError('Mac completion changed during verification')
    verify_mapping(stage,completion['artifact_sha256'])
    write_new(stage/'executor_acceptance.json',{'format':'r5_mac_tree_acceptance_v1','request_sha256':request['request_sha256'],'completion':completion,'verification':verified,'server_fit_executed':False});stage.rename(target)
    return {'request_sha256':request['request_sha256'],'completion_sha256':file_sha256(job/'completion.json'),'verification':verified}

class Controller:
    def __init__(self,config,output,plan_path,spool_root,*,project=ROOT,legacy_run=None,handoff_path=None,operations=None,fixture=False):
        if operations is not None and not fixture:raise ValueError('Synthetic operation injection only')
        self.ops=operations or NativeOperations();self.fixture=fixture
        if fixture and getattr(self.ops,'fixture_backend',False) is not True:raise ValueError('Explicit synthetic backend required')
        self.project=Path(project).resolve();self.config=config;self.output=Path(output).resolve();self.spool=Path(spool_root).resolve();self.plan_path=Path(plan_path);self.plan=validate_plan(plan_path,config,self.project);self.plan_sha=file_sha256(plan_path)
        self.legacy=Path(legacy_run).resolve() if legacy_run else None;self.handoff=None
        if self.legacy and self.output.is_relative_to(self.legacy):raise ValueError('New RUN must remain outside immutable legacy RUN')
        if self.legacy:
            if handoff_path is None:raise ValueError('Explicit handoff receipt required for legacy import')
            self.handoff_path=Path(handoff_path);self.handoff=validate_handoff(handoff_path,self.legacy,self.output,canonical_sha256(config),alive=(lambda *a:False) if fixture else process_alive);self.handoff_sha=file_sha256(handoff_path)
        elif handoff_path is not None:raise ValueError('Handoff receipt without legacy RUN')
        if self.output.exists():raise FileExistsError('Fresh RUN only, no partial reuse')
        self.codes={name:file_sha256(_safe(self.project,name)) for name in REQUIRED_CODE|TREE_EXTRA}
        self.output.mkdir(parents=True);write_new(self.output/'executor_plan.json',self.plan)
        if self.handoff:write_new(self.output/'controller_handoff_receipt.json',self.handoff)

    def verify(self):
        verify_mapping(self.project,self.codes);verify_mapping(self.project,self.plan['code_sha256'])
        if file_sha256(self.plan_path)!=self.plan_sha:raise ValueError('Executor plan changed')
        if self.handoff and file_sha256(self.handoff_path)!=self.handoff_sha:raise ValueError('Immutable handoff receipt changed')

    def fit_fold(self,number):
        cfg=self.config;folder=self.project/Path(cfg['dataset'])/f'fold_{number:02d}';f=self.ops.fold(folder,cfg['seed']);out=self.output/f'fold_{number:02d}'
        if out.exists():raise FileExistsError('Fresh fold only')
        out.mkdir();source_fold=self.legacy/f'fold_{number:02d}' if self.legacy else None;legacy_context=None
        try:
            inner,inner_meta=f.load('inner')
            if source_fold and source_fold.exists():
                legacy_context=_json(_safe(source_fold,'context.json'))
                if legacy_context.get('config')!=cfg or legacy_context.get('config_sha256')!=canonical_sha256(cfg) or legacy_context.get('training_binding')!=f.binding or legacy_context.get('code_sha256')!={k:self.codes[k] for k in REQUIRED_CODE}:raise ValueError('Legacy config/source/package/seed/core pins differ')
                r=legacy_context['prior'];p=_safe(source_fold,r['filename'])
                if not self.fixture:
                    from pipeline.grid_inputs_r5 import load_shared_prior
                    load_shared_prior(p,expected_binding=f.binding,expected_file_sha256=r['file_sha256'],expected_prior_sha256=r['payload_sha256'])
                if file_sha256(p)!=r['file_sha256']:raise ValueError('Legacy TRAIN prior changed')
                shutil.copyfile(p,out/r['filename']);prior=r
            else:
                train,_=f.load('train');prior=self.ops.prior(train,f.binding,out/'shared_background_prior.json');del train
            context={'format':'r5_fold_context_v1','data_fold':str(Path(cfg['dataset'])/f'fold_{number:02d}'),'training_binding':f.binding,'config':cfg,'config_sha256':canonical_sha256(cfg),'prior':prior,'code_sha256':{k:self.codes[k] for k in REQUIRED_CODE}}
            context_path=out/'context.json';write_new(context_path,context);paths={};pilots=[];imports=[]
            def fit_nn(kind,observation,relative):
                target=out/relative;used=False
                if legacy_context:used=import_legacy(source_fold,target,relative,legacy_context,kind,cfg['observation_candidates'][observation],cfg,self.handoff,self.ops,self.project,fixture=self.fixture)
                if not used:self.ops.nn_fit(context_path,kind,observation,target)
                self.verify();completed_nn(target,context,kind,cfg['observation_candidates'][observation],cfg,self.ops,self.project,fixture=self.fixture)
                if used:imports.append(relative)
                return target
            for observation in cfg['observation_candidates']:
                rel=f'preprocessing_pilot/{observation}/bg_cnn';path=fit_nn('bg_cnn',observation,rel);s=_json(path/'summary.json');pilots.append({'observation':observation,'inner_rmse':s['best_inner_rmse'],'model_path':rel,'summary_sha256':file_sha256(path/'summary.json')})
            winner=min(pilots,key=lambda p:p['inner_rmse']);observation=winner['observation'];paths['bg_cnn']=out/winner['model_path'];selection_path=out/'preprocessing_selection.json'
            write_new(selection_path,{'fit_split':'inner','candidates':pilots,'selected':observation,'outer_or_Test_used':False,'training_binding':f.binding})
            mac_acceptance=None
            for kind in [k for k in cfg['core_models']+cfg['comparisons'] if k!='bg_cnn']:
                if kind!='lgbm_physics':paths[kind]=fit_nn(kind,observation,'models/'+kind);continue
                target=out/'models/lgbm_physics';job=self.spool/f'fold_{number:02d}'
                request={'format':'r5_mac_tree_request_v1','job_id':self.output.name+f'.fold{number:02d}','config_sha256':canonical_sha256(cfg),'training_binding':f.binding,'observation_name':observation,'observation_policy':cfg['observation_candidates'][observation],
                    'context_sha256':file_sha256(context_path),'selection_sha256':file_sha256(selection_path),'context_source_path':str(context_path),'selection_source_path':str(selection_path),'cnn_summaries':[{'relative_path':p['model_path']+'/summary.json','sha256':p['summary_sha256']} for p in pilots],'prior_sha256':prior['file_sha256'],'expected_core19_sha256':self.codes,'tree_settings':cfg['tree'],'output_model_path':str(target),'executor_plan_sha256':self.plan_sha}
                request['request_sha256']=canonical_sha256(request);mac_acceptance=wait_mac(job,request,context_path,selection_path,target,self.ops,self.project,wait_seconds=.001 if self.fixture else 15);paths[kind]=target;self.verify()
            candidates={k:self.ops.read_predictions(p/'inner_predictions.npz') for k,p in paths.items()};w=self.ops.weights(inner['db'])
            policy=select_inner_policy(inner['share'],{k:p['share'] for k,p in candidates.items()},inner_meta.reference_group.to_numpy(),fit_split='inner',binding=f.binding,explicit_exclusions={},enable_logit_mask_candidate=cfg['ensemble']['enable_logit_mask_candidate'],neural_masks={k:p['cell_mask'] for k,p in candidates.items() if 'cell_mask' in p},energy_weights=w)
            policy_path=out/'frozen_ensemble_policy.json';write_new(policy_path,policy);policy_sha=file_sha256(policy_path)
            ip=self.ops.ensemble(policy,candidates,w,f.binding,'inner');im={k:self.ops.metrics(inner,inner_meta,p) for k,p in candidates.items()};im['ensemble']=self.ops.metrics(inner,inner_meta,ip);write_new(out/'inner_metrics.json',im)
            if file_sha256(policy_path)!=policy_sha:raise ValueError('Policy changed before OUTER gate')
            outer,outer_meta=f.load('outer',frozen_policy=policy);op={}
            for kind,path in paths.items():
                destination=out/'outer_predictions'/f'{kind}.npz';self.ops.predict(context_path,kind,observation,path,policy_path,destination);self.verify();op[kind]=self.ops.read_predictions(destination)
            ow=self.ops.weights(outer['db']);ep=self.ops.ensemble(policy,op,ow,f.binding,'outer');om={k:self.ops.metrics(outer,outer_meta,p) for k,p in op.items()};om['ensemble']=self.ops.metrics(outer,outer_meta,ep);write_new(out/'outer_metrics.json',om)
            np.savez_compressed(out/'outer_oof.npz',**ep,truth=outer['share'],presence=outer['presence'],sample_id=outer_meta.sample_id.to_numpy(dtype=str),**{k:outer_meta[k].to_numpy(dtype=bool if k in {'boundary_artifact_candidate','urn_8321_nominal_warning'} else str) for k in ['boundary_artifact_candidate','urn_8321_nominal_warning','site','background_group','reference_group']})
            write_new(out/'executor_provenance.json',{'format':FORMAT,'tree_executor':'mac_spool_v1','server_tree_fit_executed':False,'Mac_acceptance':mac_acceptance,'legacy_imports':imports,'legacy_parent_run_claimed_completed':False,'executor_plan_sha256':self.plan_sha,'config_unchanged_from_legacy':legacy_context is not None,'training_binding':f.binding,'fixture_only':self.fixture})
            summary={'status':'completed','fold':number,'training_binding':f.binding,'selected_observation':observation,'models':{k:str(p.relative_to(out)) for k,p in paths.items()},'frozen_policy_sha256':policy_sha,'context_sha256':file_sha256(context_path),'inner_rmse':im['ensemble']['rmse'],'outer_rmse':om['ensemble']['rmse'],'outer_rmse_excluding_boundary':om['ensemble'].get('oof_rmse_excluding_boundary'),'outer_changed_policy':False,'Test_accessed':False,'official_submission':False,'executor_variant':FORMAT,'fixture_only':self.fixture}
            summary['artifact_sha256']={p.relative_to(out).as_posix():file_sha256(p) for p in out.rglob('*') if p.is_file()};write_new(out/'summary.json',summary);return summary
        except BaseException as e:
            write_new(out/'FAILED.json',{'status':'failed','type':type(e).__name__,'error':str(e),'partial_run_reusable':False,'server_tree_fit_executed':False});raise

    def fit(self,folds):
        if not folds or len(set(folds))!=len(folds) or any(type(i) is not int or i not in range(5) for i in folds):raise ValueError('Explicit unique folds0..4 required')
        summaries=[]
        for i in folds:
            summaries.append(self.fit_fold(i));atomic(self.output/'run_state.json',{'status':'running','config_sha256':canonical_sha256(self.config),'completed_folds':summaries,'executor_variant':FORMAT})
        state={'status':'completed' if len(folds)==5 else 'completed_requested_folds','config_sha256':canonical_sha256(self.config),'folds':summaries,'outer_selected_any_policy':False,'Test_accessed':False,'official_submission':False,'executor_variant':FORMAT,'fixture_only':self.fixture}
        if len(folds)==5:state['pooled_oof']=self.ops.aggregate([self.ops.read_predictions(self.output/f'fold_{i:02d}/outer_oof.npz') for i in folds])
        atomic(self.output/'run_state.json',state);return state

def self_test():
    """Five-fold synthetic orchestration; mock neural/Mac workers, no fit/GPU."""
    import threading
    from tempfile import TemporaryDirectory
    from pipeline.inference_r5 import original_energy_weights
    from pipeline.hierarchical_ensemble_r5 import apply_policy
    class Column:
        def __init__(self,a):self.a=np.asarray(a)
        def astype(self,dtype):return self.a.astype(dtype)
        def to_numpy(self,dtype=None):return self.a.astype(dtype) if dtype is not None else self.a.copy()
    class Metadata:
        def __init__(self,n,fold,role):
            self.cols={k:Column(v) for k,v in {'sample_id':[f'f{fold}_{role}_{i}' for i in range(n)],'reference_group':np.resize(['rA','rB','rC'],n),'site':np.resize(['sA','sB'],n),'background_group':np.resize(['bA','bB'],n),'boundary_artifact_candidate':np.zeros(n,bool),'urn_8321_nominal_warning':np.zeros(n,bool)}.items()}
        def __getattr__(self,k):return self.cols[k]
        def __getitem__(self,k):return self.cols[k]
    with TemporaryDirectory(prefix='r5_mac_controller_fixture_') as tmp:
        base=Path(tmp);project=base/'project';project.mkdir();output=project/'models/new';legacy=project/'models/legacy';source=legacy/'fold_00';source.mkdir(parents=True);spool=base/'spool'
        for name in REQUIRED_CODE|TREE_EXTRA:
            p=project/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# synthetic source identity, never executed\n')
        for name in ['ops/runner_r5_mac.py','ops/mac_lgbm_r5.py']:
            p=project/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# synthetic executor pin, never executed\n')
        codes={k:file_sha256(project/k) for k in REQUIRED_CODE};ops_codes={k:file_sha256(project/k) for k in ['ops/runner_r5_mac.py','ops/mac_lgbm_r5.py']}
        cfg={'version':'r5_training_integration_v1','protocol':'source_group_nested5_vtuad_passage_r4','dataset':'data/synthetic/fixture','seed':42,
            'core_models':['bg_cnn','tfgrid_fdy_conformer','lgbm_physics','timemoe_public','dasheng'],'comparisons':['fdy','itransformer'],
            'model_settings':{k:{} for k in NEURAL},'observation_candidates':{'raw':{'quantization_decimals':None,'endpoint_dropout_probability':0.},'resolution_01':{'quantization_decimals':1,'endpoint_dropout_probability':0.},'resolution_01_endpoint':{'quantization_decimals':1,'endpoint_dropout_probability':.2}},
            'training':{'epochs':100,'patience':7,'optimizer':'adam','evaluation_batch':128},'foundation_training_overrides':{'evaluation_batch':2},'tree':{'maximum_trees':1500,'patience':40,'threads':4},'ensemble':{'enable_logit_mask_candidate':True,'ridge':.05,'epsilon_rmse':.0001}}
        def binding(i):return {'fold':i,'seed':42,'train_examples':48,'source_hash':'a'*64,'train_npz_sha256':'b'*64,'data_package_hash':canonical_sha256({'fixture_fold':i})}
        def arrays(n):
            y=np.linspace(.1,.8,n).astype('float32');y[::5]=0;db=np.full((n,15,25),40,dtype='float32');q=np.broadcast_to(y[:,None,None],db.shape).copy()
            return {'db':db,'share':y,'presence':(y>0).astype('float32'),'cell_share':q}
        data={r:arrays(n) for r,n in [('train',48),('inner',48),('outer',24)]}
        def prediction(role,kind,observation):
            delta={'bg_cnn':{'raw':.04,'resolution_01':.02,'resolution_01_endpoint':.03}[observation],'tfgrid_fdy_conformer':.03,'timemoe_public':.008,'dasheng':-.008,'fdy':.025,'itransformer':-.025,'lgbm_physics':-.02}[kind]
            q=np.broadcast_to(np.clip(data[role]['share']+delta,0,1)[:,None,None],data[role]['db'].shape).copy();w=original_energy_weights(data[role]['db']);p={'share':(q*w).sum((1,2)),'score':np.clip(data[role]['presence']*.8+.1,0,1)}
            if kind!='lgbm_physics':p['cell_mask']=q
            return p
        def prior_record(folder,b):
            p={'fit_split':'train','training_binding':b};p['shared_prior_payload_sha256']=canonical_sha256(p);write_new(folder/'shared_background_prior.json',p)
            return {'filename':'shared_background_prior.json','file_sha256':file_sha256(folder/'shared_background_prior.json'),'payload_sha256':p['shared_prior_payload_sha256']}
        original_context={'format':'r5_fold_context_v1','data_fold':'data/synthetic/fixture/fold_00','training_binding':binding(0),'config':cfg,'config_sha256':canonical_sha256(cfg),'prior':prior_record(source,binding(0)),'code_sha256':codes};write_new(source/'context.json',original_context)
        def nn_files(path,context,kind,observation):
            path.mkdir(parents=True);pred=prediction('inner',kind,observation);np.savez_compressed(path/'inner_predictions.npz',**pred)
            settings=dict(cfg['training'])
            if kind in {'timemoe_public','dasheng'}:settings.update(cfg['foundation_training_overrides'])
            b=context['training_binding'];identity={'binding':b,'synthetic_fixture':True};receipt={'epoch':0,'examples':48,'clean_primary_once':True,'heldout_arrays_opened':False,'source_identity_sha256':canonical_sha256(identity)}
            weights={'adapter_only':False,'state_dict':{}}
            if kind in {'timemoe_public','dasheng'}:weights={'adapter_only':True,'adapter':{'manifest':{'family':'dasheng_1.2b' if kind=='dasheng' else 'timemoe_public_200m_active_453m_total','fixture_mode':True,'pretrained_claim':False},'state':{}}}
            checkpoint={'format':'r5_neural_checkpoint_v1','kind':kind,'training_binding':b,'shared_prior_sha256':context['prior']['payload_sha256'],'observation_policy':cfg['observation_candidates'][observation],'training_settings':settings,'epoch':0,'model_code_sha256':codes,'weights':weights,'dynamic_source_identity':identity,'best_epoch_source_receipt':receipt}
            write_new(path/'best.pt',checkpoint);write_new(path/'dynamic_source_receipts.json',{'source_identity':identity,'completed_epochs':{'0':receipt}});(path/'epochs.jsonl').write_text('{"epoch":0,"inner_rmse":0.04}\n')
            write_new(path/'summary.json',{'status':'completed','kind':kind,'fixture_only':True,'outer_or_Test_accessed':False,'training_binding':b,'observation_policy':cfg['observation_candidates'][observation],'settings':settings,'best_epoch':0,'best_inner_rmse':float(np.sqrt(np.mean((pred['share']-data['inner']['share'])**2))),'checkpoint_sha256':file_sha256(path/'best.pt'),'prediction_sha256':file_sha256(path/'inner_predictions.npz'),'dynamic_source_identity':identity,'dynamic_source_receipts_file':'dynamic_source_receipts.json'})
        raw='preprocessing_pilot/raw/bg_cnn';nn_files(source/raw,original_context,'bg_cnn','raw')
        before={str(p):file_sha256(p) for p in source.rglob('*') if p.is_file()}
        handoff=base/'handoff.json';write_new(handoff,{'format':'r5_controller_handoff_v1','legacy_run_path':str(legacy),'new_run_path':str(output),'config_sha256':canonical_sha256(cfg),'old_driver_stopped_or_dead':True,'old_driver_dead_confirmed':True,'newcontroller_start_allowed':True,'child_finished_exit0':True,'legacy_driver_pid':44444,'current_child_pid':55555,'complete_summary_sha256':file_sha256(source/raw/'summary.json')})
        proc=base/'fake_proc/77777';proc.mkdir(parents=True);(proc/'stat').write_text('77777 (fixture) '+' '.join(['Z']+['0']*18+['123']))
        assert process_alive(77777,'123',_proc_root=proc.parent) is False
        try:process_alive(77777,'999',_proc_root=proc.parent)
        except ValueError:pass
        else:raise AssertionError('Foreign reused PID accepted')
        plan=base/'plan.json';write_new(plan,{'format':'r5_mac_executor_plan_v1','tree_executor':'mac_spool_v1','user_approved':True,'server_tree_fit_allowed':False,'config_sha256':canonical_sha256(cfg),'code_sha256':ops_codes})
        calls=[];outer_opened=[];worker_errors=[];mac_jobs=[]
        class FakeFold:
            def __init__(self,i):self.i=i;self.binding=binding(i)
            def load(self,role,*,frozen_policy=None):
                if role=='outer':
                    assert (output/f'fold_{self.i:02d}/frozen_ensemble_policy.json').is_file() and frozen_policy['frozen_before_outer'];outer_opened.append(self.i)
                return data[role],Metadata(len(data[role]['share']),self.i,role)
        class FakeOperations(NativeOperations):
            fixture_backend=True
            def fold(self,path,seed):return FakeFold(int(Path(path).name[-2:]))
            def prior(self,train,b,path):return prior_record(path.parent,b)
            def checkpoint(self,path):return _json(path)
            def nn_fit(self,context,kind,observation,target):
                assert kind in NEURAL;calls.append((kind,observation));nn_files(target,_json(context),kind,observation)
            def predict(self,context,kind,observation,model,policy,destination):
                destination.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(destination,**prediction('outer',kind,observation))
            def weights(self,db):return original_energy_weights(db)
            def metrics(self,a,m,p):return {'rmse':float(np.sqrt(np.mean((a['share']-p['share'])**2)))}
            def ensemble(self,p,branches,w,b,role):
                share=apply_policy(p,{k:v['share'] for k,v in branches.items()},evaluation_split=role,binding=b,neural_masks={k:v['cell_mask'] for k,v in branches.items() if 'cell_mask' in v},energy_weights=w)
                return {'share':share.astype('float32'),'score':sum(v*branches[k]['score'] for k,v in p['weights'].items()).astype('float32')}
            def verify_mac(self,context,selection,stage,expected_config_sha,root):
                r=_json(stage/'producer_summary.json');assert r['format']=='mac_lgbm_r5_producer_v1' and r['config_sha256']==expected_config_sha
                return {'status':'verified','fit_performed':False,'OUTER_or_Test_opened':False,'scope':'synthetic mocked verifier, not a production LightGBM verification'}
            def aggregate(self,parts):return {'rmse':float(np.sqrt(np.mean((np.concatenate([p['truth'] for p in parts])-np.concatenate([p['share'] for p in parts]))**2)))}
        operations=FakeOperations()
        def worker():
            try:
                for i in range(5):
                    job=spool/f'fold_{i:02d}';deadline=time.monotonic()+15
                    while not (job/'request.json').exists():
                        if time.monotonic()>deadline:raise RuntimeError('Synthetic request timeout')
                        time.sleep(.001)
                    # Exclusive file exists before its writer finishes the small JSON.
                    while True:
                        try:r=_json(job/'request.json');break
                        except ValueError:time.sleep(.001)
                    assert r['request_sha256']==canonical_sha256({k:v for k,v in r.items() if k!='request_sha256'}) and r['executor_plan_sha256']==file_sha256(plan)
                    mac_jobs.append(r['job_id']);tree=base/f'mac_{i}';tree.mkdir();np.savez_compressed(tree/'inner_predictions.npz',**prediction('inner','lgbm_physics',r['observation_name']))
                    for name in ['share.txt','presence.txt']: (tree/name).write_text('synthetic Mac artifact, not a trained model\n')
                    for name in ['prototypes.npz','physics_pca.json']:(tree/name).write_bytes(b'synthetic fixture bytes')
                    write_new(tree/'summary.json',{'status':'completed','kind':'lgbm_physics','fixture_only':True})
                    env={'platform':'synthetic fixture'};write_new(tree/'producer_summary.json',{'format':'mac_lgbm_r5_producer_v1','status':'completed','fixture_only':True,'config_sha256':r['config_sha256'],'producer_environment':env})
                    staging=job/'staging';staging.mkdir();archive=staging/'artifacts.zip'
                    with zipfile.ZipFile(archive,'w') as z:
                        for name in TREE_FILES:z.write(tree/name,name)
                    write_new(job/'completion.json',{'format':'r5_mac_tree_completion_v1','job_id':r['job_id'],'request_sha256':r['request_sha256'],'archive_sha256':file_sha256(archive),'producer_receipt_sha256':file_sha256(tree/'producer_summary.json'),'artifact_sha256':{n:file_sha256(tree/n) for n in TREE_FILES},'producer_environment':env})
            except BaseException as e:
                worker_errors.append(str(e))
                if 'job' in locals() and job.exists() and not (job/'FAILED.json').exists():write_new(job/'FAILED.json',{'error':str(e)})
        thread=threading.Thread(target=worker,daemon=True);thread.start();c=Controller(cfg,output,plan,spool,project=project,legacy_run=legacy,handoff_path=handoff,operations=operations,fixture=True);state=c.fit([0,1,2,3,4]);thread.join(2)
        assert not worker_errors and len(mac_jobs)==5 and state['status']=='completed' and outer_opened==list(range(5))
        assert calls.count(('bg_cnn','raw'))==4 and len(calls)==39 and all(kind!='lgbm_physics' for kind,_ in calls)
        assert all(file_sha256(p)==h for p,h in before.items()) and _json(output/'executor_plan.json')['server_tree_fit_allowed'] is False
        rejected=[]
        try:validate_handoff(handoff,legacy,output,canonical_sha256(cfg),alive=lambda *a:True)
        except ValueError:rejected.append('live_legacy_controller')
        else:raise AssertionError('Live legacy writer accepted')
        target=base/'partial_target';partial=source/'partial/bg_cnn';partial.mkdir(parents=True)
        try:import_legacy(source,target,'partial/bg_cnn',original_context,'bg_cnn',cfg['observation_candidates']['raw'],cfg,_json(handoff),operations,project,fixture=True)
        except ValueError:rejected.append('partial_child_no_resume')
        else:raise AssertionError('Partial NN accepted')
        p=spool/'fold_00/completion.json';old=p.read_bytes();bad=_json(p);bad['request_sha256']='0'*64;p.write_text(json.dumps(bad))
        try:unpack_mac(spool/'fold_00',_json(spool/'fold_00/request.json'),base/'bad_stage')
        except ValueError:rejected.append('wrong_Mac_request_binding')
        else:raise AssertionError('Wrong completion accepted')
        p.write_bytes(old)
        # The real NativeOperations path isolates irreversible verifier hooks.
        verifier=project/'ops/mac_lgbm_r5.py';old_verifier=verifier.read_bytes();verifier.write_text("import sys,json\ndef audit(event,args):\n if event=='subprocess.Popen': raise RuntimeError('child-only guard')\nsys.addaudithook(audit)\nprint(json.dumps({'status':'verified','fit_performed':False,'OUTER_or_Test_opened':False}))\n")
        isolated=NativeOperations().verify_mac(output/'fold_00/context.json',output/'fold_00/preprocessing_selection.json',output/'fold_00/models/lgbm_physics',canonical_sha256(cfg),project)
        assert isolated['status']=='verified'
        subprocess.run([sys.executable,'-c','pass'],check=True)
        verifier.write_bytes(old_verifier)
        try:Controller(cfg,output,plan,spool,project=project,operations=operations,fixture=True)
        except FileExistsError:rejected.append('partial_or_existing_run_reuse')
        else:raise AssertionError('Existing RUN accepted')
        return {'passed':True,'scope':'fivefold SYNTHETIC protocol fixture only; mock NN and mock Mac worker/verifier; no performance claim','fixture_only':True,'server_LightGBM_fit_calls':0,'Mac_mock_jobs':5,'NN_mock_fit_calls':39,'completed_NN_imports':1,'source_files_unchanged':True,'matching_zombie_inactive_foreign_reused_PID_rejected':True,'native_verifier_audit_hook_subprocess_isolation_checked':True,'outer_opened_only_after_new_INNER_policy':outer_opened,'guards_rejected':rejected,'real_TRAIN_INNER_OUTER_Test_GPU_SSH_accessed':False,'code_sha256':{'ops/runner_r5_mac.py':file_sha256(__file__)}}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path);p.add_argument('--output',type=Path);p.add_argument('--preflight-report',type=Path);p.add_argument('--executor-plan',type=Path);p.add_argument('--spool-root',type=Path);p.add_argument('--legacy-run',type=Path);p.add_argument('--handoff-receipt',type=Path);p.add_argument('--folds',type=int,nargs='+');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:print(json.dumps(self_test(),indent=2,allow_nan=False));return
    if any(getattr(a,n) is None for n in ['config','output','preflight_report','executor_plan','spool_root','folds']):p.error('Explicit config/output/preflight/executor-plan/spool-root/folds required')
    from pipeline.runner_r5 import validate_config
    import torch
    cfg=_json(a.config);validate_config(cfg);proof=_json(a.preflight_report)
    if not torch.cuda.is_available():raise RuntimeError('Explicit CUDA host required; no CPU production fallback')
    if proof.get('ready_for_gpu_pilot') is not True or proof.get('config_sha256')!=canonical_sha256(cfg) or set(proof.get('code_sha256',{}))!=REQUIRED_CODE:raise ValueError('Matching complete fresh core preflight required')
    verify_mapping(ROOT,proof['code_sha256'])
    for name,value in proof.get('fixture_reports',{}).items():
        if file_sha256(_safe(ROOT/'reports',name))!=value['sha256']:raise ValueError('Preflight fixture receipt changed')
    c=Controller(cfg,a.output,a.executor_plan,a.spool_root,legacy_run=a.legacy_run,handoff_path=a.handoff_receipt);print(json.dumps(c.fit(a.folds),indent=2,allow_nan=False))

if __name__=='__main__':main()
