"""Fresh, explicitly selected Mac LightGBM variant; original R5 is immutable.

Production fit requires all three original CNN summaries and their frozen INNER
selection. Only TRAIN/INNER data are opened. No SSH, model transfer, OUTER,
ensemble selection, GPU use, or modification of the original runner occurs.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib.metadata,json,math,os,sys,tempfile,time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
FORMAT='mac_lgbm_r5_producer_v1'
REPLAY_ATOL=1e-6
# The other five tree files are already pinned by the original17 context.
# These two release-verified helpers close the remaining19-file dependency set.
EXTRA_TREE_SHA256={
    'pipeline/train_lgbm.py':'704b77a00c89c636ada7881facb6deaf36afceaec35e7c4742b75478e650e42d',
    'pipeline/prototypes.py':'6349bd51e42b2c68986d44148644f63c9cfea5eae42a117dc86c5ca6da7b481a'}

def canonical_sha256(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def array_sha256(a):
    import numpy as np
    a=np.asarray(a);h=hashlib.sha256(json.dumps({'shape':list(a.shape),'dtype':str(a.dtype)},sort_keys=True,separators=(',',':')).encode())
    h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()

def inner_identity(arrays,metadata):
    from pipeline.observation_augmentation_r5 import prepare_query_observations
    w=prepare_query_observations(arrays['db'],quantization_decimals=None)['energy_weights']
    return {'examples':len(arrays['db']),
        'sample_id_order_sha256':canonical_sha256(metadata.sample_id.astype(str).tolist()),
        'raw_db_sha256':array_sha256(arrays['db']),'raw_energy_weights_sha256':array_sha256(w),
        'share_sha256':array_sha256(arrays['share']),'presence_sha256':array_sha256(arrays['presence']),
        'cell_share_sha256':array_sha256(arrays['cell_share'])}

def environment():
    import lightgbm
    versions={}
    for name in ['numpy','scipy','scikit-learn','lightgbm']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    versions['lightgbm']=lightgbm.__version__
    library=Path(lightgbm.basic._LIB._name)
    u=os.uname()
    return {'python':sys.version,'python_executable':sys.executable,'packages':versions,
        'system':u.sysname,'release':u.release,'machine':u.machine,'logical_CPUs':os.cpu_count(),
        'native_library':str(library),'native_library_sha256':sha(library),
        'model_threads':4,'thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}}

def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def _guard(folder,root):
    data=(Path(root)/'data').resolve()
    allowed={(folder/n).resolve() for n in ['readiness_manifest.json','train.npz','validation.npz',
        'train_sample_metadata.parquet','inner_validation_metadata.parquet',
        'sources/manifest.json','sources/background_manifest.parquet','sources/reference_manifest.parquet']}
    def check(value):
        p=Path(os.fsdecode(value)).resolve()
        if p.is_relative_to(data) and p not in allowed:raise ValueError('TRAIN/INNER-only IO guard: '+str(p))
        if p.suffix.lower() in {'.wav','.flac','.mp3'}:raise ValueError('External audio is forbidden')
        return p
    def audit(event,args):
        if event in {'socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'}:raise RuntimeError('Network/subprocess forbidden in Mac tree job')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=check(args[0])
            if p.is_relative_to(data) and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):raise ValueError('Dataset writes forbidden')
    sys.addaudithook(audit)
    import numpy as np,pandas as pd
    nl,pr=np.load,pd.read_parquet
    def load(p,*a,**k):check(p);return nl(p,*a,**k)
    def parquet(p,*a,**k):check(p);return pr(p,*a,**k)
    np.load,pd.read_parquet=load,parquet

def validate_inputs(context_path,selection_path,*,expected_config_sha,root=ROOT,fixture=False):
    from pipeline.runner_r5 import CODE_NAMES as RUNNER_NAMES,validate_config
    from pipeline.lgbm_r5 import CODE_NAMES as TREE_NAMES
    from pipeline.fold_data_r5 import FoldDataR5
    from pipeline.grid_inputs_r5 import load_shared_prior
    context_path=Path(context_path).resolve();selection_path=Path(selection_path).resolve();root=Path(root).resolve()
    if context_path.parent!=selection_path.parent:raise ValueError('Keep original context/selection copies in the same immutable inbox')
    c=json.loads(context_path.read_text());s=json.loads(selection_path.read_text());cfg=c['config']
    if c.get('format')!='r5_fold_context_v1' or canonical_sha256(cfg)!=c.get('config_sha256'):raise ValueError('Context format/config SHA mismatch')
    if c['config_sha256']!=expected_config_sha or validate_config(cfg)!=expected_config_sha:raise ValueError('Explicit frozen configuration SHA differs')
    if cfg['seed']!=42:raise ValueError('Original seed42 required')
    if (cfg['training']['epochs'],cfg['training']['patience'],cfg['training']['optimizer'])!=(100,7,'adam'):raise ValueError('User neural contract differs')
    if not fixture and cfg['tree']!={'maximum_trees':1500,'patience':40,'threads':4}:raise ValueError('Frozen tree budget/threads differs')
    if set(c.get('code_sha256',{}))!=set(RUNNER_NAMES):raise ValueError('Exact original17 runner code closure required')
    tree_names={'pipeline/'+n for n in TREE_NAMES}
    code={p:sha(ROOT/p) for p in set(RUNNER_NAMES)|tree_names}
    for p,h in c['code_sha256'].items():
        if code[p]!=h:raise ValueError('Original context code SHA mismatch: '+p)
    for p,h in EXTRA_TREE_SHA256.items():
        if code.get(p)!=h:raise ValueError('Original tree helper release SHA mismatch: '+p)
    if len(RUNNER_NAMES)!=17 or len(tree_names)!=7 or len(code)!=19:raise ValueError('Unexpected frozen code closure')
    fnum=c['training_binding']['fold'];folder=Path(c['data_fold'])
    if not fixture and str(folder)!=(Path(cfg['dataset'])/f'fold_{fnum:02d}').as_posix():raise ValueError('Portable original data-fold path required')
    if not folder.is_absolute():folder=root/folder
    folder=folder.resolve();_guard(folder,root)
    f=FoldDataR5(folder,cfg['seed'],project_root=root)
    if f.binding!=c['training_binding']:raise ValueError('TRAIN binding differs from original context')
    if s.get('fit_split')!='inner' or s.get('outer_or_Test_used') is not False or s.get('training_binding')!=f.binding:raise ValueError('Invalid CNN INNER-only selection binding')
    names=list(cfg['observation_candidates']);candidates=s.get('candidates',[])
    if [x.get('observation') for x in candidates]!=names:raise ValueError('All three original CNN candidates/order required')
    source_files={context_path.name:sha(context_path),selection_path.name:sha(selection_path)}
    for item in candidates:
        name=item['observation'];score=item['inner_rmse']
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or score<0:raise ValueError('Invalid CNN INNER score')
        expected=f'preprocessing_pilot/{name}/bg_cnn'
        if item['model_path']!=expected:raise ValueError('Unexpected CNN summary path')
        p=context_path.parent/expected/'summary.json'
        if sha(p)!=item['summary_sha256']:raise ValueError('Original CNN summary changed: '+name)
        v=json.loads(p.read_text())
        if v.get('status')!='completed' or v.get('kind')!='bg_cnn' or v.get('fixture_only') is not fixture or v.get('outer_or_Test_accessed') is not False:raise ValueError('Completed original CNN evidence required')
        if v.get('training_binding')!=f.binding or v.get('observation_policy')!=cfg['observation_candidates'][name] or v.get('best_inner_rmse')!=score:raise ValueError('CNN score/policy/binding differs from selection')
        source_files[expected+'/summary.json']=sha(p)
    winner=min(candidates,key=lambda x:x['inner_rmse'])['observation']
    if s.get('selected')!=winner:raise ValueError('CNN winner is not canonical candidate-list minimum')
    record=c['prior'];prior_path=context_path.parent/record['filename']
    if Path(record['filename']).name!=record['filename']:raise ValueError('Prior filename must remain relative')
    load_shared_prior(prior_path,expected_binding=f.binding,expected_file_sha256=record['file_sha256'],expected_prior_sha256=record['payload_sha256'])
    source_files[record['filename']]=sha(prior_path)
    paths=[folder/n for n in ['readiness_manifest.json','train.npz','validation.npz',
        'train_sample_metadata.parquet','inner_validation_metadata.parquet','sources/manifest.json']]
    source_info=json.loads((folder/'sources/manifest.json').read_text())
    for n,h in source_info.get('metadata_sha256',{}).items():
        p=folder/'sources'/n
        if sha(p)!=h:raise ValueError('Frozen source metadata changed')
        paths.append(p)
    inputs={p.relative_to(root).as_posix():sha(p) for p in paths}
    return {'context':c,'selection':s,'config':cfg,'fold':f,'folder':folder,'selected':winner,
        'policy':cfg['observation_candidates'][winner],'context_sha256':sha(context_path),
        'selection_sha256':sha(selection_path),'input_sha256':inputs,'code_sha256':code,
        'source_copy_sha256':source_files,'source_copy_parent':context_path.parent,'root':root}

def _unchanged(v):
    for p,h in v['input_sha256'].items():
        if sha(v['root']/p)!=h:raise ValueError('Data changed during Mac job: '+p)
    for p,h in v['code_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Core19 changed during Mac job: '+p)
    for p,h in v['source_copy_sha256'].items():
        if sha(v['source_copy_parent']/p)!=h:raise ValueError('Original source copy changed during Mac job: '+p)

def _replay(v,output,inner,*,fixture=False):
    import numpy as np
    from pipeline.lgbm_r5 import predict_tree
    with np.load(output/'inner_predictions.npz',allow_pickle=False) as z:saved={k:z[k] for k in ['share','score']}
    checks={}
    for batch in [1,7,256]:
        pred=predict_tree(inner['db'],v['fold'].binding,v['policy'],output,allow_fixture=fixture,batch_size=batch)
        errors={}
        for k in saved:
            if saved[k].shape!=(len(inner['db']),) or not np.isfinite(pred[k]).all() or ((pred[k]<0)|(pred[k]>1)).any():raise ValueError('Invalid replay predictions')
            errors[k]=float(np.max(np.abs(saved[k].astype(float)-pred[k].astype(float))))
            if not np.allclose(saved[k],pred[k],atol=REPLAY_ATOL,rtol=0):raise ValueError('Mac/cross-environment INNER replay mismatch')
        checks[str(batch)]=errors
    return {'batch_sizes':[1,7,256],'absolute_tolerance':REPLAY_ATOL,'relative_tolerance':0,
        'maximum_absolute_errors':checks,'query_accepts_db_only':True}

def verify_job(context,selection,output,receipt,*,expected_config_sha,root=ROOT,fixture=False):
    v=validate_inputs(context,selection,expected_config_sha=expected_config_sha,root=root,fixture=fixture);output=Path(output).resolve()
    r=json.loads(Path(receipt).read_text())
    if r.get('format')!=FORMAT or r.get('status')!='completed' or r.get('fixture_only') is not fixture:raise ValueError('Completed compatible Mac producer receipt required')
    if canonical_sha256({k:x for k,x in r.items() if k!='receipt_payload_sha256'})!=r.get('receipt_payload_sha256'):raise ValueError('Producer receipt payload changed')
    for k,x in [('context_sha256',v['context_sha256']),('selection_sha256',v['selection_sha256']),
        ('config_sha256',v['context']['config_sha256']),('training_binding',v['fold'].binding),
        ('selected_observation',v['selected']),('observation_policy',v['policy']),
        ('data_files_sha256',v['input_sha256']),('core_code_sha256',v['code_sha256'])]:
        if r.get(k)!=x:raise ValueError('Producer provenance mismatch: '+k)
    if r['producer_environment']['packages']['lightgbm']!='4.7.0':raise ValueError('Expected actual native Mac LightGBM4.7.0 producer')
    if sha(output/'summary.json')!=r['output_summary_sha256']:raise ValueError('Tree summary changed')
    summary=json.loads((output/'summary.json').read_text())
    if summary['artifact_sha256']!=r['artifact_sha256']:raise ValueError('Artifact declaration changed')
    inner,metadata=v['fold'].load('inner')
    if inner_identity(inner,metadata)!=r['inner_identity']:raise ValueError('INNER order/rawdb/raw-weight/target identity differs')
    replay=_replay(v,output,inner,fixture=fixture);_unchanged(v)
    env=environment()
    return {'status':'verified','receipt_payload_sha256':r['receipt_payload_sha256'],
        'replay':replay,'consumer_environment':env,
        'producer_consumer_package_versions_identical':env['packages']==r['producer_environment']['packages'],
        'OUTER_or_Test_opened':False,'fit_performed':False}

def fit_job(context,selection,output,receipt,*,expected_config_sha,root=ROOT,fixture=False):
    output=Path(output).resolve();receipt=Path(receipt).resolve()
    if output.exists():
        if receipt.is_file():return {'status':'noop_existing_completed_receipt_verified',
            'verification':verify_job(context,selection,output,receipt,expected_config_sha=expected_config_sha,root=root,fixture=fixture)}
        raise FileExistsError('Fresh Mac output required; partial directories cannot resume')
    if receipt.exists() or receipt.is_symlink():raise FileExistsError('Fresh receipt required')
    v=validate_inputs(context,selection,expected_config_sha=expected_config_sha,root=root,fixture=fixture);env=environment()
    if env['packages']['lightgbm']!='4.7.0':raise ValueError('Native LightGBM4.7.0 required')
    if not fixture and (sys.platform!='darwin' or Path(env['native_library']).suffix!='.dylib'):raise ValueError('Production producer must be the native Mac CPU runtime')
    if 'torch' in sys.modules:raise ValueError('Mac LightGBM must remain isolated from Torch')
    if not fixture and not output.is_relative_to(ROOT/'models/group_cv_r5_mac_lgbm_producers_r1'):raise ValueError('Explicit separate Mac producer staging version required')
    if output.is_relative_to(Path(context).resolve().parent):raise ValueError('Do not write into the original context/run directory')
    claim=output.with_name(output.name+'.claim.json')
    write_new(claim,{'format':FORMAT,'status':'claimed_once_no_resume','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'context_sha256':v['context_sha256'],'selection_sha256':v['selection_sha256'],'training_binding':v['fold'].binding})
    begin=time.perf_counter()
    from pipeline.lgbm_r5 import fit_tree
    train,tm=v['fold'].load('train');inner,im=v['fold'].load('inner')
    identity=inner_identity(inner,im)
    summary,_=fit_tree(train,inner,v['fold'].binding,v['policy'],v['config']['tree'],output,fixture=fixture)
    replay=_replay(v,output,inner,fixture=fixture);_unchanged(v)
    r={'format':FORMAT,'status':'completed','fixture_only':fixture,
        'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'context_sha256':v['context_sha256'],'config_sha256':v['context']['config_sha256'],
        'selection_sha256':v['selection_sha256'],'training_binding':v['fold'].binding,
        'selected_observation':v['selected'],'observation_policy':v['policy'],
        'CNN_selection_source_sha256':v['source_copy_sha256'],'data_files_sha256':v['input_sha256'],
        'core_code_sha256':v['code_sha256'],'runner_code_sha256':v['context']['code_sha256'],
        'tree_code_sha256':summary['code_sha256'],'producer_script_sha256':sha(__file__),
        'producer_environment':env,'inner_identity':identity,'inner_replay':replay,
        'output_summary_sha256':sha(output/'summary.json'),'artifact_sha256':summary['artifact_sha256'],
        'prior_provenance':{'used_by_tree':False,'record':v['context']['prior'],
            'reason':'Tree fits its own TRAIN PCA/prototypes; shared grid8 prior is preserved provenance only.'},
        'tree_executor_declaration':v['config'].get('tree_executor'),
        'trained_array_roles':v['fold'].opened_roles,'actual_tree_fit':True,
        'formal_model_training':not fixture,'core_or_original_R5_run_modified':False,
        'remote_transfer_SSH_GPU_OUTER_Test_or_ensemble_selection_performed':False,
        'data_and_core19_SHA_unchanged_after_fit':True,'partial_resume_allowed':False,
        'original_full_fold_completion_required_for_tree_fit':False,
        'consumer_must_verify_completed_NN_children_before_ensemble_freeze':True,
        'elapsed_seconds_including_replays':time.perf_counter()-begin}
    r['receipt_payload_sha256']=canonical_sha256(r)
    producer=output/'producer_summary.json';write_new(producer,r)
    if receipt!=producer:write_new(receipt,r)
    return r

def self_test():
    import copy,numpy as np,pandas as pd
    from ops.verify_lgbm_runner_r5 import generated_components
    from pipeline.grid_inputs_r5 import fit_shared_prior,save_shared_prior
    from pipeline.runner_r5 import CODE_NAMES
    with tempfile.TemporaryDirectory(prefix='mac_lgbm_r5_fixture_',dir='/private/tmp') as t:
        root=Path(t);folder=root/'data/fold_00';(folder/'sources').mkdir(parents=True)
        train=generated_components(64,921);inner=generated_components(16,922)
        files={}
        for role,data,name in [('train',train,'train'),('inner',inner,'validation')]:
            p=folder/(name+'.npz');np.savez_compressed(p,**data);files[p.name]=sha(p)
            md=pd.DataFrame({'sample_id':[f'passage_r4_f00_{name}_{i:07d}' for i in range(len(data['db']))],
                'validation_id':np.arange(len(data['db'])),'is_outer_oof':False,'effective_share':data['share'],'fold':0})
            p=folder/('train_sample_metadata.parquet' if role=='train' else 'inner_validation_metadata.parquet');md.to_parquet(p,index=False);files[p.name]=sha(p)
        source_hash=canonical_sha256({'self_test':'numeric TRAIN/INNER only'})
        write_new(folder/'sources/manifest.json',{'source_hash':source_hash,'metadata_sha256':{}})
        gate={'ready_to_train':True,'protocol':'source_group_nested5_vtuad_passage_r4','fold':0,
            'source_version_dir':str(folder/'sources'),'source_hash':source_hash,
            'source_version_manifest_sha256':sha(folder/'sources/manifest.json'),'file_sha256':files,
            'data_package_hash':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            'partitions':{'train':{'examples':64}},'validation_role':'inner_early_stopping','outer_validation_role':'outer_oof_reporting_only'}
        for k in ['source_checks_passed','reference_group_and_duplicate_isolation_passed','background_week_and_purge_isolation_passed','sealed_holdout_locked']:gate[k]=True
        for k in ['challenge_test_accessed','sealed_arrays_accessed','calibration_accessed','model_evaluated_on_sealed','frozen_v1_data_modified']:gate[k]=False
        write_new(folder/'readiness_manifest.json',gate)
        from pipeline.fold_data_r5 import FoldDataR5
        binding=FoldDataR5(folder,42,project_root=root).binding
        inbox=root/'inbox';inbox.mkdir();prior=fit_shared_prior(train,binding)
        record=save_shared_prior(inbox/'shared_background_prior.json',prior,expected_binding=binding)
        cfg=json.loads((ROOT/'configs/r5_training_gpu_r1.json').read_text());cfg['tree']={'maximum_trees':4,'patience':2,'threads':4}
        context={'format':'r5_fold_context_v1','data_fold':str(folder),'training_binding':binding,'config':cfg,
            'config_sha256':canonical_sha256(cfg),'prior':record,'code_sha256':{p:sha(ROOT/p) for p in CODE_NAMES}}
        write_new(inbox/'context.json',context);candidates=[]
        for (name,policy),score in zip(cfg['observation_candidates'].items(),[.4,.3,.35]):
            relative=f'preprocessing_pilot/{name}/bg_cnn';p=inbox/relative/'summary.json'
            write_new(p,{'status':'completed','kind':'bg_cnn','fixture_only':True,'outer_or_Test_accessed':False,
                'training_binding':binding,'observation_policy':policy,'best_inner_rmse':score})
            candidates.append({'observation':name,'inner_rmse':score,'model_path':relative,'summary_sha256':sha(p)})
        selection={'fit_split':'inner','outer_or_Test_used':False,'training_binding':binding,'candidates':candidates,'selected':'resolution_01'}
        write_new(inbox/'preprocessing_selection.json',selection)
        out=root/'mac_models/fold_00';receipt=root/'producer_receipt.json'
        expected=canonical_sha256(cfg)
        r=fit_job(inbox/'context.json',inbox/'preprocessing_selection.json',out,receipt,expected_config_sha=expected,root=root,fixture=True)
        replay=verify_job(inbox/'context.json',inbox/'preprocessing_selection.json',out,receipt,expected_config_sha=expected,root=root,fixture=True)
        same=fit_job(inbox/'context.json',inbox/'preprocessing_selection.json',out,receipt,expected_config_sha=expected,root=root,fixture=True)
        assert same['status']=='noop_existing_completed_receipt_verified'
        rejected=[]
        partial=root/'mac_models/partial';partial.mkdir()
        try:fit_job(inbox/'context.json',inbox/'preprocessing_selection.json',partial,root/'missing_receipt.json',expected_config_sha=expected,root=root,fixture=True)
        except FileExistsError:rejected.append('partial_output_resume')
        else:raise AssertionError('Partial output was resumed')
        claimed=root/'mac_models/claimed_once';write_new(claimed.with_name(claimed.name+'.claim.json'),{'claimed':True})
        try:fit_job(inbox/'context.json',inbox/'preprocessing_selection.json',claimed,root/'claimed_receipt.json',expected_config_sha=expected,root=root,fixture=True)
        except FileExistsError:rejected.append('exclusive_job_claim')
        else:raise AssertionError('Previously claimed job was fitted')
        assert not claimed.exists()
        for name,change in [('wrong_winner',lambda x:x.update(selected='raw')),('outer_used',lambda x:x.update(outer_or_Test_used=True))]:
            bad=copy.deepcopy(selection);change(bad);p=inbox/(name+'.json');write_new(p,bad)
            try:validate_inputs(inbox/'context.json',p,expected_config_sha=expected,root=root,fixture=True)
            except ValueError:rejected.append(name)
            else:raise AssertionError('Invalid selection accepted')
        bad=copy.deepcopy(context);bad['code_sha256']['pipeline/neural_r5.py']='0'*64;p=inbox/'bad_context.json';write_new(p,bad)
        try:validate_inputs(p,inbox/'preprocessing_selection.json',expected_config_sha=expected,root=root,fixture=True)
        except ValueError:rejected.append('actual_code_SHA_mismatch')
        else:raise AssertionError('Changed core accepted')
        with (out/'physics_pca.json').open('a') as f:f.write(' ')
        try:verify_job(inbox/'context.json',inbox/'preprocessing_selection.json',out,receipt,expected_config_sha=expected,root=root,fixture=True)
        except ValueError:rejected.append('artifact_tampering')
        else:raise AssertionError('Tampered artifact accepted')
        return {'status':'passed synthetic Mac producer fixture','formal_training_performed':False,
            'real_dataset_SSH_GPU_OUTER_Test_accessed':False,'numeric_TRAIN':64,'numeric_INNER':16,
            'tiny_trees':4,'replay':replay['replay'],'guards_rejected':rejected,
            'completed_receipt_verify_only_noop':True,'core19_unchanged':True}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['fit','verify','self-test'])
    for name in ['context','selection','output','receipt']:p.add_argument('--'+name,type=Path)
    p.add_argument('--expected-config-sha')
    a=p.parse_args()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
    if a.action=='self-test':result=self_test()
    else:
        if any(getattr(a,n) is None for n in ['context','selection','output','receipt','expected_config_sha']):p.error('Explicit --context --selection --output --receipt --expected-config-sha required')
        result=(fit_job if a.action=='fit' else verify_job)(a.context,a.selection,a.output,a.receipt,expected_config_sha=a.expected_config_sha)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
