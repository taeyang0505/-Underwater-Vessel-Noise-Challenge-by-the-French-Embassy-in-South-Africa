"""Time original R5 TRAIN-only features and short CPU LightGBM fits.

Diagnostic only: no validation, early stopping, predictions, RMSE, saved
boosters or production reuse. A separate worker is killed after at most 300s;
completed stage/iteration timings survive in its private temporary journal.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,resource,subprocess,sys,tempfile,time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def rss():
    n=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if sys.platform=='darwin' else n*1024)

def write_journal(path,state):
    state['peak_RSS_bytes']=rss()
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def worker(args):
    os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
    begin=time.perf_counter();journal=args.workdir/'progress.json'
    state={'status':'running speed diagnostic','stage':'imports','timing_seconds':{},'fits':{},
        'diagnostic_only':True,'neural_GPU_SSH_or_network_used':False,
        'INNER_OUTER_Test_or_external_source_arrays_opened':False,
        'validation_ES_predictions_or_RMSE_computed':False,
        'boosters_saved_or_available_for_production_reuse':False}
    write_journal(journal,state)
    try:
        import numpy as np
        import pandas as pd
        import lightgbm as lgb
        from pipeline.fold_data_r5 import FoldDataR5
        from pipeline.lgbm_r5 import feature_binding,transform
        from pipeline.physics_features_r5 import fit_physics_pca,save_physics_pca
        from pipeline.observation_augmentation_r5 import prepare_query_observations
        from pipeline.train_lgbm import training_prototypes
        config=json.loads(args.config.read_text());policy=config['observation_candidates'][args.observation]
        folder=ROOT/config['dataset']/'fold_00';data_root=(ROOT/'data').resolve()
        allowed={(folder/p).resolve() for p in ['readiness_manifest.json','sources/manifest.json','train.npz','train_sample_metadata.parquet']}
        def check(path):
            p=Path(path).resolve()
            if p.is_relative_to(data_root) and p not in allowed:raise RuntimeError('Non-TRAIN data read blocked: '+str(p))
            if p.suffix.lower() in {'.wav','.flac','.mp3'}:raise RuntimeError('External audio blocked')
            return p
        def audit(event,a):
            if event in {'socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'}:raise RuntimeError('Network/subprocess blocked in diagnostic worker')
            if event=='open' and isinstance(a[0],(str,bytes,os.PathLike)):
                p=check(os.fsdecode(a[0]))
                if p.is_relative_to(data_root) and a[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):raise RuntimeError('Source-data write blocked')
        sys.addaudithook(audit)
        np_load,pd_read=np.load,pd.read_parquet
        def safe_load(p,*a,**k):check(p);return np_load(p,*a,**k)
        def safe_parquet(p,*a,**k):check(p);return pd_read(p,*a,**k)
        np.load,pd.read_parquet=safe_load,safe_parquet
        state.update(library_versions={'numpy':np.__version__,'lightgbm':lgb.__version__},
            hardware={'machine':os.uname().machine,'system':os.uname().sysname,'logical_CPUs':os.cpu_count(),
                      'LightGBM_threads':4,'OMP_NUM_THREADS':os.environ['OMP_NUM_THREADS'],
                      'OPENBLAS_NUM_THREADS':os.environ['OPENBLAS_NUM_THREADS']},
            observation_name=args.observation,observation_policy=policy)
        def stage(name,fn):
            state['stage']=name;write_journal(journal,state);t=time.perf_counter();out=fn()
            state['timing_seconds'][name]=time.perf_counter()-t;write_journal(journal,state)
            print(json.dumps({'completed_stage':name,'seconds':state['timing_seconds'][name]}),flush=True)
            return out
        # The stdout line above is diagnostic progress, never a performance score.
        fold=stage('readiness_binding_TRAIN_hash',lambda:FoldDataR5(folder,config['seed']))
        train,meta=stage('TRAIN_load_metadata_hash_label_checks',lambda:fold.load('train'))
        if len(train['db'])!=53244:raise ValueError('Expected unchanged full fold0 TRAIN')
        state['training_binding']=fold.binding;state['rows']=len(train['db']);del meta
        feature=stage('feature_quantization',lambda:prepare_query_observations(train['db'],quantization_decimals=policy['quantization_decimals'])['feature_db'])
        binding=feature_binding(fold.binding,policy)
        def pca_fit():
            pca=fit_physics_pca(feature,binding,split='train')
            save_physics_pca(args.workdir/'physics_pca.json',pca,expected_binding=binding);return pca
        pca=stage('full_TRAIN_PCA_fit_and_temp_save',pca_fit)
        prototype=stage('full_TRAIN_positive_prototypes_fit_and_temp_save',lambda:training_prototypes(
            {'db':feature,'presence':train['presence']},args.workdir/'prototypes.npz',fold.binding['data_package_hash'],fold.binding['seed'],feature_version=3))
        X=stage('full_TRAIN_original_transform_1324',lambda:transform(train['db'],policy,pca,fold.binding,args.workdir/'prototypes.npz'))
        if X.shape!=(53244,1324) or not np.isfinite(X).all():raise ValueError('Full original feature matrix mismatch')
        state.update(feature_shape=list(X.shape),feature_dtype=str(X.dtype),feature_matrix_bytes=int(X.nbytes),prototype=prototype)
        reg=lgb.LGBMRegressor(objective='regression_l2',n_estimators=50,learning_rate=.035,num_leaves=31,
            min_child_samples=50,colsample_bytree=.85,subsample=.85,subsample_freq=1,max_bin=127,n_jobs=4,
            random_state=fold.binding['seed'],verbosity=-1)
        clf=lgb.LGBMClassifier(n_estimators=30,num_leaves=15,min_child_samples=50,max_bin=127,n_jobs=4,
            random_state=fold.binding['seed'],verbosity=-1)
        for name,template,target,rounds in [('regression',reg,train['share'],50),('presence',clf,train['presence'],30)]:
            params=template.get_params()
            for key in ['n_estimators','importance_type','class_weight']:params.pop(key,None)
            params['num_threads']=params.pop('n_jobs');params['metric']='None'
            if name=='presence':params['objective']='binary'
            info={'rounds_requested':rounds,'rounds_completed':0,'params':params,'iteration_seconds':[]}
            state['fits'][name]=info
            def construct():
                d=lgb.Dataset(X,label=target,params=params,free_raw_data=True);d.construct();return d
            dataset=stage(name+'_Dataset_construct_bins',construct)
            state['stage']=name+'_boosting';write_journal(journal,state)
            t=time.perf_counter();previous=[t]
            def callback(env):
                now=time.perf_counter();info['iteration_seconds'].append(now-previous[0]);info['rounds_completed']=env.iteration+1
                if env.iteration==0 or (env.iteration+1)%10==0:write_journal(journal,state)
                previous[0]=time.perf_counter()
            callback.order=99;callback.before_iteration=False
            booster=lgb.train(params,dataset,num_boost_round=rounds,callbacks=[callback])
            info['fit_wall_seconds_including_callback_journals']=time.perf_counter()-t
            info['first_iteration_including_booster_initialization_seconds']=info['iteration_seconds'][0]
            late=np.asarray(info['iteration_seconds'][10:])
            info['later_rounds_11_onward']={'count':len(late),'mean_seconds':float(late.mean()),
                'median_seconds':float(np.median(late)),'p10_seconds':float(np.quantile(late,.1)),
                'p90_seconds':float(np.quantile(late,.9))}
            info['booster_retained_or_saved']=False
            del booster,dataset
            write_journal(journal,state)
            print(json.dumps({'completed_boosting':name,'rounds':rounds,'fit_seconds':info['fit_wall_seconds_including_callback_journals'],
                              'late_seconds_per_round':info['later_rounds_11_onward']['mean_seconds']}),flush=True)
        overhead=sum(state['timing_seconds'].values())
        projection={}
        for scenario,quantile in [('later_round_p10_scenario','p10_seconds'),('later_round_mean_scenario','mean_seconds'),('later_round_p90_scenario','p90_seconds')]:
            total=overhead
            for name,maximum in [('regression',1500),('presence',300)]:
                info=state['fits'][name];total+=info['first_iteration_including_booster_initialization_seconds']+(maximum-1)*info['later_rounds_11_onward'][quantile]
            projection[scenario]=total
        state.update(status='passed TRAIN-only Mac CPU speed diagnostic',stage='complete',
            elapsed_seconds=time.perf_counter()-begin,opened_static_roles=fold.opened_roles,
            full_TRAIN_feature_preparation_seconds=sum(v for k,v in state['timing_seconds'].items() if k.startswith(('feature_','full_TRAIN_'))),
            one_fold_maximum_1500reg_300presence_CPU_seconds_scenarios=projection,
            extrapolation_limit='Linear timing scenarios, not guaranteed bounds. Initial data/PCA/prototype/features/bin overhead counted once; first iteration separated from later rounds. Actual early-stopping iteration, INNER/OUTER transform/prediction, other folds, GPU waits, and long-run tree complexity were not measured.',
            iteration_timing_note='After-iteration intervals exclude this profiler callback journal IO; first interval includes Booster initialization. No eval_set or metric was requested.',
            temporary_preprocessing_state_for_production_reuse=False,
            code_sha256={p:sha(ROOT/p) for p in ['ops/profile_lgbm_mac_r5.py','pipeline/lgbm_r5.py','pipeline/physics_features_r5.py','pipeline/train_lgbm.py','pipeline/mixtures.py','pipeline/fold_data_r5.py']})
        if fold.opened_roles!=['train']:raise AssertionError('Non-TRAIN role opened')
        write_journal(journal,state)
    except BaseException as e:
        state.update(status='failed speed diagnostic',error_type=type(e).__name__,error=str(e),elapsed_seconds=time.perf_counter()-begin)
        write_journal(journal,state);raise

def main(args):
    if args.worker:return worker(args)
    out=args.output.resolve()
    if out.exists():raise FileExistsError('Fresh diagnostic report required')
    if not 1<=args.timeout_seconds<=300:raise ValueError('Time budget must be <=300 seconds')
    work=Path(tempfile.mkdtemp(prefix='r5_lgbm_mac_speed_',dir='/private/tmp'))
    cmd=[sys.executable,'-u','-m','ops.profile_lgbm_mac_r5','--worker','--workdir',str(work),'--config',str(args.config.resolve()),'--observation',args.observation]
    timeout=False;t=time.perf_counter()
    try:
        r=subprocess.run(cmd,cwd=ROOT,timeout=args.timeout_seconds,check=False);exit_code=r.returncode
    except subprocess.TimeoutExpired:timeout=True;exit_code=None
    p=work/'progress.json';report=json.loads(p.read_text()) if p.exists() else {'status':'failed before journal'}
    if timeout:report.update(status='timed out; only completed stages/rounds reported',timeout_seconds=args.timeout_seconds)
    report.update(worker_exit_code=exit_code,worker_killed_at_time_limit=timeout,wall_seconds=time.perf_counter()-t,
        temporary_journal=str(p),created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    for name in ['physics_pca.json','prototypes.npz']:
        path=work/name
        if path.exists():path.unlink()
    report['temporary_preprocessing_files_removed']=True
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'report':str(out),'status':report['status'],'wall_seconds':report['wall_seconds']},indent=2))
    if exit_code!=0:raise SystemExit(2)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'reports/R5_LIGHTGBM_MAC_SPEED_PROFILE.json')
    p.add_argument('--config',type=Path,default=ROOT/'configs/r5_training_integration.json');p.add_argument('--observation',default='raw')
    p.add_argument('--timeout-seconds',type=int,default=300);p.add_argument('--worker',action='store_true');p.add_argument('--workdir',type=Path)
    main(p.parse_args())
