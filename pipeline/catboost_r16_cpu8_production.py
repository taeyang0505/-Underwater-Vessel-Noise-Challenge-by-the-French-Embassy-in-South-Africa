"""Explicit CPU8 production variant; original CPU2 module stays immutable.

Each worker fits one D0/H0-fold pair. Ownfold CV selects trees, shared fit
is loaded only after both selected models exist. No Test/report inputs.
"""
from pathlib import Path
import datetime,gc,json,resource,shutil,sys,time
from zoneinfo import ZoneInfo

from pipeline.catboost_r16_peft import (VERSION,DEFAULTS,FEATURE_COUNTS,sha,write,
    source_binding,verify,dependency_status,tree_feature_matrix)
from pipeline.r16_multisupport_data import ROOT,OUT as DATA,role,guard
from pipeline.training_r16_multisupport import metric

RUN=ROOT/'models/r16_catboost_peft_cpu8_r1'
OPS=ROOT/'ops/r16_catboost_peft_cpu8_r1'
BENCHMARK_CACHE=ROOT/'data/cache/r16_pretrained_catboost_benchmark_r1/fold_00_D0'
RSS_LIMIT=12*2**30
DISK_MINIMUM=3*2**30


def timestamp():return datetime.datetime.now(ZoneInfo('Asia/Seoul')).isoformat()


def pins():
    p=source_binding()
    for name in ('pipeline/catboost_r16_cpu8_production.py','ops/run_r16_catboost_cpu8.py'):
        p[name]=sha(ROOT/name)
    return p


def peak_rss():return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def disk_guard():
    free=shutil.disk_usage(ROOT).free
    if free<DISK_MINIMUM:raise OSError(f'Free disk below3GiB: {free}bytes; no unrelated files deleted')
    return free


def update(stage,**values):
    value={'phase':stage,'updated_at_kst':timestamp(),**values}
    write(OPS/'progress.json',value)
    print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)


def check_plan():
    plan=json.loads((OPS/'PLAN.json').read_text())
    if plan['CPU_threads']!=8 or plan['maximum_trees']!=1500 or plan['early_stopping_rounds']!=40:
        raise ValueError('Frozen CPU8 maximum1500/ES40 contract differs')
    verify(plan['source_sha256'])
    if plan['data_manifest_sha256']!=sha(DATA/'manifest.json'):raise ValueError('Data identity changed')
    if not dependency_status(True)['available']:raise RuntimeError('Pinned CatBoost dependency missing')
    return plan


def verify_completed(folder,plan):
    folder=Path(folder);s=json.loads((folder/'summary.json').read_text())
    if s['status']!='completed' or s['format']!=VERSION or s['production_variant']!='CPU8_r1':
        raise ValueError('Completed production pair required; partial never reused')
    if s['code_sha256']!=plan['source_sha256'] or s['data_manifest_sha256']!=plan['data_manifest_sha256']:
        raise ValueError('Completed pair binding changed')
    for name,h in s['artifact_sha256'].items():
        if sha(folder/name)!=h:raise ValueError('Completed pair artifact changed')
    return s


def check_benchmark_cache(plan):
    p=BENCHMARK_CACHE/'manifest.json'
    if sha(p)!=plan['benchmark_cache_manifest_sha256']:raise ValueError('Benchmark cache manifest changed')
    m=json.loads(p.read_text())
    if m['status']!='completed' or m['data_manifest_sha256']!=plan['data_manifest_sha256']:
        raise ValueError('Benchmark cache/data mismatch')
    verify(m['source_sha256'])
    for name,e in m['features'].items():
        if sha(BENCHMARK_CACHE/name)!=e['sha256']:raise ValueError('Benchmark feature bytes changed')
    return m


def fit_pair_cpu8(folder,fold,regime,*,smoke_rows=None,smoke_iterations=4):
    import numpy as np
    from catboost import CatBoostRegressor,CatBoostClassifier
    sys.addaudithook(guard);plan=check_plan();folder=Path(folder).resolve()
    if folder.exists():raise FileExistsError('Fresh pair directory required; no automatic partial reuse')
    if fold not in range(5) or regime not in ('D0','H0'):raise ValueError('Fold0..4,D0/H0 required')
    disk_guard();folder.mkdir(parents=True);temp=folder/'temporary_features';temp.mkdir()
    train=role(fold,regime,'train');cv=role(fold,regime,'cv')
    if set(train['base_window_index'])&set(cv['base_window_index']):raise ValueError('TRAIN/CV source overlap')
    manifest=json.loads((DATA/'manifest.json').read_text())
    if len(train['share'])!=manifest['folds'][str(fold)]['roles']['train'][regime]['n'] or len(cv['share'])!=1000:
        raise ValueError('Data schema/count differs')
    if smoke_rows is not None:
        ix=np.random.default_rng(2026102817+fold).permutation(len(train['share']))[:smoke_rows]
        train={k:v[ix] for k,v in train.items()}
    params=dict(DEFAULTS,thread_count=8,iterations=1500 if smoke_rows is None else smoke_iterations,
        random_seed=2026102817+fold)
    receipt={'format':VERSION,'production_variant':'CPU8_r1','status':'fixed_before_fit',
        'fold':fold,'regime':regime,'parameters':params,'early_stopping_rounds':40,
        'train_n':len(train['share']),'CV_n':len(cv['share']),
        'early_stopping_role':'existing ownfold CV1000',
        'ensemble_fit_role':'sharedlocal_fit2400/400backgrounds,afterbothfits',
        'feature_counts':FEATURE_COUNTS,'code_sha256':plan['source_sha256'],
        'data_manifest_sha256':plan['data_manifest_sha256'],
        'dependency':dependency_status(True),'disposable_smoke_only':smoke_rows is not None,
        'Test_used':False,'locked_report_used':False,'memory_guard_bytes':RSS_LIMIT,
        'temporary_cleanup_scope':'onlythisnewpairtemporary_featuresdirectory'}
    write(folder/'PLAN.json',receipt)
    began=time.perf_counter();pred={};history={};iterations={};feature_receipts={}
    cache=check_benchmark_cache(plan) if fold==0 and regime=='D0' and smoke_rows is None else None
    try:
        for kind in FEATURE_COUNTS:
            disk_guard();verify(plan['source_sha256']);iscls=kind=='classification';key='presence' if iscls else 'share'
            feature_started=time.perf_counter();paths={};reused=cache is not None
            for name,data in [('train',train),('cv',cv)]:
                filename=f'{kind}_{name}.npy'
                if reused:
                    path=BENCHMARK_CACHE/filename
                    x=np.load(path,mmap_mode='r',allow_pickle=False)
                    if tuple(x.shape)!=(len(data['share']),FEATURE_COUNTS[kind]):raise ValueError('Reused shape differs')
                    e=dict(cache['features'][filename],path=str(path.relative_to(ROOT)),retained=True)
                else:
                    path=temp/filename
                    x=tree_feature_matrix(data['db'],kind,path)
                    e=dict(path=str(path.relative_to(ROOT)),shape=list(x.shape),bytes=path.stat().st_size,
                        sha256=sha(path),retained=False)
                paths[name]=path;feature_receipts[filename]=e;del x
            feature_seconds=time.perf_counter()-feature_started;gc.collect()
            if peak_rss()>RSS_LIMIT:raise MemoryError('Feature extraction peak RSS exceeded12GiB before fit')
            x=np.load(paths['train'],mmap_mode='r',allow_pickle=False)
            v=np.load(paths['cv'],mmap_mode='r',allow_pickle=False)
            abort={'reason':None};fit_started=time.perf_counter();timings=[]
            update('training',fold=fold,regime=regime,kind=kind,round=0,max_rounds=params['iterations'],
                feature_seconds=feature_seconds,reused_benchmark_cache=reused,peak_rss_bytes=peak_rss(),
                disk_free_bytes=disk_guard())
            class Health:
                def after_iteration(self,info):
                    elapsed=time.perf_counter()-fit_started;timings.append({'round':int(info.iteration),'elapsed_seconds':elapsed})
                    if peak_rss()>RSS_LIMIT:abort['reason']='processpeakRSSexceeds12GiB'
                    metrics={scope:{key:float(val[-1]) for key,val in a.items() if len(val)} for scope,a in info.metrics.items()}
                    if any(not np.isfinite(z) for a in metrics.values() for z in a.values()):abort['reason']='nonfinitemetric'
                    if info.iteration==1 or info.iteration%25==0 or abort['reason']:
                        value={'fold':fold,'regime':regime,'kind':kind,'round':int(info.iteration),
                            'max_rounds':params['iterations'],'elapsed_fit_seconds':elapsed,
                            'metrics':metrics,'peak_rss_bytes':peak_rss(),'abort_reason':abort['reason']}
                        write(folder/'progress.json',value);update('training',**value)
                    return abort['reason'] is None
            model=(CatBoostClassifier(loss_function='Logloss',eval_metric='AUC',**params) if iscls else
                CatBoostRegressor(loss_function='RMSE',eval_metric='RMSE',**params))
            model.fit(x,train[key],eval_set=(v,cv[key]),use_best_model=True,early_stopping_rounds=40,
                callbacks=[Health()],verbose=False)
            if abort['reason']:raise RuntimeError('Boundedmemory/numericguard: '+abort['reason'])
            model.save_model(str(folder/(kind+'.cbm')))
            pred[kind]=(model.predict_proba(v,thread_count=8)[:,1] if iscls else np.clip(model.predict(v,thread_count=8),0,1))
            if not np.isfinite(pred[kind]).all():raise FloatingPointError('NonfiniteCVpredictions')
            history[kind]=model.get_evals_result()
            iterations[kind]={'best_iteration_0based':int(model.get_best_iteration()),
                'retained_trees':int(model.tree_count_),'executed_rounds':len(timings),
                'stop_reason':'maximum_trees' if len(timings)==params['iterations'] else 'early_stopping_ES40',
                'fit_seconds':time.perf_counter()-fit_started,'feature_seconds':feature_seconds,
                'reused_benchmark_cache':reused}
            write(folder/(kind+'_iterations.json'),timings)
            write(folder/(kind+'_completed.json'),{'status':'completed','iteration':iterations[kind],
                'model_sha256':sha(folder/(kind+'.cbm')),'source_sha256':plan['source_sha256']})
            del model,x,v;gc.collect()
            if not reused:
                for path in paths.values():path.unlink()
            write(folder/'feature_receipts.json',feature_receipts)
            update('task_completed',fold=fold,regime=regime,kind=kind,iteration=iterations[kind],peak_rss_bytes=peak_rss())
        with (folder/'ownfold_cv.npz').open('wb') as f:np.savez_compressed(f,
            share=pred['regression'],classifier_prob=pred['classification'],target_share=cv['share'],sample_id=cv['sample_id'])
        cvmetrics=metric(cv['share'],pred['regression'],pred['classification'])
        # Both trained artifacts are now fixed. Sharedfit cannot affect trees.
        selection=role(0,'D0','selection')
        if set(train['base_window_index'])&set(selection['base_window_index']):raise ValueError('TRAIN/ensemblefit overlap')
        pp={}
        for kind in FEATURE_COUNTS:
            m=CatBoostClassifier() if kind=='classification' else CatBoostRegressor()
            m.load_model(str(folder/(kind+'.cbm')))
            x=tree_feature_matrix(selection['db'],kind)
            pp[kind]=m.predict_proba(x,thread_count=8)[:,1] if kind=='classification' else np.clip(m.predict(x,thread_count=8),0,1)
            if not np.isfinite(pp[kind]).all():raise FloatingPointError('Nonfinite ensemblefit predictions')
            del m,x;gc.collect()
        with (folder/'selection.npz').open('wb') as f:np.savez_compressed(f,
            share=pp['regression'],classifier_prob=pp['classification'],target_share=selection['share'],sample_id=selection['sample_id'])
        write(folder/'history.json',history);verify(plan['source_sha256']);temp.rmdir()
        summary={**receipt,'status':'completed','iterations':iterations,'seconds':time.perf_counter()-began,
            'ownfold_CV_metrics':cvmetrics,'selection_metrics':metric(selection['share'],pp['regression'],pp['classification']),
            'checkpoint_and_ensemble_fit_roles_disjoint':True,'peak_process_rss_bytes':peak_rss(),
            'artifact_sha256':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}}
        write(folder/'summary.json',summary);update('pair_completed',fold=fold,regime=regime,
            seconds=summary['seconds'],CV=cvmetrics,selection=summary['selection_metrics'],
            summary_sha256=sha(folder/'summary.json'),peak_rss_bytes=peak_rss())
        return summary
    except Exception as exc:
        # Clean only newlygenerated temporary features, never source/cache/model.
        if temp.is_dir():
            for p in temp.glob('*.npy'):p.unlink()
            if not any(temp.iterdir()):temp.rmdir()
        write(folder/'FAILED.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}',
            'explicit_new_run_required':True,'partial_reuse_forbidden':True,'updated_at_kst':timestamp()})
        raise
