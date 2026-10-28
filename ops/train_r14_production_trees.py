"""Explicitly approved R14 fivefold full-tree CPU production stage.

Reuses only the two completed Fold0 full5252 learners, trains all eight
remaining learners (regression plus independent classifier each) sequentially
on CPU2, and never opens query/report arrays. Old diagnostics are immutable.
"""
from __future__ import annotations
import argparse,json,os,sys,time,shutil,subprocess,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE_CONFIG='configs/r13_full_training_v1.json'
BUNDLE='data/synthetic/group_cv_physics_r13_variants_v2'
OUT='models/r14_production_trees_r1'
VARIANTS=('baseline','highshare')
ROWS=[78869,78069,78362,78753,78124]
DATA_HASH='bb1dc465007a70798968ce14040b59643715538716b9e2c1f11c77e72c27d3c5'
USER_SHA='54bae81c2c0aa8b5e5fa299cf6aebbbe0af98b69d68f41add3c3b1802a212ba6'
USER_FILE='/Users/hongtaeyang/.codex/attachments/aa675a70-ab73-4ba9-bf20-9dea6a9a792c/붙여넣은 텍스트.txt'
EXPECTED_FOLD0_FREEZE_SHA='a25811965e0edf8669555db19606be740c72f70e24a1505f867c30e731bdbdfe'

def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def token(v):
    import hashlib
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def local(p):
    q=(ROOT/p).resolve()
    if not q.is_relative_to(ROOT):raise ValueError('Project-contained run/config files required')
    return q
def env():
    from ops.train_r12_lightgbm import environment,guard
    environment(2);sys.addaudithook(guard)
def peak():
    n=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if sys.platform=='darwin' else n*1024)
def disk():
    free=shutil.disk_usage(ROOT).free
    if free<2*1024**3:raise OSError('Less than2GiB free; preserve artifacts and stop, never delete existing data')
    return free
def verify(p):
    for n,h in p['source_pins'].items():
        if sha(ROOT/n)!=h:raise ValueError('Frozen source/config/data changed:'+n)
    if sha(p['user_instruction_file'])!=USER_SHA:raise ValueError('Approval attachment changed')

def approved(config,approval):
    c=read(config);a=read(approval)
    if c.get('production_recipe_approved') is not True or a.get('production_recipe_approved') is not True:
        raise PermissionError('Explicit new production approval true required; no old guard bypass')
    if a.get('user_instruction_sha256')!=USER_SHA or sha(USER_FILE)!=USER_SHA:
        raise PermissionError('Exact latest human instruction hash required')
    expected={'source_config_path':SOURCE_CONFIG,'bundle_path':BUNDLE,'bundle_data_hash':DATA_HASH,
        'source_seed':2026102713,'folds':[0,1,2,3,4],'input_channels':14}
    if c.get('version')!='r14_production_mac_v1' or any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Approved R14 source/full14 specification differs')
    tree={'folds':[0,1,2,3,4],'variants':list(VARIANTS),'regressor_and_classifier_each':True,
        'reuse_completed_fold0_root':'models/r14_tree_full_r1','feature_count':5252,'maximum_trees':1500,
        'early_stopping_patience':40,'threads':2,'maximum_training_rows':100000,'feature_batch_size':512,'executor':'mac_cpu'}
    if any(c.get('tree',{}).get(k)!=v for k,v in tree.items()):raise ValueError('Approved full-tree policy differs')
    return c,a

def separation(train,validation):
    import numpy as np,pandas as pd
    if set(train.base_window_index)&set(validation.base_window_index):raise ValueError('TRAIN/selection primary overlap')
    if set(train.background_group.astype(str))&set(validation.background_group.astype(str)):raise ValueError('TRAIN/selection group overlap')
    gap=float('inf');count=0
    for sensor,v in validation.groupby('sensor_group'):
        t=train[train.sensor_group==sensor]
        if not len(t):continue
        x=np.sort(pd.to_datetime(t.start_export_clock).astype('int64').to_numpy());y=pd.to_datetime(v.start_export_clock).astype('int64').to_numpy();ix=np.searchsorted(x,y)
        d=np.minimum(abs(y-x[np.clip(ix,0,len(x)-1)]),abs(y-x[np.clip(ix-1,0,len(x)-1)]));g=float(np.maximum(d/1e9-900,0).min()/3600)
        gap=min(gap,g);count+=len(v)
    if count and gap<24:raise ValueError('900second support TRAIN/selection gap less than24h')
    return {'primary_ID_intersection':0,'background_group_intersection':0,'same_sensor_rows_checked':count,
        'minimum_900second_support_gap_hours':gap if count else None,'required_hours':24,
        'time_basis':'same sensor export clock; independently verified UTC not asserted'}

def fold0_reuse(m,source_cfg):
    from pipeline.trees_ensemble_r13 import CODE_NAMES
    origin=ROOT/'models/r14_tree_full_r1'
    if sha(origin/'ALL_TREES_FROZEN.json')!=EXPECTED_FOLD0_FREEZE_SHA:raise ValueError('Exact previously frozen Fold0 registry required')
    g=read(origin/'ALL_TREES_FROZEN.json')
    if g['all_models_frozen'] is not True or token({k:v for k,v in g.items() if k!='freeze_sha256'})!=g['freeze_sha256']:
        raise ValueError('Original full Fold0 global freeze required')
    out={}
    for variant in VARIANTS:
        arm=variant+'P1';folder=origin/arm;done=read(folder/'COMPLETED.json');s=read(folder/'model/summary.json');b=s['training_binding']
        entry=g['models'][arm]
        expected_files={str(x.relative_to(origin)):sha(x) for x in (folder/'model').iterdir() if x.is_file()}
        if (entry['completion_path']!=str((folder/'COMPLETED.json').relative_to(ROOT))
            or entry['completion_sha256']!=sha(folder/'COMPLETED.json') or entry['artifact_sha256']!=expected_files):
            raise ValueError('Original frozen completion/artifact map differs')
        expected=m['folds']['0']['variant_roles'][variant]['selection']
        if ((folder/'FAILED.json').exists() or done['status']!='completed' or s['status']!='completed' or s['fixture_only']
            or s.get('format')!='r13_physics_lgbm_v1' or s.get('kind')!='lgbm_physics' or done.get('reload1_7_exact') is not True
            or s['train_rows']!=78869 or s['inner_rows']!=1000 or s['n_features']!=5252
            or done['summary_sha256']!=sha(folder/'model/summary.json') or s['training_binding']!=done['binding']
            or b['R13_bundle_data_hash']!=m['data_hash'] or b['source_hash']!=m['source_hash'] or b['fold']!=0
            or b['variant']!=variant or b['include_shape'] is not True or s['seed']!=source_cfg['seed']
            or b['selection_identity']['npz_sha256']!=expected['sha256']
            or sha(folder/'epoch1_train.npz')!=b['TRAIN_materialization_sha256']
            or sha(folder/'model/basis.npz')!=m['folds']['0']['basis_sha256']):raise ValueError('Exact completed full Fold0 source/selection/PCA reuse required')
        if (s['settings']['maximum_trees']!=1500 or s['settings']['early_stopping_patience']!=40 or s['settings']['threads']!=2
            or s['include_shape'] is not True or set(s['code_sha256'])!=set(CODE_NAMES)):
            raise ValueError('Fold0 learner settings/source closure differs')
        for n,h in s['code_sha256'].items():
            if sha(ROOT/n)!=h:raise ValueError('Original fold0 model source changed')
        files={str((folder/'model'/n).relative_to(ROOT)):h for n,h in s['artifact_sha256'].items()}
        files[str((folder/'model/summary.json').relative_to(ROOT))]=sha(folder/'model/summary.json')
        for n,h in files.items():
            if sha(ROOT/n)!=h:raise ValueError('Original Fold0 artifact changed')
        out[arm]={'fold':0,'variant':variant,'model_dir':str((folder/'model').relative_to(ROOT)),'reused_completed_Fold0':True,
            'trained_under_previous_diagnostic_approval':True,'current_user_approved_exact_weight_reuse':True,
            'completion_path':str((folder/'COMPLETED.json').relative_to(ROOT)),'completion_sha256':sha(folder/'COMPLETED.json'),
            'summary_sha256':sha(folder/'model/summary.json'),'artifact_sha256':files,
            'TRAIN_rows':78869,'feature_count':5252,'selection_rows':1000,'basis_sha256':m['folds']['0']['basis_sha256']}
    return out

def prepare(config,approval,out,validate_only=False):
    import pandas as pd
    from pipeline.full_data_r13 import validate_full_bundle,DATA_CODE
    from pipeline.trees_ensemble_r13 import CODE_NAMES,feature_schema
    c,a=approved(config,approval);m=validate_full_bundle(ROOT/BUNDLE,verify_all=True);sc=read(ROOT/SOURCE_CONFIG)
    if m['data_hash']!=DATA_HASH or [m['folds'][str(i)]['TRAIN_rows'] for i in range(5)]!=ROWS:raise ValueError('Exact full fivefold source required')
    reuse=fold0_reuse(m,sc);checks={};paths=[Path(__file__),config,approval,ROOT/SOURCE_CONFIG,ROOT/BUNDLE/'manifest.json',ROOT/BUNDLE/'readiness.json']
    paths += [ROOT/n for n in sorted(set(DATA_CODE+CODE_NAMES))]
    paths += [ROOT/'models/r14_tree_full_r1/ALL_TREES_FROZEN.json']
    for info in reuse.values():
        paths += [ROOT/info['completion_path']]+[ROOT/n for n in info['artifact_sha256']]
    cols=['base_window_index','background_group','sensor_group','start_export_clock']
    for fold in range(5):
        entry=m['folds'][str(fold)];tp=ROOT/BUNDLE/f'fold_{fold:02d}/TRAIN_pool.parquet';t=pd.read_parquet(tp,columns=cols);paths += [tp,ROOT/BUNDLE/entry['basis_path']]
        if len(t)!=ROWS[fold] or len(set(t.base_window_index))!=ROWS[fold]:raise ValueError('Unique full TRAIN counts differ')
        checks[str(fold)]={}
        for variant in VARIANTS:
            e=entry['variant_roles'][variant]['selection'];metadata_name=f'fold_{fold:02d}/{variant}_selection_metadata.parquet'
            mp=ROOT/BUNDLE/metadata_name
            if sha(mp)!=m['file_sha256'][metadata_name]:raise ValueError('Fixed selection metadata changed')
            v=pd.read_parquet(mp,columns=cols)
            if len(v)!=1000:raise ValueError('Own selection1000 required')
            checks[str(fold)][variant]=separation(t,v)
            paths += [ROOT/BUNDLE/e['path'],mp]
    pins={str(p.resolve().relative_to(ROOT)):sha(p) for p in paths}
    p={'format':'r14_approved_fivefold_fulltree_production_plan_v1','production_recipe_approved':True,
        'created_unix':time.time(),'production_config':str(config.relative_to(ROOT)),'production_config_sha256':sha(config),
        'approval_file':str(approval.relative_to(ROOT)),'approval_file_sha256':sha(approval),
        'user_instruction_file':USER_FILE,'user_instruction_sha256':USER_SHA,'source_config':SOURCE_CONFIG,'bundle':BUNDLE,
        'data_hash':m['data_hash'],'source_hash':m['source_hash'],'fold_rows':ROWS,'variants':list(VARIANTS),'folds':list(range(5)),
        'new_fit_folds':[1,2,3,4],'new_tree_learners':8,'new_boosters':16,'reused_tree_learners':2,'total_tree_learners':10,
        'source_seed':sc['seed'],'tree_seed_policy':'source_seed+fold*10007','settings':dict(sc['tree'],threads=2),
        'feature_count':5252,'feature_schema_sha256':token(feature_schema()),'fold0_reuse':reuse,'source_separation':checks,
        'source_pins':pins,'free_disk_bytes_before_start':disk(),'features_saved_to_disk':False,'minimum_free_disk_bytes':2*1024**3,
        'execution':'separate fresh CPU2 child per learner, sequential; completed folders freeze; partial learner never resumed',
        'selection_policy':'own variant selection1000 for ES1500/40 only; no report/query/modelweight tuning here',
        'R13_and_R14_old_diagnostics_modified':False,'query_report_arrays_or_external_submission_or_SSH':False,
        'shared509_references_not_reference_independent_cv':True}
    p['plan_sha256']=token(p)
    if validate_only:return p
    out.mkdir(parents=True,exist_ok=False);write(out/'PRE_FIT_FREEZE.json',p)
    write(out/'fold_00/REUSE_COMPLETED_MODELS.json',{'models':reuse,'original_files_modified':False,'symlinks_or_weight_copies':False,'plan_sha256':p['plan_sha256']})
    return p

def load_plan(out):
    p=read(out/'PRE_FIT_FREEZE.json')
    if token({k:v for k,v in p.items() if k!='plan_sha256'})!=p['plan_sha256'] or p['production_recipe_approved'] is not True:raise ValueError('Approved immutable plan required')
    verify(p);return p

def materialize(source,fold,folder,p):
    import numpy as np
    from pipeline.full_data_r13 import DynamicR13Dataset,observed_bits
    from pipeline.relative_grid_r6_v2 import relative_grid_v2
    from pipeline.trees_ensemble_r13 import array_sha
    rows=DynamicR13Dataset(source,fold,epoch=1);n=len(rows)
    if n!=ROWS[fold] or not rows.epoch_audit()['primary_once']:raise ValueError('Complete primary epoch required')
    db=np.empty((n,15,25),np.float32);q=np.empty_like(db);y=np.empty(n,np.float32);z=np.empty(n,np.int64)
    fields={k:[] for k in ['site','sample_id','source_id','background_group','base_window_index','reference_index','family','explicit_no_source','highshare']}
    begin=time.perf_counter();maximum=0.
    for i in range(n):
        a=rows.raw_item(i);db[i]=a['db'];q[i]=a['cell_share'];y[i]=a['share'];z[i]=a['presence']
        for k in fields:fields[k].append(a[k])
        if (i+1)%8192==0 or i==n-1:
            write(folder/'materialize_progress.json',{'rows':i+1,'total':n,'elapsed_seconds':time.perf_counter()-begin,'peak_RSS_bytes':peak()})
    data={'db':db,'cell_share':q,'share':y,'presence':z,**{k:np.asarray(v) for k,v in fields.items()}}
    if (len(set(data['base_window_index']))!=n or not np.array_equal(y>0,z.astype(bool))
        or not np.isfinite(q).all() or ((q<0)|(q>1)).any() or np.any(q[~np.isfinite(db)]!=0)
        or (np.isfinite(db).all(2).sum(1)<3).any()):raise ValueError('Primary/labels/masks invalid')
    observed_bits(db)
    for i in range(0,n,512):
        _,w=relative_grid_v2(db[i:i+512]);maximum=max(maximum,float(abs((w*q[i:i+512]).sum((1,2))-y[i:i+512]).max()))
    if maximum>.0058:raise ValueError('Frozen quantization pooling bound failed')
    tmp=folder/'epoch1_train.tmp.npz';np.savez_compressed(tmp,**data);tmp.replace(folder/'epoch1_train.npz')
    receipt={'format':'r14_production_full_epoch1_materialization_v1','fold':fold,'variant':source.variant,'rows':n,'identity':rows.identity,
        'schedule_audit':rows.epoch_audit(),'typed_SHA':{k:array_sha(v) for k,v in data.items()},
        'NPZ_sha256':sha(folder/'epoch1_train.npz'),'max_raw_label_rounded_pool_gap':maximum,'full_primary_once':True,
        'elapsed_seconds':time.perf_counter()-begin,'peak_RSS_bytes':peak()}
    write(folder/'epoch1_manifest.json',receipt);return data,rows.identity

def worker(out,fold,variant):
    env()
    import numpy as np
    from pipeline.full_data_r13 import R13Sources,FixedR13Dataset
    from pipeline.trees_ensemble_r13 import PhysicsLGBM
    p=load_plan(out);disk()
    if (out/'FAILED.json').exists() or (out/'ALL_FIVEFOLD_TREES_FROZEN.json').exists():raise ValueError('Failed/completed aggregate cannot start another learner')
    if fold not in (1,2,3,4) or variant not in VARIANTS:raise ValueError('Only fresh remaining approved learners')
    folder=out/f'fold_{fold:02d}'/(variant+'P1');folder.mkdir(parents=True,exist_ok=False)
    write(folder/'STARTED.json',{'pid':os.getpid(),'started_unix':time.time(),'fold':fold,'variant':variant,'rows':ROWS[fold],'CPU_threads':2,'plan_sha256':p['plan_sha256']})
    try:
        policy={'format':'r14_approved_production_source_policy_v1','data_variant':variant,'include_shape':True,'production_recipe_approved':True,
            'user_instruction_sha256':USER_SHA,'production_config_sha256':p['production_config_sha256'],'approval_file_sha256':p['approval_file_sha256']}
        policy['policy_sha256']=token(policy);write(folder/'APPROVED_SOURCE_POLICY.json',policy)
        source=R13Sources(ROOT/SOURCE_CONFIG,bundle=ROOT/BUNDLE,verify_banks=True,runtime_policy=policy)
        train,identity=materialize(source,fold,folder,p);selection=FixedR13Dataset(source,fold,'selection');val=selection.raw_arrays()
        for k in ('sample_id','base_window_index','source_id','background_group'):
            if set(train[k].astype(str))&set(val[k].astype(str)):raise ValueError('TRAIN/selection overlap:'+k)
        m=read(ROOT/BUNDLE/'manifest.json');entry=m['folds'][str(fold)]
        b={'format':'r14_approved_full_tree_production_binding_v1','fold':fold,'variant':variant,'include_shape':True,
            'production_config_sha256':p['production_config_sha256'],'approval_file_sha256':p['approval_file_sha256'],
            'user_instruction_sha256':USER_SHA,'plan_sha256':p['plan_sha256'],'data_hash':p['data_hash'],'source_hash':p['source_hash'],
            'TRAIN_identity':identity,'TRAIN_npz_sha256':sha(folder/'epoch1_train.npz'),'selection_identity':selection.identity,
            'PCA_sha256':entry['basis_sha256'],'source_separation':p['source_separation'][str(fold)][variant],
            'feature_count':5252,'executor':'Mac CPU2','executor_source_sha256':sha(__file__)}
        write(folder/'BINDING_BEFORE_FIT.json',b)
        print(json.dumps({'stage':'fit_start','fold':fold,'variant':variant,'rows':ROWS[fold],'features':5252,'free_disk_bytes':disk()}),flush=True)
        tree=PhysicsLGBM(p['settings'],seed=p['source_seed']+fold*10007,basis_path=ROOT/BUNDLE/entry['basis_path'],
            basis_training_binding=entry['basis_training_binding'],include_shape=True)
        s=tree.fit(dict(train,role='train'),dict(val,role='selection'),folder/'model',b,fixture=False)
        restored=PhysicsLGBM.load(folder/'model',expected_binding=b);a=tree.predict(val['db'][:7],batch_size=7);q=restored.predict(val['db'][:7],batch_size=1)
        if any(not np.array_equal(a[k],q[k]) for k in a):raise ValueError('Saved reload batch1/7 exact parity failed')
        source.assert_unchanged();verify(p)
        info={'format':'r14_approved_production_tree_completed_v1','status':'completed','fold':fold,'variant':variant,'production_recipe_approved':True,
            'model_dir':str((folder/'model').relative_to(ROOT)),'TRAIN_rows':ROWS[fold],'selection_rows':1000,'feature_count':5252,
            'summary_sha256':sha(folder/'model/summary.json'),'artifact_sha256':{str((folder/'model'/n).relative_to(ROOT)):h for n,h in s['artifact_sha256'].items()},
            'binding':b,'best_selection_rmse':s['best_inner_rmse'],'regressor_best_round':s['best_iteration'],'classifier_best_round':s['presence_best_iteration'],
            'fit_seconds':s['elapsed_seconds'],'peak_RSS_bytes':peak(),'reload1_7_exact':True,'actual_learner_fits':1,'actual_boosters':2,
            'Test_report_or_new_policy_evaluation':False}
        write(folder/'COMPLETED.json',info);print(json.dumps({'stage':'fit_completed',**info}),flush=True)
    except BaseException as e:
        write(folder/'FAILED.json',{'error':repr(e),'partial_learner_resume_or_reuse':False,'peak_RSS_bytes':peak()});raise

def completed(out,p,persist=True):
    models={'0':p['fold0_reuse']}
    for fold in (1,2,3,4):
        models[str(fold)]={}
        for variant in VARIANTS:
            arm=variant+'P1';folder=out/f'fold_{fold:02d}'/arm;c=read(folder/'COMPLETED.json')
            if (folder/'FAILED.json').exists() or c['status']!='completed' or c['TRAIN_rows']!=ROWS[fold] or c['feature_count']!=5252:raise ValueError('Completed full learner required')
            if sha(ROOT/c['model_dir']/'summary.json')!=c['summary_sha256']:raise ValueError('Summary changed')
            for n,h in c['artifact_sha256'].items():
                if sha(ROOT/n)!=h:raise ValueError('Completed artifact changed')
            models[str(fold)][arm]={**c,'completion_path':str((folder/'COMPLETED.json').relative_to(ROOT)),
                'completion_sha256':sha(folder/'COMPLETED.json'),'reused_completed_Fold0':False}
    verify(p)
    for arm,c in models['0'].items():
        for n,h in c['artifact_sha256'].items():
            if sha(ROOT/n)!=h:raise ValueError('Reused original Fold0 changed')
    value={'format':'r14_approved_fulltree_fivefold_completed_v1','status':'completed','production_recipe_approved':True,
        'plan_sha256':p['plan_sha256'],'production_config_sha256':p['production_config_sha256'],'approval_file_sha256':p['approval_file_sha256'],
        'user_instruction_sha256':USER_SHA,'folds':[0,1,2,3,4],'models':models,'new_learner_fits':8,'reused_completed_learners':2,
        'new_boosters_fitted':16,'total_boosters':20,'source_pins':p['source_pins'],'old_diagnostics_unchanged':True,
        'Test_report_or_policy_weights_selected_here':False}
    value['freeze_sha256']=token(value)
    if persist:write(out/'ALL_FIVEFOLD_TREES_FROZEN.json',value)
    elif read(out/'ALL_FIVEFOLD_TREES_FROZEN.json')!=value:raise ValueError('Frozen fivefold registry differs')
    return value

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('action',choices=['plan','fit','worker','verify'])
    a.add_argument('--config',required=True);a.add_argument('--approval',required=True);a.add_argument('--output',default=OUT)
    a.add_argument('--validate-only',action='store_true');a.add_argument('--fold',type=int);a.add_argument('--variant',choices=VARIANTS)
    args=a.parse_args();config=local(args.config);approval=local(args.approval);out=local(args.output);env()
    if args.action=='plan':
        p=prepare(config,approval,out,args.validate_only);print(json.dumps({'status':'validated' if args.validate_only else 'plan_frozen','plan_sha256':p['plan_sha256'],'fit_performed':False}));return
    if args.action=='worker':worker(out,args.fold,args.variant);return
    if args.action=='verify':p=load_plan(out);v=completed(out,p,False);print(json.dumps({'status':'verified','freeze_sha256':v['freeze_sha256']}));return
    import fcntl
    p=load_plan(out)
    if (str(config.relative_to(ROOT))!=p['production_config'] or sha(config)!=p['production_config_sha256']
        or str(approval.relative_to(ROOT))!=p['approval_file'] or sha(approval)!=p['approval_file_sha256']):raise ValueError('Explicit runtime config/approval differs from plan')
    if any((out/f).exists() for f in ('FAILED.json','ALL_FIVEFOLD_TREES_FROZEN.json')):raise ValueError('No failed or completed driver reuse')
    with (out/'RUNNING.lock').open('x') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(json.dumps({'pid':os.getpid(),'plan_sha256':p['plan_sha256']})+'\n');lock.flush()
        try:
            done=[]
            for fold in (1,2,3,4):
                for variant in VARIANTS:
                    disk();log=out/f'fold_{fold:02d}_{variant}P1.log'
                    with log.open('x') as f:
                        command=[sys.executable,'-m','ops.train_r14_production_trees','worker','--config',str(config),'--approval',str(approval),'--output',str(out),'--fold',str(fold),'--variant',variant]
                        proc=subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT)
                    if proc.returncode:raise RuntimeError(f'Preserved failure fold{fold}/{variant},exit{proc.returncode}')
                    done.append({'fold':fold,'variant':variant});write(out/'progress.json',{'status':'running','completed_new_learners':done,'total_new_learners':8,'updated_unix':time.time()})
                    print(json.dumps({'stage':'learner_completed','fold':fold,'variant':variant,'completed':len(done),'total':8}),flush=True)
            v=completed(out,p);write(out/'progress.json',{'status':'completed','completed_new_learners':done,'freeze_sha256':v['freeze_sha256']})
        except BaseException as e:
            write(out/'FAILED.json',{'error':repr(e),'partial_learner_resume':False,'completed_artifacts_preserved':True});raise
        finally:(out/'RUNNING.lock').unlink(missing_ok=True)

if __name__=='__main__':main()
