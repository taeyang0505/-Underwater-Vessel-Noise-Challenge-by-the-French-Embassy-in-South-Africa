"""R14 full Fold0 CPU diagnostic; no R13 production guard is changed.

Uses the original full5252 PhysicsLGBM, full epoch1 TRAIN materialization,
own-variant selection for ES, and original fold-TRAIN PCA. Both learners
freeze before the same17 exposed development panels are evaluated. This
does not approve or call the production CUDA/Mac-stage orchestrator.
"""
from __future__ import annotations
from pathlib import Path
import argparse,json,os,sys,time,subprocess,resource
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CONFIG='configs/r13_full_training_v1.json'
BUNDLE='data/synthetic/group_cv_physics_r13_variants_v2'
OUT='models/r14_tree_full_r1'
REPORT='reports/r14_tree_full_r1'
VARIANTS=('baseline','highshare')
EXPECTED_DATA='bb1dc465007a70798968ce14040b59643715538716b9e2c1f11c77e72c27d3c5'

def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def token(v):
    import hashlib
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def env(threads=2):
    from ops.train_r12_lightgbm import environment,guard
    environment(threads);sys.addaudithook(guard)

def memory():
    x=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(x if sys.platform=='darwin' else x*1024)

def verify(plan):
    for n,h in plan['source_pins'].items():
        if sha(ROOT/n)!=h:raise ValueError('Pinned source/data changed:'+n)

def plan(out):
    from pipeline.full_data_r13 import validate_full_bundle,DATA_CODE
    from pipeline.trees_ensemble_r13 import CODE_NAMES,feature_schema
    m=validate_full_bundle(ROOT/BUNDLE,verify_all=True);cfg=json.loads((ROOT/CONFIG).read_text())
    if m['data_hash']!=EXPECTED_DATA or m['folds']['0']['TRAIN_rows']!=78869:raise ValueError('Exact full Fold0 identity required')
    f=m['folds']['0'];paths=[Path(__file__),ROOT/CONFIG,ROOT/BUNDLE/'manifest.json',ROOT/BUNDLE/'readiness.json',
        ROOT/BUNDLE/f['basis_path'],ROOT/BUNDLE/'fold_00/TRAIN_pool.parquet']
    paths += [ROOT/n for n in sorted(set(DATA_CODE+CODE_NAMES))]
    for v in VARIANTS:
        e=f['variant_roles'][v]['selection'];paths.append(ROOT/BUNDLE/e['path'])
        for k in ('metadata_path','recipes_path'):
            if k in e:paths.append(ROOT/BUNDLE/e[k])
    paths += [ROOT/'reports/r13_cpu_correction_r1/RESULTS.json',ROOT/'reports/r13_cpu_correction_r1/selected_policy.json',
        ROOT/'ops/evaluate_r12_ablation.py',ROOT/'ops/verify_r12_evaluation.py']
    p={'format':'r14_actual_full_tree_CPU_diagnostic_plan_v1','created_before_any_fit':True,'created_unix':time.time(),
        'variants':list(VARIANTS),'full_train_rows_each':78869,'fold':0,'epoch':1,'feature_count':5252,
        'feature_schema':feature_schema(),'seed':cfg['seed'],'settings':dict(cfg['tree'],threads=2),
        'config_path':CONFIG,'bundle_path':BUNDLE,'data_hash':m['data_hash'],'source_hash':m['source_hash'],
        'fold_basis_path':f['basis_path'],'fold_basis_sha256':f['basis_sha256'],'basis_training_binding':f['basis_training_binding'],
        'source_pins':{str(p.relative_to(ROOT)):sha(p) for p in paths},
        'source_policy':'Original R13 epoch1 dynamic raw_item, each primary exactly once; no subsampling; P1 fold-TRAIN PCA unchanged',
        'selection_policy':'Each variant uses its own prepared selection1000; source/geometry paired but highshare alpha/targets differ; only this role used for boosting ES1500/40',
        'fit_gate_report_roles':{'train':'Fold0 TRAIN78869','selection':'own variant selection1000 for ES only',
            'ensemble_fit':'not used','ensemble_gate':'not used','R13_report':'not opened; no fivefold production/report gate bypass'},
        'evaluation':'Both models durably frozen first; same17 exposed R12/R9/R10/R11 development panels only; raw regressions/classifiers preserved',
        'score_policy':'Fulltree score=share, independent classifier and fixed halfshare/halfclassifier diagnostic reported separately; no recipe/lambda selection here',
        'scope':'Authorized R14 CPU-only diagnostic, not an R13 production run; original production config/runtime approval remainfalse',
        'original_R12_R13_mutations':False,'Test_GPU_SSH_external_transmission':False,
        'resources':{'CPU_threads':2,'sequential_variants':True,'train_float32_feature_bytes':78869*5252*4,
            'working_memory_estimate_GiB':[6,12],'memory_estimate_not_measurement':True},
        'interpretation_limits':['Full vs compact comparison changes rows,source law,fold-PCA and ES observations as well as feature count',
            'Shared reference recordings and exposed development panels; no blind OOF or official loss claim']}
    p['plan_sha256']=token(p);out.mkdir(parents=True,exist_ok=False);write(out/'PRE_FIT_FREEZE.json',p)
    return p

def load_plan(out):
    p=json.loads((out/'PRE_FIT_FREEZE.json').read_text())
    if token({k:v for k,v in p.items() if k!='plan_sha256'})!=p['plan_sha256']:raise ValueError('Plan changed')
    verify(p);return p

def materialize(source,variant,folder,plan):
    import numpy as np
    from pipeline.full_data_r13 import DynamicR13Dataset,observed_bits
    from pipeline.relative_grid_r6_v2 import relative_grid_v2
    from pipeline.trees_ensemble_r13 import array_sha
    rows=DynamicR13Dataset(source,0,epoch=1);n=len(rows)
    if n!=78869 or not rows.epoch_audit()['primary_once']:raise ValueError('Full78869 once required')
    db=np.empty((n,15,25),np.float32);q=np.empty_like(db);y=np.empty(n,np.float32);presence=np.empty(n,np.int64)
    fields={k:[] for k in ['site','sample_id','source_id','background_group','base_window_index','reference_index','family','explicit_no_source','highshare']}
    began=time.perf_counter();raw_error=rounded_error=0.
    for i in range(n):
        a=rows.raw_item(i);db[i]=a['db'];q[i]=a['cell_share'];y[i]=a['share'];presence[i]=a['presence']
        for k in fields:fields[k].append(a[k])
        if i%4096==0 or i==n-1:
            write(folder/'materialize_progress.json',{'variant':variant,'rows':i+1,'total':n,'elapsed_seconds':time.perf_counter()-began,'peak_RSS_bytes':memory()})
            print(json.dumps({'stage':'materialize','variant':variant,'rows':i+1,'total':n,'seconds':time.perf_counter()-began,'peak_RSS_bytes':memory()}),flush=True)
    if len(set(fields['base_window_index']))!=n or not np.array_equal(np.sort(rows.schedule['primary_order']),np.arange(n)):
        raise ValueError('Full primary uniqueness failed')
    for k,v in fields.items():fields[k]=np.asarray(v)
    data={'db':db,'cell_share':q,'share':y,'presence':presence,**fields}
    if not np.array_equal(y>0,presence.astype(bool)) or int((y==0).sum())!=15774:raise ValueError('Epoch1 exact count/target failed')
    if np.any(q[~np.isfinite(db)]!=0) or not np.isfinite(q).all() or ((q<0)|(q>1)).any():raise ValueError('Cell labels/missing invalid')
    for i in range(0,n,512):
        _,w=relative_grid_v2(db[i:i+512]);rounded_error=max(rounded_error,float(abs((w*q[i:i+512]).sum((1,2))-y[i:i+512]).max()))
    if rounded_error>.0058:raise ValueError('Rounded pooling mismatch outside frozen quantization bound')
    mask=observed_bits(db)
    if not np.array_equal(np.asarray([int(v).bit_count() for v in mask]),np.isfinite(db).all(2).sum(1)):raise ValueError('Mask conversion')
    path=folder/'epoch1_train.npz';np.savez_compressed(path,**data)
    receipt={'format':'r14_full_epoch1_materialization_v1','variant':variant,'fold':0,'epoch':1,'rows':n,
        'identity':rows.identity,'schedule_audit':rows.epoch_audit(),'typed_SHA':{k:array_sha(v) for k,v in data.items()},
        'NPZ_sha256':sha(path),'max_raw_label_rounded_pool_gap':rounded_error,'full_primary_once':True,
        'seconds':time.perf_counter()-began,'peak_RSS_bytes':memory(),'TRAIN_only':True,
        'no_new_labels_inferred_from_SA':'Target is controlled added whole-recording energy, not natural vessel truth'}
    write(folder/'epoch1_manifest.json',receipt)
    return data,rows.identity

def worker(out,variant):
    env(2)
    import numpy as np
    from pipeline.full_data_r13 import R13Sources,FixedR13Dataset
    from pipeline.trees_ensemble_r13 import PhysicsLGBM,array_sha
    p=load_plan(out);folder=out/(variant+'P1');folder.mkdir(exist_ok=False)
    # This source-only description is NOT an approved R13 runtime policy and
    # is never passed to any production runner/launcher approval check.
    source_policy={'format':'r14_source_only_CPU_diagnostic_variant_v1','data_variant':variant,'include_shape':True,
        'production_recipe_approved':False,'R13_production_launch_authorized':False,'R14_diagnostic_only':True}
    source_policy['policy_sha256']=token(source_policy);write(folder/'DIAGNOSTIC_SOURCE_POLICY.json',source_policy)
    try:
        source=R13Sources(ROOT/CONFIG,bundle=ROOT/BUNDLE,verify_banks=True,runtime_policy=source_policy)
        train,identity=materialize(source,variant,folder,p)
        selection=FixedR13Dataset(source,0,'selection');val=selection.raw_arrays()
        if len(val['share'])!=1000:raise ValueError('Own selection1000 required')
        for key in ('sample_id','base_window_index','source_id','background_group'):
            if set(train[key].astype(str))&set(val[key].astype(str)):raise ValueError('TRAIN/selection source overlap:'+key)
        binding={'format':'r14_full_tree_diagnostic_binding_v1','plan_sha256':p['plan_sha256'],'R13_bundle_data_hash':p['data_hash'],
            'source_hash':p['source_hash'],'fold':0,'variant':variant,'include_shape':True,'TRAIN_identity':identity,
            'TRAIN_materialization_sha256':sha(folder/'epoch1_train.npz'),'selection_identity':selection.identity,
            'fold_basis_sha256':p['fold_basis_sha256'],'feature_count':5252,'diagnostic_only':True,
            'production_approval_changed':False,'executor_source_sha256':sha(__file__)}
        write(folder/'BINDING_BEFORE_FIT.json',binding)
        begin=time.perf_counter();print(json.dumps({'stage':'full_fit_start','variant':variant,'TRAIN_rows':len(train['share']),
            'selection_rows':len(val['share']),'features':5252,'feature_bytes':len(train['share'])*5252*4,'peak_RSS_bytes':memory()}),flush=True)
        tree=PhysicsLGBM(p['settings'],seed=p['seed'],basis_path=ROOT/BUNDLE/p['fold_basis_path'],
            basis_training_binding=p['basis_training_binding'],include_shape=True)
        summary=tree.fit(dict(train,role='train'),dict(val,role='selection'),folder/'model',binding,fixture=False)
        restored=PhysicsLGBM.load(folder/'model',expected_binding=binding)
        a=tree.predict(val['db'][:7],batch_size=7);b=restored.predict(val['db'][:7],batch_size=1)
        if any(not np.array_equal(a[k],b[k]) for k in a):raise ValueError('Saved full model batch1/7 parity failed')
        source.assert_unchanged();verify(p)
        receipt={'format':'r14_full_tree_completed_v1','status':'completed','variant':variant,'TRAIN_rows':78869,'features':5252,
            'binding':binding,'summary_sha256':sha(folder/'model/summary.json'),'regressor_sha256':sha(folder/'model/share.txt'),
            'classifier_sha256':sha(folder/'model/presence.txt'),'PCA_sha256':p['fold_basis_sha256'],
            'best_selection_rmse':summary['best_inner_rmse'],'best_round':summary['best_iteration'],
            'classifier_best_round':summary['presence_best_iteration'],'fit_seconds':summary['elapsed_seconds'],
            'peak_RSS_bytes':memory(),'reload1_7_exact':True,'actual_tree_learner_fit_calls':1,'actual_boosters_fitted':2,
            'gate_report_development_Test_read':False}
        write(folder/'COMPLETED.json',receipt)
        print(json.dumps({'stage':'full_fit_completed',**receipt}),flush=True)
    except BaseException as e:
        write(folder/'FAILED.json',{'error':repr(e),'partial_reuse_allowed':False,'peak_RSS_bytes':memory()});raise

def freeze(out):
    p=load_plan(out);models={}
    for variant in VARIANTS:
        folder=out/(variant+'P1');c=json.loads((folder/'COMPLETED.json').read_text())
        if (folder/'FAILED.json').exists() or c['status']!='completed' or c['TRAIN_rows']!=78869 or c['features']!=5252:raise ValueError('Both full variants must complete')
        s=json.loads((folder/'model/summary.json').read_text())
        if sha(folder/'model/summary.json')!=c['summary_sha256'] or s['fixture_only'] or s['train_rows']!=78869:raise ValueError('Exact full model identity required')
        artifacts={str(x.relative_to(out)):sha(x) for x in (folder/'model').iterdir() if x.is_file()}
        models[variant+'P1']={'completion_path':str((folder/'COMPLETED.json').relative_to(ROOT)),
            'completion_sha256':sha(folder/'COMPLETED.json'),'artifact_sha256':artifacts}
    f={'format':'r14_full_two_tree_global_freeze_v1','status':'completed','all_models_frozen':True,
        'plan_sha256':p['plan_sha256'],'models':models,'gate_report_development_Test_read_before_this_freeze':False}
    f['freeze_sha256']=token(f);write(out/'ALL_TREES_FROZEN.json',f)
    print(json.dumps({'stage':'all_full_trees_frozen','freeze_sha256':f['freeze_sha256']}),flush=True)
    return f

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','fit','worker','freeze','evaluate'])
    p.add_argument('--output',default=OUT);p.add_argument('--report',default=REPORT);p.add_argument('--variant',choices=VARIANTS)
    a=p.parse_args();out=(ROOT/a.output).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Project-contained path required')
    if a.action=='plan':env();v=plan(out);print(json.dumps({'status':'plan_frozen','plan_sha256':v['plan_sha256']}));return
    if a.action=='worker':
        if not a.variant:p.error('--variant required')
        worker(out,a.variant);return
    if a.action=='freeze':env();freeze(out);return
    if a.action=='evaluate':evaluate(out,(ROOT/a.report).resolve());return
    env();load_plan(out)
    for variant in VARIANTS:
        log=(out/(variant+'P1.log')).open('x')
        try:
            child=subprocess.run([sys.executable,'-m','ops.train_r14_tree_full','worker','--output',str(out),'--variant',variant],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        finally:log.close()
        if child.returncode:
            write(out/'FAILED.json',{'variant':variant,'exit_code':child.returncode,'partial_reuse_allowed':False});raise RuntimeError('Preserved failed full-tree worker:'+variant)
        print(json.dumps({'stage':'variant_completed','variant':variant,'receipt':str(out/(variant+'P1')/'COMPLETED.json')}),flush=True)
    freeze(out)

def evaluate(out,report):
    env(2)
    import numpy as np,pandas as pd
    from ops import evaluate_r12_ablation as parent
    from pipeline.trees_ensemble_r13 import PhysicsLGBM
    from ops.verify_r12_evaluation import cluster_ci
    p=load_plan(out);f=json.loads((out/'ALL_TREES_FROZEN.json').read_text())
    if ((out/'FAILED.json').exists() or f['all_models_frozen'] is not True
        or token({k:v for k,v in f.items() if k!='freeze_sha256'})!=f['freeze_sha256']):raise ValueError('Both full trees frozen first')
    trees={}
    for name,v in f['models'].items():
        if sha(ROOT/v['completion_path'])!=v['completion_sha256']:raise ValueError('Completion changed')
        for a,h in v['artifact_sha256'].items():
            if sha(out/a)!=h:raise ValueError('Frozen model changed:'+a)
        folder=out/name/'model';s=json.loads((folder/'summary.json').read_text())
        trees[name]=PhysicsLGBM.load(folder,expected_binding=s['training_binding'])
    prior=ROOT/'reports/r13_cpu_correction_r1';original=json.loads((prior/'RESULTS.json').read_text())
    for n,h in original['output_sha256'].items():
        if sha(prior/n)!=h:raise ValueError('Original R13 prediction changed:'+n)
    oldplan,oldf=parent.check_freeze('reports/R12_ALL17_GLOBAL_FREEZE.json')
    oldpolicy=ROOT/'reports/r12_cpu_ablation_r1/selected_policy.json'
    parent.check_policy(oldpolicy,oldf['freeze_sha256'])
    report.mkdir(parents=True,exist_ok=False);(report/'predictions').mkdir()
    write(report/'FROZEN_BEFORE_EVALUATION.json',{'all2_frozen_file_sha256':sha(out/'ALL_TREES_FROZEN.json'),
        'freeze_sha256':f['freeze_sha256'],'plan_sha256':p['plan_sha256'],'any_recipe_or_score_selection':False,
        'all_panels_development_exposed':True,'original_R13_policy_changed':False})
    panels=[parent.load_R12(oldplan,'validation_gate')]
    for role in ('eval','high_snr_long'):panels.append(parent.load_R12(oldplan,role,oldpolicy,oldf['freeze_sha256']))
    dev,r11=parent.load_development();panels.extend(dev)
    for panel in panels:
        if panel['name'].startswith('R10_unseen__'):
            meta=pd.read_parquet(ROOT/parent.R10/'unseen_source_manifest.parquet')
            if len(meta)!=len(panel['y']):raise ValueError('Unseen row metadata')
            panel['groups']=(meta.sensor_group.astype(str)+'__'+meta.recording_week_start.astype(str)).to_numpy()
            panel['cluster_key']='sensor_group_x_recording_week_start'
    for pos in parent.DEV_R11_POS:
        for neg in parent.DEV_R11_NEG:
            a,b=r11[pos],r11[neg];panels.append({'name':a['name']+'_VS_'+neg,'y':np.r_[a['y'],b['y']],
                'db':np.concatenate([a['db'],b['db']]),'sample_id':np.r_[a['sample_id'],b['sample_id']],
                'groups':np.r_[a['groups'],b['groups']],'cluster_key':a['cluster_key'],'site':np.r_[a['site'],b['site']],
                'stage':'exposed_development'})
    if len(panels)!=17:raise ValueError('Same17 required')
    rows=[];cis=[];beg=time.perf_counter()

    def metric(panel,pred):
        m=parent.metrics(panel,pred);y=np.asarray(panel['y'],float);z=y>0;reg=np.asarray(pred['share'],float);score=np.asarray(pred['score'],float);e=reg-y;high=panel['y']>.2
        m.update(score_definition=pred.get('score_definition','original stored comparator score'),
            score_brier=float(np.mean((score-z)**2)),zero_prediction_mean=float(reg[~z].mean()) if (~z).any() else None,
            high_y_rmse=float(np.sqrt(np.mean(e[high]**2))) if high.any() else None,high_y_bias=float(e[high].mean()) if high.any() else None,
            share_ge099=float(np.mean(reg>=.99)),score_ge099=float(np.mean(score>=.99)))
        if 'score_classifier' in pred:
            c=np.asarray(pred['score_classifier'],float);mix=(.5*reg+.5*c).astype(np.float32)
            m.update(classifier_auc=parent.auc(y,c),classifier_brier=float(np.mean((c-z)**2)),classifier_ge099=float(np.mean(c>=.99)),
                diagnostic_halfshare_halfclassifier_auc=parent.auc(y,mix),diagnostic_halfshare_halfclassifier_brier=float(np.mean((mix.astype(float)-z)**2)))
        else:m.update(classifier_auc=None,classifier_brier=None,classifier_ge099=None,diagnostic_halfshare_halfclassifier_auc=None,diagnostic_halfshare_halfclassifier_brier=None)
        return m

    for panel in panels:
        name=panel['name'];predictions={};full={}
        for variant in VARIANTS:
            key=variant+'P1';value=trees[key].predict(panel['db'],batch_size=128)
            value['score_definition']='share; independent classifier separately retained'
            full[key]=value;predictions['FullTree_'+key]=value
        with np.load(prior/'predictions'/(name+'.npz'),allow_pickle=False) as z:
            if not np.array_equal(z['sample_id'].astype(str),panel['sample_id'].astype(str)) or not np.array_equal(z['target_share'],panel['y']):raise ValueError('Common panel IDs/targets changed')
            order=z['model_order'].astype(str).tolist()
            for arm in ('baselineP1','highshareP1'):
                j=order.index('Tree_'+arm);predictions['CompactTree_'+arm]={'share':z['share'][:,j],'score':z['score'][:,j],
                    'score_classifier':z[arm+'_classifier'],'score_definition':'original compact .5share+.5classifier'}
            for key in ('R12_matched_share_only','R13_selected','R9_frozen_all5'):
                if key in order:
                    j=order.index(key);predictions[key]={'share':z['share'][:,j],'score':z['score'][:,j]}
        for key,value in predictions.items():rows.append({'panel':name,'model':key,'development_not_blind':True,**metric(panel,value)})
        for variant in VARIANTS:
            key=variant+'P1';a=predictions['FullTree_'+key];b=predictions['CompactTree_'+key]
            ci=cluster_ci(name,panel['y'],panel['groups'],panel['cluster_key'],(a['share'],a['score_classifier']),(b['share'],b['score_classifier']))
            cis.extend({'panel':name,'comparison':'full_minus_compact_'+key,'metric':('auc_classifier' if k=='auc_score' else k),**v} for k,v in ci.items())
        np.savez_compressed(report/'predictions'/(name+'.npz'),sample_id=panel['sample_id'],target_share=panel['y'],
            model_order=np.asarray(list(predictions)),share=np.column_stack([v['share'] for v in predictions.values()]),
            score=np.column_stack([v['score'] for v in predictions.values()]),
            baselineP1_classifier=full['baselineP1']['score_classifier'],highshareP1_classifier=full['highshareP1']['score_classifier'])
        print(json.dumps({'stage':'panel_completed','panel':name,'baselineP1':metric(panel,predictions['FullTree_baselineP1']),
            'highshareP1':metric(panel,predictions['FullTree_highshareP1'])}),flush=True)
    parent.csv_rows(report/'results.csv',rows);parent.csv_rows(report/'paired_cluster_CI.csv',cis)
    verify(p)
    for n,h in original['output_sha256'].items():
        if sha(prior/n)!=h:raise ValueError('Original output changed')
    result={'format':'r14_full5252_two_variant_diagnostic_results_v1','status':'completed','panels':17,'metric_rows':len(rows),
        'plan_sha256':p['plan_sha256'],'source_pins':p['source_pins'],'global_freeze_file_sha256':sha(out/'ALL_TREES_FROZEN.json'),
        'global_freeze_sha256':f['freeze_sha256'],'models_root':str(out.relative_to(ROOT)),
        'all_full_models_train_rows':78869,'feature_count':5252,'actual_new_tree_learners':2,'actual_boosters_fitted':4,
        'own_variant_selection_ES':True,'any_new_ensemble_or_score_selection':False,'production_R13_approval_or_code_changed':False,
        'elapsed_evaluation_seconds':time.perf_counter()-beg,'Test_GPU_SSH_or_external_transmission':False,
        'limitations':p['interpretation_limits']+['R13 report role never opened;17 exposed panels are not a completed5fold production report',
            'Full and compact PCA/source/selection data differ; cannot attribute changes solely to5252features',
            'Full score=share versus compact legacy halfshare/classifier; use classifier/share AUC separately',
            'Conditional paired bootstrap uses shared references and20 sensor-week blocks; some adjacent blocks have15min gaps']}
    result['output_sha256']={str(x.relative_to(report)):sha(x) for x in report.rglob('*') if x.is_file()};write(report/'RESULTS.json',result)

if __name__=='__main__':main()

