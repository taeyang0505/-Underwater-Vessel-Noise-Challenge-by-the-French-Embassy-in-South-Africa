"""Sequential Mac CPU full-data R16 regression/classification trees, five folds.

Fresh learners only. Shared local_fit is development selection, never OOF.
Immutable per-fold data manifests allow preparation of later folds in parallel.
"""
from __future__ import annotations
import argparse,json,sys,time,os,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline.trees_r16_production import PhysicsLGBM,sha,write,token,DEFAULTS

def guard(event,args):
    if event=='open'and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0]).replace('\\','/').lower()
        if Path(p).name in('test.csv','sample_submission.csv')or '/eda_deep/'in p or '/r15_band_comparison_'in p:
            raise PermissionError('R16 training never opens challenge query or query-derived EDA')
def read(p):return json.loads(Path(p).read_text())
def verify(pins):
    for n,h in pins.items():
        if sha(ROOT/n)!=h:raise ValueError('Immutable training artifact/source changed: '+n)

def fixture():
    import tempfile,numpy as np
    from pipeline.trees_r16_production import transform
    from pipeline.trees_ensemble_r9 import transform_features as oldreg
    from ops.r16_clean_models import temporal_features
    rng=np.random.default_rng(2026102718);d=np.round(rng.normal(78,7,(480,15,25)),1).astype(np.float32);d[:100,0]=np.nan
    a=transform(d[:16],'regression');b=oldreg(d[:16],batch_size=5)
    c=transform(d[:16],'classification');v=temporal_features(d[:16])
    assert np.array_equal(a,b)and np.array_equal(c,v)
    shifted=d[:16]+np.round(rng.normal(0,5,25),1)[None,None]
    assert np.allclose(c,transform(shifted,'classification'),rtol=0,atol=2e-6)
    y=np.clip((d[:,7:9,15:20].mean((1,2))-75)/30,0,.85);y[::5]=0
    tr={'db':d[:400],'share':y[:400],'presence':y[:400]>0,'base_window_index':np.arange(400)}
    se={'db':d[400:],'share':y[400:],'presence':y[400:]>0,'base_window_index':np.arange(400,480),'sample_id':np.arange(80).astype(str)}
    # Actual learner/reload fixture, never a production partial or reuse source.
    with tempfile.TemporaryDirectory(dir='/private/tmp',prefix='r16_tree_fixture_')as t:
        folder=Path(t)/'model';m=PhysicsLGBM({'maximum_trees':25,'early_stopping_patience':5,'minimum_leaf_rows':8})
        s=m.fit(tr,se,folder,{'fixture':True});a=m.predict(se['db']);n=PhysicsLGBM.load(folder);b=n.predict(se['db'])
        assert all(np.array_equal(a[k],b[k])for k in a)
    result={'status':'passed','full3827_matches_validated_R9_relative8':True,'classifier700_matches_R16_pilot':True,
       'feature_offsets_invariant_with_quantization_tolerance':True,'learner_reload_exact':True,'rows_train':400,'rows_selection':80,
       'feature_counts':s['feature_counts'],'code_sha256':sha(__file__),'Test_used':False}
    write(ROOT/'reports/R16_PRODUCTION_TREE_FIXTURE.json',result);print(json.dumps(result))

def worker(manifest,fold,out):
    from pipeline.r16_production_data import load_production_role,validate_production_manifest,resolve_fold_manifest
    from pipeline.trees_r16_production import CODE_NAMES
    manifest,m=resolve_fold_manifest(Path(manifest).resolve(),fold);out=Path(out).resolve();folder=out/f'fold_{fold:02d}'
    if folder.exists():raise FileExistsError('Fresh fold required; partial runs cannot be reused')
    validation=validate_production_manifest(manifest,fold,verify_sources=True)
    folder.mkdir(parents=True);start=time.time()
    # Pin source closure plus data manifest; provider additionally verifies role SHA.
    paths=[Path(__file__),ROOT/'pipeline/r16_production_data.py',manifest]+[ROOT/n for n in CODE_NAMES]
    pins={str(p.relative_to(ROOT)):sha(p)for p in paths}
    for field in('source_code_sha256','source_input_sha256'):
        for n,h in m.get(field,{}).items():
            p=ROOT/n
            if p.is_file():pins[n]=h
    verify(pins)
    plan={'status':'frozen_before_fit','fold':fold,'manifest':str(manifest.relative_to(ROOT)),'manifest_sha256':sha(manifest),
       'source_pins':pins,'settings':DEFAULTS,'seed':2026102718+10007*fold,'feature_counts':{'regression':3827,'classification':700},
       'selection':'shared fixed R16 local_fit2400 development; not OOF','data_validation':validation,'PCA_LFanchor_highshare':False,'Test_used':False}
    write(folder/'PRE_FIT_FREEZE.json',plan)
    try:
        train=load_production_role(manifest,fold,'train');selection=load_production_role(manifest,fold,'selection')
        if not len(train['share'])or len(selection['share'])!=2400:raise ValueError('Full TRAIN and exact2400 shared development selection required')
        binding={'fold':fold,'manifest_sha256':sha(manifest),'recipe':'R16_improved_mixed6_unchanged','source_refs':'A only; B/C protected'}
        def progress(v):
            write(folder/'progress.json',v);print(json.dumps({'fold':fold,**v}),flush=True)
        learner=PhysicsLGBM(seed=plan['seed']);summary=learner.fit(train,selection,folder/'model',binding,progress=progress)
        import numpy as np
        restored=PhysicsLGBM.load(folder/'model',expected_binding=binding);a=learner.predict(selection['db'][:17]);b=restored.predict(selection['db'][:17])
        if any(not np.array_equal(a[k],b[k])for k in a):raise ValueError('Production reload differs')
        verify(pins)
        done={'status':'completed','fold':fold,'train_rows':len(train['share']),'selection_rows':2400,'selection_metrics':summary['selection_metrics'],
              'model_dir':str((folder/'model').relative_to(ROOT)),'summary_sha256':sha(folder/'model/summary.json'),
              'source_pins':pins,'elapsed_seconds':time.time()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'reloaded_exact':True,'query_inference_done':False}
        write(folder/'COMPLETED.json',done);print(json.dumps({'event':'fold_complete',**done}),flush=True)
    except BaseException as e:
        write(folder/'FAILED.json',{'status':'failed','error':repr(e),'partial_reuse_allowed':False});raise

def freeze_all(out):
    out=Path(out);models={};pins={}
    for fold in range(5):
        f=out/f'fold_{fold:02d}';d=read(f/'COMPLETED.json');s=read(f/'model/summary.json')
        if d['status']!='completed'or(f/'FAILED.json').exists():raise ValueError('All clean folds required')
        for n,h in d['source_pins'].items():pins[n]=h
        for n,h in s['artifact_sha256'].items():pins[str((f/'model'/n).relative_to(ROOT))]=h
        for p in(f/'COMPLETED.json',f/'model/summary.json'):pins[str(p.relative_to(ROOT))]=sha(p)
        models[str(fold)]={'model_dir':str((f/'model').relative_to(ROOT)),'manifest_sha256':s['binding']['manifest_sha256'],'selection_metrics':s['selection_metrics']}
    verify(pins);write(out/'ALL_FIVEFOLD_TREES_FROZEN.json',{'status':'completed','all_models_frozen':True,'models':models,'artifact_sha256':pins,
          'Test_used':False,'individual_fold_CV_validation_not_yet_scored':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['fixture','worker','freeze_all']);p.add_argument('--manifest');p.add_argument('--fold',type=int);p.add_argument('--output',default='models/r16_production_trees_r1');a=p.parse_args()
    sys.addaudithook(guard)
    if a.command=='fixture':fixture()
    elif a.command=='worker':worker(a.manifest,a.fold,ROOT/a.output)
    else:freeze_all(ROOT/a.output)
