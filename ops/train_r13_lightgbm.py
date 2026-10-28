"""Four fresh R13 compact LightGBM fits; TRAIN/validation_fit only.

Baseline strict-matched vs high-share data crossed with PCA shape off/on.
The original R12 learner is reused byte-for-byte. No gate/eval/R11/Test
predictions or score-weight selection happen in this entry point. Future
recipe/score selection belongs to the all-model-frozen R13 evaluator.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
MODELS=(('baselineP0','baseline',False),('baselineP1','baseline',True),
        ('highshareP0','highshare',False),('highshareP1','highshare',True))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/synthetic/r13_cpu_correction_v1');p.add_argument('--expected-data-hash',required=True)
    p.add_argument('--config',default='configs/r13_cpu_correction_v1.json');p.add_argument('--output',default='models/r13_compact_trees_v1')
    p.add_argument('--basis',default='models/r12_shared_basis_v1/basis.npz')
    p.add_argument('--basis-sha256',default='a176bedfd772e552bb3fb310449b699b23d2fb9445af071ed0f72b05a1205113')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--validate-only',action='store_true')
    a=p.parse_args();run(a)

def run(a):
    from ops.train_r12_lightgbm import CompactPhysicsLGBM,environment,guard,sha,write,dataset,array_sha
    environment(a.threads);sys.addaudithook(guard)
    import numpy as np
    from pipeline.dataset_r13 import load_r13_arrays,validate_r13_bundle
    from pipeline.dataset_r12 import validate_r12_bundle
    from pipeline.features_r12 import load_spectral_basis
    data=(ROOT/a.data).resolve();out=(ROOT/a.output).resolve();config=(ROOT/a.config).resolve();basis_path=(ROOT/a.basis).resolve()
    if not all(p.is_relative_to(ROOT) for p in (data,out,config,basis_path)):raise ValueError('Project-containedpathsrequired')
    m=validate_r13_bundle(data);cfg=json.loads(config.read_text())
    if m.get('version')!='r13_highshare_paired_train_v1' or m.get('data_hash')!=a.expected_data_hash:raise ValueError('ExplicitapprovedR13data_hashrequired')
    if set(m['arms'])!={'baseline','highshare'}:raise ValueError('Exactly2predeclaredTRAINdataarmsrequired')
    if (cfg['seed']!=2026102712 or cfg['tree']['fresh_fits']!=4 or cfg['tree']['examples']!=12000
        or cfg['tree']['maximum_trees']!=1500 or cfg['tree']['early_stopping']!=40):raise ValueError('Predeclared4fresh-treepolicydiffers')
    parent=(ROOT/m['parent_R12_data_dir']).resolve();pm=validate_r12_bundle(parent)
    if sha(parent/'manifest.json')!=m['parent_R12_manifest_sha256'] or pm['data_hash']!=m['parent_R12_data_hash']:raise ValueError('ParentR12versiondiffers')
    if m['basis_training_binding']!=pm['basis_training_binding'] or sha(basis_path)!=a.basis_sha256:raise ValueError('FrozenpureTRAINbasisdiffers')
    basis=load_spectral_basis(basis_path,expected_sha256=a.basis_sha256,expected_training_binding=m['basis_training_binding'])
    paths=[Path(__file__),config,data/'manifest.json',data/'readiness.json',parent/'manifest.json',parent/'readiness.json',basis_path,
        ROOT/'pipeline/dataset_r13.py',ROOT/'pipeline/features_r12.py',ROOT/'ops/train_r12_lightgbm.py']
    paths += [ROOT/m['arms'][arm][key] for arm in ('baseline','highshare') for key in ('path','metadata_path','recipes_path')]
    pins={str(p.resolve().relative_to(ROOT)):sha(p) for p in paths}
    # Validation references are hashed and path-checked here, but only the
    # validation_fit array is materialized below; gate/eval remain closed.
    for role,e in m['validation_references'].items():
        for k,hk in [('path','sha256'),('metadata_path','metadata_sha256'),('recipes_path','recipes_sha256')]:
            if k not in e:continue
            p=(ROOT/e[k]).resolve()
            if not p.is_relative_to(ROOT) or sha(p)!=e[hk]:raise ValueError('Validationreferencehashdiffers:'+role)
            pins[str(p.relative_to(ROOT))]=e[hk]
    entry=m['validation_references']['validation_fit']
    if entry['sha256']!=pm['common_validation']['validation_fit']['sha256']:raise ValueError('Commonvalidation_fit1000changed')
    # A shape-enabled preflight must prove file state/provenance before any fit.
    probe=CompactPhysicsLGBM(include_shape=True,basis_path=basis_path,basis_sha256=a.basis_sha256,
        basis_training_binding=m['basis_training_binding'],settings={'threads':a.threads})
    if a.validate_only:
        print(json.dumps({'status':'preflight_passed','data_hash':m['data_hash'],'basis_sha256':a.basis_sha256,
            'requested_models':[n for n,_,_ in MODELS],'gate_eval_R11_Test_arrays_opened':False,'training_executed':False}));return m
    if out.exists():raise FileExistsError('FreshR13runrequired; partialoroldmodelreuseforbidden')
    out.mkdir(parents=True)
    frozen={'format':'r13_four_tree_fit_plan_v1','created_unix':time.time(),'data_hash':m['data_hash'],
        'data_manifest_sha256':sha(data/'manifest.json'),'basis_sha256':a.basis_sha256,'source_pins':pins,
        'models':[{'name':n,'data_arm':arm,'include_shape':shape} for n,arm,shape in MODELS],
        'fresh_fits':4,'train_rows_each':12000,'validation_fit_rows':1000,'threads':a.threads,'seed':2026102712,
        'boosting_maximum':1500,'early_stopping':40,'score_weight_selection_here':False,
        'score_note':'Importedlearnerreturnslegacyhalfshare/halfclassifierasdiagnosticonly;futureR13gatechooseslambdaafterall8modelsgloballyfreeze.',
        'gate_eval_R11_Test_values_opened':False}
    write(out/'PRE_FIT_FREEZE.json',frozen);started=time.perf_counter();completed={}
    try:
        with np.load(ROOT/entry['path'],allow_pickle=False) as z:val={k:z[k] for k in z.files}
        dataset(val,'validation_fit',expected_rows=1000)
        for name,arm,shape in MODELS:
            train=load_r13_arrays(data,arm);dataset(train,'train',expected_rows=12000)
            binding={'format':'r13_tree_source_binding_v1','R13_data_hash':m['data_hash'],'R13_manifest_sha256':sha(data/'manifest.json'),
                'data_arm':arm,'include_shape':shape,'TRAIN_npz_sha256':m['arms'][arm]['sha256'],
                'parent_R12_manifest_sha256':m['parent_R12_manifest_sha256'],'parent_R12_data_hash':m['parent_R12_data_hash'],
                'validation_fit_npz_sha256':entry['sha256'],'basis_sha256':a.basis_sha256 if shape else None,
                'basis_training_binding':m['basis_training_binding'] if shape else None,'four_fresh_fits':True}
            tree=CompactPhysicsLGBM(include_shape=shape,basis_path=basis_path if shape else None,
                basis_sha256=a.basis_sha256 if shape else None,basis_training_binding=m['basis_training_binding'] if shape else None,
                settings={'threads':a.threads},seed=2026102712)
            print(json.dumps({'stage':'fit_start','model':name,'data_arm':arm,'shape':shape,'train_rows':len(train['share'])}),flush=True)
            s=tree.fit(train,val,out/name,binding,fixture=False)
            restored=CompactPhysicsLGBM.load(out/name,expected_binding=binding)
            p=tree.predict(val['db'][:7],batch_size=7);q=restored.predict(val['db'][:7],batch_size=1)
            if any(not np.array_equal(p[k],q[k]) for k in p):raise AssertionError('Savedreloadbatch1/7parityfailed')
            if 'score_classifier' not in p:raise AssertionError('Independentclassifieroutputmissing')
            completed[name]={'summary_path':str((out/name/'summary.json').relative_to(ROOT)),'summary_sha256':sha(out/name/'summary.json'),
                'share_booster_sha256':sha(out/name/'share.txt'),'classifier_booster_sha256':sha(out/name/'presence.txt'),
                'training_binding':binding,'validation_fit_rmse':s['validation_fit_metrics']['rmse'],'feature_count':s['feature_count']}
            write(out/'progress.json',{'completed':completed,'total':4,'elapsed_seconds':time.perf_counter()-started,'gate_eval_R11_Test_values_opened':False})
            print(json.dumps({'stage':'fit_completed','model':name,'best_round':s['best_iteration'],'classifier_best_round':s['classifier_best_iteration'],
                'validation_fit_rmse':s['validation_fit_metrics']['rmse'],'elapsed_seconds':time.perf_counter()-started}),flush=True)
            for n,h in pins.items():
                if sha(ROOT/n)!=h:raise ValueError('Frozensourcechanged:'+n)
        validate_r13_bundle(data)
        write(out/'ALL_TREES_FROZEN.json',{'format':'r13_four_tree_frozen_v1','status':'completed','all_models_frozen':True,
            'new_fits':4,'completed_control_reused':0,'models':completed,'data_hash':m['data_hash'],
            'data_manifest_sha256':sha(data/'manifest.json'),'basis_sha256':a.basis_sha256,'binding':frozen,
            'model_summary_sha256':{v['summary_path']:v['summary_sha256'] for v in completed.values()},
            'gate_eval_R11_Test_values_opened':False})
        print(json.dumps({'status':'all_four_trees_frozen','models':list(completed),'elapsed_seconds':time.perf_counter()-started}),flush=True)
        return completed
    except BaseException as e:
        write(out/'FAILED.json',{'error':repr(e),'partial_reuse_allowed':False,'gate_eval_R11_Test_opened':False});raise

if __name__=='__main__':main()
