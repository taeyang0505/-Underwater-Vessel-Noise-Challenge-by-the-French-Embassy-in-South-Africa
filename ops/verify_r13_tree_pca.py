"""Tiny numeric LightGBM plumbing with the actual fold-TRAIN PCA artifact.

Four boosting rounds are a labelled fixture, never a full R13 training run.
No fixed validation/report arrays or public query CSV are opened.
"""
from pathlib import Path
import json,sys,tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pipeline.trees_ensemble_r13 import PhysicsLGBM,transform_features,file_sha


def main():
    out=ROOT/'reports/R13_TREE_PCA_CONNECTION.json'
    if out.exists():raise FileExistsError('Fresh fixture receipt required')
    bundle=ROOT/'data/synthetic/group_cv_physics_r13_variants_v2'
    manifest=json.loads((bundle/'manifest.json').read_text());fold=manifest['folds']['0']
    basis=bundle/fold['basis_path'];rng=np.random.default_rng(202610271301)
    def rows(n,role):
        db=rng.normal(100.,5.,(n,15,25)).astype(np.float32);db[::4,:2]=np.nan
        y=np.zeros(n,np.float32);y[1::2]=rng.uniform(.1,.8,len(y[1::2]))
        return {'db':db,'share':y,'presence':(y>0).astype(np.float32),'role':role}
    train=rows(64,'train');inner=rows(16,'selection')
    settings={'maximum_trees':4,'early_stopping_patience':2,'threads':1,'minimum_leaf_rows':2}
    model=PhysicsLGBM(settings,seed=42,basis_path=basis,basis_training_binding=fold['basis_training_binding'],include_shape=True)
    features=transform_features(inner['db'],basis=model.basis,include_shape=True)
    assert features.shape==(16,5252) and np.isfinite(features).all()
    folder=Path(tempfile.mkdtemp(prefix='r13-PCA-tree-',dir='/private/tmp'))/'model'
    binding={'fixture_only':True,'synthetic_data':True,'fold_PCA_sha256':file_sha(basis)}
    summary=model.fit(train,inner,folder,binding,fixture=True)
    restored=PhysicsLGBM.load(folder,expected_binding=binding,allow_fixture=True)
    p=model.predict(inner['db'],batch_size=1);q=restored.predict(inner['db'],batch_size=7)
    error=max(float(np.abs(p[k]-q[k]).max()) for k in p)
    assert error<=1e-6
    try:PhysicsLGBM.load(folder,expected_binding=binding)
    except ValueError:production_rejected=True
    else:raise AssertionError('Fixture was admitted as production')
    payload={'format':'r13_full_tree_PCA_numeric_fixture_v1','status':'passed','fixture_only':True,
        'train_rows':64,'selection_rows':16,'maximum_boosting_rounds':4,'features':5252,
        'actual_fold_TRAIN_PCA_sha256':file_sha(basis),'basis_training_binding':fold['basis_training_binding'],
        'strict_restore_and_batch1_vs7_max_abs_error':error,'production_fixture_rejected':production_rejected,
        'code_sha256':{n:file_sha(ROOT/n) for n in ('pipeline/trees_ensemble_r13.py','ops/verify_r13_tree_pca.py')},
        'Test_or_report_arrays_or_real_source_model_fit':False,'temporary_fixture_model':str(folder)}
    out.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({'status':'passed','receipt':str(out),'max_error':error}))

if __name__=='__main__':main()
