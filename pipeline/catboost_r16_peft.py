"""Optional CPU CatBoost pair on the immutable R16 5160/2033 features.

Lazy imports keep this module usable when the optional wheel is unavailable.
Regression/classification train separately and own-fold CV never uses Tree5.
No Test/CSV access or tuning on locked evaluation panels is implemented here.
"""
from pathlib import Path
import gc, hashlib, importlib.metadata, json, time

ROOT=Path(__file__).resolve().parents[1]
VERSION='r16_peft_catboost_pair_v2_ownfold_es'
EXPECTED_VERSION='1.2.8'
FEATURE_COUNTS={'regression':5160,'classification':2033}
DEFAULTS=dict(iterations=1500,learning_rate=.035,depth=8,l2_leaf_reg=3.,
              thread_count=2,task_type='CPU',grow_policy='SymmetricTree',
              allow_writing_files=False,verbose=False,random_seed=2026102817)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def dependency_status(import_check=False):
    try:version=importlib.metadata.version('catboost')
    except importlib.metadata.PackageNotFoundError:
        return dict(available=False,required_version=EXPECTED_VERSION,reason='optional package missing')
    status=dict(available=version==EXPECTED_VERSION,version=version,required_version=EXPECTED_VERSION,
                python_package_location=str(importlib.metadata.distribution('catboost').locate_file('')))
    if import_check:
        try:
            import catboost
            status['import_passed']=True;status['imported_version']=catboost.__version__
        except Exception as e:status.update(available=False,import_passed=False,error=f'{type(e).__name__}: {e}')
    return status


def source_binding():
    from pipeline.training_r16_multisupport import codes
    pins=codes();pins[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    return pins


def verify(pins):
    for name,h in pins.items():
        if sha(ROOT/name)!=h:raise ValueError('Source changed: '+name)


def tree_feature_matrix(db,kind,path=None,batch=256):
    import numpy as np
    from pipeline.training_r16_multisupport import tree_features
    if kind not in FEATURE_COUNTS:raise ValueError('regression/classification required')
    shape=(len(db),FEATURE_COUNTS[kind])
    x=(np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape) if path
       else np.empty(shape,np.float32))
    for off in range(0,len(db),batch):
        block=tree_features(db[off:off+batch],kind)
        if block.shape!=(min(batch,len(db)-off),shape[1]) or not np.isfinite(block).all():
            raise ValueError('Original feature schema/nonfinite mismatch')
        x[off:off+batch]=block
    if hasattr(x,'flush'):x.flush()
    return x


class CatBoostR16Pair:
    """One regime/fold independent regression/classification pair."""
    def __init__(self,regression,classification):self.regression=regression;self.classification=classification

    @classmethod
    def load(cls,folder):
        from catboost import CatBoostRegressor,CatBoostClassifier
        folder=Path(folder);s=json.loads((folder/'summary.json').read_text())
        if s['status']!='completed' or s['format']!=VERSION:raise ValueError('Completed same-version pair required')
        verify(s['code_sha256'])
        for name,h in s['artifact_sha256'].items():
            if sha(folder/name)!=h:raise ValueError('CatBoost artifact changed')
        a=CatBoostRegressor();b=CatBoostClassifier()
        a.load_model(str(folder/'regression.cbm'));b.load_model(str(folder/'classification.cbm'))
        return cls(a,b)

    def predict_features(self,regression_features,classification_features):
        import numpy as np
        if regression_features.shape[1]!=5160 or classification_features.shape[1]!=2033:
            raise ValueError('Exact original feature schemas required')
        p=np.clip(self.regression.predict(regression_features,thread_count=2),0,1)
        c=self.classification.predict_proba(classification_features,thread_count=2)[:,1]
        if not np.isfinite(p).all() or not np.isfinite(c).all():raise ValueError('Nonfinite predictions')
        return p,c

    def predict(self,db):
        return self.predict_features(tree_feature_matrix(db,'regression'),tree_feature_matrix(db,'classification'))


def fit_pair(folder,*,fold=0,regime='D0',iterations=1500,early_stopping_rounds=40,
             limit_rows=None):
    """Explicit fresh fit. limit_rows is disposable smoke only, never production.

    Uses existing TRAIN and own-fold CV for early stopping. Shared local_fit
    (400 unique backgrounds) is read only after both fits have completed, to
    export predictions for a subsequent, independently fitted ensemble policy.
    """
    import numpy as np
    from catboost import CatBoostClassifier,CatBoostRegressor
    from pipeline.r16_multisupport_data import role,OUT,guard
    from pipeline.training_r16_multisupport import metric
    import sys
    if not dependency_status(True)['available']:raise RuntimeError('Pinned CatBoost dependency unavailable')
    if regime not in ('D0','H0') or fold not in range(5):raise ValueError('fold0..4, D0/H0 required')
    if not 1<=iterations<=1500 or early_stopping_rounds!=40:raise ValueError('Contract: max1500 trees / ES40')
    folder=Path(folder)
    if folder.exists():raise FileExistsError('Fresh CatBoost pair only; partial artifacts are never reused')
    folder.mkdir(parents=True);sys.addaudithook(guard)
    pins=source_binding();datahash=sha(OUT/'manifest.json');train=role(fold,regime,'train');cv=role(fold,regime,'cv')
    if set(train['base_window_index'])&set(cv['base_window_index']):raise ValueError('CV/train background identity overlap')
    n=len(train['share'])
    if limit_rows is not None:
        if limit_rows<32:raise ValueError('Smoke >=32 rows')
        # Metadata-only deterministic sampling, includes both target classes.
        rng=np.random.default_rng([2026102817,fold]);ix=rng.permutation(n)[:limit_rows]
        train={k:v[ix] for k,v in train.items()};n=len(ix)
    params=dict(DEFAULTS,iterations=iterations,random_seed=2026102817+fold)
    plan=dict(format=VERSION,status='fixed_before_fit',fold=fold,regime=regime,train_n=n,
        early_stopping_role='existing own-fold CV',CV_n=len(cv['share']),
        ensemble_fit_role='shared local_fit,400 unique backgrounds; read only after both fits',parameters=params,
        early_stopping_rounds=40,feature_counts=FEATURE_COUNTS,
        data_manifest_sha256=datahash,code_sha256=pins,dependency=dependency_status(True),
        disposable_smoke_only=limit_rows is not None,Test_used=False)
    write(folder/'PLAN.json',plan);pred={};history={};rounds={};began=time.perf_counter()
    for kind in ('regression','classification'):
        temp=folder/'temporary_train_features.npy'
        x=tree_feature_matrix(train['db'],kind,temp);v=tree_feature_matrix(cv['db'],kind)
        iscls=kind=='classification';key='presence' if iscls else 'share'
        model=(CatBoostClassifier(loss_function='Logloss',eval_metric='AUC',**params) if iscls
               else CatBoostRegressor(loss_function='RMSE',eval_metric='RMSE',**params))
        model.fit(x,train[key],eval_set=(v,cv[key]),use_best_model=True,
                  early_stopping_rounds=early_stopping_rounds,verbose=False)
        model.save_model(str(folder/(kind+'.cbm')))
        p=model.predict_proba(v,thread_count=2)[:,1] if iscls else np.clip(model.predict(v,thread_count=2),0,1)
        pred[kind]=p;history[kind]=model.get_evals_result()
        rounds[kind]=dict(best_iteration_0based=int(model.get_best_iteration()),retained_trees=int(model.tree_count_))
        write(folder/'progress.json',dict(stage='completed_'+kind,iterations=rounds,seconds=time.perf_counter()-began))
        del model,x,v;gc.collect();temp.unlink()
    with (folder/'ownfold_cv.npz').open('wb') as f:np.savez_compressed(f,share=pred['regression'],classifier_prob=pred['classification'],
        target_share=cv['share'],sample_id=cv['sample_id'])
    cv_metrics=metric(cv['share'],pred['regression'],pred['classification'])
    selection=role(0,'D0','selection')
    if set(train['base_window_index'])&set(selection['base_window_index']):raise ValueError('Ensemble-fit/TRAIN overlap')
    fit_predictions={}
    for kind in ('regression','classification'):
        model=CatBoostClassifier() if kind=='classification' else CatBoostRegressor()
        model.load_model(str(folder/(kind+'.cbm')))
        x=tree_feature_matrix(selection['db'],kind)
        fit_predictions[kind]=(model.predict_proba(x,thread_count=2)[:,1] if kind=='classification'
            else np.clip(model.predict(x,thread_count=2),0,1))
        del model,x;gc.collect()
    with (folder/'selection.npz').open('wb') as f:np.savez_compressed(f,share=fit_predictions['regression'],classifier_prob=fit_predictions['classification'],
        target_share=selection['share'],sample_id=selection['sample_id'])
    write(folder/'history.json',history);verify(pins)
    summary=dict(plan,status='completed',iterations=rounds,seconds=time.perf_counter()-began,
        ownfold_CV_metrics=cv_metrics,
        selection_metrics=metric(selection['share'],fit_predictions['regression'],fit_predictions['classification']),
        checkpoint_and_ensemble_fit_roles_disjoint=True,
        artifact_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    write(folder/'summary.json',summary)
    return summary
