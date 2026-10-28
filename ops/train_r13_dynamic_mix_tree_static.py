"""Static24k tree executor repair: explicit stdlib Mapping, no monkeypatch.

The old CNN source, data and process are unchanged. Its completed100epoch
artifacts are imported only after strict completion. All failed pre-fit logs
remain available. This is one declared recipe, not repeated hyperparameter fit.
"""
from pathlib import Path
from collections.abc import Mapping
import argparse,json,shutil,time
from ops import train_r13_dynamic_mix as base
from ops.train_r12_lightgbm import (CompactPhysicsLGBM,CODE_NAMES,dataset,transform,metrics,array_sha,utc,token)
ROOT=base.ROOT
SOURCE=ROOT/'models/r13_dynamic_mix_r1'
DEST=ROOT/'models/r13_dynamic_mix_r2'
sha=base.sha
write=base.write

def fit_tree():
    np,torch=base.imports(2)
    plan=base.load_plan(DEST)
    if (DEST/'tree').exists():raise FileExistsError('Fresh actual tree fit required')
    receipt={'format':'r13_static_import_correction_before_tree_fit_v1','created_unix':time.time(),
        'source_CNN_directory':'models/r13_dynamic_mix_r1/cnn','tree_directory':'models/r13_dynamic_mix_r2/tree',
        'static_executor_source_sha256':sha(__file__),'original_recipe_source_sha256':sha(ROOT/'ops/train_r13_dynamic_mix.py'),
        'original_plan_sha256':plan['plan_sha256'],'correction':'Explicit collections.abc.Mapping import, literal24k fit method; no runtime namespace injection',
        'previous_failures':['original missing import before tree folder/fit','dynamic helper interrupted during pre-fit data validation before any tree fit'],
        'hyperparameters_source_seed_data_unchanged':True,'old_logs_preserved':True,'actual_prior_tree_fits':0}
    write(DEST/'STATIC_CORRECTION_BEFORE_TREE_FIT.json',receipt)
    m,b,h,subset,val,basis=base.arrays()
    train={k:np.concatenate([b[k],h[k]]) for k in b}
    original=train['sample_id'].astype(str).copy();train['paired_original_sample_id']=original
    train['augmentation_variant']=np.repeat(np.asarray(['baseline','highshare']),12000)
    train['sample_id']=np.char.add(original,np.char.add('__',train['augmentation_variant']))
    binding={'format':'r13_dynamic_mix_24k_tree_binding_v1','plan_sha256':plan['plan_sha256'],'data_hash':m['data_hash'],
        'baseline_TRAIN_sha256':m['arms']['baseline']['sha256'],'highshare_TRAIN_sha256':m['arms']['highshare']['sha256'],
        'validation_fit_sha256':m['validation_references']['validation_fit']['sha256'],'basis_sha256':base.BASIS_SHA,
        'two_augmentation_variants_same_source':True,'independent_source_rows':12000,'augmentation_rows':24000,
        'execution_source_sha256':sha(__file__),'execution_source_path':str(Path(__file__).resolve().relative_to(ROOT)),
        'static_import_correction_receipt_sha256':sha(DEST/'STATIC_CORRECTION_BEFORE_TREE_FIT.json'),'development_not_blind':True}
    tree=StaticDynamicMixTree(include_shape=True,basis_path=ROOT/base.BASIS,basis_sha256=base.BASIS_SHA,
        basis_training_binding=m['basis_training_binding'],settings={'threads':2},seed=base.SEED)
    s=tree.fit(train,val,DEST/'tree',binding,fixture=False)
    restored=StaticDynamicMixTree.load(DEST/'tree',expected_binding=binding)
    p=tree.predict(val['db'][:7],batch_size=7);q=restored.predict(val['db'][:7],batch_size=1)
    if any(not np.array_equal(p[k],q[k]) for k in p):raise AssertionError('Tree reload batch1/7 differs')
    base.verify_sources(plan)
    write(DEST/'TREE_COMPLETED.json',{'status':'completed','summary_sha256':sha(DEST/'tree/summary.json'),'binding':binding})
    print(json.dumps({'stage':'static_tree_completed','validation_fit_rmse':s['validation_fit_metrics']['rmse'],
        'elapsed_seconds':s['elapsed_seconds'],'best_iteration':s['best_iteration']}),flush=True)

def finish(evaluate):
    np,torch=base.imports(2);plan=base.load_plan(DEST)
    correction=json.loads((DEST/'STATIC_CORRECTION_BEFORE_TREE_FIT.json').read_text())
    if correction['static_executor_source_sha256']!=sha(__file__):raise ValueError('Static executor changed')
    c=json.loads((SOURCE/'cnn/summary.json').read_text())
    hist=[json.loads(x) for x in (SOURCE/'cnn/epochs.jsonl').read_text().splitlines()]
    if (c['status']!='completed' or c['epochs_run']!=100 or c['early_stopping'] or len(hist)!=100
        or [x['epoch'] for x in hist]!=list(range(1,101)) or c['best_epoch']!=int(np.argmin([x['validation_fit_rmse'] for x in hist]))+1
        or sha(SOURCE/'cnn/best.pt')!=c['checkpoint_sha256']):raise ValueError('Only full100 completed original CNN may be imported')
    if (DEST/'cnn').exists():raise FileExistsError('Import is one-shot')
    shutil.copytree(SOURCE/'cnn',DEST/'cnn')
    copied={p.name:sha(p) for p in (SOURCE/'cnn').iterdir() if p.is_file()}
    if any(sha(DEST/'cnn'/n)!=h for n,h in copied.items()):raise ValueError('Completed CNN byte copy differs')
    frozen=base.freeze(DEST)
    provenance={'format':'r13_static_corrected_executor_completion_v1','status':'completed','one_recipe_only':True,
        'CNN_actual_training_path':'models/r13_dynamic_mix_r1/cnn','CNN_staging_copy':'models/r13_dynamic_mix_r2/cnn',
        'tree_actual_training_path':'models/r13_dynamic_mix_r2/tree','CNN_completed100_byte_exact':copied,
        'original_parent_failure_sha256':sha(SOURCE/'FAILED.json') if (SOURCE/'FAILED.json').exists() else None,
        'static_source_path':str(Path(__file__).resolve().relative_to(ROOT)),'static_source_sha256':sha(__file__),
        'correction_receipt_sha256':sha(DEST/'STATIC_CORRECTION_BEFORE_TREE_FIT.json'),
        'no_runtime_monkeypatch':True,'actual_CNN_fits':1,'actual_tree_booster_fits':1,
        'all2_freeze_sha256':frozen['freeze_sha256'],'no_partial_checkpoint_resume':True}
    write(DEST/'CORRECTED_EXECUTOR_COMPLETION.json',provenance)
    if evaluate:
        report=ROOT/base.REPORT;base.evaluate(DEST,report)
        r=json.loads((report/'RESULTS.json').read_text());r['executor_correction_provenance']=provenance
        r['executor_correction_provenance']['receipt_sha256']=sha(DEST/'CORRECTED_EXECUTOR_COMPLETION.json')
        write(report/'RESULTS.json',r)
    print(json.dumps({'status':'completed','output':str(DEST),'evaluation':evaluate}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['tree','finish-evaluate'])
    a=p.parse_args()
    if a.action=='tree':fit_tree()
    else:finish(True)

# The immutable original fit operators are copied below, with24000 row guard.

class StaticDynamicMixTree(CompactPhysicsLGBM):
    """Exact immutable learner fit with explicit24k-row requirement."""
    def fit(self,train,validation_fit,output,binding,*,fixture=False):
        import numpy as np,lightgbm as lgb
        from pipeline.features_r12 import compact_feature_names
        td,ty,tz=dataset(train,'train',expected_rows=None if fixture else 24000)
        vd,vy,vz=dataset(validation_fit,'validation_fit',expected_rows=None if fixture else 1000)
        if len(np.unique(tz))!=2 or len(np.unique(vz))!=2:raise ValueError('BothpresenceclassesrequiredTRAINandvalidationfit')
        for key in ('sample_id','base_window_index','source_id','background_group'):
            if key in train and key in validation_fit and set(np.asarray(train[key]).astype(str))&set(np.asarray(validation_fit[key]).astype(str)):
                raise ValueError('TRAIN/validationfitsourceseparationfailed:'+key)
        if not isinstance(binding,Mapping) or not binding:raise ValueError('Explicitdataset/sourcebindingrequired')
        token(binding);folder=Path(output)
        if folder.exists():raise FileExistsError('Freshmodeloutputrequirednoreuse')
        folder.mkdir(parents=True);started=time.perf_counter()
        pins={n:sha(ROOT/n) for n in CODE_NAMES}
        plan={'format':'r12_compact_tree_fit_plan_v1','created_utc':utc(),'training_binding':dict(binding),'settings':self.settings,'seed':self.seed,
              'include_shape':self.include_shape,'basis_file_sha256':self.basis_sha256,'basis_training_binding':self.basis_training_binding,'feature_fit_on_synthetic_train':False,
              'scaler':'none;fixedrelativefeatures','PCA':'sharedTRAIN-onlypureSAbackgroundbasis;loadedfrozen' if self.include_shape else 'disabled',
              'input_identity':{'train':{k:array_sha(v) for k,v in train.items()},'validation_fit':{k:array_sha(v) for k,v in validation_fit.items()}},
              'source_code_sha256':pins,'fixture_only':bool(fixture),'gate_eval_Test_used_for_ES':False}
        write(folder/'fit_plan.json',plan)
        try:
            x=transform(td,basis=self.basis,include_shape=self.include_shape,batch_size=self.settings['feature_batch_size'])
            v=transform(vd,basis=self.basis,include_shape=self.include_shape,batch_size=self.settings['feature_batch_size'])
            cfg=self.settings;args=dict(n_estimators=cfg['maximum_trees'],learning_rate=cfg['learning_rate'],num_leaves=cfg['num_leaves'],
                min_child_samples=cfg['minimum_leaf_rows'],max_bin=127,n_jobs=cfg['threads'],random_state=self.seed,verbosity=-1,
                colsample_bytree=.85,subsample=.85,subsample_freq=1,deterministic=True,force_col_wise=True,
                data_random_seed=self.seed,feature_fraction_seed=self.seed,bagging_seed=self.seed)
            reg=lgb.LGBMRegressor(objective='regression_l2',metric='None',**args);clf=lgb.LGBMClassifier(objective='binary',metric='binary_logloss',**args)
            def clipped_rmse(truth,p):return 'clipped_share_rmse',float(np.sqrt(np.mean((np.clip(p,0,1)-truth)**2))),False
            reg.fit(x,ty,eval_set=[(v,vy)],eval_metric=clipped_rmse,callbacks=[lgb.early_stopping(cfg['early_stopping_patience'],verbose=False),lgb.log_evaluation(0)])
            clf.fit(x,tz,eval_set=[(v,vz)],callbacks=[lgb.early_stopping(cfg['early_stopping_patience'],verbose=False),lgb.log_evaluation(0)])
            self.regressor,self.classifier=reg.booster_,clf.booster_;self.regressor.save_model(str(folder/'share.txt'));self.classifier.save_model(str(folder/'presence.txt'))
            schema={'format':'r12_compact_relative_feature_schema_v1','feature_count':len(compact_feature_names(include_shape=self.include_shape)),
                'feature_names':list(compact_feature_names(include_shape=self.include_shape)),'include_shape':self.include_shape,
                'basis_sha256':self.basis_sha256,'query_inputs':['db'],'TRAIN_fitted_scaler':False,'source_code_sha256':pins}
            write(folder/'feature_schema.json',schema)
            if self.include_shape:shutil.copyfile(self.basis_path,folder/'spectral_basis.npz')
            pred=self._predict_features(v);np.savez_compressed(folder/'validation_fit_predictions.npz',**pred,
                sample_id=np.asarray(validation_fit.get('sample_id',np.arange(len(v)))),target_share=vy,target_presence=vz)
            names=['share.txt','presence.txt','feature_schema.json','validation_fit_predictions.npz','fit_plan.json']+(['spectral_basis.npz'] if self.include_shape else [])
            summary={'format':'r12_compact_physics_lgbm_v1','kind':'lgbm_physics_compact','status':'completed','fixture_only':bool(fixture),
                'settings':cfg,'seed':self.seed,'include_shape':self.include_shape,'feature_count':x.shape[1],'feature_schema_sha256':token(schema),
                'basis_sha256':self.basis_sha256,'basis_training_binding':self.basis_training_binding,'training_binding':dict(binding),'fit_plan_sha256':sha(folder/'fit_plan.json'),
                'best_iteration':int(reg.best_iteration_),'classifier_best_iteration':int(clf.best_iteration_),
                'validation_fit_metrics':metrics(validation_fit,pred),'train_rows':len(td),'validation_fit_rows':len(vd),
                'feature_matrix_bytes':{'train':x.nbytes,'validation_fit':v.nbytes},'lightgbm_version':lgb.__version__,
                'score_policy':'.5clipped_share+.5presence_classifier; no calibration','NN_epoch_policy_applicable':False,
                'artifact_sha256':{n:sha(folder/n) for n in names},'source_code_sha256':pins,
                'elapsed_seconds':time.perf_counter()-started,'gate_eval_Test_used_for_fitting':False}
            self.summary=summary;write(folder/'summary.json',summary)
            write(folder/'FROZEN_BEFORE_HELDOUT.json',{'summary_sha256':sha(folder/'summary.json'),'created_utc':utc(),
                'binding_sha256':token(binding),'no_refit_after_heldout':True,'heldout_opened':False})
            return summary
        except BaseException as e:
            write(folder/'FAILED.json',{'type':type(e).__name__,'error':str(e),'partial_reuse_allowed':False});raise

if __name__=='__main__':main()
