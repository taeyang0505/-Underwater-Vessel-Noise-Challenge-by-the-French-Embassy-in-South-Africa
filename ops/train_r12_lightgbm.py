"""R12 isolated Mac-CPU LightGBM factorial executor, never Test fitting.

Eight runs are four predeclared long/HF data arms times shape-feature off/on.
The PCA basis is fit once on released SA TRAIN-only unmixed backgrounds by
the caller, never on synthetic targets, validation, gate, eval or queries.
Early stopping uses validation_fit only; all saved models freeze before
their gate/eval metrics. This file never chooses a factorial winner or
modifies old production artifacts. Heavy training requires explicit CLI.
"""
from __future__ import annotations
import argparse
from collections.abc import Mapping
from datetime import datetime,timezone
import hashlib,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEFAULTS={'maximum_trees':1500,'early_stopping_patience':40,'threads':4,
          'learning_rate':.035,'num_leaves':31,'minimum_leaf_rows':50,
          'feature_batch_size':512}
ARMS=('L0_H0','L0_H1','L1_H0','L1_H1')
CODE_NAMES=('ops/train_r12_lightgbm.py','pipeline/features_r12.py',
            'pipeline/relative_grid_r6.py','pipeline/relative_grid_r6_v2.py',
            'pipeline/physics_features_r6_v2.py')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def token(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def array_sha(x):
    import numpy as np
    a=np.ascontiguousarray(x)
    if a.dtype.kind=='O':raise TypeError('Objectarraysnotallowed')
    h=hashlib.sha256(json.dumps([a.dtype.str,list(a.shape)]).encode());h.update(a.tobytes());return h.hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def utc():return datetime.now(timezone.utc).isoformat()
def environment(threads):
    if os.environ.get('KMP_DUPLICATE_LIB_OK'):raise ValueError('DuplicateOpenMPruntimebypassisforbidden')
    if sys.platform=='darwin' and '/opt/homebrew/opt/libomp/lib' not in os.environ.get('DYLD_LIBRARY_PATH','').split(':'):
        raise RuntimeError('StartfreshwithDYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib')
    for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[n]=str(threads)
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=os.fsdecode(args[0]).replace('\\','/').lower()
    if Path(p).name in {'test.csv','sample_submission.csv','report.npz','outer_validation.npz'} or '/sealed/' in p:
        raise PermissionError('Test/oldOUTER/unusedreport/sealednotallowed:'+p)
def safe_path(root,name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('Datasetrelativepathrequired')
    original=Path(root)/p;q=original.resolve()
    if Path(root).resolve() not in q.parents or original.is_symlink() or not q.is_file():raise ValueError('Unsafeormissingartifact:'+name)
    return q

def dataset(d,role,*,expected_rows=None):
    import numpy as np
    if not isinstance(d,Mapping) or not {'db','share','presence'}<=set(d):raise TypeError('Explicitlabelleddatasetrequired')
    x=np.asarray(d['db']);y=np.asarray(d['share'],float);z=np.asarray(d['presence'])
    if (x.ndim!=3 or x.shape[1:]!=(15,25) or x.dtype.kind!='f' or not len(x)
        or np.isinf(x).any() or not np.isfinite(x).any((1,2)).all()):raise ValueError('Invaliddb[N,15,25]')
    if expected_rows is not None and len(x)!=expected_rows:raise ValueError(role+'rowcountdiffers')
    if y.shape!=(len(x),) or z.shape!=y.shape or not np.isfinite(y).all() or ((y<0)|(y>1)).any() or not np.isin(z,[0,1]).all():raise ValueError('Invalidtargets')
    if not np.array_equal(y>0,z.astype(bool)):raise ValueError('Knownadditionpresenceandshareinconsistent')
    for k,v in d.items():
        a=np.asarray(v)
        if a.dtype.kind=='O' or not a.ndim or len(a)!=len(x):raise ValueError('Unalignedorobjectfield:'+k)
    if 'sample_id' in d and len(set(np.asarray(d['sample_id']).astype(str)))!=len(x):raise ValueError('DuplicatesampleID')
    if 'cell_share' in d:
        q=np.asarray(d['cell_share'])
        if q.shape!=x.shape or not np.isfinite(q).all() or ((q<0)|(q>1)).any() or np.any(q[~np.isfinite(x)]!=0):raise ValueError('Invalidknowncellfraction')
        from pipeline.relative_grid_r6_v2 import relative_grid_v2
        _,w=relative_grid_v2(x)
        from pipeline.relative_grid_r6 import quantization_pooling_bound
        if np.max(abs((w*q).sum((1,2))-y))>quantization_pooling_bound(.0501)+2e-6:raise ValueError('Roundedphysicalshareboundfailed')
    return x,y,z.astype(np.int64)

def metrics(data,pred):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y=np.asarray(data['share'],float);z=np.asarray(data['presence'],int);p=np.asarray(pred['share'],float);e=p-y
    auc=lambda s:float(roc_auc_score(z,s)) if len(np.unique(z))==2 else None
    r={'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),'bias':float(e.mean()),
       'target_mean':float(y.mean()),'target_rms':float(np.sqrt(np.mean(y*y))),
       'prediction_mean':float(p.mean()),'prediction_rms':float(np.sqrt(np.mean(p*p))),
       'share_auc':auc(p),'classifier_auc':auc(pred['score_classifier']),'hybrid_auc':auc(pred['score']),
       'share_saturation_le001':float(np.mean(p<=.01)),'share_saturation_ge099':float(np.mean(p>=.99)),
       'classifier_saturation_le001':float(np.mean(pred['score_classifier']<=.01)),
       'classifier_saturation_ge099':float(np.mean(pred['score_classifier']>=.99))}
    if 'share_raw' in pred:r['raw_share_clipped_fraction']=float(np.mean((pred['share_raw']<0)|(pred['share_raw']>1)))
    for key in ('site','family'):
        if key in data:
            labels=np.asarray(data[key]).astype(str);slices={}
            for k in sorted(set(labels)):
                m=labels==k;slices[k]={'n':int(m.sum()),'rmse':float(np.sqrt(np.mean(e[m]*e[m]))),'bias':float(e[m].mean()),
                    'target_mean':float(y[m].mean()),'prediction_mean':float(p[m].mean()),
                    'share_auc':float(roc_auc_score(z[m],p[m])) if len(np.unique(z[m]))==2 else None,
                    'classifier_auc':float(roc_auc_score(z[m],pred['score_classifier'][m])) if len(np.unique(z[m]))==2 else None}
            r[key+'_slices']=slices
    if 'site_slices' in r:r['site_macro_rmse']=float(np.mean([s['rmse'] for s in r['site_slices'].values()]))
    masks={'zero':y==0,'positive_to020':(y>0)&(y<=.2),'above020':y>.2}
    if 'family' in data:
        long=np.char.startswith(np.asarray(data['family']).astype(str),'long_')&(y>0)
        masks['long_positive']=long;masks['standard_positive']=(y>0)&~long
    if 'special_zero' in data:masks['special_zero']=np.asarray(data['special_zero'],bool)&(y==0)
    if 'site' in data:
        sf=np.array(['francis' in s.lower() for s in np.asarray(data['site']).astype(str)])
        masks['SF_dense_zero']=sf&(np.isfinite(data['db']).all(2).sum(1)>=13)&(y==0)
    r['protected_slices']={k:{'n':int(m.sum()),'rmse':float(np.sqrt(np.mean(e[m]**2))) if m.any() else None,
        'bias':float(e[m].mean()) if m.any() else None,'prediction_mean':float(p[m].mean()) if m.any() else None,
        'SSE':float(np.sum(e[m]**2))} for k,m in masks.items()}
    return r

def transform(db,*,basis=None,include_shape=False,batch_size=512):
    import numpy as np
    from pipeline.features_r12 import compact_relative_features,compact_feature_names
    if isinstance(db,Mapping):raise TypeError('Featuresacceptdbonlyneverlabels')
    a=np.asarray(db)
    if (a.ndim!=3 or a.shape[1:]!=(15,25) or not len(a) or a.dtype.kind!='f' or batch_size<1
        or np.isinf(a).any() or not np.isfinite(a).any((1,2)).all()):raise ValueError('Finiteobservedfloatingdbandpositivebatchrequired')
    names=tuple(compact_feature_names(include_shape=include_shape));expected=673 if include_shape else 466
    if len(names)!=expected or len(set(names))!=expected:raise ValueError('Frozencompactfeatureschemacountdiffers')
    if include_shape and basis is None:raise ValueError('ShapefeaturesrequireTRAIN-onlypurebackgroundbasis')
    result=np.empty((len(a),expected),np.float32)
    for s in range(0,len(a),batch_size):
        x=compact_relative_features(a[s:s+batch_size],basis=basis,include_shape=include_shape)
        if x.shape!=(min(batch_size,len(a)-s),expected) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Featureshape/dtype/finitecheckfailed')
        result[s:s+len(x)]=x
    return result

class CompactPhysicsLGBM:
    def __init__(self,*,include_shape=False,basis=None,basis_path=None,basis_sha256=None,basis_training_binding=None,settings=None,seed=2026102712):
        if type(include_shape) is not bool:raise TypeError('Explicitboolfeaturetoggle')
        if settings is not None and (not isinstance(settings,Mapping) or set(settings)-set(DEFAULTS)):raise ValueError('Unknowntreesettings')
        self.settings=dict(DEFAULTS,**(settings or {}));self.seed=seed;self.include_shape=include_shape
        if type(seed)is not int or not 0<=seed<2**32:raise ValueError('Unsigned32bitseedrequired')
        for k,v in self.settings.items():
            if k!='learning_rate' and (type(v)is not int or v<1):raise ValueError('Positiveintegersettingsrequired')
        if self.settings['maximum_trees']>1500 or self.settings['early_stopping_patience']>40:raise ValueError('Fixedmax1500/ES40cap')
        if not 0<self.settings['learning_rate']<=1:raise ValueError('Learningrateinvalid')
        if include_shape:
            if (basis_path is None or basis_sha256 is None or sha(basis_path)!=basis_sha256
                or not isinstance(basis_training_binding,Mapping) or not basis_training_binding):raise ValueError('ExplicitsharedTRAINbasisfileSHAandbindingrequired')
            from pipeline.features_r12 import load_spectral_basis
            loaded=load_spectral_basis(basis_path,expected_sha256=basis_sha256,expected_training_binding=dict(basis_training_binding))
            if basis is not None and token(basis)!=token(loaded):raise ValueError('Inmemorybasisdiffersfromfile')
            self.basis=loaded;self.basis_path=Path(basis_path);self.basis_sha256=basis_sha256;self.basis_training_binding=dict(basis_training_binding)
        else:self.basis=None;self.basis_path=None;self.basis_sha256=None;self.basis_training_binding=None
        self.regressor=self.classifier=None;self.summary=None

    def fit(self,train,validation_fit,output,binding,*,fixture=False):
        import numpy as np,lightgbm as lgb
        from pipeline.features_r12 import compact_feature_names
        td,ty,tz=dataset(train,'train',expected_rows=None if fixture else 12000)
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

    def _predict_features(self,x):
        import numpy as np
        if self.regressor is None or self.classifier is None:raise ValueError('Completedmodelrequired')
        raw=self.regressor.predict(x,num_threads=self.settings['threads']);p=np.clip(raw,0,1).astype(np.float32)
        c=np.clip(self.classifier.predict(x,num_threads=self.settings['threads']),0,1).astype(np.float32)
        if not np.isfinite(p).all() or not np.isfinite(c).all():raise FloatingPointError('Nonfinitemodelprediction')
        return {'share':p,'score_classifier':c,'score':(.5*p+.5*c).astype(np.float32),'share_raw':np.asarray(raw,np.float32)}

    def predict(self,db,*,batch_size=256):
        import numpy as np
        if isinstance(db,Mapping):raise TypeError('Predictionacceptsdbonly')
        a=np.asarray(db)
        if len(a)<1 or type(batch_size)is not int or batch_size<1:raise ValueError('Nonemptyqueryandpositivebatchrequired')
        parts=[self._predict_features(transform(a[s:s+batch_size],basis=self.basis,include_shape=self.include_shape,batch_size=batch_size)) for s in range(0,len(a),batch_size)]
        return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}

    @classmethod
    def load(cls,folder,*,expected_binding,allow_fixture=False):
        import lightgbm as lgb
        from pipeline.features_r12 import compact_feature_names
        folder=Path(folder);s=json.loads((folder/'summary.json').read_text());fr=json.loads((folder/'FROZEN_BEFORE_HELDOUT.json').read_text())
        if s.get('format')!='r12_compact_physics_lgbm_v1' or s.get('status')!='completed' or s.get('training_binding')!=expected_binding or (folder/'FAILED.json').exists():raise ValueError('Completedboundmodelfolderrequired')
        if s.get('fixture_only') and not allow_fixture:raise ValueError('Fixturemodelnotproduction')
        if fr['summary_sha256']!=sha(folder/'summary.json') or fr['binding_sha256']!=token(expected_binding):raise ValueError('Frozenmodelreceiptdiffers')
        if set(s['source_code_sha256'])!=set(CODE_NAMES):raise ValueError('Incompletecodeclosure')
        expected={'share.txt','presence.txt','feature_schema.json','validation_fit_predictions.npz','fit_plan.json'}|({'spectral_basis.npz'} if s['include_shape'] else set())
        if set(s['artifact_sha256'])!=expected:raise ValueError('Incompleteartifactclosure')
        if s['lightgbm_version']!=lgb.__version__:raise ValueError('LightGBMversiondiffersfromfit')
        for n,d in s['source_code_sha256'].items():
            if sha(ROOT/n)!=d:raise ValueError('Frozenmodelsourcechanged:'+n)
        for n,d in s['artifact_sha256'].items():
            if safe_path(folder,n)!=folder.resolve()/n or sha(folder/n)!=d:raise ValueError('Modelartifactchanged:'+n)
        schema=json.loads((folder/'feature_schema.json').read_text())
        if token(schema)!=s['feature_schema_sha256'] or schema['feature_names']!=list(compact_feature_names(include_shape=s['include_shape'])):raise ValueError('Featureschemadiffers')
        obj=cls(include_shape=s['include_shape'],basis_path=folder/'spectral_basis.npz' if s['include_shape'] else None,
            basis_sha256=s['basis_sha256'],basis_training_binding=s['basis_training_binding'],settings=s['settings'],seed=s['seed'])
        obj.regressor=lgb.Booster(model_file=str(folder/'share.txt'));obj.classifier=lgb.Booster(model_file=str(folder/'presence.txt'));obj.summary=s
        if obj.regressor.num_feature()!=s['feature_count'] or obj.classifier.num_feature()!=s['feature_count']:raise ValueError('Modelfeaturecountdiffers')
        return obj

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['fit','evaluate'],default='fit')
    p.add_argument('--data-dir',type=Path);p.add_argument('--arm',choices=(*ARMS,'all'))
    p.add_argument('--expected-data-hash',help='Exact approved data_hash; stale draft packages are rejected')
    p.add_argument('--include-shape',choices=['0','1','both'],default='both');p.add_argument('--basis',type=Path);p.add_argument('--basis-sha256')
    p.add_argument('--output',type=Path);p.add_argument('--validate-only',action='store_true');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--global-freeze',type=Path);p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        environment(2);sys.addaudithook(guard);numeric_fixture(a.output);return
    if not a.data_dir or not a.arm or not a.output:p.error('--data-dir --arm --output are required')
    environment(a.threads);sys.addaudithook(guard)
    execute(a)

def execute(a):
    import numpy as np,pandas as pd
    from pipeline.dataset_r12 import validate_r12_bundle,load_r12_arrays
    data_dir=a.data_dir.resolve();output=a.output.resolve()
    if ROOT not in data_dir.parents or ROOT not in output.parents:raise ValueError('Project-containednewR12pathsrequired')
    m=validate_r12_bundle(data_dir,verify_all=True)
    if not a.expected_data_hash or m.get('data_hash')!=a.expected_data_hash:raise ValueError('Explicitapproveddata_hashrequired; draft/foreignbundleforbidden')
    if (m.get('version')!='r12_cpu_data_ablation_v1' or m.get('status')!='completed'
        or m.get('all_original_source_banks_and_R9_code_unchanged') is not True
        or m.get('model_fitting_or_Test_or_unused_report_access') is not False):raise ValueError('UnapprovedR12datasetmanifest')
    if m['configuration']['seed']!=2026102712:raise ValueError('PredeclaredR12seeddiffers')
    if m['source_audit']['role_separation']['all_active_roles_minimum_support_gap_hours']<24:raise ValueError('24hsupportpurgefailed')
    for name,d in m['file_sha256'].items():
        if sha(safe_path(data_dir,name))!=d:raise ValueError('Datasetartifactchanged:'+name)
    roles=('train','validation_fit','validation_gate','eval')
    # Metadata only: no gate/eval acoustic arrays or targets are opened here.
    for arm in ARMS:
        metadata={}
        for role in roles:
            entry=m['arms'][arm][role]
            for k,dk in [('path','sha256'),('metadata_path','metadata_sha256'),('recipes_path','recipes_sha256')]:
                if sha(safe_path(data_dir,entry[k]))!=entry[dk]:raise ValueError('Datasetentryhashdiffers')
            metadata[role]=pd.read_parquet(safe_path(data_dir,entry['metadata_path']),columns=['base_window_index','background_group'])
        for i,role in enumerate(roles):
            for other in roles[i+1:]:
                if (set(metadata[role].background_group)&set(metadata[other].background_group)
                    or set(metadata[role].base_window_index)&set(metadata[other].base_window_index)):
                    raise ValueError('R12backgroundsource/groupseparationfailed:'+arm+role+other)
        for role in roles[1:]:
            if m['arms'][arm][role]!=m['arms']['L0_H0'][role]:raise ValueError('Factorialarmsmustsharefixedvalidationpanels')
    requested=list(ARMS) if a.arm=='all' else [a.arm]
    shapes=[False,True] if a.include_shape=='both' else [a.include_shape=='1']
    if True in shapes:
        if a.basis is None:raise ValueError('Shapeonrequiresroot-providedsharedpureTRAINbasis')
        basis_path=a.basis.resolve();basis_hash=sha(basis_path)
        if a.basis_sha256 and a.basis_sha256!=basis_hash:raise ValueError('ExplicitbasisfileSHAdiffers')
        from pipeline.features_r12 import load_spectral_basis
        basis=load_spectral_basis(basis_path,expected_sha256=basis_hash,expected_training_binding=m['basis_training_binding'])
        # Bind PCA to its exact raw TRAIN-only input, not a labelled mixture.
        bp=safe_path(data_dir,m['background_basis']['path'])
        with np.load(bp,allow_pickle=False) as z:
            from pipeline.features_r12 import _array_digest,_digest
            if _array_digest(z['db'])!=basis['input_db_sha256'] or _digest(z['source_id'].astype(str).tolist())!=basis['source_ids_sha256']:
                raise ValueError('PCAbasisnotfitonthisdatasetpureTRAINbackgrounds')
    else:basis_path=None;basis_hash=None
    cfg={'threads':a.threads};manifest_sha=sha(data_dir/'manifest.json')
    def model_name(arm,shape):return arm.replace('_','')+'P'+str(int(shape))
    def binding(arm,shape):
        return {'format':'r12_compact_tree_source_binding_v1','data_hash':m['data_hash'],'data_manifest_sha256':manifest_sha,
            'arm':arm,'include_shape':shape,'TRAIN_npz_sha256':m['arms'][arm]['train']['sha256'],
            'validation_fit_npz_sha256':m['arms'][arm]['validation_fit']['sha256'],
            'basis_sha256':basis_hash if shape else None,'basis_training_binding':m['basis_training_binding'] if shape else None,
            'source_group_24h_purge_passed':True,'development_not_blind_or_official':True}
    if a.validate_only:
        print(json.dumps({'status':'preflight_passed','data_manifest_sha256':manifest_sha,'requested_models':[model_name(arm,s) for arm in requested for s in shapes],
            'basis_sha256':basis_hash,'gate_eval_arrays_opened':False,'training_executed':False}));return m
    if a.phase=='fit':
        if output.exists():raise FileExistsError('FreshR12runrootrequired; nopartialreuse')
        output.mkdir(parents=True)
        runplan={'format':'r12_tree_factorial_fit_plan_v1','created_utc':utc(),'data_manifest_sha256':manifest_sha,
            'data_hash':m['data_hash'],'basis_sha256':basis_hash,'settings':dict(DEFAULTS,**cfg),'seed':2026102712,
            'models':[{'name':model_name(arm,s),'arm':arm,'shape':s} for arm in requested for s in shapes],
            'code_sha256':{n:sha(ROOT/n) for n in (*CODE_NAMES,'pipeline/dataset_r12.py','ops/build_r12_cpu_data.py')},
            'gate_eval_R11_Test_opened':False}
        write(output/'PRE_FIT_FREEZE.json',runplan);completed={}
        try:
            val=load_r12_arrays(data_dir,requested[0],'validation_fit',verify_all=False);dataset(val,'validation_fit',expected_rows=1000)
            for arm in requested:
                train=load_r12_arrays(data_dir,arm,'train',verify_all=False);dataset(train,'train',expected_rows=12000)
                for shape in shapes:
                    name=model_name(arm,shape);print(json.dumps({'stage':'fit_start','model':name,'train':len(train['share']),'shape':shape}),flush=True)
                    tree=CompactPhysicsLGBM(include_shape=shape,basis_path=basis_path if shape else None,basis_sha256=basis_hash if shape else None,
                        basis_training_binding=m['basis_training_binding'] if shape else None,settings=cfg)
                    summary=tree.fit(train,val,output/name,binding(arm,shape),fixture=False)
                    restored=CompactPhysicsLGBM.load(output/name,expected_binding=binding(arm,shape))
                    p=tree.predict(val['db'][:7],batch_size=7);q=restored.predict(val['db'][:7],batch_size=1)
                    if any(not np.array_equal(p[k],q[k]) for k in p):raise AssertionError('Productionfitreload1/7differs')
                    completed[name]=sha(output/name/'summary.json');write(output/'progress.json',{'status':'fitting','completed':completed,'last_model':name})
                    print(json.dumps({'stage':'fit_completed','model':name,'best_iteration':summary['best_iteration'],'validation_fit_rmse':summary['validation_fit_metrics']['rmse']}),flush=True)
            for n,d in runplan['code_sha256'].items():
                if sha(ROOT/n)!=d:raise ValueError('Sourcechangedmidfit:'+n)
            validate_r12_bundle(data_dir,verify_all=True)
            write(output/'ALL_TREES_FROZEN.json',{'format':'r12_trees_frozen_v1','status':'completed','all_models_frozen':True,
                'data_manifest_sha256':manifest_sha,'summary_sha256':completed,'model_summary_sha256':{str((output/n/'summary.json').relative_to(ROOT)):d for n,d in completed.items()},
                'number_models':len(completed),'gate_eval_R11_Test_opened':False})
            write(output/'progress.json',{'status':'all_trees_frozen','completed':completed,'heldout_not_opened':True})
        except BaseException as e:
            write(output/'FAILED.json',{'type':type(e).__name__,'error':str(e),'partial_reuse_allowed':False});raise
        return completed
    # One explicit global17-model barrier; do not opportunistically inspect
    # gate/eval while other candidate settings remain changeable.
    if a.global_freeze is None:raise ValueError('Evaluationrequiresrootglobalall17-model-freezereceipt')
    global_path=a.global_freeze.resolve();globalstate=json.loads(global_path.read_text())
    if (globalstate.get('format')!='r12_all_models_frozen_v1' or globalstate.get('all_models_frozen') is not True
        or globalstate.get('data_manifest_sha256')!=manifest_sha or len(globalstate.get('model_summary_sha256',{}))!=17):
        raise ValueError('Incompleteorforeigntree8+CNN9globalfreeze')
    for n,d in globalstate['model_summary_sha256'].items():
        summary_path=safe_path(ROOT,n)
        if sha(summary_path)!=d or json.loads(summary_path.read_text()).get('status')!='completed':raise ValueError('Globalmodelsummarychanged:'+n)
    evaldir=output/'heldout_evaluation'
    if evaldir.exists() or (output/'FAILED.json').exists():raise FileExistsError('Freshheldoutevaluationofunfailedrunrequired')
    evaldir.mkdir();evaluation_plan={'global_freeze_sha256':sha(global_path),'data_manifest_sha256':manifest_sha,
        'models':[model_name(arm,s) for arm in requested for s in shapes],'panels':['validation_gate','eval','high_snr_long'],
        'weights_or_models_refit':False,'known_R11_panels_not_recipe_selection':True}
    write(evaldir/'EVALUATION_PLAN.json',evaluation_plan);results={}
    for arm in requested:
        for shape in shapes:
            name=model_name(arm,shape);summary_path=output/name/'summary.json'
            if globalstate['model_summary_sha256'].get(str(summary_path.relative_to(ROOT)))!=sha(summary_path):raise ValueError('Ownmodelabsentfromglobalfreeze')
            tree=CompactPhysicsLGBM.load(output/name,expected_binding=binding(arm,shape));results[name]={}
            for role in ['validation_gate','eval','high_snr_long']:
                if role=='high_snr_long':
                    desc=m['high_snr_long']
                    if desc.get('created') is not True:
                        results[name][role]={'status':'not_created','reason':desc.get('reason')};continue
                    if sha(safe_path(data_dir,desc['path']))!=desc['sha256']:raise ValueError('HighSNRpanelchanged')
                    with np.load(safe_path(data_dir,desc['path']),allow_pickle=False) as z:d={k:z[k] for k in z.files}
                else:d=load_r12_arrays(data_dir,arm,role,verify_all=False)
                desc=m['high_snr_long'] if role=='high_snr_long' else m['arms'][arm][role]
                md=pd.read_parquet(safe_path(data_dir,desc['metadata_path']),columns=['sample_id','special_zero'])
                if not np.array_equal(md.sample_id.astype(str).to_numpy(),d['sample_id'].astype(str)):raise ValueError('Heldoutmetadataorderdiffers')
                d['special_zero']=md.special_zero.to_numpy(bool)
                dataset(d,role,expected_rows=500);pred=tree.predict(d['db']);results[name][role]=metrics(d,pred)
                np.savez_compressed(evaldir/(name+'_'+role+'_predictions.npz'),**pred,share_target=d['share'],presence_target=d['presence'],sample_id=d['sample_id'])
    if sha(global_path)!=evaluation_plan['global_freeze_sha256']:raise ValueError('Globalfreezechangedduringevaluation')
    write(evaldir/'RESULTS.json',{'format':'r12_tree_heldout_results_v1','status':'completed','evaluations':results,
        'data_manifest_sha256':manifest_sha,'global_freeze_sha256':sha(global_path),'model_or_weight_refitting':False,
        'Test_or_known_R11_recipe_selection':False})
    lines=['# R12 compact LightGBM: 동결 후 평가','',
        'TRAIN 12,000과 validation_fit 1,000만으로 tree를 학습하고, 17개 후보 전체의 summary SHA가 동결된 뒤 아래 평가를 열었습니다. PCA는 순수 SA TRAIN 배경에서만 피팅된 공유 상태입니다.',
        'Gate는 후속 recipe 선택용 개발 검증이며 독립 OOF나 실제 대회 점수가 아닙니다. 이 실행기는 gate에 맞춰 모델·가중치를 다시 피팅하지 않습니다.',
        '', '|모델|패널|N|RMSE|share AUC|classifier AUC|hybrid AUC|bias|예측RMS|share≤.01|share≥.99|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'undefined' if v is None else f'{v:.12f}'
    for name,panels in results.items():
        for role,r in panels.items():
            if r.get('status')=='not_created':continue
            lines.append('|'+ '|'.join([name,role,str(r['n'])]+[fmt(r[k]) for k in ['rmse','share_auc','classifier_auc','hybrid_auc','bias','prediction_rms','share_saturation_le001','share_saturation_ge099']])+'|')
    lines+=['','long-positive·special-zero·SF-dense-zero·target-tail 보호 slice의 수치와 표본수는 JSON에 모두 기록했습니다. 특징과 추론 입력은 db뿐이며 source ID와 recipe는 특징에 사용하지 않았습니다.',
        '', '[전체 수치와 보호 slice](RESULTS.json) / [평가 전 승인·설정](EVALUATION_PLAN.json)']
    (evaldir/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return results

def numeric_fixture(output=None):
    """Tiny synthetic API/physics fixture, never source-data model training."""
    import numpy as np,tempfile
    from pipeline.features_r12 import fit_spectral_basis,save_spectral_basis
    out=Path(output) if output is not None else Path(tempfile.mkdtemp(prefix='r12_tree_fixture_'))/'case'
    if out.exists():raise FileExistsError('Freshfixtureoutputrequired')
    out.mkdir(parents=True);rng=np.random.default_rng(612)
    pure=(70+rng.normal(0,4,(64,15,25))).astype(np.float32);bb={'fixture_only':True,'source':'generated_unmixed_backgrounds','role':'train'}
    basis=fit_spectral_basis(pure,[f'pure_{i}' for i in range(64)],training_binding=bb)
    receipt=save_spectral_basis(basis,out/'basis.npz')
    def generated(n,role):
        db0=75+rng.normal(0,5,(n,15,25));b=10**((db0-db0.max((1,2),keepdims=True))/10)
        add=np.exp(rng.normal(-3,1,(n,15,25)));add[:round(.2*n)]=0;e=b+add;q=add/e
        return {'db':np.round(10*np.log10(e)+db0.max((1,2),keepdims=True),1).astype(np.float32),
            'share':(add.sum((1,2))/e.sum((1,2))).astype(np.float32),'presence':(add.sum((1,2))>0).astype(np.int64),
            'cell_share':q.astype(np.float32),'sample_id':np.array([role+str(i) for i in range(n)]),
            'site':np.array(['synthetic_'+str(i%2) for i in range(n)]),
            'source_id':np.array([role+'_source'+str(i) for i in range(n)]),
            'background_group':np.array([role+'_group'+str(i%3) for i in range(n)])}
    train=generated(128,'train');val=generated(48,'fit');query=generated(32,'query');results=[]
    for shape in [False,True]:
        model=CompactPhysicsLGBM(include_shape=shape,basis_path=out/'basis.npz' if shape else None,
            basis_sha256=receipt['file_sha256'] if shape else None,basis_training_binding=bb if shape else None,
            settings={'maximum_trees':10,'early_stopping_patience':3,'threads':2,'minimum_leaf_rows':5})
        binding={'fixture_only':True,'case':'physical_compact'+str(shape)};folder=out/('shape'+str(int(shape)))
        summary=model.fit(train,val,folder,binding,fixture=True)
        restored=CompactPhysicsLGBM.load(folder,expected_binding=binding,allow_fixture=True)
        a=model.predict(query['db'],batch_size=32);b=restored.predict(query['db'],batch_size=1);c=restored.predict(query['db'],batch_size=7)
        if any(not np.array_equal(a[k],b[k]) or not np.array_equal(a[k],c[k]) for k in a):raise AssertionError('Savedreloadbatch1/7/32predictiondiffers')
        try:restored.predict(query)
        except TypeError:pass
        else:raise AssertionError('Labelledquerydictionaryaccepted')
        for invalid in [np.full((1,15,25),np.nan,np.float32),np.full((1,15,25),np.inf,np.float32),np.zeros((1,15,25),np.int64)]:
            try:restored.predict(invalid)
            except ValueError:pass
            else:raise AssertionError('Invalidqueryaccepted')
        try:CompactPhysicsLGBM.load(folder,expected_binding=binding)
        except ValueError:pass
        else:raise AssertionError('Fixtureweightsacceptedwithoutoverride')
        results.append({'include_shape':shape,'features':summary['feature_count'],'replay1_7_32_exact':True,
            'labelled_query_rejected':True,'allmissing_inf_integer_query_rejected':True,'fixture_production_reuse_rejected':True,'query_metrics':metrics(query,a)})
    report={'status':'passed','scope':'tiny generated numeric fixture; no actual source/TRAIN/Test/model artifacts read',
        'train_rows':128,'validation_fit_rows':48,'query_rows':32,'maximum_trees':10,'threads':2,'cases':results,
        'code_sha256':{n:sha(ROOT/n) for n in CODE_NAMES}}
    write(out/'VERIFICATION.json',report);print(json.dumps({'fixture':'passed','output':str(out),'features':[r['features'] for r in results]}));return report

if __name__=='__main__':main()
