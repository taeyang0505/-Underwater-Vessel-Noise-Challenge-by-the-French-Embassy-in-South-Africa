"""R16 production CPU trees: relative8 physics regression + temporal classifier.

PCA, LF anchors, fitted background priors and source/site IDs are not features.
Regression3827 and classification700 are deliberately separate feature spaces.
"""
from __future__ import annotations
import hashlib,json,time,gc
from pathlib import Path
import numpy as np
from pipeline.relative_grid_r6_v2 import relative_grid_v2,FEATURE_NAMES
from pipeline.physics_features_r6_v2 import physics_features_r6_v2,FEATURE_NAMES as PHYSICS_NAMES

ROOT=Path(__file__).resolve().parents[1]
TEMPORAL_CHANNELS=(0,1,2,3,6)
CODE_NAMES=('pipeline/trees_r16_production.py','pipeline/relative_grid_r6_v2.py',
            'pipeline/relative_grid_r6.py','pipeline/physics_features_r6_v2.py')
DEFAULTS={'maximum_trees':1500,'early_stopping_patience':40,'learning_rate':.035,
          'num_leaves':31,'minimum_leaf_rows':50,'max_bin':127,'threads':2,'feature_batch_size':512}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def token(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def array_sha(a):
    a=np.ascontiguousarray(a)
    if a.dtype.kind=='O':raise TypeError('Object arrays forbidden')
    return hashlib.sha256(json.dumps([str(a.dtype),list(a.shape)]).encode()+a.tobytes()).hexdigest()

def schemas():
    reg=[f'grid/{n}/t{t:02d}/f{f:02d}'for n in FEATURE_NAMES for t in range(15)for f in range(25)]
    reg +=[f'{stat}/{n}/f{f:02d}'for stat in ('mean','std_ddof0','max','min')for n in FEATURE_NAMES for f in range(25)]
    reg +=['physics/'+n for n in PHYSICS_NAMES]
    clf=[f'temporal/{FEATURE_NAMES[c]}/{stat}/f{f:02d}'for c in TEMPORAL_CHANNELS for stat in ('mean','std_ddof0','max')for f in range(25)]
    clf +=[f'temporal_band_correlation/{f:02d}/{g:02d}'for f in range(25)for g in range(f+1,25)]
    clf +=[f'observed_fraction/f{f:02d}'for f in range(25)]
    assert len(reg)==3827 and len(clf)==700
    return {'regression':reg,'classification':clf,'learned_feature_state':False,
            'classifier_excludes_profile_sqrtw_PCA_LFanchor':True,'query_inputs':['db']}

def db_check(db):
    if isinstance(db,dict):raise TypeError('Features/predict accept db only')
    d=np.asarray(db)
    if d.ndim!=3 or d.shape[1:]!=(15,25)or d.dtype.kind!='f' or not len(d):raise ValueError('floating db N15x25 required')
    if np.isinf(d).any()or not np.isfinite(d).any((1,2)).all():raise ValueError('Invalid db support')
    return d

def classification_from_grid(x,w):
    obs=w>0;count=obs.sum(1);cols=[]
    for c in TEMPORAL_CHANNELS:
        a=x[:,c].astype(np.float64)
        mean=np.divide(a.sum(1),count,out=np.zeros_like(count,dtype=float),where=count>0)
        var=np.divide(np.where(obs,(a-mean[:,None])**2,0).sum(1),count,out=np.zeros_like(mean),where=count>0)
        mx=np.where(count>0,np.max(np.where(obs,a,-np.inf),1),0)
        cols.extend([mean,np.sqrt(np.maximum(var,0)),mx])
    a=x[:,0].astype(np.float64)
    for f in range(25):
        for g in range(f+1,25):
            m=obs[:,:,f]&obs[:,:,g];n=m.sum(1);u,v=a[:,:,f],a[:,:,g]
            mu=np.divide(np.where(m,u,0).sum(1),n,out=np.zeros(len(a)),where=n>0)
            mv=np.divide(np.where(m,v,0).sum(1),n,out=np.zeros(len(a)),where=n>0)
            uu=np.where(m,u-mu[:,None],0);vv=np.where(m,v-mv[:,None],0)
            den=np.sqrt((uu*uu).sum(1)*(vv*vv).sum(1))
            cols.append(np.divide((uu*vv).sum(1),den,out=np.zeros(len(a)),where=den>1e-12)[:,None])
    cols.append(count.astype(float)/15.)
    return np.concatenate(cols,1).astype(np.float32)

def transform(db,kind,batch_size=512):
    d=db_check(db)
    if kind not in ('regression','classification'):raise ValueError('Explicit feature kind required')
    result=np.empty((len(d),3827 if kind=='regression' else 700),np.float32)
    for start in range(0,len(d),batch_size):
        part=d[start:start+batch_size];x,w=relative_grid_v2(part)
        if kind=='classification':v=classification_from_grid(x,w)
        else:
            stats=np.concatenate([fn(x,axis=2).reshape(len(x),-1)for fn in(np.mean,np.std,np.max,np.min)],1)
            v=np.concatenate([x.reshape(len(x),-1),stats,physics_features_r6_v2(part)],1).astype(np.float32)
        if not np.isfinite(v).all():raise ValueError('Nonfinite stateless feature')
        result[start:start+len(v)]=v
    return result

def measure(y,p,c):
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y,float);p=np.asarray(p,float);e=p-y;z=y>0
    return {'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),'bias':float(e.mean()),
       'share_auc':float(roc_auc_score(z,p))if len(np.unique(z))==2 else None,
       'classifier_auc':float(roc_auc_score(z,c))if len(np.unique(z))==2 else None,
       'prediction_mean':float(p.mean()),'prediction_rms':float(np.sqrt(np.mean(p*p)))}

class PhysicsLGBM:
    def __init__(self,settings=None,seed=2026102718):
        self.settings={**DEFAULTS,**(settings or{})};self.seed=int(seed);self.regressor=None;self.classifier=None;self.summary=None
        if set(self.settings)!=set(DEFAULTS)or self.settings['threads']!=2 or self.settings['maximum_trees']>1500:raise ValueError('Predeclared CPU2/max1500 contract')
    def fit(self,train,selection,folder,binding,progress=None):
        import lightgbm as lgb
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);began=time.time()
        td=db_check(train['db']);vd=db_check(selection['db']);ty=np.asarray(train['share'],float);vy=np.asarray(selection['share'],float)
        tz=np.asarray(train['presence'],int);vz=np.asarray(selection['presence'],int)
        if not np.array_equal(ty>0,tz>0)or not np.array_equal(vy>0,vz>0):raise ValueError('Physical presence mismatch')
        if len(td)>100000 or set(train['base_window_index'])&set(selection['base_window_index']):raise ValueError('Memory cap or TRAIN/selection overlap')
        pins={n:sha(ROOT/n)for n in CODE_NAMES};iterations={};valpred={};logs={};memory={}
        for kind,target,valtarget in [('regression',ty,vy),('classification',tz,vz)]:
            x=transform(td,kind,self.settings['feature_batch_size']);v=transform(vd,kind,self.settings['feature_batch_size'])
            memory[kind]={'train_feature_bytes':x.nbytes,'selection_feature_bytes':v.nbytes}
            args=dict(n_estimators=self.settings['maximum_trees'],learning_rate=self.settings['learning_rate'],num_leaves=self.settings['num_leaves'],
                min_child_samples=self.settings['minimum_leaf_rows'],max_bin=self.settings['max_bin'],n_jobs=2,random_state=self.seed,verbosity=-1,
                colsample_bytree=.85,subsample=.85,subsample_freq=1,deterministic=True,force_col_wise=True,data_random_seed=self.seed,feature_fraction_seed=self.seed,bagging_seed=self.seed)
            clf=kind=='classification';model=lgb.LGBMClassifier(objective='binary',metric='binary_logloss',**args)if clf else lgb.LGBMRegressor(objective='regression_l2',metric='None',**args)
            def clipped_rmse(y,p):return 'clipped_rmse',float(np.sqrt(np.mean((np.clip(p,0,1)-y)**2))),False
            history={}
            def callback(env):
                if env.iteration==0 or (env.iteration+1)%25==0:
                    event={'event':'boosting','kind':kind,'round':env.iteration+1,'seconds':time.time()-began,'evaluation':[list(a)for a in env.evaluation_result_list]}
                    write(folder/'progress.json',event)
                    if progress:progress(event)
            callback.order=30
            model.fit(x,target,eval_set=[(v,valtarget)],eval_metric='binary_logloss'if clf else clipped_rmse,
                callbacks=[lgb.early_stopping(self.settings['early_stopping_patience'],verbose=False),lgb.record_evaluation(history),callback])
            booster=model.booster_;iterations[kind]=int(model.best_iteration_);logs[kind]=history
            pred=booster.predict(v,num_threads=2);valpred[kind]=pred if clf else np.clip(pred,0,1)
            name='presence.txt'if clf else'share.txt';temp=folder/(name+'.tmp');booster.save_model(str(temp));temp.replace(folder/name)
            if clf:self.classifier=booster
            else:self.regressor=booster
            del x,v,model;gc.collect()
        np.savez_compressed(folder/'selection_predictions.npz',share=valpred['regression'],classifier_prob=valpred['classification'],target_share=vy,
                            target_presence=vz,sample_id=selection['sample_id'])
        write(folder/'feature_schema.json',schemas());write(folder/'boosting_history.json',logs)
        names=['share.txt','presence.txt','selection_predictions.npz','feature_schema.json','boosting_history.json']
        self.summary={'format':'r16_relative8_decoupled_physics_tree_v1','status':'completed','binding':binding,'settings':self.settings,'seed':self.seed,
          'train_rows':len(td),'selection_rows':len(vd),'feature_counts':{'regression':3827,'classification':700},'iterations':iterations,
          'selection_metrics':measure(vy,valpred['regression'],valpred['classification']),'input_sha256':{'train_db':array_sha(td),'train_y':array_sha(ty),'selection_db':array_sha(vd),'selection_y':array_sha(vy)},
          'feature_memory':memory,'artifact_sha256':{n:sha(folder/n)for n in names},'code_sha256':pins,'elapsed_seconds':time.time()-began,
          'PCA_LFanchor_highshare':False,'score_is_independent_classifier_probability':True,'Test_used':False,'early_stopping_uses_shared_development_selection_not_OOF':True}
        if any(sha(ROOT/n)!=h for n,h in pins.items()):raise ValueError('Source changed during fitting')
        write(folder/'summary.json',self.summary);return self.summary
    def predict(self,db):
        d=db_check(db)
        if self.regressor is None or self.classifier is None:raise ValueError('Completed model required')
        p=np.clip(self.regressor.predict(transform(d,'regression',self.settings['feature_batch_size']),num_threads=2),0,1)
        c=self.classifier.predict(transform(d,'classification',self.settings['feature_batch_size']),num_threads=2)
        if not np.isfinite(p).all()or not np.isfinite(c).all():raise ValueError('Invalid model output')
        return {'share':p.astype(np.float32),'classifier_prob':c.astype(np.float32),'score':c.astype(np.float32)}
    @classmethod
    def load(cls,folder,expected_binding=None):
        import lightgbm as lgb
        folder=Path(folder);s=json.loads((folder/'summary.json').read_text())
        if s['status']!='completed' or(folder/'FAILED.json').exists():raise ValueError('Only completed clean model may load')
        if expected_binding is not None and s['binding']!=expected_binding:raise ValueError('Foreign fold/source binding')
        for n,h in s['artifact_sha256'].items():
            if sha(folder/n)!=h:raise ValueError('Tree artifact changed')
        for n,h in s['code_sha256'].items():
            if sha(ROOT/n)!=h:raise ValueError('Tree source changed')
        m=cls(s['settings'],s['seed']);m.summary=s;m.regressor=lgb.Booster(model_file=str(folder/'share.txt'));m.classifier=lgb.Booster(model_file=str(folder/'presence.txt'))
        if m.regressor.num_feature()!=3827 or m.classifier.num_feature()!=700:raise ValueError('Feature contract changed')
        return m
