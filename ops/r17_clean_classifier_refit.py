#!/usr/bin/env python3
"""Separate CPU-only R17 classifier ablation; immutable R16 models are controls.

No query inputs, regression retraining, neural training, or ensemble selection.
All comparisons on previously inspected panels are retrospective diagnostics.
"""
from __future__ import annotations
import argparse, datetime, gc, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.r17_audit_common import sha,load,jsave,panel_data
from pipeline.training_r16_multisupport import tree_features

OUT=ROOT/'reports/r17_clean_pipeline_r1/refit'
MODELS=ROOT/'models/r17_clean_classifier_r1'
DATA=ROOT/'data/synthetic/r17_clean_v1'
PANELS=('local_gate','domain_holdout','native12_VS_hf_high_negative',
        'native12_VS_st_francis_dense_negative','local_report')
SEED=2026103003

def no_query(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        s=os.fsdecode(args[0]).lower()
        if Path(s).name in ('test.csv','sample_submission.csv','query_predictions.npz'):
            raise PermissionError('R17 classifier experiment has no query access')

def event(**v):
    v={'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**v}
    jsave(OUT/'progress.json',v);print(json.dumps(v,ensure_ascii=False),flush=True)

def classifier_schema():
    from ops.r17_clean_classifier_features import schema,classifier_indices
    return schema,classifier_indices

def cached(name,db):
    folder=OUT/'features';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(name+'.npy');meta=folder/(name+'.json')
    from pipeline.trees_r16_production import array_sha
    identity={'db_sha256':array_sha(db),'shape':[len(db),2033]}
    if path.exists():
        m=json.loads(meta.read_text())
        if m['input']!=identity or sha(path)!=m['sha256']:raise ValueError('Feature cache changed')
        return np.load(path,mmap_mode='r')
    start=time.perf_counter()
    x=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(db),2033))
    for i in range(0,len(db),256):
        x[i:i+256]=tree_features(db[i:i+256],'classification')
        if not i%4096:event(phase='feature_extraction',dataset=name,rows=i,total=len(db))
    x.flush();del x
    jsave(meta,{'input':identity,'sha256':sha(path),'seconds':time.perf_counter()-start})
    return np.load(path,mmap_mode='r')

def metrics(y,p,weights=None):
    from sklearn.metrics import roc_auc_score,log_loss
    y=np.asarray(y,int);p=np.asarray(p,float)
    w=np.ones(len(y))/len(y) if weights is None else np.asarray(weights)/np.sum(weights)
    return {'n':len(y),'auc':float(roc_auc_score(y,p,sample_weight=w)),
        'brier':float(w@((p-y)**2)),'logloss':float(log_loss(y,p,sample_weight=w,labels=[0,1])),
        'score_mean':float(w@p),'score_ge090':float(w@(p>=.9)),
        'negative_mean':float(np.average(p[y==0],weights=w[y==0])),
        'positive_mean':float(np.average(p[y==1],weights=w[y==1])),
        'fpr_at05':float(np.average(p[y==0]>=.5,weights=w[y==0]))}

def panel_cache():
    schema,_=classifier_schema();OUT.mkdir(parents=True,exist_ok=True)
    for name in PANELS:
        d=panel_data(name)
        x=cached(name,d['db'])
        base=load(ROOT/f'models/r16_v1_fourway_r2/predictions/{name}.npz')
        if not np.array_equal(d['sample_id'],base['sample_id']):raise ValueError('Panel order changed')
        keep={k:d[k] for k in ('sample_id','share','site','base_window_index','background_group') if k in d}
        if 'weights' in base:keep['weights']=base['weights']
        np.savez_compressed(OUT/(name+'_metadata.npz'),**keep)
        event(phase='panel_features_complete',panel=name,n=len(x))

def data_path(role,regime):return DATA/(role+'_'+regime+'.npz')

def train():
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    schema,indices=classifier_schema()
    OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    plan=OUT/'PROTOCOL.json'
    config={'status':'fixed_before_fit','seed':SEED,'maximum_trees':1500,'ES':40,
        'LGB_modes':['full','direct','aggressive'],'Cat_modes':['full','aggressive'],
        'learning_rate':.035,'LGB_leaves':31,'min_child_samples':50,'Cat_depth':8,
        'threads':4,'regimes':['D0','H0'],'early_stop_role':'earlystop',
        'ensemble_fit_never_used_for_fit_or_early_stopping':True,
        'regression_models_unchanged':True,'neural_training_steps':0,'Test_used':False,
        'selection_rule':'Report all preregistered candidates; no policy chosen on retrospective panels',
        'source_files':{str(data_path(role,r).relative_to(ROOT)):sha(data_path(role,r))
                        for role in ('train','earlystop','ensemble_fit') for r in ('D0','H0')},
        'source_code_sha256':{f:sha(ROOT/f) for f in ('ops/r17_clean_classifier_refit.py','ops/r17_clean_classifier_features.py')}}
    if plan.exists():
        if json.loads(plan.read_text())!=config:raise ValueError('Registered experiment changed')
    else:jsave(plan,config)
    predictions={};rows=[]
    started=time.perf_counter()
    for regime in ('D0','H0'):
        tr=load(data_path('train',regime));va=load(data_path('earlystop',regime));fit=load(data_path('ensemble_fit',regime))
        for a,b in ((tr,va),(tr,fit),(va,fit)):
            if set(a['base_window_index']) & set(b['base_window_index']):raise ValueError('Background role overlap')
        for d in (tr,va,fit):
            if not np.array_equal(d['share']>0,d['presence']>0):raise ValueError('Presence target mismatch')
        xx=cached('train_'+regime,tr['db']);vv=cached('earlystop_'+regime,va['db']);ff=cached('ensemble_fit_'+regime,fit['db'])
        for family,modes in (('lightgbm',('full','direct','aggressive')),('catboost',('full','aggressive'))):
            for mode in modes:
                folder=MODELS/family/regime/mode;folder.mkdir(parents=True,exist_ok=True)
                done=folder/'COMPLETED.json';idx=np.asarray(indices(mode),int)
                if done.exists():
                    rec=json.loads(done.read_text())
                    for f,h in rec['artifact_sha256'].items():
                        if sha(folder/f)!=h:raise ValueError('Completed experiment artifacts changed')
                    event(phase='verified_existing_complete',family=family,regime=regime,mode=mode)
                else:
                    start=time.perf_counter();x=np.asarray(xx[:,idx]);v=np.asarray(vv[:,idx])
                    event(phase='classifier_fit',family=family,regime=regime,mode=mode,features=len(idx),train_n=len(x))
                    if family=='lightgbm':
                        model=lgb.LGBMClassifier(n_estimators=1500,objective='binary',metric='binary_logloss',
                            learning_rate=.035,num_leaves=31,min_child_samples=50,max_bin=127,n_jobs=4,
                            random_state=SEED,verbosity=-1,colsample_bytree=.85,subsample=.85,subsample_freq=1,
                            deterministic=True,force_col_wise=True)
                        history={}
                        model.fit(x,tr['presence'],eval_set=[(v,va['presence'])],callbacks=[
                            lgb.early_stopping(40,verbose=False),lgb.record_evaluation(history)])
                        model.booster_.save_model(str(folder/'classification.txt'));best=int(model.best_iteration_)
                    else:
                        model=CatBoostClassifier(iterations=1500,loss_function='Logloss',eval_metric='Logloss',
                            learning_rate=.035,depth=8,l2_leaf_reg=3,thread_count=4,task_type='CPU',
                            grow_policy='SymmetricTree',allow_writing_files=False,verbose=False,random_seed=SEED)
                        model.fit(x,tr['presence'],eval_set=(v,va['presence']),early_stopping_rounds=40,use_best_model=True)
                        model.save_model(str(folder/'classification.cbm'));best=int(model.best_iteration_)+1
                        history=model.get_evals_result()
                    jsave(folder/'history.json',history)
                    jsave(folder/'schema.json',{**schema(mode),'original_indices':idx.tolist()})
                    rec={'family':family,'regime':regime,'mode':mode,'features':len(idx),
                        'best_iteration':best,'seconds':time.perf_counter()-start,'train_rows':len(tr['share']),
                        'earlystop_rows':len(va['share']),'protocol_sha256':sha(plan),
                        'artifact_sha256':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}}
                    jsave(done,rec);del model,x,v;gc.collect()
                if family=='lightgbm':
                    model=lgb.Booster(model_file=str(folder/'classification.txt'))
                    pred=lambda x:model.predict(x,num_threads=4)
                else:
                    model=CatBoostClassifier();model.load_model(str(folder/'classification.cbm'))
                    pred=lambda x:model.predict_proba(x,thread_count=4)[:,1]
                outputs={};key=family+'__'+regime+'__'+mode
                for name,d,source in (('earlystop_'+regime,va,vv),('ensemble_fit_'+regime,fit,ff)):
                    p=pred(np.asarray(source[:,idx]));outputs[name]=p
                    rows.append({'family':family,'regime':regime,'mode':mode,'panel':name,**metrics(d['presence'],p)})
                for name in PANELS:
                    meta=load(OUT/(name+'_metadata.npz'));source=np.load(OUT/'features'/(name+'.npy'),mmap_mode='r')
                    p=pred(np.asarray(source[:,idx]));outputs[name]=p
                    rows.append({'family':family,'regime':regime,'mode':mode,'panel':name,**metrics(meta['share']>0,p,meta.get('weights'))})
                np.savez_compressed(folder/'predictions.npz',**outputs)
                # Predictions are separately hashed, so a complete model can be replayed without retraining.
                jsave(folder/'PREDICTION_RECEIPT.json',{'sha256':sha(folder/'predictions.npz'),'protocol_sha256':sha(plan)})
                predictions[key]=outputs
                pd.DataFrame(rows).to_csv(OUT/'metrics_progress.csv',index=False)
                event(phase='classifier_complete',family=family,regime=regime,mode=mode,
                      best_iteration=rec['best_iteration'],seconds=rec['seconds'])
                del model;gc.collect()
        del tr,va,fit,xx,vv,ff;gc.collect()
    for family,modes in (('lightgbm',('full','direct','aggressive')),('catboost',('full','aggressive'))):
        for mode in modes:
            a=predictions[family+'__D0__'+mode];b=predictions[family+'__H0__'+mode]
            for name in PANELS:
                meta=load(OUT/(name+'_metadata.npz'));p=.5*(a[name]+b[name])
                rows.append({'family':family,'regime':'D0_H0_50_50','mode':mode,'panel':name,
                             **metrics(meta['share']>0,p,meta.get('weights'))})
    pd.DataFrame(rows).to_csv(OUT/'classifier_metrics.csv',index=False,encoding='utf-8-sig')
    comparisons=[]
    for name in PANELS:
        base=load(ROOT/f'models/r16_v1_fourway_r2/predictions/{name}.npz')
        for j,family in enumerate(('legacy_LGB5','legacy_Cat5','frozen_Dasheng','frozen_DINO','frozen_TimeMoE')):
            p=.5*(base['classifier_D0'][:,j]+base['classifier_H0'][:,j])
            comparisons.append({'family':family,'regime':'D0_H0_50_50','mode':'frozen_control','panel':name,
                                **metrics(base['target_share']>0,p,base.get('weights'))})
    pd.DataFrame(rows+comparisons).to_csv(OUT/'classifier_comparison.csv',index=False,encoding='utf-8-sig')
    jsave(OUT/'COMPLETED.json',{'status':'completed','elapsed_seconds':time.perf_counter()-started,
        'CPU_classifier_fits':10,'new_neural_training_steps':0,'regression_retraining':False,
        'Test_used':False,'protocol_sha256':sha(plan),'metrics_sha256':sha(OUT/'classifier_comparison.csv')})

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['panels','train']);a=p.parse_args()
    sys.addaudithook(no_query)
    if a.phase=='panels':panel_cache()
    else:train()

if __name__=='__main__':main()
