"""R18 bounded CPU information ablation. No Test access, no neural fit.

Full 17-channel representation versus the same representation plus the one
discarded common level, versus plus all raw dB cells. All models are small
LightGBM direct regressors; they do not estimate a Bayes lower bound.
"""
from pathlib import Path
import os, sys, json, time, gc, datetime
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.r17_audit_common import load, sha, jsave, panel_data
from pipeline.training_r16_multisupport import features

OUT=ROOT/'reports/r18_fundamental_audit_r1/identifiability'
MODELS=ROOT/'models/r18_regression_audit_r1'
DATA=ROOT/'data/synthetic/r17_clean_v1'
FRESH=ROOT/'data/synthetic/r17_clean_freshfit_v1'
SEED=2026103018
MODES=('relative17','relative17_common_level','relative17_raw_db')
PANELS=('local_gate','local_report','domain_holdout','native12_VS_hf_high_negative','native12_VS_st_francis_dense_negative')

def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        if Path(os.fsdecode(args[0])).name.lower() in ('test.csv','sample_submission.csv','query_predictions.npz'):
            raise PermissionError('No query inputs allowed for R18 information ablation')

def progress(**x):
    x={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**x}
    jsave(OUT/'progress.json',x)
    print(json.dumps(x),flush=True)

def math_fixtures():
    rng=np.random.default_rng(SEED)
    db=np.round(rng.normal(80,9,(20,15,25)),1)
    db[0,0]=np.nan; db[1,6:9]=np.nan
    x,w=features(db)
    o=np.isfinite(db)
    centered=db-np.nanmedian(db,axis=(1,2),keepdims=True)
    restored=10*x[:,0]+20*x[:,4]
    profile_error=float(np.max(abs(restored[o]-centered[o])))
    logw=10*np.log10(np.maximum(x[:,7].astype(float)**2,1e-300))
    logw-=np.nanmedian(np.where(o,logw,np.nan),axis=(1,2),keepdims=True)
    weight_error=float(np.max(abs(logw[o]-centered[o])))
    shifted=db+20.
    xx,ww=features(shifted)
    gain_feature_gap=float(abs(xx-x).max())
    gain_weight_gap=float(abs(ww-w).max())
    b=np.exp(rng.normal(1,1,(15,25)))
    a=np.exp(rng.normal(0,1,(15,25)))
    y=float(a.sum()/(a+b).sum())
    y_gain=float((100*a).sum()/(100*(a+b)).sum())
    # Identical fully observed E, different latent decompositions.
    energy=np.full((15,25),200.)
    counter=[]
    for fraction in (0.,.2,.5,.8):
        B=energy*(1-fraction); A=energy-B
        counter.append({'share':float(A.sum()/energy.sum()),'same_E_reconstruction_error':float(abs(B+A-energy).max())})
    # Stationary per-band additions are invisible to the classifier's
    # band-relative time features, but change normalized spectral energy.
    f=np.arange(25)
    base=np.broadcast_to(85-.7*f,(1,15,25)).copy()
    altered=base+np.broadcast_to(np.where(f>=10,8.,2.),(1,15,25))
    bx,bw=features(base); ax,aw=features(altered)
    cls_idx=[i for i in range(17) if i not in (4,7)]
    classifier_collision=float(abs(bx[:,cls_idx]-ax[:,cls_idx]).max())
    if profile_error>1e-5 or weight_error>2e-5 or gain_feature_gap>1e-4 or classifier_collision>1e-5:
        raise AssertionError('Information identity fixture failed')
    result={'profile_and_temporal_reconstruction_max_db_error':profile_error,
        'sqrt_w_reconstruction_centered_db_max_error':weight_error,
        'common_plus20db_full17_max_gap':gain_feature_gap,'common_plus20db_weight_max_gap':gain_weight_gap,
        'share_common_gain_error':abs(y-y_gain),'same_raw_db_different_y_counterexamples':counter,
        'stationary_band_addition_classifier_feature_collision_max':classifier_collision,
        'stationary_addition_regression_profile_max_gap':float(abs(bx[:,4]-ax[:,4]).max()),
        'stationary_addition_weight_max_gap':float(abs(bw-aw).max()),
        'common_offset_is_the_missing_scalar_not_all_spectral_information':True,
        'Bayes_floor_identified':False,'Test_used':False}
    jsave(OUT/'mathematical_fixtures.json',result)
    return result

def rawlevel(db):
    return np.nanmedian(np.asarray(db,float),axis=(1,2))[:,None].astype(np.float32)/100.

def transform(db,mode):
    x=np.empty((len(db),17*375),np.float32)
    for off in range(0,len(db),256):
        xx,_=features(db[off:off+256]);x[off:off+256]=xx.reshape(len(xx),-1)
    if mode=='relative17':return x
    if mode=='relative17_common_level':return np.column_stack((x,rawlevel(db)))
    if mode=='relative17_raw_db':
        raw=np.where(np.isfinite(db),db/100.,0.).reshape(len(db),-1).astype(np.float32)
        return np.column_stack((x,raw))
    raise ValueError(mode)

def metric(d,p,weights=None):
    y=np.asarray(d['share'],float);p=np.clip(np.asarray(p,float),0,1);e=p-y
    w=np.ones(len(y)) if weights is None else np.asarray(weights,float)
    w=w/w.sum()
    sites=np.asarray(d.get('site',np.repeat('all',len(y))),str)
    return {'n':len(y),'rmse':float(np.sqrt(w@(e*e))),'bias':float(w@e),
        'site_macro_RMSE':float(np.mean([np.sqrt(np.mean(e[sites==s]**2)) for s in np.unique(sites)])),
        'prediction_mean':float(w@p),'target_mean':float(w@y),
        'negative_prediction_mean':float(np.mean(p[y==0])) if (y==0).any() else None,
        'high_y_gt02_rmse':float(np.sqrt(np.mean(e[y>.2]**2))) if (y>.2).any() else None}

def main():
    import lightgbm as lgb
    OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    sys.addaudithook(guard)
    protocol={'seed':SEED,'modes':MODES,'per_regime_train_rows':12000,'same_regime_ES_rows':2000,
        'maximum_trees':1000,'early_stopping':50,'learning_rate':.035,'leaves':31,'min_leaf_rows':50,
        'threads':2,'Test_used':False,'new_neural_training':False,'new_KFold':False,
        'no_Bayes_floor_estimate':True,'model':'direct scalar LGB regressor; information ablation not NN architecture comparison',
        'primary_new_evaluation':'fresh-fit, old local_fit excluded; fixed panels retrospective',
        'source_sha256':{str((DATA/f'{r}_{g}.npz').relative_to(ROOT)):sha(DATA/f'{r}_{g}.npz')
            for r in ('train','earlystop') for g in ('D0','H0')},
        'script_sha256':sha(Path(__file__))}
    if (OUT/'PROTOCOL.json').exists():
        if json.loads((OUT/'PROTOCOL.json').read_text())!=json.loads(json.dumps(protocol)):
            raise ValueError('Protocol changed')
    else:jsave(OUT/'PROTOCOL.json',protocol)
    math_result=math_fixtures()
    panels={f'fresh_{g}':load(FRESH/f'ensemble_fit_fresh_{g}.npz') for g in ('D0','H0')}
    for name in PANELS:panels[name]=panel_data(name)
    weights={}
    for name in PANELS:
        z=load(ROOT/f'models/r16_v1_fourway_r2/predictions/{name}.npz')
        if not np.array_equal(z['sample_id'],panels[name]['sample_id']):raise ValueError('Panel ordering differs')
        weights[name]=z.get('weights')
    prediction={};rows=[];jobs=[];started=time.perf_counter()
    for mode in MODES:
        ev={name:transform(d['db'],mode) for name,d in panels.items()}
        for regime in ('D0','H0'):
            tr=load(DATA/f'train_{regime}.npz');es=load(DATA/f'earlystop_{regime}.npz')
            if set(tr['base_window_index']) & set(es['base_window_index']):raise ValueError('Source overlap')
            x=transform(tr['db'],mode);v=transform(es['db'],mode)
            folder=MODELS/mode/regime;folder.mkdir(parents=True,exist_ok=True)
            began=time.perf_counter()
            if (folder/'COMPLETED.json').exists():
                rec=json.loads((folder/'COMPLETED.json').read_text())
                if sha(folder/'regression.txt')!=rec['model_sha256']:raise ValueError('Existing model changed')
                booster=lgb.Booster(model_file=str(folder/'regression.txt'))
            else:
                progress(stage='fit',mode=mode,regime=regime,features=x.shape[1],elapsed=time.perf_counter()-started)
                model=lgb.LGBMRegressor(n_estimators=1000,learning_rate=.035,num_leaves=31,min_child_samples=50,
                    max_bin=127,n_jobs=2,random_state=SEED,verbosity=-1,colsample_bytree=.85,
                    subsample=.85,subsample_freq=1,deterministic=True,force_col_wise=True,metric='None')
                hist={}
                model.fit(x,tr['share'],eval_set=[(v,es['share'])],eval_metric='rmse',
                    callbacks=[lgb.early_stopping(50,verbose=False),lgb.record_evaluation(hist)])
                booster=model.booster_;booster.save_model(str(folder/'regression.txt'))
                jsave(folder/'history.json',hist)
                rec={'mode':mode,'regime':regime,'features':x.shape[1],'best_iteration':int(model.best_iteration_),
                    'seconds':time.perf_counter()-began,'model_sha256':sha(folder/'regression.txt'),
                    'protocol_sha256':sha(OUT/'PROTOCOL.json')}
                jsave(folder/'COMPLETED.json',rec);del model
            jobs.append(rec);pp={}
            for name,d in panels.items():
                p=np.clip(booster.predict(ev[name],num_threads=2),0,1);pp[name]=p
                rows.append({'mode':mode,'trained_regime':regime,'panel':name,**metric(d,p,weights.get(name))})
            for name,d,xx in (('train',tr,x),('earlystop',es,v)):
                p=booster.predict(xx,num_threads=2)
                rows.append({'mode':mode,'trained_regime':regime,'panel':name,**metric(d,p)})
            prediction[(mode,regime)]=pp
            np.savez_compressed(folder/'evaluation_predictions.npz',**pp)
            pd.DataFrame(rows).to_csv(OUT/'metrics_progress.csv',index=False)
            progress(stage='complete_model',**rec,elapsed=time.perf_counter()-started)
            del x,v,tr,es,booster;gc.collect()
        for name,d in panels.items():
            p=.5*(prediction[(mode,'D0')][name]+prediction[(mode,'H0')][name])
            rows.append({'mode':mode,'trained_regime':'D0_H0_50_50','panel':name,**metric(d,p,weights.get(name))})
        del ev;gc.collect()
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(jobs).to_csv(OUT/'training_jobs.csv',index=False)
    frame=pd.DataFrame(rows)
    text='# R18 상대 입력과 절대 dB 정보 대조\n\n'
    text+='같은 12,000행 TRAIN/2,000행 조기종료 자료에서 소형 직접 회귀기 6개를 학습했다. 17채널 신경망의 Bayes 최소 오차를 측정한 것이 아니다.\n\n'
    text+='## 정보 복원 검산\n\n```json\n'+json.dumps(math_result,ensure_ascii=False,indent=2)+'\n```\n\n'
    text+='`10*Ch0+20*Ch4=db-window_median`이고 `20log10(sqrt(w))`도 공통 상수 이외의 dB를 복원한다. 공통 배율은 정답 비율에서 소거되므로 절대 공통 레벨 제거만으로 0.07의 이론적 바닥이 생긴다는 근거는 없다. 단, 알려진 장비/배경 사전분포와 연결되는 절대 레벨은 통계적으로 유용할 수 있으며 교차 장비에서는 가짜 단서가 될 수도 있다.\n\n'
    text+='## D0/H0 동일 입력 50:50 회귀 결과\n\n'+frame[frame.trained_regime.eq('D0_H0_50_50')].to_markdown(index=False,floatfmt='.6f')+'\n\n'
    text+='원시 dB가 저절로 B를 알려주지는 않는다. 동일한 E에도 여러 B+A 분해가 가능하다. 모델 간 수치는 유한 데이터/트리 함수공간의 결과이며 정보이론적 하한도 대회 Public 예측도 아니다. fixed 패널은 반복 검토된 진단 자료이며 새 fresh-fit과 별도로 읽는다.\n'
    (OUT/'R18_INFORMATION_AND_REGRESSION.md').write_text(text)
    jsave(OUT/'COMPLETED.json',{'status':'completed','small_tree_fits':6,'seconds':time.perf_counter()-started,
        'new_neural_steps':0,'Test_used':False,'report_sha256':sha(OUT/'R18_INFORMATION_AND_REGRESSION.md'),
        'metrics_sha256':sha(OUT/'metrics.csv')})

if __name__=='__main__':main()
