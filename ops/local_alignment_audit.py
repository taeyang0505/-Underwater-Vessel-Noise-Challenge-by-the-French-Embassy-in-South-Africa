"""Safe local constants and TRAIN-vs-INNER source-domain diagnostics.

Challenge Test inputs are forbidden. This script has no configurable acoustic
input path and never opens Test, old OUTER or locked diagnostic arrays. AV is
fixed 5-fold grouped cross-validation on new R6 TRAIN/INNER only. It neither
tunes a generator nor estimates leaderboard performance.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import warnings
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'reports/R6_LOCAL_ALIGNMENT_AUDIT_PLAN.json'
REPORT_JSON=ROOT/'reports/R6_LOCAL_ALIGNMENT_AUDIT.json'
REPORT_MD=ROOT/'reports/R6_LOCAL_ALIGNMENT_AUDIT.md'
R5TRAIN=ROOT/'data/synthetic/group_cv_passage_r4_v1/fold_00/train.npz'
R5INNER=ROOT/'data/synthetic/group_cv_passage_r4_v1/fold_00/validation.npz'
R6DATA=ROOT/'data/synthetic/r6_cpu_guardrails_v3_schemafix'
R6SOURCE=ROOT/'data/versions/source_split_r6_guardrails_v2'
FREEZE=ROOT/'models/r6_cpu_guardrails_v2/all_checkpoints_frozen_before_heldout.json'
VERSION='safe_local_alignment_audit_v1'
SEED=20261026
PHYSICAL_NAMES=('lf_31p6_to_501p2_bump_db','high_3162_to_7943_bump_db',
                'low_high_broadband_db_pearson','all25_broadband_db_population_std')
DEFINITION={
 'LF_band_indices':list(range(13)), 'high_band_indices':list(range(20,25)),
 'regional_power':'sum linear band powers, expressed in dB using stable logsumexp; complete regional minutes only',
 'bump':'maximum regional minute dB minus median regional minute dB, >=3 observed minutes',
 'correlation':'Pearson between LF and high regional minute dB, >=3 paired minutes and nonconstant series',
 'broadband_std':'population standard deviation of all25 regional minute dB, >=3 complete minutes',
 'AV_features':'4 physical values +4 defined flags +375 window-median-relative db/20 +375 observed mask +25 bandmedian-windowmedian/20 +25 temporal band std/20 +25 band q90-q10 span/20 +25 observed fraction',
 'undefined_AV_feature_fill':0.,'number_of_features':858,
 'raw_absolute_db_site_target_ref_tags_in_features':False,'source_id_used_only_for_CV_groups':True,
 'KS_scope':'R6 TRAIN versus R6 INNER only; physical values with defined masks',
 'domain_labels':{'TRAIN':0,'INNER':1},
 'classifier':{'family':'LightGBM binary classifier','n_estimators':200,'num_leaves':15,'learning_rate':.05,
               'n_jobs':2,'random_state':SEED,'min_child_samples':20,'subsample':1.,'colsample_bytree':1.,
               'early_stopping':False,'hyperparameter_search':False},
 'CV':{'folds':5,'kind':'StratifiedGroupKFold','shuffle':True,'random_state':SEED,'group':'original primary background base_window_index'},
 'constants':[0.,.1,'same_version_TRAIN_mean','oracle_same_INNER_mean_diagnostic_only'],
 'no_Test_or_Benchmark_moment_fitting':True,
 'limitations':['Source-domain AV is not a Test-domain AV or a leaderboard-gap measurement.',
  'Source-separated groups can still share sensors/recorder footprints. Reused TRAIN windows are grouped together.',
  'AUC near .5 does not establish distribution equality; .85 is not an automatic fix threshold.',
  'R5 INNER labels come from the old R4 synthetic generator, not natural observational vessel labels.',
  'Oracle INNER mean is the best constant fitted to that INNER truth; it is a descriptive lower bound, not a deployable fit.',
  'The unknown public Benchmark score is not a target-moment equation; no .24-.27 interval is fitted.']}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def deny_test_path(path):
    p=Path(path)
    if any(part.lower()=='test' or part.lower().startswith('test.') or part.lower().startswith('test_') for part in p.parts):
        raise ValueError('Challenge Test input is forbidden; no Test values, moments or domain labels may be read')
    return p

def save_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def inputs():
    return {'R5_TRAIN_share_source':R5TRAIN,'R5_INNER_share_source':R5INNER,
        'R6_TRAIN':R6DATA/'train_r6.npz','R6_INNER':R6DATA/'inner_r6.npz',
        'R6_dataset_manifest':R6DATA/'manifest.json','R6_source_manifest':R6SOURCE/'manifest.json',
        'R6_background_metadata':R6SOURCE/'background_manifest.parquet','all_main_checkpoints_frozen':FREEZE}

def create_plan():
    paths=inputs()
    for p in paths.values():deny_test_path(p)
    gate=json.loads(FREEZE.read_text())
    if gate.get('status')!='frozen' or gate.get('all_requested_arms_completed') is not True or len(gate.get('model_summaries',{}))!=7:
        raise ValueError('All seven main checkpoints must already be frozen')
    p={'version':VERSION,'status':'predeclared_before_alignment_model_fit','definitions':DEFINITION,
       'input_paths':{k:str(v.relative_to(ROOT)) for k,v in paths.items()},
       'input_sha256':{k:sha(v) for k,v in paths.items()},'script_sha256':sha(__file__),
       'execution':'one fixed grouped AV run; no Test, model-output or locked-array access',
       'data_fields':{'R5_TRAIN':['share'],'R5_INNER':['share'],
                      'R6_TRAIN_INNER':['db','share','base_window_index']}}
    p['plan_sha256']=canonical(p);save_new(PLAN,p);return p

def verify_plan():
    p=json.loads(PLAN.read_text())
    if p.get('version')!=VERSION or p.get('definitions')!=DEFINITION or p.get('script_sha256')!=sha(__file__):raise ValueError('Fixed audit plan/code changed')
    if canonical({k:v for k,v in p.items() if k!='plan_sha256'})!=p['plan_sha256']:raise ValueError('Plan payload changed')
    for name,path in inputs().items():
        deny_test_path(path)
        if str(path.relative_to(ROOT))!=p['input_paths'][name] or sha(path)!=p['input_sha256'][name]:raise ValueError('Bound audit input changed: '+name)
    return p

def regional_db(x,indices):
    values=x[:,:,indices];ok=np.isfinite(values).all(2)
    natural=np.where(np.isfinite(values),values*np.log(10.)/10.,-np.inf)
    summed=np.logaddexp.reduce(natural,axis=2)*10./np.log(10.)
    return np.where(ok,summed,np.nan)

def physical(db):
    x=np.asarray(db,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(15,25) or not len(x) or np.isinf(x).any():raise ValueError('db[N,15,25] finite/NaN required')
    low=regional_db(x,np.arange(13));high=regional_db(x,np.arange(20,25));broad=regional_db(x,np.arange(25))
    out=np.full((len(x),4),np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for i,v in [(0,low),(1,high)]:
            count=np.isfinite(v).sum(1);value=np.nanmax(v,1)-np.nanmedian(v,1);out[:,i]=np.where(count>=3,value,np.nan)
        out[:,3]=np.where(np.isfinite(broad).sum(1)>=3,np.nanstd(broad,axis=1,ddof=0),np.nan)
    paired=np.isfinite(low)&np.isfinite(high);count=paired.sum(1)
    ml=np.divide(np.where(paired,low,0).sum(1),count,out=np.zeros(len(x)),where=count>0)
    mh=np.divide(np.where(paired,high,0).sum(1),count,out=np.zeros(len(x)),where=count>0)
    a=np.where(paired,low-ml[:,None],0);b=np.where(paired,high-mh[:,None],0)
    denominator=np.sqrt((a*a).sum(1)*(b*b).sum(1));valid=(count>=3)&(denominator>1e-12)
    out[:,2]=np.divide((a*b).sum(1),denominator,out=np.full(len(x),np.nan),where=valid)
    out[:,2]=np.clip(out[:,2],-1,1)
    return out

def av_features(db):
    x=np.asarray(db,dtype=np.float64);phys=physical(x);observed=np.isfinite(x)
    if not observed.any((1,2)).all():raise ValueError('Wholly unobserved query is unsupported')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        med=np.nanmedian(x,(1,2),keepdims=True);bm=np.nanmedian(x,axis=1)
        std=np.nanstd(x,axis=1,ddof=0);span=np.nanquantile(x,.9,axis=1)-np.nanquantile(x,.1,axis=1)
    rel=np.where(observed,(x-med)/20.,0)
    stats=np.concatenate([(bm-med[:,0])/20.,std/20.,span/20.,observed.mean(1)],1)
    out=np.concatenate([np.nan_to_num(phys,nan=0.),np.isfinite(phys).astype(float),rel.reshape(len(x),-1),observed.reshape(len(x),-1).astype(float),np.nan_to_num(stats,nan=0.)],1)
    if out.shape!=(len(x),858) or not np.isfinite(out).all():raise RuntimeError('Fixed feature schema failed')
    return out.astype(np.float32),phys

def constants(train_y,inner_y):
    train_y=np.asarray(train_y,dtype=float);y=np.asarray(inner_y,dtype=float)
    if train_y.ndim!=1 or y.ndim!=1 or not len(y) or not len(train_y) or not np.isfinite(train_y).all() or not np.isfinite(y).all() or (y<0).any() or (y>1).any():raise ValueError('Finite scalar share labels in [0,1] required')
    mean=float(y.mean());variance=float(np.var(y,ddof=0));rows={}
    for name,c in [('constant0',0.),('constant0_1',.1),('TRAIN_mean',float(train_y.mean())),('oracle_INNER_mean_diagnostic_only',mean)]:
        mse=float(np.mean((y-c)**2));identity=variance+(c-mean)**2
        if abs(mse-identity)>1e-12:raise RuntimeError('Constant MSE identity failed')
        rows[name]={'value':c,'RMSE':float(np.sqrt(mse)),'MSE':mse,'MSE_identity_error':abs(mse-identity),'fitted_on_INNER':name.startswith('oracle')}
    return {'TRAIN_examples':len(train_y),'INNER_examples':len(y),'INNER_mean':mean,'INNER_population_variance':variance,'constants':rows,
        'identity':'MSE(c)=Var_population(INNER_share)+(c-mean(INNER_share))^2',
        'label_status':'known synthetic added-component share; not natural observational vessel ground truth'}

def fixture():
    rng=np.random.default_rng(1);x=rng.normal(70,2,(12,15,25));x[0]=65;x[1,:13]=np.nan;x[2,0]=np.nan
    features,p=av_features(x);shifted,_=av_features(x+40.)
    assert features.shape==(12,858) and np.isfinite(features).all() and np.allclose(features,shifted,atol=1e-5)
    assert p[0,0]==0 and p[0,1]==0 and not np.isfinite(p[0,2]) and not np.isfinite(p[1]).any()
    a,_=av_features(x[:5]);b,_=av_features(x[5:]);assert np.array_equal(features,np.r_[a,b])
    c=constants([0.,1.],[.1,.3,.5]);assert all(v['MSE_identity_error']<1e-12 for v in c['constants'].values())
    for path in ['Test.csv','data/raw/Test.csv','data/Test/input.npz']:
        try:deny_test_path(path)
        except ValueError:pass
        else:raise AssertionError('Test path accepted')
    return {'status':'PASS','synthetic_only':True,'fixed_feature_count':858,'gain_chunk_invariance':True,
            'constant_and_undefined_safe':True,'MSE_identity_verified':True,'Test_paths_refused_before_open':True}

def run():
    start=time.monotonic()
    if REPORT_JSON.exists() or REPORT_MD.exists():raise FileExistsError('Single fixed experiment; no repeated audit fit')
    plan=verify_plan();checks=fixture()
    from scipy.stats import ks_2samp
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from lightgbm import LGBMClassifier
    import pandas as pd
    # R5: only labels are decompressed, not the acoustic grids.
    with np.load(R5TRAIN,allow_pickle=False) as z:r5train=z['share']
    with np.load(R5INNER,allow_pickle=False) as z:r5inner=z['share']
    r6=[]
    for name in ['train_r6.npz','inner_r6.npz']:
        with np.load(R6DATA/name,allow_pickle=False) as z:r6.append({k:z[k] for k in ['db','share','base_window_index']})
    train,inner=r6;bg=pd.read_parquet(R6SOURCE/'background_manifest.parquet',columns=['base_window_index','split'])
    for role,d in [('train',train),('inner',inner)]:
        if not set(d['base_window_index'])<=set(bg.loc[bg.split.eq(role),'base_window_index']):raise ValueError('AV source role mismatch')
    if set(train['base_window_index'])&set(inner['base_window_index']):raise ValueError('TRAIN/INNER primary source overlap')
    local_constants={'R5_original_fold0_INNER':constants(r5train,r5inner),'R6_guardrails_INNER':constants(train['share'],inner['share'])}
    tx,tp=av_features(train['db']);vx,vp=av_features(inner['db']);ks={}
    for i,name in enumerate(PHYSICAL_NAMES):
        a=tp[:,i];b=vp[:,i];aa=a[np.isfinite(a)];bb=b[np.isfinite(b)]
        stat=None if not len(aa) or not len(bb) else ks_2samp(aa,bb,alternative='two-sided',method='auto')
        ks[name]={'TRAIN_total':len(a),'TRAIN_finite':len(aa),'TRAIN_undefined':int((~np.isfinite(a)).sum()),
                  'INNER_total':len(b),'INNER_finite':len(bb),'INNER_undefined':int((~np.isfinite(b)).sum()),
                  'TRAIN_median':None if not len(aa) else float(np.median(aa)),'INNER_median':None if not len(bb) else float(np.median(bb)),
                  'KS_statistic':None if stat is None else float(stat.statistic),'KS_pvalue_descriptive':None if stat is None else float(stat.pvalue)}
    x=np.r_[tx,vx];y=np.r_[np.zeros(len(tx),int),np.ones(len(vx),int)]
    groups=np.r_[train['base_window_index'],inner['base_window_index']]
    if any(len(np.unique(groups[y==label]))<5 for label in [0,1]):raise ValueError('Fivefold source-group/class support absent; no fallback CV')
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED);p=np.full(len(x),np.nan);folds=[]
    for fold,(fit,held) in enumerate(splitter.split(x,y,groups)):
        if len(np.unique(y[fit]))!=2 or len(np.unique(y[held]))!=2 or set(groups[fit])&set(groups[held]):raise ValueError('Grouped CV class/isolation gate failed')
        clf=LGBMClassifier(n_estimators=200,num_leaves=15,learning_rate=.05,n_jobs=2,random_state=SEED,min_child_samples=20,
                          subsample=1.,colsample_bytree=1.,objective='binary',verbosity=-1,deterministic=True,force_col_wise=True)
        clf.fit(x[fit],y[fit]);pred=clf.predict_proba(x[held])[:,1]
        if not np.isfinite(pred).all():raise FloatingPointError('Nonfinite fixed AV probabilities')
        p[held]=pred;folds.append({'fold':fold,'fit_rows':len(fit),'held_rows':len(held),'fit_unique_groups':len(np.unique(groups[fit])),
            'held_unique_groups':len(np.unique(groups[held])),'held_class_counts':{str(k):int((y[held]==k).sum()) for k in [0,1]},
            'group_overlap':0,'AUC':float(roc_auc_score(y[held],pred))})
        print(json.dumps({'AV_fold_completed':fold,'AUC':folds[-1]['AUC']}),flush=True)
    if not np.isfinite(p).all():raise RuntimeError('Incomplete AV OOF coverage')
    av={'domain':'new R6 TRAIN=0 vs INNER=1 only','OOF_AUC':float(roc_auc_score(y,p)),'folds':folds,
        'n_features':x.shape[1],'TRAIN_rows':len(tx),'INNER_rows':len(vx),'TRAIN_unique_primary_sources':len(np.unique(train['base_window_index'])),
        'INNER_unique_primary_sources':len(np.unique(inner['base_window_index'])),'one_fixed_fit_no_tuning':True,
        'OOF_probabilities_sha256':hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()}
    verify_plan()
    report={'status':'completed','version':VERSION,'plan_sha256':plan['plan_sha256'],'script_sha256':sha(__file__),'fixture':checks,
        'constants':local_constants,'R6_TRAIN_vs_INNER_physical_KS':ks,'R6_TRAIN_vs_INNER_grouped_AV':av,'definitions':DEFINITION,
        'packages':{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','lightgbm']},
        'elapsed_seconds':time.monotonic()-start,'Test_or_locked_arrays_or_model_outputs_opened':False,
        'generator_or_regression_model_changed':False,'Benchmark_moment_matching_or_LB_forecast_performed':False}
    save_new(REPORT_JSON,report)
    lines=['# Safe local alignment audit','',
      'Challenge Test inputs were refused. Constants use synthetic TRAIN/INNER share labels; AV and KS compare only new R6 TRAIN with INNER. No main regression model, generator, ensemble or submission was changed.','',
      '## Constant predictions','',
      '| Synthetic INNER | Constant 0 RMSE | Constant .1 RMSE | TRAIN mean RMSE | Oracle INNER mean RMSE |',
      '|---|---:|---:|---:|---:|']
    for name,row in local_constants.items():
        c=row['constants'];lines.append('| '+name+' | '+' | '.join(f"{c[k]['RMSE']:.6f}" for k in ['constant0','constant0_1','TRAIN_mean','oracle_INNER_mean_diagnostic_only'])+' |')
    lines+=['','MSE(c) = population variance + (c − INNER mean)² was verified. The oracle INNER mean uses its own labels and is a descriptive best-constant reference only. The unknown public Benchmark is not a target-moment constraint.','',
      '## R6 source-domain discrimination','',f"Fixed source-group fivefold AV OOF AUC: **{av['OOF_AUC']:.6f}** ({len(tx):,} TRAIN / {len(vx):,} INNER rows; {x.shape[1]} db-only features).",'',
      '| Physical feature | TRAIN finite/total | INNER finite/total | TRAIN median | INNER median | KS statistic |','|---|---:|---:|---:|---:|---:|']
    fmt=lambda v:'undefined' if v is None else f'{v:.6f}'
    for name,k in ks.items():lines.append(f"| {name} | {k['TRAIN_finite']}/{k['TRAIN_total']} | {k['INNER_finite']}/{k['INNER_total']} | {fmt(k['TRAIN_median'])} | {fmt(k['INNER_median'])} | {fmt(k['KS_statistic'])} |")
    lines+=['','All CV folds keep repeated primary background windows together. Source IDs are excluded from features. Physical features use complete regional minutes and report undefined values; AV adds explicit defined flags and a fixed zero fill.','',
      'This is a local source-footprint diagnostic, not a Test-gap measurement. AUC near .5 does not prove equality, and .85 is not an automatic correction threshold. KS p-values are descriptive because repeated source draws are dependent. No threshold, feature, model or generation parameter was selected from these results.','',
      'Exact definitions, fixed classifier settings, per-fold AUC, bindings and finite/undefined counts are in [JSON](R6_LOCAL_ALIGNMENT_AUDIT.json).']
    with REPORT_MD.open('x') as f:f.write('\n'.join(lines)+'\n')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--run',action='store_true',help='Execute once using the already-saved fixed plan')
    mode.add_argument('--self-test',action='store_true',help='Synthetic numeric/path fixtures only')
    parser.add_argument('--test-csv',nargs='?',const='provided',help='Always rejected; Test inference-only rule')
    args=parser.parse_args()
    if args.test_csv is not None:parser.error('--test-csv is forbidden; no Test input is accepted')
    if args.self_test:print(json.dumps(fixture(),indent=2));return
    if args.run:
        r=run();print(json.dumps({'status':r['status'],'report':str(REPORT_MD),'R6_AV_OOF_AUC':r['R6_TRAIN_vs_INNER_grouped_AV']['OOF_AUC']},indent=2))
    else:
        p=create_plan();print(json.dumps({'status':'plan_saved_no_fit','plan':str(PLAN),'plan_sha256':p['plan_sha256']},indent=2))

if __name__=='__main__':main()
