"""R15 descriptive score/region-bump/activity audit, with no model execution.

All thresholds are declared before data calculation. Public query labels and
actual site labels are unknown. SF means the historical kNN estimate only.
Original SA comparisons use Fold0 TRAIN background indices, never heldout
sources, and are explicitly separated before/after the existing minute QC.
"""
from pathlib import Path
import argparse,csv,hashlib,json,time,warnings
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT='reports/r15_score_duration_r3'
QUANTILES=(.05,.10,.15,.20,.25,.30,.50,.75,.90,.95)
REGIONS={'HF3162_7943':list(range(20,25)), 'LF31_200':list(range(9)), 'LF_historical31_501':list(range(13))}
ACTIVE_THRESHOLDS=(3.,6.,10.)
TEST_SHA='3ef819192c05c4308b66564e6307e6f4f26fe806afe96ec93e2d15f25a3474d4'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def qs(v):
    a=np.asarray(v,float);a=a[np.isfinite(a)]
    return {'n_valid':len(a),**{f'p{int(q*100):02d}':float(np.quantile(a,q)) if len(a) else None for q in QUANTILES}}
def describe(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'RMS':float(np.sqrt(np.mean(a*a))) if len(a) else None,**qs(a)}
def temporal(level):
    valid=np.isfinite(level);n=valid.sum(1)
    med=np.nanmedian(level,axis=1);rel=level-med[:,None]
    d={'nobs':n,'bump':np.nanmax(rel,axis=1)}
    for k in ACTIVE_THRESHOLDS:
        active=valid&(rel>k);count=active.sum(1)
        d[f'active_gt{k:g}_count']=count
        d[f'active_gt{k:g}_fraction']=count/n
        # Missing minutes break a contiguous run; no gap bridging.
        current=np.zeros(len(level),int);longest=current.copy()
        for t in range(15):current=np.where(active[:,t],current+1,0);longest=np.maximum(longest,current)
        d[f'active_gt{k:g}_max_contiguous_run']=longest
    return d
def regions(db,chunk=2048):
    result={name:{} for name in REGIONS};levels={name:np.empty((len(db),15),float) for name in REGIONS}
    for start in range(0,len(db),chunk):
        x=np.asarray(db[start:start+chunk],float);obs=np.isfinite(x)
        if np.isinf(x).any() or not obs.any((1,2)).all():raise ValueError('Invalid source windows')
        peak=np.nanmax(x,(1,2),keepdims=True)
        energy=np.where(obs,10**((x-peak)/10.),0)
        for name,idx in REGIONS.items():
            valid=obs[:,:,idx].all(2)
            # Historical EDA uses complete25 minutes for the LF13/HF5 curves.
            if name in ('HF3162_7943','LF_historical31_501'):valid&=obs.all(2)
            power=energy[:,:,idx].sum(2)
            level=np.where(valid,10*np.log10(np.maximum(power,np.finfo(float).tiny))+peak[:,0,0,None],np.nan)
            levels[name][start:start+len(x)]=level
    for name in REGIONS:result[name]=temporal(levels[name])
    for lf in ('LF31_200','LF_historical31_501'):
        a=levels[lf];b=levels['HF3162_7943'];ok=np.isfinite(a)&np.isfinite(b);n=ok.sum(1)
        am=np.divide(np.where(ok,a,0).sum(1),n);bm=np.divide(np.where(ok,b,0).sum(1),n)
        aa=np.where(ok,a-am[:,None],0);bb=np.where(ok,b-bm[:,None],0);den=np.sqrt((aa*aa).sum(1)*(bb*bb).sum(1))
        result[lf]['corr_with_HF']=np.divide((aa*bb).sum(1),den,out=np.full(len(a),np.nan),where=den>0)
    return result
def corr(a,b,rank=False):
    a=np.asarray(a,float);b=np.asarray(b,float);ok=np.isfinite(a)&np.isfinite(b);a=a[ok];b=b[ok]
    if rank:a=pd.Series(a).rank(method='average').to_numpy();b=pd.Series(b).rank(method='average').to_numpy()
    return (float(np.corrcoef(a,b)[0,1]) if len(a)>=3 and a.std()>0 and b.std()>0 else None),int(len(a))
def rankprob(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);v=np.r_[a,b];rank=pd.Series(v).rank(method='average').to_numpy()
    p=float((rank[:len(a)].sum()-len(a)*(len(a)+1)/2)/(len(a)*len(b)))
    grid=np.unique(v);ks=float(np.max(abs(np.searchsorted(np.sort(a),grid,side='right')/len(a)-np.searchsorted(np.sort(b),grid,side='right')/len(b))))
    return p,ks

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=OUT);a=p.parse_args();out=(ROOT/a.output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'reports'):raise FileExistsError('Fresh reports directory only')
    source=read(ROOT/'reports/r15_band_comparison_r1/R14_TRAIN_source.json');r14=source['path'];r14meta=source['metadata_path']
    if not (ROOT/r14).is_file():raise FileNotFoundError('Pinned R14epoch38 source unavailable; do not silently substitute another epoch')
    bc=read(ROOT/'data/versions/background_cv_r9/manifest.json');bank=bc['background_bank']['path'];pool='data/synthetic/group_cv_physics_r13_variants_v2/fold_00/TRAIN_pool.parquet'
    paths=['ops/r15_score_duration_audit.py','reports/r15_descriptive_audit_r1/joined1500.csv',
           'reports/r15_band_comparison_r1/R14_TRAIN_source.json','reports/r15_band_comparison_r1/R9_TRAIN_source.json',
           'data/raw/Test.csv',r14,r14meta,'models/r9_mac_trees_r2/fold_00/tree_epoch1_train.npz',
           'models/r9_mac_trees_r2/fold_00/tree_epoch1_manifest.json',bank,
           'data/versions/background_cv_r9/manifest.json',pool,'pipeline/source_split_r6.py']
    pins={n:sha(ROOT/n) for n in paths}
    if pins['data/raw/Test.csv']!=TEST_SHA or pins[bank]!=bc['background_bank']['sha256'] or pins[r14]!=source['sha256'] or pins[r14meta]!=source['metadata_sha256']:raise ValueError('Pinned source bytes differ')
    oldr9=read(ROOT/'reports/r15_band_comparison_r1/R9_TRAIN_source.json')
    if pins[oldr9['path']]!=oldr9['sha256']:raise ValueError('Original R9TRAIN bytes differ')
    out.mkdir(parents=True);started=time.time()
    plan={'version':'r15_score_duration_descriptive_v3_safeUnicode','created_unix':started,'input_sha256':pins,
          'regions_zero_based':REGIONS,'region_level':'10log10(sum of measured linear band powers), complete regional frequency support; historical LF13/HF5 require complete25 minute',
          'bump':'max_observed_t(region_level)-median_observed_t(region_level)',
          'quantiles':list(QUANTILES),'active_minutes':'strict level>within-window regional median+3,+6,+10dB, no gap imputation; count plus count/Nobs and longest observed consecutive run',
          'fixed_query_groups':{'HF_high':'HFbump>15dB','LF_near_zero':'LF<=3dB','joint_rise':'LF>6dB AND LF-HF Pearson>.5'},
          'fraction_of_HFgt15_intersection':'n(condition AND HF>15)/n(HF>15), even if row n describes an unconditional LF condition',
          'SF':'historical probe_site==07 St Francis (notactualsiteGT); restnonSF',
          'rankprob':'P(score_SF>score_nonSF)+0.5tie, both directions; NOT label-AUC. KS empirical CDFdistance only, no iid p-value claims.',
          'SA_scope':'Only78869 unique Fold0TRAIN originalB window indices; before-existing-minute-QC and after-QC separately, NOT whole1715779 released records.',
          'observation_strata':['all','Nobs5','Nobs13plus','Nobs15'],
          'model_fit_inference_calibration_or_submission':False,'hidden_Test_or_actual_site_labels_available':False}
    write(out/'plan.json',plan)
    # Meaningful threshold/rank/missing unit cases, no acoustic/model fitting.
    fixture=np.asarray([[0]*11+[3,6,10,12]],float);ft=temporal(fixture)
    if [int(ft[f'active_gt{k:g}_count'][0]) for k in ACTIVE_THRESHOLDS]!=[3,2,1]:raise AssertionError('Strict threshold equality fixture')
    fixture[0,1]=np.nan;ft=temporal(fixture)
    if ft['nobs'][0]!=14 or ft['active_gt3_fraction'][0]!=3/14:raise AssertionError('Missing normalization fixture')
    pr,ks=rankprob([1,2],[2,3])
    if pr!=.125 or ks!=.5:raise AssertionError('Tie-aware rank/KS fixture')
    write(out/'NUMERIC_FIXTURE.json',{'status':'passed','strict_active_thresholds':True,'missing_denominator14':True,'tie_rankprob':pr,'KS':ks})
    joined=pd.read_csv(ROOT/'reports/r15_descriptive_audit_r1/joined1500.csv',dtype={'ID':str});qrows=[];pop=[];testfeatures=None
    def consume(name,db,ids):
        nonlocal testfeatures
        derived=regions(db);nobs=np.isfinite(db).all(2).sum(1)
        for n,count in zip(*np.unique(nobs,return_counts=True)):pop.append({'dataset':name,'Nobs':int(n),'rows':int(count),'total':len(db)})
        payload={'source_id':np.asarray(ids,dtype=str),'Nobs':nobs}
        for reg,ds in derived.items():
            for metric,v in ds.items():payload[reg+'__'+metric]=v
        np.savez_compressed(out/(name+'_derived.npz'),**payload)
        for stratum,mask in [('all',np.ones(len(db),bool)),('Nobs5',nobs==5),('Nobs13plus',nobs>=13),('Nobs15',nobs==15)]:
            for reg,ds in derived.items():
                for metric,v in ds.items():
                    if metric=='nobs':continue
                    qrows.append({'dataset':name,'stratum':stratum,'region':reg,'metric':metric,'population_rows':len(db),'stratum_rows':int(mask.sum()),**qs(v[mask])})
        if name=='Test':testfeatures=derived
    # Synthetic TRAIN arrays retain exact source ordering; labels not loaded.
    for name,path in [('R14_TRAIN_epoch38',r14),('R9_TRAIN_epoch1',oldr9['path'])]:
        with np.load(ROOT/path,allow_pickle=False) as z:db=z['db'];ids=z['sample_id'].astype(str)
        if db.shape!=(78869,15,25) or len(set(ids))!=78869:raise ValueError('Expected complete78869 TRAIN source')
        consume(name,db,ids);del db
    # Public released background indices only; preserve and then apply original QC.
    meta=pd.read_parquet(ROOT/pool,columns=['base_window_index','window_id','boundary_mask_bits','qc_observed_minutes'])
    if len(meta)!=78869 or meta.base_window_index.nunique()!=78869:raise ValueError('Unique Fold0TRAIN source pool')
    mmap=np.load(ROOT/bank,mmap_mode='r',allow_pickle=False);raw=np.asarray(mmap[meta.base_window_index.to_numpy(int)])
    consume('SA_Fold0TRAIN_original_B_preQC',raw,meta.window_id.astype(str).to_numpy())
    from pipeline.source_split_r6 import apply_minute_qc
    qc=apply_minute_qc(raw,meta.boundary_mask_bits.to_numpy(np.int64));del raw
    if not np.array_equal(np.isfinite(qc).all(2).sum(1),meta.qc_observed_minutes):raise ValueError('Original minuteQC count differs')
    consume('SA_Fold0TRAIN_original_B_QC',qc,meta.window_id.astype(str).to_numpy());del qc,mmap
    # Query parsed only as observations; labels never exist in this input.
    with (ROOT/'data/raw/Test.csv').open(newline='',encoding='utf-8-sig') as f:
        rd=csv.reader(f);header=next(rd);rr=list(rd)
    tags=('31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5','251.2','316.2','398.1','501.2','631','794.3','1000','1258.9','1584.9','1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3')
    if header!=['ID',*[f'm{t:02d}_TOL_{hz}' for t in range(1,16) for hz in tags]]:raise ValueError('Original schema differs')
    ids=[row[0] for row in rr]
    if ids!=joined.ID.tolist():raise ValueError('Query/joined order mismatch')
    db=np.asarray([[float(v) if v.strip() else np.nan for v in row[1:]] for row in rr],float).reshape(-1,15,25)
    consume('Test',db,ids);del db
    group_rows=[];corr_rows=[];sf_rows=[];rank_rows=[];predcols=[n for n in joined if n.startswith(('R5_','R9_','R14.','R14_')) and pd.api.types.is_numeric_dtype(joined[n])]
    hf=testfeatures['HF3162_7943']['bump'];high=hf>15
    joined['HF3162_7943_bump_db']=hf
    for reg in ('LF31_200','LF_historical31_501'):
        lf=testfeatures[reg]['bump'];co=testfeatures[reg]['corr_with_HF'];joined[reg+'_bump_db']=lf;joined[reg+'_corr_with_HF']=co
        conditions={'HF_gt15':high,'HF_gt15_AND_LF_le3':high&(lf<=3),'HF_gt15_AND_LF_gt6_corr_gt05':high&(lf>6)&(co>.5),'LF_le3_all':lf<=3}
        for name,mask in conditions.items():
            group_rows.append({'LF_definition':reg,'group':name,'n':int(mask.sum()),'fraction_of_all1500':float(mask.mean()),'fraction_of_HFgt15_intersection':float(mask[high].mean()),'HFgt15_denominator':int(high.sum())})
    for reg,ds in testfeatures.items():
        for metric,v in ds.items():
            if metric=='nobs':continue
            label=reg+'__'+metric;joined[label]=v
            for series in predcols:
                pe,n=corr(v,joined[series]);sp,_=corr(v,joined[series],True);corr_rows.append({'feature':label,'prediction':series,'n':n,'Pearson':pe,'Spearman':sp})
    sf=joined.probe_site.eq('07 St Francis')
    if int(sf.sum())!=487 or int((~sf).sum())!=1013:raise ValueError('Historical estimated-site counts changed')
    for series in predcols:
        for name,mask in [('estimated_SF487',sf),('estimated_nonSF1013',~sf)]:sf_rows.append({'estimated_group_not_GT':name,'prediction':series,**describe(joined.loc[mask,series])})
        pr,ks=rankprob(joined.loc[sf,series],joined.loc[~sf,series]);rank_rows.append({'prediction':series,'n_SF':487,'n_nonSF':1013,'P_SF_greater_plus_half_ties':pr,'P_nonSF_greater_plus_half_ties':1-pr,'empirical_KS_D':ks,'label_AUC_claimed':False})
    for name,rows in [('dataset_bump_duration_quantiles.csv',qrows),('population_Nobs_counts.csv',pop),('Test_LF_HF_groups.csv',group_rows),('feature_prediction_correlations.csv',corr_rows),('SF_vs_nonSF_prediction_quantiles.csv',sf_rows),('SF_vs_nonSF_rankprob_KS.csv',rank_rows)]:pd.DataFrame(rows).to_csv(out/name,index=False)
    joined.to_csv(out/'Test_joined_duration1500.csv',index=False)
    for name,h in pins.items():
        if sha(ROOT/name)!=h:raise ValueError('Input changed during audit:'+name)
    result={'status':'passed_descriptive_only','elapsed_seconds':time.time()-started,'input_sha256':pins,'script_sha256':sha(__file__),'plan_sha256':sha(out/'plan.json'),
            'datasets':['Test','R9_TRAIN_epoch1','R14_TRAIN_epoch38','SA_Fold0TRAIN_original_B_preQC','SA_Fold0TRAIN_original_B_QC'],
            'TRAIN_and_SA_rows_each':78869,'Test_rows':1500,'group_counts':group_rows,'SF_estimate_count':487,'nonSF_estimate_count':1013,
            'R9_all5_score_vs_own_share':{'n':len(joined),'Pearson':corr(joined['R9_all5.share'],joined['R9_all5.score'])[0],'Spearman':corr(joined['R9_all5.share'],joined['R9_all5.score'],True)[0],'label_AUC_claimed':False},
            'all_input_hashes_unchanged':True,'model_fit_inference_calibration_or_submission':False,'known_Test_or_actual_site_labels_available':False,
            'original_SA_scope_is_only_Fold0TRAIN_not_all_public_rows':True,
            'limits':['Observed activity counts are not known vessel passage duration; missing minutes break runs, and counts/fractions are both reported.',
                      'Regional broadband curves sum linear powers before dB. Historical LF13 differs from requested LF9. Per-band and regional quantities must not be pooled.',
                      'SF/nonSF is an old domain-probe estimate, not a label. MannWhitney rank probability and KS compare score distributions, not positive/negative AUC or correctness.',
                      'TRAIN windows are reused backgrounds/refs and not independent rows. No iid significance p-values or causal attribution is claimed.',
                      'OriginalSA comparison is TRAIN-only78869 window sources before/after already-existing QC, not all1715779 released records. No heldout background arrays or report labels read.',
                      'Test features remain unlabeled; no target scale inferred from TRAIN conditional labels, public scores or the input statistics.'],
            'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    write(out/'RESULTS.json',result);print(json.dumps({'status':result['status'],'output':str(out),'seconds':result['elapsed_seconds'],'RESULTS_sha256':sha(out/'RESULTS.json'),'groups':group_rows}))

if __name__=='__main__':main()
