"""One predeclared CPU LightGBM reference-isolation probe; never production.

Metadata-selected A211/B197 is fixed before any new model fit. TRAIN and
early stopping use A only. One fresh, paired A/B comparison uses identical
complete-observation backgrounds, zero/family/SNR/EQ random streams. No
Test, old OUTER or unused R9 report arrays are read. No split search after
seeing scores and no mutation of the production R9 data, code or weights.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SEED=2026102604
SPLIT=ROOT/'reports/r10_diagnostics/reference_split_feasibility_r1/spectral_day_affinity_closest_raw_clip_half.csv'
CONFIG=ROOT/'configs/r9_full_training_v1.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def utc():return datetime.now(timezone.utc).isoformat()
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=os.fsdecode(args[0]).replace('\\','/').lower()
    if (Path(p).name in {'test.csv','sample_submission.csv','outer_validation.npz','report.npz'}
        or '/test/' in p or '/sealed/' in p):raise PermissionError('Probe forbids Test/old OUTER/report/sealed values: '+p)

def metrics(y,z,p,s,site):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y,float);p=np.asarray(p,float);z=np.asarray(z,int);s=np.asarray(s,float)
    d=p-y
    out={'n':len(y),'rmse':float(np.sqrt(np.mean(d*d))),'mae':float(np.mean(abs(d))),
         'bias':float(d.mean()),'mean_true':float(y.mean()),'mean_pred':float(p.mean()),
         'true_share_quantiles':dict(zip(['min','p10','p50','p90','max'],np.quantile(y,[0,.1,.5,.9,1]).tolist())),
         'controlled_addition_auc':float(roc_auc_score(z,s)) if len(np.unique(z))==2 else None,
         'known_positive_count':int(z.sum()),'site_slices':{}}
    for k in sorted(set(site)):
        m=site==k;e=d[m];out['site_slices'][str(k)]={'n':int(m.sum()),'rmse':float(np.sqrt(np.mean(e*e))),
            'bias':float(e.mean()),'mean_true':float(y[m].mean()),'mean_pred':float(p[m].mean())}
    return out

def summaries(frame):
    import numpy as np
    return {'clips':len(frame),'hulls':frame.hull_id.nunique(),'days':frame.recording_day.nunique(),
            'weak_class_counts':{str(k):int(v) for k,v in frame.weak_class.value_counts().items()},
            'native_minute_counts':{str(k):int(v) for k,v in frame.native_complete_minute_count.value_counts().sort_index().items()},
            'duration_seconds':{'sum':float(frame.duration_seconds.sum()),'min_median_max':np.quantile(frame.duration_seconds,[0,.5,1]).tolist()},
            'year_counts':{str(k):int(v) for k,v in frame.recording_day.str[:4].value_counts().sort_index().items()}}

def make_rows(sources,rows,refs,tag,*,hard_ids=None):
    import numpy as np
    from pipeline.dataset_r9 import make_schedule,schedule_audit
    from pipeline.sampler_r9 import synthesize
    n=len(rows);candidate=None
    if hard_ids is not None:candidate=rows.base_window_index.isin(hard_ids).to_numpy(bool)
    # tag=0 is required by the unchanged production hard-negative schedule.
    sched=make_schedule(n,1,0,SEED+tag,reference_count=len(refs),tag=0,
                        hard_negative_candidates=candidate)
    result={k:np.empty((n,15,25),np.float32) for k in ('db','cell_share')}
    result.update(share=np.empty(n,np.float32),presence=np.empty(n,np.int64),
                  base_window_index=np.empty(n,np.int64),reference_index=np.empty(n,np.int64),
                  site=np.empty(n,dtype='U64'),sample_id=np.array([f'r10_ref_{tag}_{i:05d}' for i in range(n)]))
    recipes=[]
    for i,position in enumerate(sched['primary_order']):
        row=rows.iloc[int(position)];zero=bool(sched['explicit_no_source'][i])
        ref=-1 if zero else int(refs[int(sched['reference_index'][i])])
        # Separate continuous EQ/SNR from geometry draws. With one shared RNG,
        # different source lengths can consume different integer draws and
        # accidentally change SNR; this wrapper preserves the same laws and
        # exactly pairs the continuous controls across A/B.
        class RenderRNG:
            def __init__(self):
                self.continuous=np.random.default_rng(np.random.SeedSequence([SEED,tag,i,0x5249,1]))
                self.geometry=np.random.default_rng(np.random.SeedSequence([SEED,tag,i,0x5249,2]))
            def uniform(self,*a,**k):return self.continuous.uniform(*a,**k)
            def integers(self,*a,**k):return self.geometry.integers(*a,**k)
            def choice(self,*a,**k):return self.geometry.choice(*a,**k)
        rng=RenderRNG()
        out=synthesize(sources.background(row),None,rng,
            reference_minute_power=None if zero else sources.reference_minutes[ref],
            family=sources.config['generator']['family_names'][int(sched['family_index'][i])],
            snr_range=(-10.,10.),eq_max_db=6.,no_source=zero)
        for k in ('db','cell_share','share','presence'):result[k][i]=out[k]
        result['base_window_index'][i]=int(row.base_window_index);result['reference_index'][i]=ref;result['site'][i]=str(row.site)
        recipe=out['recipe'];recipe.update(reference_index=ref,background_index=int(row.base_window_index),
            reference_clip_id=None if zero else str(sources.references.iloc[ref].clip_id),
            family_index=int(sched['family_index'][i]),hard_negative=bool(sched['hard_negative'][i]))
        recipes.append(recipe)
    if not np.isfinite(result['db']).all():raise AssertionError('Full-observation subset was not complete')
    return result,recipes,schedule_audit(sched,len(refs))

def source_separation(train,held):
    import numpy as np,pandas as pd
    from pipeline.source_split_r6 import nearest_gap
    if set(train.base_window_index)&set(held.base_window_index):raise AssertionError('BG source index overlap')
    if set(train.background_group)&set(held.background_group):raise AssertionError('BG source group overlap')
    minimum=None
    for sensor,h in held.groupby('sensor_group'):
        t=train[train.sensor_group.eq(sensor)]
        if not len(t):continue
        sec=lambda x:pd.to_datetime(x).to_numpy(dtype='datetime64[ns]').astype('int64')//10**9
        gap=np.maximum(nearest_gap(sec(h.start_export_clock),sec(t.start_export_clock))-900,0)
        v=float(gap.min());minimum=v if minimum is None else min(minimum,v)
        if v<86400:raise AssertionError('TRAIN/selection support gap below24h')
    return {'base_index_overlap':0,'background_group_overlap':0,'minimum_shared_sensor_support_gap_seconds':minimum,
            'required_minimum_seconds':86400,'source_group_and_24h_support_purge_passed':True}

def run(args):
    import numpy as np,pandas as pd
    from pipeline.dataset_r9 import R9Sources,DATA_CODE
    from pipeline.trees_ensemble_r9 import PhysicsLGBM,CODE_NAMES,array_sha
    sys.addaudithook(guard)
    out=args.output.resolve()
    if out.exists():raise FileExistsError('Fresh probe directory required; no partial reuse')
    out.mkdir(parents=True);start=time.perf_counter()
    source=ROOT/'data/versions/vtuad_full_r9/reference_manifest.parquet'
    trainmeta=ROOT/'data/versions/background_cv_r9/fold_00/train_manifest.parquet'
    selmeta=ROOT/'data/versions/background_cv_r9/fold_00/selection_manifest.parquet'
    names=sorted(set(DATA_CODE)|set(CODE_NAMES)|{str(Path(__file__).relative_to(ROOT))})
    pins={str(p.relative_to(ROOT)):sha(p) for p in [source,trainmeta,selmeta,CONFIG,SPLIT]}
    codepins={n:sha(ROOT/n) for n in names}
    plan={'format':'r10_reference_transfer_probe_plan_v1','created_utc':utc(),'seed':SEED,
          'reference_partition':'fixed metadata-besttested spectral211/197; no score-based selection',
          'split_source':str(SPLIT.relative_to(ROOT)),'source_sha256':pins,'code_sha256':codepins,
          'train_examples':16000,'early_stop_A_examples':1000,'paired_comparison_examples_per_reference_pool':1000,
          'full_observed_minutes_only':15,'fold':0,'TRAIN_background_role':'train','all_heldout_background_role':'selection',
          'early_stop_vs_comparison_background_indices_disjoint':True,
          'paired_A_B_same_background_zero_family_EQ_SNR_rng':True,
          'continuous_vs_geometry_rng':'independent streams preserve Uniform laws and exact EQ/SNR pairing despite different native lengths',
          'pairing_limit':'Native source duration controls crop/start RNG range, so time occupancy and true share need not match.',
          'sampler':'unchanged pipeline.sampler_r9.synthesize minute-first; allwhole-recordingaddedenergy',
          'no_source_fraction':.2,'family_probabilities':[.5,.25,.25],'snr_db':[-10.,10.],'smooth_eq_max_db':6.,
          'TRAIN_hard_negative_schedule':'existing R9 candidate metadata; requested10% allrows if available; no duplicated primary',
          'tree_settings':json.loads(CONFIG.read_text())['tree'],
          'production_mutated':False,'Test_or_unused_report_opened':False,'single_predeclared_probe_no_tuning':True}
    write(out/'predeclared_plan.json',plan)
    print(json.dumps({'stage':'plan_frozen','plan_sha256':sha(out/'predeclared_plan.json'),'elapsed_seconds':time.perf_counter()-start}),flush=True)
    sources=R9Sources(CONFIG,ROOT,verify_banks=True)
    assignment=pd.read_csv(SPLIT,dtype={'hull_id':str,'recording_day':str,'clip_id':str})
    full=sources.references.copy();full['hull_id']=full.hull_id.astype(str)
    split=full.merge(assignment[['clip_id','role']].rename(columns={'role':'probe_role'}),on='clip_id',validate='one_to_one')
    A=split[split.probe_role.eq('A')].copy();B=split[split.probe_role.eq('B')].copy()
    if (len(A),len(B))!=(211,197):raise AssertionError('Metadata frozen split211/197 differs')
    for key in ('hull_id','recording_day','sha256'):
        if set(A[key])&set(B[key]):raise AssertionError('Reference cross-pool overlap:'+key)
    at=pd.to_datetime(A.utc_begin,utc=True).astype('int64').to_numpy();ae=pd.to_datetime(A.utc_end,utc=True).astype('int64').to_numpy()
    bt=pd.to_datetime(B.utc_begin,utc=True).astype('int64').to_numpy();be=pd.to_datetime(B.utc_end,utc=True).astype('int64').to_numpy()
    overlaps=((at[:,None]<be[None,:])&(bt[None,:]<ae[:,None])).sum()
    if overlaps:raise AssertionError('Cross-reference UTC overlap')
    reference_audit={'A':summaries(A),'B':summaries(B),'excluded_clips':101,
                     'cross_hull_day_bytes_time_overlap':0,'shared_provider_receiver_corpus':True,
                     'weak_class_is_not_new_natural_vessel_ground_truth':True}
    write(out/'reference_metadata_audit.json',reference_audit)
    A.to_csv(out/'reference_A_metadata.csv',index=False);B.to_csv(out/'reference_B_metadata.csv',index=False)
    train=sources.role_metadata(0,'train');held=sources.role_metadata(0,'selection')
    separation=source_separation(train,held)
    train=train[train.qc_observed_minutes.eq(15)].reset_index(drop=True);held=held[held.qc_observed_minutes.eq(15)].reset_index(drop=True)
    if len(train)<16000 or len(held)<2000:raise ValueError('Insufficient unique complete TRAIN/selection backgrounds')
    train=train.iloc[np.random.default_rng(SEED+81).choice(len(train),16000,replace=False)].reset_index(drop=True)
    horder=np.random.default_rng(SEED+82).choice(len(held),2000,replace=False)
    early=held.iloc[horder[:1000]].reset_index(drop=True);compare=held.iloc[horder[1000:]].reset_index(drop=True)
    if set(early.base_window_index)&set(compare.base_window_index):raise AssertionError('Earlystop/comparison repeatedbackground')
    for name,frame in [('train',train),('selection_A',early),('compare',compare)]:frame.to_parquet(out/(name+'_background_metadata.parquet'),index=False)
    bg_audit={'source_separation':separation,'all_complete_observation':True,'sampling':'unique random within each existing role; no Test matching',
              'counts':{k:len(v) for k,v in [('train',train),('early',early),('compare',compare)]},
              'site_counts':{k:{str(s):int(n) for s,n in v.site.value_counts().items()} for k,v in [('train',train),('early',early),('compare',compare)]}}
    write(out/'background_metadata_audit.json',bg_audit)
    hn=sources.hard_negative_metadata(0);hardids=set(hn.loc[hn.hard_negative_candidate,'base_window_index'])
    ar=A.reference_row_index.to_numpy(int);br=B.reference_row_index.to_numpy(int)
    td,tr,ta=make_rows(sources,train,ar,100,hard_ids=hardids)
    vd,vr,va=make_rows(sources,early,ar,200)
    # Evaluation arrays are generated after fitting/freeze below; B never enters any fitting or early-stopping feature matrix.
    for name,data,recipes,audit in [('train_A',td,tr,ta),('selection_A',vd,vr,va)]:
        np.savez_compressed(out/(name+'.npz'),**data);write(out/(name+'_recipes.json'),recipes);write(out/(name+'_schedule.json'),audit)
    binding={'format':'r10_reference_transfer_probe_binding_v1','plan_sha256':sha(out/'predeclared_plan.json'),
             'reference_split_csv_sha256':sha(SPLIT),'fit_reference_pool':'A211','never_fitted_reference_pool':'B197',
             'source_identity':sources.identity,'comparison_not_production':True}
    print(json.dumps({'stage':'data_ready_fitting_CPU','elapsed_seconds':time.perf_counter()-start,'train':len(td['db']),'selection':len(vd['db'])}),flush=True)
    tree=PhysicsLGBM(sources.config['tree'],seed=SEED)
    summary=tree.fit(td,vd,out/'tree_A',binding,fixture=False)
    frozen={'model_summary_sha256':sha(out/'tree_A/summary.json'),
            'artifacts':summary['artifact_sha256'],'binding':binding,'frozen_utc':utc(),
            'evaluation_seen':False,'no_retuning_after_evaluation':True}
    write(out/'FROZEN_BEFORE_COMPARISON.json',frozen)
    print(json.dumps({'stage':'tree_frozen_comparing','best_iteration':summary['best_iteration'],'elapsed_seconds':time.perf_counter()-start}),flush=True)
    restored=PhysicsLGBM.load(out/'tree_A',expected_binding=binding)
    ep=tree.predict(vd['db'][:7],batch_size=7);rp=restored.predict(vd['db'][:7],batch_size=1)
    if any(not np.array_equal(ep[k],rp[k]) for k in ep):raise AssertionError('Strict tree save/reload/batch replay differs')
    results={};compare_data={};predictions={};recipe_lists={}
    for name,refs in [('A',ar),('B',br)]:
        data,recipes,audit=make_rows(sources,compare,refs,300)
        pred=restored.predict(data['db'],batch_size=256)
        np.savez_compressed(out/('comparison_'+name+'.npz'),**data,**{'pred_'+k:v for k,v in pred.items()})
        write(out/('comparison_'+name+'_recipes.json'),recipes);write(out/('comparison_'+name+'_schedule.json'),audit)
        results[name]=metrics(data['share'],data['presence'],pred['share'],pred['score'],data['site'])
        compare_data[name]=data;predictions[name]=pred;recipe_lists[name]=recipes
    a,b=compare_data['A'],compare_data['B']
    if not np.array_equal(a['base_window_index'],b['base_window_index']):raise AssertionError('Pairedbackgroundorderchanged')
    if not np.array_equal(a['presence'],b['presence']):raise AssertionError('Paired controlled zero/presence changed')
    for ra,rb in zip(recipe_lists['A'],recipe_lists['B']):
        for key in ('no_source_request','requested_full_peak_snr_db','reference_eq_db','family_index','background_index'):
            if ra[key]!=rb[key]:raise AssertionError('Paired render controls differ:'+key)
    ydelta=b['share'].astype(float)-a['share'];ediff=(predictions['B']['share']-b['share'])**2-(predictions['A']['share']-a['share'])**2
    earlypred=restored.predict(vd['db'],batch_size=256)
    results['early_stopping_A']=metrics(vd['share'],vd['presence'],earlypred['share'],earlypred['score'],vd['site'])
    for name,digest in pins.items():
        if sha(ROOT/name)!=digest:raise AssertionError('Frozen source/config/split mutated:'+name)
    for name,digest in codepins.items():
        if sha(ROOT/name)!=digest:raise AssertionError('Source code mutated:'+name)
    sources.assert_unchanged(verify_banks=True)
    report={'format':'r10_reference_transfer_probe_results_v1','status':'completed','created_utc':utc(),
            'plan_sha256':sha(out/'predeclared_plan.json'),'frozen_before_comparison_sha256':sha(out/'FROZEN_BEFORE_COMPARISON.json'),
            'reference_metadata':reference_audit,'background_metadata':bg_audit,'tree_summary':summary,
            'evaluations':results,'paired_controls':{'background_order_exact':True,'no_source_family_SNR_EQ_exact':True,
              'paired_samples':1000,'share_target_B_minus_A_mean':float(ydelta.mean()),
              'share_target_B_minus_A_RMS':float(np.sqrt(np.mean(ydelta**2))),
              'squared_error_B_minus_A_mean':float(ediff.mean()),'B_minus_A_RMSE':results['B']['rmse']-results['A']['rmse'],
              'B_minus_A_AUC':results['B']['controlled_addition_auc']-results['A']['controlled_addition_auc'],
              'source_duration_target_distribution_not_equalized':True},
            'strict_save_reload_batch1_vs7_exact':True,'train_A_schedule':ta,'early_A_schedule':va,
            'TRAIN_and_selection_input_sha256':summary['input_sha256'],'all_source_code_data_production_unchanged':True,
            'Test_oldOUTER_unused_report_opened':False,'production_fullsize_R9_replication':False,
            'limitations':['One16000-rowCPUtreeprobe; notfull79000-rowdynamicR9or4NNexperiment.',
              'AandBdifferentmetadata-derivedday/hull/duration/weak-classdistributions; effectsnotpurehullcausality.',
              'Bothsidesretainthesamecorpusreceiver/provider; notnew-sensorbenchmark.',
              'Acomparisonreferencesareseenintraining,butwindows/generatorseedaredifferent;Bonlycomparison.',
              'Completeminute-onlysubsetremovesmissingnessconfoundbutnarrowsdeploymentcoverage.',
              'AUCispresenceofknownwholeaddedrecording,notverifiednaturalvesseltruth.'],
            'elapsed_seconds':time.perf_counter()-start,'script_sha256':sha(Path(__file__))}
    write(out/'RESULTS.json',report)
    text=['# R10 참조 A학습 → A/B 비교 CPU probe','',
          '메타데이터로 사전에 고정한 A211/B197 분할을 한 번만 사용했습니다. 생산 R9 코드·데이터·모델은 바꾸지 않았습니다.',
          'TRAIN 16,000개 및 early stopping 1,000개는 A만 사용했습니다. 모델을 저장·동결한 뒤, 동일한 완전관측 배경 1,000개에 A/B를 각각 추가해 비교했습니다.',
          '', '|패널|N|RMSE|known-addition AUC|평균 정답|평균 예측|','|---|---:|---:|---:|---:|---:|']
    for name in ['early_stopping_A','A','B']:
        r=results[name];text.append(f"|{name}|{r['n']}|{r['rmse']:.6f}|{r['controlled_addition_auc']:.6f}|{r['mean_true']:.6f}|{r['mean_pred']:.6f}|")
    text+=['',f"B−A RMSE 차이: {report['paired_controls']['B_minus_A_RMSE']:+.6f}. AUC 차이: {report['paired_controls']['B_minus_A_AUC']:+.6f}.",
           '', 'A/B는 hull·녹음일·UTC 구간·파일 bytes가 겹치지 않습니다. 다만 같은 제공자/수신기 코퍼스이며, 길이와 약한 클래스 분포도 다릅니다. 이 차이를 특정 hull 암기 또는 실제 대회 성능의 단일 원인으로 단정할 수 없습니다.',
           '동일한 배경·no-source 20%·family·SNR·EQ 난수를 썼지만 원참조 길이가 달라 시간 점유율과 정답 share 분포가 같지는 않습니다. 이 분포를 결과에 맞춰 보정하지 않았습니다.',
           '모든 표적은 합성 시 실제로 추가한 전체 녹음의 에너지 비율입니다. 자연 상태 선박의 절대 정답 라벨이 아닙니다.',
           f"CPU 실행 {report['elapsed_seconds']:.1f}초. 회귀 best boosting round {summary['best_iteration']}, 분류 best round {summary['presence_best_iteration']}. 100-epoch 신경망 정책은 tree에 적용되지 않습니다.",
           '', '[수치·출처·설정 전체](RESULTS.json) / [사전 계획](predeclared_plan.json) / [참조 분포](reference_metadata_audit.json) / [배경 분리](background_metadata_audit.json)']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'stage':'completed','evaluations':{k:{m:v for m,v in r.items() if m in ('rmse','controlled_addition_auc','mean_true','mean_pred')} for k,r in results.items()},'elapsed_seconds':report['elapsed_seconds']}),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reports/r10_diagnostics/reference_transfer_probe_r2')
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write(a.output/'FAILED.json',{'type':type(e).__name__,'error':str(e),'partial_reuse_allowed':False})
        raise

if __name__=='__main__':main()
