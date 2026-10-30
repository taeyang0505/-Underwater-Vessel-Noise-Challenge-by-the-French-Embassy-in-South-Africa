"""Timing-only recovery on SHA-verified full-size cached R16 features.

Uses CatBoost's public fit path with an explicit thread count. The library
does not expose thread_count on Pool.quantize, so quantization/startup is
reported as an estimated fixed overhead, never an exact isolated timer.
"""
from pathlib import Path
import gc,json,resource,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
CACHE=ROOT/'data/cache/r16_pretrained_catboost_benchmark_r1/fold_00_D0'
OUT=ROOT/'reports/r16_pretrained_ft_r1'


def main(budget_seconds=360):
    import numpy as np
    from catboost import Pool,CatBoostRegressor,CatBoostClassifier
    from pipeline.catboost_r16_peft import sha,write,DEFAULTS,FEATURE_COUNTS,verify
    from pipeline.r16_multisupport_data import role,guard,OUT as DATA
    from ops.benchmark_r16_catboost_cpu import render
    sys.addaudithook(guard);start=time.perf_counter();deadline=start+budget_seconds
    old=json.loads((OUT/'CATBOOST_FULLSIZE_TIMING.json').read_text())
    if old.get('fits'):raise ValueError('Do not overwrite successful prior fit timing')
    write(OUT/'CATBOOST_FULLSIZE_FEATURES_API_RECEIPT.json',old)
    meta=json.loads((CACHE/'manifest.json').read_text())
    if meta['status']!='completed' or meta['data_manifest_sha256']!=sha(DATA/'manifest.json'):raise ValueError('Cache/data mismatch')
    verify(meta['source_sha256'])
    for name,a in meta['features'].items():
        if sha(CACHE/name)!=a['sha256']:raise ValueError('Feature matrix SHA changed')
    train=role(0,'D0','train');cv=role(0,'D0','cv')
    if (len(train['share']),len(cv['share']))!=(61446,1000):raise ValueError('Fullsize metadata required')
    report={**old,'status':'running','stage':'cached_fit_started','fits':{},'CPU8_fits':{},
        'first_attempt_API_error_preserved':'CATBOOST_FULLSIZE_FEATURES_API_RECEIPT.json',
        'recovery_script_sha256':sha(__file__),'recovery_budget_seconds':budget_seconds,
        'quantization_timing_method':'PublicfitCPUthreads; first iteration includes quantization+startup; fixed overhead is fit_seconds-steady_median*tree_count estimate, not exact quantization.',
        'original_feature_extraction_seconds':sum(sum(a.values()) for a in old['feature_extraction'].values())}
    rss=lambda:int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    def commit(stage):
        report.update(stage=stage,recovery_elapsed_seconds=time.perf_counter()-start,
            elapsed_seconds=old['elapsed_seconds']+time.perf_counter()-start,peak_process_rss_bytes=rss())
        write(OUT/'CATBOOST_FULLSIZE_TIMING.json',report)
        print(json.dumps({'stage':stage,'seconds':time.perf_counter()-start,'peak_RSS_GiB':rss()/2**30,
            'fits':{k:{a:b for a,b in v.items() if a!='per_iteration'} for k,v in report['fits'].items()},
            'CPU8_fits':{k:{a:b for a,b in v.items() if a!='per_iteration'} for k,v in report['CPU8_fits'].items()}}),flush=True)
    def run(kind,threads,trees):
        x=np.load(CACHE/f'{kind}_train.npy',mmap_mode='r',allow_pickle=False)
        v=np.load(CACHE/f'{kind}_cv.npy',mmap_mode='r',allow_pickle=False)
        cls=kind=='classification';key='presence' if cls else 'share'
        begin=time.perf_counter();pool=Pool(x,train[key],thread_count=threads);valid=Pool(v,cv[key],thread_count=threads)
        pool_seconds=time.perf_counter()-begin
        args=dict(DEFAULTS,iterations=trees,thread_count=threads)
        model=(CatBoostClassifier(loss_function='Logloss',eval_metric='AUC',**args) if cls else
            CatBoostRegressor(loss_function='RMSE',eval_metric='RMSE',**args))
        ticks=[];began=time.perf_counter()
        class Timer:
            def after_iteration(self,info):
                ticks.append({'iteration':int(info.iteration),'elapsed_seconds':time.perf_counter()-began})
                if info.iteration%16==0:print(json.dumps({'stage':'benchmark_iteration','kind':kind,'threads':threads,**ticks[-1]}),flush=True)
                return time.perf_counter()<deadline and rss()<12*2**30
        model.fit(pool,eval_set=valid,use_best_model=False,callbacks=[Timer()],verbose=False)
        elapsed=time.perf_counter()-began;deltas=np.diff([a['elapsed_seconds'] for a in ticks])
        steady=deltas[7:] if len(deltas)>8 else deltas
        median=float(np.median(steady)) if len(steady) else elapsed/max(1,len(ticks))
        actual=model.get_all_params()
        result=dict(pool_create_seconds=pool_seconds,quantization_seconds=None,
            quantization_or_startup_fixed_overhead_estimate_seconds=max(0.,elapsed-median*len(ticks)),
            fit_fixed_overhead_estimate_seconds=max(0.,elapsed-median*len(ticks)),
            first_iteration_seconds=ticks[0]['elapsed_seconds'] if ticks else None,
            fit_seconds=elapsed,iterations_observed=len(ticks),retained_trees=int(model.tree_count_),
            CPU_threads=threads,steady_median_seconds_per_tree=median,
            steady_p10_seconds_per_tree=float(np.quantile(steady,.1)) if len(steady) else None,
            steady_p90_seconds_per_tree=float(np.quantile(steady,.9)) if len(steady) else None,
            per_iteration=ticks,peak_process_rss_bytes=rss(),early_stopping_disabled_for_timing=True,
            quantization_parameters={k:actual.get(k) for k in ('border_count','feature_border_type','nan_mode')})
        del model,pool,valid,x,v;gc.collect();return result
    commit('cache_verified_training_started')
    try:
        for kind in FEATURE_COUNTS:
            report['fits'][kind]=run(kind,2,64);commit('CPU2_'+kind+'_completed')
            if time.perf_counter()>=deadline:break
        if len(report['fits'])==2:
            for kind in FEATURE_COUNTS:
                if deadline-time.perf_counter()<20:break
                report['CPU8_fits'][kind]=run(kind,8,16);commit('CPU8_'+kind+'_completed')
        scenarios={}
        for label,fits in [('CPU2',report['fits']),('CPU8',report['CPU8_fits'])]:
            if len(fits)!=2:continue
            for rounds in (250,750,1500):
                fit=sum(a['pool_create_seconds']+a['fit_fixed_overhead_estimate_seconds']+
                    rounds*a['steady_median_seconds_per_tree'] for a in fits.values())
                extraction=report['original_feature_extraction_seconds']
                scenarios[f'{label}_{rounds}']=dict(CPU_threads=2 if label=='CPU2' else 8,iterations_per_task=rounds,
                    one_regime_fold_pair_fit_seconds=fit,
                    one_pair_total_with_fresh_feature_extraction_seconds=fit+extraction,
                    D0H0_x5_10pairs_fit_hours=10*fit/3600,
                    D0H0_x5_10pairs_total_fresh_features_hours=10*(fit+extraction)/3600,
                    assumption='Fold0D0 boundediteration timing scaled10pairs; ES tree count unknown')
        report.update(status='completed' if len(report['fits'])==2 and all(a['iterations_observed']==64 for a in report['fits'].values()) else 'partial_wall_budget',
            extrapolation=scenarios,extra_CPU8_run=bool(report['CPU8_fits']),
            extrapolation_limits=['NoES timing run; eventual best tree count and ES40overhead unknown.',
                'Fixed overhead includes estimated quantization/startup; publicPool.quantize has no thread_count parameter.',
                'CPU8 is a distinct measured setting; existing production default remainsCPU2.',
                '64/16tree timing may not predict later boostinground cost exactly; cache/thermal/otherjobs affect time.',
                'Fold0D0 extrapolation applies10pairs toD0H0x5; panel evaluation/export excluded.',
                'Feature extraction was measuredCPU2and isreused; CPU8featureextraction wasnotbenchmarked.'])
        verify(meta['source_sha256']);commit('finished');render_local(report)
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}'
        commit('failed');render_local(report);raise


def render_local(d):
    lines=['# CatBoost 전체 크기 Mac CPU 시간 실측','',
        f"상태: **{d['status']}**. 실제 Fold0 D0 TRAIN 61,446행 / own-CV 1,000행.",
        '동결된 기존 특징: 회귀5,160개 / 독립 분류2,033개. depth8, lr0.035. 생산 모델을 저장하거나 Test·고정 report·shared fit을 읽지 않았습니다.','',
        '## 특징 생성', '', '| 태스크 | TRAIN(초) | CV(초) |','|---|---:|---:|']
    for k,a in d['feature_extraction'].items():lines.append(f"| {k} | {a['train_seconds']:.2f} | {a['cv_seconds']:.2f} |")
    lines+=['','## 실제 boosting 시간','',
        '첫 iteration에는 양자화·초기화가 포함됩니다. CatBoost 공개 `Pool.quantize`에 thread_count가 없어, 이를 정확한 별도 양자화 타이머로 보고하지 않고 전체fit에서 안정 구간을 뺀 고정비 추정으로 표시했습니다.','',
        '| 스레드 | 태스크 | 측정 트리 | Pool생성(초) | 양자화 포함fit(초) | 첫iteration(초) | 안정구간 초/트리 | 고정비 추정(초) |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for fits in (d.get('fits',{}),d.get('CPU8_fits',{})):
        for k,a in fits.items():lines.append(f"| {a['CPU_threads']} | {k} | {a['iterations_observed']} | {a['pool_create_seconds']:.2f} | {a['fit_seconds']:.2f} | {a['first_iteration_seconds']:.2f} | {a['steady_median_seconds_per_tree']:.3f} | {a['fit_fixed_overhead_estimate_seconds']:.2f} |")
    lines+=['','## ES 시나리오 외삽','',
        '최대1,500트리·ES40을 실제로 돌려 보기 전 최적 트리 수는 알 수 없습니다. 아래 트리 수는 회귀/분류 각각의 실제 실행 트리 수에 대한 가정입니다. 일반화 성능이나 ES발동점을 예측하지 않습니다.','',
        '| 스레드 | 각태스크 트리 수 | 1쌍fit | 새특징포함1쌍 | D0/H0×5=10쌍fit | 새특징포함10쌍 |',
        '|---:|---:|---:|---:|---:|---:|']
    for a in d.get('extrapolation',{}).values():lines.append(f"| {a['CPU_threads']} | {a['iterations_per_task']} | {a['one_regime_fold_pair_fit_seconds']/60:.1f}분 | {a['one_pair_total_with_fresh_feature_extraction_seconds']/60:.1f}분 | {a['D0H0_x5_10pairs_fit_hours']:.2f}시간 | {a['D0H0_x5_10pairs_total_fresh_features_hours']:.2f}시간 |")
    lines+=['',f"회복 실행 최대RSS: {d.get('peak_process_rss_bytes',0)/2**30:.2f}GiB. 최초 특징생성 기록은 별도API영수증에 보존했습니다.",'',
        '캐시: `data/cache/r16_pretrained_catboost_benchmark_r1/fold_00_D0/manifest.json`. 원천·소스·각mmapSHA검증 후 재사용 가능합니다. 기존 CPU2 본학습 설정을 CPU8로 몰래 변경하지 않았습니다.','',
        '## 한계','']+['- '+s for s in d.get('extrapolation_limits',[])]
    (OUT/'CATBOOST_FULLSIZE_TIMING.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
