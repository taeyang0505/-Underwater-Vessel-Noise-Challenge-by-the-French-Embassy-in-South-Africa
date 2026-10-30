"""Bounded full-size CPU2 CatBoost timing. Never a production training run.

Reads only immutable Fold0 D0 TRAIN/CV. Stores reusable exact feature mmap,
not fitted models; records feature extraction, quantization, fit and RSS.
The wall-time guard stops iterative fitting after the next tree completes.
"""
from pathlib import Path
import argparse, datetime, gc, json, os, resource, sys, time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/r16_pretrained_ft_r1'
CACHE=ROOT/'data/cache/r16_pretrained_catboost_benchmark_r1/fold_00_D0'


def main(budget_seconds=480,trees=64,extra_threads8=False):
    import numpy as np
    from catboost import Pool,CatBoostRegressor,CatBoostClassifier
    from pipeline.catboost_r16_peft import sha,write,source_binding,verify,DEFAULTS,FEATURE_COUNTS,dependency_status
    from pipeline.r16_multisupport_data import role,OUT as DATA,guard
    from pipeline.training_r16_multisupport import tree_features
    sys.addaudithook(guard)
    if CACHE.exists():raise FileExistsError('Fresh benchmark cache only; inspect existing receipt rather than overwrite')
    CACHE.mkdir(parents=True);started=time.perf_counter();deadline=started+budget_seconds
    if trees not in range(1,65):raise ValueError('Bounded benchmark only,1..64 trees')
    pins=source_binding();pins[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    datahash=sha(DATA/'manifest.json');manifest=json.loads((DATA/'manifest.json').read_text())
    train=role(0,'D0','train');cv=role(0,'D0','cv')
    if (len(train['share']),len(cv['share']))!=(61446,1000):raise ValueError('Exact full-size Fold0D0 dataset required')
    if set(train['base_window_index'])&set(cv['base_window_index']):raise ValueError('CV/train overlap')
    report=dict(status='running',format='r16_catboost_fullsize_benchmark_v1',
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        train_n=len(train['share']),CV_n=len(cv['share']),regime='D0',fold=0,
        CPU_threads=2,maximum_benchmark_trees=trees,wall_budget_seconds=budget_seconds,
        dependency=dependency_status(True),parameters=dict(DEFAULTS),
        data_manifest_sha256=datahash,source_sha256=pins,
        training_npz=manifest['folds']['0']['roles']['train']['D0'],
        CV_npz=manifest['folds']['0']['roles']['cv']['D0'],
        shared_local_fit_read=False,Test_used=False,locked_report_panels_read=False,
        production_training=False,feature_extraction={},fits={})
    def rss():return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    def commit(stage):
        report.update(stage=stage,elapsed_seconds=time.perf_counter()-started,peak_process_rss_bytes=rss())
        write(OUT/'CATBOOST_FULLSIZE_TIMING.json',report)
        print(json.dumps({'stage':stage,'elapsed_seconds':report['elapsed_seconds'],
              'peak_process_rss_GiB':rss()/2**30,'feature_extraction':report['feature_extraction'],
              'fits':report['fits']},allow_nan=False),flush=True)
    def too_long():return time.perf_counter()>deadline
    def memory_guard():
        if rss()>12*2**30:raise MemoryError('Benchmark12GiBprocess RSS guard')
    try:
        commit('feature_extraction_started')
        cachemeta={'status':'running','source_sha256':pins,'data_manifest_sha256':datahash,
                   'train_n':61446,'CV_n':1000,'dtype':'float32','features':{},
                   'source_role':'own-fold0 D0 TRAIN and CV only','Test_used':False}
        for kind,width in FEATURE_COUNTS.items():
            elapsed={}
            for label,d in [('train',train),('cv',cv)]:
                path=CACHE/f'{kind}_{label}.npy';begin=time.perf_counter()
                x=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(d['share']),width))
                for off in range(0,len(x),256):
                    a=tree_features(d['db'][off:off+256],kind)
                    if a.shape!=(min(256,len(x)-off),width) or not np.isfinite(a).all():raise ValueError('Original features mismatch')
                    x[off:off+256]=a
                    memory_guard()
                x.flush();del x;gc.collect()
                elapsed[label+'_seconds']=time.perf_counter()-begin
                cachemeta['features'][path.name]=dict(shape=[len(d['share']),width],bytes=path.stat().st_size,sha256=sha(path))
            report['feature_extraction'][kind]=elapsed
            write(CACHE/'manifest.json',cachemeta);commit('features_'+kind+'_completed')
            if too_long():
                report['status']='partial_wall_budget';commit('stopped_after_feature_extraction');render(report);return
        cachemeta['status']='completed';write(CACHE/'manifest.json',cachemeta)
        report['cache_manifest_sha256']=sha(CACHE/'manifest.json');commit('all_features_completed')
        for kind in FEATURE_COUNTS:
            iscls=kind=='classification';key='presence' if iscls else 'share'
            x=np.load(CACHE/f'{kind}_train.npy',mmap_mode='r',allow_pickle=False)
            v=np.load(CACHE/f'{kind}_cv.npy',mmap_mode='r',allow_pickle=False)
            begin=time.perf_counter();pool=Pool(x,train[key],thread_count=2);valid=Pool(v,cv[key],thread_count=2)
            pool_seconds=time.perf_counter()-begin;memory_guard()
            # These are CatBoost's default numeric quantization settings. The
            # same TRAIN borders are applied to CV; no query fitting occurs.
            begin=time.perf_counter();pool.quantize(border_count=254,feature_border_type='GreedyLogSum',
                task_type='CPU',thread_count=2,random_seed=DEFAULTS['random_seed'])
            borders=CACHE/f'{kind}_borders.tsv';pool.save_quantization_borders(str(borders))
            valid.quantize(input_borders=str(borders),task_type='CPU',thread_count=2)
            quant_seconds=time.perf_counter()-begin;memory_guard()
            params=dict(DEFAULTS,iterations=trees)
            model=(CatBoostClassifier(loss_function='Logloss',eval_metric='AUC',**params) if iscls
                   else CatBoostRegressor(loss_function='RMSE',eval_metric='RMSE',**params))
            ticks=[];fit_started=time.perf_counter()
            class Timer:
                def after_iteration(self,info):
                    ticks.append({'iteration':int(info.iteration),'elapsed_seconds':time.perf_counter()-fit_started})
                    if info.iteration%16==0:print(json.dumps({'stage':'training_benchmark','kind':kind,**ticks[-1]},allow_nan=False),flush=True)
                    return not too_long() and rss()<12*2**30
            # Deliberately no early stopping: this measures bounded tree cost,
            # not a selected model or performance experiment.
            model.fit(pool,eval_set=valid,use_best_model=False,callbacks=[Timer()],verbose=False)
            fit_seconds=time.perf_counter()-fit_started
            deltas=np.diff([a['elapsed_seconds'] for a in ticks])
            steady=deltas[7:] if len(deltas)>8 else deltas
            seconds_per_tree=float(np.median(steady)) if len(steady) else fit_seconds/max(1,len(ticks))
            actual=model.get_all_params()
            report['fits'][kind]=dict(pool_create_seconds=pool_seconds,quantization_seconds=quant_seconds,
                fit_seconds=fit_seconds,iterations_observed=len(ticks),retained_trees=int(model.tree_count_),
                CPU_threads=2,per_iteration=ticks,steady_median_seconds_per_tree=seconds_per_tree,
                steady_p10_seconds_per_tree=float(np.quantile(steady,.1)) if len(steady) else None,
                steady_p90_seconds_per_tree=float(np.quantile(steady,.9)) if len(steady) else None,
                fit_fixed_overhead_estimate_seconds=max(0.,fit_seconds-seconds_per_tree*len(ticks)),
                quantization_parameters={k:actual.get(k) for k in ('border_count','feature_border_type','nan_mode')},
                peak_process_rss_bytes=rss(),early_stopping_disabled_for_timing=True,
                train_objective=actual.get('loss_function'))
            del model,pool,valid,x,v;gc.collect();commit('benchmark_'+kind+'_completed')
            if too_long():break
        complete=set(report['fits'])==set(FEATURE_COUNTS) and all(a['iterations_observed']==trees for a in report['fits'].values())
        report['status']='completed' if complete else 'partial_wall_budget'
        if complete:
            scenarios={}
            for rounds in (250,750,1500):
                per_pair=sum(a['pool_create_seconds']+a['quantization_seconds']+a['fit_fixed_overhead_estimate_seconds']+
                    rounds*a['steady_median_seconds_per_tree'] for a in report['fits'].values())
                extraction=sum(sum(a.values()) for a in report['feature_extraction'].values())
                scenarios[str(rounds)]={'iterations_per_task':rounds,'one_regime_fold_pair_fit_seconds':per_pair,
                    'one_pair_total_with_fresh_feature_extraction_seconds':per_pair+extraction,
                    'D0H0_x5_10pairs_fit_hours':10*per_pair/3600,
                    'D0H0_x5_10pairs_total_fresh_features_hours':10*(per_pair+extraction)/3600,
                    'assumption':'Both tasks use stated rounds; Fold0D0 cost scaled10x, no assumption of early-stop epoch'}
            report['extrapolation']=scenarios
            report['extrapolation_limits']=['Early stopping may retain250/750/1500 or another tree count; not measured by this no-ES benchmark.',
                'Other Fold/regime feature distributions and CatBoost quantization may change iteration cost.',
                'File cache pressure, concurrent browser/MPS jobs, thermal throttling and battery state affect Mac wall time.',
                'Prediction/export/fixed-panel audit and queue interruptions are excluded from fit timing.',
                'No speed ratio to4090 inferred; CatBoost benchmark is CPU only.']
        report['extra_CPU8_requested']=extra_threads8
        report['extra_CPU8_run']=False
        verify(pins);commit('finished');render(report)
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}';commit('failed');render(report);raise


def render(d):
    lines=['# CatBoost Mac CPU 전체 크기 실측 시간', '',
        f"상태: **{d['status']}**. Fold0 D0 TRAIN 61,446행 / own-fold CV 1,000행. 공식 Test·고정 report 패널·shared ensemble-fit은 읽지 않았습니다.", '',
        '모델 본학습이 아닌 최대 64트리 timing 전용 실행입니다. 같은 5,160/2,033 특징, depth8·lr0.035·CPU2 설정을 사용했습니다.', '',
        '| 태스크 | 특징 생성 TRAIN/CV(초) | Pool 생성(초) | 양자화(초) | 측정 트리 | fit(초) | 안정 구간 초/트리 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for kind,f in d.get('feature_extraction',{}).items():
        a=d.get('fits',{}).get(kind,{})
        lines.append('| {} | {:.2f} / {:.2f} | {} | {} | {} | {} | {} |'.format(kind,f['train_seconds'],f['cv_seconds'],
            *[f'{a[k]:.3f}' if isinstance(a.get(k),float) else str(a.get(k,'미측정')) for k in
              ('pool_create_seconds','quantization_seconds','iterations_observed','fit_seconds','steady_median_seconds_per_tree')]))
    if d.get('extrapolation'):
        lines+=['', '## ES 시점별 예상 범위', '',
                '학습 시에는 최대1,500트리/ES40입니다. 아래는 실제 안정 구간 tree 시간을 외삽한 가정이며, ES 시점을 미리 안다는 뜻이 아닙니다.', '',
                '| 회귀·분류 각각의 실제 학습 트리 수 가정 | 1개 D/H·Fold 쌍 fit | 새 특징 포함 1쌍 | D0/H0×5=10쌍 fit | 새 특징 포함 10쌍 |',
                '|---|---:|---:|---:|---:|']
        for rounds,a in d['extrapolation'].items():
            lines.append(f"| {rounds} | {a['one_regime_fold_pair_fit_seconds']/60:.1f}분 | {a['one_pair_total_with_fresh_feature_extraction_seconds']/60:.1f}분 | {a['D0H0_x5_10pairs_fit_hours']:.2f}시간 | {a['D0H0_x5_10pairs_total_fresh_features_hours']:.2f}시간 |")
    lines+=['',f"프로세스 최대 RSS: {d.get('peak_process_rss_bytes',0)/2**30:.2f}GiB. 전체 작업 경과: {d.get('elapsed_seconds',0):.1f}초.", '',
            '특징 mmap과 원천/소스 SHA manifest는 `data/cache/r16_pretrained_catboost_benchmark_r1/fold_00_D0/`에 저장했습니다. 향후 재사용은 소스·데이터·특징 schema 해시가 모두 일치할 때만 허용합니다.', '',
            '## 한계', '']+['- '+s for s in d.get('extrapolation_limits',['전체 타이밍이 완료되지 않아 전체 예상 시간을 확정하지 않았습니다.'])]
    (OUT/'CATBOOST_FULLSIZE_TIMING.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--budget-seconds',type=int,default=480)
    p.add_argument('--trees',type=int,default=64);p.add_argument('--extra-threads8',action='store_true');a=p.parse_args()
    main(a.budget_seconds,a.trees,a.extra_threads8)
