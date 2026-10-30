"""Render requested300/600/1500 scenarios from measured fullsize timing."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline.catboost_r16_peft import write,sha
from ops.benchmark_r16_catboost_cached_cpu import render_local


if __name__=='__main__':
    p=ROOT/'reports/r16_pretrained_ft_r1/CATBOOST_FULLSIZE_TIMING.json';d=json.loads(p.read_text())
    if d['status']!='completed':raise ValueError('Both CPU2 full64 timings required')
    # The first API failure is a separate preserved attempt, never an active
    # training error or an unexplained stale top-level failure field.
    d.pop('error',None)
    d['summary_script_sha256']=sha(__file__)
    d['feature_extraction_10pairs_minutes']=10*d['original_feature_extraction_seconds']/60
    d['remaining_feature_extraction_9pairs_minutes']=9*d['original_feature_extraction_seconds']/60
    d['total_requested_models']=20;d['total_requested_regime_fold_pairs']=10
    d['extrapolation']={}
    for label,fits in [('CPU2',d['fits']),('CPU8',d['CPU8_fits'])]:
        if len(fits)!=2 or any(a['iterations_observed']<8 for a in fits.values()):continue
        for n in (300,600,1500):
            fit=sum(a['pool_create_seconds']+a['fit_fixed_overhead_estimate_seconds']+
                n*a['steady_median_seconds_per_tree'] for a in fits.values())
            extraction=d['original_feature_extraction_seconds']
            d['extrapolation'][f'{label}_{n}']=dict(CPU_threads=2 if label=='CPU2' else 8,
                iterations_per_task=n,one_regime_fold_pair_fit_seconds=fit,
                one_pair_total_with_fresh_feature_extraction_seconds=fit+extraction,
                D0H0_x5_10pairs_fit_hours=10*fit/3600,
                D0H0_x5_10pairs_total_fresh_features_hours=10*(fit+extraction)/3600,
                D0H0_x5_10pairs_total_with_one_existing_cache_hours=(10*fit+9*extraction)/3600,
                assumption='Both tasks execute givenrounds; fixed61446rows and5160/2033features approximation, not upperbound or guarantee')
    d['extrapolation_limits'] += [
        'CPU2 measured64trees/task; CPU8measured16trees/task. Extrapolationto1500 is long-range andnot a hardupperbound/timeguarantee.',
        'TenpairsmeansD0/H0x5folds=20models(10regressors+10classifiers), not10models.',
        'Allfresh-featuretiming addsabout40.09min, measuredCPU2. OneFold0D0featurecache isalreadycomplete, leavingabout36.08minfor9newpairs.']
    write(p,d);render_local(d)
    md=p.with_suffix('.md');s=md.read_text()
    s+='\n## 데이터·열 수를 고정한 근사라는 점\n\n'
    s+='- D0/H0 × 5 Fold = **10쌍, 회귀10개 + 분류10개 = 총20개 모델**입니다.\n'
    s+=f"- 새 특징을 10쌍 모두 처음 생성한다면 **{d['feature_extraction_10pairs_minutes']:.2f}분**이 별도입니다. 현재 Fold0D0 캐시1쌍은 완성되어 나머지9쌍은 같은속도가정 **{d['remaining_feature_extraction_9pairs_minutes']:.2f}분**입니다.\n"
    s+='- CPU2는64트리, CPU8은16트리까지만 실측했습니다. 위300/600/1500표는 선형 외삽이며 **최대시간 보장·hard upper bound가 아닙니다**. ES40이 언제 작동할지는 본학습 전 알 수 없습니다.\n'
    s+='- 초기 Pool.quantize(thread_count) API불일치는 `CATBOOST_FULLSIZE_FEATURES_API_RECEIPT.json`에 보존했습니다. 후속public-fit방식 CPU2/CPU8벤치마크는 정상 완료했고, 데이터나 학습 수치 실패가 아닙니다.\n'
    md.write_text(s)
    print(json.dumps({'status':d['status'],'feature_10pairs_minutes':d['feature_extraction_10pairs_minutes'],
        'scenarios':d['extrapolation']},ensure_ascii=False),flush=True)
