"""R13 paired TRAIN audit/build/verify; no models, Test or validation values."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
        name=Path(str(args[0])).name.lower()
        if name in ('test.csv','sample_submission.csv'):
            raise PermissionError('R13 TRAIN builder forbids Test input')
    if event in ('socket.connect','subprocess.Popen','os.system'):raise PermissionError('No network/process actions')


def fixture():
    import numpy as np
    from pipeline.dataset_r12 import synthesize_r12
    from pipeline.dataset_r13 import components,_render,reconstruct_reference_from_recipe,solve_observed_mean_mixture,highshare_schedule
    rng=np.random.default_rng(12);cases=0;maxerr=0.;gainerr=0.
    for n in [1,3,8,18]:
        native=np.exp(rng.normal(size=(n,25)))
        for family in ['native_inside','native_short','native_edge']+(['long_native'] if n>=15 else [])+(['long_smooth_bell'] if 5<=n<=14 else []):
            bg=np.round(rng.normal(74,4,(15,25)),1);bg[[0,3,6],:]=np.nan
            old,recipe=synthesize_r12(bg,native,seed=500+n,family=family)
            placed=reconstruct_reference_from_recipe(native,recipe);b,a,e,o,top=components(bg,placed,recipe['alpha']);again,qa=_render(b,a,e,o,top)
            for k in again:assert np.array_equal(again[k],old[k],equal_nan=np.asarray(old[k]).dtype.kind=='f')
            for u in [.25,.55,.85]:
                new,audit=solve_observed_mean_mixture(bg,placed,u)
                maxerr=max(maxerr,abs(audit['raw_share_float64']-u));assert abs(float(new['share'])-u)<3e-8
                assert np.array_equal(np.isfinite(new['db']),np.isfinite(bg))
                gained,_=solve_observed_mean_mixture(bg+10,placed,u)
                ge=float(np.max(abs(gained['cell_share']-new['cell_share'])));gainerr=max(gainerr,ge);assert ge==0
                assert np.allclose(gained['unrounded_db'],new['unrounded_db']+10,equal_nan=True,atol=1e-12)
                cases+=1
    def rejects(fn):
        try:fn()
        except (ValueError,TypeError):return True
        raise AssertionError('Invalid physics silently accepted')
    rejects(lambda:solve_observed_mean_mixture(np.full((15,25),np.nan),np.ones((15,25)),.5))
    rejects(lambda:solve_observed_mean_mixture(np.full((15,25),75.),np.zeros((15,25)),.5))
    rejects(lambda:solve_observed_mean_mixture(np.full((15,25),75.),np.ones((15,25)),1.))
    fake={'share':np.resize(np.array([0,.1,.1,.1,.1]),12000),'presence':np.resize(np.array([0,1,1,1,1]),12000),
          'family':np.full(12000,'native_inside',dtype='U32')}
    positive=np.flatnonzero(fake['presence']>0);fake['family'][positive[:2880]]='long_smooth_bell'
    # Subset exactly halves each declared source stratum, as immutable R12.
    subset=np.r_[np.flatnonzero(fake['presence']==0)[:1200],positive[:1440],positive[2880:2880+3360]]
    chosen,requested,strata=highshare_schedule(fake,subset);again=highshare_schedule(fake,subset)
    assert chosen.sum()==2400 and np.array_equal(chosen,again[0]) and np.array_equal(requested,again[1],equal_nan=True)
    return {'status':'passed','fixture_only':True,'physical_R12_replay_and_highshare_cases':cases,'max_float64_share_target_error':maxerr,
        'common_gain_q_error':gainerr,'invalid_empty_support_and_target_rejected':True,'highshare2400_and_CNN1200_exact':True,
        'exact_LHS_seed_replay':True,'acoustic_Test_validation_model_access':False}


def reports(folder):
    from pipeline.dataset_r13 import local,read,sha,write
    p=local(folder);m=read(p/'manifest.json');a=read(p/'distribution_and_physics_audit.json');v=read(p/'independent_verification.json')
    target=ROOT/'reports/R13_DATASET_CONSTRUCTION.json'
    value={'status':'completed','data_dir':str(p),'data_hash':m['data_hash'],'manifest_sha256':sha(p/'manifest.json'),
        'readiness_sha256':sha(p/'readiness.json'),'independent_verification':v,'distribution':a,
        'existing_R12_sources_models_or_outputs_changed':False,'model_training_evaluation_Test_validation_values_read':False}
    write(target,value)
    b=a['baseline'];h=a['highshare'];rows=[]
    for name in ['all','positive','long_positive','standard_positive','support_le2_positive','support_ge12_positive']:
        x=b['strata'][name]['share'];y=h['strata'][name]['share']
        rows.append(f"|{name}|{x['n']}|{x['mean']:.6f}|{y['mean']:.6f}|{x['quantiles']['0.5']:.6f}|{y['quantiles']['0.5']:.6f}|")
    md='''# R13 paired TRAIN: 고비율 추가 에너지 보강\n\n**완료: TRAIN 생성·물리 검증만 수행했습니다. 모델 학습·평가·Test/공통 validation 값 읽기는 0회입니다.** 기존 R12 strict site/sensor/recorder/effective-mask matched v2 baseline은 외부 원본을 그대로 참조합니다. baseline `load_control`과 R13 baseline loader의 key/dtype/array bytes가 같습니다.\n\n## 고정 법칙\n\nTRAIN12,000개에서 zero2,400개(20%), positive9,600개를 유지합니다. long-positive2,880개(30%)도 그대로입니다. 양성의 정확25%인2,400개만 alpha를 변경합니다: long720/standard1,680. 기존 CNN6,000개 subset에도 동일비율 highshare1,200개(long360/standard840)입니다. standard는 원래 native inside/short/edge이며 실제 오디오가 반드시 짧다는 의미가 아닙니다. source·site·sensor·recorder·정확 mask·참조·EQ·배치·sample ID·presence는 모두 보존합니다. 나머지9,600개 행의 공통 NPZ 필드 bytes는 같습니다.\n\n목표 u는 각 long/standard×CNN membership 층에서 독립 fixed-seed LHS `Uniform(.25,.85)`로 사전 고정했습니다. 임의 max(old y,u)나 share/minute-SNR cap을 넣지 않습니다. 따라서 일부 원래 높은 y는 낮아질 수 있으며 up/down 개수를 아래에 투명하게 표시합니다. Test/LB의 비율에 맞춘 분포가 아닙니다.\n\n## 물리식과 정보 가역성\n\nR12의 저장 source·EQ·native power·crop/route를 다시 조립하여 기존9,600개 positive의 raw db/q/y를 전부 정확히 replay했습니다. 반올림 db에서 역추정하거나 모델로 pseudo-label을 만들지 않았습니다. 같은 관측 O에서 `alpha=u·sum(B)/[(1−u)·sum(O R)]`, `A=alpha·O R`, `E=B+A`, `q=A/E`, `y=sum(A)/sum(E)`입니다. 같은 support의 합 비율은 관측분 mean 비율과 같습니다. raw q/y는float64로 검증하고 기존규격float32로 저장하므로 serialization roundoff는 별도 검사합니다. 입력만0.1dB로 반올림하며 w도 같은 관측 격자에서 계산합니다.\n\n정답은 추가한 **녹음 전체**의 에너지 비율입니다. 순수 선박 에너지나 자연 선박 부재 GT가 아닙니다. 현재 baseline 소스는 median-background/full-reference-peak SNR 스케일을 사용하며 명시적인 share cap은 없습니다. 현재 y분포만으로 고비율 성능 문제의 원인을 cap이라고 확정하지 않습니다. 통계적 long bell은 기존 R12 법칙을 그대로 유지하며 새로 측정한 WAV라고 부르지 않습니다.\n\n## TRAIN 분포\n\n|층|행 수|baseline y평균|highshare y평균|baseline y중앙값|highshare y중앙값|\n|---|---:|---:|---:|---:|---:|\n'''+ '\n'.join(rows)+f'''\n\n선정2,400개 중 증가 {a['selected_target_changes']['increased']}개, 감소 {a['selected_target_changes']['decreased']}개, 동률 {a['selected_target_changes']['unchanged']}개입니다. 전체 y≥.25/.5/.7/.85 개수는 baseline `{b['high_share_counts']}`, highshare `{h['high_share_counts']}`입니다. family·관측지원·requested peak SNR·실제 global SNR·분위수 전체는 [분포 JSON](R13_DATASET_CONSTRUCTION.json)에 있습니다.\n\n## 격리와 준비 상태\n\n모든 source는 기존 R9 Fold0 TRAIN에 속하며 기존 source별24h purge 증명을 계승합니다. 새 reference/source/split은 추가하지 않았습니다. 공통 validation_fit/gate/eval/high-SNR는 복사 없이 기존 R12 파일 path/SHA만 참조합니다. 이전 TRAIN-only basis의 binding과 CNN subset을 재사용합니다. Fold1..4 동적 fullscale 자료가 준비됐다는 뜻이 아니며 그 원천 검증은 별도입니다.\n\n`pipeline.dataset_r13.load_r13_arrays(folder,'baseline'|'highshare')`가 TRAIN dict를 반환하고 `validate_r13_bundle(folder)`가 manifest·source/file SHA·readiness·독립 물리 QA를 검사합니다. 모든 arm/validation/subset 경로는 project-relative, `file_sha256`만 bundle-relative입니다. 새 data hash는 `{m['data_hash']}`입니다.\n'''
    (ROOT/'reports/R13_DATASET_CONSTRUCTION.md').write_text(md)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fixture','audit','build','verify'])
    p.add_argument('--output',default='data/synthetic/r13_cpu_correction_v1');a=p.parse_args();sys.addaudithook(guard)
    # Allow bytewise SHA streams but prohibit parsing any fixed-role values.
    import numpy as np
    original_load=np.load
    def train_only_load(file,*args,**kwargs):
        if isinstance(file,(str,Path)) and Path(file).name.lower() in ('validation_fit.npz','validation_gate.npz','eval.npz','high_snr_long.npz','report.npz','outer_validation.npz'):
            raise PermissionError('R13 builder cannot parse validation/locked values')
        return original_load(file,*args,**kwargs)
    np.load=train_only_load
    from pipeline.dataset_r13 import baseline_inputs,audit_distribution,build_r13,verify_r13_dataset,write,sha
    if a.mode=='fixture':
        result=fixture();write(ROOT/'reports/R13_DATASET_NUMERIC_FIXTURE.json',result)
    elif a.mode=='audit':
        arrays,m,recipes,_,_=baseline_inputs();result={'source_control_hash':m['data_hash'],'source_train_sha256':m['train']['sha256'],
            'source_cap_statement':'R12 synthesize_r12 has no hard share cap; recipe requested peak SNR and observed global SNR differ by support/background/reference scaling.',
            'distribution':audit_distribution(arrays,recipes),'TRAIN_only':True,'Test_validation_model_read':False}
        write(ROOT/'reports/R13_DATASET_BASELINE_AUDIT.json',result)
    elif a.mode=='build':
        build_r13(a.output);result=reports(a.output)
    else:result=verify_r13_dataset(a.output)
    print(json.dumps({'status':result.get('status','audited'),'mode':a.mode,'output':a.output,'data_hash':result.get('data_hash')},ensure_ascii=False))
if __name__=='__main__':main()
