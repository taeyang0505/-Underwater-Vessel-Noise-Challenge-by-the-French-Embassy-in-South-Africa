"""Render the fixed CPU ablation's complete results, without policy selection."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=ROOT/'reports';result=json.loads((r/'R6_CPU_ABLATION_RESULTS.json').read_text());data=json.loads((r/'R6_DATASET_AUDIT.json').read_text());source=json.loads((r/'R6_SOURCE_SPLIT_AUDIT.json').read_text())
    cfg=json.loads((ROOT/'configs/r6_cpu_ablation_v1.json').read_text());out=r/'r6_figures';out.mkdir(exist_ok=True)
    names=[s['arm']['name'] for s in result['summaries']];cnn=names[:4]
    short={'r4law_legacy8':'R4 law / legacy8','r4law_relative6':'R4 law / relative6','r6law_legacy8':'R6 law / legacy8','r6law_relative6':'R6 law / relative6','r6law_legacy_tree':'R6 tree / legacy','r6law_relative_tree':'R6 tree / relative'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(11,6),layout='constrained')
    for ax,name in zip(axes.flat,cnn):
        rows=[json.loads(l) for l in (ROOT/'models/r6_cpu_ablation_v1'/name/'epochs.jsonl').read_text().splitlines()]
        ep=[x['epoch'] for x in rows]
        ax.plot(ep,[x['train_rmse'] for x in rows],label='TRAIN online RMSE',lw=1.3)
        ax.plot(ep,[x['inner_site_macro_rmse'] for x in rows],label='INNER site-macro',lw=1.3)
        ax.plot(ep,[x['inner_rmse'] for x in rows],label='INNER window RMSE',ls=':',lw=1.2)
        ax.set(title=short[name],xlabel='Epoch',ylabel='RMSE');ax.grid(alpha=.15);ax.legend(fontsize=7)
    fig.suptitle('CPU pilot: own-law INNER curves (different target laws; compare heldout panels)')
    fig.savefig(out/'01_cpu_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,(role,rows) in zip(axes,result['heldout_results'].items()):
        keys=names+['TRAIN_r6_mean'];vals=[rows[k]['rmse'] for k in keys]
        ax.barh(np.arange(len(keys)),vals,color=['#999999','#547ca3','#c09a68','#4b8c79','#aa8aaf','#667c56','#dddddd'])
        ax.set_yticks(np.arange(len(keys)),[short.get(k,k) for k in keys],fontsize=7);ax.invert_yaxis();ax.set(xlabel='Window RMSE',title=role.replace('_','\n'));ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('All frozen arms on common locked synthetic panels; not official Test')
    fig.savefig(out/'02_locked_panels.png',dpi=160);plt.close(fig)
    rows=[]
    for s in result['summaries']:
        m=s['best_inner_metrics'];epochs=str(s.get('epochs_run',s.get('best_iteration')));best=str(s.get('best_epoch',s.get('best_iteration')))
        rows.append(f"| {s['arm']['name']} | {best} / {epochs} | {m['rmse']:.6f} | {m['site_macro_rmse']:.6f} | {m['presence_auc']:.6f} | {s['elapsed_seconds']/60:.2f} |")
    scale=[]
    for name,d in data['diagnostics'].items():
        auc=d['high_band_fraction_presence_AUC'];av='고정 후 아래 진단' if auc is None else f'{auc:.6f}'
        scale.append(f"| {name} | {d['examples']:,} | {d['mean_share']:.6f} | {d['zero_fraction']:.4f} | {d['constant0_RMSE']:.6f} | {d['constant0_1_RMSE']:.6f} | {av} |")
    tables=[]
    for role,rr in result['heldout_results'].items():
        lines=[f'### {role}', '', '| 모델/대조 | RMSE | Site-macro RMSE | MAE | Bias | AUC |','|---|---:|---:|---:|---:|---:|']
        for key,m in rr.items():
            lines.append(f"| {key} | {m['rmse']:.6f} | {m['site_macro_rmse']:.6f} | {m['mae']:.6f} | {m['bias']:+.6f} | {m['presence_auc']:.6f} |")
        tables.append('\n'.join(lines))
    contrasts=[]
    for role,rr in result['heldout_results'].items():
        for a,b,label in [('r4law_legacy8','r4law_relative6','R4 생성 고정: 8→6채널'),('r6law_legacy8','r6law_relative6','R6 생성 고정: 8→6채널'),('r4law_relative6','r6law_relative6','6채널 고정: R4→R6 생성'),('r6law_legacy_tree','r6law_relative_tree','tree: legacy→relative')]:
            delta=rr[b]['rmse']-rr[a]['rmse'];contrasts.append(f'| {role} | {label} | {delta:+.6f} |')
    recovery=[]
    for v in source['site_retention']:
        recovery.append(f"| {v['site']} | {v['old_clean_train_windows']:,} | {v['old_boundary_excluded_train_windows']:,} | {v['train']:,} | {v['inner']:,} | {v['cross_sensor']:,} |")
    known=' / '.join(f'{role}: {source["background_role_counts"][role]:,}' for role in source['background_role_counts'])
    diag=[]
    for role,d in result['heldout_data_diagnostics'].items():
        diag.append(f"| {role} | {d['high_band_fraction_presence_AUC']:.6f} | {d['rounded_weight_target_gap_max']:.8f} |")
    total_seconds=sum(s['elapsed_seconds'] for s in result['summaries'])
    text=f'''# R6 구조 개편·Mac CPU ablation 결과

기준: 2026-10-26 KST. **소형 CPU 실험을 완료했다. 공식 Test 재평가·5-fold OOF·대형 모델 재개 결과는 아니다.** 모든 사전 정의한 모델과 평가 패널을 보고하며, 결과를 보고 seed·모수·모델 조합을 다시 고르지 않았다.

## 1. 요청 사항 반영과 수정한 전제

- absolute db/100, supervised q0, Bhat/E가 없는 relative6을 주 개편안으로 구현했다. legacy8은 원인 비교용 대조군에만 남겼다.
- 실제 boundary row에 해당하는 minute만 NaN으로 가렸다. 원본·R4/R5는 수정하지 않았다.
- 무 timewarp minute-first/짧은 실제 crop, 넓은 smooth EQ 및 low-pass를 추가하고, 다른 generator family·센서를 분리한 패널을 만들었다.
- feature와 pooling w를 모두 0.1dB 관측에서 계산했다. 알려진 성분에서 구한 원래 share/q는 변경하지 않았고 양자화 근사 오차를 별도 검사했다.
- **Benchmark 또는 Test 통계에 맞춘 target prior/SNR/요약 통계 matching은 수행하지 않았다.** Benchmark 예측이 확인되지 않아 RMSE²를 E[y²]로 역산할 수 없다. 공식 규정의 Test inference-only 조건을 유지한다.
- 높은 HF 비율의 AUC가 .55 이하가 되도록 조정하거나 상수 baseline을 .24~.27로 맞추지 않았다. 아래 실제 결과를 그대로 보고한다.
- 암기·과대추정·특정 사이트 오판이라는 인과는 이 실험 전에는 확정되지 않았다. 이번 작은 대조의 차이도 공식 LB 실패의 단일 원인 증명은 아니다.

구체적인 사전 Diff/실행 방침은 [R6_EXECUTION_PLAN.md](R6_EXECUTION_PLAN.md), 실제 숫자 설정은 [고정 config](../configs/r6_cpu_ablation_v1.json)에 있다.

## 2. 원천 분리와 QC 복구

원천 배경 수: **{known}**. 기존 Fold0 TRAIN/INNER에 허용됐던 원천만 사용했다. 예전 OUTER·예약 calibration/domain/sealed는 사용하지 않았다.

sensor_group 4 전체가 metadata hash로 선정된 cross-sensor다. 이 선택은 모델 성능·Test 지역 구성에서 정하지 않았다. 그 그룹은 St Francis에 해당하므로 이번 fit TRAIN에는 없고, 마지막 평가에만 등장한다. 이는 지금 실험에서의 holdout이며 과거 연구자가 전혀 보지 않은 새로운 코퍼스라는 의미는 아니다.

| Site | 예전 clean TRAIN | 예전 경계 제외 TRAIN | 새 TRAIN 원천 | 새 INNER 원천 | 새 cross-sensor |
|---|---:|---:|---:|---:|---:|
{chr(10).join(recovery)}

허용 pool에서 실제 후보 minute **4,356개**를 가렸고, 창 전체 삭제로 빠졌던 TRAIN **3,545개**를 복구했다. 추가 시간 경계 purge **97개**, 같은 센서 역할 간 support gap 최소 **24시간**을 확인했다. 후보 minute는 첫/끝 외의 내부 위치에도 있어 첫/끝 일괄 삭제는 하지 않았다.

TRAIN 참조 51 clips/9 groups, INNER 18/3, cross-generator 18/3, cross-sensor 7/2다. hull/day/UTC/SHA를 역할 간 분리했다. 새 독립 녹음이나 외부 음원을 추가 수집한 것은 아니다.

제약: Cape Recife/Riy Banks/Bird Island는 TRAIN에 있지만 이 source pool에서 INNER가 없는 site가 있다. 따라서 INNER macro가 모든 TRAIN site를 대표한다고 주장할 수 없다. cross-generator와 cross-sensor 패널도 가용 원천의 제한된 검증이다. [원천 감사](R6_SOURCE_SPLIT_AUDIT.md).

## 3. 생성·입력 설계

새 배율은 목표 y를 먼저 정해 역산하지 않는다. 변형한 전체 참조의 창 crop/mask 전 분 단위 peak가 배경 관측 분 broadband 중앙값 대비 SNR **Uniform[−10,+10]dB**가 되도록 한다. 이는 선언한 실험 범위이며 실제 대회 SNR의 추정이 아니다. EQ는 smooth tilt/곡선·low-pass를 합쳐 ±12dB 안에 둔다.

TRAIN/INNER 가족은 minute-first와 1~3분 native crop, locked generator 가족은 edge-straddling과 2초-first-no-warp다. 원 파워 궤적을 반복하거나 임의로 길게 늘리지 않는다. CPA가 창 밖이면 남은 tail을 다시 증폭하지 않는다.

배치는 관측 mask와 독립적인 기하학적 후보에서 한다. 참조가 미관측 구간에만 들어가면 관측된 추가 에너지는 0이고 y/presence도 0이다. 따라서 caller의 **no-source 20%**와 실제 관측 y=0 비율은 다를 수 있다. 이는 정답 조작이 아니라 관측 셀 에너지 정의를 그대로 적용한 결과다.

relative6: band median 상대 dB/10, band q20 상대 dB/15, 인접 관측 ΔdB/10, 저역≤200Hz 상승과 중고역≥300Hz 상승의 연속 동조, band median의 창 전체 상대 profile/20, 관측 mask. query의 true q/y나 site/recorder ID는 입력하지 않는다. profile과 energy w는 band별 정적 EQ에 불변이 아니므로 모든 장비 응답에 100% 불변이라고 부르지 않는다.

원래 정답을 q와 unrounded energy로 복원하는 검사는 통과했다. 반올림 energy로 같은 q를 pooling하면 약간 다를 수 있고, ±.05dB에 대한 수학적 최대 차이는 약 **.0057564**다. float32 저장 오차를 포함해 이 bound를 검사했다. [채널 fixture](R6_RELATIVE_FEATURES_FIXTURE.md), [지원·양자화 수정](R6_PREFIT_SUPPORT_CORRECTION.md).

## 4. 실제 저장한 CPU 예제의 분포

동일 source split에서 R4/R6 생성 law와 8/6채널을 분리해 비교했다. 총 **21,504개 역할 예제**(R4 대조 포함)를 저장했다. 핵심 R6 TRAIN은 8,192개다. TRAIN은 site 동수 quota를 사용하므로 작은 site의 원천은 반복된다. 이 숫자를 독립 21,504개 녹음으로 세지 않는다.

| 저장 세트 | n | mean share | y=0 비율 | 상수0 RMSE | 상수.1 RMSE | 단일 HF비율 AUC |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(scale)}

TRAIN/INNER에서는 단일 통계 AUC를 기술 진단으로 기록했다. locked 패널의 해당 진단은 모든 checkpoint 고정 전에는 계산하지 않았고 최종 평가 뒤에만 아래에 추가했다. 이 수치를 목표로 생성기를 재조정하지 않았다.

| locked 패널 | HF 비율 AUC | rounded-w의 원정답 복원 최대 차이 |
|---|---:|---:|
{chr(10).join(diag)}

새 생성기의 label 분산이 바뀌면 RMSE도 바뀐다. 따라서 서로 다른 INNER의 낮은 RMSE만으로 개선이라고 결론내리지 않고, 공통 locked target에 대한 결과와 상수 대조를 함께 본다.

## 5. 학습 설정과 실제 INNER 결과

CNN은 width32·3 residual blocks·dropout .1, relative6 **57,698 parameters**, legacy8 **58,274 parameters**다. Adam LR .001/weight decay .0003, 최대 **100epochs/ES7**, batch128, gradient clip1.0을 사용했다. loss는 share MSE + .25 energy-weighted cell MSE + .05 presence BCE다. epoch마다 고정 pilot TRAIN 조합을 셔플했다.

checkpoint 선택은 각 law의 source-separated INNER **site-macro RMSE**다. 공식 목적이 window RMSE라는 점을 유지해 둘을 모두 보고한다. tree 회귀는 최대1500trees/ES40 및 같은 macro 기준, presence tree는 INNER binary logloss로 선택했다. CPU feature·CNN/tree를 별도 프로세스로 분리했다. 학습 시간 합은 약 **{total_seconds/60:.2f}분**이다(사전 데이터 준비·최종 평가 시간 제외).

| Arm | best epoch/iteration / 실행 epoch 또는 best iteration | INNER RMSE | INNER site-macro | AUC | 분 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

![실제 CPU 학습곡선](r6_figures/01_cpu_curves.png)

R4 law control도 같은 새 source split·분 마스킹·양자화·tiny 구조를 사용했다. 예전 1.488M CNN과 53k 동적 TRAIN을 그대로 다시 돌린 재현 실험은 아니다. 채널 대조는 absolute/prior/상대 특징 여러 요소를 함께 바꾸므로 각각의 독립 기여량을 분해하지 않는다.

## 6. 모든 checkpoint 고정 뒤 공통 패널 평가

{chr(10).join(tables)}

![공통 locked 평가 비교](r6_figures/02_locked_panels.png)

모든 패널은 우리 합성 정답이다. cross-generator에는 원천 참조 변경도 포함돼 있어 순수한 연산자 하나의 효과만 분리한 실험은 아니다. cross-sensor와 joint 패널은 같은 배경 원천 draw를 공유하며 서로 독립 표본이 아니다. 공개 점수 예측으로 환산하지 않는다.

## 7. 사전 정의 대조의 RMSE 변화

아래 Δ는 **개편−대조**다. 음수면 같은 패널에서 RMSE가 낮아졌고 양수면 나빠졌다. 결과와 무관하게 모두 보고한다.

| 패널 | 대조 | Δ RMSE |
|---|---|---:|
{chr(10).join(contrasts)}

이 결과가 허용하는 결론은 선언된 원천·생성 조건·단일 seed·작은 CPU 예산에서의 비교다. 한 source collection의 결과로 실제 public RMSE .348의 원인 기여량을 확정하거나 최종 LB .90을 보장할 수 없다. 상수 대조보다 못한 패널이나 AUC가 낮은 경우도 그대로 보존한다. 결과를 본 뒤 대형 모델을 자동 재개하거나 이 패널에 맞춰 설정을 바꾸지 않는다.

## 8. 검증·재현과 남은 일

- 완료: 분 단위 QC/source 분리, 6채널·4생성 가족 수치 fixture, 네 CNN/두 LightGBM CPU fit, 모든 공통 패널 평가·가중치·예측·곡선 저장.
- 미완료: 완전한5fold production, 전체 배경 동적 합성 학습, 대형 backbone 성능 비교, 공식 제출 성능 검증.
- 최초 pre-fit 실패는 보존했다: Torch/LightGBM 스레드 교착과 sparse-observation edge geometry 오류. 실제 모델 성능을 보기 전에 수정했으며 실패 partial run을 학습에 재사용하지 않았다.
- Test·Benchmark 분포 matching, hidden source 추정, 공개 점수 calibration, SSH/GPU/제출은 하지 않았다. 기존 동결 R4/R5 core 17개 SHA도 최종 확인한다.

[기계 판독 전체 결과](R6_CPU_ABLATION_RESULTS.json) · [데이터 감사](R6_DATASET_AUDIT.json) · [고정 사전 계획](R6_EXECUTION_PLAN.md) · [source 검증](R6_SOURCE_SPLIT_AUDIT.verification.json) · [CPU 엔진 fixture](R6_CPU_RUNNER_FIXTURE.json)
'''
    (r/'R6_REBUILD_AND_CPU_ABLATION_REPORT.md').write_text(text)
    print('Wrote reports/R6_REBUILD_AND_CPU_ABLATION_REPORT.md')


if __name__=='__main__':main()
