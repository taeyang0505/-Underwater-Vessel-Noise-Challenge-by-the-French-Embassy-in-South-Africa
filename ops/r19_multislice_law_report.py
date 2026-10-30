"""Assemble the local R19 gate and paired-law diagnostic reports.

This assembler trains/infers nothing and never opens challenge query rows.
It consumes only the registered local diagnostic outputs.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import numpy as np
import pandas as pd
import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/r19_multislice_law_transfer_r1'
GATE = OUT / 'gate'
PANELS = OUT / 'law_panels'
INFER = OUT / 'inference'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def table(frame):
    return frame.to_markdown(index=False, floatfmt='.6f')


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec='seconds')
    selection = read(GATE / 'SELECTION_LOCK.json')
    manifest = read(PANELS / 'manifest.json')
    assert selection['selected'] == 'old_B1'
    assert manifest['total_rows'] == 4000
    post = GATE / 'postlock_all_candidates'
    summary = pd.read_csv(post / 'all_27_diagnostic_summary.csv')
    gm = pd.read_csv(GATE / 'gate_candidate_metrics.csv')
    rm = pd.read_csv(GATE / 'report_candidate_metrics_AFTER_LOCK.csv')
    chosen = ['old_B1', 'old_B2', 'new_B2']
    fields = ['panel', 'candidate', 'slice', 'n', 'rmse', 'bias']
    gate_head = gm[gm.candidate.isin(chosen) & gm['slice'].isin(['all', 'y=0', '0<y<=0.10'])][fields]
    report_head = rm[rm.candidate.isin(chosen) & rm['slice'].isin(['all', 'y=0', '0<y<=0.10'])][fields]
    preserved = pd.read_csv(PANELS / 'background_distribution_preservation.csv')
    physical = pd.read_csv(PANELS / 'law_target_stratified_physical_summary.csv')
    all_physical = physical[physical.target_bin.eq('all') & physical.feature.isin(['share', 'LF_energy_fraction', 'bump_high_db', 'bump_low_db'])]
    phys = all_physical.pivot(index='law', columns='feature', values='p50').reset_index()
    physical.to_csv(OUT / 'ALL_PHYSICAL_FEATURES_BY_TARGET_BIN.csv', index=False)
    exposure = read(PANELS / 'SOURCE_EXPOSURE.json')
    lf_identity = pd.read_csv(OUT / 'LF_MIXTURE_IDENTITY.csv')
    done = (INFER / 'COMPLETED.json').exists()
    law_section = '동결 모델 CPU 추론 진행 중. 결과를 관측하기 전에 패널·모델·비중을 고정했다.'
    if done:
        matrix = pd.read_csv(INFER / 'LAW_TRANSFER_MATRIX.csv')
        overall = matrix[matrix['slice'].eq('all')][['law', 'candidate', 'n', 'rmse', 'bias', 'score_AUC_zero_vs_slice']]
        weak = matrix[matrix['slice'].eq('0<y<=0.10')][['law', 'candidate', 'n', 'target_mean', 'prediction_mean', 'rmse', 'bias', 'score_AUC_zero_vs_slice']]
        paired = pd.read_csv(INFER / 'PAIRED_LAW_VS_VANILLA.csv')
        rows = pd.read_csv(INFER / 'LAW_TRANSFER_ROW_PREDICTIONS.csv')
        pooled = []
        for source, frame in [('six_custom_mixture', rows[~rows.law.eq('vanilla_addition')]), ('vanilla_addition', rows[rows.law.eq('vanilla_addition')])]:
            assert len(frame) == frame.sample_id.nunique() == 2000
            y = frame.target_share.to_numpy()
            for model in ['old_B1', 'new_B2']:
                for name, ix in [('all', np.ones(len(y), bool)), ('y=0', y == 0), ('0<y<=0.05', (y > 0) & (y <= .05)), ('0.05<y<=0.15', (y > .05) & (y <= .15)), ('y>0.15', y > .15), ('0<y<=0.10', (y > 0) & (y <= .10))]:
                    error = frame[model + '_share'].to_numpy()[ix] - y[ix]
                    pooled.append({'panel': source, 'candidate': model, 'slice': name, 'n': int(ix.sum()), 'rmse': float(np.sqrt(np.mean(error**2))), 'bias': float(np.mean(error))})
        pooled = pd.DataFrame(pooled)
        pooled.to_csv(OUT / 'POOLED_CUSTOM_VANILLA_METRICS.csv', index=False)
        law_section = f'''### 3.1 법칙별 동결 모델 평가

{table(overall)}

### 3.2 약한 양성의 오차와 분류 순위

아래 AUC는 해당 약한 양성과 같은 법칙의 무추가 음성을 합친 진단값이다. 양성만의 AUC를 계산한 것이 아니다.

{table(weak)}

### 3.3 같은 원천을 비교한 paired 결과

각 custom 법칙335/330행은 서로 다른 원천이다. 따라서 vanilla 전체2,000행과 직접 비교한 차이에는 표본 구성도 섞인다. 아래는 각 custom의 sample_id와 정확히 같은 vanilla 부분집합을 비교했다. 법칙마다 실현 y가 달라지므로 같은 y를 고정한 인과 비교는 아니다.

{table(paired)}

전체 custom2,000행과 vanilla2,000행은 같은2,000원천을 한 번씩 포함한다. 전체·요청한 y구간별 결과는 [POOLED_CUSTOM_VANILLA_METRICS.csv](POOLED_CUSTOM_VANILLA_METRICS.csv)에 보존했다.

{table(pooled[pooled['slice'].isin(['all', 'y=0', '0<y<=0.10'])])}

### 3.4  이번 대조에서 확인한 것

- Vanilla 전체는 old_B1 RMSE0.058544/AUC0.974641, new_B2 RMSE0.057083/AUC0.984966이었다. 그러나 같은 vanilla에서 약한 양성 RMSE는 old0.031304→new0.039411, bias는+0.009056→+0.015686, 무추가 RMSE는0.035854→0.044275로 악화했다. **인위적 EQ를 제거해도 새 모델의 저share 바닥 오차가 남는다.**
- 같은 ID의 무추가 관측·share·score 예측은 custom/vanilla 사이 최대차0이었다. 양성에 가한 법칙을 없애도 이미 학습된 모델의 음성 응답은 바뀌지 않는다. 이 비교는 EQ를 제거해 다시 학습했을 때의 효과를 시험한 것이 아니다.
- 같은 원천의 vanilla 대비 independent_subband_snr 전체 RMSE는 old0.058404→0.084878, new0.060272→0.089458로 크게 늘었다. alpha_preEQ_inserted도 old0.071446→0.081521, new0.064941→0.073885로 늘었다. 반면 shelf6/12 계열은 이 부분집합의 전체 RMSE를 줄인 경우도 있다. **6개 법칙을 모두 같은 방향의 실패 원인으로 묶을 수 없다.**
- 모든 법칙의 전체 AUC는 old0.956439~0.988305, new0.975379~0.993651였다. 이번 자연 원음 길이·SNR·배치 대조에서는 **공식 AUC약0.573 붕괴가 재현되지 않았다.** Vanilla가 공식 생성법을 대표하거나 실전 문제가 해결됐다는 결론은 내릴 수 없다.

[고정 vanilla-y 구간의 paired 오차](PAIRED_FIXED_VANILLA_BIN_EFFECTS.csv)도 별도 계산했다. 여기서는 vanilla의 y로 ID를 고정한 뒤 각 법칙의 실제 정답에 대해 오차를 비교한다. 법칙별 자기 y구간 비교와 다른 통계이며, 이 추가 진단으로 게이트 정책을 바꾸지 않았다.
'''
    text = f'''# R19 다중 구간 채택 게이트와 원음 합성 대조

갱신: {now}. 추가 학습0회. RunPod/GPU/MPS 사용0회. 작업1은 저장 예측 연산만, 작업2는 로컬 CPU의 동결 모델 추론만 수행했다. Test 행·라벨·저장 query 예측은 이번 후보 선별이나 법칙 평가에 읽지 않았다. 이전 보고서에 있는 집계 통계는 맥락 설명으로만 참고했다.

## 1. 채택 결론: old_B1 유지

사전 등록27후보를 검산했으며 **내용이 바뀐 후보는 모두 22축 중 일부에서 실패**했다. old_B1과 같은 출력인 δ=0, ρ=1만 원본과 함께 통과했다. 이 결론은 미리 정한 후보군에 대한 것이며 모든 가능한 후처리가 실패한다는 뜻은 아니다. 새 제출 CSV나 기존 공식 제출 파일은 만들거나 변경하지 않았다.

- old_B1: 기존 R17 B1, LightGBM60% + DINO25% + Dasheng15%, 각 D0/H0=50:50.
- old_B2: 같은 구세대 모델의50/30/20. 아직 제출하지 않았어도 이번 기준에서는 자동 채택 대상이 아니다.
- new_B2: R18의50/30/20. 모델 종류만 같을 뿐 입력·학습원천·학습범위가 다르다.

### 1.1 엄격한 기준과 선택 순서

local_gate·domain_holdout·native12/HF·native12/SF 각각 전체 RMSE, 무추가 RMSE/양의 편향, 약한 양성(0<y≤0.10) RMSE/양의 편향을 검사한다. native12 두 패널은 실제 활성≥12분 양성 RMSE도 추가하여 총22축이다. 이는22개의 독립 평가 데이터셋이라는 뜻은 아니다. 양의 편향은 max(mean(pred−y),0), 허용 수치오차는1e−6다. 한 축의 개선으로 다른 축의 악화를 상쇄하지 않는다. 기존 native12의 양성/음성50:50 가중치를 보존했다.

fit에서 보정 파라미터를 먼저 고정하고 gate에서 채택/기각한 뒤 SELECTION_LOCK을 저장했다. local_report는 그 후 최종 확인에만 사용했다. report와 gate는 과거 개발에도 사용됐으므로 새로운 blind test라는 보증은 없다. 신규 [게이트 API](../../pipeline/multislice_gate_r19.py)는 필수 패널·구간·배열이 누락되면 자동 기각한다. 기존 R18 finalizer/선택 정책은 역사 기록으로 동결하고, 향후 새 finalizer에서 이 모듈을 호출하도록 [연결 계약](gate/INTEGRATION_README.md)을 별도 저장했다.

{table(summary)}

### 1.2 전체·음성·약한 양성을 따로 보면

{table(gate_head)}

old_B2는 미관측/장시간 전체 오차 일부를 줄였지만 local_gate 음성·약한 양성 및 SF 음성 등에 손실이 생겨11/22축에서 실패했다. old_B1의 미세 절편0.0025와 softgateρ=.9도 각각8축·4축에서 실패했다. fit에서 선택된 보정은 결국 δ=0, ρ=1이었다. 단순히 예측을 낮추면 이미 과소예측하던 장시간 양성의 오차가 커졌다.

### 1.3 선택 동결 후 report 확인

{table(report_head)}

[27후보 전수 부록](gate/postlock_all_candidates/R19_ALL_REGISTERED_CANDIDATES_DIAGNOSTIC.md)에는 fit에서 제외된 보정까지 gate를 사후 진단한 결과를 보존했다. 이 추가 표로 선택 정책을 바꾸지 않았다. 개별 old/new 모델·D0/H0·50:50 전체 성적은 [개별 모델 지표](gate/individual_models_and_fixed_blends_all_regimes.csv)에 있다.

## 2. 2,000원천의 paired 소형 패널

vanilla2,000행(추가1,600·무추가400)과 custom6법칙 합계2,000행을 생성했다. 총4,000개의 관측 행이며 고유 배경 ID는2,000개다. 같은 녹음기·인접 시간의 상관이 있어 통계적 독립 표본2,000개라는 뜻은 아니다. 각 source ID에 vanilla1개와 custom1개를 배정했고 법칙 할당은 추론 전에 고정했다. custom별335/335/335/335/330/330행이며 각각 추가80%·무추가20%다. 초기 부분 생성 뒤 모델 결과를 보기 전에 최종4,000행 평가로 범위를 줄였으며 [범위 수정 이력](law_panels/PRE_INFERENCE_SCOPE_REVISION.json)을 보존했다. 동결 모델 추론은 최종4,000행에만 수행했다.

### 2.1 원음 합성의 정확한 의미

공식 `minute_band_power`와 검산된 기존 캐시에서 얻은 분별 선형 에너지를 사용했다. vanilla는 주파수별 EQ를 적용하지 않고 단일 스칼라α만 곱한 `E=B+αA`다. α는 원음 전체 peak 대비 요청 SNR[-10,+10]dB에서 정했다. **EQ를 없앴어도 α 분포·자연 crop·삽입 위치는 가정으로 남는다. 이것이 공식 숨은 Test 생성기라는 뜻은 아니다.** 길이15분 이하의 완전분은 자연 길이를 쓰고, 초과분은 연속15분만 자르며 늘리기·반복·평탄화·fade는 쓰지 않았다.

원음의 선택 구간별 대역 비중은 스칼라 곱 전후 최대5.55e−16 차이로 보존됐다. 모든 클립을 LF25/MF58/HF10 같은 고정 비율로 강제하지 않았다. LF는31.6~199.5Hz, MF는251.2~2511.9Hz, HF는3162.3~7943.3Hz로 정의했다. 관측 O와 배경 B가 달라지면 혼합물의 대역 비중은 원음과 다르다. 이번 custom post_EQ 범위는 현재 TRAIN과 같은±6dB이며 과거 일부 고정 패널의±12dB와 구별한다. 자연 전체 길이/최대15분 crop도 구 D0의 짧은 crop 분포와 동일하지 않다.

`E=B+A` 최대오차1.78e−15, raw y 적분식 오차0, 반올림 관측 w와 raw q를 함께 쓴 풀링 차이는 최대0.001473이었다. 원음 기록에는 ambient도 포함되므로 y는 새로 더한 기록 전체의 에너지 비율이다. 순수 선박 성분을 분리해 얻은 정답이라고 하지 않는다.

### 2.2 primary78869 배경 분포 보존

{table(preserved)}

R18 축소 배경에서 HF p90이낮아진 사실과 그 차이가 공식 실패의 원인이라는 인과 주장은 구분한다. 이번에는 primary78869의 사이트×녹음기×관측밀도 층별 비율을 근사 보존했다.

### 2.3 합성된 물리 특징 분포

아래는 각 법칙의 전체 중앙값이다. 0<y≤.05/.05<y≤.15/y>.15별 p10/p50/p90과 평균은 [전체 조건부 분포 CSV](ALL_PHYSICAL_FEATURES_BY_TARGET_BIN.csv)에 수록했다. custom끼리 다른 source 부분집합이므로 동일 원천에서 EQ·정규화 순서를 포함한 법칙 전체의 효과는 [동일 ID 물리 비교](law_panels/R19_PAIRED_PHYSICAL_EFFECTS.md)로 확인한다.

{table(phys)}

전체 custom의 HF bump 중앙값8.421692dB, vanilla8.600224dB였다. 이번 자연 원음·SNR·배치 조건에서는 EQ 제거만으로 HF 돌출이 대폭 늘지 않았다. 같은 ID에서 independent_subband_snr는 vanilla보다 HF bump 평균−4.0138dB/LF 비중+.0760, shelf12는 HF bump+1.2949dB/LF 비중−.03855였다. 여섯 법칙이 모두 같은 방향의 왜곡을 일으키지는 않는다.

### 2.4 평균 통계가 비슷해도 추가 성분의 서명은 달라진다

합성 정답을 아는 각 행에서 `LF(E)=(1−y)·LF(B)+y·LF(A)`가 성립한다. 여기 LF는 동일 관측 O로 적분한 저주파 에너지 비중이다. 독립 검산 최대 잔차는5.55e−16이었다. 이 식에 서로 다른 집합의 중앙값을 대입해 Test 정답을 역산하지 않는다.

{table(lf_identity)}

vanilla의 추가 성분 LF 비중 중앙값은0.250983인데 independent_subband_snr에서는0.837986, shelf12에서는0.025312, shelf12_lfminus6에서는0.006383이었다. **추가 성분의 스펙트럼 서명이 법칙에 따라 크게 바뀐다.** 특히 저/고대역을 각 배경 전력 기준으로 따로 정규화하는 법칙은 원음의 단순 볼륨 변경과 다르다. 서로 반대 방향의 효과가 섞여 전체 LF 중앙값이 비슷해 보여도, 같은 특징에서 기대하는 y의 조건관계는 다를 수 있다. 이 사실만으로 특정 법칙이 공식 실패의 주원인이라고 확정하지는 않는다.

## 3. Law-Transfer Matrix: 동결 모델의 CPU 평가

{law_section}

## 4. 해석의 범위와 다음 채택 조건

1. 원음 합성도 추가 비율·세기·배치·관측 조건을 정해야 하는 정방향 모델이다. EQ 없는 합성이 공식 정답 생성법이라는 근거는 없다. 이번 결과로 Test의 양성/음성 비율이나 숨은 라벨을 역산하지 않는다.
2. 이 패널의 배경은 구 R16 **Fold0 TRAIN 배경과1,563/2,000**, R18 TRAIN과300/2,000이 중복된다. 추가로 구 LightGBM Fold0~4의 TRAIN 배경 합집합을 검산하면 **2,000/2,000 모두 그중 일부 모델에 노출**됐다. 모든 개별 Fold가2,000개를 전부 봤다는 뜻은 아니다. [Fold별 원천 노출 검산](SOURCE_EXPOSURE_FOLD_SCOPE.json)에 근거를 보존했다. 참조465개도 R18에 노출됐다. 따라서 법칙 민감도 대조이며 독립 일반화 성능이나 공식 점수 예측이 아니다. old/new 모델의 학습량·Fold·특징·참조 다양성도 함께 다르다.
3. 법칙별 양성 target 분포가 달라진다. 같은 y 구간끼리도 포함된 원천이 달라질 수 있어, 동일 source의 paired 비교와 y구간별 비교를 따로 보고한다. 각 custom AUC는 음성66~67개에 근거하므로 작은 차이를 우열 확정으로 해석하지 않는다.
4. 새 패널은 동일 CPU FP32 경로로 모든 법칙을 추론했다. 과거 CUDA BF16 저장 출력과 bit-exact 재현은 아니며 local_gate32행×16모델 경로의 정밀도 차이를 receipt에 기록했다(회귀 최대0.003430, 분류 최대0.010637). 큰 분류 차이가 나온32행의 AUC는 같았지만 작은 차이는 신중히 해석한다. 이것은32행 probe의 결과이며 전체 패널의 CUDA 차이 상한을 보증하지 않는다. 기존 저장 예측을 쓰는 작업1 게이트는 이 정밀도 차이의 영향을 받지 않는다.
5. 후속 대규모 데이터 재생성이나 재학습은 실행하지 않았다. 후보는 먼저 위 multi-slice 기준을 통과해야 하며 통과하지 못하면 old_B1을 유지한다. 이 감사의 탐색 결과로 기존 report를 다시 최적화하거나 Test 예측 평균을 맞추지 않는다.

## 5. 산출물·실행 재현

- [저장예측 게이트 상세](gate/R19_MULTI_SLICE_GATE.md), [27후보 전수 부록](gate/postlock_all_candidates/R19_ALL_REGISTERED_CANDIDATES_DIAGNOSTIC.md)
- [동결 정책 JSON](gate/ADOPTION_POLICY.json), [미래 연결 안내](gate/INTEGRATION_README.md)
- [원음 패널·정답 수식 검증](law_panels/R19_VANILLA_PANEL_DIAGNOSTICS.md), [동일 원천 물리 비교](law_panels/R19_PAIRED_PHYSICAL_EFFECTS.md)
- [법칙별 전체 결과 CSV](inference/LAW_TRANSFER_MATRIX.csv), [동일 source 추론 비교](inference/PAIRED_LAW_VS_VANILLA.csv)
- [동결 CPU 추론·정밀도 상세](inference/FROZEN_LAW_INFERENCE_REPORT.md), [고정 vanilla-y 구간 paired 오차](PAIRED_FIXED_VANILLA_BIN_EFFECTS.csv)
- [독립 패널 정답 검산](INDEPENDENT_PANEL_VERIFICATION.json)

재현은 `.venv_r17_eda/bin/python` 환경에서 `ops/r19_multislice_gate.py`, `ops/r19_vanilla_law_panels.py`, `ops/r19_frozen_law_inference.py`, `ops/r19_multislice_law_report.py` 순서다. 이미 동결된 선택·프로토콜은 임의로 덮어쓰지 않는다. 모든 모델·원본·기존 제출물은 보존했다.
'''
    target = OUT / 'R19_MULTI_SLICE_AND_VANILLA_LAW_REPORT.md'
    target.write_text(text)
    css = 'body{max-width:1300px;margin:30px auto;padding:0 28px;background:#f5f8fb;color:#17334e;font:16px/1.75 system-ui}main{background:white;padding:30px;border:1px solid #d9e4ee;border-radius:14px}table{display:block;overflow-x:auto;border-collapse:collapse;font-size:13px}td,th{border:1px solid #d9e4ee;white-space:nowrap;padding:7px 10px}th{background:#eaf2fa}h1,h2,h3{color:#16466f}h2{margin-top:40px}a{color:#0966b7}code{background:#eff3f8;padding:2px 4px}'
    nav = '<p><b>R19 · 재학습 없음 · CPU 대조</b> · <a href="R19_MULTI_SLICE_AND_VANILLA_LAW_REPORT.md">MD 원문</a> · <a href="gate/R19_MULTI_SLICE_GATE.md">게이트</a> · <a href="law_panels/R19_VANILLA_PANEL_DIAGNOSTICS.md">물리 대조</a></p>'
    (OUT / 'START.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>R19 게이트·원음 합성 대조</title><style>' + css + '</style><main>' + nav + markdown.markdown(text, extensions=['tables','fenced_code']) + '</main></html>')
    if done:
        files = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ('ARTIFACT_SHA256.json', 'AUDIT_COMPLETED.json')]
        hashes = {str(p.relative_to(ROOT)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(files)}
        write_json(OUT / 'ARTIFACT_SHA256.json', hashes)
        write_json(OUT / 'AUDIT_COMPLETED.json', {'status':'completed','created_KST':now,'report':str(target.relative_to(ROOT)), 'report_sha256':sha(target), 'artifact_manifest_sha256':sha(OUT / 'ARTIFACT_SHA256.json'), 'files':len(hashes),'new_training':False,'CPU_frozen_inference_only':True,'GPU_MPS_RunPod_used':False,'new_submission_CSV':False,'selected_candidate':'old_B1','Test_row_or_query_prediction_read':False})
    print(json.dumps({'status':'completed' if done else 'inference_pending', 'report':str(target), 'created_KST':now}, ensure_ascii=False))


if __name__ == '__main__':
    main()
