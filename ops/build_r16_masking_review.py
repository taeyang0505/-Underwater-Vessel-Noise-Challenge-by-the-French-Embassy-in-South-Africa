"""Build a local, evidence-linked status page. Never load Test or model weights."""
from pathlib import Path
import datetime
import hashlib
import html
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/r16_pretrained_masking_r1'
OPS = ROOT / 'ops/r16_pretrained_masking_r1'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simple_markdown(text):
    """Small local renderer for this generated report's headings/lists/tables."""
    def inline(value):
        value = html.escape(value)
        value = re.sub(r'`([^`]+)`', r'<code>\1</code>', value)
        value = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', value)
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', value)
    lines = text.splitlines()
    parts = []
    i = 0
    in_list = False
    while i < len(lines):
        line = lines[i]
        if line.startswith('|'):
            if in_list:
                parts.append('</ul>'); in_list = False
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    tag = 'th' if not rows else 'td'
                    rows.append('<tr>' + ''.join(f'<{tag}>{inline(c)}</{tag}>' for c in cells) + '</tr>')
                i += 1
            parts.append('<table>' + ''.join(rows) + '</table>')
            continue
        if line.startswith('- '):
            if not in_list:
                parts.append('<ul>'); in_list = True
            parts.append('<li>' + inline(line[2:]) + '</li>')
        else:
            if in_list:
                parts.append('</ul>'); in_list = False
            heading = re.match(r'^(#{1,6}) (.*)', line)
            if heading:
                level = len(heading.group(1))
                parts.append(f'<h{level}>' + inline(heading.group(2)) + f'</h{level}>')
            elif line.strip():
                parts.append('<p>' + inline(line) + '</p>')
        i += 1
    if in_list:
        parts.append('</ul>')
    return '\n'.join(parts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = read(OPS / 'RUN_PLAN.json')
    split = read(OPS / 'LOCAL_FIT_SPLIT.json')
    receipts = []
    for family in plan['families']:
        for variant in plan['variants']:
            pf = read(OPS / f'PREFLIGHT_{family}_{variant}.json')
            assert pf['status'] == 'passed'
            assert pf['source_sha256'] == plan['source_sha256']
            assert pf['local_fit_split_sha256'] == plan['local_fit_split_sha256']
            for task, row in pf['models'].items():
                assert row['passed'] and row['frozen_base_unchanged']
                receipts.append({
                    'family': family, 'variant': variant, 'task': task,
                    'device': row['device'], 'microbatch': row['microbatch'],
                    'adaptation_parameters': row['adaptation_parameters'],
                    'receipt': f'../../ops/r16_pretrained_masking_r1/PREFLIGHT_{family}_{variant}_{task}.json',
                })
    assert len(receipts) == 12
    for relative, expected in plan['source_sha256'].items():
        assert digest(ROOT / relative) == expected, relative
    cat_state = read(ROOT / 'ops/r16_catboost_peft_cpu8_r1/state.json')
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')
    facts = {
        'updated_at_kst': stamp,
        'state': 'ready_neural_training_not_started',
        'neural_training_completed': 0,
        'preflight_branches': receipts,
        'neural_plan_sha256': digest(OPS / 'RUN_PLAN.json'),
        'local_fit_split_sha256': digest(OPS / 'LOCAL_FIT_SPLIT.json'),
        'sources_current': True,
        'catboost_state_snapshot': cat_state,
        'new_RMSE_AUC_comparison_available': False,
        'new_submission_csv_created': False,
        'Test_used': False,
    }
    (OUT / 'READINESS_FACTS.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2) + '\n')
    state = {
        'phase': 'ready_masked_neural_training_not_started', 'updated_at_kst': stamp,
        'run_plan_sha256': facts['neural_plan_sha256'], 'preflight_branches_passed': 12,
        'new_neural_production_started': False,
        'preflight_batch1_is_connection_test_not_tuned_production': True,
        'next_step': 'Select execution environment; benchmark production batch before long neural jobs. No GPU purchase or deployment.',
        'catboost_run': 'ops/r16_catboost_peft_cpu8_r1/state.json',
        'scratch_additional_folds_cancelled': True,
        'official_submission': False,
    }
    (OPS / 'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    md = f'''# R16 사전학습 PEFT + 입력 마스킹 전환

기록 시각: {stamp}. **구현과 실제 MPS 연결 검증 완료 / 새 신경망 본학습·성능 비교는 미실행 / CatBoost CPU 본학습 진행 중.**

이 문서는 현재 구현을 설명한다. 기존 `r16_training_current_v2.png`는 Scratch 실행의 과거 도식이며 새 사전학습 실행 도식으로 사용하지 않는다. 성능이 개선됐거나 대회 분포 간극이 해결됐다고 판정한 문서는 아니다.

## 1. 실제 완료 범위

| 항목 | 확인된 상태 |
|---|---|
| 기존 Scratch 기준점 | D0/H0 Wide CNN과 D0/H0 5-Fold LightGBM 완료분을 동결·평가 |
| 기존 Scratch 나머지 | FDY D0는 19 epoch 부분 실행, iTransformer·TFGrid 미시작. 5종 완성 앙상블 없음 |
| 새 공개 사전학습 모델 | DINOv2-small, Dasheng-Base, TimeMoE 공개 가중치 연결 |
| 새 마스킹 | `off / spec / profile / both` 구현. 기본 비교는 off와 both |
| 실제 Mac GPU 검사 | 3개 모델 × 회귀/분류 × off/both = 12개 분기, 각 3 optimizer step 통과 |
| 신경망 성능 실험 | 미실행. 위 disposable step은 본학습·RMSE 검증이 아님 |
| CatBoost | CPU8에서 D0/H0 × Fold0~4 회귀·분류 순차 본학습 중 |
| 새 PEFT 제출 CSV | 아직 생성되지 않음 |
| Scratch Fold1~4 | 추가 실행 보류 유지 |

현재 CatBoost 상태는 [실시간 상태 파일](../../ops/r16_catboost_peft_cpu8_r1/state.json)에서 확인한다. 본 문서 생성 당시 완료 쌍은 **{len(cat_state['completed_pairs'])}/10**이다.

## 2. 변경하지 않은 데이터·물리 규격

- Fold0 TRAIN은 **D0 61,446건, H0 61,446건**이다. 서로 다른 두 데이터 구성이고 독립 원천 122,892개를 뜻하지 않는다.
- 기존 5-Fold TRAIN 합계는 구성당 307,304행이며 Fold 사이 학습 원천 재사용이 포함된다.
- 참조는 **A′ 75클립·13개 연결 그룹**이다. 로컬 보유 509클립 전체를 새로 편입한 실행이 아니다. B19/C′25 참조는 분리 상태를 유지한다.
- 기존 R9 Fold0 TRAIN 배경 78,869창 안의 nested source 분할, 900초 support와 24시간 purge를 유지한다.
- D0는 자연 크롭·배치, H0는 가능한 원음 전체/긴 연속 support를 사용한다. 인위적 평탄 15분, 반복·시간 신축, highshare 강제 할당을 추가하지 않았다.
- 회귀 입력 `[N,17,15,25]`, 분류 입력은 원본 Ch4와 Ch7을 제외한 `[N,15,15,25]`다.
- 절대 `db/100`, `q0`, PCA 잔차, LF anchor는 새 입력에 없다.

추가 전 배경 B, 추가 성분 A, 관측 마스크 O에 대해 원래 정답은 `q=A/(B+A)`, `y=Σ(OA)/Σ(O(B+A))`이다.

모델은 `q_hat=sigmoid(mask_logits)`, `share_hat=Σ(w*q_hat)`을 사용한다. 입력·가중치는 기존 0.1 dB 관측 격자 기준이고 q/y는 원래 에너지 정밀도를 유지한다. 이 기존 양자화 잔차는 마스킹 전후 동일하며, 마스킹으로 라벨을 다시 정하지 않는다.

## 3. 입력 마스킹의 정확한 동작

| 기능 | 설정 | 보존되는 항목 |
|---|---|---|
| 시간·주파수 context 마스킹 | 샘플당 확률 0.5, 주파수 띠 1~2개·폭 1~2대역, 시간 폭 1~2분 | 원본 관측 채널 O, 별도 w/q/y/z |
| 가림 상한 | 관측 셀의 최대 30%, 최소 한 관측 분 문맥 유지 | 가려진 셀도 원래 회귀 정답과 손실에 포함 |
| 정적 프로파일 dropout | 회귀 원본 Ch4 전체를 확률 0.4로 0 처리 | 분류에는 원래 Ch4가 없으므로 이 dropout을 적용하지 않음 |
| 검증·추론 | augmentation 비활성, 원본 특징 그대로 | 입력이 비트 단위로 동일 |

원본17채널에서 먼저 가린 후 분류15채널을 선택한다. 원본 Ch5 관측 마스크를 가짜 결측으로 바꾸지 않는다. 입력의 `sqrt(w)` 특징을 문맥 가림에 포함하더라도 **풀링에 쓰는 별도 w는 원래 값**이다. 주변 셀에서 가려진 q를 추론하게 만드는 정규화이며 관측 에너지 자체를 제거하는 조작이 아니다.

실제 O를 지우는 observation dropout은 관측 문제와 타깃을 바꾸므로 별도의 w/y 재계산이 필요하다. 이번 입력-only 마스킹에는 섞지 않았다. 추론 시 낮은 확률을 이유로 q를 강제로 0으로 만드는 후처리도 기본값에 넣지 않았다.

`off`=가림 없음, `spec`=블록 가림, `profile`=Ch4 dropout, `both`=둘 다. 시드·배열 순서·초기화는 대응 비교에서 같게 유지하고 증강 RNG는 별도 기록한다. 마스킹 단독/프로파일 단독은 원인 분리용 선택 실행이며 현재 기본 본학습 계획에는 off/both만 고정했다.

검증 근거: [마스킹 세부 검증](MASKING_VALIDATION.md), [CPU 계약](../r16_pretrained_ft_r1/feature_masking_contract.json), [실제 MPS 계약](../r16_pretrained_ft_r1/feature_masking_contract_mps.json).

## 4. 사전학습 모델과 독립 회귀·분류 경로

| 모델 | 회귀 경로 전체 파라미터 | Stage2 학습 파라미터(회귀/분류) | 역할 |
|---|---:|---:|---|
| DINOv2-small | 22,278,913 | 241,537 / 236,546 | 공개 2D 사전학습 후보 |
| Dasheng-Base | 85,892,481 | 483,073 / 473,090 | 공개 오디오 사전학습 후보 |
| TimeMoE | 453,688,193 | 591,233 / 598,274 | 공개 시계열 MoE 후보 |

TimeMoE의 `200M`은 이 구현에서 로드한 전체 파라미터 수와 같지 않다. Dasheng은 현재 **Base**이며 1.2B로 표기하지 않는다. 공개 ConvNeXt-V2 가중치는 이용 조건 검토 후 이번 후보에서 제외했고 DINOv2-small로 대체했다.

375개 셀을 보존하는 stride1 입력 어댑터와 마스크 디코더를 연결한다. 기존 오디오/시계열 입력을 그대로 받는 사용법과 다르므로 사전학습 이득은 동일 자료 실측으로 확인해야 한다. 이 대회에서 이미 우월성이 입증된 SOTA라고 부르지 않는다.

회귀17채널과 분류15채널은 **서로 다른 Stem·LoRA·head·optimizer**를 사용한다. 분류 손실이 회귀 q를 바꾸지 않는다. 회귀와 독립 분류가 같은 5,041파라미터 CNN을 반복하는 구조를 사용하지 않는다.

백본 원래 가중치는 고정하되, 학습할 Stem/LoRA까지 gradient가 도달해야 하므로 학습 전체를 `no_grad()`로 감싸지 않는다. 12개 실제 MPS 분기에서 원래 백본 fingerprint 불변·유한 loss/gradient·업데이트 연결을 확인했다.

## 5. 학습 스케줄과 새 local_fit 역할 분리

| 단계 | 학습 대상 | 학습률/종료 |
|---|---|---|
| Warm-up 3 epoch | 새 Stem·Head, 원래 백본·LoRA 동결 | Adam 1e-3 |
| Stage2 최대25 epoch | Stem·Head + rank8 LoRA/허용 Norm | Adam 2e-4→1e-5 cosine, patience7 |
| 정규화 | weight_decay1e-3, dropout0.1, gradient clip1.0 | effective batch64 |

회귀 손실은 `MSE(share_hat,y)+0.25*Σw(q_hat-q)^2`, 독립 분류 손실은 `BCEWithLogits`다. 회귀 체크포인트는 선택 집합 RMSE 최소, 분류는 AUC 최대를 별도로 보관한다. Global minimum을 보장하지는 않는다.

| 역할 | 고유 배경 | 합성 행 | 용도 |
|---|---:|---:|---|
| local_fit checkpoint selection | 199 | 1,194 | 매 epoch ES·체크포인트 선택 |
| local_fit ensemble fit | 201 | 1,206 | 체크포인트 고정 후 capped 가중치 적합 |
| 각 D/H own-fold CV | 별도 기존 집합 | 1,000 | 새 NN에서는 체크포인트 고정 후 보고 |

동일 배경의 6개 연산자를 같은 역할에 배정했다. 두 local_fit 역할의 배경·그룹 중복은0, 최소 support 간격은 **218.75시간**이다. 모델 예측값·Test로 분할을 골라내지 않았다.

한계: 두 역할에 같은 센서와 외부 참조 B가 존재할 수 있다. 따라서 이 분할 자체가 완전한 unseen sensor/reference 검증은 아니다. Cape Recife는 한 배경 그룹뿐이라 두 역할 모두에 넣을 수 없다. 기존 LightGBM·Scratch 모델은 과거 local_fit 전체400배경으로 선택됐으므로 새 분할이 그 이력을 소급해서 제거하지 않는다. local_fit 점수는 개발용으로 명시하고, 미관측 사이트·별도 참조·채널 변형 패널을 별도 보고한다.

## 6. 비교 기준과 현재 실측

기존 완료 모델을 같은 고정 패널에 통과시켜 재측정한 기준점이다. 아래는 새 PEFT 성능이 아니다.

| 패널 | Scratch CNN D/H RMSE / 독립 AUC | 기존 Tree5 D/H RMSE / 독립 AUC |
|---|---|---|
| local_gate | 0.069555 / 0.941829 | 0.073714 / 0.965345 |
| local_report | 0.070474 / 0.954701 | 0.080636 / 0.966569 |
| unseen-site(domain_holdout) | 0.092270 / 0.892116 | 0.089171 / 0.957962 |
| native12 + HF 음성 | 0.111909 / 0.897657 | 0.122569 / 0.951631 |
| native12 + SF dense 음성 | 0.110210 / 0.976260 | 0.118238 / 0.973165 |

[완료 Scratch 기준점 상세](../r16_pretrained_ft_r1/SCRATCH_BASELINE_STATUS.md)에 각 모델·epoch·source SHA와 own-fold CV가 있다. 일부 패널은 과거 개발 중 열람했으므로 완전히 새로운 비공개 시험으로 취급하지 않는다.

CatBoost 첫 완성쌍 Fold0 D0는 own-fold CV **RMSE0.104045 / 독립 AUC0.924563**, 공유fit **0.100178 / 0.948294**, 10분18초, peak RSS6.375GiB다. 단일 쌍 결과이며 전체 Tree 앙상블 개선을 입증한 수치는 아니다. [CatBoost 실행 보고서](../r16_pretrained_ft_r1/CATBOOST_PRODUCTION_START.md).

## 7. 비교 실행·최종화 순서

1. 이미 완료된 CNN·Tree 기준점과 코드/데이터 SHA를 고정한다. 미완료 Scratch 모델을 완료로 간주하지 않는다.
2. Fold0 D0/H0에서 사전학습 모델 각각을 off/both 같은 원천·초기화·스케줄로 비교한다. 한 variant는3모델×2조건×2task=12작업, 기본 off/both는24작업이다.
3. 체크포인트 고정 후 own-fold CV, local_gate/report, unseen-site·별도 참조·native-long/HF를 보고한다. 필요할 때 spec/profile만 원인 분리한다. 새 NN 전체5-Fold는 이번 Fold0 비교 범위가 아니다.
4. 독립 ensemble-fit1206행에서 회귀·분류 정책을 따로 적합한다. **LightGBM5 최소50%, 각 NN family 최대25%, D0:H0=50:50** 유지. CatBoost는10쌍 모두 완료·해시 검증됐을 때 선택 편입한다. 개선이 없는 후보는 가중치0도 허용한다.
5. score는 독립 분류기 확률만 사용하고 share 가중치는0이다. 정책 고정 후 고정 패널 평가·Test 기술 통계·후보 CSV를 생성한다. Test로 가중치·정답·후처리를 튜닝하지 않는다.

현재 새 신경망은 **실제 MPS 연결 검사만 완료**했다. 1샘플 microbatch는 메모리·연결 확인용이며 긴 본학습 권장 배치가 아니다. 대형 MPS 본학습을 CPU Tree와 동시에 장시간 시작하지 않았다. 사용자가 GPU 임대를 검토 중이므로 실행 환경·실제 배치 속도 확인 후 본학습으로 넘어가야 한다. CUDA 속도는 아직 측정하지 않았고 GPU 구매·배포를 실행하지 않았다.

새 후보 CSV 예정 경로는 `submissions/r16_pretrained_masking_r1/both/submission_r16_pretrained_masked.csv`이다. **현재 존재하는 완성 파일로 안내하지 않는다.** 추론 기본값은 augmentation off이다. Mask TTA·q 강제삭제·평활화는 검증 없는 자동 적용 대상이 아니다.

## 8. 코드·재현 근거

- [새 학습 구현](../../pipeline/training_r16_pretrained_masked.py)
- [마스킹 구현](../../pipeline/r16_feature_masking.py)
- [실행 CLI](../../ops/run_r16_pretrained_masked.py)
- [독립 최종화·caps 검증](../../ops/finalize_r16_pretrained_masked.py)
- [고정 실행 계획](../../ops/r16_pretrained_masking_r1/RUN_PLAN.json)
- [local_fit 분할·한계](../../ops/r16_pretrained_masking_r1/LOCAL_FIT_SPLIT.json)
- [최종화 실패 조건 fixture](FINALIZER_FIXTURE_V1.json)
- [현재 준비 상태·12개 분기](READINESS_FACTS.json)

기존 실행 파일과 데이터셋을 덮어쓰지 않고 새 `r16_pretrained_masking_r1` 경로를 사용한다. 부분 실행은 자동 재사용하지 않으며, 완료 ledger가 모두 일치할 때만 건너뛴다. epoch별 latest/best와 optimizer·scheduler·RNG 상태를 저장하지만 부분 실행 자동 resume 기능을 보장하지는 않는다.

## 9. 공개 방법의 의미

[SpecAugment 원 논문](https://arxiv.org/abs/1904.08779)은 음성인식 입력의 시간·주파수 가림을 제안했다. 여기서는 파형·물리 정답을 바꾸지 않는 특징 정규화로 적용했다. [Dasheng 공식 프로젝트](https://github.com/RicherMans/Dasheng), [DINOv2 논문](https://arxiv.org/abs/2304.07193), [TimeMoE 논문](https://arxiv.org/abs/2409.16040)의 사전학습 영역은 서로 다르다. 사전학습이나 마스킹을 사용했다는 사실만으로 이 대회 일반화·수렴 시간·공식 점수 개선이 보장되지는 않는다.
'''
    (OUT / 'PIPELINE.md').write_text(md)
    try:
        import markdown
        body = markdown.markdown(md, extensions=['tables', 'fenced_code'])
    except ImportError:
        body = simple_markdown(md)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R16 PEFT + Masking</title><style>body{font:16px/1.75 -apple-system,BlinkMacSystemFont,sans-serif;color:#17212e;background:#f3f6fa;margin:0}main{max-width:1120px;margin:28px auto;padding:28px 38px;background:white;border-radius:14px}h1,h2{line-height:1.35}h2{margin-top:40px;border-bottom:1px solid #dce5ed;padding-bottom:10px}table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow:auto}th,td{border:1px solid #dce5ed;text-align:left;padding:9px 12px}th{background:#edf3f8}a{color:#155eaa}code{font-size:.9em;background:#edf3f8;padding:2px 4px}pre{white-space:pre-wrap}strong{color:#172f4b}</style><main>''' + body + '</main></html>'
    (OUT / 'START.html').write_text(page)
    print(json.dumps({'report': str(OUT / 'PIPELINE.md'), 'preflight_branches': len(receipts), 'source_pins_verified': len(plan['source_sha256']), 'catboost_pairs_snapshot': len(cat_state['completed_pairs'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
