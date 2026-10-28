"""Summarize frozen experiment results; no fitting, inference or query-data access."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/r16_support_temporal_r1"
ARMS = ("D0F0", "D0F1", "D1F0", "D1F1", "historical_R16_mixed")


def main():
    result = json.loads((OUT / "evaluation/RESULTS.json").read_text())
    qa = json.loads((OUT / "INDEPENDENT_RESULTS_QA.json").read_text())
    assert result["status"] == "completed" and qa["status"] == "passed"
    assert not result["small_gate_passed"]
    rows = {(r["panel"], r["arm"], r["model"]): r for r in result["rows"]}

    def row(panel, arm):
        return rows[panel, arm, "fixed60T40N"]

    lines = [
        "# R16 자연 support·시간 구조 2×2 실측 결과",
        "",
        "**결론: 시간 특징 단독은 주 교차 검증 RMSE를 개선했지만, 자연 길이+HF 음성 보강 결합안은 본체 적용 기준을 통과하지 못했다.** 네 조건의 소형 CNN을 각각 100epoch까지 학습했고, 독립 LightGBM 회귀·분류와 고정 결합을 평가했다. 기존 본체 CNN은 27epoch 체크포인트를 보존한 정지 상태다. 실패한 새 조합으로 본체를 재개하거나 제출 CSV를 만들지 않았다.",
        "",
        "이번 수치는 합성 로컬 평가다. Test 입력·정답·Test 유래 EDA 및 공식 제출을 사용하지 않았으며, 공식 LB 개선 또는 데이터 간극 해소를 뜻하지 않는다.",
        "",
        "## 1. 실행한 네 조건",
        "",
        "| 조건 | 데이터 | 특징 |",
        "|---|---|---|",
        "| D0F0 | 새 공통 A′ 원천, 기존 짧은 crop·원래 음성 | 기존 R16 상대 특징 |",
        "| D0F1 | D0와 동일 | 시간 지속성·곡률·동조 추가 |",
        "| D1F0 | 자연 길이 보존 + recorder/mask 일치 HF 음성 교체 | 기존 특징 |",
        "| D1F1 | D1과 동일 | 시간 지속성·곡률·동조 추가 |",
        "",
        "D0/D1은 각각 TRAIN 4,000건(양성 3,200·무추가 800), local_fit 2,400건이다. 양성의 참조 ID·원배경·관측 마스크·연산자·EQ·SNR 난수는 대응시켰다. D1의 HF 음성 배경 ID는 교체했지만 site/sensor/recorder/15bit 관측 마스크를 일치시켰다. D0 자체도 새 A′ 풀과 길이 층화 추첨을 사용하므로 **과거 R16 mixed와 같은 데이터는 아니다.** 과거 모델은 별도 historical 행으로 재평가했다.",
        "",
        "CNN은 Adam, 100epoch, dropout 0.1, early stopping 없음이다. 회귀·분류의 최적 체크포인트를 local_fit에서 각각 보관했다. Tree는 최대 600round/early stopping 40이다. 회귀 손실은 Share MSE + 0.25×에너지 가중 cell MSE이며, 독립 분류 BCE가 회귀 마스크 파라미터를 바꾸지 않는다.",
        "",
        "표의 결합은 사전 고정된 **share=0.6 Tree+0.4 CNN**, **score=0.5 Tree 독립 분류 확률+0.5 NN 독립 분류 확률**이다. score에 share를 넣거나 잠긴 평가 결과로 가중치를 맞추지 않았다.",
        "",
        "| 조건 | CNN 전체 epoch | 최적 회귀 epoch | 최적 분류 epoch | 회귀+독립 분류 파라미터 | 조건별 실행 초 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS[:4]:
        folder = ROOT / "models/r16_support_temporal_r1" / arm
        c = json.loads((folder / "COMPLETED.json").read_text())
        assert len((folder / "epochs.jsonl").read_text().splitlines()) == 100
        lines.append(f"| {arm} | {c['epochs']} | {c['best_regression_epoch']} | {c['best_classifier_epoch']} | {c['parameter_counts']['total']:,} | {c['elapsed_seconds']:.1f} |")
    lines += [
        "",
        "네 조건은 Mac CPU에서 4프로세스로 병렬 실행했다. 표의 초를 직렬 합산한 시간이 실제 대기시간은 아니다. 동결 후 전 평가 추론은 269.7초였다. D1 분류기의 최적 epoch가 1이었다는 사실도 숨기지 않는다. 100epoch를 모두 실행했지만 뒤의 체크포인트가 더 낫지는 않았다.",
        "",
        "## 2. 자료 공백은 실제로 보완했는가",
        "",
        "| 항목 | D0 | D1 |",
        "|---|---:|---:|",
        "| 양성 실제 활성 분 중앙값 | 3분 | 5분 |",
        "| 활성 ≥12분 양성 | 0/3,200 | 500/3,200 (15.625%) |",
        "| 전체 TRAIN share 평균 | 0.145799 | 0.194907 |",
        "| 전체 TRAIN share RMS | 0.235432 | 0.298823 |",
        "| HF 상위 q90 음성으로 교체 | 0/800 | 798/800 (99.75%) |",
        "",
        "인위적 평탄15·시간 신축/반복·추가 fade·highshare 강제 할당·PCA/LF anchor·절대 dB를 도입하지 않았다. 실제 1분 참조 전력의 자연 전체 길이와 연속 11~15분 구간을 사용했다. 시간 특징은 rolling 2/4/8, lag 1~3, 시간 곡률·지속성·대역 간 시간 상관/고유값이며 고정 대역 offset에 불변이다. 추가 9개 격자 채널과 1,333개 통계 특징을 구현했다. 창의 결측을 압축해 서로 떨어진 분을 인접 분으로 취급하지 않는다.",
        "",
        "**HF 음성 ‘보강’을 이번 구현에서는 사실상 전량 교체로 적용했다.** 무추가 800건 중 798건이 TRAIN 내부 HF q90 초과 배경이 되어, 일반적인 조용한 음성 형태가 거의 사라졌다. 사이트·마스크를 맞춘 것과 음성의 시간 변화 분포를 유지하는 것은 별개의 문제다. 이 강도는 과도했으며 최종 데이터 정책으로 채택하지 않는다. 동시에 자연 support 증가로 타깃 평균·RMS도 달라졌다. 다만 두 변경이 D1에 묶여 있으므로 이번 결과만으로 어느 하나의 단독 인과효과를 확정할 수 없다.",
        "",
        "원천 분할은 A′75클립·13컴포넌트 / B19·5 / C′25·4다. 장시간 컴포넌트 전체 9클립을 TRAIN으로 이동하고 다른 실제 12분 컴포넌트를 평가에 남겼다. 전체 509개 그래프는 한 컴포넌트이므로, 독립성은 bridge 원천을 제외한 선택 119클립 그래프에 조건부다. **독립 ≥15분 평가 원천은 확보되지 않았으며**, 새 장시간 평가는 단 하나의 실제 12분 참조에 한정된다. 기존 R11 native15는 새 모델에 대해 참조를 공유하는 진단이다.",
        "",
        "## 3. 주 교차 검증: 시간 특징은 이득, 데이터 결합은 악화",
        "",
        "각 셀은 RMSE / 독립 분류 score ROC-AUC다. 각 축은 400개 배경 조합에 6개 정방향 연산자를 적용한 2,400행이며, 같은 조합의 변형들이므로 2,400개 독립 원천이라는 뜻은 아니다.",
        "",
        "| 조건 | 별도 배경 | 별도 참조·크롭 C′ | 미관측 사이트 |",
        "|---|---:|---:|---:|",
    ]
    primary = ("local_gate__POOLED", "local_report__POOLED", "domain_holdout__POOLED")
    for arm in ARMS:
        cells = [f"{row(p,arm)['rmse']:.6f} / {row(p,arm)['auc_classifier']:.6f}" for p in primary]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "D0F1은 통제군 D0F0 대비 세 축의 RMSE를 모두 낮췄다. 반면 D1F1은 세 축의 분류 AUC가 약 0.12~0.13 하락했다. 이 표의 고정 CNN+Tree 결합과 과거 보고서의 CNN 단독 수치(0.0677/0.0580/0.0857), 또는 과거 C 분할의 수치를 동일 실험처럼 비교하면 안 된다.",
        "",
        "## 4. 원천 독립 native12 통과 vs 자연 HF/SF 음성",
        "",
        "양성 2,400행(실제 12분 참조 1개×400개 배경×6연산자), HF 음성 359건, SF dense 음성 168건이다. 다음 RMSE와 예측 평균은 클래스별 총 가중치를 0.5씩 준 균형 수치다. 실제 행 비율 RMSE와 share 단독 AUC도 원본 JSON/CSV에 함께 저장했다.",
        "",
        "| 조건 | native12 vs HF RMSE / score AUC | native12 vs SF RMSE / score AUC | HF 음성 score 평균 | SF 음성 score 평균 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        hf = row("native12_VS_hf_high_negative", arm)
        sf = row("native12_VS_st_francis_dense_negative", arm)
        lines.append(f"| {arm} | {hf['rmse']:.6f} / {hf['auc_classifier']:.6f} | {sf['rmse']:.6f} / {sf['auc_classifier']:.6f} | {hf['negative']['classifier_mean']:.6f} | {sf['negative']['classifier_mean']:.6f} |")
    lines += [
        "",
        "D1F1은 native12 양성 예측 평균을 0.185973→0.223796으로 높여 실제 평균 0.279177에 가까워졌다. 양성 bias는 −0.093205→−0.055381로 줄었다. 그러나 독립 분류기는 HF 음성 확률을 0.671302→0.797636으로 오히려 올렸고, SF 음성도 0.425833→0.810522로 올렸다. **장시간 회귀 RMSE 개선과 추가 유무 순위 개선은 동시에 달성되지 않았다.** D0F1도 장시간 RMSE는 악화되어 ‘시간 특징만으로 모든 조건 해결’이라고 할 수 없다.",
        "",
        "## 5. 같은 배경으로도 분류 실패가 남는가: 독립 사후 검산",
        "",
        "새 추론·학습 없이 저장 예측을 배경 ID로 맞춰 재집계했다. 패널 전체의 양성/음성 사이트 비율이 다르다는 교란을 추가 점검한 기술 통계이며, 이 결과로 모델을 다시 선택하거나 가중치를 적합하지 않았다.",
        "",
        "| 대조 | D0F0 score AUC | D1F1 score AUC | 표본 구성 |",
        "|---|---:|---:|---|",
    ]
    conditional = qa["same_background_and_SF_conditional_diagnostics"]
    selections = (
        ("hf_high_negative", "same_background_ID", "같은 HF 원배경 ID", "33개 배경: 양성198/음성33"),
        ("st_francis_dense_negative", "St_Francis_only", "St Francis 사이트 내", "양성126/음성168"),
        ("st_francis_dense_negative", "same_background_ID", "같은 St Francis 원배경 ID", "21개 배경: 양성126/음성21"),
    )
    for negative, scope, title, count in selections:
        cells = []
        for arm in ("D0F0", "D1F1"):
            item = next(x for x in conditional if x["negative"] == negative and x["scope"] == scope and x["arm"] == arm and x["model"] == "fixed60T40N")
            cells.append(f"{item['auc_classifier']:.6f}")
        lines.append(f"| {title} | " + " | ".join(cells) + f" | {count} |")
    lines += [
        "",
        "같은 SF 배경 21개에서 D0F0은 양성/음성 score 평균 0.660775/0.464333이지만 D1F1은 0.804340/0.813812로 역전했다. D1F1의 TPR/FPR@0.5가 모두 1.0이다. **실패는 pooled 사이트 구성 차이만으로 설명되지 않는다.** 양성 prior 쪽으로 확률이 쏠리는 현상이 CNN·Tree 양쪽에서 확인된다. 자연 HF가 모두 단발성이고 시간 특징으로 100% 분리된다는 주장은 이 실험에서 입증되지 않았다.",
        "",
        "## 6. 사전 기준 판정과 본체 상태",
        "",
        "평가를 읽기 전에 [PROTOCOL](PROTOCOL.md)에 고정한 조건대로 D1F1 대 D0F0를 판정했다. 세 주 축 RMSE 악화 ≤max(0.005,5%) 및 AUC 하락 ≤0.02, native12 대 HF/SF AUC 각각 ≥0.75, HF 음성 분류 확률의 절대0.10 또는 상대25% 감소를 요구했다.",
        "",
    ]
    for c in result["checks"]:
        lines.append(f"- `{c['name']}`: {'통과' if c['passed'] else '미달'}")
    lines += [
        "",
        "**6개 조건 모두 미달이다.** 따라서 수정 데이터·특징 결합안을 본체에 적용하지 않았다. 기존 R16 Tree 5-Fold 및 CNN 27epoch 체크포인트는 보존했고, FDY/iTransformer/TFGrid 추가 본학습과 새 공식 제출 CSV 생성을 시작하지 않았다. 평가를 본 뒤 기준을 완화하거나 score를 share와 재혼합하지 않았다.",
        "",
        "이번에 확보한 부분 이득은 D0F1의 주 교차 RMSE 감소와 D1의 자연 장시간 회귀 개선이다. 남은 핵심 문제는 **길이 보존의 효과와 거의 전량 HF 음성 교체의 효과를 분리하지 못했다는 점**, 그리고 변경 데이터에서 독립 분류기가 일반 음성을 구분하지 못한다는 점이다. 다음에 추가 실험을 한다면 일반 음성을 유지한 제한적 HF 보강과 support-only 조건을 분리하는 것이 우선이다. 현재 관문 결과를 성공으로 바꾸기 위한 추가 탐색은 이 실행에서 하지 않았다.",
        "",
        "## 7. 무결성·재현성과 한계",
        "",
        f"- 독립 검산: 동결 artifact {qa['frozen_artifact_SHA_verified']}개 SHA, 예측 NPZ {qa['prediction_SHA_verified']}개 SHA 일치. 평가 {qa['evaluation_metric_rows_independently_recomputed']}행을 다시 계산한 최대 절대 오차 {qa['maximum_metric_absolute_error']:.3e}.",
        "- 900초 창 support 기준 24시간 purge와 선택 원천 분리 유지. E=B+A, q=A/E, y=ΣOA/ΣOE의 raw 풀링 오차 ≤4.44e−16. 입력만 0.1dB 반올림, raw 정답 보존. 반올림 에너지 풀링과 raw y 차이는 D0≤0.001632/D1≤0.001347이며 선언한 수치 상한 내다.",
        "- source20개/data artifact58개 SHA pin 및 numeric NPZ 전수 검사 통과. 소형 새 데이터 폴더 약165MB. 생성·연산자·paired 행 수를 독립 관측 수로 세지 않았다.",
        "- 평가 전용 첫 두 시도는 과거 R11 파일의 presence/cell_share 생략 규격 처리에서 실패/정지했다. 최종 adapter는 presence를 y>0으로 복원하고, q가 없는 파일은 모든 y=0임을 확인한 경우에만 q=0으로 복원한다. 양성 q 누락은 오류로 유지했다. 실패본과 로그를 별도 attempt 폴더에 보관하고 최종 평가는 모든 예측을 처음부터 다시 계산했다. 학습 가중치·정책·실패 partial 예측은 재사용/변경하지 않았다.",
        "- 장시간 독립 참조는 컴포넌트1개, 실제12분 클립1개뿐이다. 수천 개 합성 조합을 수천 개 독립 장시간 원천으로 해석할 수 없고, reference-level 일반화 신뢰구간을 주장하지 않는다.",
        "- 이 실험은 1회 고정 seed/소형 모델 비교다. 정답 없는 실제 Test 전달성을 확정하거나 Public 점수를 예측하지 않는다. checkpoint 선택용 fit과 평가용 gate/report/domain을 구분했다.",
        "",
        "## 8. 상세 파일",
        "",
        "- [전 조건·개별 NN/Tree·연산자·사이트 수치 CSV](evaluation/metrics.csv)",
        "- [전체 예측 지표·사전 관문 JSON](evaluation/RESULTS.json)",
        "- [독립 결과 검산 보고서](INDEPENDENT_FINDINGS.md) · [재계산 영수증](INDEPENDENT_RESULTS_QA.json)",
        "- [데이터 생성·분할·support 전체 설명](DATA_PLAN_AND_LIMITS.md)",
        "- [사전 고정 실험안](PROTOCOL.md) · [코드·모델·데이터 동결](GLOBAL_FREEZE.json)",
        "- [실행 현황 HTML](index.html)",
    ]
    path = OUT / "R16_SUPPORT_TEMPORAL_ABLATION.md"
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": "summarized", "path": str(path), "gate_passed": result["small_gate_passed"]}))


if __name__ == "__main__":
    main()
