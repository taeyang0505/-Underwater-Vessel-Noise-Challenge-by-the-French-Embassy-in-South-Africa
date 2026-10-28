"""Render the read-only execution audit without reading challenge Test values."""
from pathlib import Path
import html
import json

ROOT = Path(__file__).resolve().parent
esc = html.escape
audit = json.loads((ROOT/'reports/frozen_inference_audit.json').read_text())
policy = json.loads((ROOT/'models/ensemble_foundation/postprocess.json').read_text())

def table(head, rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+esc(str(x))+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'

paths = [
    ['제출 아티팩트', '공식 제출 CSV를 파싱하지 않고 바이트 SHA만 비교. 정책 freeze와 제출 manifest의 SHA가 일치.', '확인'],
    ['기본 7개 모델', 'base_run_name으로 기존 checkpoint / LightGBM text / prototype를 참조. 새 ensemble 폴더의 빈 경로를 읽는 누락 없음.', '확인'],
    ['CNN / FDY / iTransformer', '학습과 같은 cnn_channels 및 energy weights, 저장된 state, eval, inference_mode 사용.', '확인'],
    ['TabM', '학습과 같은 fixed_features, sigmoid 후 멤버 평균, 같은 checkpoint 사용.', '확인'],
    ['LightGBM 3개', '원래 fixed / physics / prototype 특징과 저장된 booster 사용. 별도 프로세스로 Torch/OpenMP 충돌 회피.', '확인'],
    ['TabICL', '28,544,991개 파라미터의 vendor 가중치 고정. 학습 context 16,000행, 특징 96개, 4멤버. 저장 context index 재사용.', '확인'],
    ['Kumo Large', '215,683,191개 파라미터의 vendor 가중치 고정. 학습 context 4,096행, 특징 96개, 4멤버. seed42 context 추출 재현.', '확인'],
    ['앙상블 / 보정', '정책 models 순서대로 share와 score를 쌓고 동일 weights 적용. share에만 저장 affine 및 [0,1] clipping 적용.', '확인'],
    ['진단 score', 'Foundation score_added_vessel은 share 복사 proxy. 별도 보정된 presence probability가 아님. 현재 primary share loss에는 쓰이지 않음.', '한계 명시'],
    ['CSV 매핑', '고정 m01…m15 / 25개 공통 대역 순서로 reshape하고 입력 ID 순서를 유지하는 코드 경로 확인.', '정적 코드 확인'],
]
history = []
for row in audit['small_neural_training_evidence']:
    history.append([row['model'], row['epochs_recorded'],
                    str(row['tensors_different_from_seeded_initialization'])+'/'+str(row['floating_state_tensors']),
                    f"{row['parameter_L2_change_relative_to_initialization']:.4f}",
                    f"{row['first_validation_RMSE']:.6f}",
                    f"{row['best_validation_RMSE']:.6f}",
                    f"{row['last_validation_RMSE']:.6f}"])
active = [[name, f'{weight*100:.6f}%'] for name, weight in zip(policy['models'], policy['weights']) if weight > 1e-8]
err = audit['replay_errors']
sections = [
    '<header><small>2026-10-24 · READ-ONLY EXECUTION AUDIT</small><h1>추론 연결은 재현됐고, 학습 충분성은 별도 문제입니다</h1><p>제출 이후 구현 누락 가능성을 검토한 독립 감사입니다. 대회 Test 값·분포, 봉인된 NOAA 자료와 공개 점수는 모델 선택이나 보정에 사용하지 않았습니다.</p></header>',
    '<section><h2>확인된 결론</h2><p><strong>현재 9개 후보의 추론을 연결하지 않았거나 저장된 정책이 적용되지 않았다는 증거는 발견하지 못했습니다.</strong> 자체 validation의 고정 8개 위치에서 전체 추론 함수를 다시 실행해 저장된 예측으로 계산한 결과와 비교했습니다. share 최대 절대오차는 '+f'{err["share_max_abs_error"]:.3g}'+', score는 '+f'{err["score_max_abs_error"]:.3g}'+f'였고 {audit["elapsed_seconds"]:.1f}초가 걸렸습니다. 허용 오차 4×10⁻⁵보다 작습니다.</p><p>이는 저장 모델→특징→문맥→앙상블→보정의 실행 일치 검증입니다. 8개 자체 검증 예시의 재현은 전체 실해양 분포에서의 정확성·완전 수렴·대회 SOTA를 입증하지 않습니다. CSV의 ID/형식/범위 검증은 root의 별도 산출물 검증에 기록됐으며 이번 감사에서는 원시 Test나 제출 예측 분포를 분석하지 않았습니다.</p></section>',
    '<section><h2>코드 경로 확인</h2>'+table(['구간','실제 동작','판정'],paths)+'</section>',
    '<section><h2>소형 신경망의 학습 실행 증거와 남은 한계</h2>'+table(['모델','기록 epoch','초기값과 다른 float state','상대 float-state L2 변화','첫 검증 RMSE','최선 검증 RMSE','마지막 검증 RMSE'],history)+'<p>동일 seed의 초기 state와 저장 checkpoint를 비교했고 학습 이력도 함께 확인했습니다. float state에는 일부 running buffer가 포함됩니다. 이 증거는 초기 가중치 그대로 제출한 상태가 아님을 뒷받침합니다. 그러나 기존 학습은 8 epoch이며, CNN·FDY·iTransformer는 마지막 epoch가 기록 중 최선입니다. 더 학습하면 개선될 가능성은 남지만 개선량이나 실전 전이를 단정할 수 없습니다.</p><p>앞으로의 gradient 학습은 사용자 지시대로 <strong>최대 100 epoch / early stopping 7 / Adam</strong>을 사용하고, 독립 원천 validation의 최선 checkpoint를 복원합니다. 이 조건으로 아직 v1 제출 모델을 재학습한 것은 아닙니다.</p></section>',
    '<section><h2>사전학습 모델의 크기와 실제 학습 방식</h2><p>TabICL 28.5M과 Kumo 215.7M은 작은 from-scratch 신경망과 다르게 이미 사전학습된 모델입니다. 이번 실행은 가중치를 고정한 in-context prediction이므로 대회 데이터로 gradient fine-tuning하지 않았습니다. fit은 context·변환·캐시 준비입니다. 그러므로 frozen ICL에 epoch나 Adam을 적용했다고 표시해서는 안 됩니다.</p><p>현재 최고 벤치마크 모델을 가져왔다는 사실만으로 이 대회의 SOTA라고 부를 근거는 없습니다. 1B 이상의 모델도 과제·입력 표현·가중치 이용 조건과 검증 결과가 맞으면 비교할 수 있습니다. 현재의 15×25 대역 에너지 입력을 원시 오디오 기반 대형 모델에 곧바로 넣을 수 있다고 가정하지 않습니다. 크기를 키우는 선택은 독립 원천 검증과 학습 곡선에서 판단해야 합니다.</p></section>',
    '<section><h2>9개 후보와 실제 비중</h2>'+table(['0보다 유의하게 큰 비중의 모델','동결된 비중'],active)+'<p>9개 후보를 평가하고 추론 코드도 실행하지만, 최종 share에 유의하게 기여하는 모델은 4개입니다. 나머지는 비중 0 또는 부동소수점 수준입니다. 그림에 모든 후보를 그리는 것과 모든 모델이 최종 예측에 기여하는 것은 다릅니다. 실제 실행의 기준은 <code>models/ensemble_foundation/postprocess.json</code>입니다. 이 감사에서 비중이나 보정은 바꾸지 않았습니다.</p></section>',
    '<section><h2>성능 문제에 대해 이 감사가 말할 수 없는 것</h2><p>실행이 맞아도 합성 학습의 데이터 생성 과정이 실제 평가와 다르면 성능이 낮을 수 있습니다. 현재 합성 자료의 정답은 의도적으로 추가한 에너지 비율이며, 원래 관측된 선박의 에너지 정답이 아닙니다. 원천 그룹이 적고, 짧은 음원을 15분 자료로 확장하는 포락선, 추가 비율 prior, 해역·센서 응답 등이 실제 관측과 어긋날 가능성이 있습니다. 이것들은 점검할 가설이며 단 한 번의 공개 점수로 원인을 확정하지 않습니다.</p><p>v2에서는 실제 연속 15분 원천, 센서 그룹/시간 purge, 원천 단위 보정 검증과 독립 training-source probe를 통해 이 가설들을 검증해야 합니다. 공개 점수에 맞춘 후처리, 테스트 집단 분포를 이용한 보정, 봉인 평가의 반복 튜닝은 하지 않습니다. 데이터가 충분하다거나 100 epoch가 해결을 보장한다고 주장할 수 없습니다.</p></section>',
    '<section><h2>재현 기록</h2><p>정책 SHA-256: <code>'+esc(audit['policy_sha256'])+'</code><br>제출 CSV SHA-256: <code>'+esc(audit['submission_csv_sha256'])+'</code><br>자체 검증 인덱스: <code>'+esc(str(audit['own_validation_indices']))+'</code></p><p>실측 JSON: <a href="frozen_inference_audit.json">frozen_inference_audit.json</a> · 실행 코드: <a href="../pipeline/audit_frozen_inference.py">audit_frozen_inference.py</a> · 이후 모델 계획: <a href="model_scale_plan.html">model_scale_plan.html</a></p><p>이 감사는 기록된 제출 시점의 코드 hash가 일치하는 snapshot에서 완료됐습니다. 이후 새 학습 설정 변경은 기존 제출의 epoch·optimizer 기록을 소급 변경하지 않습니다. 별도 validation metadata의 submitted=false 잔여값은 현재 제출 manifest의 submitted=true 및 공식 제출 ID와 구분해야 하는 문서 상태 문제이며, CSV hash나 모델 추론을 바꾸지는 않습니다.</p></section>',
]
style = 'body{font:16px/1.75 -apple-system,BlinkMacSystemFont,sans-serif;background:#f5f7f3;color:#17241d;margin:0}header,section{max-width:1160px;margin:24px auto;padding:30px;background:white;border:1px solid #dce5d6}h1{font-size:36px;line-height:1.25}h2{font-size:24px}small{color:#617b64}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #dae1d5}th{background:#edf2e8}.scroll{overflow:auto}a{color:#256243}code{font-size:12px;overflow-wrap:anywhere}'
(ROOT/'reports/frozen_inference_audit.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>동결 추론 재현 및 학습 충분성 감사</title><style>'+style+'</style><body>'+''.join(sections)+'</body></html>')
print('Created reports/frozen_inference_audit.html')
