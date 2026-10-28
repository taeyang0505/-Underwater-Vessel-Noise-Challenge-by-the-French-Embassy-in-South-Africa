"""Render current preparation status from verified local manifests."""
from pathlib import Path
import html
import json

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    cv = read('data/synthetic/group_cv_nested5_pilot_v1/cv_manifest.json')
    data = read('reports/group_cv_data_verification.json')
    source = read('reports/group_cv_data_audit.json')
    trainer = read('reports/training_policy_v2_verification.json')
    runner = read('reports/group_cv_runner_verification.json')
    config = read('configs/group_cv_training.json')
    status = {'version': 'nested_group_cv_readiness_20261024', 'date_timezone': '2026-10-24 Asia/Seoul',
        'data_ready': cv['ready_to_train'] and data['passed'], 'preprocessing_ready': True,
        'training_code_fixture_passed': trainer['status'] == 'passed' and runner['passed'],
        'GPU_execution_measured': False, 'full_CV_training_executed': False,
        'original_observations_preserved': 1715779, 'background_windows': source['background_windows'],
        'background_sensor_week_blocks': source['background_groups'],
        'reference_clips': source['reference_clips'], 'reference_connected_components': source['reference_components'],
        'native_long_connected_components': len(source['native_components']),
        'n_splits': cv['n_splits'], 'per_fold': {'train': 16000, 'inner_validation': 3000, 'outer_validation': 3000},
        'synthetic_examples_total': 110000, 'outer_oof_examples': cv['outer_oof_examples'],
        'max_saved_energy_target_error': max(part['max_energy_target_error'] for f in data['folds'] for part in f['partitions'].values()),
        'positive_draw_became_zero_count': sum(part['positive_became_zero'] for f in data['folds'] for part in f['partitions'].values()),
        'source_hash': cv['source_hash'], 'training': config['training'],
        'models': config['models'], 'foundation_models': config['foundation_models'],
        'historical_submission': {'id': 'cJy4iTDK', 'score': .658591231, 'rmse': .341408769,
            'neural_training': '8 epochs / AdamW', 'pretrained': 'Frozen ICL; not weight fine-tuning'},
        'target_from_user_screenshot': {'score_to_exceed': .904607408, 'rmse_to_beat': .095392591,
            'use': 'Reporting target only; never derive generator priors or calibration from leaderboard scores'},
        'limitations': ['Synthetic sample count is not the independent source count.',
            'Only five independent connected components contain native 15-minute reference material.',
            'Run-edge and nominal sampling-rate warnings are retained as unresolved flags, not automatically corrected.',
            'Source-disjoint synthetic CV does not prove competition generalization or natural-vessel ground truth.',
            'GPU memory, timing, model convergence and leaderboard superiority remain unmeasured.']}
    (ROOT / 'reports/training_readiness.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
    models = [('CNN / FDY-CRNN / iTransformer / TabM', 'GPU 가중치 학습', '최대 100 epoch · patience 7 · Adam · dropout 0.1'),
        ('TabICLv2 28.5M / Kumo Large 215.7M', 'GPU 권장 · 고정 가중치 ICL', 'fold별 train context만 사용 · outer 전에 inner 예측/정책 고정'),
        ('LightGBM / Physics / Retrieval', 'CPU 트리 학습', 'fold train만으로 특징·prototype 구성 · 내부 validation으로 중단')]
    rows = ''.join('<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in row) + '</tr>' for row in models)
    content = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>현재 데이터·Group CV 학습 준비 상태</title><style>
*{box-sizing:border-box}body{margin:0;background:#f6f7f4;color:#18251e;font:16px/1.7 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:48px 24px}h1{font:600 42px/1.2 Georgia,serif}h2{font-size:23px}section{background:white;border:1px solid #dce4db;border-radius:8px;padding:26px;margin:22px 0}.label{color:#526c5b;font-size:13px;letter-spacing:.7px}.notice{border-left:5px solid #ae6328;background:#fff8ee}a{color:#216344}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:13px 10px;border-bottom:1px solid #e4e8e0;vertical-align:top}th{background:#eff4ec}.scroll{overflow:auto}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}.card{border:1px solid #dce6d8;padding:18px}.number{font:600 31px Georgia,serif}.flow{font-size:18px;background:#eef4e9;padding:22px;border-radius:5px}code{overflow-wrap:anywhere;font-size:12px}li{margin:6px 0}small{color:#536557}</style>
<main><div class="label">VERIFIED PREPARATION STATUS · 2026-10-24 KST</div><h1>데이터·전처리 완료.<br>Nested 5-fold 본학습 준비.</h1>
<p>현재 데이터와 실행 설정의 상태입니다. GPU 본학습·OOF 성능·수렴은 아직 측정하지 않았습니다.</p>
<section class="cards"><div class="card"><div class="number">110,000</div>5-fold 합성 예제<br><small>독립 원천 녹음 수가 아님</small></div><div class="card"><div class="number">15,000</div>외부 OOF 평가 예제<br><small>아직 예측·평가 미실행</small></div><div class="card"><div class="number">26 / 5</div>참조 연결 그룹 / native 그룹<br><small>공유 선박·날짜·구간 연결</small></div><div class="card"><div class="number">100 / 7</div>최대 epoch / patience<br><small>Adam · 최선 checkpoint 복원</small></div></section>
<section><h2>분할과 전처리</h2><div class="flow">원천 분리 → Train 16,000 → 내부 Val 3,000에서 중단·앙상블 선택 → 정책 고정 → 외부 Val 3,000 평가<br>위 절차를 5-fold로 순환</div><p>원천 그룹 기준 약 60% / 20% / 20%입니다. 시간 경계 24시간 purge 후 실제 배경 비율은 fold마다 다릅니다. NPZ 예제 개수는 증강 수이므로 이 비율과 같지 않습니다.</p><ul><li>배경 100,965개 윈도우 · 258개 장비-주간 블록, 외부 1,171개 클립을 합성 전에 분리했습니다.</li><li>2초 Hann PSD → 25대역 적분 → 선형 에너지 1분 평균 → 15×25 입력. 결측은 NaN·관측 마스크로 보존합니다.</li><li>관측 셀의 추가 에너지 비율과 셀별 정답을 함께 저장했습니다. 저장 정답 오차 최대 5.50×10⁻⁷, 잘못된 양성→0 변환 0건입니다.</li><li>모델별 입력 채널·특징은 로더의 고정 변환으로 만듭니다. 각 fold의 변환·context·prototype은 해당 train 자료만 사용합니다.</li><li>calibration·도메인 평가·새 sealed·대회 Test는 CV 구성과 모델 선택에서 제외했습니다.</li></ul></section>
<section><h2>실행 모델과 최적화</h2><div class="scroll"><table><tr><th>모델</th><th>장치·학습 방식</th><th>설정</th></tr>MODELS</table></div><p>기본 Adam, 학습률 0.001, plateau 스케줄러, gradient clipping 1.0입니다. NAdam·Nesterov SGD·dropout 0.05/0.2·contrastive weight 0.01은 명시적 별도 비교 옵션입니다. Contrastive는 검증 전 기본 비활성(weight 0)입니다.</p><p>선택 기준은 RMSE입니다. MAE·bias·R²·Spearman·오차 분위수·에너지 비율 구간·관측량·원천 그룹별 오차, controlled-addition AUC/AP/Brier/ECE를 진단에 추가했습니다. 공유 참조 조합을 독립 원천으로 간주한 CI는 계산하지 않습니다.</p><p>비볼록 신경망의 전역 최솟값을 보장하지 않습니다. 스케줄러·복수 seed·optimizer 비교는 검증 RMSE와 안정성으로 판단합니다.</p></section>
<section class="notice"><h2>기존 제출과 현재 준비를 구분합니다</h2><p>기존 제출 cJy4iTDK는 <strong>0.658591231 / RMSE 0.341408769</strong>입니다. 당시 NN은 8 epoch·AdamW였고 단일 holdout을 사용했습니다. 이번 Group CV·100 epoch·Adam 설정으로 다시 학습한 결과가 아닙니다.</p><p>사용자 첨부 화면의 목표는 score 0.904607408 초과·RMSE 0.095392591 미만입니다. 이 숫자는 목표 기록에만 사용하고 데이터 생성 분포나 보정에 역으로 맞추지 않습니다.</p><p>실제 15분 참조의 독립 원천이 5개로 제한되며, 장비 경계 이상치와 nominal sample-rate 경고는 물리 원인 미확인 flag입니다. 계산 검증 완료가 모든 물리 문제 해결이나 경쟁 성능 보장을 뜻하지 않습니다.</p></section>
<section><h2>GPU 실행</h2><p>RTX 4090 24GB 1장, RAM 64GB 이상, SSD 여유 100GB를 초기 기준으로 잡았습니다. 5-fold·seed 1·9개 후보의 학습/평가 예산은 약 2–4시간이며 CUDA 실측 전 추정입니다. 처음에는 메모리와 처리 시간을 측정한 뒤 실제 예산을 갱신합니다.</p><pre>bash run_group_cv_gpu.sh
bash run_group_cv_gpu.sh --execute</pre><p>첫 명령은 데이터와 가중치 해시 사전검증만 합니다. 두 번째가 실제 학습입니다. 공식 제출·공개 게시·sealed 평가를 자동 수행하지 않습니다.</p></section>
<section><h2>근거 파일</h2><ul><li><a href="data_quality_report.html">데이터 품질·분할 보고서</a> / <a href="group_cv_data_verification.json">5-fold 데이터 독립 검증</a></li><li><a href="group_cv_runner_verification.json">작은 독립 예제의 전체 CV 실행 검증</a> / <a href="training_policy_v2_verification.json">학습 정책 검증</a></li><li><a href="../configs/group_cv_training.json">실행 설정</a> / <a href="../delivery/GPU_GROUP_CV_README.txt">GPU 인계 안내</a></li><li><a href="frozen_inference_audit.html">기존 제출 추론 감사</a> / <a href="training_readiness.json">현재 상태 JSON</a></li><li><a href="report_before_group_cv.html">이전 EDA·단일 holdout 연구 기록 보관본</a></li><li>방법 문서: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html">GroupKFold</a> · <a href="https://docs.pytorch.org/docs/2.14/generated/torch.optim.NAdam.html">NAdam</a> · <a href="https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html">Nesterov SGD</a></li></ul></section></main></html>'''.replace('MODELS', rows)
    (ROOT / 'reports/training_readiness.html').write_text(content)
    current = ROOT / 'reports/report.html'
    backup = ROOT / 'reports/report_before_group_cv.html'
    if not backup.exists():
        legacy = ROOT / 'audit/report_before_group_cv.html'
        backup.write_bytes(legacy.read_bytes() if legacy.exists() else current.read_bytes())
    # Keep historical report links intact below an explicit current-status note.
    original = backup.read_text()
    notice = '<div style="padding:22px 38px;background:#eaf2e4;border-bottom:2px solid #72916b"><strong>현재 상태 업데이트 · 2026-10-24 KST</strong><br>5-fold 데이터 110,000개와 전처리·실행 검증 완료. GPU 본학습 미실행. 아래 내용은 이전 단일 holdout 기록입니다. <a style="font-weight:bold" href="training_readiness.html">현재 Group CV 준비 상태와 설정 보기 →</a></div>'
    current.write_text(original.replace('<body>', '<body>' + notice, 1))
    print(json.dumps({k: status[k] for k in ['data_ready', 'preprocessing_ready', 'training_code_fixture_passed', 'full_CV_training_executed']}, indent=2))


if __name__ == '__main__':
    main()
