"""Readable local index; reads completed summaries only, no fitting or inference."""
from pathlib import Path
import html
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = json.loads((ROOT / 'models/r6_fixed100_v1/results.json').read_text())
    assert result['status'] == 'completed'
    rows = ''
    for summary in result['summaries']:
        assert summary['epochs_run'] == 100
        best = summary['best_inner_window']
        rows += '<tr><td>' + html.escape(summary['arm']['name']) + '</td><td>' + str(best['epoch']) + '</td><td>' + format(best['metrics']['rmse'], '.6f') + '</td><td>100</td></tr>'
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R6 · 100 epoch & 공개 원천 조사</title><style>body{margin:0;background:#f5f6f3;color:#17251e;font:17px/1.7 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:48px 26px}h1{font-size:34px;line-height:1.3}h2{margin-top:12px}a{color:#246a54}article{background:white;border:1px solid #dbe2dc;border-radius:14px;padding:28px;margin:24px 0}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:12px;text-align:left;border-bottom:1px solid #dbe2dc}img{width:100%;height:auto}.tag{color:#52735e;letter-spacing:.12em;font-size:14px}.note{border-left:4px solid #b78953;padding:8px 18px;background:#faf6ed}.scroll{overflow:auto}</style><main><div class="tag">UNDERWATER VESSEL NOISE · R6 · 2026-10-26</div><h1>공개 원본 데이터 조사와<br>100 epoch CPU 실험</h1><article><h2>공개 데이터의 출처</h2><p>대회가 제공한 남아공 실측자료가 이미 배경 원천입니다. 전체 10개 사이트·1,715,779행과 동일한 별도 공개 archive/DOI는 확인하지 못했습니다. St Francis 일부와 시기·기기·지역이 겹치는 2022년 연구는 찾았습니다.</p><p><a href="R6_PUBLIC_TRAINING_PROVENANCE_REVIEW.md">원천 조사와 물리적 단서</a> · <a href="R6_PUBLIC_TRAINING_PROVENANCE_CATALOG.md">검색 catalogue</a> · <a href="R6_PUBLIC_TRAINING_PROTOCOL_CODE_REVIEW.md">관측 support 구현 점검</a> · <a href="R6_TEST_GAP_CLAIMS_REVIEW.md">Test gap 해석 검토</a></p></article><article><h2>5개 작은 CNN 비교군 · 각각 100 epoch 완료</h2><p>Adam, early stopping 없음. epoch별 가중치 500개를 보존했고 최저 window RMSE, 최저 site-macro RMSE, 마지막 epoch를 별도로 저장했습니다. 실행 약 22.56분, GPU 사용 없음.</p><div class="note">아래는 checkpoint 선택에 사용한 자체 INNER 성능입니다. 5-Fold OOF나 공식 Test 점수가 아니며, global minimum을 보장하지 않습니다. 대형 백본 본학습은 재개하지 않았습니다.</div><div class="scroll"><table><thead><tr><th>비교군</th><th>최저 window epoch</th><th>INNER RMSE</th><th>실제 실행 epoch</th></tr></thead><tbody>''' + rows + '''</tbody></table></div><p><a href="R6_FIXED100_TRAINING_REPORT.md">전체 결과 · ES7 비교 · 학습 설정</a> · <a href="R6_FIXED100_COMPLETION_VERIFICATION.json">500개 checkpoint 실행 검증</a> · <a href="R6_REVIEW_START.html">앞선 R6 데이터·비교 실험</a></p></article><article><h2>전체 학습 곡선</h2><img alt="5개 비교군의 100 epoch 학습 곡선" src="r6_fixed100_figures/01_fixed100_curves.png"><p>최저값은 24~45 epoch에서 기록됐으며, 100번째 epoch 값은 모든 비교군에서 그 최저값보다 높았습니다.</p></article><article><h2>최적 checkpoint와 마지막 epoch</h2><img alt="이전 ES 선택값과 이번 기록 최소값 비교" src="r6_fixed100_figures/02_former_es_comparison.png"><p>중단 지표와 checkpoint 선택 지표를 구분해야 합니다. 상대 채널+전역 문맥 CNN의 최선 epoch 31은 window 기준 ES7을 적용해도 중단 전에 이미 포함됩니다.</p></article><article><h2>실제 학습률</h2><img alt="고정된 macro 기반 scheduler의 실제 학습률" src="r6_fixed100_figures/03_learning_rates.png"></article></main></html>'''
    target = ROOT / 'reports/R6_FIXED100_START.html'
    target.write_text(page)
    print(json.dumps({'html': str(target), 'arms': len(result['summaries'])}))


if __name__ == '__main__':
    main()
