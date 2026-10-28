"""Read-only progress renderer for the four-arm CPU ablation; never opens query data."""
from pathlib import Path
import datetime, html, json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r16_support_temporal_r1'
MODELS=ROOT/'models/r16_support_temporal_r1'

def main():
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')
    metrics=[]
    for arm in ('D0F0','D0F1','D1F0','D1F1'):
        path=MODELS/arm/'progress.json'
        d=json.loads(path.read_text())if path.exists()else{}
        completed=(MODELS/arm/'COMPLETED.json').exists()
        metrics.append(f"<tr><td>{arm}</td><td>{'완료'if completed else '학습 중'if d else '대기'}</td><td>{d.get('epoch','—')}/100</td><td>{d.get('best_regression_rmse',float('nan')):.6f}</td><td>{d.get('best_classifier_auc',float('nan')):.6f}</td></tr>")
    result=OUT/'evaluation/RESULTS.json';extra=''
    if result.exists():
        r=json.loads(result.read_text())
        rows={(x['panel'],x['arm']):x for x in r['rows']if x['model']=='fixed60T40N'}
        summary=[]
        for arm in ('D0F0','D0F1','D1F0','D1F1'):
            cells=''.join(f"<td>{rows[p,arm]['rmse']:.6f} / {rows[p,arm]['auc_classifier']:.6f}</td>"for p in ('local_gate__POOLED','local_report__POOLED','domain_holdout__POOLED'))
            summary.append('<tr><td>'+arm+'</td>'+cells+'</tr>')
        long=[]
        for arm in ('D0F0','D0F1','D1F0','D1F1'):
            cells=''.join(f"<td>{rows[p,arm]['rmse']:.6f} / {rows[p,arm]['auc_classifier']:.6f}</td>"for p in ('native12_VS_hf_high_negative','native12_VS_st_francis_dense_negative'))
            long.append('<tr><td>'+arm+'</td>'+cells+'</tr>')
        extra='<section><h2>결과: 시간 특징 단독 개선 · 결합안 미달</h2><p>시간 특징 단독은 세 주 교차 RMSE를 낮췄습니다. 자연 길이＋HF 음성 보강 결합안은 독립 분류 AUC가 악화되어 사전 본체 적용 기준 6개를 모두 통과하지 못했습니다. 기존 본체 CNN 27epoch와 Tree 5-Fold는 보존했고 본체 재개·새 제출 CSV 생성을 진행하지 않았습니다.</p><p><a href="R16_SUPPORT_TEMPORAL_ABLATION.md">최종 상세 보고서</a> · <a href="INDEPENDENT_FINDINGS.md">독립 재검산</a> · <a href="evaluation/metrics.csv">개별 모델·연산자 전 수치</a></p><h3>주 교차 평가 — RMSE / 독립 분류 AUC</h3><table><tr><th>조건</th><th>별도 배경</th><th>별도 참조 C′</th><th>미관측 사이트</th></tr>'+''.join(summary)+'</table><h3>실제 독립 native12 대조 — 균형 RMSE / AUC</h3><table><tr><th>조건</th><th>자연 HF 음성</th><th>St Francis dense 음성</th></tr>'+''.join(long)+'</table><p class="note">HF 보강이 무추가 800개 중798개 교체가 되어 일반 음성 범위를 크게 좁혔습니다. support와 HF 교체가 묶인 데이터 요인이므로 어느 하나의 인과효과를 단정하지 않습니다. 같은 SF 원배경 대조에서도 AUC 0.801965→0.330688로 악화됐습니다.</p><p>435개 지표행 독립 재계산 최대 오차2.89e−15. 공식 Test 성능 개선을 입증한 결과가 아닙니다.</p></section>'
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>R16 support·시간 구조 비교</title>
<style>body{font:16px/1.7 system-ui;background:#f5f7fa;color:#172638;max-width:1100px;margin:40px auto;padding:0 24px}section{background:white;padding:24px;border:1px solid #dce4ee;border-radius:12px;margin:20px 0}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:10px;border-bottom:1px solid #e5ebf2}th{background:#edf3f8}a{color:#205d9c}.note{color:#65522c}</style>
<h1>R16 자연 길이·시간 구조: 2×2 소형 비교</h1><p>확인 시각 '''+html.escape(stamp)+''' · 수동 갱신한 시점 스냅샷</p>
<section><h2>실행 범위</h2><p>기존 본체 CNN은 27epoch 완료 후 정지 상태로 보존했습니다. 새 소형 CNN·Tree의 Mac CPU 네 조건 비교를 완료했습니다. 각 CNN100epoch·Adam·early stopping 없음입니다.</p><table><tr><th>조건</th><th>자료</th><th>특징</th></tr><tr><td>D0F0</td><td>새 공통 A′ + 기존 삽입 방식</td><td>기존 상대 특징</td></tr><tr><td>D0F1</td><td>D0 동일</td><td>시간 지속성·동조 추가</td></tr><tr><td>D1F0</td><td>자연 길이 보존 + HF 음성 보강</td><td>기존 특징</td></tr><tr><td>D1F1</td><td>D1 동일</td><td>시간 지속성·동조 추가</td></tr></table></section>
<section><h2>학습 상태</h2><table><tr><th>조건</th><th>상태</th><th>epoch</th><th>최적 fit RMSE</th><th>최적 fit 분류 AUC</th></tr>'''+''.join(metrics)+'''</table><p class="note">위 fit은 체크포인트 선택용입니다. 잠긴 별도 참조·사이트·장시간 평가는 모든 모델 동결 후에만 실시합니다. fit 개선을 공식 점수로 환산하지 않습니다.</p></section>
<section><h2>새 데이터에서 확인한 사실</h2><p>A′75/B19/C′25클립. 12분 이상 활성 양성은 D0 0건 → D1 500건이며, D1 활성 시간 중앙값은 5분입니다. 명시적 무추가800개 중798개를 동일 site/sensor/recorder/mask의 TRAIN 자연 HF 배경으로 교체했고 후보가 없는2개는 유지했습니다.</p><p>학습 native≥15분은 하나의 컴포넌트, 독립 장시간 평가는 다른12분 컴포넌트 하나입니다. 기존 R11 native15는 새 모델에 대해 참조를 공유하는 진단입니다. 독립15분 전이를 입증했다고 주장하지 않습니다.</p></section>
'''+extra+'''<section><h2>검증·근거</h2><p><a href="PROTOCOL.md">사전 고정 비교안</a> · <a href="DATA_PLAN_AND_LIMITS.md">자료 생성·크기·한계</a> · <a href="INDEPENDENT_DESIGN_REVIEW.md">독립 설계 검토</a> · <a href="INDEPENDENT_DATA_QA.json">독립 자료 전수 QA</a> · <a href="INDEPENDENT_CODE_QA.json">독립 코드 QA</a></p><p>Test 및 Test 유래 통계로 생성기나 정책을 맞추지 않습니다. 공식 제출은 하지 않습니다. 평탄15분·시간 신축·PCA/LF anchor·highshare를 도입하지 않습니다.</p></section></html>'''
    p=OUT/'index.html';tmp=p.with_suffix('.tmp');tmp.write_text(page);tmp.replace(p)
    print(json.dumps({'status':'rendered','updated_kst':stamp,'path':str(p)}))

if __name__=='__main__':main()
