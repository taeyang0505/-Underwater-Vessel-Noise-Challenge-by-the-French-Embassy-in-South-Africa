"""Render a dated, offline status snapshot without opening query data."""
from pathlib import Path
import csv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import html,json

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def write(p,s):
    p=Path(p);q=p.with_name(p.name+'.tmp');q.write_text(s);q.replace(p)
def main():
    now=datetime.now(timezone.utc).astimezone(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    run=ROOT/'models/r16_production_nn_fold0_r1'
    progress=read(run/'progress.json')if(run/'progress.json').exists()else{}
    all_nn=(run/'ALL_COMPLETED.json').exists()
    candidate=(ROOT/'submissions/r16_main_r1/submission_r16.csv').exists()
    failed=(run/'FAILED.json').exists()
    paused=(run/'PAUSED.json').exists()
    pause_receipt=read(run/'PAUSED.json')if paused else None
    phase='r16_candidate_created'if candidate else('r16_neural_failed'if failed else('r16_paused_for_known_failure_review'if paused else('r16_neural_complete_finalizing'if all_nn else'r16_production_nn_fold0_running')))
    status_text='신경망 4종 Fold 0과 후보 CSV 생성 완료'if candidate else('신경망 실행 실패; 자동 재시작 없음'if failed else('알려진 일반화 실패를 검토하기 위해 완전 epoch 경계에서 일시 정지; 자동 재개 없음'if paused else('신경망 4종 Fold 0 완료, 후속 동결 평가/후처리 확인 필요'if all_nn else'신경망 4종 Fold 0 순차 본학습 중')))
    rows=[]
    for fold in range(5):
        p=ROOT/f'models/r16_production_trees_r1/fold_{fold:02d}/model/summary.json'
        s=read(p);m=s['selection_metrics']
        rows.append((fold,s['train_rows'],m['rmse'],m['classifier_auc']))
    nn=[]
    for kind in ('bg_cnn','fdy','itransformer','tfgrid_fdy_conformer'):
        p=run/kind/'epochs.jsonl';a=[json.loads(v)for v in p.read_text().splitlines()if v.strip()]if p.exists()else[]
        last=a[-1]if a else None
        nn.append((kind,last))
    lines=['# R16 실행 현황',f'\n확인 시각: {now} (Asia/Seoul)',
      '\nR11 고정 스트레스 평가와 본체 Tree 회귀·독립 분류 5-Fold는 완료했다. '+status_text+'.',
      '\n## 주 교차 검증과 남은 스트레스',
      '\n| 평가 | 모델 | RMSE | 독립 분류 AUC / 동결 score AUC |',
      '|---|---|---:|---:|',
      '| 별도 배경 local_gate | 본체 Tree5 | 0.076535 | 0.970384 (독립 분류) |',
      '| 각 Fold의 자체 CV 5,000건 결합 | 본체 Tree | 0.077040 | 0.968760 (독립 분류) |',
      '| R11 native 15분 vs 자연 HF 변동 | 본체 Tree5 | 0.139986 | 0.259200 (독립 분류) |',
      '| R11 native 15분 vs St Francis 고밀도 | 본체 Tree5 | 0.146637 | 0.337596 (독립 분류) |',
      '| R11 평탄 15분 vs 자연 HF 변동 | R16 소형 60% Tree + 40% CNN | 0.134024 | 0.184856 (score) |',
      '| R11 native 15분 vs 자연 HF 변동 | R16 소형 60% Tree + 40% CNN | 0.134767 | 0.177716 (score) |',
      '\n위 행들은 서로 다른 평가 자료다. 같은 의미의 점수로 직접 비교하거나 공식 LB로 환산하지 않는다. R11의 실패는 남아 있으며, 주 기준은 사전 지정한 별도 참조·크롭, 미관측 사이트 및 독립 LF/HF 연산자 전이다. R11 결과로 레시피를 다시 튜닝하지 않았다.',
      '\n추가 확인: 본체 Fold0 TRAIN 양성 49,157건의 실제 추가 활성 시간 중앙값은 2분, 최대 10분이며 12분 이상은 0건이다. 15분 이상 native 참조 3개는 모두 C 홀드아웃의 같은 연결 컴포넌트에 있다. 이를 TRAIN으로 옮기면 기존 C 평가는 독립 검증으로 재사용할 수 없다. 장시간 coverage와 자연 HF 음성 구분을 보강한 뒤 새 미관측 참조 평가에서 확인해야 한다. 이 결함을 Public 실패의 유일한 원인으로 단정하지 않는다.',
      '\n## 완료 Tree5의 공유 selection 지표',
      '\n| Fold | TRAIN 행 | selection RMSE | 독립 분류 AUC |','|---:|---:|---:|---:|']
    lines += [f'| {f} | {n:,} | {rm:.6f} | {au:.6f} |'for f,n,rm,au in rows]
    lines += ['\nselection은 2,400건의 공유 개발셋이며 위 수치를 전체 OOF라고 부르지 않는다.',
      '\n## 신경망 Fold 0',
      '\n| 모델 | 완료 epoch / 100 | 최신 TRAIN RMSE | 최적 selection RMSE | 최적 독립 분류 AUC |','|---|---:|---:|---:|---:|']
    for kind,last in nn:
        if last:lines.append(f"| {kind} | {last['epoch']} / 100 | {last['train_online_rmse']:.6f} | {last['best_regression_rmse']:.6f} | {last['best_classifier_auc']:.6f} |")
        else:lines.append(f'| {kind} | 대기 | — | — | — |')
    lines += ['\n4개 모델 모두 100epoch, Adam, early stopping 없음. 회귀/분류 최적 체크포인트를 각각 보관한다. 현재 단계에서는 신경망 Fold 1~4를 시작하지 않는다.',
      '\n## 고정 안전 계약',
      '\n- relative8 유지; PCA·LF anchor·q0·절대 dB·highshare 배제.',
      '- 회귀는 Share MSE + 0.25 × 에너지 가중 cell MSE. 분류는 파라미터/gradient가 분리된 독립 BCE.',
      '- 최종 회귀 앙상블 Tree5 ≥50%, 신경망 아키텍처별 ≤25%. score의 share 비중은 0.',
      '- TRAIN 원천/그룹 분리 및 24시간 purge, 정밀 정답 보존, 0.1dB 입력·가중치 유지.',
      '- 최종 모델·정책 동결 전 Test를 읽지 않는다. 공식 업로드하지 않는다.',
      '\n## 파일',
      '\n- [R11 전체 결과](r16_r11_stress_r1/SUMMARY.md)',
      '- [본체 Tree5 장시간/HF 실패 감사](r16_problem_gate_r1/tree_stress/SUMMARY.md)',
      '- [장시간 자료 공백 및 특징 검토](r16_problem_gate_r1/FEATURE_FAILURE_REVIEW.md)',
      '- [본체 Tree 평가](r16_production_tree_evaluation_r1/SUMMARY.md)',
      '- [신경망 연결 검증](R16_NEURAL_PRODUCTION_READINESS.md)',
      '- [신경망 실제 진행 JSON](../models/r16_production_nn_fold0_r1/progress.json)',
      '- [감독 실행 상태](../ops/r16_production_neural_r1/state.json)',
      '\n`submission_r16.csv`: 4개 신경망 완료 → 제한 앙상블 적합/정책 동결 → 교차 검증 → 동결 Test 추론 후 생성. 현재는 미생성.']
    if paused:lines += ['\n## 실행 보류', '\n사용자 지시에 따라 알려진 실패를 먼저 검토한다. 체크포인트를 보존했으며 신경망 후속 모델, 자동 재개 및 최종화 실행을 보류한다. 앞서 제시한 연속 실행 12~15시간 ETA는 현재 적용하지 않는다.']
    if (ROOT/'reports/r16_support_temporal_r1/index.html').exists():
        completed_small=(ROOT/'reports/r16_support_temporal_r1/evaluation/RESULTS.json').exists()
        small_text='네 조건 각100epoch와 독립 검산을 완료했다. 시간 특징 단독은 주 교차 RMSE를 개선했으나 데이터 결합안은 사전 기준 미달이다. 수정 데이터 본체 재개와 후보 CSV 생성은 진행하지 않았다.'if completed_small else'별도 버전에서 자연 support/HF 음성 범위와 시간 구조 특징을 2×2로 비교한다. 기존 본체 체크포인트는 변경하지 않는다.'
        lines += ['\n## 승인된 소형 보완 실험', '\n'+small_text+' [소형 비교 결과](r16_support_temporal_r1/index.html) · [사전 기준](r16_support_temporal_r1/PROTOCOL.md)']
    latest=ROOT/'reports/r16_multisupport_r1/RESULTS.json'
    if latest.exists():
        lines += ['\n## 최신 다중 support·타깃 보존 검산',
          '\n[방법 A/B 실측 및 F1 본체 연결 검사](r16_multisupport_r1/index.html). 저장 예측 50:50 결합은 장시간/HF RMSE 한 항목 미달, 타깃 보존 재학습도 동시 통과하지 못했다. 본체 추가 학습·CSV 생성은 실행하지 않았고, 지시된 동시 통과 조건과 본체 완주 조건의 우선순위 답변을 기다린다.']
    md='\n'.join(lines)+'\n';write(ROOT/'reports/R16_PRODUCTION_STATUS.md',md)
    # Plain offline HTML: the timestamp deliberately identifies a snapshot.
    table=''.join(f'<tr><td>{f}</td><td>{n:,}</td><td>{rm:.6f}</td><td>{au:.6f}</td></tr>'for f,n,rm,au in rows)
    nt=''
    for kind,last in nn:
        if last:nt+=f"<tr><td>{kind}</td><td>{last['epoch']} / 100</td><td>{last['train_online_rmse']:.6f}</td><td>{last['best_regression_rmse']:.6f}</td><td>{last['best_classifier_auc']:.6f}</td></tr>"
        else:nt+=f'<tr><td>{kind}</td><td>대기</td><td>—</td><td>—</td><td>—</td></tr>'
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>R16 본체 실행 현황</title>
    <style>body{{font:16px/1.7 system-ui;margin:40px auto;max-width:1100px;padding:0 24px;background:#f6f8fc;color:#15243a}}h1{{font-size:32px}}section{{background:white;padding:24px;border:1px solid #dce3ef;border-radius:12px;margin:22px 0}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #e4e9f2}}th{{background:#edf3fa}}.warn{{border-left:5px solid #cf7c25}}small{{color:#566780}}a{{color:#1b5ca2}}</style>
    <h1>R16 본체 실행 현황</h1><small>확인 시각 {html.escape(now)} · 고정 시점 스냅샷</small>
    <section><b>완료:</b> R11 스트레스 평가, 본체 Tree 회귀·분류 5-Fold<br><b>상태:</b> {html.escape(status_text)}<br><b>최근 작업:</b> {html.escape(str(progress.get('kind','대기')))} Fold 0 · epoch {progress.get('epoch','—')}/100<br><b>submission_r16.csv:</b> {'생성됨'if candidate else'미생성'}</section>
    <section><h2>본체 Tree5 평가</h2><p>별도 배경 gate: RMSE <b>0.076535</b> · 독립 분류 AUC <b>0.970384</b><br>각 Fold 자신의 CV 결합: RMSE <b>0.077040</b> · AUC <b>0.968760</b></p><table><tr><th>Fold</th><th>TRAIN</th><th>공유 selection RMSE</th><th>독립 분류 AUC</th></tr>{table}</table><small>공유 selection 지표는 OOF가 아닙니다. 공식 리더보드 성능을 보장하지 않습니다.</small></section>
    <section><h2>신경망 Fold 0</h2><table><tr><th>모델</th><th>완료 epoch</th><th>TRAIN RMSE</th><th>최적 selection RMSE</th><th>최적 독립 AUC</th></tr>{nt}</table><p>각100epoch · Adam · early stopping 없음 · 회귀/분류 체크포인트 별도 보관</p></section>
    <section class="warn"><h2>R11 스트레스는 일부 실패</h2><p>소형 혼합 모델에서 평탄15분 vs 자연 HF 변동 RMSE <b>0.134024</b>, score AUC <b>0.184856</b>.<br>native15분 vs 자연 HF 변동 RMSE <b>0.134767</b>, score AUC <b>0.177716</b>.</p><p>완료된 본체 Tree5에서도 native15분 vs 자연 HF 변동 RMSE <b>0.139986</b>, 독립 분류 AUC <b>0.259200</b>로 실패가 확인됐습니다. 양성 확률 평균 0.2676, 음성 0.5013으로 순위가 뒤집혔습니다. 0.99 이상 포화율은 0.8%여서 확률 포화만으로 설명되지 않습니다. <a href="r16_problem_gate_r1/tree_stress/SUMMARY.md">본체 실패 감사</a></p><p>이 결과로 간극 해결을 선언하지 않습니다. 사전 지정한 별도 참조·크롭/미관측 사이트/독립 채널 전이를 주 기준으로 유지하고 레시피를 다시 바꾸지 않았습니다.</p></section>
    <section><h2>다음 단계</h2><p>Wide CNN → FDY-CRNN → iTransformer → TFGrid. 네 모델 완료 후 Tree ≥50%, 각 NN ≤25%의 제한 앙상블과 독립 분류 score를 고정합니다. 이후 잠긴 교차 검증 및 Test 추론으로 후보 CSV를 만듭니다. 공식 업로드는 하지 않습니다.</p><a href="R16_PRODUCTION_STATUS.md">전체 Markdown 현황</a> · <a href="r16_r11_stress_r1/SUMMARY.md">R11 전체 결과</a> · <a href="r16_production_tree_evaluation_r1/SUMMARY.md">Tree 검증</a> · <a href="../ops/r16_production_neural_r1/state.json">감독 상태 JSON</a></section></html>'''
    coverage='<section class="warn"><h2>확인된 학습 자료 공백</h2><p>Fold0 TRAIN 양성 <b>49,157건 중 12분 이상 추가 성분이 이어지는 사례는 0건</b>입니다. 활성 시간 중앙값은 2분, 최대 10분입니다. 15분 이상 native 참조 3개는 C 홀드아웃의 동일한 연결 컴포넌트에 있습니다.</p><p>C 참조를 학습으로 옮기면 기존 C 평가는 독립 검증으로 다시 쓸 수 없습니다. 원천 분리와 24시간 purge를 유지한 새 검증 설계가 필요합니다. 이 사실만으로 Public 실패의 단독 원인을 확정하지 않습니다.</p><a href="r16_problem_gate_r1/FEATURE_FAILURE_REVIEW.md">자료·특징 진단 근거</a></section>'
    page=page.replace('<section><h2>다음 단계</h2>',coverage+'<section><h2>다음 단계</h2>')
    if paused:page=page.replace('<h2>다음 단계</h2><p>Wide CNN', '<h2>현재 실행 보류</h2><p><b>알려진 실패를 먼저 검토하며 자동 재개 및 후보 CSV 생성은 보류합니다. 기존 12~15시간 ETA는 현재 적용하지 않습니다.</b></p><p>향후 재개할 때의 순서: Wide CNN')
    if (ROOT/'reports/r16_support_temporal_r1/index.html').exists():
        small_note='각100epoch·독립 검산 완료. 시간 특징 단독은 주 교차 RMSE 개선, 데이터 결합안은 사전 관문 미달입니다. 기존 본체는 정지 상태로 보존하고 수정 데이터 본체 재개·새 CSV 생성은 진행하지 않았습니다.'if completed_small else'기존 본체는 정지 상태로 보존합니다. 소형 비교 결과를 확인한 뒤 수정 데이터의 본체 연결 여부를 판단합니다.'
        page=page.replace('<section><b>완료:</b>', '<section><b>자연 support·시간 구조 2×2 소형 비교:</b> <a href="r16_support_temporal_r1/index.html">결과 열기</a><p>'+small_note+'</p></section><section><b>완료:</b>')
    if (ROOT/'reports/r16_clean_support_r1/index.html').exists():
        clean_result=read(ROOT/'reports/r16_clean_support_r1/evaluation/RESULTS.json')if(ROOT/'reports/r16_clean_support_r1/evaluation/RESULTS.json').exists()else None
        clean_note=('평가 완료: '+('관문 통과 후보가 있습니다.'if clean_result['any_gate_passed']else'사전 관문을 통과한 후보가 없습니다.'))if clean_result else'원래 음성800건 유지 및680건 유지/120건 교체 두 조건을 F1과100epoch/Adam으로 새로 비교하고 있습니다.'
        page=page.replace('<section><b>완료:</b>', '<section><b>최신 클린 음성 복원 비교:</b> <a href="r16_clean_support_r1/index.html">진행·결과 열기</a><p>'+clean_note+'</p></section><section><b>완료:</b>')
    if latest.exists():
        page=page.replace('<section><b>완료:</b>', '<section><b>최신 저장 예측 결합·타깃 보존 실측:</b> <a href="r16_multisupport_r1/index.html">결과와 F1 본체 연결 검사 열기</a><p>50:50 결합은 장시간/HF RMSE 한 항목 미달입니다. 타깃 평균/RMS를 정확히 맞춘 재학습도 동시 통과하지 못했습니다. 본체 진입 조건의 우선순위를 질문한 상태이며 추가 본학습·CSV 생성은 미실행입니다.</p></section><section><b>완료:</b>')
    write(ROOT/'reports/R16_PRODUCTION_START.html',page)
    old=ROOT/'ops/r16_execution_state.json'
    if old.exists():
        previous=ROOT/'ops/r16_execution_state_preproduction_snapshot.json'
        if not previous.exists():write(previous,old.read_text())
    state={'phase':phase,'updated_kst':now,'R11_stress':'completed_with_failures_not_a_recipe_gate',
       'Tree5':'completed','NN_fold0_progress':progress,'pause_receipt':pause_receipt,'NN_Fold1_to4':'not_started',
       'known_failure':{'native15_hf_tree5_classifier_auc':0.2592,'train_positive_rows':49157,'positive_support_ge12_minutes':0,'report':'reports/r16_problem_gate_r1/FEATURE_FAILURE_REVIEW.md'},
       'ensemble_contract':'tree5>=.5 eachNN<=.25 score_share0; not fitted until all complete',
       'submission_r16_csv':'created'if candidate else'not_yet_created','Test_used_for_training_or_policy':False,
       'report':'reports/R16_PRODUCTION_START.html','execution_state':'ops/r16_production_neural_r1/state.json'}
    write(old,json.dumps(state,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'rendered','progress':progress,'updated_kst':now},ensure_ascii=False))

if __name__=='__main__':main()
