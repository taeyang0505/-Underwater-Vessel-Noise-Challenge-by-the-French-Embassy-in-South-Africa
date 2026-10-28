"""Render the clean-support run from immutable saved artifacts only."""
from pathlib import Path
import datetime,html,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r16_clean_support_r1'
MODELS=ROOT/'models/r16_clean_support_r1'
ARMS=('S1_H0_F1','S1_H15_F1')
def read(p):return json.loads(p.read_text())
def main():
 OUT.mkdir(parents=True,exist_ok=True);stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')
 ready=read(OUT/'DATA_READY.json')if(OUT/'DATA_READY.json').exists()else None
 progress=[]
 for arm in ARMS:
  p=MODELS/arm/'progress.json';v=read(p)if p.exists()else{}
  complete=MODELS/arm/'COMPLETED.json'
  progress.append(f"<tr><td>{arm}</td><td>{'완료'if complete.exists()else'실행 중'if v else'준비'}</td><td>{v.get('epoch','—')}/100</td><td>{v.get('best_regression_rmse',float('nan')):.6f}</td><td>{v.get('best_classifier_auc',float('nan')):.6f}</td></tr>")
 extra='';res=OUT/'evaluation/RESULTS.json'
 if res.exists():
  d=read(res);old=read(ROOT/'reports/r16_support_temporal_r1/evaluation/RESULTS.json')
  ix={(r['panel'],r['arm'],r['model']):r for r in d['rows']};oi={(r['panel'],r['arm'],r['model']):r for r in old['rows']}
  md=['# R16 원래 음성 복원 + 자연 support/F1 실측','',f"최종 갱신: {stamp}",'',
      '**사전 본체 관문: '+('통과 후보 있음'if d['any_gate_passed']else'동시 통과 후보 없음')+'**. 두 후보 모두 Adam으로 100epoch를 실행했다. 음성 과잉 교체 제거로 장시간 분류가 크게 회복됐지만, 기존 D0F1의 일반 교차 회귀 성능을 동시에 보존하지 못했다. Test 및 공식 제출을 사용하지 않았다. 알려진 개발 패널의 재평가이며 새로운 blind 평가가 아니다.','',
      '## 1. 정확히 바꾼 것','',
      '두 조건의 양성 3,200건은 이전 D1 배열을 그대로 유지했다. 실제 활성 시간 중앙값은 5분, 12분 이상은 500건이다. H0는 음성 800건을 이전 D0 배열 그대로 유지했고, H15는 680건 유지·120건만 TRAIN의 HF 후보로 교체했다. 사이트/센서/녹음기/관측 마스크는 4,000행 모두 D0와 동일하며, E=B+A 및 셀/창 정답 재계산 최대 오차는 0이다. HF 교체 120개는 사전 고정한 층별 배분과 해시 순서로 골랐으며 평가 점수로 고르지 않았다.','',
      'F1·seed·모델 초기화·optimizer·epoch·local_fit 선택·Tree 설정은 이전 D0F1과 같다. 새로운 PCA/LF 앵커/절대 레벨/강제 highshare/평탄화/시간 신축은 추가하지 않았다. 과잉 음성 교체 제거의 분리 실험이다.','',
      '회귀는 Tree 60% + CNN 40%, 자체 score는 독립 Tree 분류기 50% + 독립 CNN 분류기 50%다. `+D0score`는 새 회귀를 그대로 두고 기존 D0F1의 저장된 독립 분류 score만 붙인 사전 지정 대조다. 회귀를 score에 섞거나 평가 후 가중치를 재적합하지 않았다.','',
      '## 2. 주 교차 평가','',
      '각 셀은 회귀 RMSE / 독립 분류 score AUC다. 각 축은 400개 원배경 조합 × 6개 생성 연산자 = 2,400행이다. 같은 고정 패널과 결합 비율로 비교하며, 행 수가 독립 녹음 수를 뜻하지 않는다.','',
      '| 조건 | 별도 배경 | 별도 참조 C′ | 미관측 사이트 |','|---|---:|---:|---:|']
  primary=('local_gate__POOLED','local_report__POOLED','domain_holdout__POOLED')
  oldrow=[]
  for p in primary:
   v=oi[p,'D0F1','fixed60T40N'];oldrow.append(f"{v['rmse']:.6f} / {v['auc_classifier']:.6f}")
  md.append('| 기존 D0F1 | '+' | '.join(oldrow)+' |')
  for arm in ARMS:
   for method,label in [('fixed60T40N','자체score'),('reg_plus_D0F1score','+D0score')]:
    v=[ix[p,arm,method]for p in primary];md.append('| '+arm+' '+label+' | '+' | '.join(f"{x['rmse']:.6f} / {x['auc_classifier']:.6f}"for x in v)+' |')
  md+=['','## 3. 독립 native12 vs HF/SF 대조','',
      '원천은 독립 12분 컴포넌트 1개다. 양성 2,400행/HF 음성 359행/SF 음성 168행이며 클래스별 총가중치 0.5씩으로 균형 RMSE를 계산했다. 2,400개의 독립 선박 녹음이 아니다. 인위적인 full_smooth15 조건이 본체 승격을 막는 기준인 것은 아니다.','',
      '| 조건 | HF 균형 RMSE / AUC | SF 균형 RMSE / AUC | HF 음성 score 평균 | SF 음성 score 평균 |','|---|---:|---:|---:|---:|']
  for arm in ('D0F1','D1F1',*ARMS):
   for method,label in ([('fixed60T40N','기준'if arm=='D0F1'else'이전 과잉 교체')]if arm in('D0F1','D1F1')else[('fixed60T40N','자체 score'),('reg_plus_D0F1score','+D0score')]):
    v=[oi[p,arm,method]if arm in('D0F1','D1F1')else ix[p,arm,method]for p in('native12_VS_hf_high_negative','native12_VS_st_francis_dense_negative')]
    md.append('| '+arm+' '+label+' | '+' | '.join(f"{x['rmse']:.6f} / {x['auc_classifier']:.6f}"for x in v)+' | '+' | '.join(f"{x['negative']['classifier_mean']:.6f}"for x in v)+' |')
  md+=['','## 4. 체크포인트 선택과 실행','',
      '| 조건 | 실행 epoch | 회귀 best epoch | 분류 best epoch | fit NN RMSE | fit NN AUC |','|---|---:|---:|---:|---:|---:|']
  for arm in ARMS:
   c=read(MODELS/arm/'COMPLETED.json');md.append(f"| {arm} | {c['epochs']} | {c['best_regression_epoch']} | {c['best_classifier_epoch']} | {c['fit_nn_metrics']['rmse']:.6f} | {c['fit_nn_metrics']['auc_classifier']:.6f} |")
  elapsed=[read(MODELS/a/'COMPLETED.json')['elapsed_seconds']for a in ARMS]
  md+=['',f"두 조건의 학습은 각각 {elapsed[0]:.1f}/{elapsed[1]:.1f}초였으며 CPU 병렬 실행으로 약 {max(elapsed)/60:.1f}분이 걸렸다. 고정 평가에는 {d['elapsed_seconds']:.1f}초가 추가됐다. 코드 작성·자료 검산·보고서 시간은 별도다.",'',
      'TRAIN 4,000건, local_fit 2,400건이며, 신경망은 회귀 28,249개 + 독립 분류 5,041개 = 총 33,290개 파라미터다. Adam·dropout 0.1·조기 종료 없이 100epoch를 유지했다. 최적 epoch는 local_fit에서만 선택했다. 이전 best epoch=1은 1epoch 후 학습을 중단했다는 뜻이 아니며, 100epoch 전체 실행 후 선택된 결과다. 이번 분류 best epoch는 44/45로 초기 최적점 현상은 반복되지 않았다.','',
      '## 5. 사전 관문','',
      '| 후보 | 방법 | 통과 | 실패 항목 |','|---|---|---|']
  for g in d['candidate_gates']:
   names={'local_gate':'별도 배경','local_report':'별도 참조 C′','domain_holdout':'미관측 사이트','st_francis_dense_negative_negative_preservation':'SF 음성 share 보존'}
   md.append(f"| {g['arm']} | {g['method']} | {'통과'if g['passed']else'미달'} | "+', '.join(names.get(c['name'],c['name'])for c in g['checks']if not c['passed'])+' |')
  md+=['','주 교차 각 축의 D0F1 대비 RMSE 악화 ≤ max(0.005, 기준의 5%), AUC 하락 ≤ 0.02; native12-HF RMSE ≤ 0.11 및 AUC ≥ 0.75; native12-SF RMSE ≤ 0.12 및 AUC ≥ 0.75; 음성 score 평균 증가 ≤ 0.02 및 share 평균 증가 ≤ 0.01을 학습 전에 고정했다. 후보 선택 순서도 H0 자체 → H0+D0score → H15 자체 → H15+D0score로 사전 고정했다. 결과를 보고 기준을 완화하지 않았다.','',
      '## 6. 검증 파일과 한계','',
      '[사전 계획](PROTOCOL.md) · [데이터 재계산](DATA_RECOMPUTED_QA.json) · [동결 근거](GLOBAL_FREEZE.json) · [전체 수치 JSON](evaluation/RESULTS.json) · [개별 모델/연산자 CSV](evaluation/metrics.csv)','',
      '원천 A′75/B19/C′25 분리와 24시간 purge를 유지했다. 전체 509개 그래프의 독립성을 주장하지 않으며, 독립 장시간 평가는 12분 원천 하나에 제한된다. 평가 반복 노출에 따른 개발 과적합 가능성을 배제하지 않는다. 로컬 관문 통과와 공식 LB 성능 개선은 다르다.']
  if(OUT/'RESULTS_RECOMPUTED_QA.json').exists():
   q=read(OUT/'RESULTS_RECOMPUTED_QA.json');md+=['',f"저장 예측 {q['prediction_files']}파일 및 동결 {q['frozen_artifacts']}파일의 SHA를 검산하고, {q['recomputed_rows']}개 지표 행을 별도 수식으로 재계산했다. 최대 오차는 {q['maximum_metric_absolute_error']:.3e}다. 재검산 중 새 학습·추론은 하지 않았다. [재검산 영수증](RESULTS_RECOMPUTED_QA.json)"]
  bg=read(OUT/'BACKGROUND_ONLY_DESCRIPTIVE_QA.json')
  md+=['','## 7. 교정 효과와 남은 문제의 해석','',
      '**과잉 교체를 제거한 효과는 확인됐다.** 다른 설정을 유지한 H0/H15에서 native12-HF AUC가 이전 D1F1의 0.5974에서 0.9239/0.9563으로 회복됐고 장시간 균형 RMSE도 약 0.10을 유지했다. 이 조건에서는 과잉 HF 음성 교체가 분류 붕괴에 크게 기여했다는 근거다. 구현 강도가 과했던 것은 제 오류였으며 이번 두 조건에서 교정했다.','',
      '반면 두 개선 요소를 결합하면 모든 일반 성능도 자동 보존된다는 주장은 결과와 맞지 않는다. H0 자체 score의 일반 AUC는 0.873~0.913, H15는 0.841~0.887로 D0F1 약 0.955에 미달했다. 기존 D0F1 score를 붙이면 그 AUC를 그대로 보존하지만 회귀 RMSE는 변하지 않는다. 이 AUC 보존은 같은 저장 score를 재사용한 결과이므로 새로운 분류 성능 향상으로 세지 않는다.','',
      '### 순수 기저 배경 B의 분포는 정상화됐는가?','',
      '| 조건 | 양성 B의 HF bump 중앙값(dB) | 음성 B의 HF bump 중앙값(dB) | 양성/음성 B의 HF KS |','|---|---:|---:|---:|']
  ob=bg['old_D1_reference'];md.append(f"| 이전 D1 (798/800 교체) | {ob['positive_background_HF_median']:.4f} | {ob['negative_background_HF_median']:.4f} | {ob['HF_KS_positive_vs_zero']:.5f} |")
  for arm in ARMS:
   b=bg['rows'][arm];md.append(f"| {arm} | {b['positive_background_HF_median']:.4f} | {b['negative_background_HF_median']:.4f} | {b['background_HF_KS_by_label']:.5f} |")
  md+=['','이는 TRAIN의 순수 B에 대한 기술 통계다. 낮은 KS는 측정한 HF 변동 분포가 가까워졌다는 뜻이며, 모든 특징의 모집단 일치나 배경 단독 분류 AUC=0.5를 보장하지 않는다.','',
      '### 무추가 배경의 회귀 오탐도 보존됐는가?','',
      '| 조건 | HF 음성 share 평균 | SF 음성 share 평균 |','|---|---:|---:|']
  for arm in('D0F1',*ARMS):
   rows=[oi[p,arm,'fixed60T40N']if arm=='D0F1'else ix[p,arm,'fixed60T40N']for p in('native12_VS_hf_high_negative','native12_VS_st_francis_dense_negative')]
   md.append('| '+arm+' | '+' | '.join(f"{r['negative']['prediction_mean']:.6f}"for r in rows)+' |')
  md+=['','정답이 0인 SF 음성의 share 평균이 0.048595 → 0.067180/0.065420으로 올라 사전 허용 증가폭 +0.01을 넘었다. 이는 score 평균이나 threshold 기반 오탐률과 다른 지표다. HF 음성 회귀는 유지·개선됐지만 SF까지 완전히 해결된 것은 아니다.','',
      '### 남은 원인 후보와 실행 결정','',
      '음성을 복원해도 양성은 D1 그대로이므로 TRAIN share 평균/RMS가 D0의 0.145799/0.235432에서 0.194907/0.298823으로 바뀐다. 자연 support 변화에 따른 확인된 차이다. 이번 실험만으로 이 변화가 잔여 악화의 유일 원인이라고 단정하거나 Test 정답 분포를 추정해 보정하지 않는다. 일반 회귀 성능을 보존하면서 장시간 이득을 얻는 문제가 남았다.','',
      '요청한 두 파일럿과 기존 분류기 분리 결합 비교는 완료했다. 전 조건 동시 통과에 따른 본체 승격 조건은 충족되지 않아, 기존 본체 CNN의 epoch 27 체크포인트와 완료 Tree 5-Fold를 보존하고 추가 본학습은 시작하지 않았다. 미달 후보를 최종본으로 표시한 submission_r16.csv도 생성하지 않았다. 새 소형 탐색을 자동으로 더 시작하지 않았다.','',
      '[실행 코드](../../ops/r16_clean_support_check.py) · [별도 검산 코드](../../ops/verify_r16_clean_support.py) · [TRAIN 배경 기술 통계](BACKGROUND_ONLY_DESCRIPTIVE_QA.json)']
  (OUT/'R16_CLEAN_SUPPORT_RESULTS.md').write_text('\n'.join(md)+'\n')
  # Plain HTML tables are directly viewable without a Markdown plugin.
  from html import escape
  def table(panelnames):
   h='<table><tr><th>조건</th>'+''.join('<th>'+escape(n)+'</th>'for _,n in panelnames)+'</tr>'
   for arm in('D0F1',*ARMS):
    v=[oi[p,arm,'fixed60T40N']if arm=='D0F1'else ix[p,arm,'fixed60T40N']for p,_ in panelnames]
    h+='<tr><td>'+arm+'</td>'+''.join(f"<td>{r['rmse']:.6f} / {r['auc_classifier']:.6f}</td>"for r in v)+'</tr>'
   return h+'</table>'
  extra='<section><h2>관문: '+('통과 후보 있음'if d['any_gate_passed']else'동시 통과 후보 없음')+'</h2><p><strong>장시간 분류는 회복됐지만 일반 교차 회귀 성능을 동시에 보존하지 못했습니다.</strong></p><p>native12-HF AUC: 이전 D1F1 0.5974 → H0 0.9239 / H15 0.9563. 두 조건 모두 100epoch 완료, 분류 best epoch 44/45.</p><p><a href="R16_CLEAN_SUPPORT_RESULTS.md">전체 결과·원인 해석 MD</a> · <a href="evaluation/metrics.csv">모든 개별 모델 수치</a> · <a href="RESULTS_RECOMPUTED_QA.json">184행 재검산</a></p><h3>주 교차 RMSE / AUC</h3>'+table(list(zip(primary,['별도 배경','별도 참조','미관측 사이트'])))+'<h3>native12 균형 RMSE / AUC</h3>'+table([('native12_VS_hf_high_negative','HF 음성'),('native12_VS_st_francis_dense_negative','SF 음성')])+'<p>기존 D0F1 분류기를 붙이면 AUC는 보존되지만 회귀 오차는 그대로입니다. SF 음성 share 평균도 0.0486 → 0.0672/0.0654로 증가했습니다.</p><p>본체 CNN epoch 27과 완료 Tree 5-Fold를 보존했습니다. 추가 본학습 및 submission_r16.csv 생성은 실행하지 않았습니다. 공식 점수가 아닌 로컬 개발 평가입니다.</p></section>'
 page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>R16 clean support</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:40px auto;background:#f5f7fa;color:#172638;padding:0 24px}section{background:white;border:1px solid #dce4ee;border-radius:12px;padding:24px;margin:20px 0}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}a{color:#205d9c}</style><h1>R16 자연 길이 + 원래 음성 복원</h1><p>'+html.escape(stamp)+' · 수동 갱신 스냅샷</p><section><p>H0: 원래음성800건유지. H15:680건유지/120건교체. 양성D1자연길이와F1·seed·학습설정유지. 기존본체CNN27epoch는보존합니다. Test를사용하지않습니다.</p><table><tr><th>조건</th><th>상태</th><th>epoch</th><th>best fit RMSE</th><th>best fit 분류AUC</th></tr>'+''.join(progress)+'</table><p>fit은체크포인트선택용이며공식점수가아닙니다.</p></section>'+extra+'<section><a href="PROTOCOL.md">사전 기준</a> · <a href="DATA_RECOMPUTED_QA.json">정답·배열별 재검산</a></section></html>'
 (OUT/'index.html').write_text(page)
 print(json.dumps({'status':'rendered','result_ready':res.exists(),'path':str(OUT/'index.html')}))
if __name__=='__main__':main()
