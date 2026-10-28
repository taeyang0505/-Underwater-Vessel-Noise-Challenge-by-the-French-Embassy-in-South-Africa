"""Human-readable saved-prediction blend and target-matched support results."""
from pathlib import Path
import json,html,datetime,re
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r16_multisupport_r1'
TM=ROOT/'reports/r16_target_matched_support_r1'
def read(p):return json.loads(p.read_text())
def inline(line):
 escaped=html.escape(line)
 escaped=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:'<a href="'+m.group(2)+'">'+m.group(1)+'</a>'if not re.match(r'^[a-z]+:',m.group(2),re.I)else m.group(0),escaped)
 escaped=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',escaped)
 return re.sub(r'`([^`]+)`',r'<code>\1</code>',escaped)
def main():
 r=read(OUT/'RESULTS.json');feasible=read(OUT/'CONSTANT_WEIGHT_FEASIBILITY.json')
 old=read(ROOT/'reports/r16_support_temporal_r1/evaluation/RESULTS.json');oi={(v['panel'],v['arm'],v['model']):v for v in old['rows']}
 stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')
 md=['# R16 다중 support 결합 및 타깃 보존 실측','',f'갱신: {stamp} (Asia/Seoul)','',
     '**결과: 방법 A와 B를 모두 실행·검산했으나 사전 관문 전체의 동시 통과는 확인되지 않았다.** 가장 균형적인 사전 고정 H0 50:50 결합은 장시간/HF RMSE 한 항목만 미달한다. 이 결과를 통과로 표시하거나 본체 완료·공식 점수 개선으로 간주하지 않는다.','',
     '## 1. 요청한 방법 A: 저장 예측만 결합','',
     '새 학습·추론 없이 D0F1과 S1_H0_F1/H15_F1의 저장 배열을 사용했다. sample_id와 정답 배열 일치를 확인했다. 기존의 Tree60%+CNN40% 회귀와 독립 Tree50%+CNN50% 분류를 각 전문가의 출력으로 사용했다.','',
     '`share=(1-ws)*share_D0+ws*share_S1`, `score=(1-wc)*score_D0+wc*score_S1`이다. 회귀 ws는 local_fit MSE의 구속 최소해, 분류 wc는 local_fit AUC의 사전 0.0~1.0(0.1 간격) 격자에서 선택했다. 평가 패널에서 가중치를 고르지 않았다. 고정 60:40·50:50·65:35 회귀와 분류 50:50도 사전 지정 대조로만 계산했다.','',
     '| S1 전문가 | 정책 | S1 회귀 비중 ws | S1 분류 비중 wc | fit RMSE | fit AUC |','|---|---|---:|---:|---:|---:|']
 for c in r['candidates']:md.append(f"| {c['arm']} | {c['policy']} | {c['ws']:.4f} | {c['wc']:.2f} | {c['fit_metrics']['rmse']:.6f} | {c['fit_metrics']['auc_classifier']:.6f} |")
 md+=['','### 평가 결과 (RMSE / 독립 score AUC)','',
      '| 전문가·정책 | 별도 배경 | 별도 참조 C′ | 미관측 사이트 | native12/HF | native12/SF | 전 관문 |','|---|---:|---:|---:|---:|---:|---|']
 panels=('local_gate','local_report','domain_holdout','native12_VS_hf_high_negative','native12_VS_st_francis_dense_negative')
 ix={(v['arm'],v['policy'],v['panel']):v for v in r['rows']}
 for c,g in zip(r['candidates'],r['gates']):
  a=[ix[c['arm'],c['policy'],p]for p in panels]
  md.append('| '+c['arm']+' / '+c['policy']+' | '+' | '.join(f"{v['rmse']:.6f} / {v['auc_classifier']:.6f}"for v in a)+f" | {'통과'if g['passed']else'미달'} |")
 md+=['','**방법 A는 전 관문을 동시에 통과하지 못했다.** local_fit은 두 조건 모두 S1 회귀 비중 0을 선택했다. H0 50:50 회귀 결합은 일반 세 축을 보존했지만 native12/HF RMSE=0.117553으로 기준 0.11에 미달한다.','',
      '### 고정 가중평균으로 전 관문을 만족하는 다른 비율이 존재하는가?','',
      '저장 예측 오차를 `e(a)=e_D0+a*(p_S1-p_D0)`로 두면 `MSE(a)=A*a²+B*a+C`다. 각 RMSE 제한의 해 구간을 계산했다. 아래는 결과 진단용 부등식 해석이며 이 구간으로 정책을 다시 선택하지 않았다.','',
      '| 전문가 | 장시간/HF 기준이 요구하는 최소 S1 비중 | 별도 참조 기준이 허용하는 최대 S1 비중 | 공통 구간 |','|---|---:|---:|---|']
 for arm,d in feasible['results'].items():
  con={c['panel']:c for c in d['constraints']};lo=con['native12_VS_hf_high_negative']['allowed_S1_weight_for_RMSE'][0];hi=con['local_report']['allowed_S1_weight_for_RMSE'][1]
  md.append(f"| {arm} | {lo:.4f} | {hi:.4f} | {'있음'if d['RMSE_and_negative_share_feasible']else'없음'} |")
 md+=['','MSE의 볼록성은 혼합 MSE가 개별 MSE의 가중 평균 이하임을 보장한다. 모든 패널에서 더 좋은 모델을 동시에 이기거나 각 임계값을 만족함을 보장하지 않는다. AUC에는 같은 볼록성 보장도 없다. 위 두 구간의 충돌만으로 각 D0+S1 고정 회귀 비율 하나가 전 관문을 통과할 수 없음을 확인했다.','',
      '## 2. 요청한 방법 B: 자연 support를 유지한 관측 총에너지 정규화','',
      '원래 음성 800건은 H0/D0 배열 그대로 유지했다. 양성은 S1_H0_F1의 원천·궤적·마스크·LF/HF 상대 스케일을 유지하고, 각 행의 관측 추가 에너지에 공통 배율을 곱했다. 목표는 동일 pair의 D0 share이며 Test 값이나 라벨 분포로 정하지 않았다.','',
      '`k = y_D0/(1-y_D0) * sum(OB)/sum(OA_S1)`',
      '`A_new = k*A_S1; E_new = B+OA_new; q_new = OA_new/E_new; y_new = sum(OA_new)/sum(OE_new)`','',
      '정답 수치를 독립적으로 덮어쓰지 않고 물리 성분에서 다시 계산했다. raw dB는 원래 공통 offset을 유지하고 관측 입력만 0.1dB로 반올림했다.','']
 if (TM/'DATA_READY.json').exists():
  v=read(TM/'DATA_READY.json')['statistics']
  md+=['| 항목 | 실측 |','|---|---:|',f"| TRAIN / 양성 / 음성 | {v['rows']} / {v['positive']} / {v['zero']} |",
       f"| share 평균 / RMS | {v['target_mean']:.9f} / {v['target_rms']:.9f} |",f"| D0 share와 최대 차이 | {v['target_difference_from_D0_max']:.3e} |",
       f"| 양성 활성 중앙값 / 12분 이상 건수 | {v['active_positive_median']:.0f}분 / {v['active_ge12']}건 |",'| 원래 음성 배열 / 양성 support 보존 | 모두 일치 |','']
 result=TM/'evaluation/RESULTS.json'
 if result.exists():
  v=read(result);arm='S1_TM_H0_F1';vi={(a['panel'],a['arm'],a['model']):a for a in v['rows']}
  md+=['### 방법 B 최종 고정 평가','', '| 조건 | 별도 배경 | 별도 참조 C′ | 미관측 사이트 | native12/HF | native12/SF |','|---|---:|---:|---:|---:|---:|']
  oldpanels=tuple(p+'__POOLED'if p in panels[:3]else p for p in panels)
  base=[oi[p,'D0F1','fixed60T40N']for p in oldpanels]
  md.append('| 기존 D0F1 | '+' | '.join(f"{a['rmse']:.6f} / {a['auc_classifier']:.6f}"for a in base)+' |')
  for method in ('fixed60T40N','reg_plus_D0F1score'):
   a=[vi[p,arm,method]for p in oldpanels];md.append('| 타깃 보존 / '+method+' | '+' | '.join(f"{x['rmse']:.6f} / {x['auc_classifier']:.6f}"for x in a)+' |')
  md+=['',f"**방법 B 종합 관문: {'통과 후보 있음'if v['any_gate_passed']else'동시 통과 후보 없음'}.**",'']
  for g in v['candidate_gates']:md.append('- '+g['method']+': '+('통과'if g['passed']else'미달 — '+', '.join(c['name']for c in g['checks']if not c['passed'])))
  md+=['','### St Francis 무추가 음성 보존','',
       '| 조건 | share 예측 평균 | D0 대비 변화 | 기준 ≤ 0.058595 |','|---|---:|---:|---|',
       '| D0F1 | 0.048595 | 기준 | 통과 |',
       '| 방법 A: D0 + H0 50:50 | 0.057887 | +0.009293 | 통과 |',
       '| 방법 B: 타깃 보존 S1 | 0.077305 | +0.028710 | 미달 |','',
       '방법 B는 4,000건 TRAIN으로 100epoch/Adam/early stopping 없음 학습을 완료했다. 회귀 최적 epoch16, 분류 최적 epoch30을 local_fit에서 각각 선택했다. 학습 468.1초, 동결 평가 61.7초가 걸렸다. 소형 신경망은 회귀 28,249개·독립 분류 5,041개 파라미터이며 본체 학습 결과가 아니다.']
  md+=['','## 3. 원인 해석과 본체 실행 상태','',
       '타깃 주변분포의 상승은 확인된 변화지만, 모든 일반 패널에서 양의 편향이 생겼다는 주장은 실제 저장 결과와 다르다. 별도 배경 bias는 D0F1 -0.006394 → 원래 S1_H0 -0.013099로 더 음수였다. SF 무추가 share 상승과 모든 일반 창의 과대추정을 같은 현상으로 단정할 수 없다. 방법 B는 이 타깃 상승을 실제로 제거했을 때 나머지 성능까지 회복되는지를 직접 검증한 실험이다.','',
       '본체 재개는 작은 검증이 자동 성공한다는 전제 대신 위 실측과 명시된 동시 충족 조건으로 판단한다. 기존 8채널 epoch27 체크포인트에 17채널 F1 입력을 바꿔 이어 학습하지 않는다. 데이터·입력 규격이 달라지면 새 버전과 새 학습이 필요하다.','',
       '**현재 상태:** 기존 Tree5와 CNN epoch27은 보존·정지 상태다. 새 본체 학습 및 submission_r16.csv 생성은 미실행이다. 사용자 지시의 ‘전 관문 통과 후 본체 진입’과 ‘이번 턴 본체 완주’가 실제 미달 결과에서 충돌하므로, 미달을 명시한 진행 또는 동시 통과 기준 유지 중 우선순위를 질문한 상태다. 답변 없이 기준을 낮추거나 새 장시간 학습을 시작하지 않는다.']
 else:
  md+=['방법 B는 별도 100epoch/Adam 학습 및 동결 후 평가 중이다. 아직 성공/실패를 선결론으로 표시하지 않는다.']
 md+=['','## 4. 독립 재검산과 본체 입력 사전 검사','']
 for path,label,key in [(OUT/'RECOMPUTED_QA.json','방법 A','maximum_absolute_error'),(TM/'RESULTS_RECOMPUTED_QA.json','방법 B','maximum_metric_absolute_error')]:
  if path.exists():
   qa=read(path);md.append(f"- {label}: {qa['recomputed_rows']}개 지표 행 재계산, 최대 차이 {qa[key]:.3e}, {qa['status']}.")
 md+=['- 두 평가의 총 132개 지표 행을 재계산했다. 방법 B의 52개 동결 산출물과 예측20파일을 확인했다. 이 재검산 단계에는 새 학습·추론·Test 사용이 없다.']
 for device in ('CPU','MPS'):
  path=OUT/f'FULL_F1_PREFLIGHT_{device}.json'
  if path.exists():
   qa=read(path)
   md+=['',f"### {device} 본체 F1 연결 검사: {qa['status']}",'',
        '| 본체 | 회귀 파라미터 | 독립 분류 파라미터 | CPU/장치 최대 차이 | 저장·재로드 최대 차이 |','|---|---:|---:|---:|---:|']
   for row in qa['rows']:md.append(f"| {row['kind']} | {row['regression_parameters']:,} | {row['classification_parameters']:,} | {row['CPU_device_max_abs']:.3e} | {row['checkpoint_reload_max_abs']:.3e} |")
   md+=['','실제 TRAIN의 음성1건·자연 장시간 양성1건으로 4개 본체의 순전파·역전파·일회성 Adam 업데이트·물리 풀링·누락 셀0·독립 분류 gradient 분리·F1 입력 gradient·저장/재로드를 확인했다. 검사용 가중치는 임시 폴더에서 삭제했고 본체 학습 재개 또는 성능 검증으로 세지 않는다. MPS 결과가 있는 경우 자동 CPU fallback을 끈 검사다.','',f'[검사 영수증]({path.name})']
 md+=['','## 재현 파일','',
      '[방법 A 실행 코드](../../ops/r16_multisupport_blend.py) · [A 사전 계획](PROTOCOL.json) · [fit에서 고정한 정책](POLICY_FROZEN.json) · [A 전체 결과](RESULTS.json) · [고정 가중치 해 구간](CONSTANT_WEIGHT_FEASIBILITY.json)', '',
      '[방법 B 실행 코드](../../ops/r16_target_matched_support.py) · [B 사전 계획](../r16_target_matched_support_r1/PROTOCOL.md) · [B 데이터 재검산](../r16_target_matched_support_r1/DATA_RECOMPUTED_QA.json) · [B 지표 재검산](../r16_target_matched_support_r1/RESULTS_RECOMPUTED_QA.json)', '',
      '[새 F1 본체 연결 코드](../../pipeline/models_r16_f1_preflight.py) · [본체 입력 검사 코드](../../ops/verify_r16_f1_full_models.py) · [A 지표 재검산](RECOMPUTED_QA.json)', '',
      'Test/공식 제출 사용 0회. 원천 분리와 24시간 purge를 계승했다. 독립 장시간 평가는 12분 컴포넌트 하나에 제한되며 개발 패널은 반복 노출됐다. 이 수치를 공식 LB 또는 완전한 분포 간극 해소로 환산하지 않는다.']
 (OUT/'MULTISUPPORT_TARGET_MATCHED_RESULTS.md').write_text('\n'.join(md)+'\n')
 # Minimal Markdown-to-HTML for local review: escape before emitting tables.
 blocks=[];table=False
 for line in md:
  if line.startswith('|'):
   if set(line.replace('|','').replace(':','').strip())=={'-'}:continue
   if not table:blocks.append('<table>');table=True
   blocks.append('<tr>'+''.join('<td>'+inline(a.strip())+'</td>'for a in line.strip('|').split('|'))+'</tr>');continue
  if table:blocks.append('</table>');table=False
  if line.startswith('#'):
   n=len(line)-len(line.lstrip('#'));blocks.append(f'<h{n}>'+html.escape(line[n:].strip())+f'</h{n}>')
  elif line:blocks.append('<p>'+inline(line)+'</p>')
 if table:blocks.append('</table>')
 page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>R16 결합·타깃 보존 실측</title><style>body{font:16px/1.7 system-ui;max-width:1450px;margin:40px auto;padding:0 24px;background:#f7f9fc;color:#203049}table{width:100%;border-collapse:collapse;background:white;font-variant-numeric:tabular-nums;font-size:14px}td{padding:10px;border-bottom:1px solid #d7dfeb}h1,h2{color:#173650}p{max-width:1000px}</style><p><a href="MULTISUPPORT_TARGET_MATCHED_RESULTS.md">전체 MD 파일 열기</a></p>'+''.join(blocks)+'</html>'
 (OUT/'index.html').write_text(page)
 print(json.dumps({'report':str(OUT/'index.html'),'method_B_complete':result.exists()}))
if __name__=='__main__':main()
