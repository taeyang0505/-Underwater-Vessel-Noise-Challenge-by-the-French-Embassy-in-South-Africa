"""Saved R20 hard-panel outputs only: stratified metrics and Korean report.

No model fitting, inference, thresholds selected from outcomes, or Test reads.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops.r20_frozen_hard_inference import metrics
from ops.r20_hard_panels import guard
OUT=ROOT/'reports/r20_integrated_cpu_diagnostics_r1/hard_panels'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 sys.addaudithook(guard)
 if not(OUT/'inference/COMPLETED.json').exists():raise ValueError('Frozen inference incomplete')
 rows=pd.read_csv(OUT/'inference/ALL_INDIVIDUAL_AND_BLEND_ROW_PREDICTIONS.csv')
 phys=pd.read_csv(OUT/'ALL_PANEL_PHYSICAL_ROWS.csv');q=rows.merge(phys[['sample_id','hf_2to8_bump_db','bump_high_db','bump_low_db','LF_energy_fraction']],on='sample_id',validate='one_to_one')
 allm=pd.read_csv(OUT/'inference/ALL_INDIVIDUAL_AND_BLEND_METRICS.csv');summaries=[]
 models=['old_B1','new_B2']+[f'{g}_{f}'for g in ['old','new']for f in ['lightgbm','dinov2_small','dasheng_base']]
 # Predetermined source criteria from generation protocol, never outcome-selected.
 for panel in ['hardneg1_natural','hardneg3_full15']:
  d=q[q.panel.eq(panel)];z=d.physical_presence.astype(bool)
  requested_OR=(d.bump_high_db.ge(12)|d.bump_low_db.ge(8)|d.LF_energy_fraction.le(.65))
  for label,nmask in [('all_natural_negative',~z),('requested_HF3to8_OR_LFbump_OR_LFfraction',(~z)&requested_OR),
    ('HF3to8_bump>=12',(~z)&d.bump_high_db.ge(12)),('HF2to8_bump>=12',(~z)&d.hf_2to8_bump_db.ge(12)),
    ('LF31to501_bump>=8',(~z)&d.bump_low_db.ge(8)),('LF_fraction<=.65',(~z)&d.LF_energy_fraction.le(.65))]:
   ix=z|nmask
   if not nmask.any():continue
   for model in models:
    summaries.append({'panel':panel,'negative_subset':label,'negative_n':int(nmask.sum()),'model':model,
      **metrics(d.target_share[ix],z[ix],d[model+'_share'][ix],d[model+'_score'][ix])})
 hf=pd.DataFrame(summaries);hf.to_csv(OUT/'HARDNEGATIVE_SUBGROUP_CROSS_METRICS.csv',index=False)
 h1=q[q.panel.eq('hardneg1_natural')];req=(h1.bump_high_db.ge(12)|h1.bump_low_db.ge(8)|h1.LF_energy_fraction.le(.65))
 excluded=h1[h1.physical_presence.eq(0)&~req]
 excluded[['sample_id','base_window_index','site','recorder','bump_high_db','hf_2to8_bump_db','bump_low_db','LF_energy_fraction']].to_csv(OUT/'HF_BAND_DEFINITION_EXCLUDED_FIVE.csv',index=False)
 # Same B/reference/geometry under each scale law. Per-source paired deltas, not a generator fit.
 d=rows[rows.panel.eq('scale_laws')&rows.target_share.gt(0)];paired=[]
 for model in models:
  base=d[d.law.eq('native_peak')].set_index('base_window_index')
  for law in ['integrated_target','HF_SNR_scalar']:
   a=d[d.law.eq(law)].set_index('base_window_index').loc[base.index]
   if not np.array_equal(a.reference_index,base.reference_index):raise ValueError('Paired reference differs')
   paired.append({'model':model,'law':law,'n':len(a),'target_mean':float(a.target_share.mean()),'target_rms':float(np.sqrt(np.mean(a.target_share**2))),
    'target_delta_vs_peak':float((a.target_share-base.target_share).mean()),'share_delta_vs_peak':float((a[model+'_share']-base[model+'_share']).mean()),
    'score_delta_vs_peak':float((a[model+'_score']-base[model+'_score']).mean()),
    'rmse':float(np.sqrt(np.mean((a[model+'_share']-a.target_share)**2))),
    'bias':float((a[model+'_share']-a.target_share).mean())})
 pd.DataFrame(paired).to_csv(OUT/'SAME_SOURCE_SCALE_PAIR_EFFECTS.csv',index=False)
 overall=allm[(allm['slice']=='all')&allm.model.isin(['old_B1','new_B2'])]
 physical=overall[overall.label_semantics.eq('physical_added_recording')]
 cf=overall[~overall.label_semantics.eq('physical_added_recording')]
 columns=['panel','model','n','target_mean','share_mean','rmse','bias','score_AUC','share_AUC','score_ge090_fraction']
 details=allm[(allm.label_semantics.eq('physical_added_recording'))&allm.model.isin(models)&allm['slice'].eq('all')]
 ds=details[['panel','model','rmse','bias','score_AUC','score_PR_AUC','brier','ece10']]
 scale=allm[allm.panel.eq('scale_laws')&allm.model.isin(['old_B1','new_B2'])&allm['slice'].str.startswith('law:')]
 hold=json.loads((OUT/'SOURCE_HOLDOUT_AUDIT.json').read_text());amb=json.loads((OUT/'AMBIENT_AUDIT.json').read_text());ref=json.loads((OUT/'REFERENCE_TRAINING_EXPOSURE_SUPPLEMENT.json').read_text())
 ph=pd.read_csv(OUT/'PANEL_PHYSICAL_SUMMARY.csv')
 phcols=['panel','kind','law','n','share_mean','LF_energy_fraction_p50','bump_high_db_p50','bump_low_db_p50']
 h=hf[hf.model.isin(['old_B1','new_B2'])][['panel','negative_subset','negative_n','model','rmse','bias','score_AUC','share_AUC']]
 requested=h[h.panel.eq('hardneg1_natural')&h.negative_subset.eq('requested_HF3to8_OR_LFbump_OR_LFfraction')]
 requested.to_csv(OUT/'HARDNEG1_REQUESTED_DEFINITION_PRIMARY.csv',index=False)
 strength=[]
 for panel in ['hardneg1_natural','hardneg3_full15']:
  d=q[q.panel.eq(panel)];y=d.target_share;z=d.physical_presence.astype(bool)
  for label,positive in [('0<y<=.03',z&y.le(.03)),('.03<y<=.10',y.gt(.03)&y.le(.10)),('.10<y<=.30',y.gt(.10)&y.le(.30)),('y>.30',y.gt(.30))]:
   ix=(~z)|positive
   if not positive.any():continue
   for model in models:
    strength.append({'panel':panel,'positive_strength':label,'positive_n':int(positive.sum()),'negative_n':int((~z).sum()),'model':model,
     **metrics(y[ix],z[ix],d[model+'_share'][ix],d[model+'_score'][ix])})
 pd.DataFrame(strength).to_csv(OUT/'POSITIVE_STRENGTH_VS_NATURAL_NEGATIVE.csv',index=False)
 parity=[]
 for p in(OUT/'inference').glob('*_*.json'):
  if p.name not in ['old_dinov2_small.json','new_dinov2_small.json','old_dasheng_base.json','new_dasheng_base.json']:continue
  v=json.loads(p.read_text())
  parity.extend([{'model':p.stem,**r}for r in v.get('CPU_vs_historical_precision_probe',[])])
 pd.DataFrame(parity).to_csv(OUT/'CPU_FP32_VS_HISTORICAL_BF16_PARITY.csv',index=False)
 maxdiff=max(v['max_abs']for v in parity)if parity else None
 text=f'''# R20 네 종류 원천 분리 Hard 패널·스케일 진단

학습·정책 적합·Test 조회 없이, 사전 동결된 4,000행(고유 db 3,150행)에 old/new LGB·DINO·Dasheng 동결 가중치를 CPU FP32로 적용했다. old_B1은 60/25/15%, new_B2는 50/30/20%, 각 D0:H0는 50:50이다. score는 독립 분류 확률의 같은 고정 비중이며 share를 섞지 않았다. 분류 임계값 0.5/0.9는 진단 통계이며 이번 결과로 새 정책을 고르지 않았다.

## 1. 주요 결과: 공식 타깃 정의

아래 AUC는 새로 추가한 **녹음 전체**의 존재 여부이며, 자연 배경 속 실제 선박 유무를 묻는 지표가 아니다.

{physical[columns].to_markdown(index=False,floatfmt='.6f')}

HardNeg1 주 판정표는 사용자 요청의 `bump_high_db=3~8kHz` 정의를 엄밀히 적용한 **495음성+고정500양성=995행**이다. 최초 동결 패널은 `2~8kHz` 기준을 썼으며 5개 음성만 요청 OR 조건에서 벗어났다. 원본 패널과 SHA 및 추론을 바꾸지 않고 정의를 정확히 맞춘 하위군 평가다. 제외5개 ID·물리값은 `HF_BAND_DEFINITION_EXCLUDED_FIVE.csv`에 보존했다. 위 1,000행 전체는 보조 결과이고 아래995행이 요청 정의 주 결과다.

{requested.to_markdown(index=False,floatfmt='.6f')}

각 패널은 의도적으로 실패 조건을 강화한 진단 집합이다. AUC가 0.55 근처로 내려왔는지 여부를 데이터·SNR·후처리 선택 목표로 삼지 않았다. 이 결과만으로 공식 Test의 원인이나 점수를 확정할 수 없다.

## 2. 배경·참조 원천 독립성 및 적용 범위

- 배경은 원본 domain_holdout {hold['domain_total']:,}창 중 원래 경계분 QC 후 Nobs≥13인 {hold['domain_dense_Nobs_ge13']:,}창에서 선택했다. 자연활성 OR 조건 적격은 {hold['natural_active_eligible']:,}창, HF bump<2 조용한 적격은 {hold['quiet_HFbump_lt2_eligible']:,}창이었다.
- old LGB 5-fold TRAIN 합집합 {hold['old_LGB5_union_backgrounds']:,}창과 new TRAIN {hold['new_R18_backgrounds']:,}창에 대해 배경 ID·사이트·recorder·sensor_group 중복은 모두 0이다. 공통 센서가 없으므로 같은 센서의 24시간 이내 support 재사용도 없다. 물리 기기 시리얼 번호 독립까지 증명한 것은 아니다.
- 자연활성 음성 500개의 HF≥12/LF bump≥8/LF비율≤.65 조건 수는 각각 {hold['natural_negative_criteria_counts']['HF_ge12']}/{hold['natural_negative_criteria_counts']['LFbump_ge8']}/{hold['natural_negative_criteria_counts']['LF_fraction_le065']}개로 겹칠 수 있다. 전체 OR 표본을 전부 고HF 음성으로 부르지 않는다. 선택 사이트 구성은 {json.dumps(hold['selected_natural_site'],ensure_ascii=False)}다. Saldhana 쏠림이 있어 사이트 전체 대표성을 주장하지 않는다.
- 정상 양성·scale에 쓴 예약44 참조는 old 참조 TRAIN 합집합 {ref['old_union_reference_count']}개 및 new 실제 TRAIN {ref['new_actual_train_reference_count']}개와 원음 SHA 중복 0이다. 다만 VTUAD 전체의 hull/day 전이 연결그래프가 하나라 완전한 참조 그래프 독립은 아니다.
- domain_holdout는 과거 진단에도 쓰인 원천이다. **학습 원천 미노출**과 **연구자가 한 번도 보지 않은 잠금 최종셋**은 구별해야 한다.
- 기존 R11/R16/R18에도 HF-high 자연 음성·SF 밀집 음성이 있었다. 따라서 ‘기존 모든 y=0이 조용했다’는 전제는 틀리다. 이번은 모든 old fold의 훈련 배경과 겹치지 않는 조건에서 해당 실패 가능성을 다시 분리한 진단이다.

## 3. 자연 HF/LF 활동 음성의 하위군 교차

양성 전체와 표시한 음성 하위군을 합쳐 계산했다. 하위군별 행 수와 유병률이 다르므로 전체 패널과 RMSE/AUC를 직접 동급 비교하지 않는다.

{h.to_markdown(index=False,floatfmt='.6f')}

## 4. Ambient16 / weak-background22: 두 라벨을 혼동하면 안 됨

전용 ambient는 16개·32kHz·완전 분 {amb['ambient_minutes_total']}분(최소 {amb['ambient_complete_minutes_min']}분), weak-background는 22개이며 원음 SHA 중복 0이다. 6.31/7.94kHz 대역은 이 32kHz 원음들의 Nyquist 16kHz 아래다. 모든 2초 power cache slice SHA를 검사했고 새 원음 다운로드나 재변환을 하지 않았다. ZIP 경로·SHA·16개 멤버 목록 및 CC-BY-4.0 출처는 별도 CSV/JSON에 있다.

old_B1의 실제 참조 TRAIN에는 weak-background22가 0개였고 new_B2에는 22개 모두 들어 있었다. 전용 ambient16은 양쪽에서 제외됐다. ambient를 더하면 `physical_share>0`이고 `physical_presence=1`이다. ‘배가 없는 참조’라고 가정해 `vessel_label=0`으로 부르는 것은 **별도 가상 문제**이며, 약한 background class 자체도 실제 선박 부재를 증명하지 않는다.

아래는 그 가상 라벨을 썼을 때의 민감도다. **공식 성능 AUC/RMSE로 해석 금지**:

{cf[columns].to_markdown(index=False,floatfmt='.6f')}

같은 B·같은 support·같은 물리 share를 가진 ship/ambient/weak-background 쌍 차이는 `MATCHED_AMBIENT_MINUS_SHIP.csv`에 있다. 전체 추가 녹음 에너지를 추정하는 현재 타깃에서는 ambient와 ship의 비슷한 share 예측이 올바를 수도 있다. 비슷한 출력만으로 장비 편법을 확정할 수 없다.

## 5. Full15 정적 스트레스와 세 가지 scale 법칙

Full15 양성은 원음 평균 스펙트럼을 15분 일정하게 반복 배치한 **인공 식별성 스트레스**다. 자연 장시간 통과 재현이나 Test 생성법으로 주장하지 않는다. 원래 조용한 B를 고르고 최종 bump를 측정했으며, 모델 출력·AUC를 보고 표본을 다시 뽑지 않았다. 정상/장시간/음성 특성은 다음과 같다.

{ph[phcols].to_markdown(index=False,floatfmt='.6f')}

scale 세 법칙은 같은 B·참조·crop·위치를 사용한다. native peak는 SNR U[-10,10]dB, integrated는 사전 Beta(1.4,6.6)(모집단 평균 .175), HF SNR은 U[5,20]dB다. 모두 한 스칼라 alpha만 써서 주파수 비율을 변형하지 않았다. integrated의 목표 평균은 사용자 지시의 **진단용 toy prior**이며 Test/LB에서 역산하거나 적합한 값이 아니다.

{scale[['model','slice','n','target_mean','share_mean','rmse','bias','score_AUC']].to_markdown(index=False,floatfmt='.6f')}

## 6. 개별 모델 및 수치적 한계

{ds.to_markdown(index=False,floatfmt='.6f')}

CPU FP32와 역사적 CUDA BF16은 가중치와 특징이 같아도 소규모 차이가 생긴다. 각 branch 첫32행 probe의 최대 절대 차이는 {maxdiff}이며 branch별 표는 `CPU_FP32_VS_HISTORICAL_BF16_PARITY.csv`다. 이번 비교는 동일 CPU 정밀도에서 수행했다. 모델·어댑터 SHA 및 frozen backbone fingerprint를 로드 전후 검사했다.

이 실험은 추가 학습 0회·optimizer step 0·정책 선택 0회·Test 조회 0회다. 원본 모델, 기존 CSV, 합성 코드, 학습 데이터 및 원음은 수정하지 않았다.
'''
 (OUT/'R20_HARD_PANEL_DIAGNOSTICS.md').write_text(text)
 files=[p for p in OUT.rglob('*')if p.is_file()and p.name not in ['ARTIFACT_SHA256.json','REPORT_COMPLETED.json','inference_run.log','report_run.log','progress.json']]
 manifest={str(p.relative_to(OUT)):{'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(files)}
 (OUT/'ARTIFACT_SHA256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 (OUT/'REPORT_COMPLETED.json').write_text(json.dumps({'status':'completed','report_sha256':sha(OUT/'R20_HARD_PANEL_DIAGNOSTICS.md'),'artifact_count':len(manifest),'script_sha256':sha(__file__)},indent=2)+'\n')
 print(physical[columns].to_string(index=False))
if __name__=='__main__':main()
