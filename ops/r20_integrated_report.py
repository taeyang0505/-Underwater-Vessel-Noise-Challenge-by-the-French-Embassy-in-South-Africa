"""Assemble completed R20 CPU diagnostic outputs; no model or Test access.

Requires the four independently frozen branches. Does not change inference
policy, choose a submission, train a model, or optimize any public-score fit.
"""
from __future__ import annotations
import datetime as dt, hashlib, html, json
from pathlib import Path
import pandas as pd
import numpy as np
import markdown
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r20_integrated_cpu_diagnostics_r1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(d):
 d=d.copy()
 for c in d:
  if d[c].dtype=='object':d[c]=d[c].map(lambda x:x.replace('|',' / ') if isinstance(x,str) else x)
 return d.to_markdown(index=False,floatfmt='.6f')
def verify_map(folder,mapping):
 checks=[]
 for name,v in mapping.items():
  expected=v if isinstance(v,str) else v.get('sha256')
  if not expected:continue
  p=folder/name
  if not p.exists():p=ROOT/name
  actual=sha(p)
  checks.append({'path':str(p.relative_to(ROOT)),'expected':expected,'actual':actual,'matches':actual==expected})
 return checks
def main():
 a=OUT/'alignment';h=OUT/'hard_panels';p=OUT/'physical_profiles';q=OUT/'quiet_features'
 required=[a/'COMPLETED.json',p/'COMPLETED.json',q/'COMPLETED.json',h/'R20_HARD_PANEL_DIAGNOSTICS.md',h/'inference/FOUR_GROUP_BLEND_METRICS.csv',h/'HARDNEGATIVE_SUBGROUP_CROSS_METRICS.csv']
 for f in required:
  if not f.exists():raise FileNotFoundError('Wait for completed diagnostic '+str(f))
 stamp=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec='seconds')
 cov=read(a/'COVERAGE_AND_SEARCH_ACCOUNTING.json')
 score=pd.read_csv(a/'OFFICIAL_PUBLIC_SCORES_SUPPLIED.csv')
 b=pd.read_csv(a/'B1_B2_KEY_PANEL_SUMMARY.csv')
 key=pd.read_csv(a/'OFFICIAL4_KEY_SLICES.csv')
 post=pd.read_csv(a/'PUBLIC_ALIGNMENT_POSTHOC_ALL_TRIALS.csv')
 hard=pd.read_csv(h/'inference/FOUR_GROUP_BLEND_METRICS.csv')
 individual=pd.read_csv(h/'inference/ALL_INDIVIDUAL_AND_BLEND_METRICS.csv')
 requested=pd.read_csv(h/'HARDNEG1_REQUESTED_DEFINITION_PRIMARY.csv')
 physicalpanels=pd.read_csv(h/'PANEL_PHYSICAL_SUMMARY.csv')
 inversion=pd.read_csv(h/'inference/PAIRWISE_RANK_INVERSION.csv')
 sub=pd.read_csv(h/'HARDNEGATIVE_SUBGROUP_CROSS_METRICS.csv')
 phy=pd.read_csv(p/'test_descriptive_only/DESCRIPTIVE_COMPARISON_MEDIANS.csv')
 quiet=pd.read_csv(q/'QUIET_ESTIMATOR_METRICS_ALL_SLICES.csv')
 corr=pd.read_csv(q/'TOP30_RESIDUAL_FEATURES_BY_PANEL.csv')
 counts=read(p/'dataset_counts.json');freeze=read(q/'COEFFICIENT_FREEZE.json')
 qa=read(q/'INDEPENDENT_NUMERIC_QA.json')
 # Copy compact summary tables only; all raw arrays retain their original SHA.
 for name,df in [('PUBLIC_SCORE_AND_LOCAL_SCOPE.csv',score),('R20_PHYSICAL_MEDIAN_COMPARISON.csv',phy),('R20_HARD_PANEL_BLEND_SUMMARY.csv',hard)]:df.to_csv(OUT/name,index=False)
 wide=key[key['slice'].isin(['all','zero','weak_000_010','zero_HF_ge12'])].copy()
 wide.to_csv(OUT/'R20_FOUR_PUBLIC_MODELS_LOCAL_SLICE_SUMMARY.csv',index=False)
 qb=quiet[(quiet['slice']=='all')&quiet.estimator.isin(['old_B1_saved','quiet_K2_TRAIN_median_ratio','quiet_K3_TRAIN_median_ratio','quiet_K5_TRAIN_median_ratio'])].copy()
 qb.to_csv(OUT/'R20_QUIET_ESTIMATOR_SUMMARY.csv',index=False)
 # Public ranking is reported post hoc. No optimization or policy fitting here.
 valid=post[post.n_matched.ge(4)&post.spearman_public_rmse.notna()]
 best=valid.sort_values(['spearman_public_rmse','pearson_public_rmse'],ascending=False).head(8)
 weak=wide[(wide.panel=='local_report')&wide.model.isin(['R17_B1','R18_B2'])]
 cols=[x for x in ['panel','model','label_semantics','slice','n','rmse','bias','score_AUC','share_AUC','score_mean','score_ge090_fraction'] if x in hard]
 hm=hard[hard.model.isin(['old_B1','new_B2'])].copy()
 # Primary hard-panel rows include explicit slice/semantic names, never silently
 # replace physical labels by the counterfactual vessel-presence labels.
 hm=hm[hm['slice'].eq('all')] if 'slice' in hm else hm
 hm.to_csv(OUT/'R20_OLD_B1_NEW_B2_HARD_PANEL_CORE.csv',index=False)
 zero_hard=individual[individual.model.isin(['old_B1','new_B2'])&individual['slice'].eq('y=0')&individual.label_semantics.eq('physical_added_recording')]
 scale_hard=individual[individual.model.isin(['old_B1','new_B2'])&individual.panel.eq('scale_laws')&individual['slice'].str.startswith('law:')]
 scale_physical=physicalpanels[physicalpanels.panel.eq('scale_laws')]
 targetcols=['dataset','LF_31to200_energy_fraction','MF_251to2512_energy_fraction','HF_3162to7943_energy_fraction','HF_3162to7943_bump_db','low_31to501_bump_db','LF_31to200__HF_3162to7943_corr']
 native=qb[qb.panel.isin(['local_gate','domain_holdout','native12_VS_hf_high_negative','vanilla_addition'])]
 featuretop=corr[corr.panel.isin(['local_gate','domain_holdout','native12_VS_hf_high_negative','vanilla_addition'])].groupby('panel',sort=False).head(5)
 s=[]
 s+=['# R20 로컬 검증·고난도 대조·정밀 물리 프로파일 통합 CPU 진단',f'\n완료 시각: {stamp}\n']
 s+=['## 1. 결론과 실행 범위\n',
 '기존 LightGBM·DINOv2-small·Dasheng-Base를 동결한 상태로 4대 분석을 실행했다. **Public과 1:1로 정렬된 Val Loss를 확립하거나 실전 간극을 해결했다고 결론 내릴 근거는 아직 없다.** 고난도 조건에서 성능이 바뀌는 양상과 기존 검증의 대표성 부족을 구분해 수치로 보존했다. 모델 재학습·GPU/MPS·RunPod 사용·공식 제출·새 제출 CSV 생성·생산 예측 보정은 하지 않았다.\n',
 '분석1은 저장 예측만 집계했다. 분석2는 사전 동결한 새 패널 4,000행에 CPU FP32 동결 추론만 수행했다. 분석3은 저장 Test 입력을 포함한 기술 통계이며 Test 라벨/출처 매칭/적합은 없다. 분석4는 순수 TRAIN SA 배경에서 보정계수 c를 통계 추정했다. 따라서 전체 작업을 ‘아무 적합도 없음’이라 표현하지 않는다. 예측 모델의 optimizer step은 0이다.\n',
 '| 분석 | 실제 실행 및 상세 산출물 |\n|---|---|\n| 1. 저장 검증 지표·Public 관계 | 16패널, 384개 유효 모델·패널 조합, 14슬라이스, 5,376행 지표. 모델 명칭 34개는 별칭·결합을 포함하며 34개 독립 학습 모델이 아니다. |\n| 2. 고난도 패널 | 4그룹 각1,000행, 합계4,000행. 동일 관측 재사용 후 고유3,150입력. 정확한 요청 밴드 기준은 하위군으로 별도 검사. |\n| 3. 물리 프로파일 | 10개 자료 전수. SA 적격 256,029창, primary QC 78,869창, VTUAD509+ambient16, R16/R18 D0/H0, vanilla2,000, Test1,500. |\n| 4. Quiet-B·물리 잔차 | TRAIN 배경11,700창에서 K=2/3/5 계수 고정, 평가8패널, 사전 정의643특징 상관, 패널별 상위30개 보존. |\n']
 s+=['이번 고난도 결과의 핵심은 **자연 활성 배경만으로는 분류 붕괴가 재현되지 않았고**, ambient/weak-background를 음성으로 재정의한 가상 문제와 인위적 평탄15분 조건에서는 붕괴가 나타났다는 점이다. 원래 추가 에너지 정의에서 자연 HardNeg1 AUC는 old/new 약0.939/0.972, ambient 패널은0.962/0.984였다. 같은 ambient 출력을 가상 음성으로 바꾸면0.624/0.623, 인위적 평탄15분에서는0.191/0.346이었다. 따라서 낮은 수치 자체가 공식 Test 원인 재현의 증거는 아니다. 정확한3~8kHz 요청정의995행 값은 아래별도표를우선한다.\n']
 s+=['## 2. 공식 제출 순위와 로컬 지표의 정합성\n',table(score),'\n',
 '6개 공식 제출을 같은 로컬 패널에서 모두 비교할 저장 예측은 없었다. R16 Tree5/V1/R17 B1/R18 B2 4종은 현대 고정5패널에서 공통 비교된다. R9 all5는 별도 과거500행 fit/gate이고 R9 Tree5는 적격 라벨 예측이 없어 누락으로 남겼다. 다른 표본을 억지로 연결하거나 R9 재추론으로 빈칸을 채우지 않았다.\n',
 'Public RMSE와의 관계는 14개 단독 슬라이스 및4개 동일 가중 조합을 패널별로 검사한 **사후 기술 통계**다. 총288회 중4모델 이상 공통인88회에서 최고 Spearman=0.8이며 완전한 순위 일치는 없다. 한정 후보의 관측 최댓값이고 N=4·공유 베이스 모델·다중 비교이므로 새로운 검증 손실로 채택하지 않았다. 공식 악화량 +0.010748026을 맞추기 위한 구간 가중치 적합은 하지 않았다.\n',table(best[[x for x in ['panel','candidate','n_matched','spearman_public_rmse','pearson_public_rmse','local_order_low_error_first'] if x in best]]),'\n',
 '가장 높은 Pearson 0.964379가 나온 report의 무추가·HF≥12 구간은 **30행**에 불과하다. 해당 표본과 알려진 Public 순위를 보고 발견한 관계를, 독립적인 미래 성능 예측 법칙으로 해석하면 안 된다. 전체 RMSE의 Public 선형 상관은 낮고 오차 폭까지 재현하지 못한다.\n',
 '### R17 B1과 R18 B2의 report 세부 오차\n',table(weak[['model','slice','n','rmse','mae','bias','brier','ece10','roc_auc']]),'\n',
 'R18의 전체 로컬 성적 개선과 약한 양성/일부 무추가 구간 악화는 동시에 일어났다. 전체 평균 RMSE 하나로 채택하면 이 실패를 가릴 수 있다. R19의 다중구간 게이트와 동결 old_B1 유지 결론은 이번 진단으로 변경하지 않았다.\n',
 '### 모든 B1/B2의 패널 전체 성적\n',table(b[[x for x in ['panel','model','n','rmse','bias','roc_auc','brier','ece10'] if x in b]]),'\n',
 'RMSE·MAE·Bias·R²는 회귀 예측, Brier·ECE·ROC-AUC·PR-AUC는 독립 분류 score로 계산했다. PR-AUC는 average precision, ECE는 고정10구간이다. 단일 클래스 구간 AUC와 타깃 분산0 구간 R²는 미정의로 남긴다. native12 비교는 저장된 양성/음성50:50 표본 가중치를 유지한다. 전체 상세표는 [5,376행 지표 CSV](alignment/ALL_MODELS_PANELS_SLICES_METRICS.csv)에 있다.\n']
 s+=['## 3. 새 고난도 패널: 물리 라벨과 가상 선박 라벨을 구분\n',
 '자연 SA 무추가는 추가 녹음 A=0이므로 추가 에너지 y=0이다. 반면 ambient16이나 weak-background22를 더하면 A의 에너지가 생긴다. 이들을 ‘선박이 아닐 것’이라는 이유만으로 공식 y=0이라 부르지 않았다. **정방향 추가 에너지 라벨**과 **비선박을 음성으로 가정한 counterfactual 라벨**을 별도 평가한다. weak-background 클래스명은 클립에 실제 선박이 없음을 보증하지 않는다. 주최 측이 dedicated ambient를 실제 Test에 넣었는지와 어떤 라벨 정책을 썼는지는 공개 자료만으로 확인되지 않아 어느 가상 패널도 공식 숨은 Test의 라벨을 검증한 것으로 취급하지 않는다.\n',
 '패널 설정은 모델 예측 전에 SHA로 동결했다. AUC≈0.57을 만들도록 조건을 사후 조절하지 않았다. 다음 표의 label_semantics와 slice를 함께 읽어야 한다.\n',table(hm[cols]),'\n',
 '### HardNeg-1 밴드 정의와 하위군\n',
 table(requested),'\n',
 '### 무추가 예측 오차·양성/음성 순위 역전\n',table(zero_hard[[x for x in ['panel','model','n','rmse','bias','share_mean','score_mean','score_ge090_fraction','score_ge050_fraction'] if x in zero_hard]]),'\n',
 table(inversion[inversion.model.isin(['old_B1','new_B2'])]),'\n',
 '초기 사전등록은 2~8kHz bump≥12를 사용해 요청에서 기존에 쓰던 3~8kHz 정의와 차이가 있었다. 500개 자연 음성 중495개는 3~8kHz bump≥12 OR 31.6~501.2Hz bump≥8 OR LF에너지비중≤0.65도 만족한다. 요청 정의의 주 비교는 고정500양성+495적격음성이며, 원래1,000행 패널과 제외5개의 차이를 별도로 보존했다. 입력·추론을 바꾸지 않고 저장 출력 하위집합으로 계산했다. OR 조건에는 고LF 변동만 큰 창도 들어가므로 모든 음성이 고HF라고 해석하지 않는다.\n',
 '각 하위군의 RMSE·AUC·양성/음성 순위 역전은 [고난도 하위군 교차표](hard_panels/HARDNEGATIVE_SUBGROUP_CROSS_METRICS.csv), 개별 모델과 D0/H0는 [개별 전수 지표](hard_panels/inference/ALL_INDIVIDUAL_AND_BLEND_METRICS.csv)에 있다.\n',
 '### 원천 분리·ambient 정체·스케일 법칙\n',
 '자연 음성과 정상 추가용 배경은 old LightGBM5 TRAIN 합집합78,869창/new TRAIN12,000창과 배경ID·사이트·녹음기·센서 중복0인 domain holdout을 사용했다. 정상 추가용 예약44개 참조는 두 세대 TRAIN과 SHA 중복0이다. 전체 원음 연결 그래프의 독립까지 보장하는 것은 아니다. old 계열 TRAIN 참조 합집합75개와 new465개를 실제 사용 목록으로 확인했다.\n',
 '전용 ambient16은32kHz·533완전분으로 양쪽 TRAIN에서 제외됐고, weak-background22는 new465 TRAIN에 포함됐다. 따라서 두 원음군의 성적 차이를 무조건 새로운 원음 전이 효과로 해석할 수 없다. 원음/아카이브/바이트 중복/노출은 [원음 정체 감사](hard_panels/REFERENCE_IDENTITY_AUDIT.csv)와 [TRAIN 노출 보완](hard_panels/REFERENCE_TRAINING_EXPOSURE_SUPPLEMENT.json)에 있다.\n',
 '장시간 평탄 패널은 식별성 경계 진단이며 실제 Test의 대표 양성이라고 주장하지 않는다. peak/관측총량/고주파 SNR 스케일은 사전 고정한 정방향 가정이다. 목표 평균0.15~0.18 조건도 시험지 라벨을 추정해 맞춘 결과가 아니다. 이를 통해 어느 라벨·스케일 가정에서만 점수가 무너지는지 구분하고, 그 가정을 공식 생성 규칙으로 단정하지 않는다. 상세 레시피와 모든 수치는 [고난도 보고서](hard_panels/R20_HARD_PANEL_DIAGNOSTICS.md)에 보존했다.\n']
 s+=['### 스케일 법칙별 직접 비교\n',table(scale_physical[[x for x in ['kind','law','n','share_mean','share_p50','LF_energy_fraction_p50','bump_high_db_p50','bump_low_db_p50'] if x in scale_physical]]),'\n',table(scale_hard[[x for x in ['model','slice','n','target_mean','share_mean','rmse','bias','score_AUC'] if x in scale_hard]]),'\n']
 s+=['## 4. 25대역 × 시간 프로파일: 실제 분포 차이는 남아 있다\n',table(phy[targetcols]),'\n',
 '각 값은 **창별 지표의 중앙값**이므로 LF/MF/HF 중앙값 세 개를 더해1이 되어야 하는 것은 아니다. 개별 창의25대역 에너지 비율 합과 자료별 평균 비율 합은1로 검산했다. LF=31.6~199.5Hz, MF=251.2~2511.9Hz, HF=3162.3~7943.3Hz이며 bump_low는 별도로31.6~501.2Hz를 쓴다. 밴드 파워는 선형 합 후 dB로 요약한다.\n',
 'SA 원본1,715,779행에서 중복 clock분11개를 제외한1,715,768행을 연결했다. 비어 있지 않은15분/5분stride 창410,571개 중15분 완전 관측296,798개, 기존 row eligibility를 모두 만족한256,029개를 전수 분석했다. 주 배경풀78,869창의 경계 NaN 처리도 별도 분석했다. 중첩 창 수와 독립 녹음 수를 혼동하지 않는다.\n',
 'VTUAD509 원음은1,978완전분·클립길이 중앙값3분, ambient16은533분·길이30~37분이다. 반복·신축 없이 native 길이를 사용했다. native clip의 시간 bump와15분 합성창 bump의 차이는 관측 길이도 섞인 차이다. native 위치표에서 길이 밖 결측은 SA 녹음기 결측과 구분한다. SA dB와 비보정 디지털 원음 전력의 절대 단위는 일치가 검증되지 않아 공통 정규화·차분·상관만 비교했다.\n',
 'Test 전체 통계는 양성·음성이 섞인 주변분포이며, 이 표만으로 Test 음성의 HF bump나 실제 추가 비율/정답 평균을 알 수 없다. **전체 물리 모양의 차이는 확인됐지만, 이것만으로 어느 생성 법칙이 정답인지 또는 AUC 붕괴의 단일 원인을 확정할 수 없다.** Test 기술표는 별도 폴더에 격리했고 생성기/계수/모델 선택에 전달하지 않았다.\n',
 '자료별 25대역 p10/p50/p90, 15분 위치별 결측/에너지, 지속성0~15분 히스토그램,25×25 상관 및 slope/curvature는 [물리 프로파일 안내](physical_profiles/R20_PHYSICAL_PROFILES.md)의 CSV에 모두 있다. 상관은 유효 공통 분3개 이상에서 계산했고 상수 대역의 미정의 상관은0으로 채우지 않았다.\n']
 s+=['## 5. Quiet-Minutes 배경 복원: 고정 계수만으로 개선되지 않았다\n',
 'K∈{2,3,5}에서 다중대역 상대 레벨 순위와 실제 인접 분 변화율 순위를 같은 비중으로 더해 가장 낮은K분을 선택했다. 결측 시계 구간을 압축하지 않았고 고립된 관측분은 변화율0 및 별도 flag로 기록했다. Nobs<K면 해당 추정은NaN이다. 따라서 K=5의domain holdout 유효 표본은2,370행으로 다른K의2,400행과 다르다.\n',
 f'순수 R18 TRAIN12,000배경 중 vanilla와 정확히 같은300배경을 제외한 **{freeze["TRAIN_rows"]:,}창**에서 c[f,K]=median(full-band mean E / quiet-K mean E)를 계산·동결한 뒤 평가 라벨을 읽었다. gate/report/domain/native/fresh-fit은 계수 TRAIN과 그룹 중복0이다. vanilla는 exact중복을 제거해도161그룹 공유와 기존 모델의 학습 원천 노출이 남는 법칙 진단이다.\n',
 r'$$\widehat B_f=c_{f,K}\,\mathrm{mean}_{t\in quiet_K}E_{t,f},\qquad\widehat y=\mathrm{clip}\left(1-\frac{\sum_f n_f\widehat B_f}{\sum_{t,f}O_{t,f}E_{t,f}},0,1\right).$$'+'\n',table(native[['panel','estimator','n','rmse','bias','predictive_R2']]),'\n',
 '이 K 후보들은 기존old_B1보다 크게 나쁘고 직접 예측 R²가 음수였다. 이는 공통 단위/가중치 차원 오류가 아님을 독립 검산했다. TRAIN 고정계수는 자연 배경의 비정상성, 드문 큰 저주파 과도음, 장시간 추가 성분으로 오염된 quiet분을 해결하지 못한다. 이 방법을 물리 상한이나 자동 영점 후처리로 채택하지 않았다.\n',
 '상수 관측E=200은 B=200,A=0(y=0)과 B=100,A=100(y=0.5)에서 완전히 같다. c=1인quiet추정은 양쪽을0으로 내므로, 조용한 분의 존재를 보장하지 않고 관측만으로 분해를 항상 복원할 수 없다. 이것은 구조적 반례이며 실제 자료의 Bayes error가0.07이라는 추정은 아니다. 공통+20dB 불변·Nobs<K·부분대역결측0·표본/셀 가중치 구분을 [독립 검산](quiet_features/INDEPENDENT_NUMERIC_QA.json)에 보존했다.\n',
 '### 잔차 상관 상위 특징 예시\n',table(featuretop[['panel','feature','n','pearson','spearman','univariate_affine_in_sample_R2_equals_r_squared']]),'\n',
 '실제 저장 A/B/E로도 보완 검산했다. full15 양성500개의 선택 quiet분은 모든K에서 A=0 비율0이었고, K=3에서 quiet 내부 실제 추가분율 평균0.233769였다. 자연 과도음 음성은 A=0이 맞지만 같은 K=3의 B총량 추정은 평균41.93% 과소추정되어 양의 오탐을 만들었다. 추가 성분 오염과 변동 배경 대표성 실패는 서로 다른 문제다. [실제 성분 보완표](quiet_features/QUIET_TRUE_COMPONENT_SUPPLEMENT.md)에 원시 컴포넌트의 임의 단위 정렬까지 명시했다.\n',
 '643개 입력 물리 특징을 y 및 y−old_B1과 비교했다. 관측수·결측 위치도 원인 분리용 감사 변수로만 포함했으며 생산 분류기 입력에 되살리지 않았다. 이 상관 스캔은 **행별 비가중 기술 통계**이며 native12 RMSE의50:50 가중과 다르다. r²는 같은 표본의 절편 포함 단변량 선형 설명력이지, 새 모델을 학습해 얻은 독립 검증 R²가 아니다. 다중 탐색·원천 재사용·패널 간 반복 때문에 높은 상관만으로 실전 개선을 보장할 수 없다. TOL 파워에는 파형 위상이 없어 위상 지수를 만들지 않고 envelope 동조/lag만 계산했다.\n',
 '모든 구간별RMSE/MAE/Bias/R², 모든643특징의 두 타깃 상관, 패널별잔차상위30개 및 행별quiet추정은 [Quiet-B 보고서](quiet_features/R20_QUIET_BACKGROUND_FEATURES.md)와 해당 CSV에 있다.\n']
 s+=['## 6. 이번에 확인한 것과 아직 확인하지 못한 것\n',
 '| 항목 | 이번 실행의 결론 |\n|---|---|\n| 로컬 성적과 Public 순위/오차 폭의 1:1 일치 | 미확립. 공통4제출·고정88유효비교의 최고Spearman0.8. 사후 관계를새loss로사용하지 않음. |\n| 모든 로컬 무추가가 조용한가 | 아님. 기존R11 HF hard-negative와 이번 자연활성 하위군을분리해확인. |\n| ambient를 더하면 공식 y=0인가 | 아님. 추가녹음 에너지는 양수. 선박유무 가정은별도counterfactual. |\n| Quiet-B만으로 RMSE를낮출수있는가 | 이번고정K·TRAIN계수에서는실패. 후처리미채택. |\n| 17채널이없애버린 절대스케일 때문에 Bayes floor0.07인가 | 이번자료로증명불가. 관측분해비식별 반례는있지만 실제최소오차값은모름. |\n| 학습·합성·관측 정의의전달성문제가남아있는가 | 전체물리분포차이와 로컬/Public불일치가남음. 어느기제가공식실패의주원인인지는숨은Test성분없이단일확정불가. |\n',
 '향후 변경은 기존old_B1 대비 전체·무추가·약한양성·장시간을 함께 보는 R19 게이트 원칙을 유지해야 한다. 이번보고서에 여러번 사용된local_report를 새 ‘미사용시험지’라고 부를수없다. 새검증설계가필요할때에는Public순위에맞춘계수가아닌사전물리규칙·새원천분리·독립확인셋으로평가해야한다. 현재 사용자지시범위에서는 모델/정책/제출을변경하지않고진단까지완료했다.\n',
 '## 7. 상세 산출물과 재현 코드\n',
 '- [분석1 지표·순위 보고서](alignment/R20_LOCAL_ALIGNMENT_METRICS.md) / [전체지표 CSV](alignment/ALL_MODELS_PANELS_SLICES_METRICS.csv) / [모델·패널 누락표](alignment/MODEL_PANEL_COVERAGE.csv)\n- [분석2 고난도 보고서](hard_panels/R20_HARD_PANEL_DIAGNOSTICS.md) / [개별·결합 전수 CSV](hard_panels/inference/ALL_INDIVIDUAL_AND_BLEND_METRICS.csv) / [하위군 교차 CSV](hard_panels/HARDNEGATIVE_SUBGROUP_CROSS_METRICS.csv)\n- [분석3 물리 프로파일](physical_profiles/R20_PHYSICAL_PROFILES.md) / [Test 기술통계 전용](physical_profiles/test_descriptive_only/R20_TEST_DESCRIPTIVE_COMPARISON.md)\n- [분석4 Quiet-B·잔차 보고서](quiet_features/R20_QUIET_BACKGROUND_FEATURES.md) / [전체 상관 CSV](quiet_features/ALL_PHYSICAL_FEATURE_CORRELATIONS.csv) / [패널별 상위30 CSV](quiet_features/TOP30_RESIDUAL_FEATURES_BY_PANEL.csv)\n',
 '재현 소스: ops/r20_saved_alignment.py, ops/r20_hard_panels.py, ops/r20_physical_profiles.py, ops/r20_quiet_background_features.py, ops/r20_integrated_report.py. 각 실행의 사전등록·입력SHA·계수동결·모델SHA·CPU추론진행/정밀도검사를해당폴더에보존했다. 후속실행은기존동결파일을덮어쓰지않는새버전출력을사용해야한다.\n']
 report=OUT/'R20_INTEGRATED_DIAGNOSTICS_REPORT.md';report.write_text('\n'.join(s))
 # Verify completed component receipts before recording the master manifest.
 checks=[]
 checks+=verify_map(a,read(a/'OUTPUT_SHA256.json'))
 checks+=verify_map(q,read(q/'ARTIFACT_SHA256.json'))
 checks+=verify_map(p,read(p/'COMPLETED.json')['files'])
 checks+=verify_map(h,read(h/'ARTIFACT_SHA256.json'))
 checks+=verify_map(h/'inference',read(h/'inference/OUTPUT_SHA256.json'))
 # Hard inference output receipt is verified inside the branch; root also hashes
 # every final generated file, including model-weight/source pins as metadata.
 if any(not x['matches'] for x in checks):raise ValueError('Sub-artifact digest mismatch')
 write(OUT/'CROSS_BRANCH_SHA_VERIFICATION.json',{'checks':len(checks),'mismatches':0,'results':checks})
 body=markdown.markdown(report.read_text(),extensions=['tables','fenced_code'])
 styles='body{font:16px/1.7 -apple-system,BlinkMacSystemFont,sans-serif;color:#163047;background:#f4f7fb;margin:0}main{max-width:1450px;margin:30px auto;padding:32px;background:white;border-radius:18px}h1,h2,h3{color:#133c65}h2{margin-top:38px}table{border-collapse:collapse;display:block;overflow:auto;font-size:13px;max-width:100%;margin:18px 0}td,th{padding:8px 10px;border:1px solid #d4dfe9;white-space:nowrap}th{background:#eaf2fc}a{color:#086ea6}code{background:#edf2f7;padding:2px 4px}p{max-width:1100px}.banner{background:#eaf3fc;border-left:5px solid #2276ae;padding:18px}pre{overflow:auto}'
 htmltext='<!doctype html><html lang="ko"><meta charset="utf-8"><title>R20 CPU 통합 진단</title><style>'+styles+'</style><main><div class="banner">R20 완료 · 모델 재학습 0회 · 공식 제출 0회 · 생산 정책 변경 없음<br><a href="R20_INTEGRATED_DIAGNOSTICS_REPORT.md">종합 Markdown</a> · <a href="ARTIFACT_SHA256.json">전체 SHA-256</a> · <a href="FILE_INDEX.csv">파일 목록</a></div>'+body+'</main></html>'
 (OUT/'START.html').write_text(htmltext)
 # Receipts omit themselves to avoid circular hashes; individual completion
 # receipts within branches are stable and included in the master inventory.
 root_receipts={OUT/'ARTIFACT_SHA256.json',OUT/'AUDIT_COMPLETED.json',OUT/'FILE_INDEX.csv'}
 files=[f for f in sorted(OUT.rglob('*')) if f.is_file() and f not in root_receipts]
 inventory=[{'path':str(f.relative_to(OUT)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in files]
 pd.DataFrame(inventory).to_csv(OUT/'FILE_INDEX.csv',index=False)
 mapping={r['path']:r['sha256'] for r in inventory};mapping['FILE_INDEX.csv']=sha(OUT/'FILE_INDEX.csv')
 write(OUT/'ARTIFACT_SHA256.json',mapping)
 completion={'status':'completed','completed_KST':stamp,'report':str(report.relative_to(ROOT)),'report_sha256':sha(report),
  'master_manifest_sha256':sha(OUT/'ARTIFACT_SHA256.json'),'files':len(mapping),'verified_sub_receipts':len(checks),
  'new_model_training':False,'GPU_MPS_RunPod_used':False,'frozen_CPU_hard_panel_inference':True,
  'TRAIN_only_statistical_quiet_coefficient_fit':True,'Test_used_for_descriptive_statistics_only':True,
  'Test_label_reverse_engineering':False,'new_submission':False,'production_policy_changed':False,
  'code_sha256':{f:sha(ROOT/f) for f in ('ops/r20_saved_alignment.py','ops/r20_hard_panels.py','ops/r20_frozen_hard_inference.py','ops/r20_hard_panel_report.py','ops/r20_physical_profiles.py','ops/r20_quiet_background_features.py','ops/r20_quiet_component_validation.py','ops/r20_integrated_report.py')}}
 write(OUT/'AUDIT_COMPLETED.json',completion)
 print(json.dumps(completion,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
