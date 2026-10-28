from pathlib import Path
import json,html,datetime,importlib.metadata
import pandas as pd
ROOT=Path(__file__).resolve().parent
R=ROOT/'reports';R.mkdir(exist_ok=True)
read=lambda p:json.loads((ROOT/p).read_text())
run='models/fullsources_cpu_a6d7f3a5_seed42'
eda=read('eda/summary.json');sa=read('data/processed/sa_preparation_manifest.json')
ext=read('data/processed/external_preparation_manifest.json');teacher=read('models/weak_teacher/manifest.json')
experiment=read(run+'/run_manifest.json');post=read(run+'/postprocess.json')
noaa=read(run+'/noaa_independent_assessment.json') if (ROOT/run/'noaa_independent_assessment.json').exists() else None
threads=read('scratch/discussions/thread_bodies.json')
literature=[
('FDY convolution','2022 · INTERSPEECH','https://arxiv.org/abs/2203.15296','https://github.com/frednam93/FDY-SED','주파수 이동 불변 가정을 완화. 25대역·15분 마스크 회귀로 독립 구현해 비교. 원래 SED 성적을 대회 성적으로 해석하지 않음.'),
('Frequency Dynamic Convolutions','2025','https://arxiv.org/abs/2506.12785','https://github.com/frednam93/FDY-SED','DFD/PFD/MDFD 확장 조사. 짧은 시간축에서는 모델 확대보다 검증 우선.'),
('iTransformer','2024 · ICLR','https://arxiv.org/abs/2310.06625','https://github.com/thuml/iTransformer','주파수 25개를 토큰으로 변환. 시간 예측 백본을 추가 에너지 마스크 회귀로 변경.'),
('ModernTCN','2024 · ICLR','https://openreview.net/pdf?id=vpJMJerXHU','https://github.com/luodhhh/ModernTCN','시간·변수 축 컨볼루션 후보. 이번 CPU 실험에는 미포함; 장기 예측 성능이 15분 회귀 우위를 보장하지 않음.'),
('TabM','2025 · ICLR','https://arxiv.org/abs/2410.24210','https://github.com/yandex-research/tabm','공식 Apache-2.0 패키지. 멤버별 MSE를 평균하는 학습 방식 적용.'),
('TabICLv2','2026 · ICML','https://proceedings.mlr.press/v306/qu26a.html','https://github.com/soda-inria/tabicl','최신 표형 기반 모델 비교 후보. 별도 GPU·체크포인트·전처리 감사 필요. 미실행. 이 대회 SOTA라고 주장하지 않음.'),
('TabR','2024 · ICLR','https://arxiv.org/abs/2307.14338','https://github.com/yandex-research/tabular-dl-tabr','검색을 특징에 결합하는 연구. 실제 구현은 개별 클립 검색 대신 학습 그룹 집계 프로토타입.'),
('ModernNCA','2025 · ICLR','https://arxiv.org/abs/2407.03257','https://openreview.net/pdf?id=JytL2MrlLT','이웃 기반 표형 학습의 비교 근거. 단일 외부 클립 식별에는 사용하지 않음.'),
('TabPFN','2025 · Nature','https://www.nature.com/articles/s41586-024-08328-6','','표형 기반 모델 계열 조사. 크기·가중치 라이선스·규칙 적합성 확인 전 실행 제외.'),
('LightGBM','2017 · NeurIPS','https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html','https://github.com/microsoft/LightGBM','L2 회귀와 별도 추가 여부 분류. 기본/물리 특징/프로토타입을 동일 조건 비교.'),
('CatBoost','2018 · NeurIPS','https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html','https://github.com/catboost/catboost','대안 GBDT 후보; 현재 CPU 비교에는 미포함.'),
('InceptionTime','2020 · DMKD','https://arxiv.org/abs/1909.04939','https://github.com/hfawaz/inceptiontime','다중 시간 스케일 컨볼루션 후보. 짧은 입력에 맞는 커널 폭 검토 필요.'),
('TSMixer','2023','https://arxiv.org/abs/2303.06053','https://github.com/google-research/google-research/tree/master/tsmixer','축 혼합 MLP 후보. TabM과 비교할 후속 실험.'),
('BEATs','2023 · ICML','https://arxiv.org/abs/2212.09058','https://github.com/microsoft/unilm/tree/master/beats','파형 오디오 사전학습. 25대역 요약값에 원본 파형이 복원되는 것은 아님.'),
('AudioMAE','2022 · NeurIPS','https://arxiv.org/abs/2207.06405','https://github.com/facebookresearch/AudioMAE','외부 파형 표현의 후보. 대회 요약 입력과 해상도가 달라 직접 적용하지 않음.'),
('OceanBEATs corrected release','2026 · revision in review','https://huggingface.co/BiologgingSolutions/OceanBEATs/blob/main/README.md','','수정된 2026년 9월 encoder/head 쌍 조사. 폐기된 이전 가중치는 제외. 일부 원천 학습 자료의 비공개성과 입력 차이 때문에 현재 파이프라인에는 미사용.'),
('AudioRAG','2026 · Audio-AAAI','https://arxiv.org/abs/2602.10656','','오디오 질의응답·검색 벤치마크. 에너지 비율 RMSE 개선의 직접 근거는 아님.'),
('TangoFlux','2026 · ICLR','https://arxiv.org/abs/2412.21037','https://github.com/declare-lab/TangoFlux','텍스트→오디오 flow 모델. 생성 음질이 수중 음향 물리·정확한 에너지 라벨을 보장하지 않아 주 합성기로 미채택.'),
('MixIT','2020 · NeurIPS','https://arxiv.org/abs/2006.12701','https://research.google/pubs/unsupervised-sound-separation-using-mixture-invariant-training/','혼합물 기반 비지도 분리 연구. 25대역 요약값에서 유일한 실제 선박 성분을 복원할 수 있다는 보장은 없음.'),
('UATFSN','2025 · Ocean Engineering','https://doi.org/10.1016/j.oceaneng.2025.123234','','수중 음향 분류의 경량 모델 조사. 클래스 정확도와 추가 에너지 비율 회귀는 과제가 다름.'),
('UATR-SLM','2026','https://arxiv.org/abs/2601.18086','','수중 오디오 모델 계열 조사. 원본 파형·분류 과제와의 입력/목표 차이로 현재 실험 제외.'),
('ONCship','2026 · preprint','https://arxiv.org/abs/2610.19722','https://huggingface.co/datasets/zcao123/ONCship','대규모 추가 자료 후보. 중복·라이선스·다운로드 규모 검토 전 수집/학습에 사용하지 않음.'),
('HearMyShip dataset paper','2025 · Scientific Data','https://doi.org/10.1038/s41597-025-04584-x','https://doi.org/10.57760/sciencedb.13099','연구 논문의 라이선스와 데이터 CC0를 구분. 원본 배포 MD5와 일치하는 WAV만 사용.'),
('VTUAD v1.3.2','2026 · data release','https://doi.org/10.5281/zenodo.23037895','https://doi.org/10.17616/R3RW43','strict interval 전체와 파일별 체크섬 사용. AIS 라벨은 약한 시나리오 라벨.'),
('POMDP planning','1998 · Artificial Intelligence','https://doi.org/10.1016/S0004-3702(98)00023-X','','현재 과제에는 행동·환경 전이·순차 보상이 없어 POMDP/RL을 주 학습기로 도입할 근거가 부족함.')]
(R/'literature_evidence.json').write_text(json.dumps([dict(zip(['work','year_venue','primary','project','decision'],x)) for x in literature],ensure_ascii=False,indent=2))
E=html.escape
def table(headers,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+E(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def link(url,label):return '<a href="'+E(url,quote=True)+'">'+E(label)+'</a>'
def block(title,body,id=''):return f'<section id="{id}"><h2>{title}</h2>{body}</section>'
body='''<header><span class="eyebrow">UNDERWATER VESSEL NOISE · 2026-10-24</span><h1>관측에서 예측까지</h1><p class="lead">실제 데이터 감사, 출처가 검증된 외부 음원, 제어 에너지 합성, 모델 비교와 추론까지.</p><p>CPU 실험과 데이터 구축 완료. GPU 다중 시드 최적화와 대회 공식 성능 확인은 아직 진행하지 않았습니다.</p></header>
<nav><a href="#task">과제</a><a href="#data">규모·EDA</a><a href="#pipeline">전체 구조</a><a href="#validation">검증·결과</a><a href="#research">연구·판단</a><a href="#discussion">52개 토론</a><a href="#handover">실행·전달</a></nav>'''
body+=block('추정해야 하는 것은 “추가된 기준 음원의 에너지 비율”', '''<p>대회 입력은 원본 파형이 아닌 <strong>15분 × 25개 대역의 dB 요약값</strong>입니다. 원본 남아공 관측값에는 확인된 정답이 없습니다. 제공되지 않은 AIS를 있다고 가정하거나 자연 선박 존재를 정답으로 만드는 방식은 사용하지 않습니다.</p><div class="equation">E = B + αR<br>y = Σ(OαR) / Σ(OE)<br>ŷ = Σ(OqE) / Σ(OE),　0 ≤ q ≤ 1</div><p>O는 관측 마스크입니다. 결측은 무음이 아닙니다. B에 원래 선박 소리가 있어도 y=0일 수 있으며 R에 포함된 주변음도 추가된 기준 에너지에 포함됩니다. 순위 산식은 <strong>1 − RMSE</strong>이므로 주 학습 손실은 MSE입니다. 추가 여부 ROC AUC는 진단 지표로 구분합니다.</p><p>원천 관측에 대한 자연 선박 정답은 현재 확인할 수 없습니다. 모든 1,715,779행에 외부 교사 점수를 계산했지만 독립 검증 AUC가 0.35~0.54에 그쳐 학습용 의사 라벨 사용을 차단했습니다. 점수·불확실성·오류 상태를 보존하며 확인된 정답으로 표기하지 않습니다.</p>''','task')
body+=block('실제 확보한 데이터와 결측 분포',table(['자료','크기 / 수량','처리 상태'],[
['대회 원본','97.8 MB · 1,715,779행 × 74열','10개 관측소 Parquet; 62개 TOL + 메타/QC'],
['공통 모델 입력','1,715,779행 × 25대역','공통 대역 결측 0; 모든 행 원본 위치 보존'],
['모든 TOL 대역','106,378,298셀 중 27,102,411셀 결측','약 25.48%; 녹음기별 대역 지원 범위 차이 포함'],
['원본 품질 문제','653개 −156.5 dB 평탄 행; 16,896행 길이 &lt;55초','삭제하지 않고 품질 상태 표시; 시간 윈도우 입력에서는 제외'],
['남아공 모델 윈도우','133,641 × 15 × 25','실제 시각으로 고정 15분 구간; 최소 3분 관측; 공백 NaN 유지'],
['Test.csv','1,500행 × 376열 · 약 2.83 MB','18,350 / 562,500셀 결측 = 3.2622%; 묘사 EDA만 수행'],
['VTUAD','525 strict WAV · 42.05시간 · 아카이브 4.64 GB','아카이브 MD5와 모든 WAV SHA-256 검증'],
['HearMyShip','1,146 WAV 엔트리 · 중복 제거 1,145개 · 6.38시간 · 3.32 GB','원본 ScienceDB 크기/MD5 전부 검증; 미러 불일치는 원본으로 보완'],
['NOAA 독립 자료','실제 연속 15분 × 3개 · 48.1 MB','동일 배치의 3개 날짜; 학습/보정/모델 선택에 미사용'],
['현재 합성 집합','학습 16,000 + 검증 3,000 + 보정 3,000 + 별도 관측소 3,000','총 25,000개 · 압축 38.1 MB; 각 샘플 15×25 입력과 셀별 정답 포함']])+'''<p>원천자료 합계는 약 8.10 GB로 원래 제공 파일 97.8 MB 대비 82.8배입니다. 외부 자료는 파형, 대회 자료는 대역 요약이므로 이 배수가 독립 정보량 증가를 뜻하지는 않습니다. 합성은 같은 원천을 재조합합니다.</p><p>Test에는 결측이 있는 윈도우 725개, 완전한 윈도우 775개가 있으며, 결측은 전 대역이 빠진 분 단위입니다. 이러한 Test 분포를 합성 마스크나 사전확률 설정에 사용하지 않았습니다. 학습에는 Test 값·빈도·통계·적대적 분류기를 사용하지 않습니다.</p><div class="gallery">'''+''.join(f'<figure><img src="../eda/figures/{p.name}"><figcaption>{E(p.stem)}</figcaption></figure>' for p in sorted((ROOT/'eda/figures').glob('*.png')))+'''</div>''','data')
body+=block('설계와 구현', '''<figure class="wide"><a href="../figures/pipeline_paper_final.png"><img src="../figures/pipeline_paper_final.png"></a><figcaption>요청한 참고 이미지의 논문 도식 스타일을 따라 새로 생성한 제안 구조. 그림 속 곡선·성능 막대는 모식도이며 실측 결과는 아래 표에 있습니다.</figcaption></figure><ol><li><strong>수집·감사:</strong> 원본 파일·체크섬·실제 음원 길이·출처·라이선스를 기록합니다. 1,146개 HearMyShip 파일 중 1개 중복을 제거합니다.</li><li><strong>분리 우선:</strong> 재사용 녹음기로 연결된 관측소 그룹을 유지하고, 학습/검증/보정은 관측 일자 단위로 분리합니다. Fish Hoek·Sea Point·Saldanha 연결 그룹은 별도 도메인 평가로 둡니다. 시간적으로 붙은 날짜의 자기상관은 잔여 한계이며 후속 GPU 실험에는 주 단위 purge 검증을 권장합니다.</li><li><strong>외부 자료 전처리:</strong> 실제 표본률·첫 채널 → 완전한 2초 Hann PSD → 25개 base-10 대역 적분 → 완전한 분은 선형 에너지 평균. 절대 µPa 보정을 주장하지 않습니다. 20초 CPA 음원을 측정된 1분/15분으로 표기하지 않습니다.</li><li><strong>합성:</strong> CPU 기반 수치 합성기. 제로 추가를 포함한 고정 사전분포에서 원하는 비율 u를 뽑고, α = uΣ(OB)/[(1−u)Σ(OR)]를 풉니다. 관측된 셀에서 최종 정답을 재계산합니다. 시간 배치·주파수 응답 변화·복수 음원·일부 대리 시간 포락선을 사용하며 가정을 recipe에 기록합니다.</li><li><strong>모델 입력:</strong> 직접 회귀는 대역별/분별 특징·상대 dB·관측 마스크를 사용합니다. 마스크 모델은 상대 dB, 절대 dB, 정규화 에너지, 관측 마스크 4채널을 받습니다. 전역 Test 통계는 계산하지 않습니다.</li><li><strong>학습:</strong> MSE + 작은 추가 여부 BCE 보조 손실. 마스크 계열은 에너지 가중 셀 MSE도 사용합니다. TabM은 멤버 손실 평균을 적용합니다. 비지도 실제 관측을 외부 교사 정답으로 강제하지 않습니다.</li><li><strong>후처리:</strong> 검증 집합에서 비음수 합 1 앙상블 가중치를 정합니다. 별도 보정 집합의 2분할 교차 적합으로 선형 보정을 선택합니다. 예측을 [0,1]에 제한합니다. Test를 열기 전에 정책을 고정하고 해시를 기록합니다.</li></ol>''','pipeline')
metricrows=[]
for k in post['models']:
 metricrows.append([E(k)]+[f'{experiment["results"][k+"_"+s]["rmse"]:.5f}' for s in ['validation','calibration','domain_holdout']])
metricrows.append(['앙상블 + 보정']+[f'{post["results"][s]["calibrated_rmse"]:.5f}' for s in ['validation','calibration','domain_holdout']])
body+=block('실제 CPU 실험과 검증의 범위', '''<p>전체 외부 자료를 전처리한 후 학습 16,000개, 세 평가 집합 각 3,000개를 고정 생성했습니다. 신경망은 최대 8 epoch, 트리는 최대 800개, 시드는 42입니다. <strong>아래 값은 로컬 합성 RMSE이며 대회 점수가 아닙니다.</strong> 학습에 사용하지 않은 원본 관측소 그룹의 배경과 외부 원천을 함께 분리했습니다.</p>'''+table(['모델','검증 RMSE','보정 집합 RMSE','별도 관측소 RMSE'],metricrows)+f'''<p>검증에서 물리 특징 LightGBM은 {experiment['results']['lgbm_physics_validation']['rmse']:.5f}, 프로토타입 추가형은 {experiment['results']['lgbm_retrieval_validation']['rmse']:.5f}입니다. 검색 특징의 개선은 작으며 별도 관측소에서는 반대로 약간 나빠졌습니다. RAG 효과가 확립됐다고 해석하지 않습니다.</p><p>선택된 비중: {E(', '.join(k+' '+format(float(w)*100,'.1f')+'%' for k,w in zip(post['models'],post['weights']) if float(w)>.00001))}. 보정식은 clip({post['affine'][0]:.4f}×예측 {post['affine'][1]:+.4f}, 0, 1)입니다. 이 계수는 Test에 맞춘 값이 아닙니다.</p>'''+(f'''<p class="callout">동결 후 NOAA 실제 15분 기준 음원을 사용하는 독립 합성 평가: RMSE <strong>{noaa['rmse']:.5f}</strong>, 샘플 {noaa['n']}개. 참조 음원은 동일 센서의 3개 날짜뿐이므로 이 수를 독립 선박 수로 해석하지 않습니다. 학습 정답 평균 상수 기준 RMSE 0.31934보다 나빠 이 평가에서는 일반화에 실패했습니다. 이 결과로 모델을 재선택하지 않았습니다.</p>''' if noaa else '<p>NOAA 동결 평가 결과는 실행 후 생성됩니다.</p>')+'''<p>검증 보조 교사 모델은 AIS 시나리오와 주변음 구간 분류를 제대로 일반화하지 못했습니다. 따라서 모든 정상 행은 uncertain 상태, 653행은 invalid 상태로 남깁니다. 자연 선박 라벨/순수 배경 인증은 제공할 수 없습니다. 합성 target은 알고 더한 에너지로 정확히 정의됩니다.</p>''','validation')
body+=block('RAG·프롬프트·RL·POMDP에 대한 판단', '''<p><strong>구현한 검색 방식:</strong> 학습 집합의 외부 음원에서 먼저 원천 그룹 평균을 만들고, 최소 3개 그룹이 모인 스펙트럼 프로토타입을 구성했습니다. 윈도우와 프로토타입의 코사인 유사도를 LightGBM에 입력합니다. 개별 음원 검색·Test 출처 식별·숨겨진 정답 복원은 하지 않습니다.</p><p><strong>오디오 RAG/LLM 프롬프트:</strong> 입력이 파형이 아니므로 일반 오디오 질의응답 모델에 원본 오디오가 있는 것처럼 전달하지 않습니다. 논문/음향 지식은 합성기·특징 설계에 반영할 수 있지만, 문장 생성 품질과 연속 에너지 비율 오차는 다른 목표입니다. 프롬프트로 임의 확률을 받아 정답으로 사용하는 경로는 제외했습니다.</p><p><strong>RL:</strong> 현재는 알려진 정답을 가진 정적 회귀 문제이므로 직접 MSE 최적화가 평가 지표에 맞습니다. 실제 측정 정답이 없는 상태에서 스스로 만든 보상으로 RL을 돌리면 합성기 편향을 보상으로 강화할 수 있습니다. 온라인 정책/행동 과제도 없어 GRPO 등을 채택하지 않았습니다.</p><p><strong>POMDP:</strong> 결측 관측은 있으나 행동·환경 전이·순차 보상은 없습니다. 관측 마스크와 불확실성으로 처리합니다. 향후 실제 AIS·시계열 행동 문제가 주어질 때 별도 검토할 수 있습니다.</p><p><strong>Gemini 대화에서 반영한 것:</strong> FDY 계열, iTransformer, 에너지 마스크, 분위수 바닥·저고주파 비율 특징. 수정한 가정: AIS가 제공된다는 설명, 순수 배경을 판별할 수 있다는 전제, 마스크 모델과 앙상블이 반드시 우수하다는 단정. NMF나 추가 생성 모델은 현재 비교 결과 없이 성능 개선으로 표기하지 않습니다.</p>''','research')
body+=block('문헌·프로젝트 조사와 채택 근거',table(['연구 / 공개 시점','일차 출처 / 프로젝트','판단'],[[E(a)+'<br><small>'+E(b)+'</small>',link(c,'논문/공식 자료')+('<br>'+link(d,'프로젝트') if d else ''),E(e)] for a,b,c,d,e in literature])+'''<p>초기 Crossref 검색에서 제목이 다른 논문으로 잘못 매칭된 항목은 인용에 쓰지 않았습니다. 공식 arXiv·학회·저자 프로젝트·데이터 저장소를 우선했습니다. 수중 선박 분류나 일반 표형 벤치마크의 SOTA가 이 대회 비율 회귀 SOTA를 뜻하지는 않습니다.</p>''')
special={35370:'운영진: VTUAD 허용·출처 고지·자체 검증 보정. Test는 추론 전용.',35368:'운영진: Test 분포로 합성기 조정, Test 적대적 검증도 금지.',35420:'일 2회, 총 200회. 00 UTC 초기화; LB 탐색 대신 자체 검증.',35443:'코드 공개 실험: 로컬 개선과 공식 일반화 차이, 제로 추가 오탐을 강조.',35467:'전체 hull 분리와 실제 15분 참조, NOAA 독립 평가 설계 참고.',35491:'고정 CLAP에 인공 sonification 사용 사례. 원본 오디오 복원이 아니며 현재 미채택.',35492:'원본 CC0 MD5 대조, 짧은 CPA의 한계와 독립 평가 구분.',35498:'참가자의 고지 절차 개선 요청. 새 운영 규정으로 해석하지 않음.',35500:'측정된 연속 분과 대리 포락선 구분, 실제 미관측 구간 처리 참고.',35521:'대상·절대 레벨 질문. 답변 없음; 알려진 사실처럼 채택하지 않음.',35525:'NOAA 실제 15분 고정 holdout 제안. 라이선스는 현행 공식 NOAA 문서로 별도 검증.',35537:'World Bank 배송 밀도는 설명용. Test에 위치가 없어 예측 특징으로 미사용.'}
body+=block('공개 토론 52개와 모든 댓글 확인 기록', '''<p>수집한 52개 게시글 모두 본문을 확보했고, 표시 댓글 수와 확보한 댓글 수가 일치합니다. 외부 자료 선언이 다수를 차지합니다. 운영진 답변과 참가자의 제안·주장을 구분했습니다. 아래 링크와 로컬 audit으로 누락 여부를 확인할 수 있습니다.</p>'''+table(['글 ID / 제목','댓글','반영할 내용'],[[link(x['url'],str(x['id'])+' · '+x['title']),str(x['captured_replies'])+'/'+str(x['expected_replies']),E(special.get(x['id'],'외부 자료의 버전·접근·라이선스·분리·전처리 고지 사례. 해당 글의 허용 주장을 운영진 승인으로 대신하지 않음.'))] for x in sorted(threads,key=lambda x:x['id'])])+'''<p><a href="../scratch/discussions/reading_audit.csv">전체 확인 CSV</a> · <a href="../scratch/gemini_reference.txt">Gemini 참고 대화 보관본</a></p>''','discussion')
body+=block('GPU와 전달 파일', '''<p><strong>현재 합성기는 CPU 수치 계산이며 GPU가 필요하지 않습니다.</strong> 주 학습 모델은 수십만 파라미터 규모입니다. GPU 미실측 추정으로 16 GB 한 장이면 현재 구성 학습이 가능하며 24 GB 한 장을 권장합니다. RAM 32 GB 이상, 여유 디스크 30 GB 이상을 권장합니다. 여러 모델/시드는 순차 실행할 수 있습니다. 48~80 GB는 대규모 TabICLv2 문맥 실험 등 선택적 확장에만 검토합니다.</p><p>본격 GPU 실행 설정: 시드당 합성 학습 200,000개, 각 평가 집합 12,000개, 최대 40 epoch, 시드 42/43/44. GPU 종류와 실제 환경이 정해지지 않아 VRAM·학습 시간·다중 시드 개선은 아직 실측하지 못했습니다. 전체 171만 행이 전부 신경망 supervised 정답 샘플인 것은 아닙니다.</p><div class="code">python -m pip install -r requirements.txt<br>python -m pipeline.verify<br>bash run_gpu.sh<br>python -m pipeline.postprocess models/&lt;run&gt;<br>python -m pipeline.evaluate_noaa models/&lt;run&gt;<br>python -m pipeline.infer models/&lt;run&gt; data/raw/Test.csv outputs/submission_candidate.csv --device cuda</div>'''+table(['파일','용도'],[
[link('../figures/pipeline_paper_final.png','논문 스타일 전체 도식'),'새로 생성한 이미지. open-science 그림 도구 사용 안 함.'],
[link('../eda/summary.json','EDA 요약'),'실제 스키마·결측·품질·원본 크기'],
[link('../data/processed/sa_row_manifest.parquet','원본 행 manifest'),'모든 행의 원본 파일/위치·QC·포함 상태'],
[link('../data/processed/sa_all_rows_weak_labels.parquet','전 행 모델 점수'),'1,715,779행; 약한 점수·불확실성·실제 정답 아님'],
[link('../data/processed/external_clip_manifest.parquet','외부 음원 manifest'),'1,670개 고유 처리 음원·기간·출처·분리'],
[link('../data/synthetic/fullsources_cpu_a6d7f3a5_seed42/train.npz','합성 학습 집합'),'db/share/presence/cell_share; recipe 별도'],
[link('../'+run+'/run_manifest.json','CPU 학습 결과'),'모델별 비교와 설정'],
[link('../'+run+'/postprocess.json','동결 후처리 정책'),'앙상블·보정 계수와 평가'],
[link('../outputs/submission_cpu_candidate.csv','로컬 추론 후보 CSV'),'1,500행. 공식 대회에 제출하지 않음.'],
[link('../external_data_declaration_DRAFT.txt','외부 자료 공개 고지 초안'),'계정 명의 공개 게시 전 사용자 허가 필요'],
[link('../run_gpu.sh','GPU 실행 스크립트'),'공식 제출·외부 게시 자동 수행 없음'],
[link('eda_pipeline.ipynb','EDA/전처리 확인 노트북'),'경로 자동 탐지·원본 감사·시각화·합성 확인']])+'''<p>대용량 파일은 macOS 문서 동기화의 자동 비우기로 실행이 멈추는 문제를 피하려고 프로젝트의 .nosync 경로에 보관하고 기존 경로는 심볼릭 링크로 유지했습니다. 원본은 삭제하지 않았습니다.</p><p class="callout">완료: 수집·전처리·25,000개 합성·전 행 점수 산출·7개 구성 CPU 학습·앙상블·보정·고정 추론. 남은 외부 의존 작업: GPU 서버에서 다중 시드 본 학습, 사용자 허가 후 출처 고지 및 공식 제출. 현재 결과를 대회 SOTA나 자연 선박 정답 정확도로 표현하지 않습니다.</p>''','handover')
style='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f8f9f6;color:#17221d;font:16px/1.75 -apple-system,BlinkMacSystemFont,"Noto Sans KR",sans-serif}header,section,nav{max-width:1280px;margin:auto}header{padding:76px 38px 40px}.eyebrow{font:12px/1.4 monospace;letter-spacing:2px;color:#53705d}h1{font:600 58px/1.15 Georgia,serif;margin:18px 0}h2{font-size:27px;margin-top:0;letter-spacing:-.5px}.lead{font-size:23px;max-width:880px}nav{display:flex;gap:25px;flex-wrap:wrap;padding:18px 38px;border-top:1px solid #cfd8ce;border-bottom:1px solid #cfd8ce}a{color:#226243;text-decoration:none}a:hover{text-decoration:underline}section{padding:38px;background:white;margin-top:24px;border:1px solid #e2e8de;border-radius:5px}p{max-width:1120px}.equation{font:italic 24px/1.9 Georgia,serif;background:#eef3e9;border-left:4px solid #819c73;padding:24px 34px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:13px 15px;text-align:left;vertical-align:top;border-bottom:1px solid #e0e6dc}th{background:#edf2e9;white-space:nowrap}tr:nth-child(even){background:#fafbf8}td:first-child{min-width:170px}.gallery{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:24px}figure{margin:0;padding:12px;background:#fff;border:1px solid #e3e8df}img{max-width:100%;height:auto;display:block}figcaption{font-size:12px;color:#64715f;margin-top:9px}.wide{margin-bottom:26px}.callout{background:#f2f0e6;padding:20px 25px;border-left:4px solid #b4a46d}.code{background:#16231b;color:#dce8da;font:13px/2 monospace;padding:22px;overflow:auto}li{padding:6px 0}small{color:#64715f}footer{max-width:1280px;margin:25px auto;padding:30px;color:#66705f}@media(max-width:760px){header,section{padding:25px}h1{font-size:40px}.gallery{grid-template-columns:1fr}nav{padding:15px;gap:15px}}'''
(R/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Underwater Vessel Noise — 데이터·모델 연구 기록</title><style>'+style+'</style><body>'+body+'<footer>실측 결과와 설계 제안을 구분한 재현 기록 · 2026-10-24</footer></body></html>')
cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
def code(s):cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(True)})
md('# Underwater Vessel Noise: 실제 EDA와 전처리 감사\nTest는 묘사 EDA와 동결 추론에만 사용합니다. 약한 교사 점수는 실제 정답이 아닙니다.')
code('from pathlib import Path\nimport json, numpy as np, pandas as pd\nfrom IPython.display import display, Image\nROOT=Path.cwd()\nif ROOT.name == "reports": ROOT=ROOT.parent\nassert (ROOT/"eda/summary.json").exists()\nsummary=json.loads((ROOT/"eda/summary.json").read_text())\nsummary')
md('## 원본 파일·스키마·결측\n모든 원본 파일의 체크섬 감사와 열별 결측은 eda/tables에 있습니다.')
code('tables=sorted((ROOT/"eda/tables").glob("*.csv"))\n[(p.name, p.stat().st_size) for p in tables]')
code('for p in sorted((ROOT/"eda/figures").glob("*.png")):\n    display(Image(filename=str(p)))')
md('## 1,715,779행 보존 및 약한 점수의 제한')
code('rows=pd.read_parquet(ROOT/"data/processed/sa_row_manifest.parquet")\nweak=pd.read_parquet(ROOT/"data/processed/sa_all_rows_weak_labels.parquet")\nassert len(rows)==len(weak)==1715779\nassert rows.row_id.equals(weak.row_id)\ndisplay(weak.weak_label_status.value_counts())\njson.loads((ROOT/"models/weak_teacher/manifest.json").read_text())["metrics"]')
md('## 외부 원천과 분리\nHearMyShip의 hull 독립성은 검증되지 않았습니다. VTUAD hull/day 충돌은 유지하되 학습 참조에서 제외합니다.')
code('ext=pd.read_parquet(ROOT/"data/processed/external_clip_manifest.parquet")\next.groupby(["source","split"]).agg(clips=("clip_id","size"),hours=("duration_seconds",lambda x:x.sum()/3600))')
md('## 모델 입력과 에너지 정답\n합성 함수는 Test를 읽지 않습니다. 결측 셀은 모든 에너지 합에서 제외됩니다.')
code('import sys\nsys.path.insert(0,str(ROOT))\nfrom pipeline.mixtures import MixtureSampler,cnn_channels\ndb,y,z,q,recipe=MixtureSampler("train",seed=42).draw(8)\nchannels,weights=cnn_channels(db)\nassert np.allclose((q*weights).sum((1,2)),y,atol=2e-6)\nprint(db.shape,channels.shape,y)')
md('## CPU 비교 결과와 후처리\n공식 leaderboard 성능이 아닙니다. 평가 집합은 학습에 쓰지 않습니다.')
code(f'run=ROOT/"{run}"\nresult=json.loads((run/"run_manifest.json").read_text())\ndisplay(pd.DataFrame(result["results"]).T)\njson.loads((run/"postprocess.json").read_text())')
md('## 재실행\n터미널에서 `python -m pipeline.verify`, GPU 준비 후 `bash run_gpu.sh`를 실행합니다. 공개 고지와 공식 제출은 자동 수행하지 않습니다.')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':5}
(R/'eda_pipeline.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2))
print('Created report and notebook; literature sources',len(literature),'discussion posts',len(threads))
