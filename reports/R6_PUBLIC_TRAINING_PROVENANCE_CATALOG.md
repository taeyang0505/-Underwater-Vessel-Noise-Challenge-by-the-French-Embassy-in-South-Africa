# 공개 South African TRAIN 원천 조사

조사일: **2026-10-26**. 조사 범위는 대회가 공개한 **학습 측정자료의 출처**입니다. Test 배열·숨겨진 원천/시각·추가 전 배경은 조사하지 않았습니다. 원시 데이터 다운로드, 제공자 연락, 외부 메시지, 학습은 실행하지 않았습니다.

**확인한 결론:** 공식 제공 기관과 관련 연구는 찾았습니다. 그러나 대회와 정확히 같은 **10개 site·1,715,779행 전체**를 연결하는 대회 외부의 기존 공개 dataset DOI·catalogue accession·다운로드는 이번 조사에서 검증하지 못했습니다. 이는 “그런 공개 자료가 존재하지 않는다”는 뜻은 아닙니다. 일부 catalogue는 JavaScript 전용이거나 접근 오류가 있어 전수 확인에도 제한이 있었습니다.

## 확정된 출처와 로컬 metadata

공식 대회는 실측자료 제공자로 **SAEON, SANBI, SA Underwater Radiated Noise Consortium(SA URN)**를 명시합니다. 데이터 페이지의 1,715,779개 무라벨 측정·10개 site·분 단위 주파수 대역 요약이라는 설명은 로컬 패키지와 일치합니다. 원래 audio나 자연 선박 정답 label은 이 패키지에 없습니다. [공식 소개](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge), [공식 데이터](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data)

허용된 로컬 파일 `data/extracted/README.md`, `DATA_DICTIONARY.md`, `stations.csv`, `recorders.csv`만 읽어 다음을 확인했습니다.

- Station metadata 10행의 record 수 합계와 recorder coverage metadata 17행의 합계가 모두 **1,715,779**입니다. 17행은 coverage 항목 수이며 독립 물리 녹음기 17대를 인증하지 않습니다.
- 공개 TRAIN export-clock 범위는 **2015-04-21~2025-12-06**입니다. UTC 정합은 미검증입니다.
- README는 이전 acoustic training archive와 같은 측정행을 station별 Parquet으로 재포장했고 새로운 기간을 추가하지 않았다고 설명합니다.
- 주파수 대역값은 export dB convention이며, 파일명에서 유도한 sample rate와 절대압력 calibration은 검증되지 않았습니다. 작업용 좌표도 정확한 deployment 위치의 증명이 아닙니다.
- 이 네 문서에서 원본 dataset DOI/accession을 찾지 못했습니다. README의 review-package 문구는 새로운 재배포 licence를 부여하지 않습니다. 현재 대회 이용에는 공식 규정이 적용됩니다.

## Catalogue·연구별 확인 결과

| 출처 | 공개 접근/식별자 | 대회 TRAIN과의 관계 | 일치 여부와 제한 |
|---|---|---|---|
| [SAEON ODP catalogue](https://catalogue.saeon.ac.za/) | 공식 catalogue. 직접 열면 JS shell | 공식 제공 기관의 data discovery 경로 | 정확한 전체 archive record/DOI/download **미확인**. 검색 index와 shell 확인만으로 존재 여부를 확정할 수 없음 |
| [SAEON Observations Database](https://observations.saeon.ac.za/) | 공식 환경 관측 portal. Indexed landing 확인, 직접 접근 timeout | 가능한 기관 저장 경로 | 동일 station roster·TOL schema·record 수·checksum을 묶는 record **미확인** |
| [MIMS](https://data.ocean.gov.za/), [공개 catalogue](https://data.ocean.gov.za/mims/catalog/) | DFFE 해양 자료 archive·공개 catalogue 접근 확인 | SAEON wiki에서도 연결하는 기관 저장 경로 | 정확한 hydrophone/TOL release record **미확인**. 일부 acoustic/sound 검색 URL은 도구에서 접근되지 않았고 전체 record를 열거하지 않음 |
| [SANBI marine pressures / Sound Seas](https://nba.sanbi.org.za/content/marine/marine_pressures.html) | 공식 기관 설명의 indexed text 확인; 직접 HTML403 | SoundSeas/ACEP soundscape 연구 맥락 확인 | 연구 programme 설명이며 대회 전체 file manifest·dataset DOI가 아님 |
| [Underwater Chatter for the Win, 2022](https://www.mdpi.com/2077-1312/10/6/746) | **논문 DOI `10.3390/jmse10060746`**. Publisher indexed text 및 [저자기관 목록](https://cmst.curtin.edu.au/research/marine-soundscapes/) 확인; 직접 접근429 | St Francis·Algoa Bay 연구로 **부분 원천과 관련 가능** | 논문 DOI는 dataset DOI가 아님. 전체 10site/1,715,779행과의 정확한 export·file 관계는 검증되지 않음 |
| [Baseline characterization of inshore reef soundscapes…, 2026](https://www.nature.com/articles/s41598-026-64663-w) | **논문 DOI `10.1038/s41598-026-64663-w`**, 2026-08-03. Primary indexed text 확인 | Algoa Bay 4sites, SoundSeas/DFFE/SAEON 연구 지원 맥락 | 관련 연구지만 전체 대회 archive의 dataset DOI/download를 확인하지 못함. 직접 page 접근에도 제한이 있었음 |
| [Dryad: two sub-Antarctic islands](https://datadryad.org/dataset/doi:10.5061/dryad.9s4mw6mtf) | **실제 dataset DOI `10.5061/dryad.9s4mw6mtf`**, 2025-11-06. 공개 file/README metadata 확인 | 남아공 기관의 관련 수중음향 연구 | **다른 자료**. Prince Edward/Marion Islands, 2021–2022, 시간 단위·광대역 집계로 mainland 10site minute-TOL archive와 다름 |
| [BioSoundSCape / NASA ORNL DAAC](https://doi.org/10.3334/ORNLDAAC/2372) | **실제 dataset DOI `10.3334/ORNLDAAC/2372`**. [공식 data paper](https://www.nature.com/articles/s41597-025-05685-3) | 남아공·SAEON 협력이라는 공통점 | **다른 자료**. Fynbos 육상 ARU 521sites·825,832분 WAV로 이번 해양 대역 export의 기원이 아님 |

## 2022 논문과 St Francis의 관련성을 어디까지 말할 수 있나

로컬 TRAIN recorder metadata의 St Francis 항목은 `SM2M_Cape_StFrancis_SM2M.1.48000`, **169,488행**, **2015-04-21~2018-02-28**입니다. 논문의 연구 지역과 겹쳐 원천 연구를 설명할 유력한 관련 문헌입니다. 그러나 장소·기기명·시기 유사성만으로 이 행들이 논문 분석에 사용된 바로 그 export라고 확정할 수 없습니다. 원본 file 목록·checksum·export 처리 이력의 연결이 필요합니다.

특히 이번 전체 TRAIN에는 **2025년 측정**도 있습니다. **2022년 논문 하나가 전체 archive를 설명하는 동일 dataset이라는 주장은 성립하지 않습니다.** 해당 논문은 관련 연구/부분 lineage 후보로 표시합니다.

## 이용 조건은 자료별로 확인해야 함

SAEON 정책은 일반적으로 CC BY-SA4.0을 권장하지만 제공자가 다른 CC 조건·custom licence·embargo·접근 불가를 선택할 수 있다고 명시합니다. MIMS에도 서로 다른 licence와 embargo 항목이 있습니다. **기관 portal 또는 논문의 licence를 이번 대회 원본에 자동 적용할 수 없습니다.** [SAEON 공식 data policy](https://odpwiki.saeon.ac.za/saeon_data_policy)

2026 Scientific Reports 논문의 표시 licence는 **CC BY-NC-ND4.0**이지만 이는 논문에 대한 조건이며 원시 녹음의 공개 licence를 입증하지 않습니다. Dryad의 해당 record에서도 이번 조회로 licence 조항을 별도 확인하지 못해 repository 일반 관행만으로 추정하지 않았습니다. NASA BioSoundSCape의 공개 조건도 이 대회 자료에 전이되지 않습니다.

정확한 동일 자료로 인증하려면 **제공자 release/accession·해당 site/기간 명세·dataset version/DOI·file checksum 또는 row/export lineage·명시적 source-data licence**가 필요합니다. 추가 자료가 나와도 비교 범위는 이미 허용된 공개 TRAIN에 한정하며 보호·제외된 기간을 확보하는 경로로 사용하지 않습니다.

검색 범위·기관별 access 상태·licence·일치/불일치·로컬 metadata SHA는 [JSON catalogue](R6_PUBLIC_TRAINING_PROVENANCE_CATALOG.json)에 기록했습니다.
