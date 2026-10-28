# R4 시간축 증강 독립 검증

전체 판정: **True**. 실제 음향 배열이나 Test/calibration/sealed 배열은 읽지 않았습니다.

- 고정 원천: 119 clips / 22 selected components, source hash `694cf18c9daabc69a77807f6e0ede33ca451443fbfb17d754ac191f0206463ef`.
- 시간축 수치 oracle: 117 cases, 최대 정규화 오차 1.61e-13.
- 합성 fixture: 720 examples / 15 fold-role pools, mask·parent UTC/길이/SHA 보존, 청크 재현 통과.
- 실제 예정 크기의 단일 고정 schedule: 모든 사전 gate **True**; conditional-u 평균차 최대 0.015180 (기준 .05).
- 시드 재시도, 예제 삭제, 실패 양성의 zero 재라벨링은 하지 않았습니다. 의도적 overlap 실패는 reference 호출 1회 후 실패했습니다.

| Fold | Role | N | Zero | Coarse conditional-u mean difference | All predeclared gates |
|---:|---|---:|---:|---:|---|
| 0 | validation | 3000 | 600 | 0.009141 | True |
| 0 | outer_validation | 3000 | 600 | 0.011597 | True |
| 1 | validation | 3000 | 600 | 0.015180 | True |
| 1 | outer_validation | 3000 | 600 | 0.006086 | True |
| 2 | validation | 3000 | 600 | 0.003299 | True |
| 2 | outer_validation | 3000 | 600 | 0.002123 | True |
| 3 | validation | 3000 | 600 | 0.003756 | True |
| 3 | outer_validation | 3000 | 600 | 0.006041 | True |
| 4 | validation | 3000 | 600 | 0.003504 | True |
| 4 | outer_validation | 3000 | 600 | 0.011855 | True |

평균 band power 보존은 fade/EQ **전**의 timewarp에만 적용됩니다. 시간 적분 power×duration은 M/K만큼 바뀝니다. 새 파형의 위상·Doppler를 생성하거나 실제 독립 원천 수를 늘린다는 의미가 아닙니다.

이 검증은 full-size schedule의 분포를 확인하며, 실제 배경 마스크에 따른 source 선택·숫자 검증은 production materialization 후 별도 전수 QA가 필요합니다.

[전체 수치·해시](R4_TIMEWARP_INDEPENDENT_QA.json)
