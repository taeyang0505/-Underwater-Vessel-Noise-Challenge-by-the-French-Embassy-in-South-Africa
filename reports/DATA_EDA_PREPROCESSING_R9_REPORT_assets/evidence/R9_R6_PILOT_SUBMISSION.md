# R9: R6 local diagnostic pilot — 업로드 금지

**실제 공식 제출을 하지 않았습니다. 사용자 요청에 따라 local diagnostic만 생성했으며 이 파일을 업로드하지 않습니다.**

모델은 고정100epoch R6 relative8/global의 사전 지정된 best-window epoch31입니다. 보정·새 모델 선택·분포 맞춤은 없습니다.

CSV: `submissions/r6_pilot_r1/submission_r6_pilot.csv`, SHA `e2cf56255a9fc14ce9b4b27e7c772ebe0f03401b7e1cbe1dd194cb70c0ee1558`.
CPU 추론 1.777초, 1,500행×3열, ID·범위·pooling·CSV roundtrip 검증 통과.

| 출력 | 평균 | 모표준편차 | RMS | 최소 | 최대 |
|---|---:|---:|---:|---:|---:|
| R5_share | 0.354658 | 0.269158 | 0.445229 | 0.002225 | 0.961188 |
| R6_share | 0.316468 | 0.230261 | 0.391372 | 0.002715 | 0.950653 |
| R5_score | 0.920091 | 0.208614 | 0.943444 | 0.026482 | 0.999930 |
| R6_score | 0.654767 | 0.121143 | 0.665879 | 0.110687 | 0.975145 |

| R6−R5 | 평균차 | MAE | RMS 차이 | 동일 출력 Pearson |
|---|---:|---:|---:|---:|
| share | -0.038190 | 0.082032 | 0.141438 | 0.862584 |
| score | -0.265324 | 0.308382 | 0.328804 | 0.405194 |

고정 HIGH(3162–7943Hz)·SAFE(1000–5012Hz) bump와 각 예측의 Pearson, 동일 관측 mask의 첫분 missing/observed strata는 JSON에 보존했습니다. 전부 출력 설명용이며 정답·원인·보정 근거로 쓰지 않습니다. **St Francis 유사 집단의 수치는 제공하지 않습니다.** Test의 실제 site 정답이 없고, 결측 패턴은 site label이 아닙니다.

R5의 자유 presence head와 R6의 고정 share+q-contrast score는 서로 다른 비보정 진단값입니다. 단순 평균·상관 차이가 성능 우열을 뜻하지 않습니다. 이 통계로 현재/향후 R9 생성기·SNR·EQ·sampling·모델·후처리를 바꾸지 않습니다. 원 R5 파일·R6 모델·데이터는 변경하지 않았습니다.

[추론 전 동결 계획](../../R9_R6_PILOT_SUBMISSION_PLAN.json) · [전체 수치·SHA·receipt](../../R9_R6_PILOT_SUBMISSION.json)
