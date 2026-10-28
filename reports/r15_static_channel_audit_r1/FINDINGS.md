# R15 추가: 실제 R14 PCA·LF-anchor 입력 채널 감사

기존에 맵 캐시가 없었던 항목을 미측정 상태로 끝내지 않고, **고정된 Fold0 TRAIN PCA와 실제 공식 전처리만** 적용했다. 모델 백본·체크포인트 로드·모델 예측·학습·재보정은0회다. 이전 감사는 그대로 보존하고 이 폴더에 추가 결과를 저장했다.

## 실제 입력 식과 일치 확인

- basis: `data/synthetic/group_cv_physics_r13_variants_v2/fold_00/basis.npz`, SHA `8cc263b3a43f0715daf755bed142b2e45b4373030a1e89fa7c499d77283d2d8d`
- basis는 Fold0 TRAIN의 원천2,000개 배경에서만 적합한 동일 파일이다. query에서 PCA를 새로 적합하지 않았다.
- 공식 NumPy `features_r14_anchor.channels`와 Torch CPU FP64 `features_r14_production.ProductionFrontend`를1,500개 전부 비교했다.
- 최대채널 차이 `1.421085492696024e-15`; 관측에너지 w 차이 `0.0`. 별도 부분결측 숫자fixture도통과했다.
- LF=31.6–158.5Hz, MF=251.2–3162.3Hz의 **평균 dB차**를사용한다. linear broadband에너지비나 실제선박fraction과동일한값이아니다.
- PCA signed/positive RMS는safe0..22채널을관측cell에서요약하고10배해dB로표현한다. LF-anchor residual은공식frozenTRAIN평균모양을저역에고정한잔차이며20배해dB로표현한다. 실제배경/추가성분의정답을제공하는것은아니다.

## 실제 정적 채널 요약과 저장 예측의 상관

| 특징 | R9 share P / S | R14 share P / S | R14 score P / S | Wide CNN share P / S |
|---|---:|---:|---:|---:|
| PCA_signed_residual_RMS_db | 0.0430 / 0.0933 | 0.0409 / 0.0906 | 0.0425 / 0.0795 | 0.0352 / 0.0804 |
| PCA_positive_residual_RMS_db | 0.0374 / 0.0910 | 0.0380 / 0.0911 | 0.0398 / 0.0804 | 0.0330 / 0.0811 |
| MF_minus_LF_mean_db | 0.1765 / 0.2079 | 0.1366 / 0.1597 | 0.1296 / 0.1378 | 0.1244 / 0.1388 |
| LF_anchor_residual_RMS_db | 0.2121 / 0.2673 | 0.1647 / 0.2240 | 0.1687 / 0.2139 | 0.1507 / 0.2059 |
| LF_anchor_positive_fraction_mean | 0.1762 / 0.2184 | 0.1332 / 0.1677 | 0.1307 / 0.1473 | 0.1207 / 0.1462 |

P=Pearson, S=Spearman. PCA잔차 RMS와R14예측의주변상관은작다. 이값들만으로모델이정적특징을전혀쓰지않는다고결론낼수도없고, 정적특징이Public오류를일으켰다고확정할수도없다. 비선형상호작용·다른특징과의결합은이일변량상관에서격리되지않는다.

## 낮은 시간변화 × 실제 정적채널 상위25%

low-temporal 조건은이전감사에서고정한bump_high와std_broadband의각하위25%교집합이다. 새정적채널의각상위25%와별도로교차했다. 예측이나숨겨진정답에맞춰그룹문턱을고르지않았다.

| 그룹 | N | R9 share 평균 | R14 share 평균 | Wide CNN 평균 |
|---|---:|---:|---:|---:|
| all | 1500 | 0.264744 | 0.303326 | 0.309823 |
| low_temporal_predeclared | 138 | 0.055353 | 0.052362 | 0.050678 |
| low_temporal_AND_high_PCA_positive_residual_RMS_db | 10 | 0.076246 | 0.090467 | 0.084927 |
| low_temporal_AND_high_MF_minus_LF_mean_db | 19 | 0.080784 | 0.070305 | 0.061130 |
| low_temporal_AND_high_LF_anchor_residual_RMS_db | 8 | 0.076946 | 0.062961 | 0.051235 |

작은교차그룹의N은10/19/8이다. PCA상위교차에서는R14평균이R9보다높고, MF−LF·anchor잔차상위교차에서는낮다. 따라서세정적요약을모두하나로묶어‘정적값높은평탄창을큰배로오판한다’고단정할수없다. 이것은오탐률/성능비교가아닌출력크기비교다.

## 파일과 한계

- [전수실제채널요약](static_channels1500.csv) · [공식14채널명](channel_definitions.csv)
- [저장예측·기존특징·실제정적채널결합](joined1500.csv)
- [모든실제채널–예측상관](feature_prediction_correlations.csv)
- [교차그룹통계](lowtemporal_x_actualstatic_groups.csv) · [추정유형별정적입력](site_static_stats.csv)
- [사전계획/입력SHA](plan.json) · [공식구현전수일치/출력SHA](RESULTS.json)

Test 정답과 실제site는여전히없다. probe_site는기존kNN의추정유형이며rawclassifier역산·라벨추정·LB목표맞춤·새가중치/보정·제출파일생성은하지않았다.14채널은기존관측값의결정적변환이며새물리정보를추가한것이아니다.
