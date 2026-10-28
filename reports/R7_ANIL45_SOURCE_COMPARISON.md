# Anil45 공개 외부 데이터 고지와 우리 원천 사용 비교

**같은 VTUAD v1.3.2 원천을 쓴다는 점은 확인된다. 그렇다고 학습에 쓴 clip 수·합성 정답·분할·모델까지 같다는 근거는 없다.** 이 감사는 현재 보존 manifest·코드·다운로드 기록만 읽었다. 신규 다운로드/모델 실행/원천 복원은 하지 않았다.

## 1. 공개 글에서 확인된 범위

[Zindi discussion35531](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/35531)은 Anil45의 외부 데이터 고지다. 부모 에이전트가 로그인된 Whale의 전체 accessibility tree를 직접 확인했고 [구조화 관측](R7_ANIL45_POST_OBSERVATION.json)에 출처를 남겼다. 이 에이전트의 웹 reader는 해당 페이지를 열지 못했으므로 그 관측과 독립 웹 확인을 구분한다.

고지는 VTUAD rebuilt v1.3.2/CC BY4.0/ONC를 명시하고, strict vessel3parts·strict background·같은 record의12per-class archives를 열거한다. starter의 minute_band_power로25대역 변환, uncalibrated, Test inference-only를 밝힌다. 표시된 댓글은0이다. **실제로 학습에 포함한 clip/group 수, 모든 파일의 사용 여부, 합성 recipe·배율·target prior·train/validation분할·모델·앙상블·후처리는 공개하지 않았다.** 공식 평가 점수는 `1−RMSE`이므로 당시 score.904607408을 RMSE로 환산할 수 있으며, 사용자 screenshot의 실제 RMSE 표시는 **.095392591**이다. 다만 그 점수나 rank2라는 사실만으로 원천 사용량 또는 특정 데이터·모델의 단독 성능 기여를 추론할 수는 없다. 표시 자릿수 반올림 차이는 구분한다.

독립적으로 [공식 Zenodo record](https://zenodo.org/records/23037895)를 확인했다. strict509vessel+16dedicated background와 기존12per-class container는 **같은525scenario의 서로 다른cut view**다. 둘을 합쳐 독립 원천이 두 배라고 세면 안 된다. v1.3.2는 일부 옛 container의 interval 밖/반복된 확장 문제를 문서화한다. 이 사실만으로 그 작성자가 어떤 구간을 어떻게 사용했는지는 알 수 없다.

## 2. 우리가 실제 다운로드한 WAV archive

보존 download manifest에 기록된 WAV archive volume은 아래 **4개**, 논리적으로는 vessel ZIP stream1개와 background ZIP1개다. part를 MultipartReader로 연결해 읽으므로 별도 합본 ZIP을 복제할 필요가 없다.

| 실제 파일 | bytes | 다운로드 당시 검증 MD5 |
|---|---:|---|
| README_vtuad_strict_wav_vessel_parts.txt | 1,152 | `43bab1eba183cdfac74130cdfa135620` |
| vtuad_strict_wav_vessel.zip.part01 | 1,237,445,900 | `6e3fe8245218faa4f94868531e272344` |
| vtuad_strict_wav_vessel.zip.part02 | 1,237,445,900 | `af6408581f6981f45ffdbce811e9f558` |
| vtuad_strict_wav_vessel.zip.part03 | 1,237,445,899 | `dfa59398483a651765a6265df76febad` |
| vtuad_strict_wav_background.zip | 926,037,631 | `d28ad0fb6ccc98341ba4f7e953d9c68f` |

합계 **4,638,376,482bytes**다. 현재 파일 크기는 기록과 모두 일치하고 download_complete=true이며, 변환 inventory는525WAV의 provider per-WAV SHA 검증을 모두 기록한다. 이번 감사에서4.6GB를 다시 MD5계산하거나 audio를 열지는 않았다. **현재 프로젝트 전체 inventory와 보존 download manifest에는 `vtuad_wav_<class>.zip`12개가0개이며, 현재 prepare pipeline에도 해당 archives 소비 경로가 없다.** precomputed log-mel 역시 우리25band 경로에서 쓰지 않았다. registry_anonymized.csv를 사용했고, metadata.csv의 별도 로컬 파일은 없다.

## 3. 다운로드·변환·positive 참조 편입은 서로 다른 수다

- **변환 inventory525=509vessel archive+16dedicated ambient controls.** strict background ZIP도 다운로드하고 band bank에 변환했다. 전체를119개만 내려받은 것이 아니다.
- vessel archive의 archive_role과 weak_class는 다르다. archive_role=vessel_passage 안에도 weak_class=background인22개가 있다. 이것은 별도 background archive의16개와 같은 집합이 아니다.
- 과거 R2 CV positive 자격 pool은218개다. 나머지307개는 reserved calibration6/domain15, hull/day conflict260, cross-day conflict1, train/validation에 있으나 자격 없는25개로 나뉜다. 마지막25개는 **dedicated ambient10개+vessel-role 비positive15개**다. 과거 보고서에서 이 자격 없는25개를 모두 background라고 표현한 문구는 archive_role 구분상 부정확하다. 또한307개가 모두 background라는 뜻도 아니다. 이 명칭 정정은 기존 제외 정책을 해제하거나 해당 clip을 자동 복원하는 근거가 아니다.
- R4/R5는218개 안에서 bridge99개를 추가로 제외해 **119clips/22selected components**를 사용했다.119는 전체5fold pool이다. 개별 Fold0 TRAIN은69/12, INNER25/5, oldOUTER25/5다. dedicated ambient16개는 이 positive reference pool에0개다.
- R6 guardrails는 oldOUTER25/5를 다시 제외한 **94/17**만 사용한다. dedicated ambient positive0개다. 아래 역할별 pool과 합성 예제 수는 별개다.

| R6 역할 | clips | components |
|---|---:|---:|
| cross_generator | 18 | 3 |
| cross_sensor | 7 | 2 |
| inner | 18 | 3 |
| train | 51 | 9 |

초기 MixtureSampler도 archive_role=vessel_passage만 positive로 고른다. 이전 NativeFirstSampler는 별도 ambient 경로를 지원하지만 기본확률0이며, 그 API의 존재만으로 ambient가 실제 특정 모델 학습에 쓰였다고 단정하지 않는다. 현재 R4/R5/R6의 위 manifest에서16ambient는 positive로 편입되지 않았다. 이번에 reserved307이나 bridge99를 복원하지 않았다.

전체525corpus의 보수적 hull/day/interval graph는 여전히1component다.22는 선택 pool 안의 직접 격리 단위이지22개의 새 hydrophone/해역/독립 물리 녹음 환경이 아니다. [기존 selection 감사](r4_reference_selection.md).

## 4. 같은25대역 front-end가 같은 합성 법칙을 뜻하지 않는다

| 단계 | 우리 구현에서 확인한 처리 | 공개 고지와 비교 가능한 범위 |
|---|---|---|
| audio→power | native fs/one channel,2s Hann/detrend PSD,base10 band edges,선형power 적분;30개segment가 완전한1분 | starter minute_band_power와 같은 핵심 변환 정의. uncalibrated digital energy이며 pressure calibration 아님 |
| 중간 bank | complete2s power와 complete-minute power를 둘 다 보존 | 고지의 minute helper 언급만으로 내부2s자료 사용/미사용을 추론할 수 없음 |
| R4/R5 | 2s power trajectory를 누적선형power 기반으로1/3..3배 단조warp;120..900s/fade/EQ≤4dB;배치 후30bin평균;정해진share에서alpha 역산 | 이 exact warp/범위/정답 prior가 Anil45 방법이었다는 공개 근거 없음 |
| R6 minute_first | 먼저 native complete60s 평균 후 minute placement;warp없음;EQ≤6/no lowpass;full-reference peak SNR로alpha,share는성분에서계산 | 고지의 front-end와 비교할 수 있지만 합성 scaling 동등성은 미공개 |
| R6 short/cross | 실제1–3분 contiguous crop+제어 envelope 또는edge/whole2s trajectory;no warp | 공식 hidden generator나 작성자 합성의 재현이라고 주장하지 않음 |

정답 q는 추가한 **whole measured recording** 성분의 수학적 몫이지 사람이 분리 검증한 순수 선박 성분이 아니다. AIS 자동 scenario label 역시 수동 청취한 vessel-only ground truth와 다르다. [우리 front-end](../pipeline/audio.py) · [R4/R5 sampler](../pipeline/passage_sampler_r4.py) · [R6 sampler](../pipeline/robust_sampler_r6.py).

따라서 지금 확인할 수 있는 차이는 ‘우리 strict-only 다운로드/조건부 reference subset/명시된 생성 law’와 ‘상대의 더 넓은 파일 고지/미공개 학습법’이다. 상대가12개 archive 전체로 더 많은 독립 데이터를 학습했다거나 특정 모델·ensemble을 썼다고 추정해 성능 차이를 설명하지 않는다. 같은 공식 source를 사용한 사실은 우리 작은 synthetic INNER가 공식 평가를 대표한다는 증거가 아니다.

[기계 판독 전체 facts·입력 SHA](R7_ANIL45_SOURCE_COMPARISON.json) · [보존 다운로드 기록](../external/vtuad_v1.3.2/download_manifest.json) · [공식 record](https://zenodo.org/records/23037895)
