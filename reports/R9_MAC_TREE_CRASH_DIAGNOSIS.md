# R9 Mac LightGBM 종료 원인과 환경 검증

첫 실행 `models/r9_mac_trees_r1`은 PID2679가2026-10-26 15:27:33(KST)에 `SIGSEGV`로 종료됐다. 완료된 tree 모델은 없다. 실패 출력, 전체 stdout, 원래 progress와 stale lock을 보존했고 `FAILED.json`으로 결과 가져오기를 차단했다. 자동 재시작이나 미완료 모델 재사용은 하지 않았다.

정확한 PID와 시각이 일치하는 macOS crash report는 `EXC_BAD_ACCESS / KERN_INVALID_ADDRESS 0x580`을 기록한다. 오류 스레드의 `__kmp_suspend_initialize_thread`와 `kmp_flag_64::wait` 호출이 서로 다른 `libomp.dylib` image를 교차한다. 당시 프로세스는 Torch, Homebrew, scikit-learn의 OpenMP runtime 세 개를 함께 로드했다. 이는 라이브러리 충돌이라는 강한 증거다. 메모리 부족으로 단정할 근거는 없으며, 정확한 boosting iteration은 로그로 확인되지 않는다.

[LightGBM 공식 FAQ §10·§16](https://lightgbm.readthedocs.io/en/latest/FAQ.html)은 여러 OpenMP runtime의 충돌이 비정상 종료를 일으킬 수 있으며 단일 runtime을 사용하도록 설명한다. [동일 프로젝트의 macOS/PyTorch 관련 이슈](https://github.com/lightgbm-org/LightGBM/issues/6595)도 같은 종류의 충돌을 다룬다. 이번 원인 판단은 그 공개 사례와 더불어 실제 crash stack 및 아래 로컬 재현에 근거한다.

## source 데이터를 사용하지 않은 재현

생산 CLI의 전체 모듈 import 후 합성 배열과 실제3,827열 feature transform,4threads, 최대8round의 작은 tree fixture를 실행했다. 실제 TRAIN·Test·report, GPU 또는 SSH는 사용하지 않았다.

| 조건 | 합성 TRAIN 행 | 로드된 OpenMP runtime | 결과 |
|---|---:|---:|---|
| 원래 환경 |2,048|3|동일 SIGSEGV 재현(exit139)|
| Homebrew 단일 runtime,독립 process A |2,048|1|PASS|
| 동일 조건,독립 process B |2,048|1|PASS|
| 동일 조건,크기 증가 |8,192|1|PASS|

환경 변경은 **`DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib` 한 항목**뿐이다. `_dyld_get_image_name`과 `threadpoolctl`로 시작·종료 모두 실제 로드된 runtime이 하나인지 확인했다. 고정된 실제 라이브러리는 Homebrew `libomp22.1.8`이며 SHA는 환경 lock에 보관했다. 패키지 버전, R9 source/config, 스레드 수, 학습 정책을 바꾸지 않았다. `KMP_DUPLICATE_LIB_OK`는 사용하지 않았다.

성공한 세 fixture 모두 저장·재로딩 및 batch1/7/256 예측의 최대 차이가0이었다. 같은2,048행 독립 process 두 개의 share/score/classifier 예측도 bitwise 동일했다. 이 결과는 실행 환경과 artifact 연결 검증이며 성능 개선이나 전체79k행 학습 성공의 증거는 아니다.

## 다음 실행의 경계

실패한r1을 유지하고, 이 단일 runtime 환경을 lock 검증한 **새r2 출력 경로**에서만 원래의5fold 순차 Mac CPU 학습을 다시 시작하는 것이 제안된 조치다. 최대1,500round/ES40/threads4 및 원래 데이터·seed·source·config는 그대로 유지한다. 실제 전체 크기의 메모리, 처리량 및 모든 native 오류의 부재는 작은 fixture만으로 보장하지 않는다. r1의 materialized TRAIN이나 일부 모델을r2에 재사용하지 않는다.

이 문서 작성 시점에는 환경 수정 후 전체 source 학습을 시작하지 않았으며, 부모 실행 담당자의 검토를 기다린다. GPU 신경망 실행은 별도 작업으로 이 Mac 전용 환경변수를 전달하지 않는다.

- [기계 판독 진단](R9_MAC_TREE_CRASH_DIAGNOSIS.json)
- [환경 lock](R9_MAC_TREE_ENVIRONMENT_LOCK.json)
- [정확한 crash 추출](R9_MAC_OPENMP_DIAGNOSTIC/crash_extract.json)
- [독립 숫자 재현 코드](R9_MAC_OPENMP_DIAGNOSTIC/numeric_probe.py)
- [원래 환경 재현 stdout](R9_MAC_OPENMP_DIAGNOSTIC/r9_omp_baseline_2048/stdout.log)
- [8,192행 성공 결과](R9_MAC_OPENMP_DIAGNOSTIC/r9_omp_homebrew_8192_c/result.json)
