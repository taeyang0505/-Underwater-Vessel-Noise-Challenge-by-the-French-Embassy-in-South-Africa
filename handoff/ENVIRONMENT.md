# 실행 환경과 새 장비 준비

## 실제 사용 중인 환경

| 항목 | CPU 데이터·Tree | MPS 신경망 |
|---|---|---|
| Python | 3.12.14 arm64 | 3.14.5 arm64 |
| numpy | 2.3.5 | 2.5.4 |
| torch | 2.14.1 | 2.14.1 |
| torchvision | 불필요 | 0.29.1 |
| 장치 | CPU | Apple Metal MPS |

하드웨어는 Apple M4 Pro GPU20코어 / 통합메모리24GB. OS·CPU·GPU가 메모리를 공유하며 RTX4090의24GB전용VRAM과 다르다.

CPU interpreter:
```
/Users/hongtaeyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```
CPU 환경:
```
PYTHONPATH=.:/private/tmp/underwater-ml-deps:/private/tmp/underwater-eda-deps
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1
```
MPS interpreter:
```
/Users/hongtaeyang/Documents/Codex/2026-10-27/new-chat/.venv/bin/python
```
MPS 환경:
```
PYTHONPATH=.:ops/r14_production_runtime_py
PYTORCH_ENABLE_MPS_FALLBACK=0
```

현재 MPS runtime에는 SciPy/pandas/sklearn/LightGBM이 있다고 가정하면 안 된다. CPU가 featurecache를 만들고 MPS가 tensor를 사용한다. `ops/r14_production_runtime_py`는 임시 pure-Python dependency shim이며 제3자파일약1977개는 공개에 vendoring하지 않았다. 새 환경에는 정식 requirements로 설치한다.

CPU 추가 실측 패키지: pandas2.2.3, pyarrow26.0.0, scipy1.18.1, scikit-learn1.9.1, lightgbm4.7.0, soundfile0.14.0, matplotlib3.11.2, joblib1.6.0. 별도 파일에 버전/환경을 기록한다.

## 새 환경 설치 (현재 학습 환경을 건드리지 않음)

```bash
# macOS에서 이미 설치됐으면 생략
brew install libomp
bash handoff/setup_env.sh cpu /path/to/new-cpu-venv /path/to/python3.12
bash handoff/setup_env.sh mps /path/to/new-mps-venv /path/to/python3.14
```

setup은 새 venv 의존성만 설치하며 학습을 시작하지 않는다. pins는 관측 버전 기록이다. 다른OS/아키텍처의 wheel 제공이나 미래 설치 성공을 보장하지 않는다. 설치 실패시 버전을 몰래 바꾸지 말고 별도lock/fixture로 수치검증한다.

```bash
/path/to/new-mps-venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.backends.mps.is_built(), torch.backends.mps.is_available())
PY
```

샌드박스에서 MPS unavailable이어도 실제 감독 프로세스에는 사용 가능할 수 있다. 학습 로그와 실제1step을 근거로 판단한다. CPUfallback0 계약을 유지한다.

## 현재 러너 명령 / 중복 실행 주의

현재 run은 이미 실행 중이므로 아래를 다시 실행하지 않는다.

```
ops/run_r16_multisupport_full.py all
```

감독의 stage는 build→trees→cache→preflight→neural→finalize. CPU/MPS를 선택하지만 interpreter 절대경로가 이Mac에 고정돼 있다. 새 환경에서는 versioned portability 변경과 연결fixture가 필요하다. 원본실행의 frozen코드 상수를 수정하면 안 된다.

후속 `ops/run_r16_multisupport_fivefold.py`는 첫CSV barrier를 기다리는 waiter가 이미 실행 중. CLI는 source/`--help`로 확인하고 중복 시작하지 않는다. 옛 `requirements_gpu_cv.txt`의 TabM/TabICL/Kumo환경은 현재 최소환경이 아니다.

CUDA 이전은 자동 지원으로 간주하지 않는다. 별도버전으로 CUDA연산호환/정밀도/캐시동등성/1epoch 속도/메모리를 먼저 확인한다. 현재 paired17채널 전체 실행의 RTX4090시간은 확정 실측이 아니다.

## 디스크·메모리

작성시 원본디스크 여유 약12GiB. 후속runner는 Foldcache 생성전6GiB이상 요구하고 해당Fold8모델완료 후 hash검증된 파생cache만 제거한다. 원본/NPZ/model을 수동삭제하지 않는다. CPUTree/MPS동시대규모작업은 통합메모리경합 위험이 있으므로 새 병렬학습을 추가하지 않는다.

## WiFi / 덮개 / sleep

WiFi단절만으로는 로컬 학습이 멈추지 않는다. OSsleep이나 덮개닫힘은 설정에 따라 실행이 일시정지될 수 있다. `caffeinate -i`만으로 closedlid 동작을 보장하지 않는다. 사용자가 Amphetamine을 직접 설정한 이력이 있지만 현재상태를 추측하거나 몰래변경하지 않는다. 돌아온후 process가 살아있다면 재초기화하지 말고 진행log를 확인한다.
