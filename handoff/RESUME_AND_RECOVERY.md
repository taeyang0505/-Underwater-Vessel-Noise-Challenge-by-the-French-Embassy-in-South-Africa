# 새 검증된 복구 경로 — 2026-10-28

## 2026-10-28 16:22 KST 우선 갱신 — 이전 snapshot보다 우선

- 첫 CSV 완료 뒤 사용자 지정 네 감사(score 집계/연산자 회귀/정적 입력 치환/1,000회 그룹 부트스트랩)를 별도 일회성 CPU 실행으로 연결했다. [준비 보고서](../reports/R16_POSTCSV_AUDIT_READINESS.md), [현재 상태 스냅샷](POSTCSV_AUDIT_LIVE_SNAPSHOT.json)을 먼저 읽는다. CSV 및 실제 네 감사 결과는 아직 미완료다.
- 기존 학습 PID와 waiter가 종료돼 있었음을 발견했다. source/data/cache/checkpoint/Adam/scheduler/RNG 검증 후 61epoch 경계에서 복구했으며 62epoch 이상 실제 완료를 확인했다. 현재 D0 CNN 약67epoch(스냅샷). 기존 PID나 오래된 state를 살아 있는 학습으로 보고하지 않는다.
- 별도 검증된 복구 도구: `ops/resume_r16_multisupport_lifecycle.py`, `ops/continue_r16_multisupport_verified.py`. 원본 frozen trainer는 수정하지 않았다. [검증 영수증](../ops/r16_multisupport_resume_20261028_r1/VERIFIED.json). 한 번 명시적으로 복구한 실행이며 재실행하면 거부한다. 또 죽었다고 기존 명령을 자동 반복하지 않는다.
- Documents 가상환경 패키지가 dataless 파일 읽기에서 정체됐다. 기존 설치된 로컬 Python3.12/PyTorch2.14.1에서 **MPS GPU를 유지**한다. 기존 최적 체크포인트의 저장 예측 재현 오차0, CPU/MPS Adam/RNG 복원 fixture 오차0. 레시피·손실·100epoch 계약은 그대로다.
- 현 recovery supervisor PID36954, child36956, Fold1~4 waiter37128, 사후 감사 waiter37129. 이는 snapshot PID이며 실제 command와 생존 여부를 확인한다. 중복 실행 금지.
- Release `handoff-2026-10-28`의 model ZIP은 **이 갱신 이전 checkpoint 스냅샷**이다. 새61epoch 복구 snapshot/이후 weights가 release에 올라갔다고 주장하지 않는다. 새 학습은 canonical 경로에만 있다.
- 공통 ln4 차감은 순위를 바꾸지 않는다. 패널별 rank는 코호트 의존 진단으로 한정한다. one-SE 권고는 기존 첫 CSV/정책을 덮어쓰지 않는다. Test 관측값/query 배열은 감사에서 읽지 않는다.


아래 `tested resume CLI가 없다`는 이전 상황 기록이다. 새 도구는 검증된 한 번의61epoch 복구에 한정하며 범용 자동 재시작 도구가 아니다.

---

# 작업 중단·재접속·재개 절차

## A. 대화/IDE만 바뀌고 학습이 살아 있음

1. `python3 handoff/doctor.py`로 canonicalroot의 live state를 읽는다.
2. statePID의command를 `ps`와 대조. 프로세스가살아있으면 새학습을 시작하지 않는다.
3. 최근 epochlog·progress.json과bestcheckpoint파일mtime를 확인. 같은epoch가오래면rows_seen/optimizer_steps가움직이는지 본다. TFGrid는다른모델보다느리다.
4. 다음모델로넘어가는정상경계를기다린다. 새로운agent가 frozen코드를포맷/정리하지않는다.
5. 보고서를갱신할때live값과snapshot시각을구분한다.

읽기전용 예시 (원본 root에서):
```
cat ops/r16_multisupport_full_r1/state.json
cat models/r16_multisupport_full_r1/progress.json
cat ops/r16_multisupport_fivefold_r1/state.json
ps -p 12773,21489,23279 -o pid,ppid,etime,command
```
PID는스냅샷값. 새state의PID를확인한다.

## B. 첫Fold0가완료됨

- `models/r16_multisupport_full_r1/final/COMPLETED.json` 및 첫CSV 존재/SHA/1500unique/schema/range를확인한다.
- 후속waiter가이미있으면그대로둔다. parentcomplete만보고수동후속실행하지않는다.
- 후속`handoff()`는원본코드/weights/data핀과Fold0각100epoch완료ledger를검사한다.
- 이후32회NN(4아키텍처×D0/H0×4Fold)을독립초기화한다. Fold0best로warmstart하지않는다.

## C. 프로세스가실제로죽었음 / partial checkpoint

**현재는 tested resume CLI가 없다.** `latest.pt`에optimizer/scheduler/RNG가있어도runner의`all`은partial폴더가있으면거부한다. 이는부분학습을새완료로오인하지않기위한설계다.

1. 즉시자동재시작하지않는다. FAILED/log/lastcheckpoint/manifest/코드SHA/환경을읽기전용snapshot으로보존한다.
2. latest.pt의run/regime/model/fold/epoch/globalstep/optimizer/scheduler/RNG/checkpointdata/sourcepin을대조한다. pickle형식은신뢰하는직접학습파일만로드한다.
3. 원본파일수정없이새버전에`--resume-from`와resume출처ledger를구현한다. epoch경계인지 mid-epoch인지와sampler순서/cacheddata일치부터확인한다. 저장정보가부족한경계는정확재개라고주장하지않는다.
4. deterministic소형fixture에서중단없는Nstep과중단/재개Nstep의weights·optimizer·scheduler·RNG·best선택이같은지확인한다. MPS비결정성은허용오차/재현성한계로보고한다.
5. 완료모델은기존SHA로검증하고재학습하지않는다. 미완료모델은검증된resume또는명시적새버전fresh학습으로처리한다. 기존폴더덮어쓰기/COMPLETED위조/partialbest를완료로승격금지.
6. 새복구정책이codehash계약을바꾸므로별도runID/receipt/필요한사용자확인을남긴다. 원본완료ledger는수정하지않는다.

## D. 다른장비로옮김

Release는직접학습weights snapshot이고데이터/학습캐시가아니다. [DATA_AND_RESTORE](DATA_AND_RESTORE.md)의비공개데이터복원이먼저다. 원본sourcepins는bytes기준이므로공개문서링크수정은괜찮아도학습코드포맷변경은금지. archivedcheckpoints의running/completed구분은ledger에있다.

## 완료판정

Tree완료만으로최종전체5Fold완료가아니다. NN4종×2조건×5Fold의각100epoch완료,회귀/분류bestcheckpoint,ownfoldCV,lockedpanel,정책freeze,TestCSV/ID/SHA,최종receipt가필요하다. 각아키텍처의전체Fold평균은Test/local_fit용이고,CV에는해당Fold만사용해야누수가없다.
