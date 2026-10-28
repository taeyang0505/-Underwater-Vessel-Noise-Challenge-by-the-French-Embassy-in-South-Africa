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
