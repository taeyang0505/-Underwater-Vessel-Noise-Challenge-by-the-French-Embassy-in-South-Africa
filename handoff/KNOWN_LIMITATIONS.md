# 확인된 제한과 남은 위험

1. **공식일반화 미검증:** 최신R16의Public점수는없다. local3축개선·R11개선·분포통계가LB0.78/0.90보장은아니다. 과거R5와R14는로컬개선에도공식악화했다.
2. **음원다양성:** 현재A′75/13groups,전역heldB19/C′25. strict509전체아님. ≥15분독립그룹하나뿐,이를보류한Fold4train에는≥11분참조없음.
3. **NestedCV범위:** 기존R9Fold0TRAIN배경78869내부다. 전체남아공모든해역/센서를일괄5분할한검증이아니다.
4. **합성sample종속:** paired조건/반복배경/6연산자같은400fit배경을공유.2400fit샘플이2400독립관측아님.624608은독립실녹음624608개가아님.
5. **미달승인:** D0F1+H0.5/.5의native12/HF파일럿RMSE0.117553이0.11기준미달. 사용자가이를명시한본체진입을승인했다. 무한추가진단을이유로이미승인된본체를중단하지않는다.
6. **입력정보:** absolutelevel/q0/PCA/LFanchor는제거했지만relativeprofile/sqrtw에는정적대역분포가남는다. globaldBoffset불변성과arbitraryEQ불변성은다르다. q05/q20가항상순수배경을복원하는것도아니다.
7. **정밀도:** roundeddb의w와rawq/y에최대gap.0058허용. 정확항등식으로소개하면안된다. 데이터전수QA실측은관련JSON확인.
8. **독립분류포화:** 최신Tree의75.13%score≥.9. 출력이확률처럼보여도calibration미검증. 이를단독정상/실패판정으로쓰지않는다. score높음은share높음과별개.
9. **무제한최적화아님:**100epoch/noES/최적checkpoint는globalminima보장아님. validation반복선택으로발생하는과적합가능성은남는다.
10. **후처리범위:** 실제는clip+고정평균/cappedfit결합뿐. SAM·스무딩·backgroundPCAprojection·zero-gating·temperature/power보정/RAG는적용안됨. 미적용아이디어를실제적용으로말하지않는다.
11. **사전학습대형모델:** TimeMoE-200M/Dasheng는과거연결검증만,현재17채널본체학습에미포함. 큰모델별도GPU구매/학습시작은현재인계범위아님.
12. **재개·이식성:** 현재partialresumeCLI미검증,Mac절대interpreter경로,임시runtime/PYTHONPATH의존. 새clone에는데이터없음. setupscript만으로동일실행보장안됨.
13. **공개백업한계:** 진행중weights는한파일별원자적snapshot,여러checkpoint의전역동시시각snapshot아님. latest/best가서로다른epoch인것은정상. history와epoch검증후사용.
14. **디스크:** 원본작성시여유12GiB. 무분별한cache/데이터복사금지. 원본고정데이터를지우는우회금지.
15. **원자료권한:** 대회데이터재배포허용미확인. 공개repo는원본/합성행렬/audio/pretrainedvendorweights를제외했다. 파일목록·출처·복원절차를제공하며데이터가없어도실행가능하다고속이지않는다.
