# R10 참조 hull/day 독립 분할: metadata 실현 가능성

원509clips/178hulls/33days는 실제 hull/day와 UTC overlap/bytes union graph에서 **1component**입니다. metadata만 사용했고 weak class, acoustic array, q/y/presence, 모델 성능, Test, 미사용 report는 읽지 않았습니다.

Bipartite articulation은 hull 0개/day 33개입니다. simple graph의bridge pair는117개이며그중hull degree1은117개입니다. pair전체clip을제거한뒤실제두비어있지않은clipcomponent를만드는pair는0개입니다. 이것을'소수bridgeclips로250/250'이라바꿔말할수없습니다.

Clip union graph의단일clip articulation은0개입니다. UTC자정straddling제외는0개였습니다.

사전고정16개daypartition마다hull을A/B/drop에배정하는정확한DP로retention을계산했습니다. 양쪽45%-55%조건의최대tested결과는 **211 / 197**,총408개유지·**101개제외**입니다.

정확균형±1개기준의besttested는203 / 204입니다. tested250/250가능: False.

|사전 분할|A|B|유지|제외|
|---|---:|---:|---:|---:|
|chronological_closest_raw_clip_half|203|182|385|124|
|chronological_alternating_days|193|193|386|123|
|earliest_year_vs_later_years|112|92|204|305|
|hash_greedy_raw_clip_balance_seed2026102601|195|196|391|118|
|hash_greedy_raw_clip_balance_seed2026102602|189|189|378|131|
|hash_greedy_raw_clip_balance_seed2026102603|204|190|394|115|
|hash_greedy_raw_clip_balance_seed2026102611|195|196|391|118|
|hash_greedy_raw_clip_balance_seed2026102612|193|194|387|122|
|hash_greedy_raw_clip_balance_seed2026102613|205|192|397|112|
|hash_greedy_raw_clip_balance_seed2026102621|187|188|375|134|
|hash_greedy_raw_clip_balance_seed2026102622|197|197|394|115|
|hash_greedy_raw_clip_balance_seed2026102623|187|188|375|134|
|hash_greedy_raw_clip_balance_seed2026102631|208|177|385|124|
|hash_greedy_raw_clip_balance_seed2026102632|192|192|384|125|
|hash_greedy_raw_clip_balance_seed2026102633|199|199|398|111|
|spectral_day_affinity_closest_raw_clip_half|211|197|408|101|

## 해석 한계

이는각fixeddaypartition안의최적retention이며모든가능한daypartition의전역최적balancedseparator는아닙니다. 따라서tested최선보다좋은분할이수학적으로불가능하다고주장하지않습니다.
Hull/day/UTC/bytes가양쪽에서겹치지않는실제선택부분집합을만들수는있지만,같은receiver/providercorpus를쓰므로완전히새로운해역/센서독립검증이되는것은아닙니다.
현재R9production참조509풀·모델·가중치·학습은전혀변경하지않았습니다. CSV는후보metadata분할이며자동적용하지않습니다. 현재나미래의RMSE/AUC개선을측정한결과도아닙니다.

[전체 graph·bridge·각side분포·SHA](RESULTS.json) / [전후보 요약](all_partitions.csv)
