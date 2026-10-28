# R6 fixed100 CPU execution budget

**Chosen execution:** five fresh independent CNN processes,2 Torch threads each (10 total intra-op threads on the14-core host stated by the parent). Each executes100 epochs with Adam/dropout0.1; early stopping is disabled. Source data and existing R5 core remain frozen.

| Sequential synthetic hardware fixture | Median Adam step, seconds | Peak process RSS, MiB |
|---|---:|---:|
| 1 threads | 0.201279 | 414.8 |
| 2 threads | 0.153865 | 418.9 |
| 4 threads | 0.130266 | 414.8 |

At2 threads,100×64 synthetic training steps project to16.41min per arm before validation/preparation. The optimistic no-contention five-arm wall time is **about17–19min**. This is not a concurrent benchmark.
**Operational planning:** allow **30–60min**, with **80min** as a conservative planning allowance. No finish time or speedup is guaranteed; concurrent feature preparation, CPU/memory bandwidth and thermal conditions can dominate. No new heldout evaluation is included in this run.

The short fixture RSS is about419MiB per process. Adding about55MiB TRAIN+INNER arrays and119MiB grids/weights suggests roughly600–800MiB steady state per arm, or3–4GiB across five. Full-array float64 transforms/prior preparation can temporarily exceed this sum. The task provides24GiB RAM, with existing application/compression pressure outside these processes. Actual five-arm peak RSS is unmeasured.

## Integrity and concurrency review

- All17 original R5 core source hashes match the archived GPU preflight receipt. Existing R6 code/config, seven data-array byte hashes, metadata, source and hard-negative bindings matched independently.
- Workers write separate arm folders, logs and progress files. Only the parent writes shared progress. Models use separate fresh interpreters; existing data files are read-only.
- The loop has100 epochs and no patience break. It separately retains strict INNER-window best, INNER-macro best and epoch100. Shadow ES7 is analysis from this new trajectory and never stops optimization.
- Thread2 versus prior4 can change floating-point trajectories. No old bitwise-prefix claim is made. Existing LightGBM artifacts are preserved; neural epoch changes do not require tree refitting.
- Operational issue reported to parent: an exception from a worker future can wait for remaining subprocesses at executor shutdown before outer FAILED.json is written. Early failure-status recording would make monitoring immediate.

The former locked panels have already been evaluated. Any later repeat is a development comparison, not an untouched independent test. This budget audit starts no actual source training and reads no challenge Test or prediction arrays.

Earlier4-thread serial timing projection was73.35min for all five; it is retained as historical evidence, not the chosen schedule.

Full assumptions, per-arm timings and hashes: [JSON budget](R6_FIXED100_EXECUTION_BUDGET.json).
