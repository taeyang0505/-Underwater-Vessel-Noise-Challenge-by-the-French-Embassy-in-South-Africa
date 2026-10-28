# R16 target-matched natural support: method B preregistration

User-authorized fallback after saved-prediction multi-support blending. One new small condition only: S1_TM_H0_F1.

- Preserve S1_H0_F1 positive source/reference IDs, natural contiguous support, backgrounds, O and all 800 original negatives exactly.
- For each positive, set k = y_D0/(1-y_D0) * sum(OB)/sum(OA_S1); multiply added_full_power by k; recompute E=B+OA, raw db, rounded0.1db, q=OA/E, y=sum(OA)/sum(OE). All raw y values must equal paired D0 values to1e-12; no labels overwritten independently of physics.
- Same seed2026102817, F1, 4000 TRAIN/2400local_fit, Adam/dropout0.1, NN100epoch/noES, Tree600/ES40. Reuse immutable learning code. No source/split/purge change, no Test or Test-derived EDA access.
- Select checkpoints only on local_fit. Regression Tree60%+NN40%; own score independent Tree50%+NN50%. Also retain old D0F1 score comparison, no learned blending on exposed panels.
- Evaluate exactly the inherited primary3 axes and independent native12 versus HF/SF on frozen models. Retain all thresholds in r16_clean_support_r1/PROTOCOL.md; no result-dependent relaxation. This is repeated development evaluation, not new blind evidence.
- Method A is already fully measured. This single B pilot is the only additional small fitting authorized here. If B meets criteria, start a new full F1 production version without hotpatching or resuming incompatible 8-channel checkpoints. If B does not, report the numerical limitation; do not manufacture a pass or a final full-model CSV.
- Target marginal equality is verified; complete error correction is a hypothesis, not guaranteed.
