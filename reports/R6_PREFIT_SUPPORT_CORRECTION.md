# R6 pre-fit support correction: geometry-only source placement

This definition correction was approved before any R6 model fit or model-performance inspection. The first preparation failed during `cross_sensor_generator`, after materializing the earlier roles. It encountered a background with only middle minutes observed and a short edge-straddling reference. The old requirement that the reference intersect observed minutes had no admissible placement. Neither the seed nor a desired RMSE/share prior was changed to avoid that failure.

## Corrected semantics

All four families now select placements from native source/window geometry alone. The observation mask is absent from the trajectory-placement function. There is no placement retry, source repetition, forced warp, high-share deletion or search for an observed/nonzero target.

The full selected continuous reference trajectory defines the CPA peak after EQ/CPA shaping and **before** window clipping or observation masking. Alpha remains `10^(snr_db/10) * median(observed background minute broadband power) / full_reference_minute_peak`. A source whose CPA lies outside the window is not amplified again to normalize its visible tail.

A geometrically present source may overlap only unobserved cells. For observed mask O, the known target is still `share = sum(O*A)/sum(O*(B+A))`. Thus such a sample legitimately has observed `cell_share=0`, `share=0`, and `presence=0`, while its metadata records `source_present=true`, `no_source=false`, and `source_present_but_unobserved=true`. It is not relabeled as physically having no source, nor is it removed.

The caller's fixed 20% schedule is a **no-source fraction**, not a guarantee that exactly 20% of the observed shares are zero. Total observed-zero fraction may be higher and can differ by family/site/mask support. Both quantities and the source-present-but-unobserved count must be reported separately; they must not be counterbalanced after observing outcomes.

Observed/global SNR is JSON null when observed added energy is zero, with a reason recording the zero numerator (log ratio would be minus infinity). The full-reference SNR remains finite for an injected but unobserved source. This avoids log(0) without inventing a 0dB value.

Invalid native references and impossible geometry, such as a short reference requested to straddle both window edges, still fail explicitly. The change does not weaken source/role isolation or accept invalid waveforms.

## Preservation and versioning

The original failed implementation, numeric fixture and first preparation log are preserved under `audit/r6_pre_geometry_support/`, with SHA manifest. The existing failed partial materialization is retained by the parent and must not be reused. The parent will regenerate **all roles** into a fresh v2 output using the same configuration and seeds. Geometry candidate sets changed, so matching seeds do not imply identical old samples. Source and recipe versions identify this change.

The new recipe version is `robust_sampler_r6_v2_geometry_only`. No frozen R4/R5 source, data or model was changed. No Test/OUTER inputs, leaderboard-based parameter fitting, GPU training, SSH or external posting were used by this correction.

## Numeric verification

`python ops/verify_robust_sampler_r6.py` checks:

- All four families have identical reference shaping and placement when only the background observation mask changes.
- A two-minute edge passage with only the middle three minutes observed returns q/y/presence zero, keeps a positive full-reference alpha/SNR, and records null observed/global SNR without a RuntimeWarning.
- The same sparse construction can have 20% no-source requests and 100% observed-zero targets; those are different quantities.
- Existing raw energy identity, rounding-gap bound, high-share retention, complete-native-minute use, no repeat/warp, deterministic replay and left/right/both-edge tests remain covered.

Current source SHA256:

- `pipeline/robust_sampler_r6.py`: `59a8bafbbad9c42c7628d2283ac5f9fb3550a6945686958ea207ebeb61b0722b`
- `ops/verify_robust_sampler_r6.py`: `df7e1afe509c9d1242ad69ef5ee2570fe3825b7a97c42c2d7385cc084aae26a1`

The verification is numeric only and does not establish that these synthetic laws reproduce the official hidden generator.
