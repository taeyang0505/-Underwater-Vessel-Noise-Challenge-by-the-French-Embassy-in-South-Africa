# R6 relative input numeric fixture

Status: **PASS, numeric fixture only**. No competition TRAIN/INNER/OUTER/Test files, GPU or SSH were used. The two Adam updates per model check gradients and reload compatibility; these weights are not saved or used for training.

The six db-only channels and their formulas are documented in [relative_features_r6.py](../pipeline/relative_features_r6.py). The pooling weights use the same 0.1 dB rounded observations. Known raw-precision q/y labels remain unchanged.

For |dB error| ≤0.05, fixed-q pooling changes by at most **0.0057563991** in real arithmetic: tanh(ln(10)×0.05/20). Float32 output rounding is additional. The two-cell near-worst example changes 0.5057560933 to 0.5000000000; this is an acknowledged mismatch, not a relabeling.

Common gain invariance is exact after quantization, and for pre-quantization grid-aligned gains when rounding commutes. Arbitrary gain can cross rounding boundaries. Channels 0–3 are also invariant to static band offsets under that caveat; the spectral-profile channel and energy weights are not.

## Pre-fit numerical correction

Before dataset training or held-out model evaluation, the initial hard-sign low-band fraction was replaced by `tanh(mean_observed_low(max(adjacent_delta_db, 0))/3.0)`. The fixed 3-dB divisor makes the gate continuous in its unquantized deltas; 0.1-dB input quantization still creates finite steps. The gate multiplies positive mid/high-band delta/10 only (indices 10–24); the low-band mean uses indices 0–8. Missing adjacent pairs do not enter the denominator, and no low-band pairs means zero gate. The CNN architecture, target labels and other five channels are unchanged.

The same +0.037-dB numeric stress changes the old coherence channel by 0.2677777778, and the corrected channel by 0.0143318176. [Initial fixture JSON](R6_RELATIVE_FEATURES_FIXTURE_hard_sign_prefit.json) / [initial fixture MD](R6_RELATIVE_FEATURES_FIXTURE_hard_sign_prefit.md) remain archived.

Fixtures cover missing bands/minutes, no differencing across missing minutes, hand-calculated co-rise, chunk/permutation replay, independent numerical bound, 6/8-channel gradients, energy pooling and bitwise checkpoint reload.

[Detailed JSON](R6_RELATIVE_FEATURES_FIXTURE.json)
