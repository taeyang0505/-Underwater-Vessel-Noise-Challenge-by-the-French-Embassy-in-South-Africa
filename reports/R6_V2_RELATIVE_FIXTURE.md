# R6 v2 pre-fit numeric fixture

**PASS — numeric fixture only.** No competition source data, model selection, GPU, SSH or external transmission. Two Adam updates on generated numbers check gradients only; no model checkpoint is saved.

The [pure eight-channel grid](../pipeline/relative_grid_r6_v2.py) appends temporal q05-relative dB/15 and sqrt(w) to the unchanged six-channel v1 grid. q05 is not a background label. Features and weights use rounded 0.1-dB input; original precision labels are not changed. Arbitrary gains can cross quantization bins. Profile and energy channels are not invariant to per-band offsets.

The [CNN](../pipeline/relative_models_r6_v2.py) uses the unchanged tiny local body plus optional per-minute frequency context. Means over bands 10:20, 20:25 and 15:23 are concatenated, projected and sigmoid-gated, then broadcast to all bands. An isolated hidden-tensor test verifies an explicit high-to-low gradient and perturbation path; it does not claim older GroupNorm had no frequency-global effects.

There is no learned presence head. Fixed score is 0.5×pooled share + 0.5×maximum observed temporal q contrast over bands 10–22. The temporal minimum excludes missing cells. Its clamped logit drives the BCE auxiliary term. This predeclared formula is not evidence of improved AUC or calibration. Legacy8 and relative8 use the same architecture and score; context-off is a separate supported ablation.

Tests cover pure-module Torch absence, old-grid bitwise equality, q05 arithmetic, mask/chunk/gain behavior, hand score cases, context-path gradients, finite CNN gradients, physical pooling, batch tolerance and bitwise reload. V1 source SHAs remained unchanged.

[Detailed JSON](R6_V2_RELATIVE_FIXTURE.json)
