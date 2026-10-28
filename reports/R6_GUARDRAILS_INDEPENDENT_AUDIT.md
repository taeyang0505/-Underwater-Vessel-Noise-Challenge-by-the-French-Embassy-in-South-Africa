# R6 guardrails independent integration audit

TRAIN/INNER arrays and source metadata were audited. No locked arrays/recipes, model results, old OUTER values, or reference numeric slices were opened. Existing embedded locked descriptors were inadvertently displayed once and were not used in any decision.

**PASS:** TRAIN 8,192 R4/R6 background IDs, 1,638 no-source positions and 819 hard-negative positions exactly match the deterministic source schedule. Hard-negative share/presence/cell targets are exactly zero, with no reference addition.

**PASS:** QC preserves the original NaN mask and all actual flagged minute masks in both generator datasets; every sampled row retains at least three observed minutes.

**PASS:** All TRAIN/INNER recipe reference IDs belong to the matching source role. Original 119 clips / 22 groups become 94 / 17 because exactly the original OUTER 25 / 5 are excluded.

**Limitations:** Hard-negative 819 draws represent 428 unique windows. INNER R4/R6 share primary background order but their 205 explicit no-source rows are placed independently. Own-law INNER RMSE is not a controlled common-target comparison.

Full counts, site repetitions, bindings and checks are in [JSON audit](R6_GUARDRAILS_INDEPENDENT_AUDIT.json).
