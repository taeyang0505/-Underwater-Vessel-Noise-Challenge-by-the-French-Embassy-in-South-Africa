# R5 architecture image generation record

## Initial prompt

Create a NEW, meticulous publication-quality machine-learning architecture figure. Use the attached image ONLY as the visual style reference: elegant black Times-like serif typography, white paper, thin crisp black arrows, pale sage grouped panels, subtle lavender pretrained panels, blue/orange hatched weight stacks, detailed tiny matrix and token drawings, disciplined aligned scientific layout. Do not reuse the coding-agent content. Landscape approximately 3:2. This is a top-tier conference paper figure, not a slide or infographic. Dense but readable, no cramped overlaps, no decorative gradients or cartoon icons. English text only. Preserve meaningful details and exact model specifications below.

Title-less two major panels separated by thin dashed horizontal rule. Upper about 70% height: "(a) Source-separated data construction and training". Lower: "(b) Frozen inference". Use a coherent left-to-right pipeline with scientifically correct supervision paths. All heatmaps are schematics; do not fabricate measured training curves, RMSE or leaderboard scores.

UPPER LEFT: stacked source cards in pale green:
"South African acoustic summaries" / "1,715,779 rows; unlabeled" with a time-frequency heatmap.
"VTUAD v1.3.2" / "119 clips • 22 selected groups" with waveform and hull/day source cards.
Arrow to "Split BEFORE synthesis" card: "Nested 5-fold source CV" and "24 h SA support purge". Distinct blue TRAIN and orange held-out source stacks, explicit no overlap.
Arrow to "Matched R4 recipe" card containing schematics of B, reference R, observation O; small smooth raised-cosine passage envelope and smooth EQ curve; text "bounded monotone power warp • smooth EQ ±4 dB"; formulas "E = B + αR", "q = αR / E", "y = Σ(O αR) / Σ(O E)". Role counts below: "TRAIN 267,837 • INNER 15,000 • OUTER 15,000" with footnote "5-fold role totals". All three roles follow the same recipe.
Next small "R5 observation candidates" card: "raw / 0.1 dB feature rounding" and "TRAIN-only endpoint censoring". Tiny binary-mask matrix: "After censoring: recompute O, w, y". Under it: "Rounding features preserves raw physical weights and exact labels".
Then "Shared 8-channel input" card depicting [8 × 15 × 25] layered grids: labels "relative dB, absolute dB, √w_feature, O", "band excess, Δt dB, q0, log1p(Bhat/E)". Train-fitted prior cylinder feeds the last two channels. A strict boundary: no q target or recipe ID arrow into input features.

UPPER CENTRAL / RIGHT: five model branches, grouped as trainable masks and complementary tree. Use pale lavender for the two pretrained branches, sage for M-scale.
1 "TimeMoE-200M + Stem / LoRA" / "453.9M total • 0.776M trainable". Detail: "15 time tokens • D768 • 12 layers"; frozen transformer blocks with small blue hatched LoRA modules, "public checkpoint; frozen experts".
2 "Dasheng-1.2B + Grid Stem / LoRA" / "1.136B branch • 2.143M trainable". Detail: "375 grid tokens • D1536 • 40 layers"; frozen blocks and LoRA; small text "1.2B nominal model".
3 "TFGrid + FDY + Conformer" / "2.562M; full training". Detail "5 grid blocks • frequency-dynamic kernels • 1 Conformer". Show alternating frequency/time modules and attention.
4 "Wide BG Mask CNN" / "1.488M; full training". Detail "width 128 • 5 residual blocks"; prior q0 as input only, direct sigmoid mask head, no forced output anchor.
5 complementary box "Physics / Excess Retrieval LightGBM" / "1,324 features". Train-only prototype/PCA cylinder, small tree symbols. Labels "high-band bump • low/high correlation", "mid-band peak excess • TRAIN PCA residual".
A narrow comparison box "Controls: FDY-CRNN / iTransformer".
The four neural arrows terminate at "Cell mask q̂(t,f) ∈ [0,1]" schematic grid, then "Physical energy pooling" with "w = OE / Σ(OE)" and "ŷ = Σ(w q̂)". Tree outputs ŷ directly. Supervision arrows from known q and y go ONLY to the loss box.
Loss/training box: "Adam • max 100 epochs • early stopping 7" / "dropout 0.1" / "L = share MSE + 0.25 Σw(q̂−q)² + 0.05 presence BCE". "Selection: INNER RMSE".
Neural/tree outputs feed "INNER-only safety gate": "large-backbone admission", "source-group LOO weight check", "share average (default)", "logit-mask pooling (optional)". Show rejected candidates can receive zero weight, not all branches forced in. Frozen weights stack after gate.
Far right: "OUTER assessment" / "Only after policy freeze" / "overall / boundary-excluded / site RMSE" / "GPU OOF pending". No plotted numerical scores.

LOWER PANEL: "Test.csv" / "1,500 windows • 15 × 25" schematic matrix arrow "Fixed per-window transforms" with frozen prior/PCA mini cylinders and mask O. Arrow to "Selected frozen model branches" with lock hatches, compact miniature branches named TimeMoE, Dasheng, TFGrid-FDY-Conformer, BG CNN, LightGBM. Arrows to neural mask pooling formula and tree direct share; then "Frozen INNER-selected ensemble" / "fixed linear or approved logit rule". Then "Clip to [0,1]" and "Submission.csv" stacked spreadsheet with column headers "ID", "share_added_vessel", "score_added_vessel". Small note "score: controlled-addition diagnostic".
Footer caption centered: "Figure. R5 candidate architecture with matched source-separated data and INNER-only selection." Smaller line: "CPU model connections verified; GPU CV and performance are pending. Schematic matrices; no measured performance shown." Small explicit protocol note "No Test fitting or cohort adaptation." Prioritize beautiful precision, correct arrows, exact numbers, and readable grouping.

## Scientific corrections

Precisely edit the attached R5 scientific architecture figure. Keep the beautiful existing design, dimensions, detail, serif typography, white background, sage/lavender grouped panels and exact model/count labels. Do not simplify or remove panels. Make only these scientific corrections:
1. In the upper "Matched R4 recipe" card, the right-hand heatmap is the MIXTURE, not a mask. Replace its heading "Observation O" with "Mixture E". Keep the formula E=B+αR. Keep "Observation O" only on the binary-mask example in the separate observation-candidates card. The heatmap arrow should represent B + αR -> E; place a small α before the R heatmap if possible.
2. The upper "Known targets (q,y)" box MUST NOT feed the predicted Cell mask box. REMOVE that downward target-to-mask arrow completely. The known-target box should have only a thin dashed supervision arrow to the LOSS AND TRAINING box. Put a small "supervision only" label beside that dashed arrow. Prediction arrows must originate only from the model branches.
3. The cylinder below "Shared 8-channel input" is a 32-centre aggregate TRAIN background prior. Replace "Train-fitted prior (μ,σ) + PCA" with "TRAIN-only background prior" and "(32 spectral centres)". It feeds q0 and log1p(Bhat/E) channels ONLY. PCA is NOT an 8-channel input. Retain the separate TRAIN-only prototype/PCA cylinder in the LightGBM branch.
4. In lower inference panel, replace "Frozen prior (μ,σ)" with "Frozen background prior" and "32 centres". The separate Frozen PCA cylinder is for the LightGBM feature branch, not neural channels: show its outgoing arrow pointing to the LightGBM branch, or put text "PCA → LightGBM" immediately below the separate cylinder with an appropriate thin arrow.
5. In upper TimeMoE frozen blocks, clarify the parenthetical "200M" as "200M active"; total remains 453.9M. All other precise model sizes stay unchanged: 0.776M trainable, Dasheng 1.136B branch/2.143M trainable, TFGrid-FDY-Conformer 2.562M, Wide CNN1.488M, LightGBM1324features.
6. In inference frozen-ensemble box, the displayed sum α_k yhat_k is the LINEAR DEFAULT. Add the small label "linear default" directly above that formula. Keep text "approved logit rule" as optional, with a small mathematical note "optional: q_logit = σ(Σ α logit(q))" below the formula if space allows. It must not imply the logit rule is equivalent to the scalar linear default.
Preserve all correct data counts, same-source-split recipes, model connection statuses, GPU pending status, no measured performance, and the caption. Keep all text readable without overlap.

## Final targeted corrections

Keep this entire scientific R5 diagram exactly the same, with all panels, numbers, formulas and detailed design intact. Only two tiny final corrections:
(1) In the bottom "(b) Frozen inference" panel, inside "Fixed per-window transforms", REMOVE the vertical arrow connecting the "Frozen background prior (32 centres)" cylinder down into the "Frozen PCA (transform)" cylinder. Those are two independently TRAIN-fitted artifacts; the prior is NOT an input to PCA. Keep the PCA's small outgoing arrow labeled "PCA → LightGBM". No other arrow removals.
(2) Add a small readable scientific note above or immediately beside the upper five-model branch group: "All neural branches receive the same 8-channel grid". It must be legible and not overlap any existing labels.
Do not change E, O, targets-supervision-only, parameter counts, data counts, status, or image layout. Preserve top-tier paper figure aesthetics.

