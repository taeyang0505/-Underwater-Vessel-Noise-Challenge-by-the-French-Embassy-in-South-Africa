# Initial generation prompt

Use case: scientific-educational.
Edit/redesign the supplied reference research figure. The reference is STYLE AND LAYOUT ONLY: replace ALL of its software-agent/Gemma content with the actual underwater acoustic machine-learning pipeline below. Produce a pristine, detailed top-conference methods figure, not a generic slide or infographic. High-resolution landscape, approximately 3:2, maximum available quality. White background, Times-like serif typography, strong black panel headings, precise thin black arrows, translucent pale sage group backgrounds, muted lavender frozen-foundation groups, pale blue neural layers, orange validation cards, detailed black-outlined stacked data sheets, tiny spectrogram grids, neural-layer bars and attention matrices. Match the reference's density, visual sophistication, academic typography, hatching, neat connections, restrained palette and generous local spacing. Crisp legible text. No fake performance curves, no decorative charts. All numerical results below are measured custom-synthetic OOF, NOT leaderboard scores.

Main header exact:
"R2 • Completed underwater-noise pipeline"
Small subheader:
"7 model families / 9 candidates • 5-fold nested source CV • seed 42"
Clock badge upper right: "RTX 4090 • 71 m 10 s"

Two major stacked panels separated by a thin horizontal dashed line. Panel (a) fills about 70% of the height, panel (b) 30%. Design coherent left-to-right causal flow with labeled arrows and no crossings over text. Every major arrow must have a correct source and destination. Retain technical richness rather than replacing steps with generic rectangles.

(a) heading: "Data construction, fitting and evaluation"

LEFT COLUMN: stacked source cards inside a light-sage enclosure titled "Measured sources".
Card 1: "South African acoustics"; "1,715,779 unlabeled rows"; "10 sites"; tiny 15×25 spectrogram and t / dur / TOL / QC table.
Card 2: "VTUAD v1.3.2"; "218 reference clips".
Card 3: "HearMyShip V1"; "945 reference clips".
Card 4: "NOAA SanctSound"; "8 reference clips".
Bracket under reference cards: "1,171 clips / 26 components"; tiny caution: "Native long references: 5 components".
Labels are real source measurements, NOT pseudo ground truth. Do not mark these as certified natural-vessel labels.

SECOND COLUMN: arrow to a wedge "QC + linear band energy", then an explicit split block "Split BEFORE synthesis".
Show 5 small fold cards and separate blue "TRAIN ≈53k / fold", orange "INNER 3k / fold", green "OUTER 3k / fold".
Below: "SA: 900 s support + 24 h purge"; "References: hull / day / overlap / SHA".
TRAIN source backgrounds exclude boundary candidates; INNER/OUTER keep flags for diagnostics.
A tiny external-audio conversion inset is permitted under sources: "2 s Hann PSD → 25 bands → linear minute mean".
Crucial: only TRAIN feeds train augmentation, prior fitting and gradient updates. INNER selects checkpoints/weights. OUTER is only evaluated AFTER weights are frozen.

THIRD COLUMN: a detailed rectangular "TRAIN controlled additions" module with two miniature grids B and R, an observed-mask O grid, a smooth frequency-tilt curve icon applied to R BEFORE solving its scale, and the equations below.
Labels: "Clean background B"; "Reference R"; "Smooth EQ ±4 dB"; "Background blend p=0.25"; "Short-support share cap"; "Minute SNR ≤20 dB"; "Native / dispersed multi-reference".
Equations, exactly and legibly:
"E = B + α R"
"q = α R / E"
"w = O E / Σ(O E)"
"y = Σ(w q)"
Caption: "Known ADDED energy share; natural background vessels remain part of B".
Small box underneath: "INNER / OUTER: fixed NPZ"; "Original evaluation recipes retained".
Do NOT imply the new TRAIN caps/EQ/blending were also applied to the frozen evaluation datasets.

CENTER-RIGHT MODEL COLUMN: two clearly separate enclosures, not mixed.
Upper lavender enclosure: "Frozen foundation inference — no fine-tuning".
Show two separate stacked submodules with little padlocks and attention blocks:
"TabICLv2 ≈28.5M"; "100 features • FP32"; "TRAIN context 16,000 • 4 members".
"Kumo Large ≈215.7M"; "100 features • FP32"; "TRAIN context 4,096 • 4 members".
Show train-only context as a little X,y bank connected into each model. NEVER show Adam updates to these two.
Lower sage enclosure: "Task models — fitted independently in every fold".
Neural branches drawn in parallel with recognizable architecture glyphs:
"TabM"; "1,258 features"; "direct share + presence".
"BG residual CNN"; "8 × 15 × 25"; "TRAIN-only 32-cluster background prior"; little training-only prior cylinder, residual conv blocks, output q-hat heatmap.
"FDY-inspired CRNN"; "6 × 15 × 25"; conv/frequency attention/BiGRU glyphs; output q-hat heatmap.
"iTransformer-inspired"; "6 × 15 × 25"; band tokens / transformer blocks; output q-hat heatmap.
The 3 mask branches feed ONE clearly labeled physical pooling formula:
"ŷ = Σ(w q̂)".
The TabM branch directly predicts share and bypasses cell pooling.
Beside or underneath neural branches show a compact tree enclosure:
"LightGBM ×3"; "base / physics / train-only retrieval"; "1,258 / 1,303 / 1,311 features"; tree icons and direct share arrows.
Training-only retrieval prototypes and CNN background prior are separate modules, not the same bank.
Bottom of task enclosure exact labels:
"Neural training: Adam • max 100 epochs • patience 7 • dropout 0.1"
"Share MSE + 0.03 presence BCE"
"Mask models: +0.03 energy-weighted cell MSE"
"CNN: +0.001 residual penalty"
Tiny note: "20 neural fits • 270 actual epochs".
LightGBM is NOT trained using Adam/epochs.

RIGHTMOST COLUMN: "INNER model selection" receives predictions from all 9 candidates and chooses nonnegative weights with sum=1. Clearly label "Checkpoint selection + ensemble fit". Then padlock arrow labeled "FREEZE policy"; then "OUTER evaluation".
Show a compact exact text results card with NO fabricated bars:
"Completed OOF: 15,000 windows"
"R2 ensemble RMSE 0.245585"
"CNN alone RMSE 0.238545"
"Custom synthetic OOF ≠ public score"
Tiny metrics note: "All / boundary-excluded / source-stratified RMSE".
Keep OUTER a measurement endpoint, with no arrow going back into training or ensemble fitting.

(b) heading: "Frozen inference and current status"

Left stack: "Test.csv"; "1,500 windows"; "15 × 25 bands".
Arrow through wedge "Fixed per-window transforms", then a large clean box "Frozen R2 candidates".
Inside, 4 small neural cards, 3 tree cards as a grouped "LightGBM ×3", 2 lavender padlocked foundation cards. Indicate "Saved checkpoints + TRAIN context/prior"; no fitting, no new calibration, no query-bank adaptation.
The CNN/FDY/iTransformer mini branches feed energy-weighted pooling; TabM/trees/foundations direct outputs all join "Frozen INNER weights". Then "Clip [0,1]" and "Submission-ready CSV schema" displayed as a small clean table with exact column headers:
"ID | share_added_vessel | score_added_vessel".
Under table: "R2 has NOT been officially submitted".
Show a separate orange dashed inset NOT in the core inference arrow, labeled:
"After R2: R3 ensemble experiment"
"INNER-only shrinkage λ=0.05"
"OOF RMSE 0.240328"
"Still above CNN; not promoted"
Make it unambiguous R3 is a separate completed policy experiment, not a seventh neural network, not a fine-tuning stage, not adopted as the R2 production policy.

Very bottom caption exact:
"Figure. Actual completed R2 pipeline; subsequent dataset fixes are not yet implemented."
Second caption:
"TRAIN-only fitting → INNER selection → frozen policy → OUTER measurement. No Test fitting."

Avoid all software-agent terms from the reference. No Gemma, RL, LoRA, RAG, POMDP, contrastive or Nesterov nodes: they were not executed in R2. No claim of SOTA or leaderboard success. Do not invent scalar-head neural architectures where mask pooling is specified. Keep all 9 model candidates explicit. All scientific labels in English, accurately typeset, no gibberish.

# Targeted correction prompt

Edit this exact generated R2 scientific pipeline figure with MINIMAL targeted changes. Preserve panel (a), all numbers, styles, models, layout, colors, formulas and text exactly. Do not redraw the whole figure. Only correct panel (b) and the bottom caption:
1. Change panel (b) title to "(b) Frozen inference specification — Test execution pending". It is an implementation/inference design, not an executed Test result.
2. Inside EACH of the three mask-model cards in panel (b) (BG residual CNN, FDY-inspired CRNN, iTransformer-inspired), show that its q-hat heatmap is converted to a SCALAR before joining the ensemble. Add a small precise pooling arrow and label "ŷ = Σ(w q̂)" beneath its heatmap, followed by a tiny scalar-output marker. Route the outgoing ensemble arrow from these scalar markers. It must not appear that a raw 15×25 mask is directly blended with scalar predictions. TabM, all 3 LightGBMs, TabICLv2 and Kumo are already scalar regressors and bypass this pooling. Keep all 9 candidates visible. You may slightly redistribute whitespace within panel (b) only to make these tiny equations legible.
3. Bottom caption should read:
"Figure. Completed R2 fitting and evaluation; frozen inference design. Schematic grids; measured OOF numbers."
Second line:
"TRAIN-only fitting → INNER selection → frozen policy → OUTER measurement. No Test fitting."
4. Preserve the orange separate R3 experiment box, its λ=0.05, RMSE 0.240328, not promoted, and the R2 CSV status "R2 has NOT been officially submitted".
No other content changes. Keep sharp Times-like academic serif text, exact original top-conference style. White background, high resolution.

