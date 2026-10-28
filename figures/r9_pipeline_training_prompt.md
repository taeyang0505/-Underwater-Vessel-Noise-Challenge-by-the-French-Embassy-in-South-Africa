Create a very detailed, polished, publication-quality scientific architecture figure for a top-tier machine-learning paper. The visual reference is the user's white-background academic pipeline diagram: Times-style serif typography, bold black section titles, thin precise black orthogonal arrows, small stacked document/data sheets, realistic small blue-to-yellow spectrogram grids, subtly tinted sage green grouping panels, muted lavender model panels, blue/orange hatched checkpoint stacks, fine dashed grouping borders. Keep the page almost entirely white, no glossy 3D, no dark theme, no marketing banner, no cartoon, no oversized icons. Many meaningful subcomponents, disciplined alignment and ample gutters, no crossing arrows or overlapping labels. Landscape, very high resolution with crisp legible dense text, approximately 3200 by 2100 pixels. All labels in English, use precise mathematical typography. Reproduce the exact counts and distinctions below. Include only schematic heatmaps and structural graphics, never fabricated performance curves or measured-looking bars. Each figure is a standalone full page, not combined with other figures. The small footer must say 'R9 v1 | 26 Oct 2026 | Verified source split; production GPU results pending'.
THIS IS FIGURE 2 OF 3. Title '(b) Training'. Subtitle 'R9 full background-source CV · four mask networks + Physics LightGBM'. Main diagram top 70% left-to-right; bottom 30% validation/governance lane. All text exact and highly legible. Models are candidates, no measured R9 RMSE exists.

LEFT DATA AND FEATURES:
Two small dataset cards 'TRAIN: 78,069–78,869 sources per fold / deterministic on-the-fly mixtures each epoch' and 'selection: 1,000 fixed examples / checkpoint choice only'. Show 15×25 mixture heatmap and binary NaN mask, then relative feature tokens.
Large panel 'Window-relative8, no fitted background prior', list eight channels with small band/temporal illustrations:
'1 (Q − band median_t)/10'
'2 (Q − band q20_t)/15'
'3 adjacent observed Δt Q /10'
'4 positive mid/high Δ × low-band co-rise'
'5 (band median_t − window median)/20'
'6 observed mask O'
'7 (Q − band q05_t)/15'
'8 sqrt(w), w ∝ O·10^(Q/10)'
'Q = round(db,1); NaN features = 0; NaN weights = 0.'
Small struck-out or absent-input box labeled 'No absolute db/100; no q0 or Bhat/E; no source IDs as predictors'. Avoid portraying every channel as fully receiver-invariant; profile and weights preserve spectral shape.

CENTRAL MODELS, FOUR PARALLEL NETWORK ROWS:
A large pale green grouping 'Fresh-weight task-adapted mask models'.
Row1 'Wide BG Mask CNN · 1,487,873 parameters': conv stem, five residual blocks, frequency-global context, mask head. Show broad stacked blue convolutional sheets and a small high-to-low band context bridge.
Row2 'TFGrid–FDY–Conformer · 2,562,085 parameters': FDY frequency kernels → time/frequency grid blocks → Conformer → mask head; draw frequencywise/timewise attention arrows.
Row3 'FDY-CRNN variant · 129,901 parameters': frequency-dependent conv → BiGRU → mask head.
Row4 'iTransformer variant · 295,599 parameters': 25 band tokens → cross-band attention → mask head.
A shared output panel to their right: 'q̂[t,f] = sigmoid(mask logits)' blue15×25continuousratio grid, then multiplication node with SAME input w, then large formula 'ŷ = Σ_t,f w[t,f] q̂[t,f]'. Explicit 'Continuous energy fractions, not binary segmentation'. Auxiliary diagnostic 's = 0.5 ŷ + 0.5 mid-band mask contrast' with 'fixed derived score; no free presence head'.
Below neural rows a separate sage tree branch: 'Physics LightGBM · CPU', feature illustrations '3,827 = 3,000 relative-grid + 800 temporal summaries + 27 physics'. Physics tiny examples 'low/mid/high/safe bump; cross-band correlations; defined flags; valid-minute counts'. Tree silhouette, 'share regressor + diagnostic classifier', '≤1,500 trees · early stopping 40 · train ≤100,000 rows'. No PCA or retrieval prototype shown because current R9 features have neither.

OPTIMIZATION AND CHECKPOINTS:
At right blue/orange hatched checkpoint stack with padlocks. Loss box:
'L = MSE(ŷ,y) + 0.25 Σ w(q̂−q)^2 + 0.05 BCE(s,z)'
'TRAIN site-balanced loss weights; validation metrics unweighted + site-macro'.
Settings box 'Adam · lr 10^−3 · weight decay 3×10^−4'
'100 epochs exactly · no NN early stopping'
'dropout 0.1 · gradient clip 1.0'
'CUDA BF16 · physical pooling FP32'
'Save best-window, best-macro, last, every 10 epochs'
'Selection RMSE controls scheduler/checkpoint; no global-minimum guarantee'.
No fabricated convergence curves. Instead draw a tiny blank log table with column epoch/trainRMSE/selectionRMSE and values 'pending'.

BOTTOM VALIDATION SAFETY LANE:
Four distinct cards: 'selection / 1,000 / checkpoint only' → lock icon; 'ensemble-fit / 500 / choose finite candidate' → 'ensemble-gate / 500 / one candidate vs frozen fallback' → 'report / 1,000 / evaluate only after all policies frozen'. Put arrows from frozen NN/tree predictions to fit/gate. Candidate box 'All single models; equal-all; equal-neural', 'Fallback = best single neural on ensemble-fit'. Gate 'Adopt only if fit AND gate RMSE improve ≥0.0001; otherwise fallback'. No recursive tuning with report.
Small five-fold bracket 'Folds 0…4 · 4 neural + 1 tree per fold · up to 25 fitted members'. Very visible small status badge 'Model/optimizer CPU connection checks PASS; actual production GPU fitting PENDING'.
Footer caveat 'All509 VTUAD references are shared: report estimates background-source generalization, not unseen-reference performance. No Test fitting.'.
Bottom caption 'Figure 2. R9 training and validation roles. Exact model sizes describe the adapted implementation, not full reproduction of the cited research architectures.'
