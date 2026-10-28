Create a very detailed, polished, publication-quality scientific architecture figure for a top-tier machine-learning paper. The visual reference is the user's white-background academic pipeline diagram: Times-style serif typography, bold black section titles, thin precise black orthogonal arrows, small stacked document/data sheets, realistic small blue-to-yellow spectrogram grids, subtly tinted sage green grouping panels, muted lavender model panels, blue/orange hatched checkpoint stacks, fine dashed grouping borders. Keep the page almost entirely white, no glossy 3D, no dark theme, no marketing banner, no cartoon, no oversized icons. Many meaningful subcomponents, disciplined alignment and ample gutters, no crossing arrows or overlapping labels. Landscape, very high resolution with crisp legible dense text, approximately 3200 by 2100 pixels. All labels in English, use precise mathematical typography. Reproduce the exact counts and distinctions below. Include only schematic heatmaps and structural graphics, never fabricated performance curves or measured-looking bars. Each figure is a standalone full page, not combined with other figures. The small footer must say 'R9 v1 | 26 Oct 2026 | Verified source split; production GPU results pending'.
THIS IS FIGURE 1 OF 3. Title at top left: '(a) Data Preprocessing'. Subtitle: 'Measured sources → quality control → background-source folds → minute-first controlled additions'. Three horizontal lanes, connected left to right and then down by clean arrows.

UPPER LANE, SOURCE ACQUISITION AND COMMON GRID:
At far left stack two major source cards.
Card1 pale sage title 'South African acoustics · Zindi release'. Text '10 sites · 1,715,779 rows', '74 columns = 62 TOL + 12 metadata/QC', 'Archive 94.917 MB; no waveform or vessel-share labels'. Show table columns site | recorder | t | dur | TOL_f, and a 25-band spectrogram icon.
Arrow to 'SA row QC + 15-minute binning': select 25 common bands; finite values; exclude −156.5 sentinel; 55 ≤ dur ≤ 65 s; timestamp seconds ≤ 5; deterministic duplicate-minute selection. Output '[133,641, 15, 25] float32', '200.46 MB numeric payload', 'Missing minutes stay NaN; ≥3 observed minutes'. Preserve all raw rows separately with a little archive icon.
Card2 pale sage title 'VTUAD rebuilt v1.3.2 · ONC-derived'. Text '525 WAV = 509 vessel-archive + 16 ambient controls', '32 kHz mono · 42.0467 h header total', '4.638 GB compressed / 9.688 GB WAV'. Show waveform sheet icon. Source line 'Zenodo 23037895 · CC BY 4.0'. Arrow through signal stages 'channel 0 → 2-s Hann PSD → integrate 25 bands → 30 powers averaged in LINEAR space'. Result 'Native minute matrices R[m,25]' with '509 references · 1,978 complete minutes = 32.9667 h' and '33.1633 h reference header total'. Show 30 tiny segments grouped into a single minute. No resampling, waveform repetition, time stretching or forced fade.
Separate small ambient branch '16 ambient clips · 8.8833 h → separate nuisance/control analysis', explicitly 'Not automatic y=0 labels'.
A thin provenance strip under source cards: 'Common centers: 31.6–7,943.3 Hz; 25 fixed bands (upper integration edge 8,912.5 Hz). Preserve header, source ID, original role and SHA-256.' Small side box 'Historical only: HearMyShip 1,145 unique WAV / 6.383 h; NOAA primary 12 × 15 min / 3 h. Not active R9 references.' Source tiny footnotes Zindi competition URL and doi:10.5281/zenodo.23037895.

MIDDLE LANE, SPLIT BEFORE SYNTHESIS:
Large pale sage box 'Background-source 5-fold CV'. Show five colored buckets F0 F1 F2 F3 F4, with one orange held-out bucket and four blue TRAIN buckets. Input 'Existing allowed pool: 100,965 windows · 258 sensor-week groups'. Gate 'Mask actual boundary-flagged MINUTES, not whole windows', '6,254 minutes masked; 100,964 windows survive QC'. Then '≥24 h gap between 900-s supports, same sensor'. Verified callout 'Cross-role source overlap = 0; time-gap violations = 0'.
Under buckets print a compact exact count table:
'Fold | TRAIN | selection | fit | gate | report'
'0 | 78,869 | 9,417 | 3,417 | 3,511 | 3,584'
'1 | 78,069 | 9,824 | 3,104 | 3,859 | 3,237'
'2 | 78,362 | 10,271 | 3,432 | 3,274 | 2,929'
'3 | 78,753 | 10,675 | 3,971 | 3,195 | 2,427'
'4 | 78,124 | 10,183 | 3,487 | 3,462 | 2,889'
Label these 'UNIQUE SOURCE WINDOWS, not synthetic example counts'.
Beside this, VT reference bank cylinder connects to TRAIN and validation generation with label 'All 509 references shared across background folds'. Box in restrained orange 'Background-source CV; NOT reference-independent validation'.

LOWER LANE, PHYSICAL SYNTHESIS AND DATA CONTRACT:
Panel1 'Deterministic epoch recipe' with native-minute grid R and background grid B and mask O. Family branches 'native inside 50%', 'short 1–3 min 25%', 'edge-straddling 25%'; all '100% minute-first'. Text '20% explicit no-source; geometry independent of NaN mask', 'Smooth EQ ≤ ±6 dB; full-reference peak SNR ∈ [−10,+10] dB'.
Panel2 formulas in large readable centered text:
'α = 10^(SNR/10) × median_t Σ_f B / max_native_t Σ_f R_EQ'
'A = α R_placed ; E = B + A'
'q = A/E ; y = Σ(O A) / Σ(O E)'
'Known added-recording fraction; natural background may already contain vessels'.
Then two output arrows: blue 'Q(db) = round(10 log10 E, 1)' → 'relative8 input [8,15,25]' and orange 'raw q / y / presence' → 'supervision only'. Small line 'w ∝ O·10^(Q(db)/10); raw labels unchanged; rounding mismatch bounded'.
Final stacked dataset cards: 'Dynamic TRAIN: every TRAIN background once per epoch; all 509 references cycled; site-balanced loss weights'. Second: 'Fixed validation per fold: selection 1,000 / ensemble-fit 500 / gate 500 / report 1,000'. Line '5-fold totals: 392,177 TRAIN draws per epoch + 15,000 fixed validation examples. Reused sources ≠ independent recordings.' Distinguish source-manifest PASS from dataset materialization and GPU execution, no checkmarks claiming completed full training.
Bottom caption: 'Figure 1. Audited source lineage and R9 preprocessing contract. Sources are split before any controlled mixture is generated.'
