#!/usr/bin/env python3
"""Deep descriptive EDA for the Zindi Underwater Vessel Noise Challenge.

Reads data/raw/Test.csv and data/extracted/* and writes figures, tables and
summary.json into eda_deep/. Test.csv is only described: nothing is fitted,
calibrated or pseudo-labelled on test inputs. The kNN domain probe is fitted on
training windows only and merely applied to test windows.

Run from anywhere:  python3 -I eda_deep/deep_eda.py
"""
from pathlib import Path
import json
import os
import time
import warnings

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/uvn-mpl')
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from scipy.stats import ks_2samp

warnings.filterwarnings('ignore', category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
RAW, EXT = ROOT / 'data/raw', ROOT / 'data/extracted'
OUT = ROOT / 'eda_deep'
FIG, TAB = OUT / 'figures', OUT / 'tables'
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
SUMMARY = {}
T0 = time.time()
RNG = np.random.default_rng(0)

# Reference data-viz palette (light mode).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
TRAIN_C, TEST_C = SERIES[0], SERIES[1]
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
BLUES = LinearSegmentedColormap.from_list(
    'blues', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
DIVERGE = LinearSegmentedColormap.from_list(
    'div', ['#184f95', '#6da7ec', '#f0efec', '#ef9a99', '#b52f2e'])
BLUES.set_bad('#f0efec')
DIVERGE.set_bad('#ffffff')

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 150, 'font.size': 8.5,
    'axes.facecolor': SURF, 'figure.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 9.5, 'axes.titleweight': 'bold',
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.6, 'legend.frameon': False, 'text.color': INK,
    'legend.fontsize': 7.5,
})


def log(msg):
    print(f'[{time.time() - T0:6.1f}s] {msg}', flush=True)


def save(fig, name):
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight')
    plt.close(fig)


def table(df, name):
    df.to_csv(TAB / f'{name}.csv', index=False)
    return df


def db2lin(x):
    return np.power(10.0, np.asarray(x, dtype=np.float64) / 10.0)


def lin2db(x):
    return 10.0 * np.log10(x)


def freq_axis(ax):
    ax.set_xscale('log')
    ticks = [31.6, 100, 316, 1000, 3162, 7943]
    ax.set_xticks(ticks)
    ax.set_xticklabels(['31.6', '100', '316', '1k', '3.2k', '7.9k'])
    ax.minorticks_off()
    ax.set_xlabel('band centre (Hz)')


def q(x, qs=(5, 25, 50, 75, 95)):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return {f'p{k:02d}': float(np.percentile(x, k)) for k in qs} if x.size else {}


def kv(series):
    return {str(k): round(float(v), 4) for k, v in series.items()}


def rowcorr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a = np.where(m, a, np.nan)
    b = np.where(m, b, np.nan)
    am = a - np.nanmean(a, 1, keepdims=True)
    bm = b - np.nanmean(b, 1, keepdims=True)
    return np.nansum(am * bm, 1) / np.sqrt(np.nansum(am ** 2, 1) * np.nansum(bm ** 2, 1))


# --------------------------------------------------------------------------- load
log('loading test')
test = pd.read_csv(RAW / 'Test.csv', dtype={'ID': str})
FEAT = test.columns[1:]
BANDS = [c.split('_', 1)[1] for c in FEAT[:25]]
FREQ = np.array([float(b[4:]) for b in BANDS])
XT = test[FEAT].to_numpy(np.float32).reshape(-1, 15, 25)
LOW = FREQ <= 520            # 31.6-501 Hz: where large-ship energy concentrates
HIGH = FREQ >= 3000          # 3.2-7.9 kHz
EVT_LOW = (FREQ >= 50) & (FREQ <= 520)
EVT_REF = FREQ >= 1580

log('loading training parquet')
stations = pd.read_csv(EXT / 'stations.csv')
SITE_NAME = dict(zip(stations.site, stations.site_name))
frames = [pq.read_table(f, columns=['site', 'recorder', 'sample_rate_hz', 't', 'dur'] + BANDS).to_pandas()
          for f in sorted((EXT / 'acoustics').glob('*.parquet'))]
tr = pd.concat(frames, ignore_index=True)
del frames
tr['site_id'] = tr.site.str.split().str[0].astype(int)
tr = tr.sort_values(['site_id', 'recorder', 't'], kind='stable').reset_index(drop=True)
LABEL = {s: f'{int(s.split()[0]):02d} {SITE_NAME[s]}' for s in tr.site.unique()}
tr['label'] = tr.site.map(LABEL)
SITES = [LABEL[s] for s in sorted(tr.site.unique(), key=lambda s: int(s.split()[0]))]


def rec_short(rec):
    base, sr = rec.rsplit('.1.', 1)
    return f'{base[:20]} ({int(sr) // 1000}k)'


V = tr[BANDS].to_numpy(np.float32)
TSEC = tr.t.values.astype('datetime64[s]').astype(np.int64)
SENT = (V < 0).any(1)
ONGRID = TSEC % 60 == 0
FULL = tr.dur.values >= 55
BB = lin2db(db2lin(V).sum(1)).astype(np.float32)   # broadband 25-band level

# Duty-cycled recorders can log a start-up (or shut-down) transient on the edge minute of each
# contiguous run. Measure the median broadband excess of edge minutes over their run median and
# mask edges whose excess exceeds 3 dB for that recorder.
EDGE = np.zeros(len(tr), bool)
edge_rows = []
for rec, g in tr.groupby('recorder', sort=True):
    idx = g.index.values
    idx = idx[ONGRID[idx] & ~SENT[idx]]
    m = TSEC[idx] // 60
    brk = np.r_[True, np.diff(m) != 1]
    rid = np.cumsum(brk)
    first = brk
    last = np.r_[brk[1:], True]
    runlen = np.bincount(rid)[rid]
    excess = BB[idx] - pd.Series(BB[idx]).groupby(rid).transform('median').values
    multi = runlen >= 3
    e_first = float(np.median(excess[first & multi])) if (first & multi).any() else 0.0
    e_last = float(np.median(excess[last & multi])) if (last & multi).any() else 0.0
    if e_first > 3:
        EDGE[idx[first]] = True
    if e_last > 3:
        EDGE[idx[last]] = True
    edge_rows.append({'recorder': rec, 'runs': int(rid.max()), 'median_first_minute_excess_db': round(e_first, 2),
                      'median_last_minute_excess_db': round(e_last, 2),
                      'masked_rows': int(EDGE[idx].sum())})
table(pd.DataFrame(edge_rows), 'run_edge_artifacts')
GOOD = ~SENT & ONGRID & FULL & ~EDGE
SUMMARY['train_edge_masked_rows'] = int(EDGE.sum())
SUMMARY['train_rows'] = int(len(tr))
SUMMARY['train_good_rows'] = int(GOOD.sum())
SUMMARY['train_offgrid_rows'] = int((~ONGRID).sum())
SUMMARY['train_sentinel_rows'] = int(SENT.sum())

# ------------------------------------------------- recorder duty cycle + aligned windows
log('duty cycles and clock-aligned 15-minute windows')
rec_rows, win_list, win_meta, raw_pats = [], [], [], []
for (sid, rec), g in tr.groupby(['site_id', 'recorder'], sort=True):
    idx = g.index.values
    keep = idx[ONGRID[idx] & ~SENT[idx] & ~EDGE[idx]]
    m = TSEC[keep] // 60
    brk = np.r_[True, np.diff(m) != 1]
    runs = np.bincount(np.cumsum(brk))[1:]
    gaps = np.diff(m)[np.diff(m) != 1]
    # Windows aligned to quarter hours with at most 2 absent minutes.
    starts = np.unique(m // 15 * 15)
    grid = starts[:, None] + np.arange(15)
    pos = np.clip(np.searchsorted(m, grid), 0, len(m) - 1)
    hit = m[pos] == grid
    ok = (15 - hit.sum(1)) <= 2
    W = np.full((ok.sum(), 15, 25), np.nan, np.float32)
    hk, pk = hit[ok], pos[ok]
    W[hk] = V[keep[pk[hk]]]
    pats = [','.join(str(k + 1) for k in np.flatnonzero(~h)) or 'none' for h in hk]
    short_dur = np.zeros(ok.sum(), bool)
    short_dur[np.flatnonzero(hk.any(1))] = [(~FULL[keep[p[h]]]).any() for h, p in zip(hk, pk) if h.any()]
    # Missing-minute patterns from raw availability (edge artefacts not masked), the closest
    # analogue of how blank cells could arise in Test.csv.
    raw = idx[ONGRID[idx] & ~SENT[idx]]
    mr = TSEC[raw] // 60
    gr = np.unique(mr // 15 * 15)[:, None] + np.arange(15)
    hr = mr[np.clip(np.searchsorted(mr, gr), 0, len(mr) - 1)] == gr
    hr = hr[(15 - hr.sum(1)) <= 2]
    raw_pats.append(pd.DataFrame({'label': g.label.iloc[0], 'recorder': rec,
                                  'pattern': [','.join(str(k + 1) for k in np.flatnonzero(~h)) or 'none' for h in hr]}))
    win_list.append(W)
    win_meta.append(pd.DataFrame({
        'label': g.label.iloc[0], 'recorder': rec,
        'start': pd.to_datetime(starts[ok] * 60, unit='s'), 'pattern': pats,
        'n_missing': 15 - hk.sum(1), 'has_short_dur': short_dur}))
    span_min = m.max() - m.min() + 1
    rec_rows.append({
        'site': g.label.iloc[0], 'recorder': rec, 'nominal_sr_hz': int(g.sample_rate_hz.iloc[0]),
        'rows': len(g), 'first_t': g.t.min(), 'last_t': g.t.max(),
        'span_days': round(span_min / 1440, 1), 'coverage_in_span': round(len(m) / span_min, 3),
        'offgrid_rows': int((~ONGRID[idx]).sum()), 'dur_lt55_rows': int((~FULL[idx]).sum()),
        'sentinel_rows': int(SENT[idx].sum()),
        'median_run_min': float(np.median(runs)), 'max_run_min': int(runs.max()),
        'modal_run_min': int(pd.Series(runs).mode().iloc[0]),
        'modal_gap_min': int(pd.Series(gaps).mode().iloc[0]) if gaps.size else 0,
        'share_rows_in_runs_ge15': round(runs[runs >= 15].sum() / runs.sum(), 3),
        'aligned_windows_le2missing': int(ok.sum()),
    })
REC = table(pd.DataFrame(rec_rows), 'recorder_duty_cycle')
WT = np.concatenate(win_list)
MT = pd.concat(win_meta, ignore_index=True)
RAWPAT = pd.concat(raw_pats, ignore_index=True)
del win_list
SUMMARY['train_aligned_windows'] = int(len(WT))
log(f'{len(WT):,} aligned training windows')

# ------------------------------------------------------------- 01 site map + timeline
log('fig 00/01 site map / timeline')
fig, ax = plt.subplots(figsize=(11, 5.2))
st = stations.copy()
st['label'] = st.site.map(LABEL)
ax.scatter(st.longitude_candidate, st.latitude_candidate, s=14 + 110 * np.sqrt(st.minutes / st.minutes.max()),
           color=TRAIN_C, edgecolor=SURF, linewidth=1.5, zorder=3)
alt = st.dropna(subset=['latitude_alternative'])
ax.scatter(alt.longitude_alternative, alt.latitude_alternative, s=40, facecolor='none',
           edgecolor=TRAIN_C, linewidth=1.2, zorder=3)
# Label positions in data coordinates, joined to the site by a hairline leader.
label_xy = {'01 Richards Bay': (29.6, -28.4), '02 Durban': (28.6, -30.4), '13 Saldanha Bay': (18.4, -31.6),
            '10 Sea Point': (19.6, -32.5), '09 Fish Hoek': (19.2, -35.0), '07 St Francis': (22.6, -35.2),
            '03 Cape Recife': (24.2, -35.6), '06 Riy Banks': (26.4, -35.2), '05 St Croix': (24.4, -32.4),
            '04 Bird Island': (27.2, -33.0)}
for _, r in st.iterrows():
    tx, ty = label_xy[r.label]
    ax.annotate(f"{r.label}\n{r.minutes:,} min", (r.longitude_candidate, r.latitude_candidate), xytext=(tx, ty),
                textcoords='data', fontsize=7.5, color=INK2, ha='center', va='center',
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.7, shrinkA=6, shrinkB=4))
ax.set_xlim(16.5, 33.5)
ax.set_ylim(-36.2, -27.6)
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_title('Monitoring sites: marker area ~ sqrt(released minutes); hollow ring = alternative Saldanha coordinate')
ax.set_aspect(1 / np.cos(np.deg2rad(31)))
fig.tight_layout()
save(fig, '00_site_map')

fig, ax = plt.subplots(figsize=(11, 4.8))
order = REC.sort_values(['site', 'first_t'], ascending=[False, True]).reset_index(drop=True)
day = tr.t.dt.floor('D')
cnt = pd.DataFrame({'rec': tr.recorder, 'site': tr.label, 'day': day}).value_counts().reset_index()
for i, r in order.iterrows():
    d = cnt[(cnt.rec == r.recorder) & (cnt.site == r.site)]
    ax.scatter(d.day, np.full(len(d), i), s=2, marker='s', color=TRAIN_C, alpha=0.75, linewidth=0)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f'{s} | {rec_short(r)}' for s, r in zip(order.site, order.recorder)], fontsize=6.8)
ax.set_title('Daily coverage per site | recorder (export clock)')
ax.grid(axis='y', visible=False)
fig.tight_layout()
save(fig, '01_timeline')

# --------------------------------------------------------------- 02 spectra per site
log('fig 02 recorder spectra')
spec_rows = []
for (lab, rec), g in tr[GOOD].groupby(['label', 'recorder'], sort=True):
    P = np.percentile(V[g.index.values], [10, 50, 90], axis=0)
    for k, name in enumerate(['p10', 'p50', 'p90']):
        spec_rows.append({'site': lab, 'recorder': rec, 'stat': name, **dict(zip(BANDS, P[k].round(2)))})
SPEC = table(pd.DataFrame(spec_rows), 'recorder_spectrum_quantiles')
tv = XT.reshape(-1, 25)
tv = tv[np.isfinite(tv).all(1)]
TP = np.percentile(tv, [10, 50, 90], axis=0)
fig, axes = plt.subplots(2, 5, figsize=(16, 6.2), sharex=True, sharey=True)
for ax, lab in zip(axes.ravel(), SITES):
    ax.fill_between(FREQ, TP[0], TP[2], color=MUTED, alpha=0.12, linewidth=0)
    ax.plot(FREQ, TP[1], color=MUTED, ls='--', lw=1.2, label='TEST p50 (p10-p90 grey)')
    for j, rec in enumerate(SPEC[SPEC.site == lab].recorder.unique()):
        s = SPEC[(SPEC.site == lab) & (SPEC.recorder == rec)].set_index('stat')[BANDS]
        ax.fill_between(FREQ, s.loc['p10'], s.loc['p90'], color=SERIES[j], alpha=0.12, linewidth=0)
        ax.plot(FREQ, s.loc['p50'], color=SERIES[j], label=rec_short(rec))
    ax.set_title(lab)
    freq_axis(ax)
    ax.legend(loc='lower left', fontsize=6)
for ax in axes[:, 0]:
    ax.set_ylabel('band level (dB, export convention)')
fig.suptitle('Median band-level spectrum per recorder (shaded p10-p90) vs. pooled TEST cells', color=INK)
fig.tight_layout()
save(fig, '02_spectra_by_site_recorder')

# ----------------------------------------------------- 03 which bands carry the energy
log('fig 03 band energy share')
rows = []
for lab in SITES:
    sel = GOOD & (tr.label.values == lab)
    L = db2lin(V[sel])
    rows.append((lab, (L / L.sum(1, keepdims=True)).mean(0)))
Lt = db2lin(XT)
bt = np.nansum(Lt, 1)
rows.append(('TEST windows', (bt / bt.sum(1, keepdims=True)).mean(0)))
ESH = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows], columns=BANDS)
table(ESH.reset_index().rename(columns={'index': 'set'}).round(4), 'band_energy_share')
cum200 = ESH.loc[:, FREQ <= 200].sum(1)
cum500 = ESH.loc[:, FREQ <= 520].sum(1)
SUMMARY['energy_share_le200Hz'] = cum200.round(3).to_dict()
SUMMARY['energy_share_le500Hz'] = cum500.round(3).to_dict()
fig, ax = plt.subplots(figsize=(13, 4.4))
im = ax.imshow(ESH.values * 100, aspect='auto', cmap=BLUES, vmin=0)
ax.set_yticks(range(len(ESH)))
ax.set_yticklabels([f'{i}  (<=200 Hz: {c:.0%})' for i, c in zip(ESH.index, cum200)])
ax.set_xticks(range(25))
ax.set_xticklabels(BANDS, rotation=60, fontsize=7)
ax.grid(False)
ax.axhline(len(ESH) - 1.5, color=SURF, lw=3)
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label('mean share of minute (or window) linear energy, %')
ax.set_title('Where the linear energy lives: the target is an energy ratio, so the loudest (low) bands dominate it')
fig.tight_layout()
save(fig, '03_band_energy_share')

# ----------------------------------------------------------------- 04 diurnal / season
log('fig 04 diurnal + seasonal')
hour = tr.t.dt.hour.values
month = tr.t.dt.month.values
# Recorders at one site differ by up to ~20 dB, so remove each recorder's own median first;
# otherwise a recorder swap masquerades as a diurnal or seasonal effect.
REC_MED = pd.DataFrame(np.where(GOOD[:, None], V, np.nan)).groupby(tr.recorder.values).transform('median').values
VD = V - REC_MED
BBD = BB - pd.Series(np.where(GOOD, BB, np.nan)).groupby(tr.recorder.values).transform('median').values
pick = {'63 Hz': 3, '126 Hz': 6, '1 kHz': 15, '5 kHz': 22}
fig, axes = plt.subplots(2, 5, figsize=(16, 5.6), sharex=True)
di_rows = []
for ax, lab in zip(axes.ravel(), SITES):
    sel = GOOD & (tr.label.values == lab)
    for j, (nm, b) in enumerate(pick.items()):
        dev = pd.Series(VD[sel, b]).groupby(hour[sel]).median()
        ax.plot(dev.index, dev.values, color=SERIES[j], label=nm, marker='o', ms=2.5)
        di_rows += [{'site': lab, 'band': nm, 'hour': int(h), 'median_dev_db': round(float(v), 2)}
                    for h, v in dev.items()]
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_title(lab)
    ax.set_xticks([0, 6, 12, 18, 23])
axes[0, 0].legend(loc='upper left')
for ax in axes[1]:
    ax.set_xlabel('hour of export clock (TZ unverified)')
for ax in axes[:, 0]:
    ax.set_ylabel('median - recorder median (dB)')
fig.suptitle('Diurnal pattern of band levels per site (each recorder centred on its own median)', color=INK)
fig.tight_layout()
save(fig, '04_diurnal_by_site')
table(pd.DataFrame(di_rows), 'diurnal_band_deviation')

mon = pd.DataFrame({'label': tr.label.values[GOOD], 'month': month[GOOD], 'bb': BBD[GOOD]})
MM = mon.groupby(['label', 'month']).bb.median().unstack().reindex(SITES)
MC = mon.groupby(['label', 'month']).size().unstack().reindex(SITES)
MM = MM.where(MC >= 2 * 1440)      # hide months with < 2 recorded days
table(MM.round(2).reset_index(), 'monthly_broadband_deviation')
fig, ax = plt.subplots(figsize=(9, 4.2))
lim = 8
im = ax.imshow(MM.values, cmap=DIVERGE, vmin=-lim, vmax=lim, aspect='auto')
for i in range(MM.shape[0]):
    for j in range(MM.shape[1]):
        if np.isfinite(MM.values[i, j]):
            ax.text(j, i, f'{MM.values[i, j]:+.1f}\n{MC.values[i, j] / 1440:.0f}d', ha='center', va='center',
                    fontsize=6, color=INK)
ax.set_xticks(range(12))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set_yticks(range(len(SITES)))
ax.set_yticklabels(SITES)
ax.grid(False)
fig.colorbar(im, ax=ax, pad=0.01).set_label('median broadband - recorder median (dB)')
ax.set_title('Seasonality of broadband level (cell: dB deviation / recorded days; <2 days hidden)')
fig.tight_layout()
save(fig, '05_monthly_broadband')

# ------------------------------------------------------- 06 correlation + PCA structure
log('fig 06 band correlation / PCA')
gi = np.flatnonzero(GOOD)
sub = RNG.choice(gi, size=min(400_000, gi.size), replace=False)
Xc = VD[sub].astype(np.float64)
C = np.corrcoef(Xc.T)
Xz = (Xc - Xc.mean(0)) / Xc.std(0)
U, S, VT = np.linalg.svd(Xz, full_matrices=False)
evr = S ** 2 / (S ** 2).sum()
SUMMARY['pca_explained_variance_top5'] = evr[:5].round(3).tolist()
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={'width_ratios': [1.1, 0.7, 1.2]})
ax = axes[0]
im = ax.imshow(C, cmap=DIVERGE, vmin=-1, vmax=1)
ax.set_xticks(range(0, 25, 3))
ax.set_xticklabels(BANDS[::3], rotation=60, fontsize=7)
ax.set_yticks(range(0, 25, 3))
ax.set_yticklabels(BANDS[::3], fontsize=7)
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
ax.set_title('Band-band correlation (dB, recorder-median removed)')
ax = axes[1]
ax.bar(np.arange(1, 9), evr[:8] * 100, color=TRAIN_C, width=0.6)
for k in range(8):
    ax.text(k + 1, evr[k] * 100 + 0.8, f'{evr[k]:.0%}', ha='center', fontsize=7, color=INK2)
ax.set_xlabel('principal component')
ax.set_ylabel('explained variance (%)')
ax.set_title('PCA scree')
ax = axes[2]
for k in range(3):
    ax.plot(FREQ, VT[k] * np.sign(VT[k].sum() or 1), color=SERIES[k], marker='o', ms=3, label=f'PC{k + 1} ({evr[k]:.0%})')
ax.axhline(0, color=AXIS, lw=0.8)
freq_axis(ax)
ax.set_ylabel('loading')
ax.legend()
ax.set_title('PCA loadings: level / tilt / mid-band bump')
fig.tight_layout()
save(fig, '06_band_correlation_pca')
table(pd.DataFrame(C, index=BANDS, columns=BANDS).round(3).reset_index(), 'band_correlation')

# -------------------------------------------------- 07 temporal autocorrelation (ACF)
log('fig 07 temporal ACF')
lags = np.array([1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 360, 720, 1440])
acf_rows = []
for (lab, rec), g in tr.groupby(['label', 'recorder'], sort=True):
    idx = g.index.values
    idx = idx[GOOD[idx]]
    m = TSEC[idx] // 60
    if len(m) < 500:
        continue
    full = np.full(m.max() - m.min() + 1, np.nan)
    full[m - m.min()] = BB[idx]
    for L in lags:
        a, b = full[:-L], full[L:]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 200:
            acf_rows.append({'site': lab, 'recorder': rec, 'lag_min': int(L), 'pairs': int(ok.sum()),
                             'acf': float(np.corrcoef(a[ok], b[ok])[0, 1])})
ACF = table(pd.DataFrame(acf_rows), 'broadband_acf')
acf_site = (ACF.groupby(['site', 'lag_min'])[['acf', 'pairs']]
            .apply(lambda d: np.average(d.acf, weights=d.pairs)).unstack(0))


def window_features(W):
    pres = np.isfinite(W).all(axis=2)
    lin = db2lin(W)
    nmin = pres.sum(1)
    tot = np.nansum(lin, axis=(1, 2))
    Lall = lin2db(np.where(pres, np.nansum(lin, 2), np.nan))
    Llow = lin2db(np.where(pres, np.nansum(lin[:, :, LOW], 2), np.nan))
    Lhigh = lin2db(np.where(pres, np.nansum(lin[:, :, HIGH], 2), np.nan))
    band_lin = np.nansum(lin, axis=1)
    mspec = np.nanmean(W, axis=1)
    # Per-band 20th percentile over present minutes (missing minutes are whole rows, sorted last);
    # same linear interpolation as np.nanpercentile but much faster on 100k windows.
    srt = np.sort(lin, axis=1)
    pos = 0.2 * (nmin - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, nmin - 1)
    w = (pos - lo)[:, None]
    ar = np.arange(len(W))
    q20 = srt[ar, lo] * (1 - w) + srt[ar, hi] * w
    f = pd.DataFrame({
        'level_db': lin2db(tot / nmin),
        'low_level_db': np.nanmean(Llow, 1),
        'bump_low_db': np.nanmax(Llow, 1) - np.nanmedian(Llow, 1),
        'range_low_db': np.nanmax(Llow, 1) - np.nanmin(Llow, 1),
        'bump_high_db': np.nanmax(Lhigh, 1) - np.nanmedian(Lhigh, 1),
        'std_broadband_db': np.nanstd(Lall, 1),
        'mean_abs_step_db': np.nanmean(np.abs(np.diff(Lall, axis=1)), 1),
        'peak_pos_low': np.nanargmax(np.where(np.isfinite(Llow), Llow, -np.inf), 1) + 1,
        'centroid_hz': 10 ** ((band_lin * np.log10(FREQ)).sum(1) / band_lin.sum(1)),
        'tilt_db': mspec[:, FREQ <= 130].mean(1) - mspec[:, FREQ >= 1990].mean(1),
        'max_minute_share': np.nanmax(np.where(pres, np.nansum(lin, 2), np.nan), 1) / tot,
        'max_band_share': band_lin.max(1) / tot,
        'top_band_hz': FREQ[band_lin.argmax(1)],
        'corr_low_high': rowcorr(Llow, Lhigh),
        'lag1_acf_broadband': rowcorr(Lall[:, :-1], Lall[:, 1:]),
        'naive_share_q20': 1 - (q20 * nmin[:, None]).sum(1) / tot,
        'naive_share_min': 1 - (np.nanmin(lin, axis=1) * nmin[:, None]).sum(1) / tot,
        'n_missing': 15 - nmin,
    })
    return f, Llow


log('window features')
FT, LlowT = window_features(XT)
FTR, LlowTR = window_features(WT)
FTR = pd.concat([MT.drop(columns=['n_missing']).reset_index(drop=True), FTR], axis=1)
FT.insert(0, 'ID', test.ID.values)
pattern_test = [','.join(str(k + 1) for k in np.flatnonzero(~np.isfinite(w).all(1))) or 'none' for w in XT]
FT['pattern'] = pattern_test
table(FT.round(5), 'test_window_features')
site_w = 1.0 / FTR.label.map(FTR.label.value_counts()).values   # site-balanced weights

steps_tr = np.abs(np.diff(lin2db(np.where(np.isfinite(WT).all(2), np.nansum(db2lin(WT), 2), np.nan)), axis=1)).ravel()
steps_te = np.abs(np.diff(lin2db(np.where(np.isfinite(XT).all(2), np.nansum(db2lin(XT), 2), np.nan)), axis=1)).ravel()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
others = [s for s in acf_site.columns if s not in SITES[:5]]
for s in others:
    ax.plot(acf_site.index, acf_site[s], color=AXIS, lw=1)
ax.plot([], [], color=AXIS, lw=1, label='other sites')
for j, s in enumerate([s for s in SITES if s in acf_site.columns][:5]):
    ax.plot(acf_site.index, acf_site[s], color=SERIES[j], marker='o', ms=3, label=s)
ax.set_xscale('log')
ax.set_xticks([1, 5, 15, 60, 360, 1440])
ax.set_xticklabels(['1', '5', '15', '60', '360', '1440'])
ax.minorticks_off()
ax.set_xlabel('lag (minutes)')
ax.set_ylabel('autocorrelation of broadband dB')
ax.legend(fontsize=6.5)
ax.set_title('Background memory: broadband level stays correlated for hours')
ax = axes[1]
bins = np.linspace(0, 12, 61)
for x, c, nm in [(steps_tr, TRAIN_C, 'train aligned windows'), (steps_te, TEST_C, 'TEST windows')]:
    x = x[np.isfinite(x)]
    ax.hist(x, bins=bins, density=True, histtype='step', color=c, lw=1.8, label=f'{nm} (median {np.median(x):.2f} dB)')
ax.set_yscale('log')
ax.set_xlabel('|minute-to-minute change| of broadband level (dB)')
ax.set_ylabel('density (log)')
ax.legend()
ax.set_title('Short-term dynamics: TEST has more large steps')
fig.tight_layout()
save(fig, '07_temporal_dynamics')
SUMMARY['abs_step_quantiles'] = {'train': q(steps_tr, (50, 90, 99)), 'test': q(steps_te, (50, 90, 99))}

# ------------------------------------------------------------- 08 missing patterns
log('fig 08 missing patterns')


def pat_bucket(p):
    return p if p in ('none', '1', '15', '14,15', '1,2') else 'other'


cols = ['none', '1', '15', '14,15', '1,2', 'other']
MP = (RAWPAT.assign(b=RAWPAT.pattern.map(pat_bucket)).groupby(['label', 'recorder']).b
      .value_counts(normalize=True).unstack().reindex(columns=cols).fillna(0))
MP.index = [f'{a} | {rec_short(b)}' for a, b in MP.index]
pooled = RAWPAT.pattern.map(pat_bucket).value_counts(normalize=True).reindex(cols).fillna(0)
tp = pd.Series(pattern_test).map(pat_bucket).value_counts(normalize=True).reindex(cols).fillna(0)
MP.loc['TRAIN pooled (quarter-hour aligned)'] = pooled
MP.loc['TEST'] = tp
table(MP.round(4).reset_index().rename(columns={'index': 'set'}), 'missing_pattern_by_recorder')
SUMMARY['test_missing_patterns'] = pd.Series(pattern_test).value_counts().to_dict()
fig, ax = plt.subplots(figsize=(9, 6.6))
im = ax.imshow(MP.values * 100, cmap=BLUES, vmin=0, vmax=100, aspect='auto')
for i in range(MP.shape[0]):
    for j in range(MP.shape[1]):
        v = MP.values[i, j] * 100
        if v >= 0.5:
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=7, color=SURF if v > 55 else INK)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels([f'missing: {c}' for c in cols], rotation=20)
ax.set_yticks(range(len(MP)))
ax.set_yticklabels(MP.index, fontsize=7)
ax.axhline(len(MP) - 2.5, color=SURF, lw=3)
ax.grid(False)
fig.colorbar(im, ax=ax, pad=0.01).set_label('% of windows')
ax.set_title('Missing-minute patterns: raw training availability (quarter-hour aligned) vs TEST')
fig.tight_layout()
save(fig, '08_missing_patterns')
grp = FT.groupby(FT.pattern.map(pat_bucket))[['level_db', 'tilt_db', 'centroid_hz', 'bump_low_db']].median()
grp['windows'] = FT.pattern.map(pat_bucket).value_counts()
table(grp.round(3).reset_index().rename(columns={'pattern': 'missing_pattern'}), 'test_features_by_missing_pattern')

# ------------------------------------------------- 09 window feature distributions
log('fig 09 window feature distributions')
fcols = ['level_db', 'bump_low_db', 'range_low_db', 'bump_high_db', 'std_broadband_db', 'mean_abs_step_db',
         'tilt_db', 'centroid_hz', 'max_minute_share', 'corr_low_high', 'lag1_acf_broadband', 'naive_share_q20']
cmp_rows = []
for c in fcols:
    a, b = FTR[c].values, FT[c].values
    ok = np.isfinite(a)
    # site-balanced quantiles via weighted resample
    bal = RNG.choice(a[ok], size=50_000, p=site_w[ok] / site_w[ok].sum())
    cmp_rows.append({'feature': c, **{f'train_{k}': v for k, v in q(a).items()},
                     **{f'trainbal_{k}': v for k, v in q(bal).items()},
                     **{f'test_{k}': v for k, v in q(b).items()},
                     'ks_test_vs_train': ks_2samp(a[ok], b[np.isfinite(b)]).statistic,
                     'ks_test_vs_trainbal': ks_2samp(bal, b[np.isfinite(b)]).statistic})
CMP = table(pd.DataFrame(cmp_rows).round(4), 'window_feature_train_vs_test')
fig, axes = plt.subplots(3, 4, figsize=(16, 9.5))
for ax, c in zip(axes.ravel(), fcols):
    a, b = FTR[c].values, FT[c].values
    ok = np.isfinite(a)
    a, b = a[ok], b[np.isfinite(b)]
    lo, hi = np.nanpercentile(np.r_[a, b], [0.5, 99.5])
    bins = np.linspace(lo, hi, 50)
    if c == 'centroid_hz':
        bins = np.geomspace(max(lo, 1), hi, 50)
        ax.set_xscale('log')
    ax.hist(a, bins=bins, density=True, histtype='step', lw=1.6, color=TRAIN_C, label='train (pooled)')
    ax.hist(b, bins=bins, density=True, histtype='step', lw=1.8, color=TEST_C, label='TEST')
    ks = CMP.set_index('feature').loc[c]
    ax.set_title(f'{c}   KS={ks.ks_test_vs_train:.2f}')
    ax.set_yticks([])
axes[0, 0].legend(loc='upper left')
fig.suptitle('15-minute window features: training aligned windows vs TEST (KS = two-sample statistic)', color=INK)
fig.tight_layout()
save(fig, '09_window_features_train_vs_test')

# ---------------------------------------- 10 excess spectrum of transients + band ranges
log('fig 10 transient spectra')


def bump_excess(W, Llow, mask):
    pk = np.nanargmax(np.where(np.isfinite(Llow), Llow, -np.inf), 1)
    peak = W[np.arange(len(W)), pk]
    return (peak - np.nanmedian(W, axis=1))[mask]


thr_tr = np.nanpercentile(FTR.bump_low_db, 90)
EX_tr = bump_excess(WT, LlowTR, FTR.bump_low_db.values >= thr_tr)
EX_te = bump_excess(XT, LlowT, FT.bump_low_db.values >= thr_tr)
SUMMARY['bump_low_p90_train_db'] = float(thr_tr)
SUMMARY['test_share_windows_above_train_bump_p90'] = float((FT.bump_low_db >= thr_tr).mean())
SUMMARY['test_share_windows_above_train_bump_p99'] = float((FT.bump_low_db >= np.nanpercentile(FTR.bump_low_db, 99)).mean())
rng_tr = np.nanmax(WT, 1) - np.nanmin(WT, 1)
rng_te = np.nanmax(XT, 1) - np.nanmin(XT, 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
for E, c, nm in [(EX_tr, TRAIN_C, f'train windows, bump_low >= p90 ({len(EX_tr):,})'),
                 (EX_te, TEST_C, f'TEST windows above the same threshold ({len(EX_te):,})')]:
    mu = np.nanmedian(E, 0)
    lo, hi = np.nanpercentile(E, [25, 75], axis=0)
    ax.fill_between(FREQ, lo, hi, color=c, alpha=0.12, linewidth=0)
    ax.plot(FREQ, mu, color=c, marker='o', ms=3, label=nm)
freq_axis(ax)
ax.set_ylabel('peak minute - window median (dB)')
ax.legend()
ax.set_title('Spectral signature of the loudest low-band minute (median, IQR)')
ax = axes[1]
for R, c, nm in [(rng_tr, TRAIN_C, 'train'), (rng_te, TEST_C, 'TEST')]:
    ax.plot(FREQ, np.nanmedian(R, 0), color=c, label=f'{nm} median')
    ax.plot(FREQ, np.nanpercentile(R, 90, 0), color=c, ls='--', label=f'{nm} p90')
freq_axis(ax)
ax.set_ylabel('within-window range max-min (dB)')
ax.legend()
ax.set_title('Per-band temporal range inside a 15-min window')
fig.tight_layout()
save(fig, '10_transient_spectra')
table(pd.DataFrame({'band': BANDS, 'train_peak_excess_med': np.nanmedian(EX_tr, 0),
                    'test_peak_excess_med': np.nanmedian(EX_te, 0),
                    'train_range_med': np.nanmedian(rng_tr, 0), 'test_range_med': np.nanmedian(rng_te, 0),
                    'train_range_p90': np.nanpercentile(rng_tr, 90, 0),
                    'test_range_p90': np.nanpercentile(rng_te, 90, 0)}).round(3), 'band_transient_profiles')

# --------------------------------------------------------------- 11/12 example windows


def window_gallery(W, F, title, name, ids=None):
    order = np.argsort(F.bump_low_db.values)
    ok = order[np.isfinite(F.bump_low_db.values[order])]
    picks = [ok[int(p * (len(ok) - 1))] for p in [0.05, 0.25, 0.5, 0.75, 0.9, 0.97, 0.995, 1.0]]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.4))
    for ax, i in zip(axes.ravel(), picks):
        im = ax.imshow(W[i].T, origin='lower', aspect='auto', cmap=BLUES,
                       extent=[0.5, 15.5, -0.5, 24.5])
        ax.set_yticks([0, 6, 12, 18, 24])
        ax.set_yticklabels([BANDS[k][4:] for k in [0, 6, 12, 18, 24]])
        ax.set_xticks([1, 5, 10, 15])
        ax.grid(False)
        lab = ids[i] if ids is not None else f"{F.label.iloc[i]} {F.start.iloc[i]:%Y-%m-%d %H:%M}"
        ax.set_title(f'{lab}\nbump_low={F.bump_low_db.iloc[i]:.1f} dB, level={F.level_db.iloc[i]:.1f} dB', fontsize=7.5)
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05)
    for ax in axes[1]:
        ax.set_xlabel('minute position')
    for ax in axes[:, 0]:
        ax.set_ylabel('band centre (Hz)')
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    save(fig, name)


log('fig 11/12 galleries')
window_gallery(XT, FT, 'TEST windows at increasing low-band "bump" quantiles (p5 ... max); colour = dB',
               '11_test_window_gallery', ids=test.ID.values)
window_gallery(WT, FTR, 'Training windows at the same bump quantiles (natural background, no additions)',
               '12_train_window_gallery')

# -------------------------------------------------------------- 13 domain kNN probe
log('fig 13 domain probe')
med_tr = np.nanmedian(WT, axis=1)
med_te = np.nanmedian(XT, axis=1)
lab_tr = FTR.label.values
sel = np.concatenate([RNG.permutation(np.flatnonzero(lab_tr == s))[:4000] for s in SITES])
Xk, yk = med_tr[sel], lab_tr[sel]
groups = (FTR.label.values[sel] + '_' + FTR.start.dt.strftime('%G-%V').values[sel])
gkf = GroupKFold(n_splits=5)
oof = np.empty(len(sel), dtype=object)
oof_d = np.zeros(len(sel))
for trn, val in gkf.split(Xk, yk, groups):
    knn = KNeighborsClassifier(n_neighbors=25).fit(Xk[trn], yk[trn])
    oof[val] = knn.predict(Xk[val])
    oof_d[val] = NearestNeighbors(n_neighbors=1).fit(Xk[trn]).kneighbors(Xk[val])[0][:, 0] / 5.0
bacc = balanced_accuracy_score(yk, oof)
CM = pd.DataFrame(confusion_matrix(yk, oof, labels=SITES), index=SITES, columns=SITES)
table(CM.reset_index().rename(columns={'index': 'true_site'}), 'domain_probe_cv_confusion')
knn = KNeighborsClassifier(n_neighbors=25).fit(Xk, yk)
pred = knn.predict(med_te)
proba = knn.predict_proba(med_te).max(1)
d_te = NearestNeighbors(n_neighbors=1).fit(Xk).kneighbors(med_te)[0][:, 0] / 5.0
ood_thr = np.percentile(oof_d, 99)
FT['probe_site'] = pred
FT['probe_conf'] = proba
FT['probe_nn_rms_db'] = d_te
SUMMARY['domain_probe'] = {
    'cv_balanced_accuracy_week_grouped': round(float(bacc), 3),
    'test_pred_site_share': pd.Series(pred).value_counts(normalize=True).round(3).to_dict(),
    'train_oof_nn_rms_db_p50_p99': [round(float(np.median(oof_d)), 3), round(float(ood_thr), 3)],
    'test_nn_rms_db_p50': round(float(np.median(d_te)), 3),
    'test_share_beyond_train_p99': round(float((d_te > ood_thr).mean()), 3),
}
xt = pd.crosstab(FT.probe_site, FT.pattern.map(pat_bucket)).reindex(index=SITES, columns=cols).fillna(0).astype(int)
table(xt.reset_index(), 'domain_probe_vs_missing_pattern')
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={'width_ratios': [1.3, 1, 1.1]})
ax = axes[0]
share_tr = pd.Series(lab_tr).value_counts(normalize=True).reindex(SITES).fillna(0)
share_te = pd.Series(pred).value_counts(normalize=True).reindex(SITES).fillna(0)
x = np.arange(len(SITES))
ax.bar(x - 0.2, share_tr * 100, width=0.38, color=TRAIN_C, label='train aligned windows (actual)')
ax.bar(x + 0.2, share_te * 100, width=0.38, color=TEST_C, label='TEST (kNN-probe predicted)')
ax.set_xticks(x)
ax.set_xticklabels(SITES, rotation=40, ha='right')
ax.set_ylabel('% of windows')
ax.legend()
ax.set_title(f'Which site does a window resemble? (probe CV bal-acc {bacc:.2f})')
ax = axes[1]
bins = np.linspace(0, np.percentile(np.r_[oof_d, d_te], 99.5), 50)
ax.hist(oof_d, bins=bins, density=True, histtype='step', lw=1.6, color=TRAIN_C, label='train (out-of-fold)')
ax.hist(d_te, bins=bins, density=True, histtype='step', lw=1.8, color=TEST_C, label='TEST')
ax.axvline(ood_thr, color=MUTED, ls='--', lw=1)
ax.text(ood_thr, ax.get_ylim()[1] * 0.9, ' train p99', color=INK2, fontsize=7)
ax.set_xlabel('RMS dB distance to nearest training window (median spectrum)')
ax.set_yticks([])
ax.legend()
ax.set_title(f'Novelty: {(d_te > ood_thr).mean():.0%} of TEST beyond train p99')
ax = axes[2]
im = ax.imshow(xt.values, cmap=BLUES, aspect='auto')
for i in range(xt.shape[0]):
    for j in range(xt.shape[1]):
        if xt.values[i, j]:
            ax.text(j, i, xt.values[i, j], ha='center', va='center', fontsize=7,
                    color=SURF if xt.values[i, j] > xt.values.max() * 0.55 else INK)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=30)
ax.set_xlabel('TEST missing-minute pattern')
ax.set_yticks(range(len(SITES)))
ax.set_yticklabels(SITES)
ax.grid(False)
ax.set_title('Probe site x missing pattern (TEST counts)')
fig.tight_layout()
save(fig, '13_domain_probe')

# ------------------------------------------------- 14 natural vessel-like transients
log('fig 14/15 natural transient detection (heuristic)')
events = []
glitches = {}
recorded_min = {}
hour_minutes = {}
for (lab, rec), g in tr.groupby(['label', 'recorder'], sort=True):
    idx = g.index.values
    idx = idx[GOOD[idx]]
    m = TSEC[idx] // 60
    recorded_min[lab] = recorded_min.get(lab, 0) + len(m)
    hm = pd.Series((m % 1440) // 60).value_counts()
    hour_minutes[lab] = hour_minutes.get(lab, pd.Series(0, index=range(24))).add(hm, fill_value=0)
    n = m.max() - m.min() + 1
    Lf = np.full(n, np.nan)
    Rf = np.full(n, np.nan)
    lin = db2lin(V[idx])
    Lf[m - m.min()] = lin2db(lin[:, EVT_LOW].sum(1))
    Rf[m - m.min()] = lin2db(lin[:, EVT_REF].sum(1))
    medL = pd.Series(Lf).rolling(61, center=True, min_periods=10).median().values
    medR = pd.Series(Rf).rolling(61, center=True, min_periods=10).median().values
    exL, exR = Lf - medL, Rf - medR
    flag = (exL >= 6) & ((exL - exR) >= 3)
    f = np.r_[False, flag, False].astype(int)
    st_, en_ = np.flatnonzero(np.diff(f) == 1), np.flatnonzero(np.diff(f) == -1)
    for a, b in zip(st_, en_):
        if b - a < 3:
            continue
        # A passage should rise and fall inside recorded data; events touching a gap or file
        # boundary, or with implausible > 40 dB excess, are logged as artefacts instead.
        if a == 0 or b >= n or not (np.isfinite(Lf[a - 1]) and np.isfinite(Lf[b])) or np.nanmax(exL[a:b]) > 40:
            glitches[lab] = glitches.get(lab, 0) + 1
            continue
        pk = a + int(np.nanargmax(exL[a:b]))
        pm = m.min() + pk
        lo_, hi_ = np.searchsorted(m, [pm - 30, pm + 31])
        ctx = np.nanmedian(V[idx[lo_:hi_]], axis=0)
        row = V[idx[np.searchsorted(m, pm)]]
        events.append({'site': lab, 'recorder': rec, 'start': pd.to_datetime((m.min() + a) * 60, unit='s'),
                       'duration_min': int(b - a), 'peak_excess_low_db': float(exL[pk]),
                       'peak_excess_ref_db': float(exR[pk]), 'hour': int(((m.min() + a) % 1440) // 60),
                       **{f'ex_{B}': float(row[k] - ctx[k]) for k, B in enumerate(BANDS)}})
EV = pd.DataFrame(events)
table(EV.drop(columns=[f'ex_{B}' for B in BANDS]).round(3), 'natural_transient_events')
rate = (EV.groupby('site').size() / pd.Series(recorded_min) * 1440).reindex(SITES).fillna(0)
SUMMARY['natural_events'] = {'n': int(len(EV)), 'per_recorded_day': rate.round(2).to_dict(),
                             'rejected_edge_or_glitch': glitches,
                             'duration_median_min': float(EV.duration_min.median()),
                             'duration_p90_min': float(EV.duration_min.quantile(0.9))}
fig, axes = plt.subplots(1, 4, figsize=(18, 4.3), gridspec_kw={'width_ratios': [1.2, 0.8, 1.1, 1.1]})
ax = axes[0]
ax.barh(range(len(SITES)), rate.values, color=TRAIN_C, height=0.6)
for i, v in enumerate(rate.values):
    ax.text(v, i, f' {v:.1f}', va='center', fontsize=7, color=INK2)
ax.set_yticks(range(len(SITES)))
ax.set_yticklabels(SITES)
ax.invert_yaxis()
ax.set_xlabel('candidate events per recorded day')
ax.set_title('Low-band transients (>=6 dB over 61-min median, >=3 min)')
ax = axes[1]
ax.hist(EV.duration_min.clip(upper=60), bins=np.arange(3, 62, 2), color=TRAIN_C)
ax.set_xlabel('event duration (min, clipped at 60)')
ax.set_ylabel('events')
ax.set_title('Duration')
ax = axes[2]
top = rate.sort_values(ascending=False).index[:4]
for j, s in enumerate(top):
    hc = EV[EV.site == s].hour.value_counts().reindex(range(24)).fillna(0)
    r = hc / hour_minutes[s].reindex(range(24)).replace(0, np.nan) * 1440
    ax.plot(range(24), r.values, color=SERIES[j], marker='o', ms=2.5, label=s)
ax.set_xticks([0, 6, 12, 18, 23])
ax.set_xlabel('start hour (export clock)')
ax.set_ylabel('events per recorded day-equivalent')
ax.legend()
ax.set_title('Diurnal event rate (top-4 sites)')
ax = axes[3]
for j, s in enumerate(top):
    E = EV[EV.site == s][[f'ex_{B}' for B in BANDS]].values
    ax.plot(FREQ, np.nanmedian(E, 0), color=SERIES[j], label=f'{s} (n={len(E)})')
ax.plot(FREQ, np.nanmedian(EX_te, 0), color=MUTED, ls='--', lw=1.4, label='TEST bump windows')
freq_axis(ax)
ax.set_ylabel('peak - 61-min median (dB)')
ax.legend(fontsize=6.5)
ax.set_title('Median excess spectrum at event peak')
fig.tight_layout()
save(fig, '14_natural_transients')

# examples: strongest moderately long events at four different sites
ex = (EV[(EV.duration_min >= 5) & (EV.duration_min <= 40)]
      .sort_values('peak_excess_low_db', ascending=False).groupby('site').head(1).head(4))
fig, axes = plt.subplots(2, 2, figsize=(15, 6.8))
for ax, (_, e) in zip(axes.ravel(), ex.iterrows()):
    g = tr[(tr.label == e.site) & (tr.recorder == e.recorder)]
    c0 = e.start - pd.Timedelta(minutes=80)
    c1 = e.start + pd.Timedelta(minutes=100)
    gg = g[(g.t >= c0) & (g.t < c1) & GOOD[g.index.values]]
    grid = pd.date_range(c0.floor('min'), c1.floor('min'), freq='min')
    M = gg.set_index(gg.t.dt.floor('min'))[BANDS].reindex(grid).values
    im = ax.imshow(M.T, origin='lower', aspect='auto', cmap=BLUES,
                   extent=[0, len(grid), -0.5, 24.5])
    s0 = (e.start - c0) / pd.Timedelta(minutes=1)
    ax.axvspan(s0, s0 + e.duration_min, ymin=0.97, ymax=1.0, color=TEST_C)
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_yticklabels([BANDS[k][4:] for k in [0, 6, 12, 18, 24]])
    ax.grid(False)
    ax.set_xlabel(f'minutes from {c0:%Y-%m-%d %H:%M} (orange bar = detected event)')
    ax.set_ylabel('band (Hz)')
    ax.set_title(f'{e.site}: +{e.peak_excess_low_db:.1f} dB low-band, {e.duration_min} min')
    fig.colorbar(im, ax=ax, pad=0.01).set_label('dB')
fig.suptitle('Candidate natural vessel-like passages in released SA data (heuristic, unverified)', color=INK)
fig.tight_layout()
save(fig, '15_natural_transient_examples')

# ------------------------------------------------------------- 16 target mechanics
log('fig 16 target mechanics')
dL = np.linspace(0, 20, 401)
share = 1 - 10 ** (-dL / 10)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
ax = axes[0]
ax.plot(dL, share, color=TRAIN_C)
for d in [0.5, 1, 3, 6, 10]:
    s = 1 - 10 ** (-d / 10)
    ax.plot([d], [s], 'o', color=TRAIN_C, ms=5, mec=SURF, mew=1.5)
    ax.annotate(f'+{d:g} dB -> {s:.2f}', (d, s), xytext=(6, -10), textcoords='offset points', fontsize=7.5, color=INK2)
ax.set_xlabel('window energy rise over background, dL (dB)')
ax.set_ylabel('share_added_vessel')
ax.set_title('share = 1 - 10^(-dL/10)')
ax = axes[1]
e = np.linspace(-3, 3, 301)
for j, s in enumerate([0.0, 0.2, 0.5, 0.8]):
    err = np.clip(1 - (1 - s) * 10 ** (e / 10), 0, 1) - s
    ax.plot(e, err, color=SERIES[j], label=f'true share {s:.1f}')
ax.axhline(0, color=AXIS, lw=0.8)
ax.axvline(0, color=AXIS, lw=0.8)
ax.set_xlabel('error of background-energy estimate (dB)')
ax.set_ylabel('share error (clipped to [0,1])')
ax.legend()
ax.set_title('A 1 dB background error costs ~0.2 at low share')
ax = axes[2]
bins = np.linspace(0, 1, 51)
ax.hist(FTR.naive_share_q20, bins=bins, density=True, histtype='step', lw=1.6, color=TRAIN_C, label='train (no additions)')
ax.hist(FT.naive_share_q20, bins=bins, density=True, histtype='step', lw=1.8, color=TEST_C, label='TEST')
ax.set_xlabel('naive share = 1 - (per-band q20 over minutes) / window energy')
ax.set_yticks([])
ax.legend()
ax.set_title('Within-window "quiet-minute" baseline: false-positive floor')
fig.tight_layout()
save(fig, '16_target_mechanics')

# --------------------------------------------------------- 17 band-support anomaly
log('12 kHz recorder band support')
sup = []
for (lab, rec), g in tr[GOOD].groupby(['label', 'recorder'], sort=True):
    X = V[g.index.values]
    sup.append({'site': lab, 'recorder': rec, 'nominal_sr_hz': int(g.sample_rate_hz.iloc[0]),
                'nyquist_hz': int(g.sample_rate_hz.iloc[0]) // 2,
                'med_6310_minus_5012': float(np.median(X[:, 23] - X[:, 22])),
                'med_7943_minus_5012': float(np.median(X[:, 24] - X[:, 22])),
                'p01_7943_minus_5012': float(np.percentile(X[:, 24] - X[:, 22], 1))})
table(pd.DataFrame(sup).round(2), 'top_band_support_by_recorder')
FT['d7943_5012'] = np.nanmedian(XT[:, :, 24] - XT[:, :, 22], 1)
SUMMARY['test_d7943_5012_quantiles'] = q(FT.d7943_5012, (1, 5, 50, 95, 99))

# ---------------------------------------- 17 site-matched comparison (probe domains)
# The pooled comparison mixes a site-composition shift with the injected component. Comparing
# each site's training windows only with TEST windows whose median spectrum resembles that site
# isolates what the controlled additions change.
log('fig 17 site-matched comparison')
mfeat = ['bump_high_db', 'corr_low_high', 'lag1_acf_broadband', 'naive_share_q20', 'mean_abs_step_db']
msites = FT.probe_site.value_counts().index[:4].tolist()
match_rows = []
fig, axes = plt.subplots(len(msites), len(mfeat), figsize=(17, 2.5 * len(msites)))
for i, s in enumerate(msites):
    tr_s, te_s = FTR[FTR.label == s], FT[FT.probe_site == s]
    for j, c in enumerate(mfeat):
        a = tr_s[c].values
        b = te_s[c].values
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        p99 = np.percentile(a, 99)
        ks = ks_2samp(a, b).statistic
        match_rows.append({'probe_site': s, 'feature': c, 'train_windows': len(a), 'test_windows': len(b),
                           'train_p50': np.median(a), 'train_p99': p99, 'test_p50': np.median(b),
                           'test_share_above_train_p99': float((b > p99).mean()), 'ks': ks})
        ax = axes[i, j]
        lo, hi = np.percentile(np.r_[a, b], [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        ax.hist(a, bins=bins, density=True, histtype='step', lw=1.6, color=TRAIN_C, label='train windows of this site')
        ax.hist(b, bins=bins, density=True, histtype='step', lw=1.8, color=TEST_C, label='TEST windows probed as this site')
        ax.axvline(p99, color=MUTED, ls='--', lw=0.9)
        ax.set_yticks([])
        ax.set_title(f'{c}  KS={ks:.2f}, {(b > p99).mean():.0%} > train p99', fontsize=8)
        if j == 0:
            ax.set_ylabel(f'{s}\ntrain {len(a):,} / TEST {len(b)}', fontsize=8)
axes[0, 0].legend(fontsize=6.5, loc='upper right')
fig.suptitle('Same-site comparison (dashed = train p99): training windows of a site vs TEST windows resembling it',
             color=INK)
fig.tight_layout()
save(fig, '17_site_matched_comparison')
MATCH = table(pd.DataFrame(match_rows).round(4), 'site_matched_feature_comparison')
SUMMARY['site_matched_bump_high'] = MATCH[MATCH.feature == 'bump_high_db'].set_index('probe_site')[
    ['train_p50', 'test_p50', 'test_share_above_train_p99', 'ks']].round(3).to_dict('index')
SUMMARY['share_bump_high_gt5db'] = {'test': round(float((FT.bump_high_db > 5).mean()), 3),
                                    'train': round(float((FTR.bump_high_db > 5).mean()), 3)}
SUMMARY['peak_pos_low_complete_windows'] = {
    'test': kv(FT[FT.n_missing == 0].peak_pos_low.value_counts(normalize=True).sort_index()),
    'train': kv(FTR[FTR.n_missing == 0].peak_pos_low.value_counts(normalize=True).sort_index())}
SUMMARY['top_band_631Hz_share'] = {'test': round(float((FT.top_band_hz == 631).mean()), 4),
                                   'train': round(float((FTR.top_band_hz == 631).mean()), 4)}

# ------------------------------------------------------------------- remaining stats
lv = XT[np.isfinite(XT)]
SUMMARY['test'] = {
    'windows': int(len(XT)), 'cells': int(XT.size), 'missing_cells': int((~np.isfinite(XT)).sum()),
    'missing_minutes': int((~np.isfinite(XT).all(2)).sum()),
    'partial_minutes': int((np.isfinite(XT).any(2) & ~np.isfinite(XT).all(2)).sum()),
    'db_min': float(lv.min()), 'db_max': float(lv.max()),
    'all_on_0p1_grid': bool(np.all(np.abs(lv * 10 - np.round(lv * 10)) < 1e-3)),
    'top_band_hz_share': kv(FT.top_band_hz.value_counts(normalize=True).head(6)),
    'peak_pos_low_share': kv(FT.peak_pos_low.value_counts(normalize=True).sort_index()),
}
SUMMARY['train_windows_peak_pos_low_share'] = kv(FTR.peak_pos_low.value_counts(normalize=True).sort_index())
SUMMARY['train_top_band_hz_share'] = kv(FTR.top_band_hz.value_counts(normalize=True).head(6))
SUMMARY['benchmark'] = {'public_score': 0.753867825, 'rmse': round(1 - 0.753867825, 6),
                        'if_all_zero_then_E[y^2]': round((1 - 0.753867825) ** 2, 5)}
SUMMARY['feature_ks'] = CMP.set_index('feature')[['ks_test_vs_train', 'ks_test_vs_trainbal']].round(3).to_dict('index')
table(FT.round(5), 'test_window_features')
FTR.drop(columns=['start']).assign(start=FTR.start.astype(str)).sample(20_000, random_state=0).round(4).to_csv(
    TAB / 'train_window_features_sample20k.csv', index=False)
(OUT / 'summary.json').write_text(json.dumps(SUMMARY, indent=2, default=str, ensure_ascii=False))
log('done')
