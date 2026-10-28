"""Training-source-only measured plots and facts for the dataset review."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parent
CV = ROOT / 'data/synthetic/group_cv_nested5_pilot_v1'
OUT = ROOT / 'reports/dataset_figures'
COLORS = {'train': '#77936b', 'validation': '#e5ac72', 'outer_validation': '#7199b5', 'purged_temporal_boundary': '#d4d8d2'}


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=210, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / (name + '.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
        'axes.titleweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': '#263629', 'text.color': '#203026', 'axes.edgecolor': '#9ea99e',
        'grid.color': '#e7ece5', 'figure.constrained_layout.use': True})
    bg = pd.read_parquet(CV / 'background_outer_assignment.parquet')
    ref = pd.read_parquet(CV / 'reference_outer_assignment.parquet')
    audit = json.loads((ROOT / 'reports/group_cv_data_audit.json').read_text())
    checks = json.loads((ROOT / 'reports/group_cv_data_verification.json').read_text())
    summaries = [json.loads((CV / f'fold_{f:02d}/readiness_manifest.json').read_text()) for f in range(5)]
    canonical = [CV / f'fold_{f:02d}' / (s + '.npz') for f in range(5) for s in ['train', 'validation', 'outer_validation']]
    facts = {'official_download_bytes': 97806656, 'official_rows': 1715779,
        'common_band_float32_bytes': 171577900, 'all_aligned_windows': 133641,
        'cv_background_windows': len(bg), 'cv_background_groups': bg.background_group.nunique(),
        'cv_reference_clips': len(ref), 'cv_reference_components': ref.reference_component.nunique(),
        'reference_by_source': ref.groupby('source').agg(clips=('clip_id','size'), seconds=('duration_seconds','sum'),
            native_long_clips=('native_long','sum'), connected_components=('reference_component','nunique')).to_dict('index'),
        'canonical_npz_bytes': sum(f.stat().st_size for f in canonical),
        'canonical_array_payload_bytes': 330880000,
        'cv_directory_logical_file_bytes_including_aliases': sum(f.stat().st_size for f in CV.rglob('*') if f.is_file()),
        'observed_minute_counts_background': bg.observed_minutes.value_counts().sort_index().to_dict(),
        'fold_partitions': [], 'challenge_Test_read': False, 'calibration_or_sealed_arrays_read': False}
    # Figure 1: units are deliberately separated across panels.
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    labels = ['Original rows\n(all preserved)', 'Eligible minute rows', 'Rows used in\n15-min windows']
    values = [1715779, 1698224, 1698184]
    ax[0].barh(labels[::-1], values[::-1], color=['#a9bba1', '#8fa681', '#77936b'])
    ax[0].set_xlim(0, 1900000); ax[0].set_xlabel('One-minute observations'); ax[0].set_title('(a) Original observations')
    for i, v in enumerate(values[::-1]): ax[0].text(v + 14000, i, f'{v:,}', va='center', fontsize=9)
    ax[0].ticklabel_format(axis='x', style='plain'); ax[0].set_xticks([0, 500000, 1000000, 1500000], ['0', '0.5M', '1.0M', '1.5M'])
    vals = [133641, len(bg), 110000]
    ax[1].bar(['Aligned\nwindows', 'CV source\nwindows', 'Synthetic\nacross folds'], vals, color=['#a9bba1', '#77936b', '#7199b5'])
    for i, v in enumerate(vals): ax[1].text(i, v + 3000, f'{v:,}', ha='center')
    ax[1].set_ylim(0, 155000); ax[1].set_ylabel('15 x 25 windows / examples'); ax[1].set_title('(b) Window units; sources are reused')
    rs = ref.groupby('source').size().reindex(['HearMyShip_V1', 'VTUAD_v1.3.2', 'NOAA_SanctSound_extension_v1'])
    ax[2].bar(['HearMyShip', 'VTUAD', 'NOAA native'], rs.to_numpy(), color=['#e5ac72', '#7199b5', '#77936b'])
    for i, v in enumerate(rs): ax[2].text(i, v + 22, f'{v:,}', ha='center')
    ax[2].set_ylim(0, 1100); ax[2].set_ylabel('CV-admitted recording clips'); ax[2].set_title('(c) External recordings: 1,171 total')
    for a in ax: a.set_axisbelow(True); a.grid(axis='x' if a is ax[0] else 'y', alpha=.7)
    save(fig, '01_dataset_scale')
    # Figure 2: actual source-group roles and actual post-purge source counts.
    units = ref.groupby('reference_component').agg(outer_fold=('outer_fold','first'), native=('native_long','sum'),
        source=('source','first'), clips=('clip_id','size')).sort_values(['outer_fold','source','clips'])
    role = np.zeros((len(units), 5), dtype=int)
    for i, r in enumerate(units.itertuples()):
        for f in range(5): role[i, f] = 2 if r.outer_fold == f else 1 if r.outer_fold == (f + 1) % 5 else 0
    fig, ax = plt.subplots(1, 2, figsize=(13, 8), gridspec_kw={'width_ratios':[1.15, 1]})
    cmap = ListedColormap([COLORS[k] for k in ['train', 'validation', 'outer_validation']])
    ax[0].imshow(role, aspect='auto', cmap=cmap, norm=BoundaryNorm([-.5,.5,1.5,2.5],3))
    short = {'HearMyShip_V1':'HMS', 'VTUAD_v1.3.2':'VT', 'NOAA_SanctSound_extension_v1':'NOAA'}
    names = [f'{short[r.source]} {r.clips:>3} clips' + ('  *' if r.native else '') for r in units.itertuples()]
    ax[0].set_yticks(range(len(units)), names, fontsize=8); ax[0].set_xticks(range(5), [f'Fold {f+1}' for f in range(5)])
    ax[0].set_title('(a) 26 connected reference components\n* includes measured native >=15-min material')
    ax[0].tick_params(length=0)
    for y in np.arange(.5,len(units),1): ax[0].axhline(y,color='white',lw=.6)
    left = np.zeros(5)
    for key, label in [('train','Train'),('validation','Inner validation'),('outer_validation','Outer validation'),('purged_temporal_boundary','24-h purge')]:
        count = np.asarray([x['background_counts'].get(key,0) for x in audit['folds']]); pct = count / len(bg) * 100
        ax[1].barh(np.arange(5), pct, left=left, color=COLORS[key], label=label)
        for f in range(5):
            if pct[f] >= 8: ax[1].text(left[f]+pct[f]/2,f,f'{pct[f]:.1f}%',ha='center',va='center',fontsize=9)
        left += pct
    ax[1].set_yticks(range(5),[f'Fold {f+1}' for f in range(5)]); ax[1].invert_yaxis(); ax[1].set_xlim(0,100)
    ax[1].set_xlabel('Percent of 100,965 source background windows'); ax[1].set_title('(b) Background roles after 24-hour purge')
    ax[1].legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2,frameon=False)
    save(fig, '02_nested_group_cv')
    # Figure 3: actual saved train arrays; no outer evaluation or Test analysis.
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))
    coverage = []; hist_edges=np.linspace(0,1,21)
    for f in range(5):
        with np.load(CV / f'fold_{f:02d}/train.npz') as d:
            y = d['share']; mins = np.isfinite(d['db']).all(2).sum(1)
            ax[0].hist(y, bins=hist_edges, histtype='step', linewidth=1.5, label=f'Fold {f+1} train')
            counts = np.bincount(mins,minlength=16); coverage.append(counts)
            facts['fold_partitions'].append({'fold':f,'train_examples':len(y),'zero_target_fraction':float((y==0).mean()),
                'observed_minute_counts_train':dict(enumerate(counts.tolist()))})
    ax[0].set_xlabel('Known controlled added-reference energy share'); ax[0].set_ylabel('Saved training examples per bin')
    ax[0].set_title('(a) Saved target distribution; zeros are intentional'); ax[0].legend(frameon=False,fontsize=8)
    pooled = np.sum(coverage,axis=0); original = np.bincount(bg.observed_minutes,minlength=16)
    x=np.arange(3,16)
    ax[1].bar(x-.2,original[x]/original.sum()*100,width=.4,color='#a9bba1',label='Source background pool')
    ax[1].bar(x+.2,pooled[x]/pooled.sum()*100,width=.4,color='#7199b5',label='Saved train examples (all folds)')
    ax[1].set_xticks(x); ax[1].set_xlabel('Observed minutes in a 15-minute window'); ax[1].set_ylabel('Percent within collection')
    ax[1].set_title('(b) Measured coverage vs controlled sampling'); ax[1].legend(frameon=False,fontsize=8)
    for a in ax: a.grid(axis='y',alpha=.7); a.set_axisbelow(True)
    save(fig, '03_training_distributions')
    # Figure 4: derive known components from saved q and mixture, not a fitted
    # separator. Both the label and panels are a numerical reconstruction.
    with np.load(CV / 'fold_00/train.npz') as d:
        valid_minutes = np.isfinite(d['db']).all(2).sum(1)
        possible=np.flatnonzero((d['share']>.35)&(d['share']<.65)&(valid_minutes>=10)&(valid_minutes<15))
        i=int(possible[0]); db=d['db'][i].astype(float); q=d['cell_share'][i].astype(float); y=float(d['share'][i])
    observed=np.isfinite(db); peak=np.nanmax(db); e=np.where(observed,10**((db-peak)/10),0.)
    b=e*(1-q); added=e*q; recovered=float(added.sum()/e.sum())
    facts['example']={'split':'fold_00/train.npz','index':i,'share':y,'recovered_share':recovered,
        'observed_minutes':int(observed.all(1).sum()),'component_origin':'Reconstructed known components E*(1-q), E*q; not a learned separation output'}
    facts['example']['unweighted_mean_cell_fraction']=float(q[observed].mean())
    fig, ax=plt.subplots(1,4,figsize=(15,4.5),sharey=True)
    cm=plt.get_cmap('viridis').copy();cm.set_bad('#e8eae5')
    for j,(component,title) in enumerate([(b,'Known background'),(added,'Known added component'),(e,'Saved mixture')]):
        z=np.where(observed,10*np.log10(np.maximum(component,1e-6)),np.nan)
        im=ax[j].imshow(z.T,origin='lower',aspect='auto',cmap=cm,vmin=-60,vmax=0,extent=[0,15,0,25])
        ax[j].set_title(f'({chr(97+j)}) {title}');ax[j].set_xlabel('Minute');ax[j].set_xticks([0,5,10,15])
    cq=plt.get_cmap('magma').copy();cq.set_bad('#e8eae5')
    imq=ax[3].imshow(np.where(observed,q,np.nan).T,origin='lower',aspect='auto',cmap=cq,vmin=0,vmax=1,extent=[0,15,0,25])
    ax[3].set_title(f'(d) Known cell fraction\nWindow share = {y:.4f}');ax[3].set_xlabel('Minute');ax[3].set_xticks([0,5,10,15])
    ax[0].set_yticks([.5,5.5,10.5,15.5,20.5,24.5],['31.6','100','316','1,000','3,162','7,943']);ax[0].set_ylabel('Band centre (Hz; index spacing)')
    fig.colorbar(im,ax=list(ax[:3]),label='dB relative to mixture peak (zero clipped at -60 dB)',shrink=.85,pad=.03)
    fig.colorbar(imq,ax=ax[3],label='Added energy / mixture energy',shrink=.85,pad=.03)
    fig.suptitle('A saved training example: known-component reconstruction; gray = unobserved',fontsize=13)
    save(fig,'04_known_component_example')
    facts['figure_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.png')}
    (ROOT/'reports/dataset_review_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2))
    print(json.dumps({'figures':list(facts['figure_sha256']),'canonical_npz_bytes':facts['canonical_npz_bytes'],
        'example':facts['example'],'Test_read':False},indent=2))


if __name__=='__main__':main()
