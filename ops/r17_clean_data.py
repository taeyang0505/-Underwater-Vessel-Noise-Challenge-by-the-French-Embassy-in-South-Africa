"""R17 metadata-balanced, mask-paired CPU experiment; no query-file access.

This is a separately versioned controlled refit data set, not a replacement
for the completed/frozen R16 production data.  Every source background gives
one no-addition example and four observed-positive examples in both regimes.
Thus the empirical full observation-pattern/presence MI is exactly zero,
including within site, recorder and n_obs strata.  Acoustic values are not
claimed independent of labels, and independent source counts are not inferred
from generated row counts or the concentration ESS.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data/synthetic/r17_clean_v1'
REPORT = ROOT / 'reports/r17_clean_pipeline_r1/data'
SEED = 202610300731


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def stable(value):
    return hashlib.sha256(f'{SEED}|{value}'.encode()).hexdigest()


def balanced_sites(frame, count, salt):
    """Without replacement, water-fill the metadata-only site quotas."""
    frame = frame.reset_index(drop=True)
    pools = {}
    for site, part in frame.groupby('site', sort=True):
        pools[site] = sorted(part.index, key=lambda i: stable(f'{salt}|{int(frame.at[i,"base_window_index"])}'))
    result = []
    while len(result) < count:
        advanced = False
        for site in sorted(pools):
            if pools[site] and len(result) < count:
                result.append(pools[site].pop())
                advanced = True
        if not advanced:
            raise ValueError(f'Insufficient independent source rows: {salt}')
    chosen = frame.loc[result].copy()
    return chosen.iloc[np.random.default_rng([SEED, len(salt)]).permutation(count)].reset_index(drop=True)


def reference_pool(source):
    from ops.r16_support_temporal_data import new_reference_split
    old, proof = new_reference_split(source)
    reserved = old[old.r16_reference_partition.isin(['B', 'C'])].copy()
    full = source.references.copy()
    full['reserved_B_Cprime'] = full.reference_row_index.isin(reserved.reference_row_index)
    full['reserved_hull_overlap'] = full.hull_id.isin(set(reserved.hull_id))
    full['reserved_day_overlap'] = full.recording_day.isin(set(reserved.recording_day))
    pool = full[~full.reserved_B_Cprime].drop_duplicates('sha256').copy().reset_index(drop=True)
    if len(reserved) != 44 or len(pool) != 465 or pool.sha256.nunique() != 465:
        raise AssertionError('Predeclared 509 minus 44 reference universe differs')
    pool.to_parquet(OUT / 'reference_pool_465.parquet', index=False)
    reserved.to_parquet(OUT / 'reserved_B19_Cprime25.parquet', index=False)
    cols = ['reference_row_index', 'clip_id', 'sha256', 'weak_class', 'native_complete_minute_count',
            'duration_seconds', 'hull_id', 'recording_day', 'utc_begin', 'utc_end',
            'reserved_B_Cprime', 'reserved_hull_overlap', 'reserved_day_overlap']
    full[cols].to_csv(REPORT / 'reference_pool_and_reservation_audit.csv', index=False)
    direct = {
        'all': len(full), 'reserved_byte_distinct_clips': len(reserved), 'eligible_byte_distinct_clips': len(pool),
        'weak_background_classes_in_eligible': int(pool.weak_class.eq('background').sum()),
        'eligible_shared_reserved_hull': int(pool.reserved_hull_overlap.sum()),
        'eligible_shared_reserved_recording_day': int(pool.reserved_day_overlap.sum()),
        'all509_transitive_reference_graph_components': int(full.reference_component.nunique()),
        'strict_reference_independence_claimed': False,
        'semantics': 'Known energy of an added measured recording, including ambient; original weak class is not a source-energy label.',
        'dedicated_ambient16_in_this_pool': False,
        'B_Cprime_still_excluded_from_all_new_roles': True,
        'old_selected_graph_proof_retained_without_extending_it_to465': proof['partition_counts'],
    }
    return pool, direct


def prepare_backgrounds():
    from pipeline.background_cv_r9 import proposed_purge_gap
    from pipeline.r16_production_data import _crossrole
    train_path = ROOT / 'data/synthetic/r16_multisupport_v1/fold_00/train_backgrounds.parquet'
    cv_path = ROOT / 'data/synthetic/r16_production_v1/fold_00/cv_validation_source_pool.parquet'
    oldfit_path = ROOT / 'reports/r16_support_temporal_r1/data/local_fit_backgrounds.parquet'
    train = pd.read_parquet(train_path)
    # Keep low-observation sources, but predeclared metadata cap avoids another
    # almost-half low-density selection set. No Test counts/statistics are read.
    sparse = train[train.qc_observed_minutes.lt(12)].copy()
    sparse = balanced_sites(sparse, min(240, len(sparse)), 'train_sparse_cap')
    dense = train[train.qc_observed_minutes.ge(12)].copy()
    # Site-balanced water filling across all observed densities; sparse share
    # is at most 10% of selected source backgrounds, not label-dependent.
    admitted = pd.concat([dense, sparse], ignore_index=True)
    chosen_train = balanced_sites(admitted, 2400, 'train_source')
    cv = pd.read_parquet(cv_path)
    cv = cv[cv.qc_observed_minutes.ge(12)].copy().reset_index(drop=True)
    mapping = {}
    for site, frame in cv.groupby('site', sort=True):
        groups = sorted(frame.background_group.unique(), key=lambda g: hashlib.sha256(('r17clean' + str(g)).encode()).hexdigest())
        mapping.update({g: 'selection' if i % 2 == 0 else 'ensemble_fit' for i, g in enumerate(groups)})
    cv['proposed_role'] = cv.background_group.map(mapping)
    cv['r17_cross_role_support_gap_hours'] = proposed_purge_gap(cv) / 3600.
    purged = cv[cv.r17_cross_role_support_gap_hours.lt(24)].copy()
    cv = cv[cv.r17_cross_role_support_gap_hours.ge(24)].copy()
    stop = cv[cv.proposed_role.eq('selection')].copy()
    fit = cv[cv.proposed_role.eq('ensemble_fit')].copy()
    oldfit = pd.read_parquet(oldfit_path)
    oldfit = oldfit[oldfit.qc_observed_minutes.ge(12)].copy()
    # The old shared-fit source blocks were already 24h-purged from ownfold
    # source pools. Adding their dense rows recovers the otherwise absent Cape.
    fit = pd.concat([fit, oldfit], ignore_index=True).drop_duplicates('base_window_index')
    chosen_stop = balanced_sites(stop, 400, 'earlystop_source')
    chosen_fit = balanced_sites(fit, 400, 'ensemblefit_source')
    frames = {'train': chosen_train, 'earlystop': chosen_stop, 'ensemble_fit': chosen_fit}
    proof = []
    names = list(frames)
    for j, name in enumerate(names):
        proof.extend({'primary': name, **p} for p in _crossrole(frames[name], {n: frames[n] for n in names[j+1:]}))
    fixed = {name: pd.read_parquet(ROOT / f'reports/r16_support_temporal_r1/data/{name}_backgrounds.parquet')
             for name in ['local_gate', 'local_report', 'domain_holdout']}
    for name, frame in frames.items():
        proof.extend({'primary': name, **p} for p in _crossrole(frame, fixed))
        frame.to_parquet(OUT / f'{name}_backgrounds.parquet', index=False)
    purged.to_parquet(OUT / 'new_earlystop_fit_purged_backgrounds.parquet', index=False)
    summary = []
    for name, frame in frames.items():
        for (site, nobs), g in frame.groupby(['site', 'qc_observed_minutes']):
            summary.append({'role': name, 'site': site, 'n_obs': int(nobs), 'unique_backgrounds': len(g), 'rows_per_regime': len(g)*5})
    pd.DataFrame(summary).to_csv(REPORT / 'background_site_density_composition.csv', index=False)
    return frames, {'cross_role_purge': proof, 'new_stop_fit_purged_sources': len(purged),
                    'source_paths_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [train_path, cv_path, oldfit_path]},
                    'metadata_selection_only': True, 'high_density_minimum_eval_n_obs': 12,
                    'ensemble_fit_oldfit_rows_used': int(chosen_fit.base_window_index.isin(oldfit.base_window_index).sum()),
                    'clean_fit_not_a_retroactive_cure_for_old_models_selection_leakage': True}


def schedule(pool, n, role):
    from ops.r16_forward_data import OPERATORS
    rng = np.random.default_rng([SEED, sum(role.encode()), 0x524546])
    zeros = np.arange(n) % 5 == 0
    positive = np.flatnonzero(~zeros)
    available = pool.reference_row_index.to_numpy(np.int64)
    count = len(positive)
    values = np.concatenate([rng.permutation(available) for _ in range((count+len(available)-1)//len(available))])[:count]
    rng.shuffle(values)
    refs = np.full(n, -1, np.int64)
    refs[positive] = values
    operators = np.full(n, 'control_eq6', dtype='<U32')
    op = np.resize(np.asarray(OPERATORS), count)
    rng.shuffle(op)
    operators[positive] = op
    cap = int(np.ceil(count/len(available)))
    return zeros, refs, operators, cap


def mutual_information(mask, label):
    table = pd.crosstab(mask, label)
    p = table.to_numpy(np.float64) / len(mask)
    independent = p.sum(1)[:, None] * p.sum(0)[None, :]
    use = p > 0
    return float(np.sum(p[use] * np.log(p[use]/independent[use])))


def materialize(source, backgrounds, pool, role, regime):
    from ops.r16_forward_data import choose_geometry, generate_operator, channel_zero_mean
    from ops.r16_support_temporal_data import natural_geometry, mask_bits
    from pipeline.sampler_r9 import smooth_eq
    # Background and observation pattern are exactly shared across labels.
    frame = backgrounds.iloc[np.repeat(np.arange(len(backgrounds)), 5)].reset_index(drop=True)
    n = len(frame)
    zeros, refs, operators, reuse_cap = schedule(pool, n, role)
    a = {'db': np.empty((n,15,25),np.float32), 'cell_share': np.empty((n,15,25),np.float32),
         'share': np.empty(n,np.float64), 'presence': (~zeros).astype(np.int64),
         'reference_index': refs, 'operator': operators, 'explicit_no_source': zeros,
         'sample_id': np.asarray([f'R17C_{role}_{i:06d}' for i in range(n)]),
         'pair_background_id': np.repeat(np.arange(len(backgrounds)),5),
         'site': frame.site.to_numpy(dtype=str), 'recorder': frame.recorder.to_numpy(dtype=str),
         'sensor_group': frame.sensor_group.to_numpy(np.int64),
         'base_window_index': frame.base_window_index.to_numpy(np.int64),
         'background_group': frame.background_group.to_numpy(dtype=str),
         'family': np.full(n,'no_source',dtype='<U40'), 'active_minutes': np.zeros(n,np.int64),
         'selected_minutes': np.zeros(n,np.int64), 'crop_begin': np.zeros(n,np.int64),
         'window_start': np.zeros(n,np.int64),
         'reference_weak_class': np.full(n,'no_source',dtype='<U32')}
    lookup = pool.set_index('reference_row_index')
    start = time.perf_counter()
    max_raw = max_round = 0.
    for i, row in frame.iterrows():
        bg = np.asarray(source.background(row), np.float64)
        rng = np.random.default_rng([SEED, sum(role.encode()), i, 0x4348])
        geom_rng = np.random.default_rng([SEED, sum(role.encode()), i, 0x4745])
        native = None if zeros[i] else source.reference_minutes[int(refs[i])]
        geometry = None if zeros[i] else (natural_geometry(bg,native,geom_rng) if regime=='H0' else choose_geometry(bg,native,geom_rng))
        eq = np.zeros(25) if zeros[i] else smooth_eq(rng,6.)
        post = np.zeros(25) if zeros[i] else channel_zero_mean(rng,6. if role=='train' else 12.)
        snr = float(rng.uniform(-10,10)); sub = rng.uniform(-10,10,2)
        value, receipt = generate_operator(bg,native,geometry,eq,post,snr,sub,str(operators[i]),zero=bool(zeros[i]))
        E,A,B = value['mixture_observed_power'],value['added_observed_power'],value['background_observed_power']
        if not np.array_equal(E,B+A) or abs(A.sum()/E.sum()-value['share'])>1e-12:
            raise AssertionError('Unrounded E=B+A or observed-share identity failed')
        if bool(value['presence']) != (not bool(zeros[i])):
            raise AssertionError('A positive insertion vanished behind a missing minute')
        for key in ['db','cell_share','share']:
            a[key][i] = value[key]
        a['active_minutes'][i] = np.any(value['cell_share']>0,axis=1).sum()
        if geometry:
            for key, old in [('selected_minutes','selected_minutes'),('crop_begin','crop_begin_minute'),('window_start','window_start_minute'),('family','family')]:
                a[key][i] = geometry[old]
            a['reference_weak_class'][i] = str(lookup.loc[int(refs[i])].weak_class)
        max_raw = max(max_raw,float(receipt['raw_pool_identity_error']))
        max_round = max(max_round,float(receipt['rounded_pool_gap']))
        if i and i%3000==0:
            print(json.dumps({'role':role,'regime':regime,'rows':i,'seconds':time.perf_counter()-start}),flush=True)
    a['effective_mask_bits'] = mask_bits(a['db'])
    a['n_obs'] = np.isfinite(a['db']).all(2).sum(1)
    if not np.array_equal(a['share']>0,~zeros):
        raise AssertionError('Declared and physically observed presence differ')
    if any(v.dtype.kind=='O' for v in a.values()):
        raise AssertionError('Object NPZ forbidden')
    observed = np.isfinite(a['db'])
    if np.any(a['cell_share'][~observed] != 0):
        raise AssertionError('Nonzero true cell share in missing cells')
    quantization_error = np.max(abs(a['db'][observed]*10 - np.round(a['db'][observed]*10)))
    if quantization_error > .0002:
        raise AssertionError('0.1dB output lattice differs')
    table = pd.DataFrame({'role':role,'regime':regime,'mask_bits':a['effective_mask_bits'],
                          'n_obs':a['n_obs'],'presence':a['presence'],'site':a['site'],'recorder':a['recorder']})
    strata = table.groupby(['site','recorder','mask_bits','n_obs']).presence.agg(['size','sum']).reset_index()
    strata['positive_fraction'] = strata['sum']/strata['size']
    if not np.all(strata['sum'].to_numpy()*5 == strata['size'].to_numpy()*4):
        raise AssertionError('Full observation pattern conditional prevalence differs')
    mi = mutual_information(a['effective_mask_bits'],a['presence'])
    if abs(mi)>1e-14:
        raise AssertionError('Empirical I(O;presence) is not zero')
    strata.to_csv(REPORT / f'{role}_{regime}_mask_label_strata.csv',index=False)
    use = pd.Series(refs[~zeros]).value_counts().sort_index()
    if len(use)!=465 or use.max()>reuse_cap:
        raise AssertionError('Reference coverage or per-role reuse cap differs')
    reuse = use.rename('uses').rename_axis('reference_index').reset_index()
    reuse['role']=role;reuse['regime']=regime
    reuse.to_csv(REPORT / f'{role}_{regime}_reference_reuse.csv',index=False)
    path = OUT / f'{role}_{regime}.npz'
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f,**a)
    path.with_suffix('.tmp').replace(path)
    with np.load(path,allow_pickle=False) as reread:
        if any(not np.array_equal(a[k],reread[k],equal_nan=True) if a[k].dtype.kind in 'fc' else not np.array_equal(a[k],reread[k]) for k in a):
            raise AssertionError('Saved data round trip differs')
    return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'n':n,'unique_backgrounds':len(backgrounds),
            'positive_rows':int((~zeros).sum()),'zero_rows':int(zeros.sum()),'share_mean':float(a['share'].mean()),
            'share_rms':float(np.sqrt(np.mean(a['share']**2))), 'reference_clips':len(use),
            'reference_reuse_min':int(use.min()),'reference_reuse_max':int(use.max()),'reference_reuse_cap':reuse_cap,
            'reference_concentration_Kish_ESS':float(use.sum()**2/np.square(use).sum()),
            'background_reuse_per_regime':5,'background_concentration_Kish_ESS':len(backgrounds),
            'full_mask_presence_MI_nats':mi,'nobs_presence_MI_nats':mutual_information(a['n_obs'],a['presence']),
            'every_site_recorder_mask_stratum_positive_fraction':.8,'post_mask_accidental_zero_count':0,
            'active_positive_minutes_p50':float(np.median(a['active_minutes'][~zeros])),
            'max_raw_pool_identity_error':max_raw,'max_rounded_pool_gap':max_round,
            'quantization_error_max':float(quantization_error),'bytes':path.stat().st_size,
            'seconds':time.perf_counter()-start}


def main():
    from pipeline.dataset_r9 import R9Sources
    from pipeline.r16_multisupport_data import guard
    sys.addaudithook(guard)
    if (OUT/'manifest.json').exists():
        raise FileExistsError('Completed R17 clean version is immutable')
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    source = R9Sources(verify_banks=False)
    pool, reference_proof = reference_pool(source)
    frames, background_proof = prepare_backgrounds()
    protocol = {'version':'r17_clean_v1','seed':SEED,'scope':'CPU controlled clean-classifier refit, not full neural training',
                'rows_per_background':5,'explicit_zero_fraction':.2,'all_full_mask_patterns_label_balanced':True,
                'reference_sampling':'uniform balanced shuffled cycles over all465; hard max ceil(n_positive/465)',
                'operator_sampling':'balanced six frozen R16 operators','regimes':['D0','H0'],
                'no_time_warp_or_flat_synthetic_long':True,'physical_targets_from_unrounded_E_B_A':True,
                'db_quantization_decimals':1,'Test_used':False,'new_kfold_training':False,
                'reference_proof':reference_proof,'background_proof':background_proof,
                'code_sha256':sha(Path(__file__))}
    write(OUT/'protocol.json',protocol);write(REPORT/'PROTOCOL_BEFORE_GENERATION.json',protocol)
    roles={}
    for name, frame in frames.items():
        roles[name]={}
        for regime in ['D0','H0']:
            roles[name][regime]=materialize(source,frame,pool,name,regime)
            write(OUT/'progress.json',{'completed':roles})
    manifest={**protocol,'status':'completed','roles':roles,'source_identity':source.identity,
              'reference_independence_limited':'Reserved44 clip bytes are excluded. Hull/day/transitive graph independence is not asserted after pool expansion.',
              'paired_background_dependence':'Five rows per background are clustered, not five independent observations.',
              'heldout_roles':'Background-source/24h disjoint, reference465 shared; report as background-generalization not unseen-reference.',
              'legacy_models_selection_leakage':'Changing this fit cannot erase legacy LightGBM early-stopping use of the old local_fit.'}
    write(OUT/'manifest.json',manifest);write(REPORT/'DATA_RECEIPT.json',manifest)
    print(json.dumps({'completed':str(OUT/'manifest.json'),'roles':roles},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
