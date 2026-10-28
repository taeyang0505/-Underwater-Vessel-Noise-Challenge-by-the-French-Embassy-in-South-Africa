"""Versioned natural-support/HF-negative R16 pilot data. No models or Test.

D0/D1 share reference draws, positive backgrounds, zero slots and channel
draws. D1 changes only natural crop support and matched zero backgrounds.
The 119-clip selected graph is conditionally source-disjoint; the graph of
the full 509 archive is connected and is NOT claimed to support this split.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, sys, time, warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops.r16_forward_data import (OPERATORS, allocate, canonical, channel_zero_mean,
    choose_geometry, dump, generate_operator, guard, qstats, reference_partition,
    safe, sha)

OUT = ROOT / 'reports/r16_support_temporal_r1/data'
OLD = ROOT / 'reports/r16_cross_operator_r1/data'
VERSION = 'r16_support_temporal_data_v1'
SEED = 202610282149
LONG_TRAIN = 'VT_R4_REF_474fe510b4e0aa460cad'
LONG_HOLD = 'VT_R4_REF_de02fbd6bbb753c153bd'
STRATA = ['site', 'sensor_group', 'recorder', 'effective_mask_bits']


def mask_bits(db):
    o = np.isfinite(db)
    if np.isinf(db).any() or (o.any(2) != o.all(2)).any():
        raise ValueError('Whole observed/NaN minutes required')
    return (o.all(2).astype(np.int64) * (1 << np.arange(15))).sum(1)


def hf_bump(db):
    """Sum linear powers in fixed 1.995..7.943 kHz bands, max minus median."""
    v = np.asarray(db, np.float64)[:, :, 18:25]
    valid = np.isfinite(v).all(2)
    natural = np.where(np.isfinite(v), v * np.log(10.) / 10., -np.inf)
    p = np.logaddexp.reduce(natural, axis=2) * 10. / np.log(10.)
    p = np.where(valid, p, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmax(p, axis=1) - np.nanmedian(p, axis=1)


def features_for_frame(source, frame):
    from pipeline.source_split_r6 import apply_minute_qc
    parts = []
    for start in range(0, len(frame), 512):
        f = frame.iloc[start:start+512]
        db = apply_minute_qc(np.asarray(source.background_bank()[f.base_window_index.to_numpy(int)]),
                             f.boundary_mask_bits.to_numpy(np.int64))
        parts.append((mask_bits(db), hf_bump(db)))
    a = frame.copy().reset_index(drop=True)
    a['effective_mask_bits'] = np.concatenate([x[0] for x in parts])
    a['hf_bump_2_8_db'] = np.concatenate([x[1] for x in parts])
    if not np.array_equal(a.effective_mask_bits.map(int.bit_count), a.qc_observed_minutes):
        raise AssertionError('QC provenance and effective masks differ')
    return a


def new_reference_split(source):
    r, original = reference_partition(source)
    r = r.rename(columns={'r16_reference_partition': 'legacy_r16_reference_partition'})
    r['r16_reference_partition'] = r.legacy_r16_reference_partition
    r.loc[r.reference_component.eq(LONG_TRAIN), 'r16_reference_partition'] = 'A'
    assert sorted(r[r.reference_component.eq(LONG_TRAIN)&r.minute_count.ge(12)].minute_count.tolist()) == [19, 20, 23]
    assert r[r.reference_component.eq(LONG_HOLD)&r.minute_count.ge(12)].minute_count.tolist() == [12]
    for col in ('hull_id', 'recording_day', 'sha256', 'reference_component'):
        if r.groupby(col).r16_reference_partition.nunique().max() != 1:
            raise AssertionError('Reference source identity crosses new partitions: ' + col)
    import pandas as pd
    overlaps = 0
    for i, a in r.iterrows():
        for _, b in r.iloc[i+1:].iterrows():
            if (a.r16_reference_partition != b.r16_reference_partition
                and pd.Timestamp(a.utc_begin) < pd.Timestamp(b.utc_end)
                and pd.Timestamp(b.utc_begin) < pd.Timestamp(a.utc_end)):
                overlaps += 1
    if overlaps:
        raise AssertionError('New reference partitions overlap in UTC')
    proof = dict(original)
    proof.update(version=VERSION, partition_rule='Legacy ABC plus predeclared whole component C-to-A; B unchanged',
        moved_component=LONG_TRAIN, independent_native12_component=LONG_HOLD,
        partition_counts={str(k): {'clips': len(g), 'components': g.reference_component.nunique(),
          'native_minutes': {str(int(x)): int(n) for x, n in g.minute_count.value_counts().sort_index().items()}}
          for k, g in r.groupby('r16_reference_partition')},
        legacy_C_results_no_longer_reference_independent=True,
        legacy_R11_native15_reference_seen_diagnostic_only=True,
        unseen_native_ge15_components_remaining=0,
        independent_long_generalization_claim_limited_to_native12=True)
    full = source.references
    proof['full509_native_minute_histogram'] = {str(int(x)): int(n) for x, n in full.native_complete_minute_count.value_counts().sort_index().items()}
    proof['selected_long_sources'] = r[r.minute_count >= 11][['clip_id', 'minute_count', 'reference_component',
         'recording_day', 'hull_id', 'legacy_r16_reference_partition', 'r16_reference_partition']].to_dict('records')
    return r, proof


def background_gaps(frames):
    """900-second support gap, not merely start-clock distance."""
    import pandas as pd
    proof = []
    for i, (a, fa) in enumerate(frames.items()):
        for b, fb in list(frames.items())[i+1:]:
            ids = len(set(fa.base_window_index) & set(fb.base_window_index))
            groups = len(set(fa.background_group) & set(fb.background_group))
            gaps = []
            for sensor, ga in fa.groupby('sensor_group'):
                gb = fb[fb.sensor_group.eq(sensor)]
                if not len(gb):
                    continue
                x = pd.to_datetime(ga.start_export_clock).astype('int64').to_numpy()/1e9
                y = np.sort(pd.to_datetime(gb.start_export_clock).astype('int64').to_numpy()/1e9)
                pos = np.searchsorted(y, x)
                delta = np.minimum(abs(x-y[np.clip(pos-1, 0, len(y)-1)]), abs(x-y[np.clip(pos, 0, len(y)-1)]))
                gaps.extend((np.maximum(delta-900., 0.)/3600.).tolist())
            gap = min(gaps) if gaps else None
            if ids or groups or (gap is not None and gap < 24.-1e-8):
                raise AssertionError('Background cross-role source/purge violation')
            proof.append(dict(a=a, b=b, source_id_overlap=ids, group_overlap=groups,
                              minimum_support_gap_hours=gap, no_shared_sensor=not bool(gaps)))
    return proof


def matched_hf_swap(source, train_frame, zero):
    """TRAIN conditional q90 only; exact four-way strata; no replacement."""
    import pandas as pd
    full = features_for_frame(source, source.role_metadata(0, 'train'))
    full['stratum_HF_q90_db'] = full.groupby(STRATA, sort=True).hf_bump_2_8_db.transform(lambda x: x.quantile(.9))
    full['hf_candidate'] = full.hf_bump_2_8_db > full.stratum_HF_q90_db
    blocked = set(train_frame.base_window_index.astype(int))
    cand = full[full.hf_candidate & ~full.base_window_index.isin(blocked)].copy()
    pools = {}
    for key, g in cand.groupby(STRATA, sort=True):
        records = g.to_dict('records')
        stable_key=[str(key[0]),int(key[1]),str(key[2]),int(key[3])]
        pools[key] = sorted(records, key=lambda x: canonical([SEED, 'HF', stable_key, int(x['base_window_index'])]))
    lookup = full.set_index('base_window_index')
    out = train_frame.copy().reset_index(drop=True)
    original = lookup.loc[out.base_window_index, STRATA].reset_index(drop=True)
    receipts = []
    for i in np.flatnonzero(zero):
        current = lookup.loc[int(out.at[i, 'base_window_index'])]
        key = tuple(current[k] for k in STRATA)
        pool = pools.get(key, [])
        if not pool:
            receipts.append(dict(row=int(i), replaced=False, reason='exact_stratum_train_q90_shortfall'))
            continue
        new = pool.pop(0)
        old = int(out.at[i, 'base_window_index'])
        for col in out.columns:
            if col in new:
                out.at[i, col] = new[col]
        receipts.append(dict(row=int(i), replaced=True, original_base_window_index=old,
            base_window_index=int(new['base_window_index']), original_hf_bump_db=float(current.hf_bump_2_8_db),
            new_hf_bump_db=float(new['hf_bump_2_8_db']), stratum_q90_db=float(new['stratum_HF_q90_db']),
            site=str(key[0]), sensor_group=int(key[1]), recorder=str(key[2]), effective_mask_bits=int(key[3])))
    after = lookup.loc[out.base_window_index, STRATA].reset_index(drop=True)
    if not original.equals(after) or out.base_window_index.duplicated().any():
        raise AssertionError('Exact matching or no-replacement invariant failed')
    if not out.loc[~zero].equals(train_frame.loc[~zero]):
        raise AssertionError('A positive background was changed')
    n = sum(x['replaced'] for x in receipts)
    full.to_parquet(OUT/'TRAIN_hf_source_features.parquet', index=False)
    dump(OUT/'HF_SWAP_RECEIPTS.json', receipts)
    return out, dict(requested=int(zero.sum()), replaced=n, shortfall=int(zero.sum())-n,
        exact_site_sensor_recorder_effective_mask_preserved=True, positive_backgrounds_unchanged=True,
        replacement_source_ids_unique=True, replacement_ids_overlap_original_train_rows=0,
        threshold_scope='Fold0 TRAIN within site/sensor/recorder/effective-mask; strictly above q90',
        source_candidates=int(full.hf_candidate.sum()), known_A_zero_not_natural_ship_absence=True,
        all_source_rows_examined=len(full), HF_scope_bands='18..24 = 1.995..7.943 kHz')


def joint_train_schedule(refs, n):
    """Metadata-only support-stratum draw shared by D0 and D1."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 0x534348]))
    zero = np.zeros(n, bool); zero[rng.choice(n, n//5, replace=False)] = True
    positive = np.flatnonzero(~zero)
    bins = np.array(['short']*1120 + ['medium']*1280 + ['long']*800, dtype='<U8')
    if len(positive) != len(bins):
        raise ValueError('Predeclared 4000-row pilot required')
    rng.shuffle(bins)
    idx = np.full(n, -1, np.int64); label = np.full(n, 'zero', dtype='<U8')
    for name, condition in [('short', refs.minute_count.between(1,4)),
                            ('medium', refs.minute_count.between(5,10)), ('long', refs.minute_count.ge(11))]:
        available = refs.loc[condition, 'reference_row_index'].to_numpy(int)
        if not len(available):
            raise ValueError('No real native source for support stratum: '+name)
        rows = positive[bins == name]; seq = []
        while len(seq) < len(rows):
            seq.extend(rng.permutation(available).tolist())
        idx[rows] = seq[:len(rows)]; label[rows] = name
    operator = np.full(n, 'control_eq6', dtype='<U32')
    assignment = np.resize(np.asarray(OPERATORS), len(positive)); rng.shuffle(assignment)
    operator[positive] = assignment
    return dict(zero=zero, reference_index=idx, support_stratum=label, operator=operator)


def natural_geometry(bg, native, rng, *, preserve_all=False):
    n = len(native)
    # Preserve all native minutes for <=10-minute recordings. Long recordings
    # contribute only real contiguous 11..15-minute slices, never interpolation.
    length = n if n <= 10 or (preserve_all and n <= 15) else int(rng.integers(11, min(15,n)+1))
    begin = int(rng.integers(n-length+1))
    part = native[begin:begin+length]
    legal = [s for s in range(16-length) if np.where(np.isfinite(bg)[s:s+length], part, 0.).sum() > 0]
    if not legal:
        raise ValueError('No observed natural support; no retry or label rewrite')
    return dict(crop_begin_minute=begin, selected_minutes=length,
        window_start_minute=int(rng.choice(legal)), family='natural_uncropped' if n==length else 'natural_long_contiguous',
        full_native_minutes=n, legal_observed_placements=len(legal),
        placement_conditioning='enumerate valid observed placements once; no target conditioning',
        source_repetition=False, time_warp=False, added_bell_or_fade=False)


def write_dataset(source, frame, refs, role, name, *, schedule=None, natural=False, long_only=False, zero_only=False):
    """One raw-data NPZ; all six operators remain the frozen R16 laws."""
    from pipeline.sampler_r9 import smooth_eq
    n = len(frame); lookup = refs.set_index('reference_row_index'); rng = np.random.default_rng([SEED, len(role), sum(role.encode())])
    if schedule is None:
        zero = np.zeros(n, bool)
        if not long_only and not zero_only:
            zero[rng.choice(n, n//5, replace=False)] = True
        if zero_only:
            zero[:] = True
        available = refs.reference_row_index.to_numpy(int); seq = []
        if (~zero).any() and not len(available):
            raise ValueError('Positive role has no reference')
        while len(seq) < int((~zero).sum()):
            seq.extend(rng.permutation(available).tolist())
        idx = np.full(n, -1, np.int64); idx[~zero] = seq[:int((~zero).sum())]
        op = np.full(n, name if name in OPERATORS else 'control_eq6', dtype='<U32')
        schedule = dict(zero=zero, reference_index=idx, operator=op, support_stratum=np.full(n, 'evaluation'))
    d = allocate(n); receipts = []; components = []; families = []; selected = []
    for i, row in frame.iterrows():
        local = np.random.default_rng(np.random.SeedSequence([SEED, sum(role.encode()), i, 0x4348414E]))
        geometry_rng = np.random.default_rng(np.random.SeedSequence([SEED, sum(role.encode()), i, 0x47454F4D]))
        bg = np.asarray(source.background(row), float); zero = bool(schedule['zero'][i])
        index = int(schedule['reference_index'][i]); native = None if zero else source.reference_minutes[index]
        geom = None if zero else (natural_geometry(bg, native, geometry_rng, preserve_all=long_only) if natural or long_only
                                 else choose_geometry(bg, native, geometry_rng, report_crop=role in ('local_report','domain_holdout')))
        eq = np.zeros(25) if zero else smooth_eq(local, 6.)
        post = np.zeros(25) if zero else channel_zero_mean(local, 6. if role=='train' else 12.)
        snr = float(local.uniform(-10.,10.)); sub = local.uniform(-10.,10.,2)
        value, rec = generate_operator(bg, native, geom, eq, post, snr, sub, str(schedule['operator'][i]), zero=zero)
        for k, v in value.items():
            d[k][i] = v
        component = '' if zero else str(lookup.loc[index].reference_component)
        components.append(component); families.append('no_source' if zero else geom['family'])
        selected.append(0 if zero else int(geom['selected_minutes']))
        receipts.append(dict(row=i, pair_id=f'r16_st_{role}_{i:05d}', background_index=int(row.base_window_index),
            reference_index=index, reference_component=component, support_stratum=str(schedule['support_stratum'][i]),
            geometry=geom, channel_seed=[SEED,sum(role.encode()),i,0x4348414E], receipt=rec))
    ids=np.asarray([x['pair_id'] for x in receipts])
    d.update(pair_id=ids, sample_id=np.asarray([x+'__'+name for x in ids]),
        base_window_index=frame.base_window_index.to_numpy(np.int64), reference_index=schedule['reference_index'],
        reference_component=np.asarray(components), site=frame.site.astype(str).to_numpy(dtype=str),
        recorder=frame.recorder.astype(str).to_numpy(dtype=str), sensor_group=frame.sensor_group.to_numpy(np.int64),
        background_group=frame.background_group.astype(str).to_numpy(dtype=str), family=np.asarray(families),
        explicit_no_source=schedule['zero'], operator=schedule['operator'], selected_minutes=np.asarray(selected,np.int64),
        support_stratum=schedule['support_stratum'], effective_mask_bits=mask_bits(d['db']))
    positive=d['presence']>0; active=(d['cell_share']>0).any(2).sum(1)
    if not np.array_equal(positive,~schedule['zero']) or any(a.dtype.kind=='O' for a in d.values()):
        raise AssertionError('Label/no-pickle invariant failed')
    if not np.allclose(d['mixture_observed_power'],d['background_observed_power']+d['added_observed_power'],rtol=0,atol=0):
        raise AssertionError('E=B+A differs')
    if np.max(abs(d['added_observed_power'].sum((1,2))/d['mixture_observed_power'].sum((1,2))-d['share']))>1e-12:
        raise AssertionError('Physical target changed')
    folder=OUT/role;folder.mkdir(parents=True,exist_ok=True);path=folder/(name+'.npz')
    np.savez_compressed(path,**d);dump(folder/(name+'_recipes.json'),receipts)
    with np.load(path,allow_pickle=False) as z:
        if any(z[k].dtype.kind=='O' for k in z.files):raise AssertionError('No-pickle reload failed')
    counts=lambda x:{str(int(k)):int(v) for k,v in zip(*np.unique(x,return_counts=True))}
    return dict(path=str(path.relative_to(OUT)), scope='bundle', sha256=sha(path), n=n,
        statistics=dict(share=qstats(d['share']),positive_count=int(positive.sum()),zero_count=int((~positive).sum()),
          native_selected_positive_minutes=counts(d['selected_minutes'][positive]),actual_positive_active_minutes=counts(active[positive]),
          actual_positive_support_median=float(np.median(active[positive])) if positive.any() else None,
          actual_positive_support_ge12=int((active[positive]>=12).sum()),
          max_raw_pool_error=max(x['receipt']['raw_pool_identity_error'] for x in receipts),
          max_rounded_pool_gap=max(x['receipt']['rounded_pool_gap'] for x in receipts),
          max_rounding_bound=max(x['receipt']['rounding_bound'] for x in receipts)))


def _entry(manifest, role, name):
    if role=='train':return manifest['training'][name]
    if role=='long_validation':return manifest['long_validation'][name]
    return manifest['panels'][role][name]


def _freeze(manifest_path, receipt):
    if receipt is None:raise PermissionError('GLOBAL_FREEZE required for C-prime report/long inputs')
    r=json.loads(Path(receipt).read_text())
    if (r.get('status')!='completed' or r.get('all_models_frozen')is not True
        or r.get('ensemble_policy_frozen')is not True or r.get('data_manifest_sha256')!=sha(manifest_path)
        or not r.get('artifact_sha256')):raise PermissionError('Incomplete or wrong freeze binding')
    for p,h in r['artifact_sha256'].items():
        if sha(safe(ROOT,p))!=h:raise ValueError('Frozen checkpoint/policy artifact changed')


def load_panel(manifest_path, role, operator, *, freeze_receipt=None):
    mp=Path(manifest_path);m=json.loads(mp.read_text());ready=json.loads((mp.parent/'readiness.json').read_text())
    if m.get('status')!='completed' or ready.get('ready')is not True or ready['manifest_sha256']!=sha(mp):
        raise ValueError('Immutable complete experiment data required')
    if role in ('local_report','domain_holdout','long_validation'):_freeze(mp,freeze_receipt)
    e=_entry(m,role,operator);p=safe(ROOT if e.get('scope')=='project' else mp.parent,e['path'])
    if sha(p)!=e['sha256']:raise ValueError('Input artifact SHA differs')
    with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    if len(d['share'])!=e['n'] or not np.array_equal(d['share']>0,d['presence'].astype(bool)):
        raise ValueError('Physical targets/schema differ')
    return d


def load_training(manifest_path, arm='D0'):
    if arm not in ('D0','D1'):raise ValueError('D0/D1 required')
    return load_panel(manifest_path,'train',arm)


def load_long_panel(manifest_path, name, operator=None, *, freeze_receipt=None):
    key=name+'__'+operator if operator else name
    return load_panel(manifest_path,'long_validation',key,freeze_receipt=freeze_receipt)


def fixture():
    rng=np.random.default_rng(17);bg=np.full((15,25),90.);native=np.exp(rng.normal(size=(23,25)))
    geom=natural_geometry(bg,native,rng)
    assert 11<=geom['selected_minutes']<=15 and not geom['time_warp'] and not geom['source_repetition']
    zeros=np.full((2,15,25),90.);zeros[1,0]=np.nan
    assert mask_bits(zeros).tolist()==[32767,32766]
    for op in OPERATORS:
        a,r=generate_operator(bg,native,geom,np.zeros(25),np.zeros(25),0.,[0.,0.],op)
        assert a['presence']==1 and np.max(abs(a['cell_share']))<=1 and r['raw_pool_identity_error']<1e-10
    return dict(status='passed',natural_contiguous_support=True,quantized_db_raw_physical_targets=True,
                model_training_or_Test_access=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','fixture']);a=ap.parse_args();sys.addaudithook(guard)
    if a.action=='fixture':print(json.dumps(fixture()));return
    if OUT.exists():raise FileExistsError('Fresh experiment path only; do not reuse partial results')
    OUT.mkdir(parents=True);start=time.perf_counter();own=sha(__file__)
    plan=dict(version=VERSION,seed=SEED,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      D0='new A-prime joint native-length reference draws; old 25percent-natural-full/otherwise1..6 geometry',
      D1='same reference IDs; natural whole short/medium and contiguous11..15-minute real long crops + exact-stratum HF zero swaps',
      paired_invariants=['positive base background','reference ID','explicit20percent zero slots','operator draw','EQ/SNR seeds'],
      train_rows_per_arm=4000,positive_native_length_stratum_fractions=dict(short=.35,medium=.40,long=.25),
      target_share_forcing=False,artificial_flat_or_warp_or_repeat=False,
      relative_features_only_model_agent_responsibility=True,Test_EDA_or_LB_target_matching=False,
      original_R16_source_data_models_untouched=True,new_pilot_not_production=True,
      long_split_limitation='Only one native>=15 component goes TRAIN; independent native12 component holds out; unseen>=15 unavailable',
      old_R11_reference_seen_after_new_training=True, report_domain_long_require_GLOBAL_FREEZE=True,
      data_builder_model_fitting=False)
    dump(OUT/'PLAN.json',plan);dump(OUT/'FIXTURE.json',fixture())
    from pipeline.dataset_r9 import R9Sources
    import pandas as pd
    source=R9Sources();r,referenceproof=new_reference_split(source);r.to_parquet(OUT/'reference_partitions.parquet',index=False)
    # Persist operational scalar metadata only. Arrow list-valued provenance
    # remains in its pinned parent parquet, avoiding pandas scalar-assignment
    # coercion of one-element ndarray cells during background swaps.
    metadata_columns=['window_id','base_window_index','site','recorder','sensor_group',
        'start_export_clock','recording_day','recording_week_start','background_group',
        'boundary_mask_bits','qc_observed_minutes']
    frames={role:pd.read_parquet(OLD/(role+'_backgrounds.parquet'),columns=metadata_columns) for role in
       ('train','local_fit','local_gate','local_report','domain_holdout')}
    refs={p:r[r.r16_reference_partition.eq(p)].copy() for p in ('A','B','C')}
    schedule=joint_train_schedule(refs['A'],len(frames['train']))
    newtrain,HFproof=matched_hf_swap(source,frames['train'],schedule['zero'])
    for role,f in frames.items():f.to_parquet(OUT/(role+'_backgrounds.parquet'),index=False)
    newtrain.to_parquet(OUT/'train_D1_backgrounds.parquet',index=False)
    gaps=background_gaps({**frames,'train':pd.concat([frames['train'],newtrain]).drop_duplicates('base_window_index')})
    dump(OUT/'SOURCE_AUDIT.json',dict(reference=referenceproof,HF_negative=HFproof,background_crossrole=gaps))
    pins={**source.code_sha256,'ops/r16_forward_data.py':sha(ROOT/'ops/r16_forward_data.py'),
       'ops/r15_forward_prototype.py':sha(ROOT/'ops/r15_forward_prototype.py'),
       str((OLD/'manifest.json').relative_to(ROOT)):sha(OLD/'manifest.json'),
       referenceproof['source_manifest']:referenceproof['source_sha256']}
    for role in frames:
        p=OLD/(role+'_backgrounds.parquet');pins[str(p.relative_to(ROOT))]=sha(p)
    for role in ('train','ensemble_fit','ensemble_gate','report'):
        p=source.bg_dir/'fold_00'/(role+'_manifest.parquet');pins[str(p.relative_to(ROOT))]=sha(p)
    m=dict(status='completed',version=VERSION,code_sha256=own,plan_sha256=sha(OUT/'PLAN.json'),
      source_identity=source.identity,source_pins=pins,training={},panels={},long_validation={},
      reference_counts=referenceproof['partition_counts'],Test_accessed=False,
      report_rows_for_parameter_selection=False,reference_unseen_for_legacy_R11=False)
    for arm,f,natural in [('D0',frames['train'],False),('D1',newtrain,True)]:
        m['training'][arm]=write_dataset(source,f,refs['A'],'train',arm,schedule=schedule,natural=natural)
        print(json.dumps(dict(stage='training_data',arm=arm,statistics=m['training'][arm]['statistics'],seconds=time.perf_counter()-start)),flush=True)
    with np.load(OUT/m['training']['D0']['path'],allow_pickle=False)as z:a0={k:z[k] for k in ('reference_index','presence','base_window_index','operator','effective_mask_bits')}
    with np.load(OUT/m['training']['D1']['path'],allow_pickle=False)as z:a1={k:z[k] for k in a0}
    for key in ('reference_index','presence','operator','effective_mask_bits'):
        if not np.array_equal(a0[key],a1[key]):raise AssertionError('D0/D1 pairing failed: '+key)
    if not np.array_equal(a0['base_window_index'][a0['presence']>0],a1['base_window_index'][a1['presence']>0]):raise AssertionError('Positive pairing failed')
    oldm=json.loads((OLD/'manifest.json').read_text())
    for role in ('local_fit','local_gate'):
        m['panels'][role]={}
        for op,e in oldm['panels'][role].items():
            p=OLD/e['path']
            if sha(p)!=e['sha256']:raise ValueError('Original B evaluation changed')
            m['panels'][role][op]={**e,'path':str(p.relative_to(ROOT)),'scope':'project','references_partition':'B unchanged'}
    for role in ('local_report','domain_holdout'):
        m['panels'][role]={}
        for op in OPERATORS:m['panels'][role][op]=write_dataset(source,frames[role],refs['C'],role,op)
        print(json.dumps(dict(stage=role,panels=6,seconds=time.perf_counter()-start)),flush=True)
    # Rebuilt C-prime long panel: real 12-minute reference only, never stretched.
    reportfull=features_for_frame(source,source.role_metadata(0,'report'))
    dense=reportfull[reportfull.qc_observed_minutes.ge(13)].copy()
    dense['_order']=[canonical([SEED,'long_dense',int(x)]) for x in dense.base_window_index]
    longbg=dense.sort_values('_order').head(400).drop(columns='_order').reset_index(drop=True)
    longrefs=refs['C'][refs['C'].reference_component.eq(LONG_HOLD)&refs['C'].minute_count.eq(12)]
    for op in OPERATORS:
        m['long_validation']['native12_positive__'+op]=write_dataset(source,longbg,longrefs,'long_validation',op,natural=True,long_only=True)
    cutoff=float(reportfull.hf_bump_2_8_db.quantile(.9));hf=reportfull[reportfull.hf_bump_2_8_db>cutoff].copy()
    hf['_order']=[canonical([SEED,'report_hf',int(x)]) for x in hf.base_window_index]
    hf=hf.sort_values('_order').head(400).drop(columns='_order').reset_index(drop=True)
    sf=reportfull[reportfull.site.eq('7 St_Francis')&reportfull.qc_observed_minutes.ge(13)].copy().reset_index(drop=True)
    empty=refs['C'].iloc[:0]
    for name,f in [('hf_high_negative',hf),('st_francis_dense_negative',sf)]:
        e=write_dataset(source,f,empty,'long_validation',name,zero_only=True)
        e.update(class_balance_evaluation='Use class-balanced0.5/0.5 sample weights paired with native12; no duplicated evaluation rows')
        m['long_validation'][name]=e
    extended_report=pd.concat([frames['local_report'],longbg,hf,sf]).drop_duplicates('base_window_index')
    extra_gaps=background_gaps({**frames,'train':pd.concat([frames['train'],newtrain]).drop_duplicates('base_window_index'),
                                'local_report':extended_report})
    dump(OUT/'EXTENDED_LONG_BACKGROUND_PURGE_QA.json',dict(status='passed',pairs=extra_gaps,
      report_and_all_extra_long_negative_backgrounds_checked=True))
    dump(OUT/'LONG_PANEL_DEFINITION.json',dict(positive=400,HF_negative=len(hf),SF_dense_negative=len(sf),
      reference_component=LONG_HOLD,real_native_minutes=12,unseen_ge15_available=False,
      hf_threshold_db=cutoff,threshold_scope='heldout report backgrounds only, evaluation definition; never TRAIN augmentation threshold',
      threshold_rule='strictly above report-pool q90; no Test statistics',
      long_HF_and_SF_class_weights='each class has total weight0.5; unequal available counts disclosed',
      long_and_negative_sources='same existing report background role; 900-second/24-hour purged against TRAIN',
      previous_development_exposure=True,independent_for_new_Aprime_models_only=True))
    for p,h in pins.items():
        if sha(ROOT/p)!=h:raise AssertionError('Frozen source changed: '+p)
    if sha(__file__)!=own:raise AssertionError('Data builder changed during build')
    m['elapsed_seconds']=time.perf_counter()-start
    m['artifact_sha256']={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')if p.is_file()}
    dump(OUT/'manifest.json',m)
    dump(OUT/'readiness.json',dict(ready=True,manifest_sha256=sha(OUT/'manifest.json'),version=VERSION,
      source_checks_passed=True,paired_data_checks_passed=True,model_training_performed=False,
      production_ready=False,report_policy_freeze_required=True))
    dump(OUT/'DATA_QA.json',dict(status='passed',manifest_sha256=sha(OUT/'manifest.json'),D0_D1_pairing=True,
      source_graph_and_hull_day_time_disjoint=True,background24hour_support_purge=True,
      raw_physical_labels_and_rounding_bound=True,exact_zero_fraction=.2,HF_negative=HFproof,
      heldout_long_native_minutes=12,no_independent_ge15_source=True,no_Test_or_model_fitting=True,
      source_pin_count=len(pins),elapsed_seconds=m['elapsed_seconds']))
    print(json.dumps(dict(status='completed',manifest_sha256=sha(OUT/'manifest.json'),elapsed_seconds=m['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
