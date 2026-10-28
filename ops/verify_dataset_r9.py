#!/usr/bin/env python3
"""R9 data-only numeric/source/worker fixture. No fitting or query access."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import tempfile
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pipeline.dataset_r9 import (
    R9Sources, DynamicR9Dataset, FixedR9Dataset, make_schedule, schedule_audit,
    array_sha, canonical, sha, validate_bundle,
)
from ops.prepare_r9_training import verify_cache


def _hash_rows(args):
    dataset, indices = args
    result=[]
    for index in indices:
        raw=dataset.raw_item(index);grid=dataset[index]
        result.append((index, {key:array_sha(np.asarray(raw[key])) for key in
                               ('db','unrounded_db','share','presence','cell_share','base_window_index')},
                       array_sha(grid['x']),array_sha(grid['w'])))
    return result


def run(config,output,bundle=None):
    start=time.monotonic()
    output=Path(output)
    if output.exists():raise FileExistsError('Fresh verification report required')
    pins={n:sha(ROOT/n) for n in ['pipeline/dataset_r9.py','ops/prepare_r9_training.py','pipeline/sampler_r9.py',
                                 'pipeline/hard_negatives_r6_v2.py','pipeline/relative_grid_r6_v2.py']}
    # Synthetic schedule fixture: all primaries once, no duplication to fill
    # a scarce hard-negative quota, exact 20% zeros, all509 references.
    hard=np.zeros(1000,bool);hard[::17]=True
    a=make_schedule(1000,1,0,20261026,hard_negative_candidates=hard)
    b=make_schedule(1000,1,0,20261026,hard_negative_candidates=hard)
    c=make_schedule(1000,2,0,20261026,hard_negative_candidates=hard)
    if (not all(np.array_equal(a[k],b[k]) for k in a) or all(np.array_equal(a[k],c[k]) for k in a)
            or a['explicit_no_source'].sum()!=200 or a['hard_negative'].sum()!=hard.sum()
            or np.any(a['hard_negative']&~a['explicit_no_source'])
            or np.any(a['hard_negative']&~a['hard_negative_candidate'])
            or len(np.unique(a['reference_index'][~a['explicit_no_source']]))!=509):
        raise AssertionError('Numeric schedule invariants failed')
    sources=R9Sources(config)
    cache=verify_cache(sources)
    dataset=DynamicR9Dataset(config,0,sources=sources)
    identity=canonical(dataset.identity)
    indices=list(range(8))
    expected=_hash_rows((dataset,indices))
    # Pickle keeps numerical minute cache, reopens only read-only bank mmap.
    copied=pickle.loads(pickle.dumps(dataset,protocol=5))
    if copied.sources._background is not None or _hash_rows((copied,indices))!=expected:
        raise AssertionError('Dataset pickle replay failed')
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        parts=list(pool.map(_hash_rows,[(dataset,indices[j::4]) for j in range(4)]))
    actual=sorted([row for part in parts for row in part])
    if actual!=expected:raise AssertionError('4 spawned worker row/grid outputs differ')
    dataset.set_epoch(2)
    changed=_hash_rows((dataset,indices))
    if changed==expected or canonical(dataset.identity)!=identity:
        raise AssertionError('Different epoch must change views, never immutable identity')
    dataset.set_epoch(1)
    if _hash_rows((dataset,list(reversed(indices))))!=list(reversed(expected)):
        raise AssertionError('Indexed access must not depend on call/batch order')
    # A true no-source hard negative must retain only the actual background.
    hard_index=int(np.flatnonzero(dataset.schedule['hard_negative'])[0])
    row=dataset.raw_item(hard_index)
    if row['share']!=0 or row['presence']!=0 or np.any(row['cell_share']):
        raise AssertionError('TRAIN hard-negative known addition must be zero')
    bundle_result=None
    if bundle:
        manifest=validate_bundle(bundle,verify_all=True)
        accesses=[]
        for fold in range(5):
            for role in ('selection','ensemble_fit','ensemble_gate'):
                ds=FixedR9Dataset(bundle,fold,role)
                old={k:array_sha(v) for k,v in ds.raw_arrays().items()}
                ds[0]
                if old!={k:array_sha(v) for k,v in ds.raw_arrays().items()}:
                    raise AssertionError('Query transform mutated frozen role')
                accesses.append(dict(fold=fold,role=role,rows=len(ds)))
            try:FixedR9Dataset(bundle,fold,'report')
            except ValueError:pass
            else:raise AssertionError('Report must refuse missing global freeze gate')
        bundle_result={'source_hash':manifest['source_hash'],'role_reads':accesses,
                       'report_numeric_arrays_read_by_verifier':False,'report_without_gate_rejected':True}
    sources.assert_unchanged()
    if any(sha(ROOT/n)!=value for n,value in pins.items()):raise AssertionError('Source code changed during fixture')
    report={'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
            'fixture_only':True,'model_training_or_GPU_or_Test_or_waveform_access':False,
            'source_acoustic_values_read':['fold0 TRAIN backgrounds','509 admitted VTUAD2s slices'],
            'prepared_nonreport_roles_read_for_integrity':bool(bundle),
            'source_identity':sources.identity,'code_sha256':pins,
            'synthetic_schedule':schedule_audit(a),'direct_cached_parity':cache,
            'actual_dynamic':dataset.epoch_audit(),
            'pickle_replay':True,'spawned_cpu_workers':4,'worker_examples':8,
            'spawn4_outputs_bitwise_equal':True,'same_epoch_replay':True,'different_epoch_changes':True,
            'batch_and_access_order_independent':True,'immutable_dataset_identity':True,
            'known_A_zero_hard_negative_verified':True,
            'bundle_verification':bundle_result,'elapsed_seconds':time.monotonic()-start}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','report':str(output),'sha256':sha(output),
                      'elapsed_seconds':report['elapsed_seconds']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/r9_full_training_v1.json')
    p.add_argument('--output',default='reports/R9_DATASET_WORKER_VERIFICATION.json')
    p.add_argument('--bundle')
    args=p.parse_args();run(args.config,args.output,args.bundle)
