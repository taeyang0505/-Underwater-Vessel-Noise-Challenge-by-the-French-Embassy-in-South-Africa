"""Numeric-only R6-v2 physical features fixture; no dataset/model I/O."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np

from pipeline.physics_features_r6_v2 import (
    FEATURE_COUNT, FEATURE_NAMES, GROUP_BANDS, SIGNAL_NAMES,
    feature_specification, physics_features_r6_v2,
)


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def oracle(db):
    """Independent scalar/group loops, direct powers on ordinary fixture dB."""
    rows = []
    for window in np.round(db.astype('float64'), 1):
        groups = {}
        for name, bands in {**GROUP_BANDS, 'broad': tuple(range(25))}.items():
            series = []
            for minute in window:
                a = minute[list(bands)]
                series.append(10*np.log10(np.power(10., a/10).sum())
                              if np.isfinite(a).all() else np.nan)
            groups[name] = np.asarray(series)
        v, d, counts = [], [], []
        for name in ['low', 'mid', 'high', 'safe']:
            a = groups[name][np.isfinite(groups[name])]
            ok = len(a) >= 2
            v.append(float(a.max()-np.median(a)) if ok else 0.)
            d.append(ok); counts.append(len(a))
        for name in ['mid', 'high', 'safe']:
            pairs = np.isfinite(groups['low']) & np.isfinite(groups[name])
            a, b = groups['low'][pairs], groups[name][pairs]
            ok = len(a) >= 3 and np.square(a-a.mean()).sum() > 1e-20 and np.square(b-b.mean()).sum() > 1e-20
            v.append(float(np.corrcoef(a, b)[0, 1]) if ok else 0.)
            d.append(ok); counts.append(len(a))
        a = groups['broad'][np.isfinite(groups['broad'])]
        ok = len(a) >= 2
        v.append(float(a.std(ddof=0)) if ok else 0.)
        d.append(ok); counts.append(len(a))
        pairs = np.isfinite(groups['low']) & np.isfinite(groups['high'])
        a = groups['high'][pairs]-groups['low'][pairs]
        v.append(float(np.median(a)) if len(a) else 0.)
        d.append(bool(len(a))); counts.append(len(a))
        rows.append(v+d+counts)
    return np.asarray(rows, dtype='float32')


def run():
    rng = np.random.default_rng(20261026)
    db = rng.normal(100., 6., (12,15,25))
    db[1] = np.nan
    db[2] = np.nan; db[2,7] = 80.
    db[3,:,:9] = np.nan
    db[4,:,24] = np.nan      # high unavailable; safe remains available
    db[5,:,10] = np.nan      # mid unavailable; safe remains available
    db[6,1::2] = np.nan
    db[7,4,18] = np.nan      # one missing bin removes that minute, never partial-sums it
    db[8] = 100.             # constant series: corr undefined, bump/std valid zero
    db[9, :, :] = np.nan; db[9,0,24] = 70.  # no complete requested group
    db.setflags(write=False)
    before = db.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        f, info = physics_features_r6_v2(db, return_diagnostics=True)
    expected = oracle(db)
    assert f.shape == (12,27) and FEATURE_COUNT == len(FEATURE_NAMES) == 27
    assert f.dtype == np.float32 and np.isfinite(f).all()
    assert np.array_equal(db, before, equal_nan=True) and not db.flags.writeable
    oracle_error = float(np.max(np.abs(f-expected)))
    assert oracle_error <= 2e-6, oracle_error
    assert np.array_equal(f[:,9:18].astype(bool), info['defined'])
    assert np.array_equal(f[:,18:], info['valid_minute_counts'])
    assert not f[1].any() and not f[9].any()
    assert (f[2,:8] == 0).all() and (f[2,9:17] == 0).all()
    assert f[2,17] == 1 and (f[2,18:] == 1).all()
    assert info['group_complete_minutes']['high'][4] == 0
    assert info['group_complete_minutes']['safe'][4] == 15
    assert info['group_complete_minutes']['mid'][5] == 0
    assert info['group_complete_minutes']['safe'][5] == 15
    assert f[7,18+1] == f[7,18+3] == 14  # mid/safe complete minutes
    assert (f[8,4:7] == 0).all() and (f[8,13:16] == 0).all()
    assert (f[8,9:13] == 1).all() and f[8,16] == 1  # genuine constant temporal signals

    ramp = np.broadcast_to(100.+np.arange(15)[None,:,None], (1,15,25)).copy()
    ramp_f = physics_features_r6_v2(ramp)[0]
    assert np.allclose(ramp_f[:4], 7., atol=1e-6)
    assert np.allclose(ramp_f[4:7], 1., atol=1e-7)
    assert abs(ramp_f[7] - np.std(np.arange(15), ddof=0)) < 1e-6
    assert abs(ramp_f[8] - 10*np.log10(5/9)) < 1e-6
    antiphase = ramp.copy(); antiphase[:,:,20:] = 100.-np.arange(15)[None,:,None]
    anti_f = physics_features_r6_v2(antiphase)
    assert abs(float(anti_f[0,5])+1.) < 1e-7

    gain_audit = {}
    for gain in [-12., -6., .1, 6., 12.]:
        g = physics_features_r6_v2(db + gain)
        error = float(np.max(abs(g-f)))
        gain_audit[str(gain)] = error
        assert error <= 2e-6
    # Quantization is intentionally internal, not delegated to caller state.
    prequantized = physics_features_r6_v2(np.round(db,1))
    assert np.array_equal(prequantized,f)
    nongrid = physics_features_r6_v2(db+.037)
    nongrid_change = float(np.max(abs(nongrid[:,:9]-f[:,:9])))
    assert nongrid_change > 0
    # Frequency-dependent response is deliberately not an invariance guarantee.
    band_shift = physics_features_r6_v2(db + np.linspace(-6,6,25)[None,None,:])
    band_change = float(np.max(abs(band_shift[:,:9]-f[:,:9])))
    assert band_change > .01

    chunks = np.concatenate([physics_features_r6_v2(db[:1]),
                             physics_features_r6_v2(db[1:7]),
                             physics_features_r6_v2(db[7:])])
    assert np.array_equal(chunks,f)
    order = rng.permutation(len(db))
    assert np.array_equal(physics_features_r6_v2(db[order]),f[order])
    assert physics_features_r6_v2(np.empty((0,15,25))).shape == (0,27)
    reject = 0
    for bad in [{'db':db,'share':np.zeros(len(db))}, np.zeros((15,25)),
                np.zeros((1,15,25),dtype='int32'), np.full((1,15,25),np.inf),
                np.zeros((1,15,24))]:
        try:
            physics_features_r6_v2(bad)
        except (TypeError,ValueError):
            reject += 1
    assert reject == 5
    probe = subprocess.run([sys.executable,'-c',
        "import sys,json,numpy as np; from pipeline.physics_features_r6_v2 import physics_features_r6_v2; "
        "a=physics_features_r6_v2(np.full((1,15,25),100.)); "
        "assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules); "
        "print(json.dumps({'torch_loaded':False,'shape':list(a.shape)}))"],
        capture_output=True,text=True,check=True,timeout=20,cwd=ROOT)
    return {
        'format':'r6_physics_features_v2_numeric_fixture_v1',
        'status':'passed', 'fixture_only':True,
        'code_sha256':{n:sha(ROOT/n) for n in [
            'pipeline/physics_features_r6_v2.py','ops/verify_physics_features_r6_v2.py']},
        'specification':feature_specification(),
        'independent_scalar_oracle_max_abs':oracle_error,
        'common_decimal_grid_gain_max_abs':gain_audit,
        'nongrid_pre_round_gain_0p037_max_signal_change':nongrid_change,
        'per_band_static_offset_max_signal_change':band_change,
        'all_nan_single_minute_constant_and_partial_band_safe':True,
        'safe_group_valid_when_7943Hz_missing':True,
        'input_unchanged_and_readonly_supported':True,
        'chunk_permutation_bitwise':True,
        'hand_computed_ramp_and_negative_correlation':True,
        'no_Torch_import_probe':json.loads(probe.stdout),
        'source_datasets_Test_OUTER_or_TRAIN_opened':False,
        'model_fit_GPU_SSH_network_submission':False,
        'no_performance_or_AUC_improvement_claim':True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='reports/R6_PHYSICS_FEATURES_V2_FIXTURE.json')
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        raise FileExistsError('Fresh fixture report required: '+str(output))
    report = run()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','report':str(output),'sha256':sha(output),
                      'feature_count':FEATURE_COUNT}))


if __name__ == '__main__':
    main()
