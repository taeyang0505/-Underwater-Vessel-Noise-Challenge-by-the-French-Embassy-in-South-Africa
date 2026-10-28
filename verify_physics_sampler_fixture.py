"""Independent synthetic-only support/EQ/blend/full-pass checks, no training."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from pipeline.native_sampler_v2 import ReferenceRecord
from pipeline.augmentation_v2 import solve_reference_scale as legacy_solve, UnobservedReferenceError
from pipeline.physics_sampler_v3 import (PhysicsAwareSampler, PhysicsAugmentationConfig,
    solve_reference_scale, smooth_reference_eq, apply_reference_eq)


def rejected(fn):
    try: fn()
    except (ValueError, UnobservedReferenceError): return
    raise AssertionError('Expected explicit rejection')


def verify_energy(values):
    db, y, z, q, recipe = values
    o = np.isfinite(db); peak = np.nanmax(db.astype('float64'), (1, 2), keepdims=True)
    e = np.where(o, 10 ** ((db - peak) / 10), 0)
    restored = (e * q).sum((1, 2)) / e.sum((1, 2))
    assert np.allclose(restored, y, atol=3e-6)
    assert np.isfinite(y).all() and np.isfinite(q).all() and (q[~o] == 0).all()
    assert np.array_equal(z, (y > 0).astype('float32'))
    assert (o.all(2).sum(1) >= 3).all()
    for k, item in enumerate(recipe['items']):
        assert np.isclose(y[k], item['desired_share'], atol=1e-6)
        assert np.isclose(y[k], item['effective_share'], atol=1e-6)
        assert not (item['requested_share'] > 0 and y[k] == 0)
        active = (q[k] > 0).any(1) & o[k].all(1)
        assert np.isclose(active.sum() / o[k].all(1).sum(), item['support_coverage'])
        if item['cap_applied']: assert item['effective_share'] <= item['support_cap'] + 1e-10
        if y[k] > 0:
            assert abs(10 * np.log10(float(y[k]) / (1 - float(y[k]))) - item['global_snr_db']) < 2e-3
            a_minutes = (e[k] * q[k]).sum(1)
            b_minutes = (e[k] * (1 - q[k])).sum(1)
            for j, recorded in enumerate(item['minute_snr_db']):
                if recorded is not None:
                    recovered = 10 * np.log10(a_minutes[j] / b_minutes[j])
                    assert abs(recovered - recorded) < 2e-3
                    assert recorded <= item['max_minute_snr_db'] + 1e-10
                    assert recovered <= item['max_minute_snr_db'] + 2e-3
            assert item['max_minute_snr_db_actual'] <= 20 + 1e-10
        ids = [r['reference_index'] for r in item['references']]
        assert len(ids) == len(set(ids))
        assert all(max(abs(np.asarray(r['reference_eq_db']))) <= 4 + 1e-10 for r in item['references'])
    json.dumps({**recipe, 'background_index': recipe['background_index'].tolist()}, allow_nan=False)


def main():
    rng = np.random.default_rng(17)
    b = rng.uniform(.1, 4, (15, 25)); r = np.zeros((15, 25)); r[3] = rng.uniform(.1, 1, 25)
    observed = np.ones((15, 25), bool)
    old = legacy_solve(b, r, observed, .9)
    new = solve_reference_scale(b, r, observed, .9)
    assert old[0] == new[0] and np.array_equal(old[1].added, new[1].added)
    alpha, capped, audit = solve_reference_scale(b, r, observed, .9, support_cap=True, single_short=True, return_audit=True)
    expected_cap = .25 + .73 / 15
    assert np.isclose(capped.share, expected_cap) and audit['cap_applied']
    gains, _ = smooth_reference_eq(rng)
    eq = apply_reference_eq(r, gains)
    assert np.allclose(apply_reference_eq(eq, -gains), r, rtol=1e-12)
    fixed_alpha_changed = legacy_solve(b, r, observed, .5)[0]
    a2, mixture = solve_reference_scale(b, eq, observed, .5)
    assert np.isclose(mixture.share, .5) and not np.isclose(a2, fixed_alpha_changed)
    rejected(lambda: solve_reference_scale(b, np.zeros_like(r), observed, .4, support_cap=True, single_short=True))
    # A reference-minute/background ratio of 1e6 gives 60dB at alpha=1.
    stress_b = np.ones((15,25)); stress_b[7] = 1e-6
    stress_r = np.ones((15,25))
    stress_u = float(stress_r.sum() / (stress_b.sum() + stress_r.sum()))
    legacy_alpha, _ = legacy_solve(stress_b, stress_r, observed, stress_u)
    before_snr = float(10 * np.log10(legacy_alpha / 1e-6))
    capped_alpha, stress_mix, stress_audit = solve_reference_scale(stress_b, stress_r, observed, stress_u,
        max_minute_snr_db=20., return_audit=True)
    assert np.isclose(before_snr,60.) and np.isclose(capped_alpha,1e-4)
    assert stress_audit['minute_snr_cap_applied'] and 0 < stress_mix.share < stress_u
    assert stress_audit['max_minute_snr_db_actual'] <= 20 + 1e-10
    assert np.isclose((stress_mix.cell_share * stress_mix.energy_weights).sum(),stress_mix.share)
    rejected(lambda: solve_reference_scale(np.zeros_like(b), stress_r, observed, .5, max_minute_snr_db=20.))
    no_background_minute = stress_b.copy(); no_background_minute[7] = 0
    rejected(lambda: solve_reference_scale(no_background_minute, stress_r, observed, .5, max_minute_snr_db=20.))
    quantized_zero = stress_b.copy(); quantized_zero[7] = 1e-310
    rejected(lambda: solve_reference_scale(quantized_zero, stress_r, observed, .5, max_minute_snr_db=20.))
    x = rng.normal(70, 4, (7, 15, 25)).astype('float32')
    meta = pd.DataFrame({'base_window_index': np.arange(7), 'site': ['a','b','a','b','c','c','a'],
        'split': ['train','train','train','train','excluded_boundary_candidate','validation','train'],
        'observed_minutes': [15]*7, 'boundary_candidate_flag': [False,False,False,False,True,False,False],
        'background_group': ['g0','g1','g2','g3','bad','held','g6']})
    refs = [ReferenceRecord('native', 'NOAA_fixture', 'n', 'train', rng.uniform(.01, 2, (450,25)),
                           verified=True, quality_full_band_good=True)]
    refs += [ReferenceRecord('short'+str(i), 'HMS_fixture', 's'+str(i), 'train', rng.uniform(.01, 2, (10,25)),
                             verified=True, quality_full_band_good=True) for i in range(3)]
    cfg = PhysicsAugmentationConfig(zero_addition_probability=0., min_positive_share=.85, max_positive_share=.85,
        background_blend_probability=1., whole_minute_dropout_probability=0.)
    order = np.array([3,0,6,1,2])
    one = PhysicsAwareSampler(x, meta, refs, seed=112, config=cfg)
    result = one.draw(len(order), background_indices=order)
    verify_energy(result)
    assert np.array_equal(result[4]['background_index'], order)
    assert all(i['secondary_background_index'] in order for i in result[4]['items'])
    assert all(i['background_blend_applied'] for i in result[4]['items'])
    assert all(i['route'] in ['high_share_native','high_share_dispersed_multi'] for i in result[4]['items'])
    dispersed = PhysicsAwareSampler(x, meta, refs, seed=13,
        config=replace(cfg, native_long_fraction=0., background_blend_probability=0.)).draw(1, background_indices=[0])
    verify_energy(dispersed)
    di = dispersed[4]['items'][0]
    assert di['route'] == 'high_share_dispersed_multi' and len(di['references']) == 3
    anchors = [q['requested_anchor_minute'] for q in di['references']]
    assert len(set(anchors)) == 3 and max(anchors) - min(anchors) >= 2
    replay = PhysicsAwareSampler(x, meta, refs, seed=112, config=cfg).draw(len(order), background_indices=order)
    for left, right in zip(result[:4], replay[:4]): assert np.array_equal(left, right, equal_nan=True)
    rejected(lambda: one.draw(1, background_indices=[0]))
    rejected(lambda: PhysicsAwareSampler(x, meta, refs, seed=2).draw(1, background_indices=[4]))
    # Single very short source: high requested share must be capped, never relabelled zero.
    short = PhysicsAwareSampler(x, meta, refs[1:2], seed=7, config=replace(cfg, background_blend_probability=0.), require_native_long=False)
    short_result = short.draw(1, background_indices=[0]); verify_energy(short_result)
    assert short_result[4]['items'][0]['cap_applied'] and short_result[1][0] <= .25 + .73 * 2 / 15 + 1e-6
    # Three disjoint observed-minute blocks: no secondary can create observations by filling NaNs.
    disjoint = np.full((3,15,25), np.nan, dtype='float32')
    disjoint[0,:3] = 60; disjoint[1,3:6] = 60; disjoint[2,6:9] = 60
    dm = pd.DataFrame({'base_window_index':[0,1,2], 'site':['a','b','c'], 'split':['train']*3,
                       'observed_minutes':[3]*3, 'boundary_candidate_flag':[False]*3})
    failed_blend = PhysicsAwareSampler(disjoint, dm, refs, seed=2, config=cfg).draw(1, background_indices=[0])
    verify_energy(failed_blend)
    assert not failed_blend[4]['items'][0]['background_blend_applied']
    assert np.array_equal(np.isfinite(failed_blend[0][0]), np.isfinite(disjoint[0]))
    # Compatible blend uses the exact mask intersection, never its union.
    partial = x[:2].copy(); partial[0,:2] = np.nan; partial[1,-2:] = np.nan
    pm = meta.iloc[:2].copy(); pm['observed_minutes'] = 13
    inter = PhysicsAwareSampler(partial, pm, refs, seed=2,
        config=replace(cfg, time_reverse_probability=0.)).draw(1, background_indices=[0])
    verify_energy(inter)
    assert np.array_equal(np.isfinite(inter[0][0]), np.isfinite(partial[0]) & np.isfinite(partial[1]))
    report = {'passed':True, 'synthetic_fixture_only':True, 'actual_data_read':False, 'model_training_executed':False,
        'checks':['legacy four-argument numerical equivalence','single-short support cap','reference EQ inverse and alpha recomputation',
          'positive overlap rejection','clean explicit primary full pass','same-clean-train secondary only','energy/q/y/SNR identity',
          'dispersed multi or native high-share route','unique reference clips without repetition','first-batch exact replay',
          'reject boundary/foreign/reused explicit primary','zero-overlap blend skipped','blend mask intersection, not imputation',
          '60dB extreme bounded to20dB in every minute','minute SNR list matches final reversed/gained outputs',
          'zero minute background and positive-to-float32-zero explicitly rejected'],
        'single_minute_cap_for_15_observed':expected_cap,
        'extreme_uncapped_minute_snr_db':before_snr,
        'extreme_capped_minute_snr_db':stress_audit['max_minute_snr_db_actual'],
        'extreme_capped_effective_share':stress_mix.share,
        'routes':[i['route'] for i in result[4]['items']] + [di['route']],
        'legacy_code_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['pipeline/native_sampler_v2.py','pipeline/augmentation_v2.py']}}
    Path('reports/physics_sampler_v3_fixture.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
