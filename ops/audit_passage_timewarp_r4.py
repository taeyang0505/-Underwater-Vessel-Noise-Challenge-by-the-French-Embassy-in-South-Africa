"""Independent R4 power/timewarp and fixed-production-schedule QA; no real arrays.

Reads only the frozen R4 reference metadata and approved generation config.
The overlap-integral oracle and sampler fixture powers are synthetic.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from pipeline.passage_sampler_r4 import (
    PassageConfigR4, PassageReference, PassageSamplerR4, warp_band_power,
    make_passage_schedule, schedule_slice, raised_cosine_power_weights,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'reports/R4_TIMEWARP_INDEPENDENT_QA.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
        return
    p = Path(args[0]).resolve()
    if str(p).startswith(str(ROOT.resolve())) and (
        p.name.lower() == 'test.csv' or p.suffix.lower() in {'.wav', '.flac', '.npy', '.npz'}
    ):
        raise RuntimeError(f'Independent audit forbids real acoustic/challenge arrays: {p.name}')


def numeric_oracle():
    rng = np.random.default_rng(18411)
    cases = 0
    max_error = 0.
    max_mean_error = 0.
    for k in [60, 90, 150, 180, 225, 300, 449, 450, 570, 600, 690]:
        low, high = max(60, int(np.ceil(k/3))), min(450, 3*k)
        for m in sorted({low, high, min(max(k, low), high), (low+high)//2}):
            for kind in ['varying', 'constant', 'impulse']:
                if kind == 'varying':
                    s = rng.lognormal(0., 2., size=(k, 25))
                elif kind == 'constant':
                    s = np.tile(np.linspace(0., 2., 25), (k, 1))
                else:
                    s = np.zeros((k, 25)); s[[0, k//2, k-1]] = np.linspace(.1, 3., 25)
                s[:, 0] = 0.
                s.setflags(write=False)
                before = hashlib.sha256(s.tobytes()).hexdigest()
                result, audit = warp_band_power(s, m)
                # Independent interval-intersection oracle; no cumulative sums.
                edge = np.arange(m+1, dtype=float)*k/m
                left = np.arange(k, dtype=float)[None, :]
                overlap = np.maximum(0., np.minimum(left+1, edge[1:, None])-
                                     np.maximum(left, edge[:-1, None]))
                expected = overlap @ s / (k/m)
                err = float(np.max(np.abs(result-expected))/max(float(s.max()), 1.))
                merr = float(np.max(np.abs(result.mean(0)-s.mean(0)))/max(float(s.max()), 1.))
                assert err < 2e-11 and merr < 2e-12, (k, m, kind, err, merr)
                assert result.shape == (m, 25) and np.isfinite(result).all() and (result >= 0).all()
                assert before == hashlib.sha256(s.tobytes()).hexdigest()
                assert audit['source_coordinate_begin'] == 0 and audit['source_coordinate_end'] == k
                assert audit['source_coordinate_step'] == k/m and audit['timewarp_ratio'] == m/k
                assert np.allclose(overlap.sum(0), 1., atol=1e-12)  # each source bin exactly once
                if m == k: np.testing.assert_allclose(result, s, rtol=1e-10, atol=1e-10)
                cases += 1; max_error = max(max_error, err); max_mean_error = max(max_mean_error, merr)
    invalid = 0
    for s, m in [(np.ones((60, 25)), 181), (np.ones((690, 25)), 229),
                 (np.ones((60, 25)), 60.9), (np.full((60, 25), np.nan), 60),
                 (-np.ones((60, 25)), 60)]:
        try: warp_band_power(s, m)
        except (ValueError, OverflowError): invalid += 1
        else: raise AssertionError('Invalid timewarp accepted')
    fade_errors = []
    for m in [60, 180, 450]:
        w = raised_cosine_power_weights(m, 30.)
        error = abs(2*w.sum()-(2*m-1.25*30))
        assert error < 1e-9 and np.allclose(w, w[::-1], atol=1e-13)
        fade_errors.append(float(error))
    return {'passed': True, 'oracle_cases': cases, 'invalid_inputs_rejected': invalid,
            'max_overlap_oracle_scaled_absolute_error': max_error,
            'max_band_mean_scaled_absolute_error': max_mean_error,
            'fade_integral_max_absolute_error': max(fade_errors),
            'input_readonly_bytes_unchanged': True,
            'energy_scope': 'Band mean power conserved before fade/EQ; integrated power x derived duration changes by M/K.'}


def schedule_summary(n, seed, cfg):
    d = make_passage_schedule(n, seed, cfg)
    replay = make_passage_schedule(n, seed, cfg)
    u, m = d['requested_share'], d['rendered_segments']
    assert np.array_equal(u, replay['requested_share']) and np.array_equal(m, replay['rendered_segments'])
    positive = u > 0; p = int(positive.sum())
    assert (~positive).sum() == round(.2*n) and (m[~positive] == 0).all()
    assert (u[positive] >= .005).all() and (u[positive] < .95).all()
    assert (m[positive] >= 60).all() and (m[positive] <= 450).all()
    ranks = np.sort((u[positive]-.005)/(.95-.005))*p - np.arange(p)
    assert (ranks >= -1e-9).all() and (ranks < 1+1e-9).all()
    seconds = m[positive]*2
    coarse = (seconds >= 300).astype(int)
    fine = np.minimum(np.searchsorted([120, 300, 600, 900, 1380], seconds, side='right')-1, 3)
    means = [float(u[positive][coarse == k].mean()) for k in [0, 1]]
    summary = {'examples': n, 'schedule_seed': seed, 'positive_examples': p,
               'zero_count': int((~positive).sum()), 'zero_fraction': float((~positive).mean()),
               'share_mean': float(u.mean()), 'positive_share_mean': float(u[positive].mean()),
               'conditional_positive_share_means': means,
               'rendered_duration_mean_seconds': float(seconds.mean()),
               'coarse_fractions': (np.bincount(coarse, minlength=2)/p).tolist(),
               'fine_fractions': (np.bincount(fine, minlength=4)/p).tolist(),
               'LHS_positive_share_rank_strata_complete': True, 'exact_replay': True}
    return summary, u[positive], seconds


def fullsize_schedules(config, cfg):
    counts = [53244, 53315, 53522, 52998, 54758]
    folds = []; all_passed = True; max_conditional = 0.
    for fold, ntrain in enumerate(counts):
        summaries = {}; arrays = {}
        for role, split in enumerate(['train', 'validation', 'outer_validation']):
            n = ntrain if role == 0 else 3000
            seed = 20261025+fold*10007+role*1000003+170003
            summaries[split], u, seconds = schedule_summary(n, seed, cfg)
            arrays[split] = (u, seconds)
        a = summaries['train']; comparisons = {}
        for split in ['validation', 'outer_validation']:
            b = summaries[split]
            c = {'mean_share_difference': abs(a['share_mean']-b['share_mean']),
                 'zero_fraction_difference': abs(a['zero_fraction']-b['zero_fraction']),
                 'coarse_duration_total_variation': float(np.abs(np.array(a['coarse_fractions'])-b['coarse_fractions']).sum()/2),
                 'fine_duration_total_variation': float(np.abs(np.array(a['fine_fractions'])-b['fine_fractions']).sum()/2),
                 'rendered_duration_KS_statistic': float(ks_2samp(arrays['train'][1], arrays[split][1], method='asymp').statistic),
                 'rendered_duration_mean_difference_seconds': abs(a['rendered_duration_mean_seconds']-b['rendered_duration_mean_seconds']),
                 'maximum_conditional_mean_difference': max(abs(x-y) for x, y in zip(a['conditional_positive_share_means'], b['conditional_positive_share_means']))}
            c['gates_passed'] = all(c[k] <= t for k, t in config['distribution_tolerances'].items())
            all_passed &= c['gates_passed']; max_conditional = max(max_conditional, c['maximum_conditional_mean_difference'])
            comparisons[split] = c
        folds.append({'fold': fold, 'splits': summaries, 'comparison_to_train': comparisons})
    return {'passed': bool(all_passed), 'scope': 'Schedule only: no background/reference power arrays read or dataset generated.',
            'seed_attempts_per_split': 1, 'predeclared_tolerances': config['distribution_tolerances'],
            'maximum_conditional_mean_difference': max_conditional, 'folds': folds}


def sampler_fixtures(frames, cfg):
    # Natural-mask-shaped synthetic background; all powers below are fixture values.
    db = np.tile(np.linspace(32., 68., 25), (4, 15, 1)).astype('float32')
    db[1, [1, 3, 9]] = np.nan; db[2, :, ::4] = np.nan
    db[3, [0, 2, 5, 8, 11, 14]] = np.nan; db.setflags(write=False)
    max_target_error = 0.; examples = 0; positive = 0; coverage = []
    for fold, frame in enumerate(frames):
        for role, split in enumerate(['train', 'validation', 'outer_validation']):
            metadata = frame[frame.split.eq(split)].to_dict('records')
            refs = []; snapshots = {}
            for row in metadata:
                k = int(row['segment_count'])
                power = (1+.35*np.sin(np.linspace(0, 5, k)))[:, None]*np.linspace(.3, 1.7, 25)[None, :]
                power.setflags(write=False); snapshots[row['clip_id']] = hashlib.sha256(power.tobytes()).hexdigest()
                refs.append(PassageReference(row, power))
            assert all(any(1/3 <= m/int(row['segment_count']) <= 3 for row in metadata) for m in range(60, 451))
            coverage.append({'fold': fold, 'split': split, 'clips': len(metadata),
                             'components': len({row['reference_component'] for row in metadata}),
                             'all_391_rendered_lengths_have_eligible_source': True})
            bg = pd.DataFrame({'split': [split]*4, 'base_window_index': range(4), 'background_group': ['fixture_BG']*4})
            n = 48; seed = 99123+fold*53+role; indices = np.arange(n) % 4
            schedule = make_passage_schedule(n, seed+170003, cfg)
            sampler = PassageSamplerR4(db, bg, refs, split, seed, cfg)
            out = sampler.draw(n, indices, schedule)
            # Chunk boundaries must not alter reference RNG or schedule alignment.
            chunked = PassageSamplerR4(db, bg, refs, split, seed, cfg)
            a = chunked.draw(17, indices[:17], schedule_slice(schedule, 0, 17))
            b = chunked.draw(n-17, indices[17:], schedule_slice(schedule, 17, n))
            for i in range(4): np.testing.assert_array_equal(out[i], np.concatenate([a[i], b[i]]))
            assert out[4]['items'] == a[4]['items']+b[4]['items']
            x, y, z, q, recipe = out; mask = np.isfinite(x)
            assert np.array_equal(mask, np.isfinite(db[indices])) and np.all(q[~mask] == 0)
            assert len(x) == n and recipe['background_index'] == indices.tolist()
            assert np.array_equal(z, (schedule['requested_share'] > 0).astype('float32'))
            xx = x.astype(float); e = np.where(mask, 10**((xx-np.nanmax(xx, (1, 2), keepdims=True))/10), 0)
            recovered = (e*q).sum((1, 2))/e.sum((1, 2))
            error = float(np.max(np.abs(recovered-y))); assert error < 3e-6
            max_target_error = max(max_target_error, error)
            lookup = {row['clip_id']: row for row in metadata}
            for j, item in enumerate(recipe['items']):
                assert item['background_index'] == int(indices[j]) and not item['time_reversed']
                assert not item['hard_snr_or_support_cap_applied'] and not item['background_blending_applied']
                assert item['requested_share'] == float(schedule['requested_share'][j])
                assert abs(item['desired_share']-item['requested_share']) < 1e-12
                assert len(item['references']) == int(z[j]); positive += int(z[j])
                if not item['references']: continue
                r = item['references'][0]; parent = lookup[r['clip_id']]; k = int(parent['segment_count']); m = int(schedule['rendered_segments'][j])
                assert r['original_duration_seconds'] == parent['duration_seconds']
                assert r['source_utc_begin'] == str(parent['utc_begin']) and r['source_utc_end'] == str(parent['utc_end'])
                assert r['source_sha256'] == parent['sha256'] and r['measured_used_seconds'] == 2*k
                assert r['derived_duration_seconds'] == 2*m and r['source_coordinate_begin'] == 0 and r['source_coordinate_end'] == k
                assert r['timewarp_ratio'] == m/k and 1/3 <= m/k <= 3
                assert not r['original_UTC_modified'] and not r['waveform_repetition'] and not r['dB_interpolation']
                assert r['linear_power_timewarp_applied'] and max(map(abs, r['reference_eq_db'])) <= 4+1e-12
            assert all(snapshots[r.metadata['clip_id']] == hashlib.sha256(r.segments.tobytes()).hexdigest() for r in refs)
            # Fail closed on fractional and nonfinite externally supplied schedules.
            for bad in [60.9, np.nan, np.inf]:
                invalid = {'requested_share': np.array([.3]), 'rendered_segments': np.array([bad]), 'schedule_seed': 1, 'scheme': 'invalid fixture'}
                try: sampler.draw(1, [0], invalid)
                except ValueError: pass
                else: raise AssertionError('Noninteger/nonfinite supplied duration accepted')
            examples += n
    # A deliberate measured-band mismatch must fail once, never resample/retry.
    ref_meta = {'source': 'VTUAD_v1.3.2', 'split': 'train', 'clip_id': 'disjoint_fixture',
                'reference_component': 'fixture', 'sha256': '0'*64, 'duration_seconds': 360.,
                'reference_eligible': True, 'verified': True, 'measured': True, 'quality_full_band_good': True}
    power = np.zeros((180, 25)); power[:, 24] = 1.
    only = np.full((1, 15, 25), np.nan); only[:, :, 0] = 50.
    bg = pd.DataFrame({'split': ['train'], 'base_window_index': [0], 'background_group': ['fixture']})
    s = PassageSamplerR4(only, bg, [PassageReference(ref_meta, power)], 'train', 42, cfg)
    calls = []; original = s._reference
    def counting(*args): calls.append(1); return original(*args)
    s._reference = counting
    try: s.draw(1, [0], {'requested_share': [.3], 'rendered_segments': [180], 'schedule_seed': 1, 'scheme': 'one-shot'})
    except RuntimeError as error: assert 'overlap' in str(error)
    else: raise AssertionError('Disjoint positive support accepted')
    assert len(calls) == 1
    return {'passed': True, 'synthetic_examples': examples, 'positive_examples': positive,
            'all_15_fold_role_pools': coverage, 'max_saved_float32_target_reconstruction_error': max_target_error,
            'original_source_metadata_and_fixture_power_bytes_unchanged': True,
            'chunked_exact_replay': True, 'no_positive_to_zero': True,
            'deliberate_overlap_failure_reference_calls': len(calls), 'failed_rows_not_retried_or_relabelled': True}


def main():
    sys.addaudithook(guard)
    source = ROOT/'data/versions/vtuad_reference_r4_v1'
    config_path = ROOT/'configs/group_cv_passage_r4_data.json'
    code_path = ROOT/'pipeline/passage_sampler_r4.py'
    config = json.loads(config_path.read_text()); cfg = PassageConfigR4(**config['passage_config'])
    manifest = json.loads((source/'manifest.json').read_text())
    files = [source/'manifest.json', config_path, code_path]+[source/name for name in manifest['file_sha256']]
    before = {str(p.relative_to(ROOT)): sha(p) for p in files}
    for name, expected in manifest['file_sha256'].items(): assert sha(source/name) == expected
    frames = [pd.read_parquet(source/f'fold_{fold:02d}_reference_manifest.parquet') for fold in range(5)]
    assert all(len(frame) == 119 and frame.reference_component.nunique() == 22 for frame in frames)
    tree = ast.parse(code_path.read_text())
    loops = {node.name: [type(n).__name__ for n in ast.walk(node) if isinstance(n, (ast.While, ast.Continue))]
             for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in ['draw', '_reference']}
    assert not any(loops.values()), loops
    numeric = numeric_oracle(); schedules = fullsize_schedules(config, cfg); fixtures = sampler_fixtures(frames, cfg)
    after = {str(p.relative_to(ROOT)): sha(p) for p in files}
    assert before == after, 'Frozen code/config/source metadata changed during audit'
    report = {'passed': numeric['passed'] and schedules['passed'] and fixtures['passed'],
              'scope': 'Independent local data-only QA; overlap oracle and sampler powers are synthetic. Production-size schedules only.',
              'source_hash': manifest['source_hash'], 'frozen_file_sha256_before_and_after': before,
              'frozen_source_metadata_unchanged': True, 'numeric_timewarp_oracle': numeric,
              'production_size_fixed_schedules': schedules, 'sampler_fixtures': fixtures,
              'draw_reference_while_or_continue_nodes': loops,
              'real_acoustic_arrays_read': False, 'Test_calibration_sealed_arrays_read': False,
              'full_dataset_generated': False, 'model_training_executed': False,
              'limitation': 'Band-power timewarp does not generate waveform phase or Doppler and does not increase independent hull/day/source count. Fade/EQ deliberately alter power, with alpha and cell/overall targets recomputed.'}
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# R4 시간축 증강 독립 검증', '',
             f'전체 판정: **{report["passed"]}**. 실제 음향 배열이나 Test/calibration/sealed 배열은 읽지 않았습니다.', '',
             f'- 고정 원천: 119 clips / 22 selected components, source hash `{manifest["source_hash"]}`.',
             f'- 시간축 수치 oracle: {numeric["oracle_cases"]} cases, 최대 정규화 오차 {numeric["max_overlap_oracle_scaled_absolute_error"]:.3g}.',
             f'- 합성 fixture: {fixtures["synthetic_examples"]} examples / 15 fold-role pools, mask·parent UTC/길이/SHA 보존, 청크 재현 통과.',
             f'- 실제 예정 크기의 단일 고정 schedule: 모든 사전 gate **{schedules["passed"]}**; conditional-u 평균차 최대 {schedules["maximum_conditional_mean_difference"]:.6f} (기준 .05).',
             '- 시드 재시도, 예제 삭제, 실패 양성의 zero 재라벨링은 하지 않았습니다. 의도적 overlap 실패는 reference 호출 1회 후 실패했습니다.', '',
             '| Fold | Role | N | Zero | Coarse conditional-u mean difference | All predeclared gates |',
             '|---:|---|---:|---:|---:|---|']
    for f in schedules['folds']:
        for split, c in f['comparison_to_train'].items():
            s = f['splits'][split]
            lines.append(f'| {f["fold"]} | {split} | {s["examples"]} | {s["zero_count"]} | {c["maximum_conditional_mean_difference"]:.6f} | {c["gates_passed"]} |')
    lines += ['', '평균 band power 보존은 fade/EQ **전**의 timewarp에만 적용됩니다. 시간 적분 power×duration은 M/K만큼 바뀝니다. 새 파형의 위상·Doppler를 생성하거나 실제 독립 원천 수를 늘린다는 의미가 아닙니다.', '',
              '이 검증은 full-size schedule의 분포를 확인하며, 실제 배경 마스크에 따른 source 선택·숫자 검증은 production materialization 후 별도 전수 QA가 필요합니다.', '',
              '[전체 수치·해시](R4_TIMEWARP_INDEPENDENT_QA.json)']
    OUTPUT.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'passed': report['passed'], 'report': str(OUTPUT), 'sampler_sha256': before[str(code_path.relative_to(ROOT))],
                      'maximum_conditional_mean_difference': schedules['maximum_conditional_mean_difference'],
                      'oracle_cases': numeric['oracle_cases'], 'fixture_examples': fixtures['synthetic_examples']}, indent=2))
    if not report['passed']: raise SystemExit(2)


if __name__ == '__main__': main()
