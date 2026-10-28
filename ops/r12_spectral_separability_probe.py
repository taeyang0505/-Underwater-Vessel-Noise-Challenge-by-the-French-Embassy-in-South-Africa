"""Frozen scalar spectral probe; no model/basis fit, Test or new eval access."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
              'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pipeline.features_r12 import load_spectral_basis, transform_channels
from pipeline.relative_grid_r6 import quantized_db

OUT = ROOT/'reports/r12_spectral_probe'
PANELS = ('full_smooth15', 'native_long15', 'st_francis_dense', 'hf_high')
SCORES = (
    'old8_profile_mid_minus_low_db',
    'old8_profile_mid_chord_excess_db',
    'old8_sqrtw_mid_energy_fraction',
    'new11_positive_PCA_residual_RMS_db',
    'new11_signed_PCA_residual_mid_mean_db',
    'new11_positive_local_curvature_RMS_db',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def scalar_features(db, basis):
    x, w = transform_channels(db, basis, include_shape=True)
    base, w8 = transform_channels(db, include_shape=False)
    if not np.array_equal(x[:, :8], base) or not np.array_equal(w, w8):
        raise AssertionError('Shape transform changed the old8 grid or physical weights')
    observed = w > 0
    count = observed.sum(1)
    # Existing profile is channel4; channel6 is q05-relative, not profile.
    profile = np.divide((x[:, 4].astype(float)*20).sum(1), count,
                        out=np.zeros_like(count, dtype=float), where=count > 0)
    low_contrast = profile[:, 10:20].mean(1)-profile[:, :9].mean(1)
    # Band index is uniformly spaced on log10 frequency. The fixed chord
    # joins index9 (251.2Hz) to22 (5011.9Hz), probing316.2..3981.1Hz.
    t = (np.arange(10, 22)-9)/13
    chord = profile[:, 9, None]*(1-t)+profile[:, 22, None]*t
    chord_excess = (profile[:, 10:22]-chord).mean(1)
    mid_energy = np.square(x[:, 7].astype(float))[:, :, 10:23].sum((1, 2))
    def reduce_map(values, bands, positive_rms):
        selected = values[:, :, bands]
        mask = observed[:, :, bands]
        denom = mask.sum((1, 2))
        if positive_rms:
            selected = np.maximum(selected, 0.)**2
        result = np.divide(np.where(mask, selected, 0.).sum((1, 2)), denom,
                           out=np.zeros(len(db)), where=denom > 0)
        return np.sqrt(result) if positive_rms else result
    pca_pos = reduce_map(x[:, 9].astype(float)*10, slice(10, 23), True)
    pca_signed = reduce_map(x[:, 8].astype(float)*10, slice(10, 23), False)
    curvature = reduce_map(x[:, 10].astype(float)*10, slice(10, 22), True)
    scores = np.stack([low_contrast, chord_excess, mid_energy, pca_pos,
                       pca_signed, curvature], axis=1)
    qdb = quantized_db(db)
    median = np.nanmedian(qdb, axis=(1, 2))
    reconstruction = x[:, 0].astype(float)*10+x[:, 4].astype(float)*20
    oracle_error = float(np.max(abs(reconstruction[observed]-(qdb-median[:, None, None])[observed])))
    if not np.isfinite(scores).all() or oracle_error > 1e-5:
        raise AssertionError('Finite scalar/information reconstruction oracle failed')
    return scores, {'old8_exact_prefix_and_weights': True,
        'old8_reconstruct_gain_centered_grid_max_error_db': oracle_error}


def auc(positive, negative):
    # Deliberately fixed direction, no inverse/max(AUC,1-AUC) selection.
    delta = positive[:, None]-negative[None, :]
    return float(np.mean(delta > 0)+.5*np.mean(delta == 0))


def main():
    if OUT.exists():
        raise FileExistsError('Fresh probe output only; no replacement or outcome retry')
    OUT.mkdir(parents=True)
    folder = ROOT/'reports/r11_diagnostics/full_support_hf_r1'
    basis_path = ROOT/'models/r12_shared_basis_v1/basis.npz'
    manifest_path = ROOT/'data/synthetic/r12_cpu_ablation_v3/manifest.json'
    inputs = [basis_path, manifest_path, Path(__file__),
        ROOT/'pipeline/features_r12.py', ROOT/'pipeline/relative_grid_r6.py',
        ROOT/'pipeline/relative_grid_r6_v2.py']
    inputs += [folder/(name+'_samples.npz') for name in PANELS]
    pins = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    plan = {'format': 'r12_frozen_spectral_scalar_probe_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'input_sha256': pins,
        'panels': list(PANELS), 'scores': list(SCORES),
        'score_direction': 'larger positive spectral excess is positive, fixed for every score/panel; never invert AUC',
        'safe_PCA_score_bands': list(range(10, 23)),
        'safe_PCA_score_centres_Hz': [float(10**(1.5+.1*j)) for j in range(10, 23)],
        'formulas': {
            SCORES[0]: 'mean(profile[10:20])-mean(profile[0:9]);profile=oldchannel4*20',
            SCORES[1]: 'mean(profile[10:22]-linear_logfrequency_chord(profile9,profile22))',
            SCORES[2]: 'sum(square(oldchannel7=sqrt(w))[:,:,10:23])',
            SCORES[3]: 'RMS positive TRAIN-PCA residual over observed t andbands10..22, newchannel9*10',
            SCORES[4]: 'mean signed TRAIN-PCA residual over observed t andbands10..22, newchannel8*10',
            SCORES[5]: 'RMS positive(2db[f]-db[f-1]-db[f+1]) over observed t,f10..21,newchannel10*10'},
        'analysis_scope': 'already-exposed fixed R11 challenges; scalar feature diagnostics, not fitted model performance or a new blind test',
        'PCA_refit_or_model_fit_or_new_val_gate_eval_or_Test_network': False,
        'threads': 1}
    # Saved BEFORE opening any panel values or computing separation statistics.
    write(OUT/'predeclared_plan.json', plan)
    manifest = json.loads(manifest_path.read_text())
    basis = load_spectral_basis(basis_path, expected_sha256=pins[basis_path.relative_to(ROOT).as_posix()],
                                expected_training_binding=manifest['basis_training_binding'])
    values, stats, oracle, targets = {}, {}, {}, {}
    for name in PANELS:
        with np.load(folder/(name+'_samples.npz'), allow_pickle=False) as z:
            db, target = z['db'], z['share']
        if name in PANELS[:2] and not np.all(target > 0):
            raise AssertionError('Expected known added positives')
        if name in PANELS[2:] and np.any(target != 0):
            raise AssertionError('Expected controlled A=0 negatives')
        values[name], oracle[name] = scalar_features(db, basis)
        targets[name] = {'n': len(db), 'known_added_share_mean': float(target.astype(float).mean()),
            'role': 'positive' if name in PANELS[:2] else 'known_A0_not_natural_vessel_absence'}
        stats[name] = {score: {'median': float(np.median(values[name][:, j])),
            'mean': float(values[name][:, j].mean()),
            'q10': float(np.quantile(values[name][:, j], .1)),
            'q90': float(np.quantile(values[name][:, j], .9))} for j, score in enumerate(SCORES)}
    table = []
    for positive in PANELS[:2]:
        for negative in PANELS[2:]:
            for j, score in enumerate(SCORES):
                table.append({'positive_panel': positive, 'negative_panel': negative,
                    'score': score, 'fixed_direction_AUC': auc(values[positive][:, j], values[negative][:, j]),
                    'positive_median': stats[positive][score]['median'],
                    'negative_median': stats[negative][score]['median'],
                    'positive_n': len(values[positive]), 'negative_n': len(values[negative])})
    with (OUT/'scalar_separation.csv').open('x') as f:
        w = csv.DictWriter(f, list(table[0])); w.writeheader(); w.writerows(table)
    np.savez_compressed(OUT/'per_window_scalar_scores.npz', **values,
                        score_names=np.asarray(SCORES))
    for relative, expected in pins.items():
        if sha(ROOT/relative) != expected:
            raise AssertionError('Pinned source/basis changed: '+relative)
    result = {'format': plan['format'], 'status': 'completed',
        'predeclared_plan_sha256': sha(OUT/'predeclared_plan.json'),
        'basis_file_sha256': sha(basis_path), 'basis_content_sha256': basis['basis_sha256'],
        'TRAIN_basis_windows': basis['unique_background_source_count'],
        'TRAIN_basis_complete_minute_spectra': basis['fitted_spectra'],
        'TRAIN_basis_safe_bands': basis['safe_band_indices'],
        'TRAIN_explained_variance_ratio': basis['explained_variance_ratio'],
        'TRAIN_explained_variance_sum': sum(basis['explained_variance_ratio']),
        'old93percent_claim_not_substituted_for_measured_state': True,
        'targets': targets, 'scalar_panel_statistics': stats, 'separation': table,
        'old8_information_oracle': oracle, 'old8_already_contains_gain_centered_db':
            'channel0*10+channel4*20=Q(db)-window_median at every observed cell, up tofloat32 rounding',
        'new11_is_deterministic_db_representation_not_new_observed_information': True,
        'model_or_basis_fit_or_model_inference': False, 'Test_new_validation_gate_eval_access': False,
        'all_original_inputs_unchanged': True,
        'limitations': [
            'AUC of a fixed scalar on known R11 panels is not a fitted model score or guaranteed energy-mask identification.',
            'Native long panel cycles7 measured references;500 windows are not500 independent long recordings.',
            'Positive background sites and actual SFdense negatives are not site-matched; fixed challenge domains/target laws differ.',
            'The rank3 85.07percent number is empirical for2000site-balanced TRAIN background windows/25937complete23bandmean-centered spectra, not a universal background law or proof of93percent.',
            'Known A=0 means no controlled addition, not natural ship absence. Whole added reference includes its ambient sound.',
            'For each cell A=q*E when true q is known. A=y*E is generally false because window y is only an energy-weighted average; it assumes q is spatially constant.',
            'Observed mixture E alone admits many nonnegative B/A decompositions; PCA residuals cannot create actual natural source truth.',
            'Positive-direction AUC is never reversed or selected by its result; all six prespecified scores andfourcrossings are preserved.']}
    write(OUT/'RESULTS.json', result)
    short_names = ['old8 mid−LF', 'old8 chord excess', 'old8 sqrt(w) mid mass',
                   'new11 positive PCA RMS', 'new11 signed PCA mean', 'new11 positive local curvature RMS']
    lines = ['# R12 고정 스펙트럼 스칼라 분리 진단', '',
        '기존 TRAIN PCA와 R11 고정 패널만 사용했다. 새 PCA fitting·모델 학습/추론·Test 또는 새 validation gate/eval 접근은 없었다. 모든 스칼라는 양의 excess가 양성이라는 방향을 사전 고정했고 AUC를 뒤집지 않았다.', '',
        f"실제 TRAIN2000/완전 분 스펙트럼{basis['fitted_spectra']:,}개에서 safe23대역·분별 공통평균 제거 후 rank3 설명분산 합은 **{100*sum(basis['explained_variance_ratio']):.4f}%**다. 과거93% 가정으로 대체하지 않는다.", '',
        '|고정 스칼라|full15/SF AUC|full15/HF AUC|native15/SF AUC|native15/HF AUC|',
        '|---|---:|---:|---:|---:|']
    for label, score in zip(short_names, SCORES):
        av = [next(r['fixed_direction_AUC'] for r in table if r['score']==score and r['positive_panel']==p and r['negative_panel']==n)
              for p in PANELS[:2] for n in PANELS[2:]]
        lines.append('|'+label+'|'+'|'.join(f'{v:.6f}' for v in av)+'|')
    lines += ['', '|고정 스칼라 중앙값|full15|native15|SF A=0|HF A=0|', '|---|---:|---:|---:|---:|']
    for label, score in zip(short_names, SCORES):
        lines.append('|'+label+'|'+'|'.join(f"{stats[p][score]['median']:.6f}" for p in PANELS)+'|')
    lines += ['', '## 정보와 정답의 한계', '',
        '실제 기존8채널의 profile은 channel4(/20), sqrt(w)는 channel7이다. channel6은 q05-relative다. 관측셀에서 `10*x0+20*x4=Q(db)−windowmedian`이므로 기존8채널만으로도 공통gain을 제거한 전체 grid를 복원할 수 있다. 새11채널은 기존db의 결정론적 좌표/귀납편향을 제공하며 새 관측 정보를 추가하는 것은 아니다. 잔차의 스칼라 분리 가능성이 신경망의 학습 성공이나 per-cell q 식별을 보장하지 않는다.', '',
        '정확한 합성 성분은 셀별 `A=q*E`다. 창 하나의 정답 비율 `y=sum(A)/sum(E)`를 모든 셀에 곱하는 `A=y*E`는 q가 모든 셀에서 같은 특수한 경우만 성립한다. 혼합 E만 관측하면 여러 비음수 B/A 분해가 가능하며, PCA 잔차를 실제 선박 정답으로 쓸 수 없다. 원참조에는 주변 배경도 포함돼 알려진 정답은 whole-recording 추가 에너지다.', '',
        'SF와 양성 배경의 사이트 구성은 동일하지 않고, native15는 실제7참조를 여러 배경에 순환한 패널이다. 위 표는 이미 노출된 R11 challenge의 특성 진단이며 새로운 blind 평가나 모델 성능표가 아니다. 어떤 스칼라도 결과에 따라 방향을 바꾸거나 생산 입력/정책으로 자동 채택하지 않았다.', '',
        '[모든 수치·입력 SHA](RESULTS.json) · [24개 비교 CSV](scalar_separation.csv) · [사전 계획](predeclared_plan.json)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'delivery_manifest.json', {'input_sha256': pins,
        'output_sha256': {p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
        'original_inputs_unchanged': True})
    print(json.dumps({'status': 'completed', 'output': str(OUT),
        'explained_variance_sum': result['TRAIN_explained_variance_sum'],
        'separation': table}, ensure_ascii=False))


if __name__ == '__main__':
    main()
