"""R17 classifier-only feature ablations and immutable R17 audit summaries.

No model fitting, query prediction, Test labels, or old artifact mutations occur.
Modes compare the original 2,033 features, deletion of 25 pure observation
fractions, and deletion of all 400 features in the audit's broad name-based
``observation_mask_proxy`` category. The latter includes real audio deltas.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AUDIT = ROOT / 'reports/r17_eda_audit_20261030/shortcut'
OUTPUT = ROOT / 'reports/r17_clean_pipeline_r1'
MODES = ('full', 'direct', 'aggressive')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


@lru_cache(maxsize=1)
def original_names():
    """Rebuild actual feature order and check the completed audit schema."""
    from pipeline.trees_r16_production import schemas
    from ops.r16_temporal_features import temporal_statistics
    extra = temporal_statistics(np.ones((1, 15, 25), np.float32))[1]
    names = schemas()['classification'] + ['F1/' + name for name in extra]
    pinned = json.loads((AUDIT / 'tree_feature_names.json').read_text())['classification']
    if names != pinned or len(names) != 2033:
        raise ValueError('Actual R16 feature order differs from the audited 2,033-column contract')
    return tuple(names)


def classifier_indices(mode='aggressive'):
    if mode not in MODES:
        raise ValueError(f'Explicit classifier feature mode must be one of {MODES}')
    names = original_names()
    remove = np.asarray([
        False if mode == 'full' else
        name.startswith('observed_fraction/') if mode == 'direct' else
        'observed' in name
        for name in names
    ], bool)
    return np.flatnonzero(~remove)


def schema(mode='aggressive'):
    names = original_names()
    ix = classifier_indices(mode)
    keep = set(ix.tolist())
    selected = [names[i] for i in ix]
    return {
        'mode': mode,
        'source_feature_count': len(names),
        'feature_count': len(ix),
        'removed_count': len(names) - len(ix),
        'feature_names': selected,
        'removed_feature_names': [name for i, name in enumerate(names) if i not in keep],
        'audited_source_schema_sha256': sha256(AUDIT / 'tree_feature_names.json'),
        'selected_schema_sha256': hashlib.sha256(json.dumps(selected, separators=(',', ':')).encode()).hexdigest(),
        'pure_observation_fraction_removed': mode != 'full',
        'all_audit_observation_mask_proxy_names_removed': mode == 'aggressive',
        'no_indirect_missingness_information_guarantee': True,
        'caveat': 'Adjacent-observed delta and contiguous-observed lag are real audio statistics with mask-dependent validity. Deleting names does not prove I(O; presence)=0; label/mask balancing must be checked separately.',
    }


def classifier_features(db, mode='aggressive'):
    """Return (N x selected_width FP32 features, ordered feature-name list)."""
    from pipeline.training_r16_multisupport import tree_features
    x = tree_features(db, 'classification')
    if x.ndim != 2 or x.shape[1] != len(original_names()):
        raise ValueError('Unexpected classifier transform width')
    ix = classifier_indices(mode)
    selected = np.ascontiguousarray(x[:, ix], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError('Non-finite selected classifier features')
    names = original_names()
    return selected, [names[i] for i in ix]


def importance_decomposition():
    """Read all 20 existing classifiers, use each library's actual metric."""
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    import pandas as pd
    names = original_names()
    categories = np.asarray([
        'pure_observation_fraction' if n.startswith('observed_fraction/') else
        'audio_adjacent_observed_delta' if 'adjacent_observed_delta' in n else
        'audio_contiguous_observed_lag' if 'observed_contiguous_lag' in n else
        'other_audio' for n in names
    ])
    rows, sources = [], {}
    for regime in ('D0', 'H0'):
        for fold in range(5):
            paths = {
                'LightGBM': ROOT / f'models/r16_multisupport_full_r1/trees/{regime}/fold_{fold:02d}/classification.txt',
                'CatBoost': ROOT / f'models/r16_catboost_peft_cpu8_r1/{regime}/fold_{fold:02d}/classification.cbm',
            }
            for family, path in paths.items():
                if family == 'LightGBM':
                    model = lgb.Booster(model_file=str(path))
                    importance = model.feature_importance(importance_type='gain').astype(float)
                    definition = 'gain'
                else:
                    model = CatBoostClassifier()
                    model.load_model(str(path))
                    importance = model.get_feature_importance(type='PredictionValuesChange', thread_count=2)
                    definition = 'PredictionValuesChange'
                if len(importance) != 2033:
                    raise ValueError('Trained classifier schema width differs')
                total = max(float(importance.sum()), 1e-30)
                for category in np.unique(categories):
                    rows.append({
                        'family': family, 'regime': regime, 'fold': fold,
                        'importance_definition': definition, 'category': category,
                        'feature_count': int((categories == category).sum()),
                        'importance_fraction': float(importance[categories == category].sum() / total),
                    })
                sources[str(path.relative_to(ROOT))] = sha256(path)
    result = pd.DataFrame(rows)
    return result, sources


def main():
    import pandas as pd
    features = OUTPUT / 'features'
    summaries = OUTPUT / 'shortcut_summary'
    features.mkdir(parents=True, exist_ok=True)
    summaries.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        s = schema(mode)
        write_json(features / f'schema_{mode}.json', s)
    pd.DataFrame([
        {'index': i, 'feature': n,
         'drop_direct': n.startswith('observed_fraction/'),
         'drop_aggressive': 'observed' in n,
         'pure_O_feature': n.startswith('observed_fraction/'),
         'audio_and_observation_joint_statistic': 'observed' in n and not n.startswith('observed_fraction/')}
        for i, n in enumerate(original_names())
    ]).to_csv(features / 'all2033_feature_decisions.csv', index=False)

    rng = np.random.default_rng(20261030)
    fixture = np.round(rng.normal(75, 5, (8, 15, 25)), 1).astype(np.float32)
    fixture[0, 0] = np.nan
    fixture[1, 4:7] = np.nan
    fixture[2, :, 24] = np.nan
    full, _ = classifier_features(fixture, 'full')
    checked = {}
    for mode in MODES:
        x, names = classifier_features(fixture, mode)
        assert np.array_equal(x, full[:, classifier_indices(mode)])
        checked[mode] = {'shape': list(x.shape), 'finite': bool(np.isfinite(x).all()),
                         'selected_columns_exactly_equal_source_transform': True,
                         'name_count_matches': len(names) == x.shape[1]}
    write_json(features / 'schema_fixture.json', {'status': 'passed', 'modes': checked, 'Test_used': False, 'training': False})

    importance, source_pins = importance_decomposition()
    importance.to_csv(features / 'classifier_importance_decomposition_all20.csv', index=False)
    imp_mean = importance.groupby(['family', 'importance_definition', 'category', 'feature_count'], as_index=False).importance_fraction.mean()
    imp_mean.to_csv(features / 'classifier_importance_decomposition_mean.csv', index=False)
    write_json(features / 'existing_classifier_sha256.json', source_pins)
    pd.DataFrame([
        {'mode': m, 'features': schema(m)['feature_count'], 'removed': schema(m)['removed_count']}
        for m in MODES
    ]).to_csv(features / 'mode_counts.csv', index=False)

    file_names = ['support_mask_by_observation_density.csv', 'insertion_boundary_jumps.csv',
                  'same_y_background_LF_error.csv', 'quantization_by_true_class.csv',
                  'same_y_005_015_error_correlations.csv']
    frames = {name: pd.read_csv(AUDIT / name) for name in file_names}
    pins = {name: sha256(AUDIT / name) for name in file_names}
    for name, frame in frames.items():
        frame.to_csv(summaries / name, index=False)
    mask = frames[file_names[0]]
    mi_rows = []
    for dataset, part in mask.groupby('dataset', sort=False):
        counts = part.n.to_numpy(float)
        positive = np.rint(counts * part.actual_positive_fraction.to_numpy(float))
        if not np.allclose(positive, counts * part.actual_positive_fraction.to_numpy(float), atol=1e-8):
            raise ValueError('Observation-density fractions do not reconstruct integer counts')
        joint = np.stack([counts - positive, positive], axis=1)
        joint /= joint.sum()
        independent = joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True)
        valid = joint > 0
        mi_rows.append({'dataset': dataset, 'n': int(counts.sum()),
                        'positive_fraction': float(positive.sum() / counts.sum()),
                        'empirical_MI_nobs_presence_bits': float((joint[valid] * np.log2(joint[valid] / independent[valid])).sum()),
                        'full_observation_pattern_MI_measured_here': False})
    mi_frame = pd.DataFrame(mi_rows)
    mi_frame.to_csv(summaries / 'observation_density_presence_MI.csv', index=False)
    density = mask[mask.n_obs.isin([5, 12, 13, 14, 15])].copy()
    density.to_csv(summaries / 'key_observation_density_rows.csv', index=False)
    boundary = frames[file_names[1]]
    boundary.to_csv(summaries / 'key_boundary_rows.csv', index=False)
    lf = frames[file_names[2]]
    lf_key = lf[(lf.condition == 'LF_background_fraction') & lf.panel.isin(['ensemble_fit', 'domain_holdout'])]
    lf_key.to_csv(summaries / 'same_y_LF_fit_and_unseen.csv', index=False)
    quant = frames[file_names[3]]
    cor = frames[file_names[4]]
    cor_key = cor[(cor.panel == 'domain_holdout') & cor.feature.isin(['bump_high_db', 'LF_energy_fraction', 'n_obs'])]
    cor_key.to_csv(summaries / 'same_y_unseen_error_correlations.csv', index=False)

    chunks = [
        '# R17 클린 분류 특징 정의와 기존 상세 감사 수치\n',
        '기존 모델·자료는 변경하지 않았다. 새 학습에 사용할 3개 특징 대조군만 정의하고, 완료된 R17 감사 상세 CSV를 재집계했다. Test 정답·Test 적합·모델 재학습을 사용하지 않았다.\n',
        '## 1. 분류 특징 제거 대조군\n',
        pd.read_csv(features / 'mode_counts.csv').to_markdown(index=False),
        '\n`direct`는 순수 관측 비율 25개만 제거한다. `aggressive`는 기존 감사에서 이름에 `observed`가 포함되어 observation_mask_proxy로 묶인 400개를 전부 제거한다. 나머지 375개는 관측 가능한 분 사이의 **실제 음향 차분·lag 통계**다. 따라서 감사의 15.34%/21.35%를 모두 음향 정보가 없는 순수 마스크 중요도로 해석할 수 없다.\n',
        imp_mean.to_markdown(index=False, floatfmt='.8f'),
        '\nLightGBM은 gain, CatBoost는 PredictionValuesChange이며 서로 다른 정의다. 특징 중요도는 모델 의존도이고 Public AUC 붕괴 원인을 인과적으로 확정하지 않는다.\n',
        '\n이름 기반 삭제 후에도 잔존 음향 통계의 유효 표본 수·0 채움·lag 지원 여부를 통해 결측 정보가 간접적으로 남을 수 있다. `I(O; presence)=0`은 별도의 **완전 관측 패턴별 양성/음성 균형**으로 확인해야 한다. `n_obs`만 균형화하면 `I(n_obs; presence)=0`만 보장한다.\n',
        '## 2. 관측 밀도별 양성 비율 및 실제 support\n',
        density.to_markdown(index=False, floatfmt='.6f'),
        '\nTRAIN의 큰 층에서는 양성 비율이 이미 약 80%다. n_obs=5 D0 80.1107%, n_obs=15 80.0714%로 차이는 0.0393%p다. 표본 11~22개인 희소 밀도층의 차이까지 전체 원인으로 확대 해석하면 안 된다. 반면 검증 패널의 일부 밀도층은 0%/100%로 극단적이며 소표본인지 반드시 함께 봐야 한다.\n',
        mi_frame.to_markdown(index=False, floatfmt='.9f'),
        '\n위 정보량은 `n_obs` 범주와 presence의 경험적 상호정보량(bits)이다. 완전한 O 행렬의 상호정보량이 아니며 둘을 같은 값으로 취급하지 않는다. 밀도 비율 차이와 사이트별 음향 차이도 구분해야 한다.\n',
        '## 3. 삽입 support 경계의 중고역 인접 분 단차\n',
        boundary.to_markdown(index=False, floatfmt='.6f'),
        '\nTRAIN D0 support 시작/끝 절대 단차 중앙값은 7.4260/7.4053 dB이고 다른 인접 분은 0.6925 dB다. H0는 7.0706/6.8839 dB, 다른 분 0.6172 dB다. 급격한 경계가 실제로 크지만, 원래 선박 통과도 변화하므로 인공 경계가 실패의 주원인인지는 별도 반사실 대조가 필요하다. 분석은 파형이 아닌 관측 가능한 인접 **분** 기준이다.\n',
        '## 4. 동일 share 0.05~0.15의 저주파 배경 분모별 오차\n',
        lf_key.to_markdown(index=False, floatfmt='.6f'),
        '\n저주파 배경 비중은 `E=10^((rounded_db-window_max)/10)`, `B_proxy=E*(1-q)`에서 재구성한 상대 성분이다. 원래 반올림 전 B 자체가 저장된 것은 아니다. 일부 구간 n=8처럼 작으므로 일반 법칙으로 단정하지 않는다.\n',
        cor_key.to_markdown(index=False, floatfmt='.6f'),
        '\n미관측 사이트의 동일 share 구간에서는 HF bump와 Tree 오차가 음의 상관이다. 이 결과를 “HF가 클수록 항상 과대추정”으로 요약하면 반대 해석이다.\n',
        '## 5. 양자화 및 물리 풀링 일관성\n',
        quant.to_markdown(index=False, floatfmt='.9f'),
        '\n양성·음성 모두 0.1 dB 격자 잔차 최대값은 0이다. 따라서 현재 저장 자료에서 격자 여부 자체로 양성/음성을 구별하는 편법은 확인되지 않았다. 반올림된 E와 원래 q의 풀링 오차 최대는 TRAIN D0 0.00176033, H0 0.00186614이며 원래 정답을 수정하지 않았다.\n',
        '## 실행 검증\n',
        '실제 변환 8개 창(결측·한 대역 전체 결측 포함)에서 full/direct/aggressive의 열 선택 일치·유한값·스키마 폭을 확인했다. 기존 분류기 20개(각 계열 D0/H0×5 Fold)의 실제 중요도를 다시 읽었다. 어떤 모델도 학습하지 않았다.\n',
    ]
    (summaries / 'R17_SHORTCUT_DETAIL_SUMMARY.md').write_text('\n\n'.join(chunks), encoding='utf-8')
    write_json(summaries / 'source_csv_sha256.json', pins)
    outputs = [p for base in (features, summaries) for p in base.iterdir()
               if p.is_file() and p.name != 'completion.json']
    write_json(summaries / 'completion.json', {
        'status': 'completed', 'model_training': False, 'Test_used': False,
        'feature_modes': {m: schema(m)['feature_count'] for m in MODES},
        'source_csv_sha256': pins,
        'output_sha256': {str(p.relative_to(OUTPUT)): sha256(p) for p in outputs},
        'code_sha256': sha256(Path(__file__)),
    })
    print(json.dumps({'status': 'completed', 'output': str(OUTPUT), 'counts': {m: schema(m)['feature_count'] for m in MODES}}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
