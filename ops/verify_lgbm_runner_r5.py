"""Tiny generated-array CPU integration check for the R5 tree runner.

This does not open real folds, Test, OUTER, calibration, sealed or historical
data. Eight trees verify wiring and artifact restoration, not model quality.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import inspect
import json
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.grid_inputs_r5 import canonical_sha256, file_sha256
from pipeline.lgbm_r5 import fit_tree, predict_tree, transform, feature_binding
from pipeline.physics_features_r5 import load_physics_pca


def rejected(action):
    try:
        action()
    except (ValueError, TypeError, FileExistsError):
        return True
    raise AssertionError('Invalid fixture operation was accepted')


def generated_components(n, seed):
    """Known B/A mixtures; saved float32 db/q rederive the physical target."""
    rng = np.random.default_rng(seed)
    times = np.arange(15, dtype=float)[None, :, None]
    bands = np.arange(25, dtype=float)[None, None, :]
    background_db = 100 + rng.normal(0, 2, (n, 1, 25)) + rng.normal(0, .7, (n, 15, 1))
    background = 10 ** ((background_db - 100) / 10)
    centre = rng.uniform(3, 12, (n, 1, 1))
    width = rng.uniform(1, 4, (n, 1, 1))
    reference = np.exp(-.5 * ((times-centre) / width) ** 2) * np.exp(-bands / 9)
    reference *= rng.uniform(.7, 1.4, (n, 1, 25))
    observed = rng.random((n, 15, 25)) > .04
    observed[::3, 0] = False; observed[1::5, -1] = False
    background = np.where(observed, background, 0)
    reference = np.where(observed, reference, 0)
    wanted = rng.uniform(.06, .85, n); wanted[::4] = 0
    scale = wanted * background.sum((1, 2)) / ((1-wanted) * reference.sum((1, 2)))
    added = scale[:, None, None] * reference
    total = background + added
    q = np.divide(added, total, out=np.zeros_like(total), where=observed).astype('float32')
    db = np.where(observed, 100+10*np.log10(np.maximum(total, 1e-300)), np.nan).astype('float32')
    peak = np.nanmax(db.astype(float), (1, 2), keepdims=True)
    energy = np.where(observed, 10 ** ((db.astype(float)-peak)/10), 0)
    share = ((q.astype(float)*energy).sum((1, 2)) / energy.sum((1, 2))).astype('float32')
    assert np.max(np.abs(share-wanted)) < 2e-6
    return {'db': db, 'cell_share': q, 'share': share, 'presence': (share > 0).astype('float32')}


def run(output):
    train = generated_components(96, 3401); inner = generated_components(24, 3402)
    before = {role: {k: v.copy() for k, v in data.items()} for role, data in [('train',train),('inner',inner)]}
    policy = {'quantization_decimals': 1, 'endpoint_dropout_probability': 0.0}
    settings = {'maximum_trees': 8, 'patience': 3, 'threads': 2}
    checks = {}
    with tempfile.TemporaryDirectory(prefix='r5-lgbm-generated-fixture-') as temporary:
        temp = Path(temporary)
        np.savez_compressed(temp/'train.npz', **train)
        binding = {'train_npz_sha256': file_sha256(temp/'train.npz'),
            'data_package_hash': canonical_sha256({'scope':'generated fixture only', 'train_rows':96}),
            'source_hash': canonical_sha256({'generator':'generated_components','seed':3401}),
            'seed':42, 'fold':0, 'train_examples':96}
        folder = temp/'tree'
        summary, fitted = fit_tree(train, inner, binding, policy, settings, folder, fixture=True)
        assert summary['n_features'] == 1258+45+8+13 == 1324
        assert summary['prototype']['prototypes'] == 8 and summary['prototype']['training_only']
        assert summary['prototype']['whole_mixture_spectrum_used'] is False
        assert summary['maximum_trees'] == 8 and 1 <= summary['best_iteration'] <= 8
        assert summary['fixture_only'] is True and summary['outer_or_Test_accessed'] is False
        checks['feature_count_1324'] = {'fixed':1258,'existing_physics':45,'TRAIN_retrieval':8,'new_physics':13}
        checks['TRAIN_positive_excess_prototype_count'] = summary['prototype']['prototypes']
        checks['maximum_trees_and_patience'] = settings
        restored = predict_tree(inner['db'], binding, policy, folder, allow_fixture=True, batch_size=7)
        rowwise = predict_tree(inner['db'], binding, policy, folder, allow_fixture=True, batch_size=1)
        whole = predict_tree(inner['db'], binding, policy, folder, allow_fixture=True, batch_size=256)
        for key in ['share','score']:
            assert np.array_equal(fitted[key], restored[key])
            assert np.array_equal(restored[key], rowwise[key]) and np.array_equal(rowwise[key],whole[key])
            assert restored[key].shape == (24,) and np.isfinite(restored[key]).all()
            assert ((restored[key]>=0)&(restored[key]<=1)).all()
        checks['saved_restore_and_batch_1_7_256_bitwise_equal'] = True

        record = summary['physics_pca']
        pca = load_physics_pca(folder/record['filename'], expected_binding=feature_binding(binding,policy),
            expected_file_sha256=record['file_sha256'], expected_payload_sha256=record['payload_sha256'])
        assert pca['training_rows'] == 96 and pca['training_targets_used'] is False
        features = transform(inner['db'], policy, pca, binding, folder/'prototypes.npz')
        chunks = np.concatenate([transform(inner['db'][:8],policy,pca,binding,folder/'prototypes.npz'),
            transform(inner['db'][8:],policy,pca,binding,folder/'prototypes.npz')])
        assert np.allclose(features,chunks,atol=1e-7,rtol=0)
        checks['feature_chunk_max_abs_error'] = float(np.max(np.abs(features-chunks)))
        assert features.shape == (24,1324)
        checks['TRAIN_PCA_rows_and_targets_policy'] = {'train_rows':96,'target_used':False,
            'quantization_decimals':pca['training_binding']['quantization_decimals']}

        altered = copy.deepcopy(inner)
        altered['share'][:] = .99; altered['presence'][:] = 1; altered['cell_share'][:] = .99
        label_changed = predict_tree(altered['db'],binding,policy,folder,allow_fixture=True,batch_size=5)
        assert all(np.array_equal(restored[k],label_changed[k]) for k in restored)
        checks['changing_query_labels_does_not_change_predictions'] = True
        checks['query_labels_not_in_predict_signature'] = not any(k in inspect.signature(predict_tree).parameters for k in ['share','cell_share','presence','labels','recipes'])
        assert checks['query_labels_not_in_predict_signature']
        checks['query_labelled_dictionary_rejected'] = rejected(lambda: predict_tree(inner,binding,policy,folder,allow_fixture=True))
        checks['fixture_artifact_rejected_by_production_default'] = rejected(lambda: predict_tree(inner['db'],binding,policy,folder))
        checks['changed_TRAIN_binding_rejected'] = rejected(lambda: predict_tree(inner['db'],dict(binding,train_npz_sha256='e'*64),policy,folder,allow_fixture=True))
        checks['changed_quantization_policy_rejected'] = rejected(lambda: predict_tree(inner['db'],binding,dict(policy,quantization_decimals=None),folder,allow_fixture=True))
        checks['output_overwrite_rejected'] = rejected(lambda: fit_tree(train,inner,binding,policy,settings,folder,fixture=True))
        for role,data in [('train',train),('inner',inner)]:
            for key,array in data.items():
                assert np.array_equal(array,before[role][key],equal_nan=True)
        checks['TRAIN_and_inner_arrays_unchanged'] = True
        with (folder/'physics_pca.json').open('a') as f:f.write(' ')
        checks['PCA_file_tampering_rejected'] = rejected(lambda: predict_tree(inner['db'],binding,policy,folder,allow_fixture=True))
        assert 'torch' not in sys.modules
        checks['Torch_not_imported'] = True
        metrics = {'tiny_inner_rmse_smoke_only':summary['best_inner_rmse'], 'best_iteration':summary['best_iteration']}

    sources = ['pipeline/lgbm_r5.py','pipeline/physics_features_r5.py','pipeline/observation_augmentation_r5.py',
        'pipeline/train_lgbm.py','pipeline/mixtures.py','pipeline/prototypes.py','pipeline/grid_inputs_r5.py',
        'ops/verify_lgbm_runner_r5.py']
    result = {'status':'PASS','scope':'CPU generated-array eight-tree integration fixture only',
        'generated_train_examples':96,'generated_inner_examples':24,'settings':settings,
        'policy':policy,'checks':checks,'smoke_metrics_not_performance_evidence':metrics,
        'real_datasets_opened':False,'Test_outer_calibration_sealed_or_history_opened':False,
        'GPU_used':False,'actual_fold_training_executed':False,
        'code_sha256':{name:file_sha256(ROOT/name) for name in sources}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/R5_LGBM_RUNNER_FIXTURE.json')
    run(parser.parse_args().output)
