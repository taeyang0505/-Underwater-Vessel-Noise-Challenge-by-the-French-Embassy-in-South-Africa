"""One explicit CPU tree pilot on original + augmented training components."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, dtype='float64') - p)**2)))


def main(a):
    project, directory, output = a.project.resolve(), a.data.resolve(), a.output.resolve()
    if output.exists() or output.is_relative_to(project / 'data') or output.is_relative_to(project / 'models'):
        raise ValueError('Use a fresh OPS pilot output')
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(project)]
    from energy_domain_augmentation import augment_components
    from pipeline.mixtures import fixed_features, physics_features
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    opened = set()
    def audit(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(project / 'data'):
                if not p.is_relative_to(directory) or p.name.startswith(('outer_', 'calibration', 'sealed', 'Test')):
                    raise PermissionError('Tree pilot blocked forbidden partition')
                opened.add(str(p.relative_to(project)))
    sys.addaudithook(audit)
    output.mkdir(parents=True)
    start = time.perf_counter()
    report = {'status': 'running', 'configuration': vars(a), 'outer_Test_accessed': False}
    def save():
        report['seconds'] = time.perf_counter() - start
        report['dataset_files_opened'] = sorted(opened)
        (output / 'manifest.json').write_text(json.dumps(report, indent=2, default=str))
    try:
        gate = json.loads((directory / 'readiness_manifest.json').read_text())
        if gate.get('ready_to_train') is not True or gate.get('challenge_test_accessed') is not False:
            raise ValueError('Verified source package required')
        for name in ['train.npz', 'validation.npz', 'validation_recipes.json']:
            if sha(directory / name) != gate['file_sha256'][name]:
                raise ValueError('Input SHA differs')
        with np.load(directory / 'train.npz', allow_pickle=False) as z:
            train = {k: z[k] for k in ['db', 'share', 'presence', 'cell_share']}
        with np.load(directory / 'validation.npz', allow_pickle=False) as z:
            val = {k: z[k] for k in ['db', 'share']}
        recipes = json.loads((directory / 'validation_recipes.json').read_text())
        items = [i for batch in recipes['batches'] for i in batch['items']]
        def route(item):
            refs = item['references']; native = [r['kind'] == 'native_contiguous_15min_crop' for r in refs]
            return 'zero' if not refs else 'native_only' if all(native) else 'mixed_native_short' if any(native) else 'single_short' if len(refs) == 1 else 'multi_short'
        routes = np.asarray([route(i) for i in items])
        if len(routes) != len(val['share']):
            raise ValueError('Recipe alignment differs')
        def transform(db):
            return np.concatenate([fixed_features(db, version=3), physics_features(db)], 1)
        features = [transform(train['db'])]; labels = [train['share']]; audits = []
        for copy in range(a.augmented_copies):
            rng = np.random.default_rng(a.seed + 100003 * (copy + 1))
            xparts = []; yparts = []
            for begin in range(0, len(train['share']), 512):
                d, info = augment_components(train['db'][begin:begin+512], train['cell_share'][begin:begin+512],
                    rng, mode='reference_smooth', probability=.75, strength_db=12.)
                xparts.append(transform(d['db'])); yparts.append(d['share']); audits.append(info)
            features.append(np.concatenate(xparts)); labels.append(np.concatenate(yparts))
        X = np.concatenate(features); y = np.concatenate(labels); V = transform(val['db'])
        report.update(data_package_hash=gate['data_package_hash'], train_rows=len(y), base_train_rows=len(train['share']),
            inner_rows=len(val['share']), n_features=X.shape[1], augmentation_audits=audits,
            source_independent_examples_increased=False,
            baseline_r1_physics_inner_rmse=.28370051829302556,
            augmentation_code_sha256=sha(Path(__file__).with_name('energy_domain_augmentation.py')))
        save(); del features, labels, xparts, yparts
        model = LGBMRegressor(objective='regression_l2', n_estimators=1000, learning_rate=.035,
            num_leaves=31, min_child_samples=100, colsample_bytree=.85, subsample=.85, subsample_freq=1,
            reg_lambda=1., max_bin=127, n_jobs=4, random_state=a.seed, verbosity=-1)
        model.fit(X, y, eval_set=[(V, val['share'])], callbacks=[early_stopping(40, verbose=False), log_evaluation(100)])
        pred = np.clip(model.predict(V), 0, 1)
        model.booster_.save_model(str(output / 'regressor.txt'))
        np.savez_compressed(output / 'inner_predictions.npz', share_true=val['share'], share=pred, route=routes)
        report.update(status='completed inner-only tree pilot', inner_rmse=rmse(val['share'], pred),
            best_iteration=int(model.best_iteration_), inner_bias=float(np.mean(pred-val['share'])),
            by_route={r: {'n': int(np.sum(routes == r)), 'rmse': rmse(val['share'][routes == r], pred[routes == r]),
                         'mean_prediction': float(pred[routes == r].mean())} for r in np.unique(routes)})
        for name in ['train.npz', 'validation.npz', 'validation_recipes.json']:
            if sha(directory / name) != gate['file_sha256'][name]:
                raise ValueError('Input changed during pilot')
        save()
        print(json.dumps({k: v for k, v in report.items() if k not in ['augmentation_audits', 'configuration']}, indent=2))
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); save(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--augmented-copies', type=int, default=2)
    main(p.parse_args())
