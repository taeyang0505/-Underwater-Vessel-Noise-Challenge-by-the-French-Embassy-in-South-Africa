"""Numeric-only verification of the completed-fivefold R13 query contract.

No acoustic files, trained models, Test, GPU, fitting or submission are used.
The injected backend deliberately cannot be enabled in production mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from pipeline.inference_r13 import FrozenInferenceR13


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NumericBackend:
    fixture_backend = True

    def __init__(self, folds=None, failure=None):
        self.folds = list(range(5)) if folds is None else folds
        self.failure = failure
        self.calls = []

    def open(self, run):
        self.run = Path(run)
        self.artifact = self.run / 'numeric_artifact.json'
        return ([{'fold': fold} for fold in self.folds],
                {str(self.artifact): digest(self.artifact)},
                {'format': 'numeric_only_verified_fixture', 'folds': self.folds})

    def predict(self, view, db, device, batch):
        if device != 'cpu':
            raise AssertionError('Fixture is CPU-only')
        self.calls.append({'fold': view['fold'], 'shape': list(db.shape), 'batch_size': batch})
        pieces = []
        for start in range(0, len(db), batch):
            mean = np.nanmean(db[start:start+batch].astype(np.float64), axis=(1, 2))
            pieces.append(mean)
        mean = np.concatenate(pieces)
        p = np.asarray(.08 + .1 * view['fold'] + (mean - 60.) * .001, np.float32)
        s = np.asarray(.15 + .12 * view['fold'] + (mean - 60.) * .002, np.float32)
        if self.failure == 'bounds':
            p[0] = 1.01
        elif self.failure == 'nonfinite':
            s[0] = np.nan
        elif self.failure == 'length':
            p = p[:-1]
        elif self.failure == 'artifact_change' and view['fold'] == 4:
            self.artifact.write_text('{"fixture": "changed"}\n')
        elif self.failure == 'boundary':
            p[:] = 0. if view['fold'] < 4 else 1.
            s[:] = 1. if view['fold'] < 4 else 0.
        return {'share': p, 'score': s}, {'fold': view['fold'], 'fixture_only': True}

    def verify_after(self, run):
        if self.failure == 'completion_change':
            return {'format': 'changed_after_prediction'}
        return {'format': 'numeric_only_verified_fixture', 'folds': self.folds}


def run_fixture():
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def rejected(name, func):
        try:
            func()
        except (ValueError, TypeError):
            checks.append(name)
        else:
            raise AssertionError('Expected fail-closed rejection: ' + name)

    db = np.broadcast_to(np.arange(11, dtype=np.float32)[:, None, None] + 70., (11, 15, 25)).copy()
    db[0, :12] = np.nan
    db[1, 2:7] = np.nan
    original = db.copy()
    with tempfile.TemporaryDirectory(prefix='r13_inference_numeric_') as temporary:
        folder = Path(temporary)
        artifact = folder / 'numeric_artifact.json'

        def engine(*, folds=None, failure=None):
            artifact.write_text('{"fixture": true}\n')
            return FrozenInferenceR13(folder, device='cpu', backend=NumericBackend(folds, failure), _fixture=True)

        first = engine()
        p1 = first.predict_db(db, batch_size=1)
        second = engine()
        p7 = second.predict_db(db, batch_size=7)
        mean = np.nanmean(db.astype(np.float64), axis=(1, 2))
        member_p = [np.asarray(.08 + .1*f + (mean-60.)*.001, np.float32) for f in range(5)]
        member_s = [np.asarray(.15 + .12*f + (mean-60.)*.002, np.float32) for f in range(5)]
        expected_p = np.mean([v.astype(float) for v in member_p], axis=0).astype(np.float32)
        expected_s = np.mean([v.astype(float) for v in member_s], axis=0).astype(np.float32)
        check('fivefold_equal_mean_share_exact', np.array_equal(p1['share'], expected_p))
        check('fivefold_equal_mean_score_exact', np.array_equal(p1['score'], expected_s))
        check('batch1_vs7_share_exact', np.array_equal(p1['share'], p7['share']))
        check('batch1_vs7_score_exact', np.array_equal(p1['score'], p7['score']))
        check('exact_folds_called_once_and_in_order', [c['fold'] for c in first.backend.calls] == list(range(5)))
        check('query_missing_mask_unchanged', np.array_equal(np.isnan(db), np.isnan(original)))
        check('query_values_unchanged', np.array_equal(db, original, equal_nan=True))
        check('output_float32', p1['share'].dtype == np.float32 and p1['score'].dtype == np.float32)
        check('query_only_receipt', p1['receipt']['query_db_only'] and not p1['receipt']['extra_calibration_or_query_fit'])
        boundary = engine(failure='boundary').predict_db(db)
        check('valid_boundary_fold_predictions_arithmetic_mean',
              np.array_equal(boundary['share'], np.full(11, .2, np.float32))
              and np.array_equal(boundary['score'], np.full(11, .8, np.float32)))
        rejected('missing_fold_rejected', lambda: engine(folds=[0, 1, 2, 3]))
        rejected('duplicate_fold_rejected', lambda: engine(folds=[0, 1, 2, 3, 3]))
        rejected('reordered_folds_rejected', lambda: engine(folds=[1, 0, 2, 3, 4]))
        rejected('production_backend_injection_rejected', lambda: FrozenInferenceR13(folder, backend=NumericBackend()))
        foreign = NumericBackend()
        foreign.fixture_backend = False
        rejected('untrusted_fixture_backend_rejected', lambda: FrozenInferenceR13(folder, backend=foreign, _fixture=True))
        for bad in ('bounds', 'nonfinite', 'length', 'artifact_change', 'completion_change'):
            rejected('prediction_' + bad + '_rejected', lambda bad=bad: engine(failure=bad).predict_db(db))
        good = engine()
        rejected('query_dictionary_labels_rejected', lambda: good.predict_db({'db': db, 'share': np.zeros(11)}))
        for name, value in [('empty', db[:0]), ('wrong_shape', db[:, :, :24]),
                            ('integer', np.ones((2, 15, 25), np.int32)),
                            ('all_missing', np.full((2, 15, 25), np.nan, np.float32))]:
            rejected('query_' + name + '_rejected', lambda value=value: good.predict_db(value))
        few = db[:1].copy(); few[0, 12] = np.nan
        rejected('less_than_three_minutes_rejected', lambda: good.predict_db(few))
        partial = db[2:3].copy(); partial[0, 0, 0] = np.nan
        rejected('partial_band_minute_rejected', lambda: good.predict_db(partial))
        infinite = db[2:3].copy(); infinite[0, 0, 0] = np.inf
        rejected('infinite_query_rejected', lambda: good.predict_db(infinite))
        for batch in (0, -1, 1., True):
            rejected('invalid_batch_' + str(batch) + '_rejected', lambda batch=batch: good.predict_db(db, batch_size=batch))
    return {
        'format': 'r13_frozen_inference_numeric_verification_v1', 'status': 'passed',
        'scope': 'Synthetic db-only mock-backend contract fixture; no production model/run verification proof',
        'examples': 11, 'input_shape': [11, 15, 25], 'five_folds': [0, 1, 2, 3, 4],
        'checks_passed': len(checks), 'checks': checks,
        'code_sha256': {'pipeline/inference_r13.py': digest(ROOT/'pipeline/inference_r13.py'),
                        'ops/verify_r13_frozen_inference.py': digest(__file__)},
        'production_static_review': {
            'completed_fivefold_guard': 'ProductionBackend.open verifies completed production run before query CSV is opened',
            'state': 'runtime recipe SHA, prepared data hash, fold-specific TRAIN-only PCA SHA/binding, architecture13 and tree5252 checked',
            'restoration': 'strict completed best_window state, code/artifact hashes, one neural loaded at a time',
            'formula': 'Frozen per-fold policy then equal arithmetic mean across folds; numerical clip[0,1] only',
        },
        'coverage_limits': ['No real neural/tree restore or GPU exercised by this mock fixture',
                            'No guarantee of CPU/GPU or real-model batch-independence from mock arithmetic',
                            'Production artifact integrity is additionally enforced by verify_run/restore_neural/PhysicsLGBM.load'],
        'real_data_or_Test_opened': False, 'fitting_or_inference_on_trained_models': False,
        'GPU_SSH_external_transmission': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/R13_FROZEN_INFERENCE_VERIFICATION.json')
    args = parser.parse_args()
    result = run_fixture()
    out = Path(args.output)
    if not out.is_absolute(): out = ROOT/out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'checks_passed': result['checks_passed'],
                      'output': str(out), 'sha256': digest(out)}))
