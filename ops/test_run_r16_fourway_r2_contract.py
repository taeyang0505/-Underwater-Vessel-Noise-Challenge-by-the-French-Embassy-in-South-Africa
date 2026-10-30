"""Synthetic tests of the actual r2 collector; no data, models or CUDA needed.

Injects NumPy-backed torch/model modules, then invokes collect_stretch itself.
The model records every forward batch and rejects forwards outside no_grad.
"""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
driver = importlib.import_module("ops.run_r16_fourway_r2")


class Tensor:
    def __init__(self, a):
        self.a = np.asarray(a)

    def float(self):
        return Tensor(self.a.astype(np.float32))

    def sigmoid(self):
        return Tensor(1 / (1 + np.exp(-self.a)))

    def __sub__(self, value):
        return Tensor(self.a - value)

    def __mul__(self, value):
        return Tensor(self.a * (value.a if isinstance(value, Tensor) else value))

    def div(self, value):
        return Tensor(self.a / value)

    def clamp(self, lo, hi):
        return Tensor(np.clip(self.a, lo, hi))

    def sum(self, axes):
        return Tensor(self.a.sum(axis=axes))

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class CollectorFixture:
    def __init__(self):
        self.strict_calls = 0
        self.no_grad_depth = 0
        self.forward_batches = []
        self.modules = {}
        torch = types.ModuleType("torch")
        torch.as_tensor = lambda a, **_: Tensor(a)
        torch.stack = lambda values, axis: Tensor(np.stack([v.a for v in values], axis))
        torch.isfinite = lambda a: np.isfinite(a.a)
        torch.set_num_threads = lambda _: None
        torch.no_grad = self.no_grad
        torch.inference_mode = lambda: self.fail_inference_mode()
        torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
        torch.mps = types.SimpleNamespace(empty_cache=lambda: None)
        torch.load = self.checkpoint
        self.modules["torch"] = torch
        original = types.ModuleType("ops.finalize_r16_pretrained_cuda_bf16")
        original.RUN = Path("/synthetic/no-artifacts")
        original.configure_variant = lambda variant: None
        original.check_checkpoint = lambda *args: None
        self.modules[original.__name__] = original
        training = types.ModuleType("pipeline.training_r16_pretrained_cuda_bf16")
        training.OPS = Path("/synthetic/no-plan")
        training.MICROBATCH = {"dasheng_base": 32, "dinov2_small": 32, "timemoe": 64}
        training.branch = self.branch
        training.base_fingerprint = lambda _: "frozen-fixture"
        self.modules[training.__name__] = training
        features = types.ModuleType("pipeline.training_r16_multisupport")
        features.features = lambda db: (db, np.ones_like(db, dtype=np.float32))
        self.modules[features.__name__] = features
        precision = types.ModuleType("pipeline.r16_cuda_precision")
        precision.autocast_context = self.autocast
        precision.require_cuda_bf16 = self.require_strict
        self.modules[precision.__name__] = precision

    @contextmanager
    def no_grad(self):
        self.no_grad_depth += 1
        try:
            yield
        finally:
            self.no_grad_depth -= 1

    @contextmanager
    def autocast(self, device):
        if device == "cuda" and not self.strict_calls:
            raise AssertionError("Strict BF16 setup must precede inference")
        yield

    def require_strict(self):
        self.strict_calls += 1
        return {"synthetic": True}

    @staticmethod
    def fail_inference_mode():
        raise AssertionError("r2 must reproduce original no_grad execution")

    @staticmethod
    def provenance(family):
        return {"model_id": family, "revision": "fixture", "weight_sha256": "fixture"}

    def checkpoint(self, path, **kwargs):
        family = Path(path).parts[-4]
        return {"provenance": self.provenance(family), "adapter_state": {},
                "permanent_base_fingerprint": "frozen-fixture"}

    def branch(self, family, task, device):
        fixture = self

        class Model:
            def eval(self):
                return self

            def load_adapter_state(self, _):
                pass

            def __call__(self, x):
                if fixture.no_grad_depth != 1:
                    raise AssertionError("Forward must execute inside exactly one no_grad context")
                fixture.forward_batches.append((family, len(x.a)))
                return x

        return Model(), self.provenance(family)

    @staticmethod
    def panels(names=("selection", "local_gate", "local_report", "query"), n=129):
        data, predictions = {}, {}
        for i, name in enumerate(names):
            db = np.linspace(-2 + i * .1, 2 + i * .1, n, dtype=np.float32)[:, None, None]
            expected = Tensor(db).float().sigmoid().sum((1, 2)).numpy().astype(float)
            data[name] = {"db": db}
            # A nonzero, accepted reconstruction delta proves exact identity is
            # restored to the stored prediction, rather than recomputed output.
            predictions[name] = {"share_" + r: np.tile((expected + .00005)[:, None], (1, 5))
                                 for r in driver.REGIMES}
        return data, predictions

    def run(self, data, predictions, folder, batch=32):
        with patch.dict(sys.modules, self.modules), patch.object(driver, "read", return_value={}):
            return driver.collect_stretch(data, predictions, "cuda", batch, folder / "progress.json")


class CollectorContracts(unittest.TestCase):
    def test_actual_collector_matches_source_microbatch_only_for_selection(self):
        fixture = CollectorFixture()
        data, predictions = fixture.panels()
        with tempfile.TemporaryDirectory() as tmp:
            result, drifts = fixture.run(data, predictions, Path(tmp), batch=7)
        self.assertEqual(fixture.strict_calls, 1)
        self.assertEqual(len(drifts), 2 * 3 * 4)
        for row in drifts:
            expected = 64 if row["panel"] == "selection" and row["family"] == "timemoe" else 32
            self.assertEqual(row["used_batch"], expected)
        # Per regime/family: first source panel uses training's family batch;
        # every other panel remains original finalizer batch32, even CLI batch7.
        expected_calls = []
        for _ in driver.REGIMES:
            for family in driver.FAMILIES:
                first = [64, 64, 1] if family == "timemoe" else [32, 32, 32, 32, 1]
                expected_calls += [(family, n) for n in first]
                expected_calls += [(family, n) for n in [32, 32, 32, 32, 1] * 3]
        self.assertEqual(fixture.forward_batches, expected_calls)
        self.assertEqual(set(result), set(data))

    def test_epsilon_zero_is_stored_identity_exact_and_positive_epsilon_monotone(self):
        fixture = CollectorFixture()
        data, predictions = fixture.panels(names=("selection",))
        original = {r: predictions["selection"]["share_" + r].copy() for r in driver.REGIMES}
        with tempfile.TemporaryDirectory() as tmp:
            result, drifts = fixture.run(data, predictions, Path(tmp))
        for r in driver.REGIMES:
            stretched = result["selection"]["stretch_" + r]
            np.testing.assert_array_equal(stretched[:, 0, :], original[r])
            self.assertTrue(np.all(np.diff(stretched, axis=1) <= 1e-7))
            # Unmodified Tree columns remain identical at every epsilon.
            np.testing.assert_array_equal(stretched[:, :, :2],
                np.repeat(original[r][:, None, :2], len(driver.EPS), axis=1))
        self.assertTrue(all(0 < row["max_identity_reconstruction_drift"] < .002 for row in drifts))

    def test_drift_over_fixed_threshold_rejected_and_receipt_written_before_error(self):
        fixture = CollectorFixture()
        data, predictions = fixture.panels(names=("selection",))
        predictions["selection"]["share_D0"][0, driver.MEMBERS.index("timemoe")] += .003
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with self.assertRaisesRegex(ValueError, r"selection/D0/timemoe batch=64"):
                fixture.run(data, predictions, folder)
            receipt = json.loads((folder / "NUMERICAL_RECONSTRUCTION.json").read_text())
            row = receipt["drifts"][-1]
        self.assertTrue(receipt["no_training"])
        self.assertEqual((row["panel"], row["regime"], row["family"], row["used_batch"]),
                         ("selection", "D0", "timemoe", 64))
        self.assertGreater(row["max_identity_reconstruction_drift"], .002)

    def test_fresh_namespace_and_driver_source_pin(self):
        self.assertEqual(driver.NAME, "r16_v1_fourway_r2")
        self.assertIn("ops/run_r16_fourway_r2.py", driver.CODE_FILES)
        self.assertNotIn("ops/run_r16_fourway.py", driver.CODE_FILES)


if __name__ == "__main__":
    unittest.main()
