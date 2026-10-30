#!/usr/bin/env python3
"""Synthetic contract fixtures for the separate R16 postprocessing driver.

Only stdlib + NumPy are required. Driver functions are loaded through AST so
no training/production import, real data, model, report, query, GPU or network
is accessed. The paired-forward fixture uses a tiny deterministic fake tensor
backend to execute the *actual* collector body, not a second implementation of
its adjustment formula.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
import csv
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "ops/run_r16_postprocess.py"


def functions_only(path: Path, names=None, **extra):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and (names is None or n.name in names)]
    namespace = dict(Path=Path, argparse=__import__('argparse'), csv=csv, datetime=datetime,
                     gc=gc, hashlib=hashlib, json=json, os=os, sys=sys, time=time, np=np,
                     ROOT=ROOT, BASE=ROOT/"models/fixture_base", OUT=ROOT/"models/fixture_out",
                     EPS=(0., .005, .01, .02), REGIMES=("D0", "H0"),
                     FAMILIES=("dasheng_base", "dinov2_small", "timemoe"),
                     MEMBERS=("lightgbm5", "catboost5", "dasheng_base", "dinov2_small", "timemoe"))
    namespace.update(extra)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class Tensor:
    def __init__(self, values): self.a = np.asarray(values)
    def float(self): return Tensor(self.a.astype(np.float32))
    def sigmoid(self): return Tensor(1/(1+np.exp(-self.a)))
    def div(self, value): return Tensor(self.a/value)
    def clamp(self, low, high): return Tensor(np.clip(self.a, low, high))
    def sum(self, axes): return Tensor(self.a.sum(axis=axes))
    def cpu(self): return self
    def numpy(self): return self.a
    def __sub__(self, other): return Tensor(self.a-other)
    def __mul__(self, other): return Tensor(self.a*(other.a if isinstance(other, Tensor) else other))
    def __array__(self, dtype=None): return np.asarray(self.a, dtype=dtype)


class Model:
    def __init__(self, calls): self.calls = calls
    def load_adapter_state(self, state): self.calls.append("load_adapter")
    def eval(self): self.calls.append("eval"); return self
    def __call__(self, x):
        self.calls.append("forward")
        q = x.a[:, 0]
        return Tensor(np.log(q)-np.log1p(-q))


def fake_modules(calls):
    torch = ModuleType("torch")
    torch.set_num_threads = lambda n: None
    torch.as_tensor = lambda a, **kw: Tensor(a)
    torch.stack = lambda seq, dim: Tensor(np.stack([x.a for x in seq], axis=dim))
    torch.isfinite = lambda x: np.isfinite(x.a)
    torch.inference_mode = nullcontext
    torch.cuda = SimpleNamespace(empty_cache=lambda: None)
    torch.mps = SimpleNamespace(empty_cache=lambda: None)
    provenance = dict(model_id="fixture", revision="fixed", weight_sha256="0"*64)
    torch.load = lambda *a, **kw: dict(provenance=provenance, permanent_base_fingerprint="base",
                                     adapter_state={})
    original = ModuleType("ops.finalize_r16_pretrained_cuda_bf16")
    original.configure_variant = lambda variant: calls.append("configure:"+variant)
    original.RUN = Path("/fixture/no_models_are_read")
    original.check_checkpoint = lambda *args: calls.append("checkpoint_verified")
    training = ModuleType("pipeline.training_r16_pretrained_cuda_bf16")
    training.branch = lambda *args: (Model(calls), provenance)
    training.base_fingerprint = lambda model: "base"
    training.OPS = Path("/fixture/no_plan_is_read")
    features = ModuleType("pipeline.training_r16_multisupport")
    def transform(db):
        db = np.asarray(db, np.float32)
        x = np.repeat(db[:, None], 17, axis=1)
        # Deliberately nonuniform physical weights, identical for all candidates.
        w = np.broadcast_to(np.arange(1, 376, dtype=np.float32).reshape(1,15,25), db.shape).copy()
        w /= w.sum(axis=(1,2), keepdims=True)
        return x, w
    features.features = transform
    precision = ModuleType("pipeline.r16_cuda_precision")
    precision.autocast_context = lambda device: nullcontext()
    ops_package = ModuleType("ops"); ops_package.__path__ = []
    ops_package.finalize_r16_pretrained_cuda_bf16 = original
    pipeline_package = ModuleType("pipeline"); pipeline_package.__path__ = []
    pipeline_package.training_r16_pretrained_cuda_bf16 = training
    pipeline_package.training_r16_multisupport = features
    pipeline_package.r16_cuda_precision = precision
    return {"torch": torch, "ops": ops_package, "pipeline": pipeline_package,
            "ops.finalize_r16_pretrained_cuda_bf16": original,
            "pipeline.training_r16_pretrained_cuda_bf16": training,
            "pipeline.training_r16_multisupport": features,
            "pipeline.r16_cuda_precision": precision}, transform


class AccessBoundary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="r16-postprocess-contract-")
        self.root = Path(self.tmp.name)
        self.base = self.root / "models/final"
        self.base.mkdir(parents=True)
        self.ns = functions_only(DRIVER, ROOT=self.root, BASE=self.base)
        self.state = {"phase": "selection"}
        self.guard = self.ns["access_guard"](self.state)
    def tearDown(self): self.tmp.cleanup()
    def open_event(self, path): self.guard("open", (str(path), "r", 0))
    def test_query_and_report_reads_are_blocked_before_decision(self):
        for path in (self.root/"data/raw/Test.csv", self.base/"query_predictions.npz",
                     self.base/"TEST_PREDICTION_AUDIT.json", self.base/"evaluation/local_report.npz",
                     self.root/"bundle/local_report/control_eq6.npz", self.base/"EVALUATION.json",
                     self.base/"RESULTS.md", self.base/"progress.json", self.base/"index.html"):
            with self.subTest(path=str(path)), self.assertRaises(PermissionError): self.open_event(path)
    def test_symlink_alias_cannot_bypass_query_or_report_guard(self):
        for i, target in enumerate((self.root/"data/raw/Test.csv", self.root/"bundle/local_report/control_eq6.npz")):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"synthetic fixture; never opened")
            link = self.root/f"innocent_alias_{i}.bin"
            link.symlink_to(target)
            with self.assertRaises(PermissionError): self.open_event(link)
    def test_report_phase_does_not_open_query_access(self):
        self.state["phase"]="report"
        self.open_event(self.root/"bundle/local_report/control_eq6.npz")
        with self.assertRaises(PermissionError): self.open_event(self.root/"data/raw/Test.csv")
        self.state["phase"]="query"
        self.open_event(self.root/"data/raw/Test.csv")
    def test_ordinary_gate_and_training_features_remain_accessible(self):
        self.open_event(self.root/"bundle/local_gate/control_eq6.npz")
        self.open_event(self.base/"selection_predictions.npz")
    def test_query_derived_eda_remains_forbidden_even_after_freeze(self):
        self.state["phase"]="query"
        for path in (self.root/"reports/eda_deep/features.npz", self.root/"reports/r15_band_comparison_test.json"):
            with self.assertRaises(PermissionError): self.open_event(path)


class PairedForwardIdentity(unittest.TestCase):
    def run_fixture(self, offset=0.0004):
        calls=[]; modules, transform=fake_modules(calls)
        writes=[]; ns=functions_only(DRIVER, read=lambda path: {}, write=lambda p,d: writes.append(d))
        db=np.linspace(.05,.95,2*15*25,dtype=np.float32).reshape(2,15,25)
        _,w=transform(db)
        # Obtain the fake backend's exact epsilon=0 reconstruction, then emulate
        # small stored FP32 drift. Identity must preserve those saved values.
        q=Tensor(np.log(db)-np.log1p(-db)).float().sigmoid().a
        original=(q*w).sum((1,2))
        saved=np.clip(original+offset,0,1)
        pred={"sample_id":np.array(["fixture0","fixture1"])}
        for regime in ("D0","H0"):
            pred["share_"+regime]=np.column_stack([np.array([.1,.3]), np.array([.2,.4]), saved, saved, saved]).astype(np.float32)
            pred["classifier_"+regime]=np.full((2,5),.6,np.float32)
        with patch.dict(sys.modules,modules):
            output,drift=ns["collect_stretch"]({"query":{"db":db}},{"query":pred},"cpu",1,Path("/fixture/progress.json"))
        return output["query"],pred,drift,calls,w
    def test_identity_is_exact_tree_is_unchanged_and_classification_is_untouched(self):
        out,pred,drift,calls,w=self.run_fixture()
        for regime in ("D0","H0"):
            np.testing.assert_array_equal(out["stretch_"+regime][:,0],pred["share_"+regime])
            np.testing.assert_array_equal(out["classifier_"+regime],pred["classifier_"+regime])
            for j in (0,1):
                np.testing.assert_array_equal(out["stretch_"+regime][:,:,j],np.repeat(pred["share_"+regime][:,j:j+1],4,axis=1))
            self.assertTrue((np.diff(out["stretch_"+regime],axis=1)<=1e-7).all())
        self.assertEqual(calls.count("forward"),12) # 3 families × 2 regimes × 2 batches.
        self.assertEqual(calls.count("eval"),6)
        self.assertEqual(len(drift),6)
        self.assertTrue(all(d["max_identity_reconstruction_drift"]<.002 for d in drift))
        self.assertNotIn("target_share",out)
        np.testing.assert_allclose(w.sum((1,2)),1,atol=1e-6)
    def test_excessive_reconstruction_drift_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"reconstruction drift"):
            self.run_fixture(offset=.003)


class SavedDataIntegrity(unittest.TestCase):
    def test_attach_rejects_id_order_and_label_corruption(self):
        fn=functions_only(DRIVER)["attach"]
        data={"sample_id":np.array(["a","b"]),"share":np.array([0.,.5]),"cell_share":np.zeros((2,15,25))}
        pred={"sample_id":np.array(["a","b"]),"target_share":data["share"].copy()}
        good=fn(pred,data)
        self.assertEqual(good["support_minutes"].tolist(),[0,0])
        with self.assertRaisesRegex(ValueError,"ID/order"):
            fn({**pred,"sample_id":pred["sample_id"][::-1]},data)
        with self.assertRaisesRegex(ValueError,"labels"):
            fn({**pred,"target_share":np.array([0.,.4])},data)
    def test_source_pin_detects_byte_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/"source.py";p.write_bytes(b"frozen source\n")
            digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            fn=functions_only(ROOT/"pipeline/training_r16_multisupport.py",names={"verify"},ROOT=root,sha=digest)["verify"]
            pins={"source.py":digest(p)};fn(pins)
            p.write_bytes(b"modified source\n")
            with self.assertRaisesRegex(ValueError,"Frozen source changed"):fn(pins)
    def test_csv_exact_roundtrip_bounds_uniqueness_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            ns=functions_only(ROOT/"ops/finalize_r16_production.py",names={"write_submission"},
                safe=lambda p:Path(p),relative=lambda p:str(Path(p).relative_to(root)),sha=digest)
            fn=ns["write_submission"];ids=np.array(["쉼표,ID","second","third"])
            pred={"share":np.array([0.,np.nextafter(.5,1.),1.]),"score":np.array([1.,.2,0.])}
            receipt=fn(root/"candidate.csv",ids,pred)
            with (root/receipt["path"]).open(newline="") as f:rows=list(csv.DictReader(f))
            self.assertEqual([r["ID"] for r in rows],ids.tolist())
            np.testing.assert_array_equal([float(r["share_added_vessel"]) for r in rows],pred["share"])
            self.assertEqual(receipt["sha256"],digest(root/"candidate.csv"))
            with self.assertRaises(FileExistsError):fn(root/"candidate.csv",ids,pred)
            with self.assertRaises(ValueError):fn(root/"duplicate.csv",np.array(["same"]*3),pred)
            with self.assertRaises(ValueError):fn(root/"nan.csv",ids,{**pred,"share":np.array([np.nan,.2,.3])})
            with self.assertRaises(ValueError):fn(root/"bounds.csv",ids,{**pred,"score":np.array([1.01,.2,.3])})


class WorkflowOrdering(unittest.TestCase):
    def test_protocol_decision_report_confirmation_query_have_strict_order(self):
        tree=ast.parse(DRIVER.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run")
        def first_call(name, text=None):
            hits=[n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                  and n.func.id==name and (text is None or any(text in x.value for x in ast.walk(n) if isinstance(x,ast.Constant) and isinstance(x.value,str)))]
            self.assertTrue(hits,(name,text));return min(hits)
        pro=first_call("write","PROTOCOL.json");choose=first_call("select")
        decision=first_call("write","DECISION_FREEZE.json");report=first_call("panel_data","local_report")
        confirmation=first_call("write","REPORT_CONFIRMATION.json");query=first_call("read_raw_csv")
        self.assertLess(pro,choose);self.assertLess(choose,decision);self.assertLess(decision,report)
        self.assertLess(report,confirmation);self.assertLess(confirmation,query)
        select_lines=[n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="select"]
        self.assertTrue(all(line<decision for line in select_lines))
        # A source verifier must surround selection/application, not only capture
        # source hashes as decorative metadata.
        verifies=[n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                  and n.func.id=="verify" and "code_sha256" in ast.unparse(n)]
        self.assertTrue(any(line<choose for line in verifies))
        self.assertTrue(any(decision<line<query for line in verifies))
        self.assertTrue(any(query<line for line in verifies))
        # Early original artifact pin construction may read metadata, but may not
        # hash original query CSV bytes before the decision is frozen.
        early=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id=="sha" and n.lineno<decision]
        self.assertFalse(any("submission" in ast.unparse(n) or "original_csv" in ast.unparse(n) for n in early))


if __name__=="__main__":
    unittest.main(verbosity=2)
