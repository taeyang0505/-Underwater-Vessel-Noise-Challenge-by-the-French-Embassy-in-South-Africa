"""Read-only baseline comparison on the already completed R2 OOF.

Never opens Test.csv, fits models, changes policies, or submits predictions.
The public benchmark algorithm is not bound to the starter by published evidence.
"""
from pathlib import Path
import ast
import hashlib
import io
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, pred):
    return {
        "n": len(y),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mae": float(np.mean(np.abs(pred - y))),
        "bias": float(np.mean(pred - y)),
    }


def check_reference_conversion(starter):
    """Compare the published converter on generated audio, never competition inputs."""
    import soundfile as sf
    from pipeline.audio import audio_band_power
    source = "\n".join("".join(c.get("source", [])) for c in starter["cells"] if c["cell_type"] == "code")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "minute_band_power")
    scope = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "official_converter_function_only", "exec"), scope)
    fs = 32000
    t = np.arange(60 * fs) / fs
    rng = np.random.default_rng(42)
    samples = (0.1 * np.sin(2 * np.pi * 100 * t) + 0.02 * np.sin(2 * np.pi * 1000 * t)
               + rng.normal(0, 0.01, len(t))).astype(np.float32)
    wav = io.BytesIO()
    sf.write(wav, samples, fs, format="WAV", subtype="FLOAT")
    expected = scope["minute_band_power"](samples, fs)
    _, actual, _ = audio_band_power(wav.getvalue())
    assert actual.shape == expected.shape == (1, 25)
    rel = np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-30)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-12)
    return {"fixture": "60s seeded generated waveform, 32kHz, float32 WAV", "bands": 25,
            "max_relative_power_error": float(rel.max()), "passed": True,
            "limitation": "Verifies numeric integration on this fixture, not recorder calibration or simulator fidelity."}


def route_summary(path, share):
    recipe = json.loads(path.read_text())
    items = [item for batch in recipe["batches"] for item in batch["items"]]
    assert len(items) == len(share)
    routes = []
    for item in items:
        refs = item["references"]
        native = any(r["kind"] == "native_contiguous_15min_crop" for r in refs)
        if not refs:
            route = "zero"
        elif native:
            route = "native_included"
        elif len(refs) == 1:
            route = "single_short"
        else:
            route = "multi_short"
        routes.append(route)
    routes = np.array(routes)
    result = {}
    for route in sorted(set(routes)):
        idx = routes == route
        result[route] = {"n": int(idx.sum()), "fraction": float(idx.mean()),
                         "share_mean": float(share[idx].mean()), "share_max": float(share[idx].max()),
                         "share_above_0.4_fraction": float(np.mean(share[idx] > 0.4))}
    return {"recipe_sha256": sha(path), "routes": result}


def main():
    oof_path = ROOT / "ops/r2_progress/oof_predictions.npz"
    meta_path = ROOT / "ops/r2_progress/oof_metadata.parquet"
    starter_path = ROOT / "data/raw/maritime_starter_notebook.ipynb"
    inputs = [oof_path, meta_path, starter_path, ROOT / "pipeline/audio.py"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    starter = json.loads(starter_path.read_text())
    code = "\n".join("".join(c.get("source", [])) for c in starter["cells"] if c["cell_type"] == "code")
    assert 'submission["share_added_vessel"] = 0.1' in code
    assert 'submission["score_added_vessel"] = 0.5' in code
    meta = pd.read_parquet(meta_path)
    with np.load(oof_path, allow_pickle=False) as z:
        assert np.array_equal(z["sample_id"], meta["sample_id"].to_numpy())
        assert meta["sample_id"].is_unique and len(meta) == 15000
        y = z["share_true"].astype(np.float64)
        assert np.isfinite(y).all() and ((0 <= y) & (y <= 1)).all()
        predictions = {
            "published_starter_constant_0.1": np.full(len(y), 0.1),
            "r2_cnn": z["cnn_share"].copy(),
            "r2_ensemble": z["ensemble_share"].copy(),
        }
    train_mean = np.empty(len(y))
    folds = []
    for fold in range(5):
        train_path = ROOT / f"data/synthetic/group_cv_physics_v2_r1/fold_{fold:02d}/train.npz"
        train_hash = sha(train_path)
        with np.load(train_path, allow_pickle=False) as z:
            train_y = z["share"].astype(np.float64)
            mean = float(train_y.mean())
            n = len(train_y)
        idx = meta["fold"].to_numpy() == fold
        assert idx.sum() == 3000
        train_mean[idx] = mean
        folds.append({
            "fold": fold, "train_n": n, "train_share_mean": mean,
            "train_sha256": train_hash,
            "outer_n": int(idx.sum()),
            "metrics": {k: metrics(y[idx], v[idx]) for k, v in predictions.items()},
            "train_routes": route_summary(train_path.with_name("train_recipes.json"), train_y),
        })
        validation_path = train_path.with_name("validation.npz")
        with np.load(validation_path, allow_pickle=False) as z:
            inner_y = z["share"].astype(np.float64)
        folds[-1]["inner_routes"] = route_summary(train_path.with_name("validation_recipes.json"), inner_y)
        folds[-1]["inner_npz_sha256"] = sha(validation_path)
    predictions["fold_train_mean_constant"] = train_mean
    for fold, item in enumerate(folds):
        idx = meta["fold"].to_numpy() == fold
        item["metrics"]["fold_train_mean_constant"] = metrics(y[idx], train_mean[idx])
    boundary = meta["boundary_candidate_flag"].to_numpy(dtype=bool)
    result = {
        "scope": "Completed own-synthetic OOF diagnosis; no fitting or leaderboard probing",
        "official_public": {
            "benchmark_score": 0.753867825,
            "benchmark_rmse_from_score": 1 - 0.753867825,
            "benchmark_rmse_displayed": 0.246132174,
            "submitted_v1_score": 0.658591231,
            "submitted_v1_rmse": 1 - 0.658591231,
            "rmse_gap": 0.753867825 - 0.658591231,
            "rmse_relative_increase": (1 - 0.658591231) / (1 - 0.753867825) - 1,
            "benchmark_same_algorithm_as_starter_authoritatively_confirmed": False,
            "source": "https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/leaderboard",
        },
        "starter": {"share_constant": 0.1, "evidence_constant": 0.5, "model_training": False},
        "audio_conversion_parity": check_reference_conversion(starter),
        "own_oof": {
            "n": len(y), "share_mean": float(y.mean()),
            "share_std": float(y.std()), "zero_fraction": float(np.mean(y == 0)),
            "metrics": {k: metrics(y, v) for k, v in predictions.items()},
            "boundary_excluded_metrics": {k: metrics(y[~boundary], v[~boundary]) for k, v in predictions.items()},
            "folds": folds,
        },
        "no_test_data_opened": True,
        "no_model_or_weight_fitting": True,
        "input_sha256": before,
        "limitations": [
            "Public benchmark and custom OOF have different examples and target generation; RMSE is not directly comparable.",
            "Constant-0.1 starter evaluation is not evidence that the public Benchmark uses the identical algorithm.",
            "OOF has already informed development; it is not a new independent final evaluation.",
            "No public score is available for R2 or R3.",
        ],
    }
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    output = ROOT / "reports/OFFICIAL_BASELINE_COMPARISON_FACTS.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "own_oof_metrics": result["own_oof"]["metrics"],
                      "official_public": result["official_public"], "inputs_unchanged": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
