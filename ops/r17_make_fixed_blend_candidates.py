#!/usr/bin/env python3
"""Export user-prescribed B1/B2 blends from pinned saved predictions only.

No model loading/inference/training, coefficient fitting, or official upload.
The separate verification process independently reloads every source and export.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submissions/r17_fixed_blends_20261030"
AUDIT = ROOT / "reports/r17_eda_audit_20261030/task1"
QUERY = ROOT / "models/r16_v1_fourway_r2/query_predictions.npz"
TEST = ROOT / "data/raw/Test.csv"
MEMBERS = ["lightgbm5", "catboost5", "dasheng_base", "dinov2_small", "timemoe"]
COLUMNS = ["ID", "share_added_vessel", "score_added_vessel"]
SPECS = {
    "B1": {"audit_key": "B_L60_DINO25_Dash15", "weights": [.60, 0., .15, .25, 0.],
           "filename": "submission_r17_B1_L60_DINO25_Dash15.csv"},
    "B2": {"audit_key": "B_L50_DINO30_Dash20", "weights": [.50, 0., .20, .30, 0.],
           "filename": "submission_r17_B2_L50_DINO30_Dash20.csv"},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def atomic_bytes(path, data):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def sources():
    protocol_path = AUDIT / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["member_order"] == MEMBERS
    pins = dict(protocol["source_sha256"])
    for rel, expected in pins.items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f"Previously pinned source changed: {rel}")
    for p in [protocol_path, AUDIT / "rows_Test_descriptive_only.csv",
              AUDIT / "Test_descriptive_statistics.csv", TEST, Path(__file__)]:
        pins[str(p.relative_to(ROOT))] = sha(p)
    with np.load(QUERY, allow_pickle=False) as z:
        q = {k: z[k] for k in ["sample_id", "share_D0", "share_H0", "classifier_D0", "classifier_H0"]}
    ids = q["sample_id"].astype(str)
    assert ids.shape == (1500,) and len(set(ids)) == 1500
    for k in ["share_D0", "share_H0", "classifier_D0", "classifier_H0"]:
        assert q[k].shape == (1500, 5) and np.isfinite(q[k]).all()
        assert np.all((q[k] >= 0) & (q[k] <= 1))
    # Only official IDs are read; Test input values never fit/select any policy.
    header, test_rows = csv_rows(TEST)
    assert header[0] == "ID" and len(header) == 376 and len(test_rows) == 1500
    assert [r["ID"] for r in test_rows] == ids.tolist()
    _, audited = csv_rows(AUDIT / "rows_Test_descriptive_only.csv")
    assert [r["sample_id"] for r in audited] == ids.tolist()
    _, statistics = csv_rows(AUDIT / "Test_descriptive_statistics.csv")
    return q, ids, audited, statistics, pins


def expected(q, spec):
    w = np.asarray(spec["weights"], dtype=np.float64)
    assert abs(w.sum() - 1) < 1e-12 and np.all(w >= 0)
    return np.column_stack([
        np.clip(.5 * (q["share_D0"] + q["share_H0"]) @ w, 0, 1),
        np.clip(.5 * (q["classifier_D0"] + q["classifier_H0"]) @ w, 0, 1),
    ])


def describe(values, reference):
    qs = np.quantile(values, [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1])
    names = ["min", "p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max"]
    return {"mean": float(values.mean()), "std": float(values.std()),
            "rms": float(np.sqrt(np.mean(values**2))),
            **dict(zip(names, map(float, qs))),
            "ge090_fraction": float(np.mean(values >= .90)),
            "R9_pearson": float(pearsonr(values, reference).statistic),
            "R9_spearman": float(spearmanr(values, reference).statistic)}


def verify(label):
    q, ids, audited, statistics, pins = sources()
    spec = SPECS[label]
    path = OUT / spec["filename"]
    header, rows = csv_rows(path)
    assert header == COLUMNS and len(rows) == 1500
    assert [r["ID"] for r in rows] == ids.tolist()
    actual = np.asarray([[float(r[c]) for c in COLUMNS[1:]] for r in rows], dtype=np.float64)
    assert actual.shape == (1500, 2) and np.isfinite(actual).all()
    assert np.all((actual >= 0) & (actual <= 1))
    calculated = expected(q, spec)
    np.testing.assert_array_equal(actual, calculated)
    old = np.asarray([[float(r[p + "_" + spec["audit_key"]]) for p in ["share", "score"]]
                      for r in audited])
    np.testing.assert_array_equal(actual, old)
    stats = {}
    for j, output in enumerate(["share", "score"]):
        reference = np.asarray([float(r["R9_" + output]) for r in audited])
        stats[output] = describe(actual[:, j], reference)
        previous = next(r for r in statistics if r["model_or_mixture"] == spec["audit_key"] and r["output"] == output)
        for name, v in stats[output].items():
            assert abs(v - float(previous[name])) <= 2e-14, (label, output, name, v, previous[name])
    for rel, digest in pins.items():
        assert sha(ROOT / rel) == digest, f"Source changed during export: {rel}"
    return {"label": label, "path": str(path), "sha256": sha(path), "rows": 1500,
            "columns": COLUMNS, "weights_in_member_order": spec["weights"],
            "D0_H0": [.5, .5], "score_share_weight": 0,
            "exact_roundtrip_to_saved_formula": True, "exact_match_to_prior_audit_rows": True,
            "prior_audit_statistics_tolerance": 2e-14,
            "unique_ids": True, "official_Test_ID_order": True,
            "finite": True, "range_0_1": True, "statistics": stats,
            "verified_in_separate_process": True, "source_sha256": pins}


def run():
    q, ids, _, _, initial_pins = sources()
    OUT.mkdir(parents=True, exist_ok=True)
    for label, spec in SPECS.items():
        pred = expected(q, spec)
        path = OUT / spec["filename"]
        # Existing candidates are preserved and verified, never silently replaced.
        if not path.exists():
            import io
            stream = io.StringIO(newline="")
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(COLUMNS)
            writer.writerows((sid, format(s, ".17g"), format(c, ".17g"))
                             for sid, (s, c) in zip(ids, pred))
            atomic_bytes(path, stream.getvalue().encode("utf-8"))
    results = []
    for label in SPECS:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--verify", label],
                                   check=True, capture_output=True, text=True)
        results.append(json.loads(completed.stdout))
    assert all(r["source_sha256"] == initial_pins for r in results)
    manifest = {"created_KST": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                "member_order": MEMBERS, "policy": "User-prescribed fixed B1/B2, independent probability score, D0/H0 50:50, clip[0,1] only",
                "no_training": True, "no_model_inference": True, "no_weight_fitting": True,
                "official_upload": False, "official_LB_for_these_candidates": "unmeasured",
                "Test_labels_used": False, "Test_used_for_weight_selection": False,
                "limitations": "Retrospective local panels have been reused; legacy LGB early stopping exposure remains. Fixed blends cannot remove that exposure. Local improvement does not guarantee Public/Private improvement.",
                "candidates": results, "source_sha256": initial_pins}
    atomic_bytes(OUT / "candidate_manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
    lines = ["# R17 B1/B2 고정 배합 제출 후보", "", "저장된 예측값만 결합했습니다. 재학습·모델 추론·가중치 적합·공식 제출은 수행하지 않았습니다.", "",
             "| 후보 | LightGBM | DINOv2-small | Dasheng-Base | CatBoost / TimeMoE | D0:H0 |",
             "|---|---:|---:|---:|---:|---:|", "| B1 | 60% | 25% | 15% | 0% | 50:50 |",
             "| B2 | 50% | 30% | 20% | 0% | 50:50 |", "",
             "회귀와 분류 모두 위 비율을 사용합니다. `score_added_vessel`은 독립 분류기 확률의 가중평균이며 `share`가 섞이는 비율은 0입니다. 후처리는 [0,1] clip만 적용했습니다.", "",
             "| 후보 | share 평균 | share RMS | share 중앙값 | score 평균 | score ≥0.90 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        s, c = r["statistics"]["share"], r["statistics"]["score"]
        lines.append(f'| {r["label"]} | {s["mean"]:.8f} | {s["rms"]:.8f} | {s["median"]:.8f} | {c["mean"]:.8f} | {100*c["ge090_fraction"]:.2f}% |')
    lines += ["", "각 파일 1,500행, 공식 Test ID 순서·중복 없음·정확한 3열·유한값·[0,1] 범위를 별도 Python 프로세스에서 재검증했습니다. 내보낸 CSV를 다시 읽은 값은 원본 NPZ 재계산 및 R17 감사 저장 행과 완전히 일치합니다. 감사 요약 통계도 허용오차 2e-14 이내로 일치합니다.", "",
              "두 후보의 공식 리더보드 성적은 미측정입니다. 기존 LightGBM 조기 종료의 선택셋 노출과 반복 사용한 로컬 패널의 한계는 남아 있습니다.", "", "## SHA-256", ""]
    for r in results:
        lines += [f'- `{Path(r["path"]).name}`: `{r["sha256"]}`']
    lines += ["", "전체 분위수, R9와의 Pearson/Spearman 상관계수, 원본 SHA 고정값은 `candidate_manifest.json`에 보존했습니다.", ""]
    atomic_bytes(OUT / "README.md", "\n".join(lines).encode())
    sums = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "SHA256SUMS.json"}
    atomic_bytes(OUT / "SHA256SUMS.json", (json.dumps(sums, indent=2) + "\n").encode())
    print(json.dumps({"output_dir": str(OUT), "candidates": [{"label": r["label"], "path": r["path"], "sha256": r["sha256"], "statistics": r["statistics"]} for r in results], "manifest_sha256": sha(OUT / "candidate_manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", choices=list(SPECS))
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify), ensure_ascii=False))
    else:
        run()
