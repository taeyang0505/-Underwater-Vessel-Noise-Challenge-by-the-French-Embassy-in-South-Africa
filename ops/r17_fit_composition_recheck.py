#!/usr/bin/env python3
"""Retrospective saved-output checks under fixed composition/weight protocols.

No training, inference, coefficient optimization, Test access, or submission.
Removing rows/reweighting old fit does not undo legacy checkpoint selection.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports/r17_eda_audit_20261030"
OUT = ROOT / "reports/r17_clean_pipeline_r1/fit_recheck"
MEMBERS = ["lightgbm5", "catboost5", "dasheng_base", "dinov2_small", "timemoe"]
MIXES = {
    "A_original": None,
    "B1_L60_DINO25_Dash15": np.array([.60, 0, .15, .25, 0]),
    "B2_L50_DINO30_Dash20": np.array([.50, 0, .20, .30, 0]),
    "D_balanced4": np.array([.25, .25, .25, .25, 0]),
}


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def kish(w):
    return float(np.sum(w)**2 / np.sum(w*w))


def median_weighted(x, w):
    order = np.argsort(x)
    return float(np.asarray(x)[order][np.searchsorted(np.cumsum(w[order]), .5*w.sum())])


def metrics(y, pred, score, w):
    return {"RMSE": float(np.sqrt(np.average((pred-y)**2, weights=w))),
            "bias": float(np.average(pred-y, weights=w)),
            "AUC": float(roc_auc_score(y > 0, score, sample_weight=w)),
            "target_mean": float(np.average(y, weights=w)),
            "prediction_mean": float(np.average(pred, weights=w))}


def table(frame, columns=None):
    return frame[columns].to_markdown(index=False, floatfmt=".6f") if columns else frame.to_markdown(index=False, floatfmt=".6f")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    feature_path = AUDIT / "eda/ensemble_fit.parquet"
    rows_path = AUDIT / "task1/rows_ensemble_fit.csv"
    npz_path = ROOT / "models/r16_v1_fourway_r2/predictions/selection.npz"
    policy_path = ROOT / "models/r16_pretrained_cuda_bf16_r1/both/final/policy.json"
    protocol_path = AUDIT / "task1/protocol.json"
    paths = [feature_path, rows_path, npz_path, policy_path, protocol_path, Path(__file__)]
    pins = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    prior = json.loads(protocol_path.read_text())
    for p in [npz_path, policy_path]:
        assert pins[str(p.relative_to(ROOT))] == prior["source_sha256"][str(p.relative_to(ROOT))]
    policy = json.loads(policy_path.read_text())
    assert policy["member_order"] == MEMBERS
    f = pd.read_parquet(feature_path)
    row = pd.read_csv(rows_path, float_precision="round_trip")
    assert f.sample_id.is_unique and row.sample_id.is_unique
    row = row.set_index("sample_id").loc[f.sample_id].reset_index()
    np.testing.assert_array_equal(row.sample_id, f.sample_id)
    with np.load(npz_path, allow_pickle=False) as z:
        ix = z["ensemble_fit_row_indices"]
        assert len(ix) == 1206
        lookup = {s: i for i, s in enumerate(z["sample_id"].astype(str))}
        index = np.asarray([lookup[s] for s in f.sample_id])
        assert set(index) == set(ix)
        q = {k: z[k][index] for k in ["share_D0", "share_H0", "classifier_D0", "classifier_H0"]}
        y = z["target_share"][index]
    np.testing.assert_array_equal(y, f.share.to_numpy())
    np.testing.assert_array_equal(y, row.target_share.to_numpy())
    predictions = {}
    s = .5*(q["share_D0"] + q["share_H0"])
    c = .5*(q["classifier_D0"] + q["classifier_H0"])
    for j, member in enumerate(MEMBERS):
        predictions[member] = (s[:, j], c[:, j])
        np.testing.assert_allclose(s[:, j], f["pred_"+member], atol=1e-15, rtol=0)
        np.testing.assert_allclose(c[:, j], f["score_"+member], atol=1e-15, rtol=0)
    for mix, w in MIXES.items():
        if w is None:
            sm = sum(.5*q["share_"+r] @ np.asarray(policy["experts"][r]["regression_weights"]) for r in ["D0", "H0"])
            cm = sum(.5*q["classifier_"+r] @ np.asarray(policy["experts"][r]["independent_classifier_weights"]) for r in ["D0", "H0"])
        else:
            sm, cm = s@w, c@w
        predictions[mix] = (np.clip(sm, 0, 1), np.clip(cm, 0, 1))
        key = {"A_original": "A_original", "B1_L60_DINO25_Dash15": "B_L60_DINO25_Dash15",
               "B2_L50_DINO30_Dash20": "B_L50_DINO30_Dash20", "D_balanced4": "D_balanced4"}[mix]
        np.testing.assert_allclose(predictions[mix][0], row["share_"+key], atol=1e-15, rtol=0)
        np.testing.assert_allclose(predictions[mix][1], row["score_"+key], atol=1e-15, rtol=0)
    density = pd.cut(f.n_obs, bins=[0, 5, 14, 15], labels=["sparse_1_5", "middle_6_14", "complete_15"])
    assert density.notna().all()
    f["density_bin"] = density.astype(str)
    protocols = [
        ("original_all_unweighted", np.ones(len(f), bool), None),
        ("dense_nobs_ge12_unweighted", (f.n_obs >= 12).to_numpy(), None),
        ("dense_nobs_ge12_site_equal", (f.n_obs >= 12).to_numpy(), "site"),
        ("all_site_equal", np.ones(len(f), bool), "site"),
        ("all_three_density_bins_equal", np.ones(len(f), bool), "density_bin"),
    ]
    comp, scores, boot, allocations = [], [], [], []
    for name, mask, balance in protocols:
        sub = f.loc[mask].copy()
        w = np.ones(len(sub)) if balance is None else 1/sub.groupby(balance)[balance].transform("count").to_numpy()
        w = w/w.sum()
        for field in ["site", "n_obs", "density_bin", "background_group"]:
            for value in sorted(sub[field].unique()):
                part = sub[field] == value
                allocations.append({"protocol": name, "field": field, "value": str(value), "rows": int(part.sum()), "weight_fraction": float(w[part].sum())})
        sw = pd.Series(w, index=sub.index)
        bw = sw.groupby(sub.base_window_index).sum().to_numpy()
        gw = sw.groupby(sub.background_group).sum().to_numpy()
        comp.append({"protocol": name, "rows": len(sub), "unique_base_windows": sub.base_window_index.nunique(),
                     "unique_background_source_groups": sub.background_group.nunique(), "sites": sub.site.nunique(),
                     "nobs5_row_fraction": float((sub.n_obs == 5).mean()), "nobs5_weight_fraction": float(w[sub.n_obs == 5].sum()),
                     "SF5_row_fraction": float(((sub.n_obs == 5)&sub.site.str.contains("St_Francis")).mean()),
                     "SF5_weight_fraction": float(w[(sub.n_obs == 5)&sub.site.str.contains("St_Francis")].sum()),
                     "row_weight_Kish_ESS": kish(w), "base_window_weight_Kish_ESS": kish(bw),
                     "background_source_weight_Kish_ESS": kish(gw),
                     "HF_bump_median_unweighted": float(sub.bump_high_db.median()),
                     "HF_bump_weighted_median": median_weighted(sub.bump_high_db.to_numpy(), w),
                     "zero_weight_fraction": float(w[y[mask] == 0].sum())})
        start = len(scores)
        for model, (pred, score) in predictions.items():
            scores.append({"protocol": name, "model": model, **metrics(y[mask], pred[mask], score[mask], w)})
        rank = pd.DataFrame(scores[start:]).set_index("model").RMSE.rank(method="min")
        for item in scores[start:]:
            item["RMSE_rank_among_9"] = int(rank[item["model"]])
        # Paired bootstrap of unique 15-min background windows, retaining all six
        # operator variants of each window. It does not create new sites/recorders.
        groups, inverse = np.unique(sub.base_window_index, return_inverse=True)
        denominator = np.bincount(inverse, weights=w)
        sums = {m: np.bincount(inverse, weights=w*(p[mask]-y[mask])**2) for m, (p, _) in predictions.items()}
        rng = np.random.default_rng(20261030)
        draws = rng.integers(0, len(groups), size=(1000, len(groups)))
        a = np.sqrt(sums["A_original"][draws].sum(1)/denominator[draws].sum(1))
        for model in list(MIXES)[1:] + ["dinov2_small", "catboost5"]:
            delta = np.sqrt(sums[model][draws].sum(1)/denominator[draws].sum(1))-a
            boot.append({"protocol": name, "model_minus_A": model, "base_windows": len(groups),
                         "source_groups": sub.background_group.nunique(), "repeats": 1000,
                         "RMSE_delta_p025": float(np.quantile(delta, .025)), "RMSE_delta_median": float(np.median(delta)),
                         "RMSE_delta_p975": float(np.quantile(delta, .975)),
                         "caution": "retrospective within-old-fit base-window bootstrap; only 3-5 source groups; no official uncertainty"})
    cf, sf, bf = pd.DataFrame(comp), pd.DataFrame(scores), pd.DataFrame(boot)
    cf.to_csv(OUT/"composition_protocols.csv", index=False)
    sf.to_csv(OUT/"fixed_models_and_blends_metrics.csv", index=False)
    pd.DataFrame(allocations).to_csv(OUT/"composition_allocations.csv", index=False)
    bf.to_csv(OUT/"paired_base_window_bootstrap.csv", index=False)
    f[["sample_id", "base_window_index", "background_group", "site", "recorder", "operator", "n_obs", "density_bin"]].to_csv(OUT/"audited_old_fit_rows.csv", index=False)
    pivot = sf.pivot(index="model", columns="protocol", values="RMSE").reset_index()
    aucpivot = sf.pivot(index="model", columns="protocol", values="AUC").reset_index()
    result = {"created_KST": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
              "scope": "retrospective old fit composition sensitivity only", "no_training": True,
              "no_inference": True, "no_weight_optimization": True, "Test_accessed": False,
              "member_order": MEMBERS,
              "mixtures": {k: None if v is None else v.tolist() for k,v in MIXES.items()},
              "composition": comp, "metrics": scores, "bootstrap": boot, "source_sha256": pins,
              "legacy_exposure": "Original LGB early stopping used all 2400 local_fit rows. Removing/reweighting current1206 rows cannot erase exposure; all existing models remain frozen."}
    (OUT/"result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    lines = ["# R17 기존 ensemble_fit 구성 편중 재검산", "",
             "저장 예측·기존 행 메타데이터만 사용했습니다. 학습·추론·가중치 최적화·Test 접근은 0회입니다. B1/B2/D 가중치와 D0:H0=50:50은 고정했습니다. A는 제출한 원본 정책을 보존했습니다.", "",
             "## 관측 밀도·사이트 구성", "", table(cf, ["protocol", "rows", "unique_base_windows", "unique_background_source_groups", "sites", "nobs5_row_fraction", "nobs5_weight_fraction", "base_window_weight_Kish_ESS", "HF_bump_median_unweighted", "HF_bump_weighted_median"]), "",
             "원본 1,206행은 고유 배경 창 201개에 생성 연산자 6종을 적용한 표본입니다. `Nobs=5 / St Francis`는 정확히 594행(49.2537%)입니다. Nobs≥12 제한은 594행을 제외해 612행·102배경·3사이트가 남고, 이때 SF 사례 자체가 모두 제외됩니다. 따라서 고밀도 제한은 순수한 결측 밀도 개입 실험이 아닙니다.", "",
             "3개 밀도 구간 동일 가중은 ≤5분, 6~14분, 15분을 각각 1/3로 고정했습니다. 해당 표본의 중간 구간은 Nobs12 78행과 Nobs14 6행으로 구성됩니다. 희소 구간의 양성/음성 관계는 이 재가중만으로 제거되지 않습니다.", "",
             "## 회귀 RMSE (작을수록 좋음)", "", table(pivot), "",
             "## 독립 분류 AUC (클수록 좋음)", "", table(aucpivot), "",
             "## 해석", "",
             "- 원본 fit에서는 CatBoost RMSE 0.093146, DINOv2 0.122158입니다. 고밀도 행만 남기면 DINOv2 0.068965, CatBoost 0.069392로 근소하게 순위가 바뀝니다. 관측 밀도·사이트 혼합이 상대 성적에 큰 영향을 주는 것은 확인됩니다.",
             "- 그러나 고밀도 3사이트를 동일 가중하면 CatBoost 0.058785, DINOv2 0.068057로 다시 CatBoost가 우수합니다. 전체 4사이트 동일 가중에서도 CatBoost가 우수합니다. 즉, `SF5가 절반이라서 Tree가 선택되었다`는 단일 설명만으로 모든 상대 순위나 공식 LB 악화를 확정할 수 없습니다.",
             "- Nobs5와 St Francis가 완전히 결합되어 있어 관측밀도와 사이트 효과가 분리되지 않습니다. 원래 fit에 없는 고밀도 St Francis를 재가중으로 만들 수 없습니다. 별도의 깨끗한 고밀도·다사이트 평가가 필요합니다.",
             "- 기존 LightGBM이 분리 전 local_fit으로 조기 종료한 노출은 남아 있습니다. 이 표는 데이터 구성 민감도 진단이며 새 독립 검증 성적이나 누수 제거 성적으로 부르면 안 됩니다.",
             "- 배경 창 단위 paired bootstrap 1,000회 결과는 `paired_base_window_bootstrap.csv`에 보존했습니다. 이 102~201개 창도 실제 배경 원천 그룹은 3~5개뿐이므로 신뢰구간은 해당 기존 표본 내 불확실성만 설명합니다.", "",
             "## 전체 RMSE·편향·AUC", "", table(sf, ["protocol", "model", "RMSE", "bias", "AUC", "prediction_mean", "RMSE_rank_among_9"]), "",
             "입력 SHA-256은 `result.json`, 출력 SHA-256은 `SHA256SUMS.json`에 보존했습니다. 후보를 새로 선택하거나 원본 정책을 변경하지 않았습니다.", ""]
    (OUT/"R17_FIT_COMPOSITION_RECHECK.md").write_text("\n".join(lines))
    for rel,digest in pins.items():
        assert sha(ROOT/rel) == digest
    sums = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "SHA256SUMS.json"}
    (OUT/"SHA256SUMS.json").write_text(json.dumps(sums, indent=2)+"\n")
    print(cf.to_string(index=False))
    print(pivot.to_string(index=False))
    print("REPORT", OUT/"R17_FIT_COMPOSITION_RECHECK.md")


if __name__ == "__main__":
    run()
