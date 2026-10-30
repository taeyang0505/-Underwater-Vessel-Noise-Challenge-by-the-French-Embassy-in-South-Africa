#!/usr/bin/env python3
"""R20 labelled saved-prediction audit: no fit/inference/Test or policy changes."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/r20_integrated_cpu_diagnostics_r1/alignment"
FINAL = ROOT / "reports/r18_production_r1/finalization"
R16 = ROOT / "models/r16_v1_fourway_r2/predictions"
R19 = ROOT / "reports/r19_multislice_law_transfer_r1"
FAMILIES = ("lightgbm", "dinov2_small", "dasheng_base")
PANELS = ("fresh_fit_D0", "fresh_fit_H0", "local_gate", "local_report", "domain_holdout",
          "native12_VS_hf_high_negative", "native12_VS_st_francis_dense_negative")
SLICE_NAMES = ("all", "zero", "positive_000_003", "positive_003_010", "positive_010_030",
               "positive_above030", "weak_000_010", "HF_bump_ge12", "LF_fraction_le065",
               "active_le3", "active_ge10", "Nobs15", "zero_HF_ge12", "weak_HF_ge12")
COMBOS = {"equal_all_zero_weak": ("all", "zero", "weak_000_010"),
          "equal_all_zero_weak_HF": ("all", "zero", "weak_000_010", "HF_bump_ge12"),
          "equal_all_zero_weak_long": ("all", "zero", "weak_000_010", "active_ge10"),
          "equal_all_zero_weak_LF": ("all", "zero", "weak_000_010", "LF_fraction_le065")}
PUBLIC = {
    "R9_Tree5": (.231871324, .548088888),
    "R9_all5": (.217535806, .555066666),
    "R16_Tree5": (.223143023, .568777777),
    "R16_V1": (.224766255, .568777777),
    "R17_B1": (.210528863, .572711111),
    "R18_B2": (.221276889, .573555555),
}
PINS: dict[str, str] = {}
READS: list[str] = []


def access_guard(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes)):
        return
    p = Path(str(args[0])); name = p.name.lower()
    if (name in {"test.csv", "sample_submission.csv", "frozen_fold_predictions.npz"}
            or "query" in name or (name.endswith(".csv") and "submission" in name)):
        raise PermissionError("R20 does not open Test/query/submission outputs: " + str(p))


sys.addaudithook(access_guard)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pin(path):
    p = Path(path); k = str(p.relative_to(ROOT))
    if k not in PINS:
        PINS[k] = sha(p); READS.append(k)
    return PINS[k]


def load(path):
    pin(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def read_json(path):
    pin(path)
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def physical(d):
    db = np.asarray(d["db"], float)
    observed = np.isfinite(db)
    hi = np.max(np.where(observed, db, -np.inf), axis=(1, 2))
    energy = np.where(observed, np.power(10., (db - hi[:, None, None]) / 10.), 0.)
    hf = energy[:, :, 20:25].sum(2)
    hf_db = np.full(hf.shape, np.nan)
    np.log10(hf, out=hf_db, where=hf > 0)
    hf_db *= 10.
    bump = np.nanmax(hf_db, axis=1) - np.nanmedian(hf_db, axis=1)
    lf = energy[:, :, :9].sum((1, 2)) / energy.sum((1, 2))
    n_obs = observed.any(2).sum(1)
    active = d.get("active_minutes")
    if active is None and "cell_share" in d:
        active = (np.asarray(d["cell_share"]) > 0).any(2).sum(1)
    if active is None:
        active = d.get("support_minutes", np.full(len(db), -1))
    return {"HF_bump": bump, "LF_fraction": lf, "n_obs": n_obs,
            "active_minutes": np.asarray(active, int)}


def slices(d, f):
    y = d["share"].astype(float); hf = f["HF_bump"] >= 12.; active = f["active_minutes"]
    return {
        "all": np.ones(len(y), bool), "zero": y == 0,
        "positive_000_003": (y > 0) & (y <= .03),
        "positive_003_010": (y > .03) & (y <= .10),
        "positive_010_030": (y > .10) & (y <= .30),
        "positive_above030": y > .30, "weak_000_010": (y > 0) & (y <= .10),
        "HF_bump_ge12": hf, "LF_fraction_le065": f["LF_fraction"] <= .65,
        "active_le3": (active >= 0) & (active <= 3), "active_ge10": active >= 10,
        "Nobs15": f["n_obs"] == 15,
        "zero_HF_ge12": (y == 0) & hf,
        "weak_HF_ge12": (y > 0) & (y <= .10) & hf,
    }


def panel_weights(d):
    w = d.get("weights", np.ones(len(d["share"])))
    # R19 weights[N,15,25] are physical cell weights, NOT observation sample weights.
    if np.ndim(w) != 1:
        w = np.ones(len(d["share"]))
    w = np.asarray(w, float)
    assert np.all(w > 0) and np.all(np.isfinite(w))
    return w


def metrics(y, s, c, w):
    n = len(y)
    if n == 0:
        return {"n": 0, "status": "empty_slice", "rmse": None, "mae": None, "bias": None,
                "r2": None, "r2_status": "empty_slice", "brier": None, "ece10": None,
                "roc_auc": None, "pr_auc": None, "auc_status": "empty_slice"}
    w = w / w.sum(); z = y > 0; err = s-y
    mse = float(w @ (err**2)); ym = float(w@y); var = float(w@((y-ym)**2))
    bins = np.minimum((c*10).astype(int), 9)
    ece = 0.
    for k in range(10):
        ix = bins == k
        if ix.any():
            mass = w[ix].sum()
            ece += abs(float(w[ix]@c[ix]) - float(w[ix]@z[ix]))
    two = np.unique(z).size == 2
    return {"n": n, "n_zero": int((~z).sum()), "n_positive": int(z.sum()),
            "status": "ok", "rmse": float(np.sqrt(mse)), "mae": float(w@abs(err)),
            "bias": float(w@err), "r2": None if var <= 1e-25 else 1.-mse/var,
            "r2_status": "zero_target_variance" if var <= 1e-25 else "ok",
            "brier": float(w@((c-z)**2)), "ece10": float(ece),
            "roc_auc": float(roc_auc_score(z,c,sample_weight=w)) if two else None,
            "pr_auc": float(average_precision_score(z,c,sample_weight=w)) if two else None,
            "auc_status": "ok" if two else "single_class",
            "prediction_mean": float(w@s), "prediction_rms": float(np.sqrt(w@(s*s))),
            "target_mean": ym, "score_mean": float(w@c), "score_ge090": float(w@(c>=.90)),
            "slice_weight_ess": float(1./(w@w)), "target_variance": var}


def ensure_pred(p, n):
    s,c = (np.asarray(v, float) for v in p)
    assert s.shape == c.shape == (n,)
    assert np.isfinite(s).all() and np.isfinite(c).all()
    assert s.min() >= -1e-6 and s.max() <= 1+1e-6 and c.min() >= -1e-6 and c.max() <= 1+1e-6
    return np.clip(s,0,1),np.clip(c,0,1)


def generations_models(arrays):
    pred = {}
    for generation in ("old", "new"):
        for family in FAMILIES:
            a = arrays[generation, family]
            for regime in ("D0", "H0"):
                pred[f"{generation}_{family}__{regime}"] = (a[f"share_{regime}"],a[f"prob_{regime}"])
            pred[f"{generation}_{family}__D0H0"] = tuple(.5*(a[k+"_D0"]+a[k+"_H0"]) for k in ("share","prob"))
        for blend, coefs in (("B1", (.60,.25,.15)), ("B2", (.50,.30,.20))):
            pred[f"{generation}_{blend}"] = tuple(sum(c*pred[f"{generation}_{f}__D0H0"][k]
                for c,f in zip(coefs,FAMILIES)) for k in (0,1))
    return pred


def load_standard(name):
    ip = FINAL/f"inputs/{name}.npz"; d = load(ip); arrays = {}
    assert d["db"].shape == (len(d["share"]),15,25)
    for generation in ("old","new"):
        for family in FAMILIES:
            p = FINAL/f"predictions/{generation}/{family}/{name}.npz"
            receipt = read_json(p.with_suffix(".json")); a = load(p)
            assert receipt["prediction_sha256"] == pin(p)
            assert receipt["input_sha256"] == pin(ip)
            assert np.array_equal(a["sample_id"], d["sample_id"])
            arrays[generation,family] = a
    pred = generations_models(arrays)
    pred["R16_Tree5"] = pred["old_lightgbm__D0H0"]
    pred["R17_B1"] = pred["old_B1"]
    pred["R18_B2"] = pred["new_B2"]
    if (R16/f"{name}.npz").exists():
        saved = load(R16/f"{name}.npz")
        assert np.array_equal(saved["sample_id"],d["sample_id"])
        assert np.allclose(saved["target_share"],d["share"],rtol=0,atol=1e-12)
        for j, family in enumerate(("lightgbm5","catboost5","dasheng_base","dinov2_small","timemoe")):
            pred[f"R16_saved_{family}"] = tuple(.5*(saved[k+"_D0"][:,j]+saved[k+"_H0"][:,j])
                for k in ("share","classifier"))
        # Original frozen policy uses regime-specific full-precision weights, not rounded percentages.
        policy = read_json(ROOT/"models/r16_pretrained_cuda_bf16_r1/both/final/policy.json")
        pred["R16_V1"] = tuple(sum(.5*(saved[k+"_"+r]@np.asarray(policy["experts"][r][wk]))
            for r in ("D0","H0")) for k,wk in (("share","regression_weights"),
                                                      ("classifier","independent_classifier_weights")))
        # The R17 B1 submission was formed directly from this original saved tensor.
        b = np.array([.60,0.,.15,.25,0.])
        p17 = tuple(.5*(saved[k+"_D0"]+saved[k+"_H0"])@b for k in ("share","classifier"))
        assert max(float(np.max(abs(x-y))) for x,y in zip(p17,pred["old_B1"])) < 1e-12
        pred["R17_B1"] = p17
    return d,pred


def load_laws():
    protocol = read_json(R19/"inference/PROTOCOL.json")
    hashes = read_json(R19/"inference/OUTPUT_SHA256.json")
    idx = load(R19/"inference/fixed_blend_predictions.npz")
    assert pin(R19/"inference/fixed_blend_predictions.npz") == hashes["fixed_blend_predictions.npz"]["sha256"]
    arrays = {}
    for gen in ("old","new"):
        for fam in FAMILIES:
            p = R19/f"inference/{gen}_{fam}.npz"
            assert pin(p) == hashes[p.name]["sha256"]
            arrays[gen,fam] = load(p)
    offset = 0
    for info in protocol["laws"]:
        path = ROOT/info["path"]; d=load(path); n=info["n"]
        assert pin(path) == protocol["input_sha256"][info["path"]]
        assert np.array_equal(idx["sample_id"][offset:offset+n],d["sample_id"])
        assert np.all(idx["law"][offset:offset+n] == info["law"])
        sub = {(g,f):{k:v[offset:offset+n] for k,v in a.items()} for (g,f),a in arrays.items()}
        pred=generations_models(sub)
        assert np.allclose(pred["old_B1"][0],idx["old_B1_share"][offset:offset+n],atol=1e-12,rtol=0)
        assert np.allclose(pred["new_B2"][0],idx["new_B2_share"][offset:offset+n],atol=2e-7,rtol=0)
        pred["R16_Tree5"]=pred["old_lightgbm__D0H0"]
        pred["R17_B1"]=pred["old_B1"];pred["R18_B2"]=pred["new_B2"]
        yield "R19_"+info["law"],d,pred
        offset += n
    assert offset == protocol["rows_total"]


def load_r9_aux():
    policy = read_json(ROOT/"submissions/r9_fold0_all5_cpu_r1/frozen_policy.json")
    w = np.asarray(policy["weights"])
    for role, filename in (("ensemble_fit","fit_member_predictions.npz"),("ensemble_gate","gate_member_predictions.npz")):
        d=load(ROOT/f"data.nosync/synthetic/group_cv_minute_full_r9_v1/fold_00/{role}.npz")
        a=load(ROOT/f"submissions/r9_fold0_all5_cpu_r1/{filename}")
        assert np.array_equal(d["sample_id"],a["sample_id"])
        pred={"R9_all5":(a["share"]@w,a["score"]@w)}
        # Fold0-only LGB is deliberately NOT labelled as official R9 Tree5.
        pred["R9_LGB_fold0_not_officialTree5"]=(a["share"][:,4],a["score"][:,4])
        yield "R9_legacy_"+role,d,pred


def correlation_rows(records, panel_names):
    lookup={(r["panel"],r["model"],r["slice"]):r for r in records}
    rows=[]
    candidates={k:(k,) for k in SLICE_NAMES}|COMBOS
    for panel in panel_names:
        for candidate,sl in candidates.items():
            matched=[]; local=[]; official=[]
            for model,(rmse,auc) in PUBLIC.items():
                sub=[lookup.get((panel,model,s)) for s in sl]
                if all(r is not None and r.get("rmse") is not None for r in sub):
                    matched.append(model);local.append(float(np.mean([r["rmse"] for r in sub])));official.append(rmse)
            valid=len(matched)>=4 and np.std(local)>0 and np.std(official)>0
            result={"panel":panel,"candidate":candidate,"slices":"|".join(sl),
                    "aggregation":"equal_mean_RMSE_no_fitting","n_matched":len(matched),
                    "models":"|".join(matched),"status":"posthoc_descriptive_only" if valid else "insufficient_common_models",
                    "pearson_public_rmse":float(pearsonr(local,official).statistic) if valid else None,
                    "spearman_public_rmse":float(spearmanr(local,official).statistic) if valid else None,
                    "local_order_low_error_first":"|".join(matched[i] for i in np.argsort(local)),
                    "public_order_same_subset":"|".join(matched[i] for i in np.argsort(official))}
            if len(sl)==1:
                mm=[];la=[];oa=[]
                for model,(_,auc) in PUBLIC.items():
                    r=lookup.get((panel,model,sl[0]))
                    if r and r.get("roc_auc") is not None:
                        mm.append(model);la.append(r["roc_auc"]);oa.append(auc)
                ok=len(mm)>=4 and np.std(la)>0 and np.std(oa)>0
                result.update({"n_auc_matched":len(mm),"auc_correlation_status":"posthoc_descriptive_only" if ok else "insufficient_common_models_or_single_class",
                               "spearman_public_auc":float(spearmanr(la,oa).statistic) if ok else None,
                               "pearson_public_auc":float(pearsonr(la,oa).statistic) if ok else None})
            rows.append(result)
    return rows


def table(rows, cols):
    def fmt(v):
        if v is None:return "미정의/미측정"
        return f"{v:.6f}" if isinstance(v,(float,np.floating)) else str(v)
    return "\n".join(["|"+"|".join(cols)+"|","|"+"|".join(["---"]*len(cols))+"|"]+
                      ["|"+"|".join(fmt(r.get(k)) for k in cols)+"|" for r in rows])


def run():
    started = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    assert (OUT/"PREREGISTERED_PROTOCOL.md").exists()
    if (OUT/"COMPLETED.json").exists():
        raise FileExistsError("Completed audit is immutable; choose a new output revision")
    pin(OUT/"PREREGISTERED_PROTOCOL.md");pin(Path(__file__))
    inputs={}; predictions={}
    for panel in PANELS:
        inputs[panel],predictions[panel]=load_standard(panel)
    for panel,d,p in load_laws():inputs[panel]=d;predictions[panel]=p
    for panel,d,p in load_r9_aux():inputs[panel]=d;predictions[panel]=p
    records=[];coverage=[];feature_rows=[];slice_rows=[]
    models=sorted(set(PUBLIC)|{m for p in predictions.values() for m in p})
    for panel,d in inputs.items():
        n=len(d["share"]); assert len(np.unique(d["sample_id"]))==n
        f=physical(d); masks=slices(d,f); pw=panel_weights(d)
        for i in range(n):
            feature_rows.append({"panel":panel,"sample_id":str(d["sample_id"][i]),"target_share":float(d["share"][i]),
                                 **{k:float(v[i]) for k,v in f.items()}})
        for sl,mask in masks.items():
            slice_rows.append({"panel":panel,"slice":sl,"n":int(mask.sum()),"n_positive":int(((d['share']>0)&mask).sum()),
                               "weight_mass":float(pw[mask].sum()/pw.sum()),
                               "active_metadata_available":bool(np.any(f["active_minutes"]>=0))})
        for model in models:
            exists=model in predictions[panel]
            coverage.append({"panel":panel,"model":model,"available":exists,"n":n if exists else 0,
                             "official_rmse":PUBLIC.get(model,(None,None))[0],
                             "status":"saved_labelled_predictions" if exists else "missing_saved_predictions_no_inference",
                             "notes":"R9 has different historical local panels; no cross-panel substitution" if model.startswith('R9') else ""})
            if not exists:continue
            s,c=ensure_pred(predictions[panel][model],n)
            for sl,mask in masks.items():
                records.append({"panel":panel,"model":model,"slice":sl,
                                **metrics(d["share"][mask].astype(float),s[mask],c[mask],pw[mask])})
    write_csv(OUT/"ALL_MODELS_PANELS_SLICES_METRICS.csv",records)
    write_csv(OUT/"MODEL_PANEL_COVERAGE.csv",coverage)
    write_csv(OUT/"SLICE_COUNTS_AND_WEIGHTS.csv",slice_rows)
    write_csv(OUT/"LOCAL_PHYSICAL_FEATURES.csv",feature_rows)
    public_rows=[{"model":m,"public_rmse":v[0],"public_auc":v[1]} for m,v in PUBLIC.items()]
    write_csv(OUT/"OFFICIAL_PUBLIC_SCORES_SUPPLIED.csv",public_rows)
    correlations=correlation_rows(records,list(inputs))
    write_csv(OUT/"PUBLIC_ALIGNMENT_POSTHOC_ALL_TRIALS.csv",correlations)
    valid=[r for r in correlations if r["spearman_public_rmse"] is not None]
    best=sorted(valid,key=lambda r:(-r["spearman_public_rmse"],-r["pearson_public_rmse"],r["panel"],r["candidate"]))
    write_csv(OUT/"PUBLIC_ALIGNMENT_POSTHOC_OBSERVED_RANKING.csv",best)
    # Verification against previously frozen old/new summary values is descriptive only.
    key={(r["panel"],r["model"],r["slice"]):r for r in records}
    keyrows=[key[p,m,"all"] for p in PANELS for m in ("old_B1","old_B2","new_B1","new_B2")]
    write_csv(OUT/"B1_B2_KEY_PANEL_SUMMARY.csv",keyrows)
    units={"panels":len(inputs),"unique_saved_model_names":len(models),"available_model_panel_pairs":sum(r["available"] for r in coverage),
           "missing_model_panel_pairs":sum(not r["available"] for r in coverage),"metric_rows":len(records),
           "slices_registered":len(SLICE_NAMES),"public_candidates_registered_per_panel":len(SLICE_NAMES)+len(COMBOS),
           "public_trials_total":len(correlations),"public_trials_nmatched_ge4":len(valid),
           "official_models":len(PUBLIC),"official_models_max_common":max(r["n_matched"] for r in correlations),
           "full_six_model_common_panel_exists":False,"maximum_observed_spearman":best[0] if best else None}
    write_json(OUT/"COVERAGE_AND_SEARCH_ACCOUNTING.json",units)
    limitations=["All public relationships are retrospective diagnostics, not loss weights or a deployable validation objective.",
        "Only four official models have common saved predictions on five modern panels; six-model ranking cannot be reconstructed without forbidden new inference.",
        "R9 all5 has two separate historical local panels only; R9 Tree5 lacks eligible labelled saved predictions.",
        "R16 V1/Tree5/R17 B1 share base predictions; four observations are correlated, not independent experiments.",
        "local_report and domain/stress panels have already been inspected in prior rounds, so this is not untouched confirmation.",
        "R19 custom and vanilla rows share 2,000 paired source identities; cross-law target distributions differ.",
        "Missing/single-class/zero-variance cells remain explicit nulls; no synthetic filling or target-label reconstruction.",
        "A pre-run structural inventory accidentally inspected shapes/dtypes of R9 frozen_fold_predictions.npz (Test predictions); no values/statistics used, script blocks this path."]
    report=["# R20 저장 예측 기반 로컬 정합성 전수 진단","",f"실행 시작: {started}","",
        "재학습·추론·보정·제출 없이 저장된 라벨 있는 로컬 예측만 집계했다. 공개 점수와 관계는 사후 설명용이며 생산 정책을 바꾸지 않았다.","",
        "## 범위와 결측","",json.dumps(units,ensure_ascii=False,indent=2),"",
        "R9 Tree5의 frozen_fold_predictions는 Test 추론 파일이므로 제외했다. R9 all5 500행 fit/gate는 현재 R16/R18 패널과 다른 샘플이므로 별도 표본으로만 수록했다. 따라서 공식 6모델의 동일 패널 순위 일치를 검증할 수 없다.","",
        "## 핵심 B1/B2 패널 전체 지표","",table(keyrows,["panel","model","n","rmse","bias","roc_auc","brier","ece10"]),"",
        "## 공개 순위와의 고정 후보 사후 상관","",
        f"사전 고정한 {len(SLICE_NAMES)}개 단독 구간과 {len(COMBOS)}개 동일 가중 평균만 검사했다. 전체 {len(correlations)}회 중 4개 이상 공식 모델이 겹친 유효 비교는 {len(valid)}회다. 아래는 한정 집합에서 관측된 상위 8개이며, 선택 편향이 포함된 관측 최댓값이다.","",
        table(best[:8],["panel","candidate","n_matched","spearman_public_rmse","pearson_public_rmse","local_order_low_error_first"]),"",
        "N=4에서는 단 24개 순열만 존재한다. 상관 1.0이 나와도 여러 후보를 검사한 뒤의 결과이며 실제 비공개 평가와 1:1 정렬되었다는 근거가 아니다. 점수 +0.01075를 맞추는 구간·가중치·손실 최적화는 하지 않았다.","",
        "## 지표의 의미와 미정의 값","",
        "각 표에는 RMSE·MAE·Bias·R²와 독립 score의 Brier·ECE10·ROC-AUC·PR-AUC가 있다. PR-AUC는 average precision이다. 양성만 또는 음성만 남는 타깃 구간은 AUC가 정의되지 않아 null/single_class로 기록했다. y=0의 R²는 null/zero_target_variance다. ECE는 동일한 고정 확률 10구간이며 극단 구간의 ECE는 그 조건부 집단에서의 값이다. native12는 저장된 양성/음성 50:50 가중치를 유지한다.","",
        "## 한계","",*["- "+s for s in limitations],"",
        "전체 결과: ALL_MODELS_PANELS_SLICES_METRICS.csv, MODEL_PANEL_COVERAGE.csv, SLICE_COUNTS_AND_WEIGHTS.csv, PUBLIC_ALIGNMENT_POSTHOC_ALL_TRIALS.csv. 파일 SHA는 OUTPUT_SHA256.json에 보존한다."]
    (OUT/"R20_LOCAL_ALIGNMENT_METRICS.md").write_text("\n".join(report)+"\n")
    write_json(OUT/"INPUT_SHA256.json",PINS)
    write_json(OUT/"ACCESS_AND_SCOPE.json",{"new_training":False,"new_inference":False,"gpu_or_mps":False,
        "Test_or_query_used_by_analysis":False,"inventory_shape_only_exception":"R9 frozen_fold_predictions.npz excluded after shape/dtype identification",
        "new_policy_or_submission":False,"read_paths":READS,"limitations":limitations})
    outputhash={str(p.relative_to(OUT)):{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(OUT.rglob('*'))
                if p.is_file() and p.name not in ('OUTPUT_SHA256.json','COMPLETED.json')}
    write_json(OUT/"OUTPUT_SHA256.json",outputhash)
    write_json(OUT/"COMPLETED.json",{"status":"completed","finished_KST":datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "source_sha256":sha(Path(__file__)),"output_manifest_sha256":sha(OUT/"OUTPUT_SHA256.json"),
        "units":units,"new_training":False,"new_inference":False})
    print(json.dumps(units,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    run()
