#!/usr/bin/env python3
"""R16 V2 metadata-only plan. Never imports model code or starts training.

The candidate config is the single source of the V2 training contract. This
planner checks source/data identities, anonymous public-weight pins, and ZIP
backup metadata. It never deserializes a tensor, reads a prediction/evaluation
file, or treats a metadata check as a successful model restore.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/r16_version2.json"
REMOTE_PROJECT = "/workspace/underwater_r16_pretrained_5090_r1/project"
MODEL_RUN = "models/r16_pretrained_cuda_bf16_r1/both"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def project_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Project path escapes workspace: {relative}")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def literal_constant(path: Path, name: str):
    """Read a literal dict without importing Torch, NumPy or model code."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Literal constant {name} not found in {path.name}")


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise ValueError(explanation)


def validate_contract(pointer: dict, candidate: dict) -> None:
    require(pointer.get("plan_only") is True, "This command is PLAN ONLY")
    require(pointer.get("production_trainer") == "pipeline/training_r16_version2.py" and
            pointer.get("production_cli") == "ops/train_r16_version2.py",
            "Only the separate V2 production trainer/explicit CLI is referenced")
    require(pointer.get("automatic_training_launch") is False, "No automatic training")
    require(pointer.get("automatic_download") is False, "No automatic download")
    require(set(candidate["families"]) == {"dasheng_base", "timemoe", "convnextv2_base_fcmae"},
            "V2 must be Dasheng + TimeMoE + ConvNeXt; DINO remains V1")
    require(set(pointer["reuse_families"]) == {"dasheng_base", "timemoe"} and
            pointer["new_family"] == candidate["new_family"] == "convnextv2_base_fcmae",
            "Only Dasheng/TimeMoE are reuse candidates; ConvNeXt is the new family")
    require(candidate["folds"] == [0] and candidate["additional_kfold"] is False,
            "Only the existing fixed split is authorized; no additional K-fold")
    require(candidate["regimes"] == ["D0", "H0"] and
            set(candidate["new_tasks"]) == {"regression", "classification"}, "Four new ConvNeXt branches only")
    require(candidate["new_convnext_training_jobs"] == 4, "Exactly four new training jobs")
    require(candidate["regression_input"] == [17, 15, 25] and
            candidate["classifier_input"] == [15, 15, 25], "Preserve input contract")
    require(candidate["classifier_channel_order_from_regression"] ==
            [0, 1, 2, 3, 6, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16], "Classifier channels/order changed")
    require(candidate["regression_classifier_shared_trainable_parameters"] is False,
            "Regression and classification trainable parameters must be separate")
    require(candidate["masking_variant"] == "both" and
            candidate["feature_only_masking_does_not_change_w_q_y"] is True,
            "Feature-only both masking must preserve w, q and y")
    require(candidate["regression_pooling"] == "sum(w * sigmoid(mask_logits))" and
            candidate["regression_loss"] == "share_MSE + 0.25 * energy_weighted_cell_MSE",
            "Physical pooling/loss changed")
    require(candidate["classifier_loss"] == "BCEWithLogits(independent_presence_logit, presence)",
            "Classifier loss must remain independent of regression mask logits")
    require(candidate["score_share_weight"] == 0, "Independent classifier score excludes share")
    require((candidate["warmup_epochs"], candidate["stage2_max_epochs"],
             candidate["early_stopping_patience"]) == (3, 25, 7), "Expected 3 + maximum 25 epochs, ES7")
    require(candidate["D0_H0_weights"] == [.5, .5], "Preserve D0/H0 50:50")
    require(candidate["regression_tree_min_weight"] == 0 and
            candidate["ensemble_policy"] == "fourway_validation_selected",
            "The capped ensemble is a comparison, not a mandatory Tree floor")
    require(candidate["Test_used_for_training_or_selection"] is False,
            "Test must not be used for training or policy selection")


def source_identity(run_plan: dict) -> dict:
    rows = []
    for relative, expected in run_plan["source_sha256"].items():
        path = project_path(relative)
        actual = sha256(path) if path.is_file() else None
        rows.append({"path": relative, "expected_sha256": expected,
                     "actual_sha256": actual, "match": actual == expected})
    return {"all_match": all(row["match"] for row in rows), "files": rows}


def trainer_readiness(pointer: dict, public: dict) -> dict:
    """Audit prepared source/receipts without importing the production runner.

    Production make_plan() loads immutable TRAIN caches; this metadata planner
    deliberately does not invoke it. Existing plan/preflight SHA receipts are
    checked instead, and no training/checkpoint/prediction is evaluated.
    """
    trainer = project_path(pointer["production_trainer"])
    cli = project_path(pointer["production_cli"])
    trainer_ast = ast.parse(trainer.read_text(encoding="utf-8"))
    functions = {node.name for node in trainer_ast.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    require({"make_plan", "check_preflight", "train_job"}.issubset(functions),
            "Prepared V2 trainer must expose plan/preflight/explicit train functions")
    # Resolve only ROOT / '<relative literal>' assignments; do not execute code.
    constants = {}
    for node in trainer_ast.body:
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div)
                and isinstance(value.left, ast.Name) and value.left.id == "ROOT"
                and isinstance(value.right, ast.Constant) and isinstance(value.right.value, str)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value.right.value
    require(constants.get("RUN") == pointer["production_model_directory"] and
            constants.get("OPS") == pointer["production_ops_directory"],
            "Metadata output paths must match actual V2 trainer constants")
    ast.parse(cli.read_text(encoding="utf-8"))
    result = {
        "trainer_source": pointer["production_trainer"], "cli_source": pointer["production_cli"],
        "trainer_sha256": sha256(trainer), "cli_sha256": sha256(cli),
        "trainer_wired": True, "separate_output_paths_match_trainer_AST": True,
        "model_directory": constants["RUN"], "ops_directory": constants["OPS"],
        "trainer_imported_or_invoked_by_planner": False,
        "source_receipt_checks_passed": False,
        "source_pinned_MPS_preflight_passed": False,
        "production_run_plan_source_pins_match": False,
        "production_plan_training_flag": None,
        "actual_training_completion_not_inferred_from_plan": True,
    }
    preflight_path = project_path(pointer["both_task_preflight"])
    run_path = project_path(pointer["production_run_plan"])
    if not preflight_path.is_file() or not run_path.is_file():
        result["status"] = "prepared_source_missing_bound_preflight_or_run_receipt"
        return result
    preflight = read_json(preflight_path)
    preflight_sources = source_identity(preflight)
    rows = []
    for row in preflight.get("rows", []):
        receipt_path = project_path(row["result"])
        receipt_sha_match = receipt_path.is_file() and sha256(receipt_path) == row["result_sha256"]
        receipt = read_json(receipt_path) if receipt_sha_match else {}
        provenance = receipt.get("provenance", {})
        stages = receipt.get("stages", row.get("stages", []))
        valid = (
            receipt_sha_match and receipt.get("status") == "passed"
            and receipt.get("task") == row["task"]
            and receipt.get("device") == preflight.get("device")
            and receipt.get("frozen_base_unchanged") is True
            and receipt.get("adapter_roundtrip_exact") is True
            and provenance.get("strict_pretrained_load") is True
            and provenance.get("weight_sha256") == preflight.get("public_weight_sha256")
            and provenance.get("output_grid") == [15, 25]
            and provenance.get("downsampling_strides") == [1, 1, 1, 1]
            and {s.get("stage") for s in stages} == {1, 2}
            and all(s.get("finite_forward_backward") and s.get("frozen_gradients_absent") for s in stages)
        )
        rows.append({"task": row["task"], "receipt": row["result"],
                     "receipt_sha256": row["result_sha256"],
                     "receipt_SHA_and_required_gates_match": bool(valid)})
    tasks_match = {row["task"] for row in rows} == {"regression", "classification"}
    preflight_match = (
        preflight.get("status") == "passed" and tasks_match
        and preflight.get("task_branches_independently_instantiated") is True
        and preflight_sources["all_match"] and all(row["receipt_SHA_and_required_gates_match"] for row in rows)
        and preflight.get("public_weight_sha256") == public["convnextv2_base_fcmae"]["file_checks"][0]["expected_sha256"]
    )
    run = read_json(run_path)
    run_sources = source_identity(run)
    run_match = (
        run.get("status") == "fixed_before_training" and run_sources["all_match"]
        and run.get("config_sha256") == sha256(project_path(pointer["candidate_config"]))
        and run.get("preflight_sha256") == sha256(preflight_path)
        and run.get("family") == "convnextv2_base_fcmae"
        and run.get("folds") == [0] and len(run.get("jobs", [])) == 4
        and run.get("runtime", {}).get("device") == preflight.get("device")
        and run.get("runtime", {}).get("python") == preflight.get("python")
        and run.get("runtime", {}).get("torch") == preflight.get("torch")
    )
    result.update({
        "status": "prepared_runner_and_source_pinned_preflight_verified" if preflight_match and run_match
                  else "prepared_runner_bound_receipts_changed_or_failed",
        "preflight_receipt": pointer["both_task_preflight"], "preflight_sha256": sha256(preflight_path),
        "source_pinned_MPS_preflight_passed": bool(preflight_match),
        "preflight_task_receipt_checks": rows, "preflight_source_identity": preflight_sources,
        "production_run_plan": pointer["production_run_plan"], "production_run_plan_sha256": sha256(run_path),
        "production_run_plan_source_pins_match": bool(run_match),
        "source_receipt_checks_passed": bool(preflight_match and run_match),
        "production_source_identity": run_sources, "runtime_from_verified_receipt": run.get("runtime"),
        "production_plan_training_flag": run.get("new_convnext_trained"),
        "planned_epoch_speed_not_production_measured": True,
    })
    return result


def public_readiness(pointer: dict, verify_bytes: bool) -> dict:
    result = {}
    for family, location in pointer["public_weight_sources"].items():
        spec = literal_constant(project_path(location["source"]), location["constant"])
        if location["key"] is not None:
            spec = spec[location["key"]]
        cache = project_path(location["cache"])
        weight = cache / "model.safetensors"
        files = {"model.safetensors": spec["weight_sha256"], **spec.get("files", {})}
        checks = []
        for name, expected in files.items():
            path = cache / name
            exists = path.is_file() and not path.is_symlink()
            actual = sha256(path) if exists and verify_bytes else None
            checks.append({"file": name, "exists": exists, "expected_sha256": expected,
                           "actual_sha256": actual,
                           "hash_match": actual == expected if verify_bytes else None})
        size = weight.stat().st_size if weight.is_file() and not weight.is_symlink() else None
        valid_size = size == spec["weight_bytes"]
        checked = verify_bytes and valid_size and all(row["hash_match"] for row in checks)
        result[family] = {
            "model_id": spec["model_id"], "revision": spec["revision"],
            "cache": location["cache"], "expected_weight_bytes": spec["weight_bytes"],
            "actual_weight_bytes": size, "size_match": valid_size,
            "public_weight_sha_verified_in_this_plan": bool(checked),
            "state": "public_file_bytes_verified" if checked else "metadata_and_existence_only",
            "license": spec.get("weights_license", spec.get("license")),
            "license_url": spec.get("weights_license_url", spec.get("license_url")),
            "file_checks": checks, "strict_tensor_load_performed": False,
            "adapter_load_performed": False, "training_performed_by_planner": False,
        }
        if family == "convnextv2_base_fcmae":
            result[family]["source_checkpoint_parameters_approx"] = 87_700_000
            result[family]["license_note"] = (
                "CC-BY-NC-4.0 weights: attribution/noncommercial terms require review "
                "against the intended use and competition conditions; no clearance is implied."
            )
    return result


def reusable_jobs(pointer: dict, run_plan: dict, sources: dict, data_match: bool,
                  split_match: bool, public: dict) -> list[dict]:
    receipt_path = project_path(pointer["version1_backup_receipt"])
    receipt = read_json(receipt_path) if receipt_path.is_file() else {"jobs": []}
    by_job = {item["job"]: item for item in receipt.get("jobs", [])}
    result = []
    for family in pointer["reuse_families"]:
        for regime in ("D0", "H0"):
            for task in ("regression", "classification"):
                job = f"{family}/{regime}/{task}"
                prefix = f"{MODEL_RUN}/{job}"
                row = {"job": job, "variant": "both", "fold": 0,
                       "new_training_planned": False,
                       "remote_branch_path": f"{REMOTE_PROJECT}/{prefix}",
                       "same_data_manifest": data_match, "same_fit_split": split_match,
                       "same_source_pins": sources["all_match"],
                       "checkpoint_restore_verified": False,
                       "status": "awaiting_verified_backup_and_exact_restore"}
                item = by_job.get(job)
                if item:
                    archive = project_path(item["local_zip"])
                    archive_match = archive.is_file() and sha256(archive) == item["sha256"]
                    row.update({"local_backup_zip": item["local_zip"],
                                "archive_sha256": item["sha256"],
                                "archive_sha_verified_in_this_plan": archive_match})
                    if archive_match:
                        # Only metadata is opened: no predictions, labels, metrics or tensors.
                        with zipfile.ZipFile(archive) as z:
                            manifest = json.loads(z.read("__metadata__/manifest.json"))
                            plan_name = f"{prefix}/PLAN.json"
                            plan_bytes = z.read(plan_name)
                            branch = json.loads(plan_bytes)
                            plan_match = hashlib.sha256(plan_bytes).hexdigest() == manifest["files"][plan_name]["sha256"]
                            p = branch.get("provenance", {})
                            base = branch.get("run_plan", {})
                            exact = (
                                plan_match and manifest["job"] == job and
                                (branch.get("family"), branch.get("regime"), branch.get("task"),
                                 branch.get("variant"), branch.get("fold")) == (family, regime, task, "both", 0)
                                and base.get("data_manifest_sha256") == run_plan["data_manifest_sha256"]
                                and base.get("local_fit_split_sha256") == run_plan["local_fit_split_sha256"]
                                and base.get("source_sha256") == run_plan["source_sha256"]
                                and branch.get("augmentation") == run_plan["masking_variants"]["both"]
                                and base.get("numerical_policy") == run_plan["numerical_policy"]
                                and p.get("model_id") == public[family]["model_id"]
                                and p.get("revision") == public[family]["revision"]
                                and p.get("weight_sha256") == public[family]["file_checks"][0]["expected_sha256"]
                                and branch.get("permanent_base_fingerprint") is not None
                                and branch.get("local_fit_split_sha256") == run_plan["local_fit_split_sha256"]
                            )
                            checkpoint = manifest["files"].get(f"{prefix}/best.pt", {})
                        row.update({"branch_metadata_exact": exact,
                                    "adapter_checkpoint_sha256_from_verified_manifest": checkpoint.get("sha256"),
                                    "adapter_checkpoint_bytes_from_verified_manifest": checkpoint.get("bytes"),
                                    "base_fingerprint_from_branch_receipt": branch.get("permanent_base_fingerprint"),
                                    "metadata_candidate": bool(exact and data_match and split_match and sources["all_match"]),
                                    "status": "metadata_candidate_pending_tensor_restore" if exact and data_match and split_match and sources["all_match"]
                                              else "identity_mismatch_no_reuse"})
                row["required_before_reuse"] = [
                    "Restore the pinned best.pt into the identical V1 branch factory/numerical policy",
                    "Recheck public base fingerprint, adapter byte SHA, task/state keys and source SHA",
                    "Verify fixed data, feature order, both masking and D0/H0 branch identity",
                    "Do not reuse V1 ensemble weights or select using Test/report performance",
                ]
                result.append(row)
    return result


def build_plan(config: Path = DEFAULT_CONFIG, verify_public_bytes: bool = False) -> dict:
    pointer = read_json(config)
    candidate_path = project_path(pointer["candidate_config"])
    candidate = read_json(candidate_path)
    validate_contract(pointer, candidate)
    old_plan_path = project_path(pointer["version1_run_plan"])
    old_plan = read_json(old_plan_path)
    data_path = project_path(pointer["data_manifest"])
    split_path = project_path(pointer["local_fit_split"])
    data_match = sha256(data_path) == old_plan["data_manifest_sha256"]
    split_match = sha256(split_path) == old_plan["local_fit_split_sha256"]
    sources = source_identity(old_plan)
    public = public_readiness(pointer, verify_public_bytes)
    trainer = trainer_readiness(pointer, public)
    reused = reusable_jobs(pointer, old_plan, sources, data_match, split_match, public)
    new = [{"family": "convnextv2_base_fcmae", "regime": regime, "task": task,
            "fold": 0, "variant": "both", "status": "planned_not_started",
            "output_path": f"{pointer['production_model_directory']}/convnextv2_base_fcmae/{regime}/{task}",
            "warmup_epochs": 3, "stage2_max_epochs": 25, "early_stopping_patience": 7}
           for regime in ("D0", "H0") for task in ("regression", "classification")]
    return {
        "format": "r16_version2_plan_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "plan_contract_valid_prepared_trainer_not_launched_by_planner", "plan_only": True,
        "version1": {"families": pointer["version1_families"], "run": "r16_pretrained_cuda_bf16_r1",
                     "sources_or_outputs_modified": False},
        "version2": {"families": candidate["families"], "factory": candidate["family_factory"],
                     "new_family": candidate["new_family"], "additional_kfold": False},
        "new_jobs": new, "new_job_count": len(new), "reuse_candidates": reused,
        "reuse_candidate_count": len(reused),
        "locally_verified_metadata_candidate_count": sum(bool(row.get("metadata_candidate")) for row in reused),
        "restored_reuse_jobs_in_this_plan": 0,
        "public_weights": public, "candidate_contract": candidate, "prepared_trainer": trainer,
        "masking_at_inference": "OFF; original observation O and physical pooling w remain ON",
        "ensemble": {"policy": "fourway_validation_selected", "Tree_floor_default": 0,
                     "candidates": candidate["ensemble_candidates"],
                     "weights": "not fitted or borrowed by this planner; independent validation decides",
                     "D0_H0_weights": [.5, .5], "score_share_weight": 0},
        "identity": {"data_manifest_match": data_match, "local_fit_split_match": split_match,
                     "version1_source_pins": sources,
                     "pointer_sha256": sha256(config), "candidate_config_sha256": sha256(candidate_path),
                     "version1_run_plan_sha256": sha256(old_plan_path),
                     "planner_sha256": sha256(Path(__file__)),
                     "new_factory_sha256": sha256(project_path("pipeline/pretrained_convnext_v2_r16.py"))},
        "execution": {"trainer_wired": trainer["trainer_wired"], "training_launched": False,
                      "MPS_or_CUDA_preflight_performed_by_planner": False,
                      "source_pinned_MPS_preflight_receipts_verified": trainer["source_pinned_MPS_preflight_passed"],
                      "explicit_convnext_training_CLI_prepared": trainer["source_receipt_checks_passed"],
                      "ready_for_full_training": False,
                      "full_pipeline_readiness_reason": "Explicit device/runtime/cache checks and compatible reused branch restore remain; policy finalizer is separate",
                      "blocking_requirements": [
                          "Invoke prepared V2 trainer only explicitly with the pinned preflight Python/Torch/device; this planner never launches it",
                          "Prepared runner revalidates immutable TRAIN/CV cache, data source and 1194/1206 role separation at execution",
                          "Restore and verify exact Dasheng/TimeMoE checkpoint identities before reuse",
                          "Measure a real TRAIN epoch; synthetic-input preflight speed is not production runtime or transfer-performance proof",
                          "Use a separate frozen-policy V2 finalizer for validation comparison and subsequent Test inference",
                          "Review CC-BY-NC-4.0 ConvNeXt weight terms for the intended use",
                      ]},
        "privacy_and_selection": {"Test_read": False, "predictions_read": False,
                                  "evaluation_metrics_read": False, "tensors_deserialized": False,
                                  "source_data_loaded": False, "network_accessed": False},
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".PLAN.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--outdir", type=Path, default=ROOT / "ops/r16_version2")
    parser.add_argument("--verify-public-bytes", action="store_true",
                        help="Hash existing pinned public files; still no tensor/model loads or download")
    args = parser.parse_args()
    plan = build_plan(args.config.resolve(), args.verify_public_bytes)
    output = args.outdir.resolve() / "PLAN.json"
    require(output.is_relative_to(ROOT), "Output must remain inside the project")
    atomic_json(output, plan)
    print("R16 V2 PLAN ONLY — no model load, training, Test or evaluation access")
    print("V1: Dasheng-Base + DINOv2-small + TimeMoE; preserved")
    print("V2: Dasheng-Base + ConvNeXtV2-Base FCMAE + TimeMoE")
    print(f"New jobs: {plan['new_job_count']} (ConvNeXt D0/H0 × regression/classification; no additional K-fold)")
    print(f"Reuse: {plan['reuse_candidate_count']} candidates; "
          f"{plan['locally_verified_metadata_candidate_count']} verified backup metadata matches; 0 tensor restores performed")
    for family, row in plan["public_weights"].items():
        print(f"Public weights {family}: {row['state']}; size_match={row['size_match']}; license={row['license']}")
    prepared = plan["prepared_trainer"]
    print(f"Separate V2 trainer prepared={prepared['trainer_wired']}; "
          f"source-pinned MPS preflight receipts verified={prepared['source_pinned_MPS_preflight_passed']}; "
          f"production run-plan pins match={prepared['production_run_plan_source_pins_match']}")
    print("No training launched. Explicit pinned-runtime execution and reused checkpoint restore remain separate.")
    print(output)


if __name__ == "__main__":
    main()
