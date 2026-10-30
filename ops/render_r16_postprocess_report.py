#!/usr/bin/env python3
"""Render an existing R16 postprocessing result dict as Korean Markdown and HTML.

This module uses only the standard library and does not read model/data files,
train models, run inference, or change an input result. The root workflow owns
all experiment execution, policy decisions and CSV validation.

API: ``render(result: dict, outdir: Path) -> dict[str, str]``.
CLI: ``python ops/render_r16_postprocess_report.py RESULT.json --outdir DIR``.
The input schema is deliberately permissive: known sections receive comparison
tables; unknown/nested fields remain visible in expandable JSON sections.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any


TITLE = "R16 동결 예측·후처리·앙상블 비교 보고서"
STEM = "R16_POSTPROCESSING_COMPARISON"
LABELS = {
    "dataset": "평가셋", "panel": "패널", "regime": "D0/H0", "family": "모델 계열",
    "model": "모델", "name": "이름", "policy": "정책", "n": "행 수", "n_backgrounds": "고유 배경 수",
    "rmse": "RMSE", "RMSE": "RMSE", "bias": "Bias", "auc": "AUC", "AUC": "AUC",
    "share_auc": "share AUC", "score_auc": "score AUC", "id": "후보 ID",
    "candidate": "후보", "status": "상태", "selected": "채택", "eligible": "통과 여부",
    "accepted": "채택 여부", "passed": "통과 여부", "reason": "판정 근거", "reasons": "판정 근거",
    "rejection_reason": "탈락 근거", "rejection_reasons": "탈락 근거", "reject_reason": "탈락 근거",
    "epsilon": "ε", "eps": "ε", "weights": "가중치", "regression_weights": "회귀 가중치",
    "classifier_weights": "분류 가중치", "score_transform": "score 변환",
    "D0_H0_weights": "D0:H0 가중치", "fit_rmse": "fit RMSE", "gate_rmse": "gate RMSE",
    "fit_auc": "fit AUC", "gate_auc": "gate AUC", "path": "경로", "csv": "CSV",
    "sha256": "SHA-256", "mean": "평균", "rms": "RMS", "std": "표준편차", "median": "중앙값",
    "min": "최솟값", "max": "최댓값", "p05": "p05", "p10": "p10", "p25": "p25",
    "p50": "p50", "p75": "p75", "p90": "p90", "p95": "p95", "p99": "p99",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _human_view(value: Any, key: str = "") -> Any:
    """Summarize large fitted reference arrays without dropping exact JSON state."""
    if isinstance(value, dict):
        return {k: _human_view(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if key in ("sorted_reference", "ecdf_reference", "fit_reference"):
            canonical = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
            lengths = [len(row) for row in value if isinstance(row, (list, tuple))]
            return {"display": "참조값 전체는 결과 JSON에 보존", "members": len(value),
                    "n_fit_per_member": lengths[0] if lengths and len(set(lengths)) == 1 else lengths,
                    "canonical_json_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}
        if len(value) > 64 and not any(isinstance(v, dict) for v in value):
            canonical = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
            return {"display": "긴 배열 전체는 결과 JSON에 보존", "items": len(value),
                    "canonical_json_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}
        return [_human_view(v) for v in value]
    return value


def _value(value: Any) -> str:
    if value is None:
        return "미제공"
    if isinstance(value, bool):
        return "예" if value else "아니요"
    if isinstance(value, float):
        return f"{value:.6f}" if abs(value) < 1e5 else f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)
    return str(value)


def _rows(value: Any, key_name: str = "name") -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(row) if isinstance(row, dict) else {key_name: row} for row in value]
    if isinstance(value, dict):
        return [{key_name: key, **(row if isinstance(row, dict) else {"value": row})}
                for key, row in value.items()]
    return [] if value is None else [{"value": value}]


def _columns(rows: list[dict[str, Any]], preferred: tuple[str, ...] = ()) -> list[str]:
    existing = list(dict.fromkeys(key for row in rows for key in row))
    return [key for key in preferred if key in existing] + [key for key in existing if key not in preferred]


def _md_escape(value: Any) -> str:
    return _value(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _table(rows: list[dict[str, Any]], preferred: tuple[str, ...] = ()) -> tuple[str, str]:
    if not rows:
        return "수치가 제공되지 않았습니다.\n", '<p class="muted">수치가 제공되지 않았습니다.</p>'
    cols = _columns(rows, preferred)
    if not cols:
        return "표에 표시할 필드가 없습니다.\n", '<p class="muted">표에 표시할 필드가 없습니다.</p>'
    md = ["| " + " | ".join(LABELS.get(c, c) for c in cols) + " |",
          "| " + " | ".join("---" for _ in cols) + " |"]
    md += ["| " + " | ".join(_md_escape(row.get(c)) for c in cols) + " |" for row in rows]
    ht = '<div class="table-scroll"><table><thead><tr>'
    ht += "".join(f"<th>{html.escape(LABELS.get(c, c))}</th>" for c in cols) + "</tr></thead><tbody>"
    for row in rows:
        ht += "<tr>" + "".join(f"<td>{html.escape(_value(row.get(c)))}</td>" for c in cols) + "</tr>"
    ht += "</tbody></table></div>"
    return "\n".join(md) + "\n", ht


def _flat(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Keep nested labels visible, without assuming any measurement semantics."""
    if isinstance(value, dict):
        result = []
        for key, entry in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(entry, dict) and entry:
                result.extend(_flat(entry, name))
            else:
                result.append({"field": name, "value": entry})
        return result
    return [{"field": prefix or "value", "value": value}]


def render(result: dict, outdir: Path) -> dict[str, str]:
    """Write MD/HTML and a lossless JSON snapshot; return their absolute paths.

    Known input sections (all optional): baseline_family_metrics, policies,
    candidates, final_report, submissions, provenance, chronology_caveat(s).
    ``baseline_family_metrics`` and ``final_report`` accept lists of row dicts;
    nested additional columns are retained rather than silently omitted.
    """
    if not isinstance(result, dict):
        raise TypeError("result must be a dict")
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    snapshot = _json(result) + "\n"
    result_path = outdir / f"{STEM}_RESULT.json"
    result_path.write_text(snapshot, encoding="utf-8")
    digest = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()
    # Preserve exact transform arrays in RESULT.json. Human-facing policy tables
    # show ECDF reference counts and hashes, not tens of thousands of floats.
    result = _human_view(result)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md = [f"# {TITLE}\n", f"생성 시각(UTC): `{now}`  \n결과 JSON SHA-256: `{digest}`\n",
          "이 보고서는 전달받은 저장 결과만 정리합니다. 렌더러는 학습·추론·Test 읽기·정책 변경을 수행하지 않습니다. "
          "아래 비교는 입력 JSON에 기재된 분할과 수치에 한정되며, 로컬 개선을 공식 리더보드 개선으로 해석하지 않습니다. "
          "긴 ECDF 참조 배열은 개수·해시로 요약하며 전체 값은 결과 JSON에 보존합니다.\n"]
    body = [f"<h1>{TITLE}</h1>", f'<p class="muted">생성 시각(UTC): {html.escape(now)}<br>결과 JSON SHA-256: <code>{digest}</code></p>',
            '<p class="notice">저장된 결과 JSON의 렌더링 보고서입니다. 학습·추론·Test 읽기·정책 변경은 실행하지 않습니다. '
            '로컬 비교 수치가 공식 리더보드 성능을 보장하지 않습니다. '
            '긴 ECDF 참조 배열은 개수·해시로 요약하며 전체 값은 결과 JSON에 보존합니다.</p>']
    used = set()

    def section(title: str, description: str = "") -> None:
        md.append(f"## {title}\n")
        body.append(f"<h2>{html.escape(title)}</h2>")
        if description:
            md.append(description + "\n")
            body.append(f"<p>{html.escape(description)}</p>")

    def add_table(rows: list[dict[str, Any]], preferred: tuple[str, ...] = ()) -> None:
        m, h = _table(rows, preferred)
        md.append(m); body.append(h)

    def details(label: str, value: Any) -> None:
        raw = _json(value)
        # A long fenced section preserves exact fields and precision in Markdown.
        fence = "`" * max(3, max((len(s) for s in re.findall(r'`+', raw)), default=0) + 1)
        md.append(f"<details><summary>{label}</summary>\n\n{fence}json\n{raw}\n{fence}\n\n</details>\n")
        body.append(f"<details><summary>{html.escape(label)}</summary><pre>{html.escape(raw)}</pre></details>")

    section("1. 비교 범위와 선택·평가 데이터",
            "체크포인트 선택용 1,194행, 가중치 fit용 1,206행, 독립 gate/report는 역할을 구분합니다. "
            "행 수와 고유 배경 수는 서로 다른 단위이며, 입력 결과에 없는 값은 추정해 채우지 않습니다.")
    context = {k: result[k] for k in ("scope", "datasets", "splits", "contract", "selection_protocol", "metric_definition") if k in result}
    used.update(context)
    if context:
        add_table(_flat(context), ("field", "value"))
    else:
        add_table([])

    section("2. 원본 모델·계열의 저장 예측 성적", "평가셋 열을 통해 checkpoint/fit/gate를 구분하여 읽습니다. 같은 행이거나 같은 원천이라는 보장은 이 표만으로 하지 않습니다.")
    baseline = result.get("baseline_family_metrics")
    used.add("baseline_family_metrics")
    add_table(_rows(baseline), ("dataset", "panel", "regime", "family", "model", "n_backgrounds", "n", "rmse", "bias", "auc", "share_auc", "score_auc"))

    section("3. 원본·Tree·선택 정책", "가중치, ε, D0:H0 결합 및 score 변환은 회귀와 독립 분류를 구분해 기록합니다. 실제 채택 여부는 입력 결과의 정책·판정 기록을 따릅니다.")
    policies = result.get("policies", {})
    used.add("policies")
    policy_items = policies.items() if isinstance(policies, dict) else enumerate(policies or [])
    for name, policy in policy_items:
        md.append(f"### {name}\n"); body.append(f"<h3>{html.escape(str(name))}</h3>")
        add_table(_flat(policy), ("field", "value"))
        details(f"정책 {name}: 전체 필드", policy)
    if not policies:
        add_table([])

    section("4. 후보 비교·채택 및 탈락 근거", "후보 상태와 탈락 사유는 결과에 기록된 내용을 그대로 표시합니다. 빈 사유를 임의로 성공 또는 실패로 바꾸지 않습니다.")
    candidates = result.get("candidates")
    used.add("candidates")
    add_table(_rows(candidates, "candidate"), ("id", "name", "candidate", "regime", "status", "selected", "eligible", "accepted", "passed", "fit_rmse", "gate_rmse", "fit_auc", "gate_auc", "epsilon", "eps", "reason", "reasons", "rejection_reason", "rejection_reasons", "reject_reason"))
    if candidates is not None:
        details("후보별 원시 결과·정밀 수치", candidates)

    section("5. 독립 최종 비교: 원본 / 선택 정책 / Tree", "local_report와 다른 독립 패널의 RMSE·Bias·AUC를 입력 결과에 따라 나란히 표시합니다. 이 단계의 평가 결과로 정책을 다시 적합했는지는 아래 이력·출처 기록을 확인해야 합니다.")
    final = result.get("final_report")
    used.add("final_report")
    add_table(_rows(final), ("dataset", "panel", "policy", "name", "regime", "n_backgrounds", "n", "rmse", "bias", "auc", "share_auc", "score_auc"))
    if final is not None:
        details("최종 평가 전체 결과", final)

    section("6. 제출 후보 CSV의 기술 통계·무결성", "예측 평균·RMS·분위수와 ID 순서·행 수·유한값·범위·SHA 등 검증 결과를 보존합니다. 통계값만으로 숨은 정답 분포 또는 공식 점수를 추정하지 않습니다.")
    submissions = result.get("submissions", {})
    used.add("submissions")
    submission_items = submissions.items() if isinstance(submissions, dict) else enumerate(submissions or [])
    for name, info in submission_items:
        md.append(f"### {name}\n"); body.append(f"<h3>{html.escape(str(name))}</h3>")
        add_table(_flat(info), ("field", "value"))
        details(f"CSV {name}: 전체 경로·통계·검사", info)
    if not submissions:
        add_table([])

    section("7. 출처·실행 순서 및 독립성 한계")
    provenance = result.get("provenance")
    used.add("provenance")
    caveats = []
    for key in ("chronology_caveat", "chronology_caveats", "limitations", "caveats", "chronology", "selection_caveat", "independence_caveat"):
        if key in result:
            caveats.append({"field": key, "value": result[key]}); used.add(key)
    if isinstance(provenance, dict):
        for key, value in provenance.items():
            if any(t in str(key).lower() for t in ("caveat", "chronolog", "limit", "independen", "leak")):
                caveats.append({"field": f"provenance.{key}", "value": value})
    if caveats:
        add_table(caveats, ("field", "value"))
    else:
        note = "실행 순서·독립성 한계가 결과 JSON에 별도로 제공되지 않았습니다. 사후 생성 시각만으로 사전 고정·완전한 독립 검증을 주장할 수 없습니다."
        md.append(note + "\n"); body.append(f'<p class="notice">{html.escape(note)}</p>')
    if provenance is not None:
        add_table(_flat(provenance), ("field", "value"))
        details("출처·해시·실행 이력 전체", provenance)

    extra = {key: value for key, value in result.items() if key not in used}
    if extra:
        section("8. 추가 결과 필드")
        details("스키마에 추가된 결과 전체", extra)
    section("재현 자료")
    md.append(f"- [렌더링 입력 JSON]({result_path.name})\n- 결과 JSON SHA-256: `{digest}`\n")
    body.append(f'<ul><li><a href="{html.escape(result_path.name)}">렌더링 입력 JSON</a></li><li>SHA-256: <code>{digest}</code></li></ul>')
    page = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>""" + TITLE + """</title><style>
body{font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo','Noto Sans KR',sans-serif;color:#203148;background:#f3f6f9;margin:0;line-height:1.65}main{max-width:1500px;margin:30px auto;padding:32px;background:white;border-radius:12px}h1{font-size:30px}h2{border-top:1px solid #d9e1e9;padding-top:24px;margin-top:38px;font-size:24px}h3{font-size:19px}.muted{color:#596c80;font-size:13px}.notice{background:#fff7e8;border-left:4px solid #b58739;padding:15px}.table-scroll{overflow-x:auto;margin:14px 0 22px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #dce4ed;padding:9px 12px;text-align:left;vertical-align:top;min-width:70px;overflow-wrap:anywhere}th{background:#edf3f9;position:sticky;top:0}tr:nth-child(even){background:#f8fafc}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f9;padding:16px;border-radius:6px;font-size:12px}code{overflow-wrap:anywhere}details{margin:14px 0}summary{cursor:pointer;color:#285c88}a{color:#285c88}@media(max-width:800px){main{padding:18px;margin:8px}h1{font-size:24px}}
</style></head><body><main>""" + "\n".join(body) + "</main></body></html>\n"
    md_path = outdir / f"{STEM}.md"
    html_path = outdir / f"{STEM}.html"
    md_path.write_text("\n".join(md), encoding="utf-8")
    html_path.write_text(page, encoding="utf-8")
    return {"markdown": str(md_path), "html": str(html_path), "result_json": str(result_path), "result_json_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    print(json.dumps(render(result, args.outdir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
