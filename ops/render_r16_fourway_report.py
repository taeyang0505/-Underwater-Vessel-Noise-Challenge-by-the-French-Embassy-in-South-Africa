#!/usr/bin/env python3
"""Korean, stdlib-only rendering of a saved R16 four-way comparison result.

The renderer performs no learning, prediction, policy selection or input-data
reads. Exact result state is preserved as JSON; long fitted reference arrays
are represented by counts and hashes in human-facing Markdown and HTML.

API: render(result: dict, outdir: Path) -> dict[str, str]
CLI: python ops/render_r16_fourway_report.py RESULT.json --outdir DIR
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any


TITLE = "R16 V1 보존 · V2 네 가지 앙상블 비교"
STEM = "R16_FOURWAY_ENSEMBLE_COMPARISON"
FAMILIES = ("lightgbm5", "catboost5", "dasheng_base", "dinov2_small", "timemoe")
LABELS = {
    "category": "후보군", "policy": "정책", "name": "이름", "source": "출처/경로", "field": "항목", "value": "값",
    "dataset": "평가셋", "panel": "패널", "regime": "D0/H0", "family": "모델 계열", "model": "모델",
    "n": "행 수", "n_backgrounds": "고유 배경 수", "rmse": "RMSE", "RMSE": "RMSE", "bias": "Bias",
    "auc": "AUC", "AUC": "AUC", "share_auc": "share AUC", "score_auc": "독립 score AUC", "classifier_auc": "독립 score AUC",
    "prediction_mean": "share 평균", "prediction_rms": "share RMS", "classifier_mean": "score 평균",
    "id": "후보 ID", "task": "학습 목적", "status": "상태", "selected": "채택", "eligible": "Gate 통과",
    "accepted": "채택 여부", "passed": "검증 통과", "reason": "판정 근거", "reasons": "판정 근거",
    "rejection_reason": "탈락 근거", "rejection_reasons": "탈락 근거", "reject_reason": "탈락 근거",
    "epsilon": "ε", "D0_H0_weights": "D0:H0", "transform": "score 변환", "method": "변환 방식",
    "lightgbm5": "LightGBM5", "catboost5": "CatBoost5", "dasheng_base": "Dasheng-Base",
    "dinov2_small": "DINOv2-small", "timemoe": "TimeMoE", "weight": "가중치", "member": "계열",
    "fit_rmse": "fit RMSE", "gate_rmse": "gate RMSE", "fit_auc": "fit AUC", "gate_auc": "gate AUC",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _brief(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _brief(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if key in ("sorted_reference", "ecdf_reference", "fit_reference") or (len(value)>64 and not any(isinstance(x,dict) for x in value)):
            raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
            lengths = [len(x) for x in value if isinstance(x, (list, tuple))]
            return {"display": "전체 값은 결과 JSON에 보존", "items": len(value),
                    "n_fit_per_member": lengths[0] if lengths and len(set(lengths))==1 else lengths,
                    "canonical_json_sha256": hashlib.sha256(raw.encode()).hexdigest()}
        return [_brief(v) for v in value]
    return value


def _value(v: Any) -> str:
    if v is None: return "미제공"
    if isinstance(v, bool): return "예" if v else "아니요"
    if isinstance(v, float): return f"{v:.6f}" if abs(v)<1e5 else f"{v:.6g}"
    if isinstance(v, (dict, list, tuple)): return json.dumps(v, ensure_ascii=False, default=str)
    return str(v)


def _rows(v: Any, key="name") -> list[dict]:
    if isinstance(v,list): return [dict(x) if isinstance(x,dict) else {key:x} for x in v]
    if isinstance(v,dict): return [{key:k, **(x if isinstance(x,dict) else {"value":x})} for k,x in v.items()]
    return [] if v is None else [{"value":v}]


def _flat(v: Any, path="") -> list[dict]:
    if isinstance(v,dict):
        rows=[]
        for k,x in v.items():
            p=f"{path}.{k}" if path else str(k)
            rows.extend(_flat(x,p) if isinstance(x,dict) and x else [{"field":p,"value":x}])
        return rows
    return [{"field":path or "value","value":v}]


def _metric_rows(v: Any, path="") -> list[dict]:
    """Find metric dictionaries without assuming category/gate nesting."""
    if isinstance(v,list):
        rows=[]
        for i,x in enumerate(v): rows.extend(_metric_rows(x,f"{path}[{i}]"))
        return rows
    if not isinstance(v,dict): return []
    metric_keys={"rmse","RMSE","bias","auc","AUC","score_auc","classifier_auc","share_auc"}
    if set(v)&metric_keys:
        row={"source":path}
        row.update({k:x for k,x in v.items() if not isinstance(x,(dict,list,tuple))})
        rows=[row]
    else: rows=[]
    for k,x in v.items():
        if isinstance(x,(dict,list,tuple)): rows.extend(_metric_rows(x,f"{path}.{k}" if path else str(k)))
    return rows


def _policies(result: dict) -> dict[str,dict]:
    """Accept direct policies and category-level selected policy containers."""
    out={}
    raw=result.get("policies",{})
    for name,value in (raw.items() if isinstance(raw,dict) else enumerate(raw or [])):
        if isinstance(value,dict): out[str(name)]=value.get("policy",value)
    for container in ("categories","category_results","fourway"):
        raw=result.get(container,{})
        if not isinstance(raw,dict): continue
        for name,value in raw.items():
            if not isinstance(value,dict): continue
            p=value.get("policy",value.get("selected_policy"))
            if isinstance(p,dict): out.setdefault(str(name),p)
            for sub,p in value.get("policies",{}).items():
                if isinstance(p,dict): out.setdefault(f"{name}/{sub}",p)
    return out


def _weight_rows(policies: dict) -> list[dict]:
    rows=[]
    for name,policy in policies.items():
        members=policy.get("member_order",FAMILIES)
        reg=policy.get("regression",{})
        weights=reg.get("member_weights",{})
        if not weights and isinstance(policy.get("experts"),dict):
            weights={r:v.get("regression_weights",[]) for r,v in policy["experts"].items()}
        for regime,w in weights.items():
            row={"policy":name,"task":"회귀","regime":regime,
                 "epsilon":reg.get("epsilon",policy.get("epsilon")),
                 "D0_H0_weights":reg.get("D0_H0_weights",policy.get("D0_H0_weights"))}
            if isinstance(w,dict): row.update(w)
            elif isinstance(w,(list,tuple)): row.update(dict(zip(members,w)))
            else: row["weights"]=w
            rows.append(row)
        cls=policy.get("classification",{})
        w=cls.get("weights")
        method=cls.get("transform",{}).get("method",cls.get("method")) if isinstance(cls.get("transform",{}),dict) else cls.get("method")
        if isinstance(w,(list,tuple)) and len(w)==2*len(members):
            for i,regime in enumerate(("D0","H0")):
                rows.append({"policy":name,"task":"독립 분류(최종 score 기여량)","regime":regime,
                             "transform":method,**dict(zip(members,w[i*len(members):(i+1)*len(members)]))})
        elif w is not None:
            rows.append({"policy":name,"task":"독립 분류","transform":method,"weights":w})
    return rows


def render(result: dict, outdir: Path) -> dict[str,str]:
    if not isinstance(result,dict): raise TypeError("result must be a dict")
    outdir=Path(outdir).resolve();outdir.mkdir(parents=True,exist_ok=True)
    exact=_json(result)+"\n";result_path=outdir/f"{STEM}_RESULT.json"
    result_path.write_text(exact,encoding="utf-8")
    digest=hashlib.sha256(exact.encode()).hexdigest();view=_brief(result)
    md=[f"# {TITLE}\n"]
    body=[f"<h1>{TITLE}</h1>"]
    stamp=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")
    note=f"생성 시각(KST): {stamp} · 결과 JSON SHA-256: {digest}"
    md.append(note+"\n");body.append(f'<p class="muted">{html.escape(note)}</p>')

    def prose(s,notice=False):
        md.append(s+"\n");body.append(f'<p class="{"notice" if notice else ""}">{html.escape(s)}</p>')
    def section(s):
        md.append(f"## {s}\n");body.append(f"<h2>{html.escape(s)}</h2>")
    def table(rows,preferred=()):
        if not rows: prose("해당 수치는 결과 JSON에 제공되지 않았습니다.");return
        cols=list(dict.fromkeys(k for row in rows for k in row));cols=[k for k in preferred if k in cols]+[k for k in cols if k not in preferred]
        md.append("| "+" | ".join(LABELS.get(k,k) for k in cols)+" |\n| "+" | ".join("---" for _ in cols)+" |\n"+
                  "\n".join("| "+" | ".join(_value(row.get(k)).replace("|","\\|").replace("\n","<br>") for k in cols)+" |" for row in rows)+"\n")
        h='<div class="scroll"><table><thead><tr>'+"".join(f"<th>{html.escape(LABELS.get(k,k))}</th>" for k in cols)+"</tr></thead><tbody>"
        h+="".join("<tr>"+"".join(f"<td>{html.escape(_value(row.get(k)))}</td>" for k in cols)+"</tr>" for row in rows)
        body.append(h+"</tbody></table></div>")
    def detail(title,value):
        raw=_json(value);fence="`"*max(3,max((len(s) for s in re.findall(r'`+',raw)),default=0)+1)
        md.append(f"<details><summary>{title}</summary>\n\n{fence}json\n{raw}\n{fence}\n\n</details>\n")
        body.append(f"<details><summary>{html.escape(title)}</summary><pre>{html.escape(raw)}</pre></details>")

    prose("V1은 기존에 등록·동결된 추론/제출 경로이며 그대로 보존합니다. V2는 별도 버전의 네 가지 후보군 비교와 유한 후처리 선택입니다. 이 렌더러는 전달된 저장 결과만 읽으며, 학습·추론·정책 선택·Test 데이터 읽기·공식 업로드를 실행하지 않습니다.")
    prose("독립성 주의: native-long/HF/SF 패널은 local_report의 배경 그룹과 겹칩니다. 따라서 report는 정책 고정 후 확인용이며, 편향 없는 독립 최종 시험이 아닙니다. 기존 개발에서 사용했던 패널의 재사용 한계도 남아 있습니다. Gate에 사용한 수치를 새 정책의 독립 검증 성능으로 표시하지 않습니다.",True)

    section("1. V1 / V2 범위와 사전 고정·선택·평가 순서")
    scope={k:view[k] for k in ("scope","pipeline_versions","protocol","datasets","splits","selection_protocol","decision","chronology") if k in view}
    table(_flat(scope) if scope else [],("field","value"))
    prose("checkpoint 선택 1,194행과 가중치 fit 1,206행은 역할이 다릅니다. 고유 배경 수·행 수·D0/H0·단일 분할/기존 Tree 5-Fold를 구분해야 합니다. 입력 JSON에 없는 값은 채우지 않습니다.")

    section("2. 개별 5개 계열의 성적: D0 / H0 / 고정 50:50")
    family=_rows(view.get("baseline_family_metrics"))
    datasets=list(dict.fromkeys(row.get("dataset",row.get("panel","미지정")) for row in family))
    pref=("dataset","panel","regime","family","model","n_backgrounds","n","rmse","bias","share_auc","score_auc","classifier_auc","auc","prediction_mean","prediction_rms")
    if not family: table([])
    for dataset in datasets:
        md.append(f"### {dataset}\n");body.append(f"<h3>{html.escape(str(dataset))}</h3>")
        table([r for r in family if r.get("dataset",r.get("panel","미지정"))==dataset],pref)
    prose("대상 축은 checkpoint / fit / local_gate / unseen-site / native-long+HF / native-long+SF입니다. 실제 제공된 패널만 표시합니다. 50:50 행은 고정 D0/H0 평균이며 그 자체로 별도 최적화된 모델은 아닙니다.")

    section("3. 네 가지 앙상블 후보군의 가중치와 점수 규칙")
    prose("제약형·자유형·신경망 전용·Tree 전용을 별도 후보군으로 비교합니다. 정책명과 실제 가중치는 아래 저장 결과가 기준입니다. 회귀와 독립 분류를 분리하며, 분류 표의 D0/H0 가중치는 최종 score에 기여하는 질량입니다.")
    policies=_policies(view)
    table(_weight_rows(policies),("policy","task","regime","epsilon","D0_H0_weights","transform",*FAMILIES,"weights"))
    for name,policy in policies.items(): detail(f"정책 {name}: 전체 규칙·변환 요약",policy)

    section("4. Gate 비교 및 유한 후처리 후보 채택·탈락 근거")
    gatekeys=("gate_metrics","category_metrics","fourway_metrics","comparison_metrics","category_results","categories","fourway")
    metrics=[]
    for key in gatekeys:
        if key in view: metrics+=_metric_rows(view[key],key)
    table(metrics,("source","category","policy","panel","dataset","regime","n","rmse","bias","share_auc","score_auc","classifier_auc","auc"))
    candidates=_rows(view.get("candidates"),"candidate")
    # Keep candidate outcome tables compact; nested per-panel metrics follow in
    # an expandable section instead of creating enormous cells.
    simple=[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in candidates]
    table(simple,("category","id","name","candidate","task","eligible","selected","status","epsilon","reason","reasons","rejection_reason","rejection_reasons"))
    if candidates: detail("유한 후보 전체 결과·판정 근거",candidates)
    for key in ("selection","selection_results","selected_categories","selected_category","fallback","fallback_reason"):
        if key in view: detail(key,view[key])

    section("5. 동결 후 local_report 확인: 원본 / 네 후보군 / 최종 선택")
    final=view.get("final_report")
    table(_rows(final) if isinstance(final,list) else _metric_rows(final,"final_report"),
          ("panel","dataset","category","policy","name","n_backgrounds","n","rmse","bias","share_auc","score_auc","classifier_auc","auc","prediction_mean","prediction_rms"))
    prose("이 표는 정책 동결 후 확인 결과입니다. 이 결과를 보고 가중치나 후처리를 다시 선택했다면 독립 확인으로 볼 수 없습니다. source/chronology 기록에서 실제 실행 순서를 함께 확인해야 합니다.",True)

    section("6. CSV 후보별 기술 통계·무결성·경로")
    submissions=view.get("submissions",{})
    items=submissions.items() if isinstance(submissions,dict) else enumerate(submissions or [])
    found=False
    for name,entry in items:
        found=True;md.append(f"### {name}\n");body.append(f"<h3>{html.escape(str(name))}</h3>")
        table(_flat(entry),("field","value"))
    if not found: table([])
    prose("평균·RMS·분위수는 출력 기술 통계입니다. 숨은 Test 정답 평균이나 공식 점수의 추정값이 아닙니다. CSV 생성·무결성 검사와 공식 제출은 구분합니다.")

    section("7. 출처 SHA·실행 이력·수치 재구성 한계")
    provenance={k:view[k] for k in ("provenance","numerical_reconstruction","chronology_caveat","chronology_caveats","limitations","caveats") if k in view}
    table(_flat(provenance) if provenance else [],("field","value"))
    prose("장시간 스트레스 한 축의 결과만으로 모든 채널·미관측 사이트 전이를 판단하지 않습니다. 로컬 비교에서 이긴 후보가 공식 리더보드에서도 개선된다는 보장은 없습니다.")

    section("재현 자료 및 누락 없는 결과")
    md.append(f"[정확한 전체 결과 JSON]({result_path.name})\n");body.append(f'<p><a href="{html.escape(result_path.name)}">정확한 전체 결과 JSON</a></p>')
    detail("전체 결과 개요 (큰 참조 배열은 SHA·개수 요약)",view)
    page='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+TITLE+'''</title><style>
body{font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo','Noto Sans KR',sans-serif;background:#f1f5f8;color:#203148;margin:0;line-height:1.65}main{max-width:1560px;margin:30px auto;background:white;padding:32px;border-radius:14px}h1{font-size:30px}h2{font-size:24px;border-top:1px solid #dce4ec;padding-top:24px;margin-top:38px}h3{font-size:19px}.notice{border-left:5px solid #b7822a;background:#fff6e6;padding:14px 18px}.muted{font-size:13px;color:#596b80;overflow-wrap:anywhere}.scroll{overflow-x:auto;margin:15px 0 24px}table{border-collapse:collapse;font-size:13px;width:100%}th,td{border:1px solid #dbe3ec;padding:9px 12px;text-align:left;vertical-align:top;min-width:65px;overflow-wrap:anywhere}th{background:#ebf2f8}tbody tr:nth-child(even){background:#f8fafc}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f9;padding:16px;font-size:12px}summary{cursor:pointer;color:#285c88}details{margin:16px 0}a{color:#285c88}@media(max-width:800px){main{margin:8px;padding:18px}h1{font-size:24px}}
</style></head><body><main>'''+"\n".join(body)+"</main></body></html>\n"
    mdpath=outdir/f"{STEM}.md";htmlpath=outdir/f"{STEM}.html"
    mdpath.write_text("\n".join(md),encoding="utf-8");htmlpath.write_text(page,encoding="utf-8")
    return {"markdown":str(mdpath),"html":str(htmlpath),"result_json":str(result_path),"result_json_sha256":digest}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json",type=Path);parser.add_argument("--outdir",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(render(json.loads(args.result_json.read_text(encoding="utf-8")),args.outdir),ensure_ascii=False,indent=2))


if __name__=="__main__": main()
