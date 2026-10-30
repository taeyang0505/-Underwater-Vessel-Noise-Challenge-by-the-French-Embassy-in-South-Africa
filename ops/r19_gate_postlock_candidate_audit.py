#!/usr/bin/env python3
"""Diagnostic-only exhaustive gate check of all 27 preregistered candidates.

SELECTION_LOCK and earlier metric files are immutable. No training/inference,
Test/query read, report-based reselection, or new calibration candidates.
"""
from pathlib import Path
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops'))
import r19_multislice_gate as g

def main():
    base=g.OUT;out=base/'postlock_all_candidates';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'PROTOCOL.json').exists(),'Immutable output already exists'
    original={p.name:g.sha(p) for p in base.iterdir() if p.is_file()}
    protocol=json.loads((base/'PROTOCOL.json').read_text())
    lock=json.loads((base/'SELECTION_LOCK.json').read_text())
    assert lock['PROTOCOL_sha256']==g.sha(base/'PROTOCOL.json')
    assert lock['selected']=='old_B1'
    for p,h in json.loads((base/'INPUT_SHA256.json').read_text()).items():assert g.sha(ROOT/p)==h,p
    g.write(out/'PROTOCOL.json',{'created_KST':g.now(),'mode':'diagnostic_only_after_lock','prior_candidate_protocol_sha256':g.sha(base/'PROTOCOL.json'),
        'original_selection_lock_sha256':g.sha(base/'SELECTION_LOCK.json'),'candidate_count':len(protocol['candidate_specs']),
        'rule':'Exactly the original 22 axes and 1e-6 tolerance; all originally registered candidates, including fit-rejected corrections, are tabulated; no adoption or reselection.',
        'report_previously_read':True,'new_training':False,'new_model_inference':False,'Test_read':False,'code_sha256':g.sha(Path(__file__))})
    panels={name:g.load_panel(name) for name in g.GATE}
    specs=protocol['candidate_specs'];metrics=g.eval_specs(panels,specs)
    checks=g.gates(metrics,list(specs))
    metrics.to_csv(out/'gate_all_27_candidate_metrics_DIAGNOSTIC_ONLY.csv',index=False)
    checks.to_csv(out/'gate_all_27_axis_checks_DIAGNOSTIC_ONLY.csv',index=False)
    checks[~checks.passed].to_csv(out/'gate_all_27_failure_reasons_DIAGNOSTIC_ONLY.csv',index=False)
    summary=checks.groupby('candidate').agg(axes=('passed','size'),failed_axes=('passed',lambda x:int((~x).sum())),max_delta=('delta','max')).reset_index()
    fit=pd.read_csv(base/'fit_candidate_ranking.csv').set_index('candidate')['fit_RMSE']
    summary['fit_RMSE']=summary.candidate.map(fit)
    summary['fit_delta_RMSE']=summary.fit_RMSE-float(fit['old_B1'])
    summary['passes_all']=summary.failed_axes.eq(0)
    summary['original_gate_eligible']=summary.candidate.isin(json.loads((base/'FIT_CALIBRATION_FREEZE.json').read_text())['retained'])
    summary.to_csv(out/'all_27_diagnostic_summary.csv',index=False)
    passing=summary[summary.passes_all].candidate.tolist()
    nonidentity=[n for n in passing if n not in ('old_B1','old_B1_offset_0','old_B1_softgate_1')]
    changed=[]
    for p,h in original.items():
        if g.sha(base/p)!=h:changed.append(p)
    assert not changed,changed
    md=f'''# R19 사전 등록27개 후보 전수 gate 검산 — 동결 후 진단 전용

완료: {g.now()}. 원래 SELECTION_LOCK **{lock['selected']}**는 그대로다. 모델 학습/추론, Test/query 읽기, 새 파라미터 탐색은0회다.

## 범위 보충

원래 보고서의 ‘모든 변경후보 실패’는 **fit에서 파라미터가 먼저 고정되어 gate에 진입한17개 후보**에 대한 결론이었다. 전수 요청을 명확히 충족하기 위해, 처음 PROTOCOL에 이미 등록했으나 fit에서 탈락한10개 보정도 같은22축으로 추가 검산했다. 사전 후보/임계값을 늘리지 않았고 report를 이미 본 뒤이므로 이 단계는 **진단만** 한다. 기존 선택이나 report 성적표를 바꾸지 않는다.

전체27개 중 통과={passing}. identity가 아닌 통과 후보={nonidentity}.

{g.table(summary)}

fit에서 제외된 미세 offset/softgate도 무추가·약한 양성의 바닥값을 낮추는 이득과 장시간 양성의 과소예측 손해를 맞바꾸는지 확인했다. 어느 축이 깨지는지는 `gate_all_27_failure_reasons_DIAGNOSTIC_ONLY.csv`에 모두 있다. all/zero/weak/active≥12는 원래 가중치를 유지했다. 허용 오차는 절대1e−6다.

## 보존

이전 파일{len(original)}개 SHA와 SELECTION_LOCK을 전후 대조했고 변경0개다. 원본 보고서·기존지표·후보·제출 CSV는 수정하지 않았다. 이 부록은 원 보고서의 범위를 명확히 하는 설명이며 결과에 맞춘 재선택이 아니다.
'''
    (out/'R19_ALL_REGISTERED_CANDIDATES_DIAGNOSTIC.md').write_text(md)
    g.write(out/'INPUT_SHA256.json',g.PINS)
    hashes={str(p.relative_to(ROOT)):{'sha256':g.sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file() and p.name not in ('OUTPUT_SHA256.json','COMPLETED.json')}
    g.write(out/'OUTPUT_SHA256.json',hashes)
    done={'status':'completed','created_KST':g.now(),'candidate_count':27,'gate_axes_per_candidate':22,'passing':passing,'nonidentity_passing':nonidentity,
        'selection_changed':False,'old_files_SHA_unchanged':len(original),'source_sha_verified':len(g.PINS),'new_training':False,'new_inference':False,'Test_read':False,
        'report_sha256':g.sha(out/'R19_ALL_REGISTERED_CANDIDATES_DIAGNOSTIC.md')}
    g.write(out/'COMPLETED.json',done);print(json.dumps(done,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
