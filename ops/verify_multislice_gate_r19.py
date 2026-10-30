#!/usr/bin/env python3
"""Exact numeric parity of reusable R19 API against frozen 27 x 22 checks."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ops'))
import r19_multislice_gate as original
from pipeline.multislice_gate_r19 import DEFAULT_POLICY,evaluate_candidate,policy_dict,policy_sha256


def main():
    out=original.OUT
    before={p.name:original.sha(p) for p in out.iterdir() if p.is_file()}
    protocol=json.loads((out/'PROTOCOL.json').read_text())
    panels={name:original.load_panel(name) for name in original.GATE}
    targets={name:d for name,(d,m) in panels.items()}
    bases={name:original.raw(m)['old_B1'][0] for name,(d,m) in panels.items()}
    predictions={candidate:{name:original.apply(spec,original.raw(m))[0] for name,(d,m) in panels.items()} for candidate,spec in protocol['candidate_specs'].items()}
    result=evaluate_candidate(targets,bases,predictions)
    old=pd.read_csv(out/'postlock_all_candidates/gate_all_27_axis_checks_DIAGNOSTIC_ONLY.csv')
    new=pd.DataFrame([c for v in result['results'].values() for c in v['checks']])
    keys=['candidate','panel','slice','metric']
    merged=old.merge(new,on=keys,suffixes=('_old','_API'),validate='one_to_one')
    assert len(merged)==27*22
    for key in ['passed','available','n']:assert np.array_equal(merged[key+'_old'],merged[key+'_API']),key
    maxdiff={key:float(np.max(np.abs(merged[key+'_old']-merged[key+'_API']))) for key in ('baseline','candidate_value','delta')}
    assert max(maxdiff.values())<1e-14,maxdiff
    assert {k:v['passed'] for k,v in result['results'].items()}=={k:bool(x.passed.all()) for k,x in old.groupby('candidate')}
    assert all(original.sha(out/k)==h for k,h in before.items()),'Prior artifact changed'
    policy={'status':'frozen_adoption_gate','policy':policy_dict(),'policy_sha256':policy_sha256(),
            'expected_axes':22,'API':'pipeline.multislice_gate_r19.evaluate_candidate(panels,bases,candidates)',
            'data_roles':{'fit':'Choose calibration parameters before gate evaluation','gate':list(DEFAULT_POLICY.panels),'local_report':'Final verification only after SELECTION_LOCK; never changes adoption','Test':'Never calibration, ranking, or gate input'},
            'prior_selection_lock_sha256':original.sha(out/'SELECTION_LOCK.json'),'prior_selection_changed':False}
    original.write(out/'ADOPTION_POLICY.json',policy)
    original.write(out/'MODULE_VERIFICATION.json',{'status':'passed','created_KST':original.now(),'comparison':'frozen 27-candidate x 22-axis prior diagnostic','axis_rows_compared':len(merged),
        'numeric_max_absolute_difference':maxdiff,'all_passed_available_n_flags_identical':True,'all_candidate_decisions_identical':True,
        'prior_files_SHA_unchanged':len(before),'selection_lock_unchanged':True,'new_training':False,'new_model_inference':False,'Test_read':False,
        'source_sha256':{n:original.sha(ROOT/n) for n in ('pipeline/multislice_gate_r19.py','ops/test_multislice_gate_r19.py','ops/verify_multislice_gate_r19.py')},
        'minimal_tests':'5 passed: identity, aggregate improvement with weak harm, missing panel, NaN prediction, empty weak slice'})
    (out/'INTEGRATION_README.md').write_text('''# R19 다중 구간 게이트 재사용 안내

`pipeline/multislice_gate_r19.py`의 `DEFAULT_POLICY`는 frozen dataclass이며 tuple로22축을 고정한다. `ADOPTION_POLICY.json`에 같은 계약과 hash를 저장했다. 이 모듈은 학습·추론·배합 적합·보정을 실행하지 않는다.

## 호출 계약

```python
from pipeline.multislice_gate_r19 import evaluate_candidate
decision = evaluate_candidate(panels, bases, candidates)
accepted = decision['results']['candidate_name']['passed']
```

- `panels[panel_name]`: `share` 정답1D, 선택적 `weights`1D, native12 두 패널은 `active_minutes`1D 필수. ID 정렬은 호출자가 모델 예측 receipt와 대조해야 한다.
- `bases[panel_name]`: 기준 old_B1의 정렬된 share 예측1D.
- `candidates[candidate_name][panel_name]`: 동일 입력/정렬의 후보 share 예측1D.
- 4필수 패널은 local_gate/domain_holdout/native12_VS_hf_high_negative/native12_VS_st_francis_dense_negative이다.
- 각 패널 전체RMSE, 무추가RMSE/양의bias, 약한양성(0<y≤.1)RMSE/양의bias, native12 두 패널의 활성≥12분 양성RMSE를 비교한다. 허용오차는1e−6이며 모든22축이 통과해야 한다.
- 필수 패널·예측·정답이 없거나 비정상길이/NaN/Inf/빈 필수 slice/0가중치 slice이면 자동 기각한다. 양의 bias는 max(가중 mean(pred−y),0)이다. 전체개선으로 하위군손실을 상쇄하지 않는다.

## 향후 연결 순서

1. fit만으로 사전 등록 후보의 보정 파라미터를 고정한다.
2. gate4패널에서 위 API를 실행한다. 후보 중 통과안만 사전 정의한 fit 순위로 채택하며 없으면 old_B1 유지다.
3. 선택을 SELECTION_LOCK에 저장하고 나서 local_report를 최종 확인한다. report를 보고 재선택하지 않는다.
4. Test는 동결된 정책의 추론에만 사용한다. Test/공식점수로 이 게이트나 임계값을 조정하지 않는다.

**기존 `pipeline/r18_production_finalization.py`와 기존 PLAN/SELECTION_LOCK/모델/후보 CSV는 동결 상태로 보존했으며 수정하지 않았다.** 미래 실험의 새 finalizer에서만 이 모듈을 import한다. 지금 R18의 과거 선택 결과를 소급해서 바꾸는 도구가 아니다. historically exposed gate/report이므로 새로운 blind test로 부르면 안 된다.

검증: `ops/test_multislice_gate_r19.py` 5개 필수 테스트 통과, `ops/verify_multislice_gate_r19.py` 기존27후보×22축594행과 수치/판정 동일 확인. 기존frozen출력은전후SHA를검증했다.
''')
    paths=[ROOT/n for n in ('pipeline/multislice_gate_r19.py','ops/test_multislice_gate_r19.py','ops/verify_multislice_gate_r19.py')]
    paths += [out/n for n in ('ADOPTION_POLICY.json','MODULE_VERIFICATION.json','INTEGRATION_README.md')]
    original.write(out/'MODULE_ARTIFACTS_SHA256.json',{str(p.relative_to(ROOT)):{'sha256':original.sha(p),'bytes':p.stat().st_size} for p in paths})
    print(json.dumps({'status':'passed','rows':len(merged),'max_abs_difference':maxdiff,'policy_sha256':policy_sha256(),'prior_files_unchanged':len(before)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
