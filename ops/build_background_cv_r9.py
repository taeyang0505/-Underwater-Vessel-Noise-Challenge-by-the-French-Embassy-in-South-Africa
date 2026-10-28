"""Build/audit backgroundsource5fold and four disjoint stage roles, no models."""
from __future__ import annotations

import argparse,json
from pathlib import Path

import numpy as np
import pandas as pd
from pipeline.background_cv_r9 import (
    ROOT,DEFAULT_OUTPUT,ROLES,VALIDATION_ROLES,PURGE_SECONDS,WINDOW_SECONDS,
    heldout_group_roles,proposed_purge_gap,final_cross_role_gaps,build,
    validate_background_cv,apply_minute_qc,sha,write,
)


def fixture():
    mapping=heldout_group_roles([f'g{i}' for i in range(51)],0)
    assert set(mapping.values())==set(VALIDATION_ROLES)
    assert pd.Series(mapping).value_counts().to_dict()=={'selection':25,'ensemble_fit':10,'ensemble_gate':8,'report':8}
    assert mapping==heldout_group_roles(list(reversed(list(mapping))),0)
    base=pd.Timestamp('2020-01-10')
    f=pd.DataFrame({'start_export_clock':[base-pd.Timedelta(seconds=PURGE_SECONDS+WINDOW_SECONDS),
                                        base-pd.Timedelta(seconds=PURGE_SECONDS),base],
                    'sensor_group':[0,0,0],'proposed_role':['train','train','report']})
    gaps=proposed_purge_gap(f)
    assert gaps[0]==PURGE_SECONDS and gaps[1]==PURGE_SECONDS-WINDOW_SECONDS and np.isinf(gaps[2])
    f['split']=f.proposed_role
    active=f.iloc[[0,2]].reset_index(drop=True)
    assert (final_cross_role_gaps(active)>=PURGE_SECONDS).all()
    # A same-time event at a distinct sensor must not trigger artificial time conflict.
    sensor=f.iloc[[2]].copy();sensor['sensor_group']=1;sensor['proposed_role']='ensemble_gate'
    assert np.isinf(proposed_purge_gap(pd.concat([f.iloc[[2]],sensor],ignore_index=True))).all()
    x=np.ones((2,15,25),dtype=np.float32);x[0,4]=np.nan
    qc=apply_minute_qc(x,np.array([1<<2,1<<7],dtype=np.uint16))
    assert np.isnan(qc[0,2]).all() and np.isnan(qc[0,4]).all() and np.isnan(qc[1,7]).all()
    assert np.isfinite(x[0,2]).all() and np.isfinite(x[1,7]).all()
    print('PASS numericmetadatafixture: deterministic4roles,24h+900sboundary,sensorseparation,sameQC')


def independent_audit(output):
    folder=ROOT/output;m=json.loads((folder/'manifest.json').read_text());gate=validate_background_cv(output)
    pool=pd.read_parquet(folder/'background_pool_manifest.parquet')
    assert len(pool)==100965 and not pool.base_window_index.duplicated().any()
    foldfacts=[];heldout_union=set()
    for fold in range(5):
        directory=folder/f'fold_{fold:02d}';parts={r:pd.read_parquet(directory/(r+'_manifest.parquet')) for r in ROLES}
        allrows=pd.concat(parts.values(),ignore_index=True)
        assert not allrows.base_window_index.duplicated().any()
        assert set(allrows.base_window_index)<=set(pool.base_window_index)
        assert allrows.qc_observed_minutes.ge(3).all()
        assert allrows.groupby('background_group').split.nunique().max()==1
        held=set(pool.loc[pool.outer_fold.eq(fold),'base_window_index'])
        assert heldout_union.isdisjoint(held);heldout_union.update(held)
        assert not(set(parts['train'].base_window_index)&held)
        assert all(set(parts[r].base_window_index)<=held for r in VALIDATION_ROLES)
        minimum=np.inf;violations=0;pair_facts=[]
        # Independent interval support audit: sorted lookup at both bracketing
        # starts for each distinct role pair, not the builder's priority pass.
        for i,left in enumerate(ROLES):
            for right in ROLES[i+1:]:
                mins=[]
                for sensor,l in parts[left].groupby('sensor_group'):
                    rr=parts[right][parts[right].sensor_group.eq(sensor)]
                    if rr.empty:continue
                    a=pd.to_datetime(l.start_export_clock).astype('int64').to_numpy()//10**9
                    b=np.sort(pd.to_datetime(rr.start_export_clock).astype('int64').to_numpy()//10**9)
                    at=np.searchsorted(b,a)
                    distance=np.full(len(a),np.inf)
                    before=at>0;after=at<len(b)
                    distance[before]=np.minimum(distance[before],a[before]-b[at[before]-1])
                    distance[after]=np.minimum(distance[after],b[at[after]]-a[after])
                    support_gap=np.maximum(distance-WINDOW_SECONDS,0)
                    violations+=int((support_gap<PURGE_SECONDS).sum());mins.append(float(support_gap.min()/3600))
                value=min(mins) if mins else None
                if value is not None:minimum=min(minimum,value)
                pair_facts.append({'left':left,'right':right,'minimum_support_gap_hours':value})
        assert violations==0 and (not np.isfinite(minimum) or minimum>=24)
        assert all(v.background_group.nunique()>=2 for v in parts.values())
        foldfacts.append({'fold':fold,'pairwise_role_gaps':pair_facts,'under24h_violations':violations,
                          'group_or_index_overlap_violations':0,
                          'roles':{k:{'windows':len(v),'groups':v.background_group.nunique(),
                                      'qc_min_observed_minutes':int(v.qc_observed_minutes.min()),
                                      'site_counts':{str(a):int(b) for a,b in v.site.value_counts().items()}}
                                   for k,v in parts.items()}})
    assert heldout_union==set(pool.base_window_index)
    gate.update({'source_pool_windows':100965,'source_groups':int(pool.background_group.nunique()),
                 'each_original_background_bucket_heldout_exactly_once_before_QC_purge':True,
                 'original_fold_assignment_and_window_identity_preserved':True,
                 'all_folds_pairwise_under24h_violations':0,'all_group_or_index_overlaps':0,
                 'folds':foldfacts,'QC':m['QC'],'input_SHA':m['input_sha256'],
                 'bank_SHA':m['background_bank']['sha256'],'no_synthetic_NPZ_or_model_fitted':True})
    return m,gate


def report(m,g,output):
    js=ROOT/'reports/R9_BACKGROUND_CV_AUDIT.json';md=ROOT/'reports/R9_BACKGROUND_CV_AUDIT.md'
    if js.exists() or md.exists():raise FileExistsError('Fresh reports required')
    write(js,g);rows=[];purge=[]
    for f in m['folds']:
        for role,v in f['role_stats'].items():rows.append(f"| {f['fold']} | {role} | {v['windows']:,} | {v['source_groups']} |")
        b=f['before_QC_and_purge'];p=f['purged_by_proposed_role']
        purge.append(f"| {f['fold']} | {b['train_windows']:,} | {b['heldout_windows']:,} | {f['QC_excluded']:,} | "+' | '.join(str(p.get(r,0)) for r in ROLES)+' |')
    text=f'''# R9 SA background5fold·4개 검증stage 역할 — 실제 구축 감사

새버전은 **기존R4허용100,965창만** 사용한다. oldquality의TRAIN91,186+validation9,779와index집합이정확히같고SAcalibration/domain/그밖의oldpurged원천을복원하지않았다. VT509확장허가는SA예약복원허가로확대하지않았다. SA원본bank·옛dataset·모델은변경하지않았다.

## 분할 정의

기존sensor-week group258개와outerbucket0..4를그대로쓴다. 각fold는heldout한bucket을제외한4개bucket이TRAIN이므로**purge전약80/20**다. 기존nested3/5TRAIN을재사용한것이아니다. bucket원창수는{m['original_outer_bucket_counts']}다.

각heldoutbucket의whole sourcegroups를모델결과와무관한SHA순서로정렬해 **selection50%/ensemble_fit20%/ensemble_gate15%/report15%**를largest-remainder수로배정했다. 창수비율이나site비율이정확히같아진다는뜻은아니다. 작은site/group은일부역할에없을수있고이를맞추려고group을쪼개거나seed를재선택하지않았다.

모든역할에같은R6minuteQC를먼저적용했다. 원row_id와실제window/minute를join해후보분만NaN으로마스킹하고원래NaN을보존한다. 실제flag원인의물리적검증을의미하지않으며첫/끝분일괄삭제가아니다. min3observed미달은제외했다. 전체허용창중flag분{m['QC']['masked_selected_minutes']:,}개/해당창{m['QC']['windows_with_masked_minutes']:,}개,QC통과{m['QC']['eligible_windows_min3_observed']:,}개다.

## 24h·900s support purge

같은sensor의savedexportclock에서 `[start,start+900s)`사이공백이24h이상이어야한다. TRAIN은전체heldout후보와비교한다. heldout내에서는report를보존하고gate는report,fit은gate+report,selection은나머지3역할에대해경계창을제외한다. 이렇게높은우선순위는고정했으며어떤모델오차도보지않았다. clock은원exportclock이며검증된UTC로재명명하지않는다.

| Fold | purge전TRAIN | purge전heldout | QC제외 | TRAINpurge | selectionpurge | fitpurge | gatepurge | reportpurge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(purge)}

| Fold | 최종역할 | 고유창 | whole sourcegroups |
|---|---|---:|---:|
{chr(10).join(rows)}

독립pair별시간감사에서 **5fold모든role pair의24h미달0, sourcegroup/index교차중복0**이다. 각역할은최소2group이있고nonempty다. 생성할고정예제수보다고유창수가적다면replacement를명시해야하며생성예제수를독립원천수로세지않는다. 이builder는샘플링/합성NPZ를만들지않았다.

## stage사용과해석한계

- selection:epoch/checkpoint선택전용. ensemble_fit:미리선언한ensemble계수적합전용. ensemble_gate:동결후수용/거절판정전용. report:모든checkpoint/ensemblepolicy동결뒤최종보고전용이다.
- reportgroup은**그fold의새TRAIN/selection/fit/gate**에서제외됐다. 다른CVfoldTRAIN에서는같은source가사용될수있다. 과거프로젝트에서전혀본적없는새publiccorpus라는뜻은아니다. 여기서는report의source값을QC목적으로만읽었고모델예측을하지않았다.
- reference509는모든fold가공유한다. **reference-independentCV가아니며**,원래SA자연선박energyGT도새로생긴것이아니다. 새보고수치는배경source개발CV로표기해야한다.
- 새manifest/readiness는**backgroundsource metadata를합성할준비**라는gate다. 아직새generator정답/고정validationNPZ/실제모델학습전체ready가아니다.

## 경로·schema·재현

`fold_XX/{{train,selection,ensemble_fit,ensemble_gate,report}}_manifest.parquet`와heldout4역할union인`validation_manifest.parquet`가있다. 제외창은`excluded_manifest.parquet`,전체원pool은`background_pool_manifest.parquet`,모든fold사전역할은`proposed_roles_allfolds.parquet`다.

행은원래`base_window_index/window_id/background_group/site/sensor_group/start_export_clock`를보존하고`group/sensor/start/end/end_export_clock`alias,`boundary_mask_bits/masked_minute_indices/raw_observed_minutes/qc_observed_minutes`,원boundary·nominalbandwarning과새`split/role/background_fold`를제공한다. 숫자창은원bank에서globalindex로읽고동일`apply_minute_qc`를적용해야한다. 원값을0으로채우지않는다.

source_hash `{m['source_hash']}`. 원bankSHA `{m['background_bank']['sha256']}`.

[새manifest](../{output}/manifest.json) · [readiness](../{output}/readiness.json) · [독립상세감사JSON](R9_BACKGROUND_CV_AUDIT.json) · [구현](../pipeline/background_cv_r9.py) · [builder](../ops/build_background_cv_r9.py)
'''
    with md.open('x') as f:f.write(text)
    return {'audit_json_sha256':sha(js),'audit_md_sha256':sha(md)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=DEFAULT_OUTPUT)
    action=p.add_mutually_exclusive_group(required=True);action.add_argument('--self-test',action='store_true');action.add_argument('--build',action='store_true');action.add_argument('--verify',action='store_true')
    a=p.parse_args()
    if a.self_test:fixture();return
    if a.build:
        fixture();build(a.output);m,g=independent_audit(a.output);h=report(m,g,a.output)
        print(json.dumps({'status':'passed','output':a.output,'source_hash':m['source_hash'],**h}))
    else:
        _,g=independent_audit(a.output);print(json.dumps(g))


if __name__=='__main__':main()
