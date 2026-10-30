#!/usr/bin/env python3
"""Frozen-prediction R19 multi-slice adoption gate. No training or inference.

Only labelled local cached outputs are read. Raw Test, query inputs/outputs,
and submission files are denied. Small calibration parameters are selected
on fresh_fit only; gate adopts or falls back; report never changes policy.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r19_multislice_law_transfer_r1/gate'
FINAL=ROOT/'reports/r18_production_r1/finalization'
FAMILIES=('lightgbm','dinov2_small','dasheng_base')
GATE=('local_gate','domain_holdout','native12_VS_hf_high_negative','native12_VS_st_francis_dense_negative')
FIT=('fresh_fit_D0','fresh_fit_H0')
TOL=1e-6
PINS={}
READ_LOG=[]

def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])); low=p.name.lower()
        if low in ('test.csv','sample_submission.csv','query.npz') or 'query' in low or 'submission' in low:
            raise PermissionError('R19 gate forbids Test/query/submission reads: '+str(p))
sys.addaudithook(guard)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def pin(p):
    p=Path(p);PINS[str(p.relative_to(ROOT))]=sha(p);return p

def write(p,x):
    text=json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.item() if isinstance(a,np.generic) else str(a))+'\n'
    tmp=Path(str(p)+'.tmp');tmp.write_text(text);tmp.replace(p)

def read(p):return json.loads(pin(p).read_text())
def now():return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec='seconds')

def load_panel(name):
    source=FINAL/'inputs'/(name+'.npz')
    keys=('share','presence','sample_id','site','base_window_index','background_group','operator','active_minutes','weights')
    with np.load(pin(source),allow_pickle=False) as z:d={k:z[k] for k in keys if k in z}
    READ_LOG.append({'stage':'report' if name=='local_report' else 'fit' if name in FIT else 'gate','panel':name,'read_KST':now(),'raw_db_loaded':False})
    n=len(d['share']);assert len(set(d['sample_id']))==n
    d['weights']=np.asarray(d.get('weights',np.ones(n)),float);d['weights']/=d['weights'].sum()
    assert np.array_equal(d['share']>0,d['presence']>0)
    members={}
    for gen in ('old','new'):
        for family in FAMILIES:
            p=FINAL/'predictions'/gen/family/(name+'.npz')
            receipt=read(p.with_suffix('.json'))
            assert receipt['input_sha256']==sha(source)
            assert receipt['prediction_sha256']==sha(pin(p))
            with np.load(p,allow_pickle=False) as z:
                assert np.array_equal(z['sample_id'].astype(str),d['sample_id'].astype(str))
                a={k:z[k].astype(float) for k in ('share_D0','share_H0','prob_D0','prob_H0')}
            for v in a.values():assert v.shape==(n,) and np.isfinite(v).all() and np.all((v>=0)&(v<=1))
            a['share_D0H0_50_50']=.5*(a['share_D0']+a['share_H0'])
            a['prob_D0H0_50_50']=.5*(a['prob_D0']+a['prob_H0'])
            members[gen,family]=a
    return d,members

def raw(members,regime='D0H0_50_50'):
    a={}
    for gen in ('old','new'):
        for family in FAMILIES:
            m=members[gen,family];a[gen+'_'+family]=(m['share_'+regime],m['prob_'+regime])
        for name,w in [('B1',(.6,.25,.15)),('B2',(.5,.3,.2))]:
            a[gen+'_'+name]=tuple(sum(wi*members[gen,f][kind+'_'+regime] for wi,f in zip(w,FAMILIES)) for kind in ('share','prob'))
    return a

def candidate_specs():
    specs={}
    for gen in ('old','new'):
        for family in (*FAMILIES,'B1','B2'):specs[gen+'_'+family]={'type':'raw','base':gen+'_'+family}
    for mix in (.25,.5,.75):specs[f'generation_mix_new{mix:g}']={'type':'generation_mix','new_fraction':mix}
    for base in ('old_B1','old_B2'):
        for delta in (0.,.0025,.005,.01):
            specs[f'{base}_offset_{delta:g}']={'type':'offset','base':base,'delta':delta}
        for rho in (.9,.95,1.):
            specs[f'{base}_softgate_{rho:g}']={'type':'softgate','base':base,'rho':rho}
    return specs

def apply(spec,a):
    kind=spec['type']
    if kind=='raw':s,c=a[spec['base']]
    elif kind=='generation_mix':
        f=spec['new_fraction'];s=(1-f)*a['old_B1'][0]+f*a['new_B2'][0];c=(1-f)*a['old_B1'][1]+f*a['new_B2'][1]
    elif kind=='offset':
        s,c=a[spec['base']];s=np.clip(s-spec['delta'],0,1)
    elif kind=='softgate':
        s,c=a[spec['base']];s=s*(spec['rho']+(1-spec['rho'])*c)
    else:raise ValueError(kind)
    return np.clip(s,0,1),np.clip(c,0,1)

def metrics(d,s,c,meta):
    y=d['share'].astype(float);w=d['weights'];active=d['active_minutes']
    masks={'all':np.ones(len(y),bool),'y=0':y==0,'0<y<=0.10':(y>0)&(y<=.1),'0<y<=0.05':(y>0)&(y<=.05),
           '0.05<y<=0.15':(y>.05)&(y<=.15),'y>0.15':y>.15,'positive':y>0,'positive_active>=12':(y>0)&(active>=12)}
    rows=[]
    for k,ix in masks.items():
        n=int(ix.sum());r={**meta,'slice':k,'n':n,'weight_mass':float(w[ix].sum())}
        if n:
            wi=w[ix]/w[ix].sum();e=s[ix]-y[ix];z=y[ix]>0;bias=float(wi@e)
            r.update(rmse=float(np.sqrt(wi@(e*e))),bias=bias,positive_bias=max(bias,0),prediction_mean=float(wi@s[ix]),target_mean=float(wi@y[ix]),
                     score_mean=float(wi@c[ix]),score_ge090=float(wi@(c[ix]>=.9)),auc=float(roc_auc_score(z,c[ix],sample_weight=wi)) if z.any() and (~z).any() else None)
        rows.append(r)
    return rows

def eval_specs(panels,specs):
    result=[]
    for panel,(d,m) in panels.items():
        a=raw(m)
        for name,spec in specs.items():result.extend(metrics(d,*apply(spec,a),{'panel':panel,'candidate':name,'regime':'D0H0_50_50'}))
    return pd.DataFrame(result)

def fit_rank(frame):
    d=frame[frame['slice'].eq('all')]
    score=d.groupby('candidate').rmse.apply(lambda x:float(np.sqrt(np.mean(x.to_numpy()**2))))
    return score.sort_values(kind='stable')

def gates(frame,candidates):
    rows=[]
    for panel in GATE:
        for sl in ('all','y=0','0<y<=0.10',*(['positive_active>=12'] if panel.startswith('native12') else [])):
            for metric in (['rmse','positive_bias'] if sl in ('y=0','0<y<=0.10') else ['rmse']):
                base=frame[frame.panel.eq(panel)&frame.candidate.eq('old_B1')&frame['slice'].eq(sl)].iloc[0]
                for name in candidates:
                    x=frame[frame.panel.eq(panel)&frame.candidate.eq(name)&frame['slice'].eq(sl)].iloc[0]
                    available=int(base.n)>0 and int(x.n)>0
                    delta=float(x[metric]-base[metric]) if available else None
                    rows.append({'candidate':name,'panel':panel,'slice':sl,'metric':metric,'n':int(x.n),
                                 'baseline':float(base[metric]) if available else None,'candidate_value':float(x[metric]) if available else None,
                                 'delta':delta,'tolerance':TOL,'passed':bool(available and delta<=TOL),'available':available})
    return pd.DataFrame(rows)

def table(frame):return frame.to_markdown(index=False,floatfmt='.6f')

def resume_render():
    """Recover an interrupted reporting phase without recomputing any selection."""
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    decision=json.loads((OUT/'SELECTION_LOCK.json').read_text())
    assert decision['PROTOCOL_sha256']==sha(OUT/'PROTOCOL.json')
    assert decision['fit_freeze_sha256']==sha(OUT/'FIT_CALIBRATION_FREEZE.json')
    freeze=json.loads((OUT/'FIT_CALIBRATION_FREEZE.json').read_text())
    before={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
    counts=pd.read_csv(OUT/'gate_adoption_summary.csv')
    gm=pd.read_csv(OUT/'gate_candidate_metrics.csv')
    rm=pd.read_csv(OUT/'report_candidate_metrics_AFTER_LOCK.csv')
    ranking=pd.read_csv(OUT/'fit_candidate_ranking.csv')
    chosen=decision['selected'];wanted=['old_B1','old_B2','new_B2',chosen]
    cols=['panel','candidate','slice','n','rmse','bias','positive_bias','auc']
    slices=['all','y=0','0<y<=0.10','positive_active>=12']
    gate=gm[gm.candidate.isin(wanted)&gm['slice'].isin(slices)][cols]
    report=rm[rm.candidate.isin(wanted)&rm['slice'].isin(slices)][cols]
    scope=pd.read_csv(OUT/'recorded_training_scope.csv')
    old=scope[scope.generation.eq('old')&scope.family.eq('lightgbm')]
    for panel in (*FIT,*GATE,'local_report'):
        ip=FINAL/'inputs'/(panel+'.npz');ih=sha(pin(ip))
        for gen in ('old','new'):
            for family in FAMILIES:
                pp=FINAL/'predictions'/gen/family/(panel+'.npz'); receipt=read(pp.with_suffix('.json'))
                assert receipt['input_sha256']==ih
                assert receipt['prediction_sha256']==sha(pin(pp))
    pin(ROOT/'reports/r18_public_gap_audit_r1/predictions/model_training_provenance.csv')
    md=f'''# R19 저장 예측 기반 다중 구간 게이트

완료: {now()}. 추가 모델 학습·모델 추론·Test/query/제출 파일 읽기 **0회**. 새 제출 CSV를 만들거나 기존 후보를 수정하지 않았다.

## 결론

**{chosen} 유지.** 모든 변경 후보가 사전 고정한22개 구간별 기준을 동시에 통과하지 못했다. old_B1은 R17 B1(LGB60%+DINO25%+Dasheng15%, 각 D0:H0=50:50)이다. old_B2는50/30/20이다. 공식 과거 점수는 후보 적합/선택에 사용하지 않았다.

## 사전 고정 기준과 실행 순서

- 4개 gate 패널(local_gate, domain_holdout, native12/HF, native12/SF) 각각 전체 RMSE, 무추가 RMSE와 양의 bias, 약한 양성(0<y≤.10) RMSE와 양의 bias를 old_B1과 비교했다. native12 두 패널은 활성≥12분 양성 RMSE도 비교했다. 총22축 전부 비악화여야 채택한다.
- 양의 bias는 max(가중 mean(pred−y),0), 허용 오차는 절대1e−6이다. native12 원래 양/음성50:50 패널 가중치를 보존했다. 개선 축으로 악화 축을 상쇄하지 않는다.
- old_B1/B2에서 offset δ∈{{0,.0025,.005,.01}}와 softgate ρ∈{{.9,.95,1}}를 각각 사전 등록했다. offset=clip(share−δ,0,1); softgate=share*(ρ+(1−ρ)*독립prob). score는 수정하지 않았다. 약한 양성/장시간의 과소예측 위험은 동일 게이트에서 검사한다.
- 각 base×보정family에서 fresh_fit_D0/H0 동등가중 MSE가 가장 낮은 파라미터1개를 먼저 고정하고, raw10개·세대혼합3개·fit선택보정4개를 gate에 비교했다. 모든27개 fit 대조값과 gate17개 결과를 보존했다.
- 선택 정책을 SELECTION_LOCK에 동결한 **뒤** local_report를 읽었다. report에서 다시 선택하지 않았다. 이 패널들은 과거 개발에도 노출됐으므로 새로운 blind test가 아니다.

## fit에서 고정한 보정 파라미터

{table(pd.DataFrame([{k:v for k,v in p.items() if k!='spec'} for p in freeze['parameters']]))}

## 전체 후보 채택 결과

{table(counts)}

개별 실패축은 `gate_failure_reasons.csv`, 각 축의 baseline/후보/delta/통과여부는 `gate_axis_checks.csv`다. 모든 fit 후보는 `fit_all_candidates_metrics.csv`에 있다.

## 주요 후보 gate 성적

{table(gate)}

## 동결 후 local_report 최종 확인

{table(report)}

회귀 보정은 독립 분류 점수를 바꾸지 않아 동일 base의 AUC는 같다. 신경망의 분류 확률을 회귀 softgate에 사용해도 회귀/분류 모델 파라미터나 학습 그래디언트를 공유한 것은 아니다. 결과가 좋아 보이는 report 행을 골라 정책을 다시 바꾸지 않았다.

## 실제 old LGB 학습 범위

{table(old[['regime','task','fold','train_rows','selection_rows','input_feature_count']])}

각 fold의 실제 TRAIN 수가 달라 ‘61,446×5 독립 배경’이라고 부를 수 없다. old/new 신경망은 단일 source split이고, D0/H0와 합성 반복은 독립 원천 수가 아니다. 전체 모델 범위는 `recorded_training_scope.csv`에 있다.

## 검증과 제한

모든 입력/예측 파일의 receipt SHA 및 샘플 ID를 최초 평가 때 확인했다. 보고서 복구 시에도21개가 아닌7개패널×6개모델 예측과 원본 입력의 receipt SHA를 재검증했다. 원본 db와 Test는 로드하지 않았다. 17후보의 선택 계산·지표·report 값은 수정하지 않았다.

보고 단계에서 pandas `DataFrame.axes` 속성과 컬럼명 `axes`가 충돌하여 본문 렌더만 중단됐다. 선택동결/모든지표는 이미 저장되어 있었다. 이 보고서는 그 파일만 재사용해 복구했으며 변경 전후 SHA를 `REPORT_RENDER_RECOVERY.json`에 기록했다. 새 보정 후보나 재선택은 없다.
'''
    (OUT/'R19_MULTI_SLICE_GATE.md').write_text(md)
    write(OUT/'INPUT_SHA256.json',PINS)
    after={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name in before}
    assert before==after
    write(OUT/'REPORT_RENDER_RECOVERY.json',{'created_KST':now(),'reason':'pandas DataFrame.axes attribute/column collision in report formatting only','registered_code_sha256':protocol['code_sha256'],
           'current_code_sha256':sha(Path(__file__)),'frozen_artifacts_preserved':before,'selection_recomputed':False,'report_metrics_recomputed':False})
    outputs={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ('OUTPUT_SHA256.json','COMPLETED.json')}
    write(OUT/'OUTPUT_SHA256.json',outputs)
    done={'status':'completed','created_KST':now(),'selected':chosen,'fit_selected_parameters':freeze['parameters'],'passing':decision['passing'],'candidates':len(protocol['candidate_specs']),
          'gate_candidates':len(freeze['retained']),'gate_axes_per_candidate':int(counts['axes'].iloc[0]),'source_files_verified':len(PINS),'source_changes':[],
          'new_model_training':False,'new_model_inference':False,'Test_or_query_read':False,'new_submission_CSV':False,
          'report_sha256':sha(OUT/'R19_MULTI_SLICE_GATE.md'),'selection_lock_sha256':sha(OUT/'SELECTION_LOCK.json')}
    write(OUT/'COMPLETED.json',done);print(json.dumps(done,ensure_ascii=False,indent=2))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    specs=candidate_specs()
    protocol={'name':'R19_multi_slice_gate_v1','registered_KST':now(),'baseline':'old_B1','candidate_specs':specs,
        'numerical_tolerance_absolute':TOL,'positive_bias':'max(weighted_mean(prediction-truth),0)',
        'fit':'fresh_fit_D0/H0 geometries equally weighted; one minimum-RMSE parameter per old_B1/B2 x offset/softgate family retained; ties prefer identity then smaller correction',
        'gate':'All four panels: overall RMSE, zero RMSE and positive bias, weak(0<y<=.10) RMSE and positive bias; additionally active>=12 positive RMSE in both native12 panels; no failed axis allowed',
        'adoption':'Among all passing fixed raw/generation candidates and fit-retained calibration candidates choose smallest fit RMSE; ties prefer old_B1; fallback old_B1 if no strict improvement',
        'report':'Loaded only after SELECTION_LOCK. Final verification never alters chosen policy. This is historically exposed local_report, not an untouched blind test.',
        'classification':'Independent probability is unchanged by regression correction; AUC reported, not used to optimize score in this run',
        'restrictions':{'model_training':False,'new_model_inference':False,'Test_or_query_reads':False,'submission_file_reads':False,'new_submission_CSV':False},
        'code_sha256':sha(Path(__file__))}
    protocol_path=OUT/'PROTOCOL.json'
    if protocol_path.exists():
        if '--resume-render' in sys.argv and not (OUT/'COMPLETED.json').exists():return resume_render()
        raise FileExistsError('Frozen output exists; select a new version rather than overwriting')
    write(protocol_path,protocol)
    fits={name:load_panel(name) for name in FIT}
    fitmetrics=eval_specs(fits,specs);fitmetrics.to_csv(OUT/'fit_all_candidates_metrics.csv',index=False)
    ranking=fit_rank(fitmetrics);ranking.rename('fit_RMSE').to_csv(OUT/'fit_candidate_ranking.csv')
    retained=[name for name,spec in specs.items() if spec['type'] in ('raw','generation_mix')]
    params=[]
    for base in ('old_B1','old_B2'):
        for kind in ('offset','softgate'):
            names=[name for name,spec in specs.items() if spec['type']==kind and spec['base']==base]
            def order(name):
                s=specs[name];change=s.get('delta',1-s.get('rho',1))
                return (float(ranking[name]),change,name)
            best=min(names,key=order);retained.append(best);params.append({'base':base,'family':kind,'selected_candidate':best,'fit_RMSE':float(ranking[best]),'spec':specs[best]})
    write(OUT/'FIT_CALIBRATION_FREEZE.json',{'created_KST':now(),'PROTOCOL_sha256':sha(protocol_path),'retained':retained,'parameters':params,'gate_not_yet_read_in_this_run':True})
    gatepanels={name:load_panel(name) for name in GATE}
    gm=eval_specs(gatepanels,{name:specs[name] for name in retained});gm.to_csv(OUT/'gate_candidate_metrics.csv',index=False)
    checks=gates(gm,retained);checks.to_csv(OUT/'gate_axis_checks.csv',index=False)
    checks[~checks.passed].to_csv(OUT/'gate_failure_reasons.csv',index=False)
    eligible=checks.groupby('candidate').passed.all();passing=eligible[eligible].index.tolist()
    selected=min(passing,key=lambda n:(float(ranking[n]),n!='old_B1',n)) if passing else 'old_B1'
    strict=float(ranking[selected])<float(ranking['old_B1'])-TOL
    if not strict:selected='old_B1'
    counts=checks.groupby('candidate').agg(axes=('passed','size'),failed_axes=('passed',lambda x:int((~x).sum())),max_delta=('delta','max')).reset_index()
    counts['fit_RMSE']=counts.candidate.map(ranking);counts['passes_all']=counts.failed_axes.eq(0);counts['selected']=counts.candidate.eq(selected)
    counts.to_csv(OUT/'gate_adoption_summary.csv',index=False)
    decision={'created_KST':now(),'PROTOCOL_sha256':sha(protocol_path),'fit_freeze_sha256':sha(OUT/'FIT_CALIBRATION_FREEZE.json'),'selected':selected,'selected_spec':specs[selected],
              'passing':passing,'strict_fit_improvement':strict,'reason':'strict multi-slice nondegradation then fit RMSE only' if selected!='old_B1' else 'No strictly fit-improving candidate passed all prespecified gate axes; preserve old_B1',
              'report_used_for_selection':False,'source_predictions_reused':True,'new_training':False,'new_inference':False,'Test_read':False}
    write(OUT/'SELECTION_LOCK.json',decision)
    report={'local_report':load_panel('local_report')}
    reportmetrics=eval_specs(report,{name:specs[name] for name in retained});reportmetrics.to_csv(OUT/'report_candidate_metrics_AFTER_LOCK.csv',index=False)
    # Show model-regime breakdown too, strictly descriptive after the lock.
    breakdown=[]
    for panel,(d,m) in {**fits,**gatepanels,**report}.items():
        for regime in ('D0','H0','D0H0_50_50'):
            for name,(s,c) in raw(m,regime).items():breakdown.extend(metrics(d,s,c,{'panel':panel,'candidate':name,'regime':regime}))
    pd.DataFrame(breakdown).to_csv(OUT/'individual_models_and_fixed_blends_all_regimes.csv',index=False)
    write(OUT/'read_order.json',READ_LOG)
    # Recorded training scope, not an invented equal-size five-fold dataset.
    provenance=pd.read_csv(pin(ROOT/'reports/r18_public_gap_audit_r1/predictions/model_training_provenance.csv'))
    provenance.to_csv(OUT/'recorded_training_scope.csv',index=False)
    oldtree=provenance[(provenance.generation=='old')&provenance.family.eq('lightgbm')]
    show=gm[gm['slice'].isin(['all','y=0','0<y<=0.10','positive_active>=12'])&gm.candidate.isin(['old_B1','old_B2',selected,'new_B2'])][['panel','candidate','slice','n','rmse','bias','positive_bias','auc']]
    reportshow=reportmetrics[reportmetrics['slice'].isin(['all','y=0','0<y<=0.10','positive_active>=12'])&reportmetrics.candidate.isin(['old_B1','old_B2',selected,'new_B2'])][['candidate','slice','n','rmse','bias','positive_bias','auc']]
    md=f'''# R19 저장 예측 기반 다중 구간 게이트

완료: {now()}. 추가 학습·모델 추론·Test/query/제출 파일 읽기·새 제출 CSV 생성 **0회**.

## 결론

채택 정책: **{selected}**. {decision['reason']}

`old_B1`은 R17 B1(LGB60%, DINO25%, Dasheng15%, 각 D0:H0=50:50)이다. 공식 점수0.789471136은 사용자가 제공한 과거 결과이며 이번 후보 선택에는 사용하지 않았다. old_B2는50/30/20이다. 수정된 출력이 공식 점수를 개선한다고 주장하지 않는다.

## 사전 정의한 채택 조건

각 local_gate·domain_holdout·native12/HF·native12/SF에서 전체 RMSE, y=0 RMSE/양의 bias, 0<y≤.10 RMSE/양의 bias가 old_B1보다 악화하지 않아야 한다. native12 두 패널은 활성≥12분 양성 RMSE도 통과해야 한다. 양의 bias는 max(mean(pred−y),0), 허용 오차는 수치상1e−6이다. native12의 원래 양/음성50:50 가중치를 유지하고 각 slice 안에서만 정규화한다. 임의 유의성 한계나 평균 개선으로 실패 축을 상쇄하지 않는다.

보정 후보는 old_B1/B2의 미세 출력 offset δ={{0,.0025,.005,.01}} 및 독립 score 소프트게이팅 ρ={{.9,.95,1}}뿐이다. offset은 clip(share−δ,0,1), softgate는 share*(ρ+(1−ρ)*p)다. score는 바꾸지 않는다. 이는 셀 물리상한이 아니며 약한 양성/장시간 손실은 같은 게이트에서 확인한다. 각 base×보정family의 파라미터는 fresh_fit_D0/H0 평균 MSE만으로 하나 고른 뒤 고정했다. gate에서는 그4후보 및 사전 고정raw/세대혼합만 비교했다. report는 SELECTION_LOCK 뒤 읽었으며 채택을 바꾸지 않았다.

## 전체 후보 채택 요약

{table(counts)}

한 후보의 gate 축 수는{int(counts['axes'].iloc[0])}개다. 전체 후보의 자세한 실패 축은 `gate_failure_reasons.csv`, fit에서 제외된 보정 후보까지의 전체 성적은 `fit_all_candidates_metrics.csv`에 있다.

## 주요 후보 gate 구간별 성적

{table(show)}

## 동결 후 local_report 확인

{table(reportshow)}

local_report 및 고정 stress 패널은 과거 개발과 사용자 보고에서 이미 노출됐다. 이번 실행 순서는 분리했지만 새 블라인드 테스트로 부를 수 없다. 보고서의 결과로 SELECTION_LOCK 정책을 재선택하지 않았다.

## 기록된 실제 학습 범위

{table(oldtree[['regime','task','fold','train_rows','selection_rows','input_feature_count']])}

old LGB는 각 fold의 실제 train_n이 다르므로 ‘61,446×5개 독립 배경’이라고 부를 수 없다. 신경망 old/new도 단일 source split이며 D0/H0 및 합성 반복 행은 독립 원천 수가 아니다. 전체 모델 범위는 `recorded_training_scope.csv`에 보존했다.

## 무결성과 제한

모든 입력 NPZ ID 정렬·receipt input SHA·prediction SHA·[0,1]/유한수 범위를 검증했다. 원본 모델·예측·제출 파일은 수정하지 않았다. 이 감사는 saved prediction의 보정/선별이며 새로운 학습이나 추론이 아니다. Test 평균·분위수·공식 점수로 파라미터를 맞추지 않았다.
'''
    (OUT/'R19_MULTI_SLICE_GATE.md').write_text(md)
    write(OUT/'INPUT_SHA256.json',PINS)
    changed=[name for name,h in PINS.items() if sha(ROOT/name)!=h];assert not changed
    outputs={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ('OUTPUT_SHA256.json','COMPLETED.json')}
    write(OUT/'OUTPUT_SHA256.json',outputs)
    done={'status':'completed','created_KST':now(),'selected':selected,'fit_selected_parameters':params,'passing':passing,'candidates':len(specs),'gate_candidates':len(retained),'gate_axes_per_candidate':int(counts['axes'].iloc[0]),
          'source_files_verified':len(PINS),'source_changes':changed,'new_model_training':False,'new_model_inference':False,'Test_or_query_read':False,'new_submission_CSV':False,
          'report_sha256':sha(OUT/'R19_MULTI_SLICE_GATE.md'),'selection_lock_sha256':sha(OUT/'SELECTION_LOCK.json')}
    write(OUT/'COMPLETED.json',done);print(json.dumps(done,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
