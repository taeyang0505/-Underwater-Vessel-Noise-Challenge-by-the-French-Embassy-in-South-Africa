"""Thirty fixed global regression/score blends: feasibility only, no policy.

Cached17-panel R13 predictions only. No inference, fitting, Test, source
matching, panel-specific coefficients or edits to original outputs.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BETAS=(0.,.25,.5,.75,1.);LAMBDAS=(0.,.05,.1,.25,.5,1.)
NORMAL=('R9_selection','R9_ensemble_gate','R10_unseen__r9_complete')
SF=('R11_full_smooth15_VS_st_francis_dense','R11_native_long15_VS_st_francis_dense')
HIGH='R12_high_snr_long'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def csvwrite(p,rows):
    with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(y,p,s):
    import numpy as np
    from ops.verify_r12_evaluation_v2 import pair_auc
    y=np.asarray(y,float);p=np.asarray(p,float);s=np.asarray(s,float);e=p-y
    return {'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'absolute_bias':float(abs(e.mean())),
        'score_AUC':pair_auc(y,s),'share_AUC':pair_auc(y,p),'presence_Brier':float(np.mean((s-(y>0))**2)),
        'prediction_mean':float(p.mean()),'prediction_RMS':float(np.sqrt(np.mean(p*p))),
        'score_mean':float(s.mean()),'score_ge099':float(np.mean(s>=.99))}
def run(a):
    import numpy as np
    from ops.verify_r12_evaluation_v2 import guard
    sys.addaudithook(guard);source=(ROOT/a.source).resolve();out=(ROOT/a.output).resolve()
    if not source.is_relative_to(ROOT) or not out.is_relative_to(ROOT) or out.exists():raise ValueError('Freshproject-containedoutputrequired')
    result=read(source/'RESULTS.json');policy=read(source/'selected_policy.json')
    if result.get('status')!='completed' or result.get('format')!='r13_cpu_evaluation_v1' or result['selected_policy']!=policy:raise ValueError('CompleteimmutableR13snapshotrequired')
    for n,h in result['output_sha256'].items():
        if Path(n).is_absolute() or '..' in Path(n).parts or sha(source/n)!=h:raise AssertionError('OriginalR13outputchanged')
    out.mkdir(parents=True);start=time.perf_counter();pin={n:sha(source/n) for n in result['output_sha256']}
    plan={'format':'r13_fixed30_global_safety_plan_v1','created_unix':time.time(),'source_results_sha256':sha(source/'RESULTS.json'),
        'source_policy_sha256':sha(source/'selected_policy.json'),'source_outputs_sha256':pin,
        'regression_R13_weight':BETAS,'score_R13_classifier_weight':LAMBDAS,
        'regression':'(1-b)*storedR12matchedshare+b*storedR13selectedshare;float64arithmeticthenfloat32',
        'score':'(1-lambda)*thatregression+lambda*R13selectedarmindependenttreeclassifier;float64thenfloat32',
        'one_global_pair_used_on_every_panel':True,'classifier_arm':policy['selected_arm'],
        'normal_primary':NORMAL[0],'normal_auxiliary':list(NORMAL[1:]),'strict_normal_all':list(NORMAL),
        'normal_AUC_constraint':'>0.9','SF_AUC_constraints':{p:'>=0.8' for p in SF},
        'highSNR_panel':HIGH,'highSNR_RMSE_constraint':'<0.15634','highSNR_abs_bias_constraint':'<0.11137',
        'threshold_precision_note':'Literaluser-specifiedroundedthresholds;notastatisticalimprovementtestversusfullprecisionbaseline.',
        'Pareto_objectives':['maximizeprimarynormalAUC','maximizefull15SF_AUC','maximizenativeSF_AUC','minimizehighSNR_RMSE','minimizehighSNR_absbias'],
        'no_new_policy_or_selection':True,'no_panel_specific_lambda_or_exceptions':True,'development_exposed_not_blind':True,
        'new_fitting_inference_Test_or_production_change':False,'script_sha256':sha(__file__)}
    write(out/'PREDECLARED_PLAN.json',plan)
    combinations=[{'b':b,'lambda':l,'name':f'b{b}_clf{l}'} for b in BETAS for l in LAMBDAS]
    data={};proof={};sourceR12=ROOT/'reports/r12_mask_matched_evaluation_r2'
    paths=sorted((source/'predictions').glob('*.npz'))
    if len(paths)!=17:raise AssertionError('Exactly17savedpanelsrequired')
    for path in paths:
        with np.load(path,allow_pickle=False) as z:
            order=z['model_order'].astype(str).tolist();y=z['target_share'].copy();ids=z['sample_id'].astype(str)
            r13=z['share'][:,order.index('R13_selected')].copy();r12=z['share'][:,order.index('R12_matched_share_only')].copy();clf=z[policy['selected_arm']+'_classifier'].copy()
            selected='Blend50_'+policy['selected_arm']+'_lambda'+str(policy['score_classifier_weight'])
            if not np.array_equal(r13,z['share'][:,order.index(selected)]):raise AssertionError('R13selectedregressionidentitydiffers')
        with np.load(sourceR12/'predictions'/path.name,allow_pickle=False) as old:
            if not np.array_equal(ids,old['sample_id'].astype(str)) or not np.array_equal(y,old['target_share']):raise AssertionError('CachedR12/R13sourceIDortargetorderdiffers')
            j=old['model_order'].astype(str).tolist().index('Blend50_L1HM1P1')
            if not np.array_equal(r12,old['share'][:,j]):raise AssertionError('R12matchedbaselinevalueschanged')
        if any(not np.isfinite(v).all() or ((v<0)|(v>1)).any() for v in (r12,r13,clf,y)):raise AssertionError('Invalidcachednumericarray')
        data[path.stem]=(y,ids,r12,r13,clf);proof[path.stem]={'R13_prediction_sha256':sha(path),
            'R12_prediction_sha256':sha(sourceR12/'predictions'/path.name),'rows':len(y),'ID_target_order_and_R12_baseline_exact':True}
    allmetrics={c['name']:{} for c in combinations};rows=[]
    for panel,(y,ids,r12,r13,clf) in data.items():
        regression=[];scores=[]
        for c in combinations:
            b,l=c['b'],c['lambda'];p=((1-b)*r12.astype(float)+b*r13.astype(float)).astype(np.float32);s=((1-l)*p.astype(float)+l*clf.astype(float)).astype(np.float32)
            m=metrics(y,p,s);allmetrics[c['name']][panel]=m;rows.append({'combination':c['name'],'b':b,'classifier_lambda':l,'panel':panel,**m});regression.append(p);scores.append(s)
        np.savez_compressed(out/(panel+'_candidates.npz'),sample_id=ids,target_share=y,combination_order=np.asarray([c['name'] for c in combinations]),
            regression=np.column_stack(regression),score=np.column_stack(scores),R12_share=r12,R13_share=r13,R13_classifier=clf)
    feasibility=[];vectors=[]
    for c in combinations:
        m=allmetrics[c['name']];norm=[m[p]['score_AUC'] for p in NORMAL];full,native=[m[p]['score_AUC'] for p in SF];h=m[HIGH]
        checks={'normal_primary_gt090':norm[0]>.9,'all_three_normals_gt090':all(v>.9 for v in norm),
            'full15_SF_ge080':full>=.8,'native15_SF_ge080':native>=.8,'highSNR_RMSE_lt015634':h['rmse']<.15634,
            'highSNR_abs_bias_lt011137':h['absolute_bias']<.11137}
        common=all(checks[k] for k in ['full15_SF_ge080','native15_SF_ge080','highSNR_RMSE_lt015634','highSNR_abs_bias_lt011137'])
        f={'combination':c['name'],'b':c['b'],'classifier_lambda':c['lambda'],
            'R9_selection_AUC':norm[0],'R9_ensemble_gate_AUC':norm[1],'R10_unseen_normal_AUC':norm[2],
            'full15_SF_AUC':full,'native15_SF_AUC':native,'highSNR_RMSE':h['rmse'],'highSNR_bias':h['bias'],
            **checks,'primary_all_constraints_feasible':checks['normal_primary_gt090'] and common,
            'strict_three_normal_all_constraints_feasible':checks['all_three_normals_gt090'] and common}
        feasibility.append(f);vectors.append([-norm[0],-full,-native,h['rmse'],h['absolute_bias']])
    vectors=np.asarray(vectors)
    for i,row in enumerate(feasibility):row['Pareto_nondominated_primary_axes']=not any(np.all(vectors[j]<=vectors[i]) and np.any(vectors[j]<vectors[i]) for j in range(len(vectors)) if j!=i)
    csvwrite(out/'feasibility_30.csv',feasibility);csvwrite(out/'all_panel_metrics_510.csv',rows)
    for n,h in pin.items():
        if sha(source/n)!=h:raise AssertionError('OriginalR13outputmodifiedduringanalysis')
    if sha(source/'selected_policy.json')!=plan['source_policy_sha256'] or sha(source/'RESULTS.json')!=plan['source_results_sha256']:raise AssertionError('Originalpolicy/resultsmodified')
    feasible=[r['combination'] for r in feasibility if r['primary_all_constraints_feasible']];strict=[r['combination'] for r in feasibility if r['strict_three_normal_all_constraints_feasible']]
    report={'format':'r13_fixed30_safety_feasibility_v1','status':'completed','plan_file_sha256':sha(out/'PREDECLARED_PLAN.json'),
        'combinations':30,'panels':17,'metric_rows':len(rows),'feasibility':feasibility,'primary_feasible_combinations':feasible,
        'strict_three_normal_feasible_combinations':strict,'any_primary_feasible':bool(feasible),'any_strict_feasible':bool(strict),
        'Pareto_nondominated_combinations':[r['combination'] for r in feasibility if r['Pareto_nondominated_primary_axes']],
        'source_identity_checks':proof,'no_policy_selected':True,'original_R13_policy_results_models_unchanged':True,
        'no_fitting_inference_Test_or_cohort_matching':True,'development_not_blind':True,
        'interpretation':'Finiteprespecifiedglobalblendfeasibilityonly;noLBpredictionandnoglobalmathematicalimpossibilityclaimoutside30candidates.',
        'next_step_if_none':'These30simplefixedblendscannotmeetallrequestedconditionssimultaneously;furthertraining-data/source-validationredesignisneededbeforeclaimingasafesolution.',
        'elapsed_seconds':time.perf_counter()-start,'script_sha256':sha(__file__)}
    write(out/'RESULTS.json',report)
    text=['# R13 전역 안전 혼합: 사전 고정30조합 feasibility','',
        '원 R13 정책·예측·모델은 그대로 두고, 같은17패널의 저장 예측만 사용했습니다. b와 classifier λ는 모든 패널에 동일하게 적용했습니다. 패널별 계수·예외·새 정책 선택은 없습니다.',
        f"기본 normal=R9 selection 기준 모든 조건 충족: **{bool(feasible)}**. 세 normal 패널 모두 AUC>0.9까지 요구한 엄격 조건 충족: **{bool(strict)}**.",
        '요구 조건은 normal AUC>0.9, full15+SF 및 native15+SF AUC≥0.8, highSNR RMSE<0.15634 및 |bias|<0.11137입니다. 주어진 소수점 threshold를 문자 그대로 비교했으며 통계적 유의 개선으로 해석하지 않습니다.',
        '', '|b|classifier λ|normal AUC|full15 SF AUC|native SF AUC|highSNR RMSE|highSNR bias|기본 모두 충족|엄격 모두 충족|','|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for r in feasibility:text.append(f"|{r['b']}|{r['classifier_lambda']}|{r['R9_selection_AUC']:.8f}|{r['full15_SF_AUC']:.8f}|{r['native15_SF_AUC']:.8f}|{r['highSNR_RMSE']:.8f}|{r['highSNR_bias']:+.8f}|{r['primary_all_constraints_feasible']}|{r['strict_three_normal_all_constraints_feasible']}|")
    text+=['','Pareto 표는 다섯 요청 축에서 다른 조합에 모두 열등하지 않은 조합을 표시할 뿐 정책을 선택하지 않습니다. 범위 밖 연속 계수나 새 모델에서도 불가능하다는 전역 증명은 아닙니다.',
        '이30개 단순 전역 혼합으로 동시 조건을 못 맞춘다면, 안전하게 해결됐다고 주장하기 전에 추가 학습 데이터와 원천 독립 검증 설계를 재검토해야 합니다. 점수만 바꾸어 에너지 회귀 오차를 고칠 수는 없습니다.',
        '', '[전체 제약·Pareto·SHA](RESULTS.json) / [30개 조합 표](feasibility_30.csv) / [17패널510행 전체 지표](all_panel_metrics_510.csv)']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n');print(json.dumps({'status':'completed','primary_feasible':feasible,'strict_feasible':strict,'elapsed_seconds':report['elapsed_seconds']}));return report
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default='reports/r13_cpu_correction_r1');p.add_argument('--output',default='reports/r13_safety_feasibility_r1');run(p.parse_args())
