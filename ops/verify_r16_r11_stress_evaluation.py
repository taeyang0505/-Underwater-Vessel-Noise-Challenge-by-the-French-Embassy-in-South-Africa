"""Read-only numerical/source verification of the completed R11 stress run."""
import csv, datetime, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops.r16_r11_stress_evaluation import OUT,R11,MODELROOT,read,sha,dump,npz,guard,inventory,csvwrite,POS,NEG

def main():
    sys.addaudithook(guard)
    import pandas as pd
    from sklearn.metrics import mean_squared_error,mean_absolute_error,roc_auc_score
    result=read(OUT/'RESULTS.json')
    for group in ('input_source_sha256','model_policy_sha256','code_sha256'):
        for p,h in result[group].items():
            if sha(ROOT/p)!=h:raise ValueError('Pinned source changed:'+p)
    for p,h in result['output_sha256'].items():
        if sha(OUT/p)!=h:raise ValueError('Output changed:'+p)
    records=inventory(); original=0; maximum=0.; count=0
    original_maps={
      'full_support_hf_r1':read(R11/'full_support_hf_r1/FROZEN_BEFORE_PREDICTIONS.json')['panel_file_sha256'],
      'asymmetric_eq_r1':read(R11/'asymmetric_eq_r1/FORWARD_BEFORE_MODEL.json')['data_artifact_sha256'],
      'asymmetric_feasible_models_r1':read(R11/'asymmetric_feasible_models_r1/FORWARD_BEFORE_MODEL.json')['artifact_sha256'],
      'st_francis_density_r2':read(R11/'st_francis_density_r2/delivery_manifest.json')['files_sha256']}
    full=pd.read_parquet(ROOT/'data/versions/vtuad_full_r9/reference_manifest.parquet')
    aa=pd.read_parquet(ROOT/'reports/r16_cross_operator_r1/data/reference_partitions.parquet')
    aa=aa[aa.r16_reference_partition.eq('A')]
    rows=[];refs_by_panel={}
    for rec in records:
        matched=[h for p,h in original_maps[rec['group']].items() if Path(p).name==rec['path'].name]
        if len(matched)!=1 or sha(rec['path'])!=matched[0]:raise ValueError('Original frozen panel SHA mismatch')
        original+=1;data=npz(rec['path']);n=len(data['share']);folder=rec['path'].parent
        if 'reference_index' in data:refs=data['reference_index'].astype(int);kind='saved reference_index'
        elif rec['group']=='st_francis_density_r2':
            bank={int(v['treatment_id']):int(v['reference_index']) for v in read(folder/'treatment_bank.json')}
            recipes=[json.loads(s) for s in (folder/(rec['mode']+'_recipes.jsonl')).read_text().splitlines()]
            refs=np.array([bank.get(int(r['treatment_id']),-1) for r in recipes]);kind='saved treatment bank/recipe'
        elif rec['mode']=='r9_original':
            refs=np.array([v['reference_index'] for v in read(folder/'positive_schedule.json')]);kind='saved positive_schedule (original recipe lacks ref index)'
        elif (folder/(rec['mode']+'_recipes.json')).exists():
            refs=np.array([v.get('reference_index',-1) for v in read(folder/(rec['mode']+'_recipes.json'))]);kind='saved per-mode recipe'
        else:refs=np.full(n,-1);kind='known A=0 background-only panel'
        if len(refs)!=n:raise ValueError('Reference row alignment differs')
        active=(data['share']>0)&(refs>=0);used=set(refs[active]);assigned=set(refs[refs>=0]);selected=full[full.reference_row_index.isin(used)]
        name='R11_'+rec['mode'] if rec['group']=='full_support_hf_r1' else 'R11_'+rec['group']+'__'+rec['mode']
        refs_by_panel[name]=refs
        rows.append({'panel':name,'reference_lineage':kind,'rows':n,'observed_positive_rows':int((data['share']>0).sum()),
          'assigned_reference_unique':len(assigned),'observed_positive_reference_unique':len(used),
          'observed_positive_reference_exact_overlap_with_R16_A':len(used&set(aa.reference_row_index)),
          'observed_positive_hull_overlap_with_R16_A':len(set(selected.hull_id)&set(aa.hull_id)),
          'observed_positive_day_overlap_with_R16_A':len(set(selected.recording_day)&set(aa.recording_day)),
          'unknown_positive_reference_rows':int(((data['share']>0)&(refs<0)).sum()),
          'no_reference_independence_claim':True})
    if original!=31 or any(r['unknown_positive_reference_rows'] for r in rows):raise ValueError('Reference lineage incomplete')
    # Independently recalculate all new small-model metrics from their saved arrays.
    bykey={(r['panel'],r['model']):r for r in result['metrics']}
    safe=OUT/'predictions_numeric';safe.mkdir()
    pool=pd.read_parquet(ROOT/'data/versions/background_cv_r9/background_pool_manifest.parquet').set_index('base_window_index')
    safe_sha={}
    for p in sorted((OUT/'predictions').glob('*.npz')):
        # The first export's pandas-derived `site` metadata is object dtype.
        # Never unpickle it: reconstruct Unicode sites from the pinned raw metadata.
        with np.load(p,allow_pickle=False) as zz:d={k:zz[k] for k in zz.files if k!='site'}
        if 'base_window_index' in d:
            d['site']=pool.loc[d['base_window_index'],'site'].to_numpy(dtype=str)
            d['reference_index']=refs_by_panel[p.stem]
        safe_file=safe/p.name
        np.savez_compressed(safe_file,**d)
        checked=npz(safe_file)
        if any(v.dtype.kind=='O' for v in checked.values()):raise ValueError('Object dtype in numeric export')
        for k in ('target_share','share','independent_classifier','frozen_score'):
            if not np.array_equal(d[k],checked[k]):raise ValueError('Numeric copy changed')
        safe_sha[str(safe_file.relative_to(OUT))]=sha(safe_file)
        y=d['target_share'].astype(float);z=y>0
        if len(set(d['sample_id'].astype(str)))!=len(y):raise ValueError('Duplicate row IDs')
        for j,model in enumerate(d['model_order'].tolist()):
            pred=d['share'][:,j].astype(float);c=d['independent_classifier'][:,j];score=d['frozen_score'][:,j]
            expected={'rmse':float(np.sqrt(mean_squared_error(y,pred))),'mae':float(mean_absolute_error(y,pred)),'bias':float((pred-y).mean())}
            for key,v in [('auc_share',pred),('auc_independent_classifier',c),('auc_frozen_score',score)]:
                expected[key]=float(roc_auc_score(z,v)) if z.any() and (~z).any() else None
            stored=bykey[p.stem,model]
            for key,value in expected.items():
                if value is None:
                    if stored[key] is not None:raise ValueError('Undefined single-class AUC was filled')
                else:
                    delta=abs(value-stored[key]);maximum=max(maximum,delta);count+=1
                    if delta>1e-12:raise ValueError('Metric mismatch')
    csvwrite(OUT/'source_reference_lineage_verified.csv',rows)
    selected_models=('control_NN','control_Tree','control_60T40N','mixed_NN','mixed_Tree','mixed_60T40N','R9_all5_frozen_Fold0','R14_fast3_exactFold0')
    table=[r for r in result['metrics'] if r['group']=='standard_R11_pair' and r['model'] in selected_models]
    csvwrite(OUT/'standard_pairs_all_models.csv',table)
    lines=['# R11 고정 스트레스 최종 검산','',
      'R16 교차 연산자 검증을 주 기준으로 유지한다. 아래는 노출된 R11의 사후 스트레스 진단이며, 이 결과로 원래 레시피·모델·정책을 변경하지 않았다.','',
      '| 조합 | 모델 | RMSE | 독립 classifier AUC | 동결 score AUC |','|---|---|---:|---:|---:|']
    for r in table:
        c=r['auc_independent_classifier'];lines.append(f"| {r['panel']} | {r['model']} | {r['rmse']:.6f} | {c:.6f}"+f" | {r['auc_frozen_score']:.6f} |" if c is not None else f"| {r['panel']} | {r['model']} | {r['rmse']:.6f} | 미저장 | {r['auc_frozen_score']:.6f} |")
    lines +=['','31개 원본 음향 패널은 최초 R11 생성 동결 manifest의 SHA와 모두 일치했다. 새 small 모델 예측 39파일에서 RMSE/MAE/bias 및 세 종류 AUC를 별도 sklearn 식으로 재계산했다.','',
      '원천 메타 보강: 첫 source_overlap.csv의 r9_original에는 원래 recipe에 reference_index가 없어서 0이 기재됐다. source_reference_lineage_verified.csv는 당시 저장된 positive_schedule로 연결을 복구한다. 이 0을 참조 미사용이나 미관측 참조로 해석하면 안 된다. 또한 모든 패널에서 할당된 참조와 실제 관측 양성에 기여한 참조를 구분했다. 원본 음향/예측/수치지표는 수정하지 않았다.','',
      '안전한 재사용 파일은 predictions_numeric/ 아래39개다. 첫 predictions/의 site 메타는 pandas object dtype이므로 pickle로 열지 않고, 동결 원천 메타에서 Unicode로 다시 기록했다. 잘못 누락된 r9_original 참조 인덱스도 저장 schedule로 채웠다. share/classifier/score/정답 배열은 첫 출력과 비트 단위로 동일하며 원형 파일은 보존했다.','',
      'R11 원천은 실제 R16 TRAIN4,000 배경과 ID/group 중복0, 같은 sensor의900초 지원 구간 최소25.25시간 이상 분리다. 참조는 대부분 R16 A와 겹친다. native_long15도 clip 자체는 A와 안 겹치지만 hull/day 중복이 있어 별도 미관측 원천 일반화로 단정하지 않는다.','',
      '평탄·늘림 full_smooth, 자연 native_long15, 비대칭 EQ, SF 관측밀도는 서로 다른 스트레스다. 시간 불변 입력이나 이 평가의 실패가 실제 시계열 복원의 불가능성을 증명하지 않는다. 대회 generator의 원인이나 공식 점수를 추론하지 않는다. SF5view는 배경 source 반복이며 sparse/dense의 시기 차이가 남는다.','',
      '[전체532지표](metrics.csv) · [정밀 참조 계보](source_reference_lineage_verified.csv) · [기존23모델의 측정 가용성](old_model_comparison_availability.csv) · [원 보고서](README.md)']
    (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    report={'status':'passed','scope':'saved R11 acoustic stresses only; no fitting/new inference in verification',
      'original_R11_frozen_sample_SHA_matches':original,'new_small_prediction_files':39,'independent_metric_values':count,
      'max_abs_metric_error':maximum,'source_reference_lineage_rows':rows,'original_artifacts_unchanged':True,
      'R16_primary_benchmark_unchanged':True,'Test_or_Testderived_read':False,
      'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'code_sha256':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
      'completed_result_sha256':sha(OUT/'RESULTS.json'),
      'numeric_prediction_file_sha256':safe_sha,'numeric_exports_allow_pickle_false_passed':True,
      'new_output_sha256':{p.name:sha(p) for p in [OUT/'source_reference_lineage_verified.csv',OUT/'standard_pairs_all_models.csv',OUT/'SUMMARY.md']}}
    dump(OUT/'FINAL_VERIFICATION.json',report);print(json.dumps({'status':'passed','metric_error':maximum,'count':count,'sha256':sha(OUT/'FINAL_VERIFICATION.json')}))

if __name__=='__main__':main()
