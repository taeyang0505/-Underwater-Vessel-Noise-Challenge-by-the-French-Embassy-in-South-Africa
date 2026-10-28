"""Saved-prediction-only R11 selection/query distribution audit.

No model loading, inference, training, calibration, threshold/weight tuning,
target recovery, or public-score/RMS matching. Query truth is unknown.
Only query availability metadata is newly extracted from the input db.
"""
from __future__ import annotations
import csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pipeline.inference_r5 import FEATURE_COLUMNS
from ops.r10_saved_prediction_audit import align

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'submissions/r9_fold0_all5_cpu_r1'
CPU=ROOT/'reports/r10_diagnostics/temporal_permutation_r1'
MEMBERS=('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer','lgbm_physics')
THRESHOLDS=(0.,.01,.05,.5,.95,.99,1.)
QUANTILES=(0.,.01,.05,.25,.5,.75,.95,.99,1.)
COMPONENT_TOLERANCE=3e-6


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def corr(x,y,method):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<2:return None
    if method=='spearman':x=rankdata(x,method='average');y=rankdata(y,method='average')
    x=x-x.mean();y=y-y.mean();den=float(np.linalg.norm(x)*np.linalg.norm(y))
    return None if den==0 else float(np.clip(np.dot(x,y)/den,-1,1))


def availability(db):
    o=np.isfinite(db);complete=o.all(2);any_=o.any(2)
    return pd.DataFrame({'Nobs':complete.sum(1),'any_observed_minutes':any_.sum(1),
       'partial_missing_minutes':(any_&~complete).sum(1),
       'first_completely_missing':~any_[:,0],'last_completely_missing':~any_[:,-1],
       'first_not_fully_observed':~complete[:,0],'last_not_fully_observed':~complete[:,-1],
       'missing_pattern':[','.join(str(i+1) for i in np.flatnonzero(~row)) or 'none' for row in complete]})


def scopes(meta,role):
    yield 'all','all',np.ones(len(meta),bool)
    for n in range(3,16):
        use=meta.Nobs.to_numpy()==n
        if use.any():yield 'Nobs_exact',str(n),use
    yield 'Nobs_group','>=13',meta.Nobs.to_numpy()>=13
    for first in (False,True):
        for last in (False,True):
            use=(meta.first_completely_missing.to_numpy()==first)&(meta.last_completely_missing.to_numpy()==last)
            if use.any():yield 'endpoint_pattern',f'first_missing={first};last_missing={last}',use
    for pattern in sorted(meta.missing_pattern.unique()):yield 'missing_pattern',str(pattern),meta.missing_pattern.eq(pattern).to_numpy()
    label='historical_inferred_type_NOT_site_GT' if role=='query' else 'released_source_site_metadata'
    field='historical_inferred_type' if role=='query' else 'source_site'
    for value in sorted(meta[field].unique()):yield label,str(value),meta[field].eq(value).to_numpy()


def main():
    start=time.monotonic();out=ROOT/'reports/r11_diagnostics/saved_query_audit_r1'
    if out.exists():raise FileExistsError('Fresh audit output required')
    out.mkdir(parents=True)
    mapping=json.loads((ROOT/'ops/r9_fold0_pilot_nn_map.json').read_text());nn={k:Path(v) for k,v in mapping.items()}
    policy=json.loads((PILOT/'frozen_policy.json').read_text());receipt=json.loads((PILOT/'inference_receipt.json').read_text())
    witness=json.loads((PILOT/'independent_verification.json').read_text());cpu_witness=json.loads((CPU/'RESULTS.json').read_text())
    if canonical({k:v for k,v in policy.items() if k!='policy_sha256'})!=policy['policy_sha256']:raise ValueError('Policy changed')
    files=[PILOT/n for n in ['query_member_predictions.npz','submission.csv','inference_receipt.json','independent_verification.json','frozen_policy.json']]
    files += [CPU/'original_CPU_predictions.npz',CPU/'RESULTS.json',ROOT/'data/raw/Test.csv',
       ROOT/'data/synthetic/group_cv_minute_full_r9_v1/fold_00/selection.npz',
       ROOT/'data/synthetic/group_cv_minute_full_r9_v1/fold_00/selection_metadata.parquet',
       ROOT/'models/r9_mac_trees_r2/fold_00/models/lgbm_physics/inner_predictions.npz',
       ROOT/'models/r9_mac_trees_r2/fold_00/models/lgbm_physics/summary.json',
       ROOT/'eda_deep/tables/test_window_features.csv',ROOT/'eda_deep/deep_eda.py',
       ROOT/'reports/R5_EDA_INTEGRATION_FACTCHECK.md',ROOT/'ops/r9_fold0_pilot_nn_map.json']
    for k in MEMBERS[:4]:files += [nn[k]/'summary.json',nn[k]/'best_window_validation_predictions.npz']
    pins={n:h for n,h in policy['source_pins'].items()}
    if any(sha(ROOT/n)!=h for n,h in pins.items()):raise ValueError('Original policy source/model pins changed')
    pins.update({str(p.resolve().relative_to(ROOT)):sha(p) for p in files})
    if sha(ROOT/'data/raw/Test.csv')!='3ef819192c05c4308b66564e6307e6f4f26fe806afe96ec93e2d15f25a3474d4':raise ValueError('Original query bytes changed')
    for name,h in witness['original_file_sha256'].items():
        if sha(PILOT/name)!=h:raise ValueError('Original saved inference output changed: '+name)
    if sha(CPU/'original_CPU_predictions.npz')!=cpu_witness['output_sha256']['original_CPU_predictions.npz']:raise ValueError('CPU selection witness differs')
    plan={'format':'r11_saved_query_prediction_audit_plan_v1','source_script_sha256':sha(__file__),'input_sha256':pins,
          'expected_query_Test_sha256':receipt['query']['raw_query_csv_sha256'],
          'expected_selection_rows':1000,'expected_query_rows':1500,'primary_precision':'saved CPU FP32 for selection and query; native CPU tree',
          'selection_GPU_q_witness':'formula identity witness only; not exact primary CPU-q equivalence',
          'thresholds_cumulative_both_le_and_ge':list(THRESHOLDS),'quantile_levels':list(QUANTILES),
          'statistics':['mean','RMS','population_std','min','max','quantiles','fixed saturation fractions','unique values'],
          'correlations':['Pearson','Spearman average ties'],'matrix_scopes':['share5x5','recovered components5x5','share-to-component5x5','global all15 variables'],
          'query_db_read_scope':'Nobs, complete/partial missing minutes, first/last missing, missing-minute pattern only',
          'inferred_type_policy':'Reuse historical per-ID probe_site only if IDs and stored availability patterns agree; never actual site GT; no new matching',
          'component_formula':'NN contrast or tree classifier=2*score-share; range roundoff tolerance3e-6; raw and clipped values both retained',
          'prohibited':['new inference or model loading','fit or calibration','query labels or label recovery','public score/RMS matching','weight/threshold tuning','new site inference','report role access','server/external transmission']}
    write(out/'predeclared_plan.json',plan)
    # No predictor/runtime library is imported by this script. Query numerical
    # values are discarded after availability-only extraction below.
    with (ROOT/'data/raw/Test.csv').open(newline='',encoding='utf-8-sig') as f:
        r=csv.reader(f);header=next(r)
        if header!=['ID',*FEATURE_COLUMNS]:raise ValueError('Canonical375column query schema differs')
        ids=[];values=[]
        for row in r:
            ids.append(row[0]);values.append([float(v) if v.strip() else np.nan for v in row[1:]])
    if len(ids)!=1500 or len(set(ids))!=1500 or canonical(ids)!=receipt['ID_order_sha256']:raise ValueError('Original query IDs/order changed')
    db=np.asarray(values,np.float32).reshape(1500,15,25);qmeta=availability(db);del db,values
    qmeta.insert(0,'sample_id',ids)
    legacy=pd.read_csv(ROOT/'eda_deep/tables/test_window_features.csv',usecols=['ID','probe_site','probe_conf','pattern','n_missing'],dtype={'ID':str})
    ix=align(ids,legacy.ID.to_numpy());legacy=legacy.iloc[ix].reset_index(drop=True)
    valid_legacy=np.array_equal(legacy.pattern.astype(str),qmeta.missing_pattern.astype(str)) and np.array_equal(legacy.n_missing,15-qmeta.Nobs)
    qmeta['historical_inferred_type']=legacy.probe_site.astype(str).to_numpy() if valid_legacy else '__unavailable__'
    qmeta['historical_probe_vote_fraction_not_calibrated_GT']=legacy.probe_conf.to_numpy() if valid_legacy else np.nan
    legacy_status={'available':bool(valid_legacy),'artifact':'eda_deep/tables/test_window_features.csv','method_source':'eda_deep/deep_eda.py lines700-725, TRAIN-fit kNN25 site-domain probe',
                   'current_ID_and_availability_alignment_exact':bool(valid_legacy),'historical_raw_Test_SHA_receipt_available':False,
                   'actual_site_ground_truth':False,'new_assignment_computed':False,
                   'meaning':'Existing spectral-type/domain-probe class output; historical input-byte provenance not independently timestamp-certified'}
    with np.load(PILOT/'query_member_predictions.npz',allow_pickle=False) as z:
        order=align(ids,z['ID']);query={k:z[k][order] for k in ('share','score','ensemble_share','ensemble_score')}
    weights=np.asarray(policy['weights'],float)
    for kind in ['share','score']:
        calculated=np.clip(query[kind].astype(float)@weights,0,1).astype(np.float32)
        if not np.array_equal(calculated,query['ensemble_'+kind]):raise ValueError('Frozen ensemble expression differs')
    with (PILOT/'submission.csv').open(newline='') as f:
        rows=list(csv.reader(f))
    if rows[0]!=['ID','share_added_vessel','score_added_vessel'] or [r[0] for r in rows[1:]]!=ids:raise ValueError('CSV IDs/schema differ')
    for j,key in [(1,'ensemble_share'),(2,'ensemble_score')]:
        if not np.array_equal(np.asarray([float(r[j]) for r in rows[1:]]),query[key].astype(float)):raise ValueError('CSV prediction bytes/roundtrip differ')
    with np.load(ROOT/'data/synthetic/group_cv_minute_full_r9_v1/fold_00/selection.npz',allow_pickle=False) as z:
        sid=z['sample_id'].astype(str);smeta=availability(z['db']);truth=z['share'].astype(float);observed=np.isfinite(z['db'])
    smeta.insert(0,'sample_id',sid);actual=pd.read_parquet(ROOT/'data/synthetic/group_cv_minute_full_r9_v1/fold_00/selection_metadata.parquet')
    smeta['source_site']=actual.iloc[align(sid,actual.sample_id.to_numpy())].site.astype(str).to_numpy()
    with np.load(CPU/'original_CPU_predictions.npz',allow_pickle=False) as z:
        if tuple(z['member_order'].astype(str))!=MEMBERS+('frozen_all5',):raise ValueError('Selection CPU matrix member order differs')
        order=align(sid,z['sample_id']);selection={key:z[key][order] for key in ('share','score')}
    direct=[]
    for j,kind in enumerate(MEMBERS[:4]):
        with np.load(nn[kind]/'best_window_validation_predictions.npz',allow_pickle=False) as z:
            order=align(sid,z['sample_id']);q=z['cell_mask'][order][:,:,10:23];o=observed[:,:,10:23]
            floor=np.min(np.where(o,q,np.inf),1,keepdims=True);exact=np.where(o,np.maximum(q-floor,0),0).max((1,2))
            recovered=2*z['score'][order].astype(float)-z['share'][order].astype(float)
            error=float(np.abs(recovered-exact).max())
            if error>COMPONENT_TOLERANCE:raise ValueError('Source NN score decomposition differs')
            direct.append({'model':kind,'witness_precision':'saved GPU BF16 operator / FP32 q+pool','max_formula_error':error,
                           'primary_CPU_vs_GPU_share_max':float(np.abs(selection['share'][:,j]-z['share'][order]).max()),
                           'direct_primary_CPU_q_available':False})
    with np.load(ROOT/'models/r9_mac_trees_r2/fold_00/models/lgbm_physics/inner_predictions.npz',allow_pickle=False) as z:
        if not np.array_equal(selection['share'][:,4],z['share']):raise ValueError('Primary selection tree/order differs')
        error=float(np.abs(2*selection['score'][:,4].astype(float)-selection['share'][:,4].astype(float)-z['score_classifier']).max())
        if error>COMPONENT_TOLERANCE:raise ValueError('Tree score decomposition differs')
        direct.append({'model':'lgbm_physics','witness_precision':'native CPU','max_formula_error':error,'direct_primary_CPU_classifier_available':True})
    distributions=[];quantiles=[];saturation=[];correlations=[];recovery=[];availability_rows=[]
    for role,meta,p,score,label in [('selection',smeta,selection['share'],selection['score'],truth),
                                  ('query',qmeta,np.column_stack([query['share'],query['ensemble_share']]),
                                   np.column_stack([query['score'],query['ensemble_score']]),None)]:
        p=p.astype(float);score=score.astype(float)
        if p.shape!=(len(meta),6) or score.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(score).all():raise ValueError('Prediction shape/finite check')
        if ((p<0)|(p>1)|(score<0)|(score>1)).any():raise ValueError('Prediction bounds')
        raw=2*score-p;violation=np.maximum(0,np.maximum(-raw,raw-1))
        if violation.max()>COMPONENT_TOLERANCE:raise ValueError('Recovered component outside numerical tolerance')
        component=np.clip(raw,0,1)
        names=list(MEMBERS)+['frozen_all5']
        data={};kind_map={}
        for j,kind in enumerate(names):
            ctype='NN_contrast_recovered' if j<4 else 'tree_classifier_recovered' if j==4 else 'weighted_mixed_component'
            for suffix,value in [('share',p[:,j]),('score',score[:,j]),(ctype,component[:,j])]:
                key=kind+'__'+suffix;data[key]=value;kind_map[key]=(kind,suffix)
            recovery.append({'role':role,'model':kind,'roundoff_tolerance':COMPONENT_TOLERANCE,
                             'raw_min':float(raw[:,j].min()),'raw_max':float(raw[:,j].max()),
                             'range_violation_max':float(violation[:,j].max()),'range_clip_count':int((raw[:,j]!=component[:,j]).sum()),
                             'NN_source_logit_clamp_epsilon':1e-6 if j<4 else None})
            meta[kind+'_saved_predicted_share']=p[:,j];meta[kind+'_saved_score']=score[:,j]
            meta[kind+'_component_raw']=raw[:,j];meta[kind+'_component_bounded']=component[:,j]
        meta['known_synthetic_added_share_label']=label if label is not None else np.nan
        meta.insert(0,'role',role);meta.to_csv(out/(role+'_ID_availability_and_saved_prediction_components.csv'),index=False)
        sharekeys=[k+'__share' for k in MEMBERS];compkeys=[k+('__NN_contrast_recovered' if j<4 else '__tree_classifier_recovered') for j,k in enumerate(MEMBERS)]
        for dimension,value,use in scopes(meta,role):
            n=int(use.sum())
            if not n:continue
            availability_rows.append({'role':role,'dimension':dimension,'slice':value,'n':n,
                'known_synthetic_target_mean':None if label is None else float(label[use].mean()),
                'known_synthetic_target_RMS':None if label is None else float(np.sqrt(np.mean(label[use]**2))),
                'query_true_share_known':False if role=='query' else None})
            for variable,values in data.items():
                v=values[use];model,component_type=kind_map[variable]
                distributions.append({'role':role,'dimension':dimension,'slice':value,'n':n,'model':model,'variable':component_type,
                    'predicted_mean':float(v.mean()),'predicted_RMS':float(np.sqrt(np.mean(v*v))),
                    'population_std':float(v.std()),'min':float(v.min()),'max':float(v.max()),'unique_values':len(np.unique(v)),
                    'component_range_within_roundoff_tolerance':bool('recovered' in component_type and np.ptp(v)<=COMPONENT_TOLERANCE)})
                quantiles.extend({'role':role,'dimension':dimension,'slice':value,'n':n,'model':model,'variable':component_type,
                                  'quantile_p':t,'value':float(np.quantile(v,t))} for t in QUANTILES)
                saturation.extend({'role':role,'dimension':dimension,'slice':value,'n':n,'model':model,'variable':component_type,
                                   'fixed_threshold':t,'fraction_le':float((v<=t).mean()),'fraction_ge':float((v>=t).mean()),
                                   'fraction_exact':float((v==t).mean())} for t in THRESHOLDS)
            for method in ('pearson','spearman'):
                for matrix,left,right in [('share5x5',sharekeys,sharekeys),('component5x5',compkeys,compkeys),('share_to_component5x5',sharekeys,compkeys)]:
                    values=np.empty((5,5))
                    for i,l in enumerate(left):
                        for j,r in enumerate(right):
                            v=corr(data[l][use],data[r][use],method);values[i,j]=np.nan if v is None else v
                            correlations.append({'role':role,'dimension':dimension,'slice':value,'n':n,'matrix':matrix,
                                                 'method':method,'left':l,'right':r,'correlation':v,'defined':v is not None})
                    if dimension=='all':pd.DataFrame(values,index=left,columns=right).to_csv(out/f'{role}_{method}_{matrix}.csv')
                if dimension=='all':
                    allkeys=[k for k in data if not k.startswith('frozen_all5__')]
                    pd.DataFrame([[corr(data[l],data[r],method) for r in allkeys] for l in allkeys],index=allkeys,columns=allkeys).to_csv(out/f'{role}_{method}_all15x15.csv')
    for name,records in [('distribution_summary',distributions),('quantiles',quantiles),('fixed_threshold_saturation',saturation),
                         ('correlations_by_metadata_slice',correlations),('component_recovery_roundoff',recovery),('availability_and_known_target_scope',availability_rows)]:
        pd.DataFrame(records).to_csv(out/(name+'.csv'),index=False)
    pd.crosstab(qmeta.Nobs,qmeta.historical_inferred_type).to_csv(out/'query_Nobs_by_HISTORICAL_INFERRED_TYPE_NOT_GT.csv')
    if any(sha(ROOT/n)!=h for n,h in pins.items()):raise ValueError('Original data/model/CSV/policy input changed')
    result={'status':'passed','plan_sha256':sha(out/'predeclared_plan.json'),'source_script_sha256':sha(__file__),
            'input_sha256':pins,'source_prediction_precision':'existing CPU FP32 selection/query; GPU BF16 q used only for independent formula witness',
            'query_source_SHA_matches_original':True,'query_ID_order_SHA':canonical(ids),'query_prediction_SHA_matches_original_witness':True,
            'CSV_saved_ensemble_exact':True,'original_models_data_policy_CSV_unchanged':True,
            'component_formula_direct_witness':direct,'legacy_inferred_type':legacy_status,
            'query_availability_counts':{'Nobs':{str(k):int(v) for k,v in qmeta.Nobs.value_counts().sort_index().items()},
                'first_completely_missing':int(qmeta.first_completely_missing.sum()),'last_completely_missing':int(qmeta.last_completely_missing.sum()),
                'partial_missing_minutes_total':int(qmeta.partial_missing_minutes.sum())},
            'known_selection_synthetic_label_mean':float(truth.mean()),'known_selection_synthetic_label_RMS':float(np.sqrt(np.mean(truth**2))),
            'query_true_label_mean_RMS':None,'query_actual_site_labels_available':False,
            'new_inference_fit_model_load_or_labels_recovered':False,'public_RMS_matching_or_weights_thresholds_tuned':False,
            'new_site_assignments_computed':False,'unused_report_or_server_or_external_transmission':False,
            'torch_imported':bool('torch' in sys.modules),
            'correlation_and_saturation_are_not_accuracy_or_causal_diagnosis':True,
            'component_exact_0_1_ties_sensitive_to_recovery_roundoff':True,
            'query_stats_not_authorized_for_training_generator_calibration_weights_or_threshold_fitting':True,
            'elapsed_seconds':time.monotonic()-start}
    result['output_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    write(out/'RESULTS.json',result)
    print(json.dumps({'status':'passed','report':str(out/'RESULTS.json'),'sha256':sha(out/'RESULTS.json'),
                      'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
