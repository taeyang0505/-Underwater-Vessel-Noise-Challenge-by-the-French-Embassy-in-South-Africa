"""Explicitly LOCAL DIAGNOSTIC ONLY R6 pilot CSV. Do not upload.

Freeze checkpoint/helpers and output statistics before Test decoding.
No fitting, calibration, model selection, source attribution, future-parameter
changes, official submission, external transmission or remote/GPU operations.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ARM='r6law_relative8_global'
RUN=ROOT/'models/r6_fixed100_v1'
MODEL=RUN/ARM
CONFIG=ROOT/'configs/r6_fixed100_v1.json'
DATA_MANIFEST=ROOT/'data/synthetic/r6_cpu_guardrails_v3_schemafix/manifest.json'
PLAN=ROOT/'reports/R9_R6_PILOT_SUBMISSION_PLAN.json'
REPORT=ROOT/'reports/R9_R6_PILOT_SUBMISSION.json'
OUTPUT=ROOT/'submissions/r6_pilot_r1'
OLD=ROOT/'submissions/r5_fold0_raw_cnn_cpu_pilot_r1/submission.csv'
CHECKPOINT_SHA='677dad7ac36fa0af5b6b9259f7b0d1cbc4d029848e43a8b2b8d263dbc4e5083a'
OLD_SHA='e32b04a2b876a66da6046e1ef1fdae6fd7e768eb9365bde5aa7227ad235c83fe'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def js(path):return json.loads(Path(path).read_text())


def require(value,message):
    if not value:raise ValueError(message)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def plan():
    cfg=js(CONFIG);manifest=js(DATA_MANIFEST)
    files=[CONFIG,DATA_MANIFEST,RUN/'freeze.json',RUN/'all_models_frozen.json',RUN/'results.json',
           MODEL/'summary.json',MODEL/'best_window.pt',MODEL/'checkpoints/epoch_031.pt',
           ROOT/'reports/R6_FIXED100_COMPLETION_VERIFICATION.json',OLD,ROOT/'data/raw/Test.csv',
           ROOT/'data/raw/sample_submission.csv',Path(__file__).resolve()]
    helpers={'pipeline/relative_grid_r6.py','pipeline/relative_grid_r6_v2.py',
             'pipeline/relative_models_r6_v2.py','pipeline/physics_features_r6_v2.py',
             *manifest['freeze']['code_sha256'],*cfg['original_frozen_core17_sha256']}
    files.extend(ROOT/n for n in sorted(helpers))
    output={'format':'r9_r6_local_pilot_preinference_plan_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'local_diagnostic_only':True,'not_to_upload':True,'official_submission':False,'selected_arm':ARM,
        'checkpoint_epoch':31,'checkpoint_sha256':CHECKPOINT_SHA,'input_file_sha256':{str(f.resolve()):sha(f) for f in files},
        'runtime':{'device':'CPU','dtype':'FP32','torch_threads':2,'interop_threads':1,'batch_size':128},
        'transforms':'same frozen relative_grid_v2;rounded0.1dB features+weights;fixed derived presence score;no calibration',
        'statistics_predeclared':{
            'outputs':['R5_share','R5_score','R6_share','R6_score'],
            'moments':['n','mean','population_std_ddof0','RMS','min','max'],
            'paired_R6_minus_R5':['mean','MAE','RMS'],'paired_same_output_correlation':'Pearson',
            'strata':'all;firstminute all25missing;firstminute all25observed;partialfirstminute ifany;shared original querymask forbothmodels',
            'fixed_db_only_features':{'HIGH_bump':'physics_features_r6_v2 column2:3162.3--7943.3Hz max-minus-median minute powerdB',
                                      'SAFE_bump':'column3:1000--5011.9Hz max-minus-median minute powerdB'},
            'feature_associations':'Pearson of each fixed bump with each4modeloutput;only defined region summaries;no feature-basedmodelselection'},
        'hidden_site_StFrancis_statistics':'not provided; Test has no verified site labels and source inference is prohibited',
        'no_Test_statistic_used_for_calibration_model_or_future_R9_parameter_change':True,
        'no_Test_labels_source_identity_matching_or_new_official_submission':True,
        'CSV_path':str(OUTPUT/'submission_r6_pilot.csv'),'fresh_run_only':True}
    write(PLAN,output);print(json.dumps({'plan':str(PLAN),'plan_sha256':sha(PLAN),'inputs':len(output['input_file_sha256'])}))


def pearson(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<2 or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def moments(a):
    a=np.asarray(a,float)
    if not len(a):return {'n':0}
    return {'n':len(a),'mean':float(a.mean()),'population_std_ddof0':float(a.std(ddof=0)),
            'RMS':float(np.sqrt(np.mean(a*a))),'min':float(a.min()),'max':float(a.max())}


def describe(ids,old,new,observed,bumps,defined):
    strata={'all':np.ones(len(ids),bool),'first_minute_all_missing':~observed[:,0].any(1),
            'first_minute_all_observed':observed[:,0].all(1),
            'first_minute_partial':observed[:,0].any(1)&~observed[:,0].all(1)}
    output={}
    for label,mask in strata.items():
        entry={'n':int(mask.sum()),'stratum_is_not_a_site_label':True,'output_summary':{},'R6_minus_R5':{},'fixed_feature_correlations':{}}
        for key in ['share','score']:
            a=old[key][mask];b=new[key][mask];delta=b.astype(float)-a.astype(float)
            entry['output_summary']['R5_'+key]=moments(a);entry['output_summary']['R6_'+key]=moments(b)
            entry['R6_minus_R5'][key]={'n':len(delta),'mean':float(delta.mean()) if len(delta) else None,
                'MAE':float(abs(delta).mean()) if len(delta) else None,'RMS':float(np.sqrt(np.mean(delta*delta))) if len(delta) else None,
                'Pearson_same_output':pearson(a,b)}
        for feature,values in bumps.items():
            use=mask&defined[feature]
            entry['fixed_feature_correlations'][feature]={'n_defined':int(use.sum()),
                **{name+'_'+key:pearson(values[use],p[key][use]) for name,p in [('R5',old),('R6',new)] for key in ['share','score']}}
        output[label]=entry
    return output


def infer(expected):
    require(sha(PLAN)==expected,'Explicit plan SHA mismatch')
    p=js(PLAN);require(not OUTPUT.exists() and not REPORT.exists(),'Fresh output required')
    for n,h in p['input_file_sha256'].items():require(sha(n)==h,'Predeclared input changed: '+n)
    cfg=js(CONFIG);freeze=js(RUN/'freeze.json');gate=js(RUN/'all_models_frozen.json');summary=js(MODEL/'summary.json')
    require(sha(CONFIG)==freeze['config_sha256']==gate['config_sha256'],'Fixed100 config binding changed')
    require(sha(MODEL/'summary.json')==gate['model_summary_sha256'][ARM],'Selected summary changed')
    require(summary['status']=='completed' and summary['epochs_run']==100 and not summary['heldout_evaluated'], 'Completed model required')
    require(summary['arm']['channels']=='relative8' and summary['prior_sha256'] is None,'Expected relative model without prior')
    require(summary['best_inner_window']['epoch']==31 and summary['best_inner_window']['checkpoint']=='best_window.pt','Selected checkpoint rule differs')
    require(sha(MODEL/'best_window.pt')==sha(MODEL/'checkpoints/epoch_031.pt')==CHECKPOINT_SHA==summary['artifact_sha256']['best_window.pt'],'Epoch31 bytes differ')
    proof=js(ROOT/'reports/R6_FIXED100_COMPLETION_VERIFICATION.json')
    require(proof['status']=='passed' and proof['config_sha256']==sha(CONFIG),'Independent completion proof differs')
    manifest=js(DATA_MANIFEST);require(sha(DATA_MANIFEST)==cfg['data_manifest_sha256'],'Training manifest changed')
    for mapping in [manifest['freeze']['code_sha256'],cfg['original_frozen_core17_sha256'],freeze['new_code_sha256']]:
        for n,h in mapping.items():require(sha(ROOT/n)==h,'Frozen producer/preprocessing code changed: '+n)
    import torch
    from pipeline.relative_grid_r6_v2 import relative_grid_v2
    from pipeline.relative_models_r6_v2 import TinyMaskCNNv2,fixed_presence_score
    from pipeline.physics_features_r6_v2 import physics_features_r6_v2
    from pipeline.inference_r5 import read_query_csv
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    checkpoint=torch.load(MODEL/'best_window.pt',map_location='cpu',weights_only=True)
    require(checkpoint['epoch']==31 and checkpoint['model_config']==summary['model_config'],'Checkpoint model metadata differs')
    model=TinyMaskCNNv2(**summary['model_config']).eval();model.load_state_dict(checkpoint['state_dict'],strict=True)
    state_before={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
    # No model or policy choice remains at the query boundary.
    query_open_utc=datetime.now(timezone.utc).isoformat();ids,db,query=read_query_csv(ROOT/'data/raw/Test.csv',expected_rows=1500)
    with (ROOT/'data/raw/sample_submission.csv').open(newline='') as f:sample=list(csv.reader(f))
    require(sample[0]==['ID','share_added_vessel','score_added_vessel'] and [r[0] for r in sample[1:]]==ids,'Sample ID/order mismatch')
    started=time.perf_counter();parts={k:[] for k in ['share','score','cell_mask']};pool_error=score_error=0.
    with torch.inference_mode():
        for begin in range(0,len(db),128):
            x,w=relative_grid_v2(db[begin:begin+128]);wt=torch.from_numpy(w)
            share,logit,q=model(torch.from_numpy(x),wt);score=logit.sigmoid()
            exact_score,pooled,_=fixed_presence_score(q,wt)
            pool_error=max(pool_error,float((share-pooled).abs().max()))
            pool_error=max(pool_error,float(abs((w.astype(float)*q.numpy()).sum((1,2))-share.numpy()).max()))
            score_error=max(score_error,float((score-exact_score.clamp(1e-6,1-1e-6)).abs().max()))
            for name,v in [('share',share),('score',score),('cell_mask',q)]:parts[name].append(v.numpy())
    new={k:np.concatenate(v) for k,v in parts.items()};prediction_seconds=time.perf_counter()-started
    require(pool_error<=3e-6 and score_error<=2e-7,'Pooling/fixed-score check failed')
    for key,value in new.items():require(np.isfinite(value).all() and ((value>=0)&(value<=1)).all(),'Invalid predictions: '+key)
    require(sha(OLD)==OLD_SHA,'Original R5 handoff CSV differs')
    with OLD.open(newline='') as f:old_rows=list(csv.reader(f))
    require(old_rows[0]==sample[0] and len(old_rows)==1501,'Old CSV shape differs')
    keyed={r[0]:(float(r[1]),float(r[2])) for r in old_rows[1:]}
    require(len(keyed)==1500 and set(keyed)==set(ids),'Original R5 ID set differs')
    old={'share':np.asarray([keyed[i][0] for i in ids]),'score':np.asarray([keyed[i][1] for i in ids])}
    features,diag=physics_features_r6_v2(db,return_diagnostics=True)
    bumps={'HIGH_bump':features[:,2],'SAFE_bump':features[:,3]}
    defined={'HIGH_bump':diag['defined'][:,2],'SAFE_bump':diag['defined'][:,3]}
    descriptions=describe(ids,old,new,np.isfinite(db),bumps,defined)
    # Description is not allowed to modify any values/model/settings.
    for n,h in p['input_file_sha256'].items():require(sha(n)==h,'Input changed during inference/statistics: '+n)
    require(state_before=={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()},'Model state changed')
    OUTPUT.mkdir(parents=True);csv_path=OUTPUT/'submission_r6_pilot.csv'
    with csv_path.open('x',newline='') as f:
        w=csv.writer(f);w.writerow(sample[0]);w.writerows(zip(ids,new['share'].astype(float).tolist(),new['score'].astype(float).tolist()))
    with csv_path.open(newline='') as f:check=list(csv.reader(f))
    values=np.asarray([[float(x) for x in r[1:]] for r in check[1:]])
    require(check[0]==sample[0] and len(check)==1501 and [r[0] for r in check[1:]]==ids,'Written CSV ID/shape mismatch')
    require(np.array_equal(values[:,0],new['share'].astype(float)) and np.array_equal(values[:,1],new['score'].astype(float)),'CSV float roundtrip differs')
    np.savez_compressed(OUTPUT/'raw_predictions.npz',sample_id=np.asarray(ids),**new)
    receipt={'format':'r9_r6_local_diagnostic_pilot_v1','status':'completed','local_diagnostic_only':True,'not_to_upload':True,'official_submission':False,
       'created_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':expected,'checkpoint_sha256':CHECKPOINT_SHA,'checkpoint_epoch':31,
       'arm':ARM,'model_config':summary['model_config'],'selected_model_summary_sha256':sha(MODEL/'summary.json'),
       'fixed100_config_sha256':sha(CONFIG),'input_file_sha256':p['input_file_sha256'],
       'query_schema':query,'query_open_after_plan_and_model_freeze_utc':query_open_utc,
       'rows':1500,'columns':sample[0],'ID_order_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
       'physical_pool_max_absolute_error':pool_error,'fixed_score_max_absolute_error':score_error,'prediction_seconds':prediction_seconds,
       'CSV_sha256':sha(csv_path),'raw_prediction_sha256':sha(OUTPUT/'raw_predictions.npz'),'CSV_roundtrip_exact':True,
       'old_R5_CSV_sha256':OLD_SHA,'statistics':descriptions,
       'StFrancis_or_other_hidden_site_statistics':'not provided; no ground-truth Test site IDs and no hidden source inference',
       'observation_strata_do_not_identify_sites':True,'no_calibration_postprocessing_or_new_model_selection':True,
       'statistics_do_not_drive_future_R9_parameters':True,'model_input_and_policy_unchanged':True,
       'device':'CPU FP32','torch_threads':2,'batch_size':128,'no_training_OUTER_cross_GPU_SSH_external_transmission':True,
       'limitations':['R5 and R6 scores are different uncalibrated diagnostic functions. Correlation is descriptive, not calibration or causality.',
                      'Statistics have no hidden target labels and cannot estimate official performance.',
                      'Original R5 e32b hash identifies local handoff bytes; actual uploaded server bytes remain unverified.']}
    write(OUTPUT/'inference_receipt.json',receipt);write(REPORT,receipt)
    (OUTPUT/'DO_NOT_UPLOAD.txt').write_text('LOCAL DIAGNOSTIC ONLY. User explicitly requested no official upload. No submission was performed.\n')
    allstats=descriptions['all'];lines=['# R9: R6 local diagnostic pilot — 업로드 금지','',
        '**실제 공식 제출을 하지 않았습니다. 사용자 요청에 따라 local diagnostic만 생성했으며 이 파일을 업로드하지 않습니다.**','',
        '모델은 고정100epoch R6 relative8/global의 사전 지정된 best-window epoch31입니다. 보정·새 모델 선택·분포 맞춤은 없습니다.',
        '',f'CSV: `submissions/r6_pilot_r1/submission_r6_pilot.csv`, SHA `{receipt["CSV_sha256"]}`.',
        f'CPU 추론 {prediction_seconds:.3f}초, 1,500행×3열, ID·범위·pooling·CSV roundtrip 검증 통과.','',
        '| 출력 | 평균 | 모표준편차 | RMS | 최소 | 최대 |','|---|---:|---:|---:|---:|---:|']
    for name,v in allstats['output_summary'].items():lines.append(f'| {name} | {v["mean"]:.6f} | {v["population_std_ddof0"]:.6f} | {v["RMS"]:.6f} | {v["min"]:.6f} | {v["max"]:.6f} |')
    lines+=['','| R6−R5 | 평균차 | MAE | RMS 차이 | 동일 출력 Pearson |','|---|---:|---:|---:|---:|']
    for name,v in allstats['R6_minus_R5'].items():lines.append(f'| {name} | {v["mean"]:.6f} | {v["MAE"]:.6f} | {v["RMS"]:.6f} | {v["Pearson_same_output"]:.6f} |')
    lines+=['','고정 HIGH(3162–7943Hz)·SAFE(1000–5012Hz) bump와 각 예측의 Pearson, 동일 관측 mask의 첫분 missing/observed strata는 JSON에 보존했습니다. '
        '전부 출력 설명용이며 정답·원인·보정 근거로 쓰지 않습니다. **St Francis 유사 집단의 수치는 제공하지 않습니다.** Test의 실제 site 정답이 없고, 결측 패턴은 site label이 아닙니다.',
        '', 'R5의 자유 presence head와 R6의 고정 share+q-contrast score는 서로 다른 비보정 진단값입니다. 단순 평균·상관 차이가 성능 우열을 뜻하지 않습니다. '
        '이 통계로 현재/향후 R9 생성기·SNR·EQ·sampling·모델·후처리를 바꾸지 않습니다. 원 R5 파일·R6 모델·데이터는 변경하지 않았습니다.',
        '', '[추론 전 동결 계획](R9_R6_PILOT_SUBMISSION_PLAN.json) · [전체 수치·SHA·receipt](R9_R6_PILOT_SUBMISSION.json)']
    (ROOT/'reports/R9_R6_PILOT_SUBMISSION.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'saved':str(csv_path),'local_diagnostic_only':True,'not_to_upload':True,'official_submission':False,
                      'CSV_sha256':receipt['CSV_sha256'],'seconds':prediction_seconds,'overall':allstats}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','infer']);p.add_argument('--expected-plan-sha')
    a=p.parse_args()
    if a.action=='plan':plan()
    else:
        if not a.expected_plan_sha:p.error('Explicit predeclared plan SHA required')
        infer(a.expected_plan_sha)


if __name__=='__main__':main()
