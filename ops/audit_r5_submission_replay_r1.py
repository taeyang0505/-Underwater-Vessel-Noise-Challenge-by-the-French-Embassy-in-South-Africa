"""R7 read-only R5 submitted-path audit: predeclare hashes, then CPU replay.

Test use is inference-only and ID-keyed comparison with archived predictions.
No query distribution summaries, fitting, calibration selection, recovery of
labels, model update, external I/O, or new official submission is performed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PREFIX=ROOT/'reports/R7_R5_SUBMISSION_PATH_AUDIT'
PLAN=Path(str(PREFIX)+'_PLAN.json')
RESULT=Path(str(PREFIX)+'_RESULTS.json')
EVIDENCE=Path(str(PREFIX)+'_evidence')
BACKUP=ROOT/'ops/server_backups/r5_gpu_r1'
RESTORED=BACKUP/'restored_first_cnn_v1'
FOLD=RESTORED/'fold_00'
MODEL=FOLD/'preprocessing_pilot/raw/bg_cnn'
SUBMISSION=ROOT/'submissions/r5_fold0_raw_cnn_cpu_pilot_r1'
CHECKPOINT_SHA='aa222697275306cf93e5778a707f7ee5fd75610ae0a0bf643d654854ee96d46f'
CSV_SHA='e32b04a2b876a66da6046e1ef1fdae6fd7e768eb9365bde5aa7227ad235c83fe'
ARCHIVE=BACKUP/'r5_snapshot_20261025T161450_550440Z_8aeccd97.zip'
ARCHIVE_SHA='12ec9d8bfb176a19c53891a3578ab32ffdacb4c1d99633a3f11b8a94c5a2fe2a'
BANDS=('31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5','251.2','316.2','398.1',
       '501.2','631','794.3','1000','1258.9','1584.9','1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def require(value,message):
    if not value:raise ValueError(message)


def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def plan():
    context=json.loads((FOLD/'context.json').read_text());data=ROOT/context['data_fold']
    inputs=[FOLD/'context.json',FOLD/'shared_background_prior.json',ARCHIVE,
            *[MODEL/n for n in ['best.pt','summary.json','epochs.jsonl','dynamic_source_receipts.json','inner_predictions.npz']],
            *[SUBMISSION/n for n in ['submission.csv','inference_receipt.json','raw_test_predictions.npz','frozen_postprocessing_policy.json']],
            ROOT/'reports/R5_CNN_MAC_INFERENCE_PROFILE.json',ROOT/'reports/R5_CNN_MAC_INFERENCE_PROFILE/inner_cpu_predictions.npz',
            ROOT/'reports/R5_CNN_PILOT_CALIBRATION/specification.json',
            ROOT/'data/raw/Test.csv',ROOT/'data/raw/sample_submission.csv',
            *[data/n for n in ['readiness_manifest.json','sources/manifest.json','train.npz','validation.npz','inner_validation_metadata.parquet']],
            ROOT/'ops/build_r5_cnn_pilot_submission.py',ROOT/'ops/profile_r5_cnn_mac_inference.py',ROOT/'ops/calibrate_r5_single_cnn.py',
            Path(__file__).resolve(),*[ROOT/n for n in context['code_sha256']]]
    for optional in [ROOT/'delivery/r5_cnn_pilot_repro_r1/reproduced_submission/submission.csv',
                     Path('/Users/hongtaeyang/Desktop/French_Embassy_Underwater_Vessel_Noise/SUBMIT_NOW/submission.csv')]:
        if optional.exists():inputs.append(optional)
    hashes={str(p.resolve()):sha(p) for p in inputs}
    p={'format':'r7_r5_submitted_path_audit_plan_v1','created_utc':datetime.now(timezone.utc).isoformat(),
       'scope':'R5 Fold0 raw CNN epoch19 replay only; one completed model, not fivefold ensemble',
       'input_file_sha256':hashes,'original_core17_expected_sha256':context['code_sha256'],
       'checkpoint_sha256':CHECKPOINT_SHA,'local_handoff_csv_sha256':CSV_SHA,
       'training_binding':context['training_binding'],'data_fold':context['data_fold'],
       'inference':{'device':'CPU','precision':'FP32','torch_threads':2,'interop_threads':1,'batch_size':128},
       'replay_tolerance':{'absolute':1e-5,'relative':0.,'scope':'CPU2 versus archivedCPU4 share,score,cell_mask; exact equality reported separately'},
       'tests':['archive+restored entry identity','checkpoint/prior/context/code binding','375 minute-major mapping + ID/mask/rejection numeric cases',
                'raw TRAIN preparation versus query inputs on numeric arrays','physical linear-energy weights and pooling',
                'full INNER3000 CPU2 replay + original CPU4/GPU archives + labels-only metrics',
                'archived Test raw-to-CSV exact byte reconstruction','Test1500 inference-only replay; ID-keyed differences only'],
       'explicit_test_inference_authorized':True,'no_Test_cohort_statistics_fitting_tuning_or_label_recovery':True,
       'no_weight_updates_or_policy_selection':True,'no_OUTER_old_cross_GPU_SSH_or_external_transmission':True,
       'actual_uploaded_Zindi_CSV_bytes_independently_verified':False,
       'outputs':[str(RESULT),str(PREFIX)+'.md',str(EVIDENCE)],'fresh_execution_no_resume':True}
    write_new(PLAN,p);print(json.dumps({'plan':str(PLAN),'plan_sha256':sha(PLAN),'files':len(hashes)}))


def metrics(y,z,p):
    from sklearn.metrics import roc_auc_score
    e=p['share'].astype(float)-y.astype(float)
    return {'n':len(y),'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(abs(e).mean()),'bias_pred_minus_truth':float(e.mean()),
            'addition_presence_AUC':float(roc_auc_score(z,p['score']))}


def comparison(current,archived,atol):
    out={}
    for k in ['share','score','cell_mask']:
        a=np.asarray(current[k]);b=np.asarray(archived[k]);require(a.shape==b.shape,'Replay shape mismatch: '+k)
        d=a.astype(float)-b.astype(float)
        out[k]={'shape':list(a.shape),'bitwise_equal':bool(np.array_equal(a,b)),
                'max_absolute_difference':float(abs(d).max()),'difference_vector_RMSE':float(np.sqrt(np.mean(d*d))),
                'different_element_count':int(np.count_nonzero(a!=b)),'within_predeclared_atol':bool(abs(d).max()<=atol)}
    return out


def csv_bytes(ids,share,score):
    stream=io.StringIO(newline='');writer=csv.writer(stream)
    writer.writerow(['ID','share_added_vessel','score_added_vessel'])
    writer.writerows(zip(ids,np.asarray(share,dtype=float).tolist(),np.asarray(score,dtype=float).tolist()))
    return stream.getvalue().encode()


def replay(expected_plan):
    require(sha(PLAN)==expected_plan,'Explicit predeclared plan SHA differs')
    p=json.loads(PLAN.read_text());require(not RESULT.exists() and not EVIDENCE.exists(),'Fresh audit outputs required')
    for path,h in p['input_file_sha256'].items():require(sha(path)==h,'Frozen input differs from plan: '+path)
    EVIDENCE.mkdir();started=time.perf_counter()
    import torch
    import pandas as pd
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42)
    from pipeline.inference_r5 import read_query_csv,original_energy_weights,FEATURE_COLUMNS
    from pipeline.grid_inputs_r5 import load_shared_prior,shared_grid_inputs
    from pipeline.fold_data_r5 import FoldDataR5
    from pipeline.neural_r5 import make_neural,restore_into,predict_neural,grid_inputs
    from pipeline.observation_augmentation_r5 import augment_training_observations,EndpointDropoutPolicy
    from ops.calibrate_r5_single_cnn import validate_frozen_policy,apply_frozen_policy
    # File reads are allowlisted. Source train.npz is hashable but never decoded.
    data_root=(ROOT/'data').resolve();allowed={Path(q).resolve() for q in p['input_file_sha256']}
    def audit(event,args):
        if event in ['socket.connect','socket.getaddrinfo','subprocess.Popen','os.system']:
            raise PermissionError('No external activity in replay audit')
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            q=Path(os.fsdecode(args[0])).resolve()
            if data_root==q or data_root in q.parents:
                if q not in allowed:raise PermissionError('Undeclared source/query access: '+str(q))
                if args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):raise PermissionError('No source writes')
            if q.suffix.lower() in ['.wav','.flac','.mp3']:raise PermissionError('No waveform access')
    sys.addaudithook(audit)
    original_np_load=np.load;loaded=[]
    def guarded_np_load(path,*args,**kwargs):
        if isinstance(path,(str,os.PathLike)):
            q=Path(path).resolve();require(q.name!='train.npz','TRAIN values must not be decoded in replay audit')
            loaded.append(str(q))
        return original_np_load(path,*args,**kwargs)
    np.load=guarded_np_load
    require(sha(MODEL/'best.pt')==CHECKPOINT_SHA and sha(ARCHIVE)==ARCHIVE_SHA,'Original model/archive differs')
    with zipfile.ZipFile(ARCHIVE) as z:
        require(z.testzip() is None,'Original archive CRC failure');archive_manifest=json.loads(z.read('__snapshot_manifest.json'))
        for item in archive_manifest['files']:
            payload=z.read(item['archive_path'])
            require(hashlib.sha256(payload).hexdigest()==item['sha256'] and len(payload)==item['bytes']
                    and sha(RESTORED/item['relative_path'])==item['sha256'],'Restored backup entry differs')
    context=json.loads((FOLD/'context.json').read_text());summary=json.loads((MODEL/'summary.json').read_text())
    require(canonical(context['config'])==context['config_sha256'],'Context config identity failure')
    require(len(context['code_sha256'])==17,'Original core17 closure missing')
    for name,h in context['code_sha256'].items():require(sha(ROOT/name)==h,'Original code changed: '+name)
    require(summary['status']=='completed' and summary['fixture_only'] is False and summary['best_epoch']==19,'Wrong completion scope')
    fold=FoldDataR5(ROOT/context['data_fold'],42);require(fold.binding==context['training_binding']==summary['training_binding'],'TRAIN binding differs')
    record=context['prior'];prior=load_shared_prior(FOLD/record['filename'],expected_binding=fold.binding,
        expected_file_sha256=record['file_sha256'],expected_prior_sha256=record['payload_sha256'])
    policy=json.loads((SUBMISSION/'frozen_postprocessing_policy.json').read_text());validate_frozen_policy(policy)
    observation=summary['observation_policy'];require(observation=={'quantization_decimals':None,'endpoint_dropout_probability':0.},'Raw policy changed')
    checkpoint=torch.load(MODEL/'best.pt',map_location='cpu',weights_only=True)
    model,factory=make_neural('bg_cnn',context['config']['model_settings']['bg_cnn'])
    restore_into(model,checkpoint,kind='bg_cnn',binding=fold.binding,prior_sha=record['payload_sha256'],policy=observation)
    model=model.to(device='cpu',dtype=torch.float32).eval();before=state_hash(model)
    source_receipts=json.loads((MODEL/'dynamic_source_receipts.json').read_text())
    er=source_receipts['completed_epochs'][str(checkpoint['epoch'])]
    require(checkpoint['receipt']==summary['receipt'] and checkpoint['training_settings']==summary['settings'],'Training receipt/settings differ')
    require(checkpoint['dynamic_source_identity']==summary['dynamic_source_identity']==source_receipts['source_identity']
            and checkpoint['best_epoch_source_receipt']==er and er['clean_primary_once'],'Dynamic epoch provenance mismatch')
    # Pure numeric schema cases. The literal ordinal mapping is independent of Test values.
    canonical_columns=tuple(f'm{m:02d}_TOL_{f}' for m in range(1,16) for f in BANDS)
    require(tuple(FEATURE_COLUMNS)==canonical_columns,'375 canonical feature mapping differs')
    unit=(80.+np.arange(3)[:,None,None]*2+np.arange(15)[None,:,None]*.5+np.arange(25)[None,None,:]*.01).astype('float32')
    unit[0,3]=np.nan;unit[1,0:4]=np.nan;unit_ids=['audit_numeric_A','audit_numeric_B','audit_numeric_C']
    mapping=EVIDENCE/'numeric_mapping.csv'
    with mapping.open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['ID',*canonical_columns]);writer.writerows([[i,*[float(v) if np.isfinite(v) else '' for v in a.ravel()]] for i,a in zip(unit_ids,unit)])
    ids,parsed,_=read_query_csv(mapping,expected_rows=3)
    require(ids==unit_ids and np.array_equal(parsed,unit,equal_nan=True),'Minute/band or ID/mask roundtrip failed')
    malformed=EVIDENCE/'numeric_bad_header.csv';lines=list(csv.reader(mapping.open()));lines[0][1],lines[0][2]=lines[0][2],lines[0][1]
    with malformed.open('x',newline='') as stream:csv.writer(stream).writerows(lines)
    rejected=False
    try:read_query_csv(malformed,expected_rows=3)
    except ValueError:rejected=True
    require(rejected,'Wrong column order must fail closed')
    mask=np.isfinite(unit);peak=np.nanmax(unit.astype(float),(1,2),keepdims=True)
    e=np.where(mask,10**((unit.astype(float)-peak)/10),0);oracle_w=(e/e.sum((1,2),keepdims=True)).astype('float32')
    require(np.array_equal(original_energy_weights(unit),oracle_w),'Independent energy oracle differs')
    q=np.where(mask,.27,0).astype('float32')
    changed,_=augment_training_observations(unit,q,epoch=1,sample_indices=np.arange(3),seed=42,
        policy=EndpointDropoutPolicy(probability=0.),split='train',quantization_decimals=None)
    train_grid,_=shared_grid_inputs(changed['feature_db'],prior,expected_binding=fold.binding)
    query_grid,query_w=grid_inputs(unit,prior,fold.binding,observation)
    require(np.array_equal(train_grid,query_grid) and np.array_equal(changed['energy_weights'],query_w),'Raw TRAIN/query transforms differ')
    direct=predict_neural(model,unit,prior,fold.binding,observation,device='cpu',batch_size=128)
    reparsed=predict_neural(model,parsed,prior,fold.binding,observation,device='cpu',batch_size=128)
    require(all(np.array_equal(direct[k],reparsed[k]) for k in direct),'Parser/direct model path differs')
    unit_pool=float(abs((oracle_w.astype(float)*direct['cell_mask']).sum((1,2))-direct['share']).max())
    require(unit_pool<=3e-6,'Numeric masked physical pool mismatch')
    # INNER labels are used only after prediction, solely for diagnostic scoring.
    inner,metadata=fold.load('inner');inner_ids=metadata.sample_id.astype(str).to_numpy();t=time.perf_counter()
    inner_pred=predict_neural(model,inner['db'],prior,fold.binding,observation,device='cpu',batch_size=128)
    inner_seconds=time.perf_counter()-t
    with np.load(ROOT/'reports/R5_CNN_MAC_INFERENCE_PROFILE/inner_cpu_predictions.npz',allow_pickle=False) as z:
        old_cpu={k:z[k] for k in z.files}
    require(np.array_equal(old_cpu['sample_id'],inner_ids),'Archived CPU INNER ID order differs')
    with np.load(MODEL/'inner_predictions.npz',allow_pickle=False) as z:old_gpu={k:z[k] for k in z.files}
    atol=p['replay_tolerance']['absolute'];inner_cpu=comparison(inner_pred,old_cpu,atol);inner_gpu=comparison(inner_pred,old_gpu,atol)
    inner_metrics=metrics(inner['share'],inner['presence'],inner_pred)
    old_cpu_metrics=metrics(inner['share'],inner['presence'],old_cpu);old_gpu_metrics=metrics(inner['share'],inner['presence'],old_gpu)
    inner_w=original_energy_weights(inner['db']);inner_pool=float(abs((inner_w.astype(float)*inner_pred['cell_mask']).sum((1,2))-inner_pred['share']).max())
    target_error=float(abs((inner_w.astype(float)*inner['cell_share']).sum((1,2))-inner['share']).max())
    np.savez_compressed(EVIDENCE/'inner_cpu2_replay.npz',sample_id=inner_ids,**inner_pred)
    # The only Test operations below are fixed parsing, inference and archived-ID comparison.
    test_open_utc=datetime.now(timezone.utc).isoformat();test_ids,test_db,query_receipt=read_query_csv(ROOT/'data/raw/Test.csv',expected_rows=1500)
    with (ROOT/'data/raw/sample_submission.csv').open(newline='') as stream:sample=list(csv.reader(stream))
    with (SUBMISSION/'submission.csv').open(newline='') as stream:submitted=list(csv.reader(stream))
    require(sample[0]==submitted[0]==['ID','share_added_vessel','score_added_vessel'],'Submission header differs')
    require([r[0] for r in sample[1:]]==test_ids==[r[0] for r in submitted[1:]] and len(set(test_ids))==1500,'Test/sample/submission ID alignment differs')
    with np.load(SUBMISSION/'raw_test_predictions.npz',allow_pickle=False) as z:old_test={k:z[k] for k in z.files}
    require(np.array_equal(np.asarray(test_ids),old_test['sample_id']),'Archived raw Test ID order differs')
    archived_share=apply_frozen_policy(policy,old_test['share']);stored_csv=csv_bytes(test_ids,archived_share,old_test['score'])
    require(hashlib.sha256(stored_csv).hexdigest()==CSV_SHA==sha(SUBMISSION/'submission.csv'),'Archived raw-to-CSV bytes do not reproduce original')
    local_handoff={q:h for q,h in p['input_file_sha256'].items() if q.endswith('/submission.csv')}
    require(all(h==CSV_SHA for h in local_handoff.values()),'Local handoff CSV bytes differ')
    t=time.perf_counter();test_pred=predict_neural(model,test_db,prior,fold.binding,observation,device='cpu',batch_size=128);test_seconds=time.perf_counter()-t
    test_diff=comparison(test_pred,old_test,atol)
    test_w=original_energy_weights(test_db)
    test_pool=float(abs((test_w.astype(float)*test_pred['cell_mask']).sum((1,2))-test_pred['share']).max())
    require(test_pool<=3e-6 and inner_pool<=3e-6 and target_error<=3e-6,'Physical pooling/INNER target identity failure')
    adjusted=apply_frozen_policy(policy,test_pred['share']);fresh_csv=csv_bytes(test_ids,adjusted,test_pred['score'])
    csv_values=np.asarray([[float(r[1]),float(r[2])] for r in submitted[1:]])
    replay_csv_diff={'share_max_absolute':float(abs(adjusted-csv_values[:,0]).max()),
                     'score_max_absolute':float(abs(test_pred['score'].astype(float)-csv_values[:,1]).max()),
                     'recomputed_CPU2_CSV_would_be_byte_identical':bool(fresh_csv==stored_csv),
                     'CPU2_in_memory_CSV_sha256':hashlib.sha256(fresh_csv).hexdigest(),'no_new_submission_CSV_written':True}
    np.savez_compressed(EVIDENCE/'test_cpu2_inference_replay.npz',sample_id=np.asarray(test_ids),**test_pred)
    require(before==state_hash(model),'Model state changed during audit')
    for path,h in p['input_file_sha256'].items():require(sha(path)==h,'Input changed during audit: '+path)
    mismatch=any(not v['within_predeclared_atol'] for d in [inner_cpu,test_diff] for v in d.values())
    r={'format':'r7_r5_submitted_path_audit_result_v1','status':'replay_mismatch_requires_review' if mismatch else 'passed',
       'created_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':expected_plan,'input_file_sha256':p['input_file_sha256'],
       'checkpoint_sha256':CHECKPOINT_SHA,'checkpoint_epoch':19,'core17_verified':17,'archive_entries_verified':len(archive_manifest['files']),
       'factory_receipt':factory,'TRAIN_prior_loaded_not_refit':True,'model_state_unchanged':True,'model_state_sha256':before,
       '375_minute_major_band_mapping_ID_mask_roundtrip_bitwise':True,'wrong_column_order_rejected':True,
       'raw_TRAIN_query_feature_and_weight_equivalence_bitwise':True,'numeric_parser_model_predictions_bitwise':True,
       'energy_weight_oracle_bitwise':True,'physical_pooling_max_abs':{'numeric':unit_pool,'INNER':inner_pool,'Test_inference_only':test_pool},
       'INNER_known_target_pooling_max_abs':target_error,'INNER_CPU2_metrics':inner_metrics,
       'archived_INNER_CPU4_metrics':old_cpu_metrics,'archived_INNER_GPU_BF16_metrics':old_gpu_metrics,
       'INNER_CPU2_vs_archived_CPU4':inner_cpu,'INNER_CPU2_vs_archived_GPU_BF16':inner_gpu,
       'INNER_id_order_sha256':canonical(inner_ids.tolist()),'Test_id_order_sha256':canonical(test_ids),
       'Test_query_schema_receipt':query_receipt,'Test_CPU2_vs_archived_CPU4_ID_keyed_numeric_replay_only':test_diff,
       'archived_Test_raw_to_submission_CSV_byte_reproduction':True,'local_handoff_csv_sha256':CSV_SHA,
       'local_handoff_copies':local_handoff,'actual_uploaded_Zindi_bytes_SHA_verified':False,
       'frozen_calibration':{'candidate':policy['candidate'],'coefficients':policy['coefficients'],'policy_file_sha256':sha(SUBMISSION/'frozen_postprocessing_policy.json'),
                             'reselected_or_refit':False},'CPU2_vs_original_CSV':replay_csv_diff,
       'precision':'CPU FP32','torch_threads':2,'archived_CPU_threads':4,'replay_atol':atol,
       'timing_seconds':{'INNER3000':inner_seconds,'Test1500_inference_only':test_seconds,'total':time.perf_counter()-started},
       'loaded_np_arrays':loaded,'opened_FoldData_roles':fold.opened_roles,
       'Test_opened_after_plan_and_checkpoint_policy_freeze_utc':test_open_utc,
       'no_Test_distribution_analysis_fit_tuning_source_matching_or_label_recovery':True,
       'no_training_OUTER_old_cross_GPU_SSH_external_or_official_submission':True,
       'limitations':['INNER already used for original epoch selection; not independent neural OOF.',
                      'Local handoff SHA does not prove actual uploaded Zindi bytes.',
                      'CPU2/CPU4 numeric equality is distinct from archived raw-to-CSV exact byte reproduction.',
                      'Implementation replay cannot prove hidden generator similarity or identify the cause of the official generalization gap.'],
       'evidence_sha256':{v.name:sha(v) for v in EVIDENCE.iterdir() if v.is_file()}}
    write_new(RESULT,r)
    md=Path(str(PREFIX)+'.md')
    md.write_text('# R7: 기존 R5 제출 경로 재현 감사\n\n'
        f'결과: **{r["status"]}**. 모델·입력 SHA 계획을 추론 전에 저장했고, 원 epoch19 CNN과 TRAIN prior·정책을 그대로 사용했습니다. '
        '원 CPU4와 이번 CPU2의 차이는 아래 수치 재현 검사로 분리했습니다.\n\n'
        '| 검사 | 결과 |\n|---|---|\n'
        '| 375열 minute-major/주파수/ID/missing mask 숫자 roundtrip | bitwise 일치; 잘못된 헤더 거부 |\n'
        '| raw TRAIN 준비와 query 특징·가중치 | bitwise 일치 |\n'
        '| 동일 배열 vs CSV parser 경유 모델 예측 | bitwise 일치 |\n'
        '| 저장 Test raw→동결 정책→원 CSV 바이트 | SHA e32b04a2…c83fe 정확 재현 |\n'
        f'| INNER CPU2 재현 RMSE | {inner_metrics["RMSE"]:.12f} |\n'
        f'| 저장 CPU4 / GPU BF16 INNER RMSE | {old_cpu_metrics["RMSE"]:.12f} / {old_gpu_metrics["RMSE"]:.12f} |\n'
        f'| INNER CPU2−CPU4 share 최대차 | {inner_cpu["share"]["max_absolute_difference"]:.12g} |\n'
        f'| Test CPU2−CPU4 share 최대차 (추론 재현 차이만) | {test_diff["share"]["max_absolute_difference"]:.12g} |\n'
        f'| 새 CPU2 추론의 원 CSV 바이트 동일 여부 | {replay_csv_diff["recomputed_CPU2_CSV_would_be_byte_identical"]} |\n\n'
        '입력/가중치/정책은 전후 SHA가 같았고 학습·보정 재선택·새 제출은 하지 않았습니다. '
        'Test는 고정 추론과 기존 예측의 ID별 재현 차이 확인만 사용했으며 분포·원천 추정이나 정답 복원은 하지 않았습니다. '
        '재추론 결과는 audit NPZ로만 저장하고 새 submission CSV는 쓰지 않았습니다.\n\n'
        '이 결과는 구현의 특정 경로 재현 증거입니다. 숨은 생성기 정합·독립 검증의 대표성·공식 성능 격차의 원인을 '
        '확정하거나 외부 분포 오류만이 원인이라고 증명하지는 않습니다. INNER는 원 CNN epoch 선택에 이미 사용됐습니다. '
        'e32b…는 로컬 handoff CSV 해시이며 실제 Zindi 업로드 바이트의 SHA는 독립 확인되지 않았습니다.\n\n'
        '[추론 전 동결 계획](R7_R5_SUBMISSION_PATH_AUDIT_PLAN.json) · '
        '[상세 수치·SHA 결과](R7_R5_SUBMISSION_PATH_AUDIT_RESULTS.json)\n')
    print(json.dumps({'status':r['status'],'results':str(RESULT),'results_sha256':sha(RESULT),
                      'INNER_RMSE':inner_metrics['RMSE'],'Test_replay_max_share_difference':test_diff['share']['max_absolute_difference'],
                      'archived_CSV_byte_reproduction':True,'seconds':r['timing_seconds']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','replay'])
    parser.add_argument('--expected-plan-sha');args=parser.parse_args()
    if args.action=='plan':plan()
    else:
        if not args.expected_plan_sha:parser.error('Explicit --expected-plan-sha required')
        replay(args.expected_plan_sha)


if __name__=='__main__':main()
