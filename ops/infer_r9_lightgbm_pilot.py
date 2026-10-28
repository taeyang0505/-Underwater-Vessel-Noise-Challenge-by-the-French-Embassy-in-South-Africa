"""Local R9 five-fold LightGBM-only pilot CSV, with no fitting or upload.

All completed model/context/code hashes and the equal-fold policy are frozen
before opening Test. Only per-window db enters PhysicsLGBM.predict. This is
not the complete R9 five-member pipeline, OOF evaluation, or a submission.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

from pipeline.prepare_sa import COMMON
from pipeline.trees_ensemble_r9 import PhysicsLGBM, canonical_sha, array_sha

ROOT=Path(__file__).resolve().parents[1]
COLS=['ID','share_added_vessel','score_added_vessel']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def regular_hashes(folder):
    result={}
    for p in sorted(Path(folder).rglob('*')):
        if p.is_symlink():raise ValueError('Symlink model artifacts are forbidden')
        if p.is_file():result[p.relative_to(folder).as_posix()]=sha(p)
    return result


def check_map(base,mapping):
    for name,digest in mapping.items():
        rel=Path(name);p=Path(base)/rel
        if rel.is_absolute() or '..' in rel.parts or p.is_symlink() or not p.is_file() or sha(p)!=digest:
            raise ValueError('Immutable artifact changed: '+name)


def omp_images():
    d=ctypes.CDLL(None);d._dyld_image_count.restype=ctypes.c_uint32
    d._dyld_get_image_name.argtypes=[ctypes.c_uint32];d._dyld_get_image_name.restype=ctypes.c_char_p
    return sorted({str(Path(d._dyld_get_image_name(i).decode()).resolve())
        for i in range(d._dyld_image_count()) if 'libomp' in d._dyld_get_image_name(i).decode().lower()})


def valid_pred(pred,n):
    for key in ['share','score','score_classifier']:
        v=np.asarray(pred[key])
        if v.shape!=(n,) or not np.isfinite(v).all() or ((v<0)|(v>1)).any():
            raise ValueError('Invalid frozen-model prediction: '+key)


def main(args):
    start=time.perf_counter();run=Path(args.run).resolve();output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('Fresh output folder required; no overwrite or partial reuse')
    if not output.is_relative_to(ROOT/'submissions'):raise ValueError('Pilot output must remain in project submissions')
    if platform.system()!='Darwin':raise ValueError('This entry point is explicitly Mac CPU only')
    lock_path=ROOT/'reports/R9_MAC_TREE_ENVIRONMENT_LOCK.json';lock=json.loads(lock_path.read_text())
    if (os.environ.get('DYLD_LIBRARY_PATH')!=lock['DYLD_LIBRARY_PATH']
            or sha(lock['resolved_libomp_path'])!=lock['libomp_sha256']
            or os.environ.get('KMP_DUPLICATE_LIB_OK','').upper() in ['TRUE','1']):
        raise ValueError('Pinned single-OpenMP environment required')
    jobs=json.loads((run/'jobs_completed.json').read_text())
    if (jobs.get('status')!='completed' or jobs.get('fixture_only') is not False
            or jobs.get('folds')!=list(range(5)) or (run/'FAILED.json').exists()):
        raise ValueError('All five completed unfailed production Mac tree jobs required')
    cfg_path=ROOT/'configs/r9_full_training_v1.json';config=json.loads(cfg_path.read_text())
    config_file_sha=sha(cfg_path)
    if canonical_sha(config)!=jobs['config_sha256']:raise ValueError('Model config differs')
    models=[];contexts=[];fold_evidence={};code={};run_before=regular_hashes(run)
    for fold in range(5):
        folder=run/f'fold_{fold:02d}';rec=json.loads((folder/'receipt.json').read_text())
        context=json.loads((folder/'context.json').read_text());binding=context['binding']
        if (sha(folder/'receipt.json')!=jobs['receipt_file_sha256'][str(fold)]
                or rec.get('status')!='completed' or rec.get('executor')!='mac_cpu'
                or rec.get('fixture_only') is not False or rec.get('binding')!=binding
                or canonical_sha({k:v for k,v in rec.items() if k!='receipt_sha256'})!=rec['receipt_sha256']
                or binding['config_file_sha256']!=config_file_sha or binding['config_sha256']!=jobs['config_sha256']
                or binding['fold']!=fold or binding['source_hash']!=jobs['source_hash']):
            raise ValueError('Completed context/receipt binding differs')
        check_map(folder,rec['artifact_sha256']);check_map(ROOT,context['code_sha256'])
        if rec['code_sha256']!=context['code_sha256']:raise ValueError('Model source closure differs')
        for n,h in context['code_sha256'].items():
            if n in code and code[n]!=h:raise ValueError('Fold source code mismatch')
            code[n]=h
        model=PhysicsLGBM.load(folder/'models/lgbm_physics',expected_binding=binding)
        if model.settings['threads']!=4:raise ValueError('Fixed CPU thread policy differs')
        models.append(model);contexts.append(context)
        fold_evidence[str(fold)]={'receipt_sha256':sha(folder/'receipt.json'),
            'context_sha256':sha(folder/'context.json'),
            'summary_sha256':sha(folder/'models/lgbm_physics/summary.json'),
            'model_artifact_sha256':model.summary['artifact_sha256'],
            'training_binding':binding,'feature_count':3827,
            'best_iteration':model.summary['best_iteration'],
            'presence_best_iteration':model.summary['presence_best_iteration']}
    images=omp_images()
    if images!=[lock['resolved_libomp_path']]:raise ValueError('Multiple OpenMP runtimes loaded')
    from threadpoolctl import threadpool_info
    if any(v['num_threads']!=4 for v in threadpool_info() if v['user_api']=='openmp'):
        raise ValueError('Four OpenMP threads required')
    code.update({'ops/infer_r9_lightgbm_pilot.py':sha(__file__),
                 'pipeline/prepare_sa.py':sha(ROOT/'pipeline/prepare_sa.py')})
    # The complete policy is persisted BEFORE reading query bytes or metadata.
    policy={'format':'r9_lightgbm_only_fivefold_pilot_v1','status':'frozen',
        'folds':list(range(5)),'fold_weights':[.2]*5,'branch':'lgbm_physics',
        'share':'arithmetic mean of frozen clipped fold shares',
        'score':'arithmetic mean of each fold .5*share+.5*presence_classifier',
        'postprocessing':'numerical clip[0,1] only','affine':[1.,0.],
        'feature_count':3827,'input':'per-window db only; existing R6 relative8+27 transform',
        'quantization_decimals':1,'full_batch_size':256,
        'predeclared_batch_replay':{'indices':[0,1,17,63,127,128,749,1499],
                                    'batch_sizes':[1,7],'atol':1e-6,'rtol':0},
        'expected_query_rows':1500,'config_file_sha256':config_file_sha,
        'jobs_completed_sha256':sha(run/'jobs_completed.json'),
        'fold_model_evidence':fold_evidence,'code_sha256':code,
        'environment_lock_sha256':sha(lock_path),'loaded_libomp':images,
        'frozen_before_Test_read':True,'fit_calibration_or_threshold_on_Test':False,
        'R9_full_five_member_pipeline_complete':False,'OOF_or_report_evaluation':False,
        'official_upload_or_submission':False,'created_utc':datetime.now(timezone.utc).isoformat()}
    policy['policy_sha256']=canonical_sha(policy)
    output.mkdir(parents=True);save(output/'frozen_policy.json',policy)
    try:
        input_path=Path(args.input).resolve();sample_path=ROOT/'data/raw/sample_submission.csv'
        input_sha=sha(input_path);sample_sha=sha(sample_path)
        test=pd.read_csv(input_path,dtype={'ID':str});sample=pd.read_csv(sample_path,dtype={'ID':str})
        features=[f'm{i:02d}_{c}' for i in range(1,16) for c in COMMON]
        if (len(features)!=375 or test.shape!=(1500,376) or list(test.columns)!=['ID',*features]
                or test.ID.isna().any() or not test.ID.is_unique
                or sample.shape!=(1500,3) or list(sample.columns)!=COLS
                or test.ID.tolist()!=sample.ID.tolist()):
            raise ValueError('Official query/schema/ID-order validation failed')
        db=test[features].to_numpy(np.float32).reshape(1500,15,25)
        if np.isinf(db).any() or not np.isfinite(db).any((1,2)).all():raise ValueError('Invalid query db/missing mask')
        ids=test.ID.tolist();predictions=[];checks={};selected=np.asarray(policy['predeclared_batch_replay']['indices'])
        for fold,model in enumerate(models):
            pred=model.predict(db,batch_size=256);valid_pred(pred,1500);foldchecks={}
            for batch in [1,7]:
                replay=model.predict(db[selected],batch_size=batch);valid_pred(replay,len(selected))
                for key in ['share','score','score_classifier']:
                    delta=float(np.max(np.abs(replay[key].astype(float)-pred[key][selected].astype(float))))
                    if delta>1e-6:raise ValueError('Fixed tree query batch dependence exceeds tolerance')
                    foldchecks[key+'_batch'+str(batch)]=delta
            checks[str(fold)]=foldchecks;predictions.append(pred)
        y=np.clip(np.mean(np.stack([v['share'].astype(np.float64) for v in predictions]),axis=0),0,1)
        s=np.clip(np.mean(np.stack([v['score'].astype(np.float64) for v in predictions]),axis=0),0,1)
        result=pd.DataFrame({'ID':ids,'share_added_vessel':y,'score_added_vessel':s})
        if not np.isfinite(result.iloc[:,1:].to_numpy()).all() or not result.iloc[:,1:].apply(lambda c:c.between(0,1)).all().all():
            raise ValueError('Invalid final prediction')
        temp=output/'submission.csv.tmp';result.to_csv(temp,index=False,float_format='%.17g')
        reread=pd.read_csv(temp,dtype={'ID':str},float_precision='round_trip')
        if (reread.shape!=(1500,3) or list(reread.columns)!=COLS or reread.ID.tolist()!=ids
                or not np.array_equal(reread.iloc[:,1:].to_numpy(),np.column_stack([y,s]))):
            raise ValueError('CSV serialization/order/precision validation failed')
        temp.replace(output/'submission.csv')
        np.savez_compressed(output/'frozen_fold_predictions.npz',ID=np.asarray(ids),
            **{f'fold_{i:02d}_{k}':v for i,p in enumerate(predictions) for k,v in p.items()},
            mean_share=y,mean_score=s)
        check_map(ROOT,code)
        if regular_hashes(run)!=run_before or sha(input_path)!=input_sha or sha(sample_path)!=sample_sha or sha(cfg_path)!=config_file_sha:
            raise ValueError('Frozen source/model/input bytes changed during inference')
        if omp_images()!=images:raise ValueError('Runtime library changed during inference')
        report={'format':'r9_lightgbm_pilot_inference_verification_v1','status':'passed','rows':1500,
            'columns':COLS,'ID_order_matches_Test_and_sample':True,'IDs_unique':True,
            'ID_order_sha256':canonical_sha(ids),'input_csv_sha256':input_sha,
            'sample_submission_sha256':sample_sha,'raw_db_array_sha256':array_sha(db),
            'policy_file_sha256':sha(output/'frozen_policy.json'),'policy_payload_sha256':policy['policy_sha256'],
            'submission_csv_sha256':sha(output/'submission.csv'),
            'fold_predictions_sha256':sha(output/'frozen_fold_predictions.npz'),
            'fold_model_evidence':fold_evidence,'code_sha256':code,'config_file_sha256':config_file_sha,
            'strict_completed_model_loads':5,'model_and_input_sha256_unchanged':True,
            'chunk_batch_replay':checks,'replay_atol':1e-6,'all_finite_and_in_unit_interval':True,
            'descriptive_output_validation_only_not_used_for_fitting':{
                'share_min':float(y.min()),'share_max':float(y.max()),'share_mean':float(y.mean()),
                'score_min':float(s.min()),'score_max':float(s.max()),'score_mean':float(s.mean())},
            'environment_lock_sha256':sha(lock_path),'loaded_libomp':images,
            'new_model_fit_Test_calibration_or_threshold_fit':False,'official_upload_or_submission':False,
            'full_R9_five_member_pipeline_or_OOF':False,'elapsed_seconds':time.perf_counter()-start}
        save(output/'verification.json',report)
        print(json.dumps({'status':'passed','CSV':str(output/'submission.csv'),
            'sha256':report['submission_csv_sha256'],'rows':1500,'elapsed_seconds':report['elapsed_seconds']}),flush=True)
    except BaseException as error:
        save(output/'FAILED.json',{'error_type':type(error).__name__,'error':str(error),'partial_reuse_or_overwrite_allowed':False})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default=str(ROOT/'models/r9_mac_trees_r2'))
    p.add_argument('--input',default=str(ROOT/'data/raw/Test.csv'))
    p.add_argument('--output',default=str(ROOT/'submissions/r9_lightgbm_5fold_cpu_r1'))
    main(p.parse_args())
