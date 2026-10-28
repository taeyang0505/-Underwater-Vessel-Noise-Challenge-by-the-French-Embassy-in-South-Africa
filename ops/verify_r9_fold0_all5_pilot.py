"""Independent frozen Fold0 pilot QA: no fitting, report read or CSV rewrite."""
from pathlib import Path
import argparse
import csv
import ctypes
from datetime import datetime,timezone
import gc
import hashlib
import json
import math
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
KINDS=('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer')
MEMBERS=KINDS+('lgbm_physics',)
INDICES=[0,1,17,63,127,128,749,1499]
TOL=3e-6


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def inside(name):
    p=(ROOT/Path(name)).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Path escapes project')
    return p
def verify(pins):
    for n,h in pins.items():
        p=inside(n)
        if not p.is_file() or p.is_symlink() or sha(p)!=h:raise ValueError('Frozen bytes changed: '+n)
def omp():
    d=ctypes.CDLL(None);d._dyld_image_count.restype=ctypes.c_uint32
    d._dyld_get_image_name.argtypes=[ctypes.c_uint32];d._dyld_get_image_name.restype=ctypes.c_char_p
    return sorted({str(Path(d._dyld_get_image_name(i).decode()).resolve()) for i in range(d._dyld_image_count())
        if 'libomp' in d._dyld_get_image_name(i).decode().lower()})


def main(a):
    start=time.perf_counter();folder=inside(a.output);destination=folder/'independent_verification.json'
    if destination.exists():raise FileExistsError('Independent receipt already exists; overwrite forbidden')
    if (folder/'FAILED.json').exists():raise ValueError('Failed pilot cannot be verified')
    plan=read(folder/'prequery_plan.json');policy=read(folder/'frozen_policy.json')
    receipt=read(folder/'inference_receipt.json');ready=read(folder/'PREQUERY_READY.json')
    if (canonical({k:v for k,v in policy.items() if k!='policy_sha256'})!=policy['policy_sha256']
            or policy['prequery_plan_sha256']!=sha(folder/'prequery_plan.json')
            or receipt['policy_file_sha256']!=sha(folder/'frozen_policy.json')
            or ready['frozen_policy_file_sha256']!=receipt['policy_file_sha256']
            or receipt['output_sha256']!=sha(folder/'submission.csv')
            or policy['member_order']!=list(MEMBERS) or policy['source_pins']!=plan['source_pins']
            or policy['weights']!=[.1,.1,.1,.1,.6]):raise ValueError('Frozen all-five policy or output identity differs')
    verify(plan['source_pins'])
    original={n:sha(folder/n) for n in ['submission.csv','query_member_predictions.npz','frozen_policy.json','prequery_plan.json','inference_receipt.json']}
    lock=read(ROOT/'reports/R9_MAC_TREE_ENVIRONMENT_LOCK.json')
    if os.environ.get('DYLD_LIBRARY_PATH')!=lock['DYLD_LIBRARY_PATH'] or os.environ.get('KMP_DUPLICATE_LIB_OK'):
        raise ValueError('Fresh single-runtime process required')
    if sha(lock['resolved_libomp_path'])!=lock['libomp_sha256']:raise ValueError('OpenMP binary changed')
    def audit(event,args):
        if event=='socket.connect':raise PermissionError('Network disabled in independent verification')
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            name=Path(os.fsdecode(args[0])).name
            if name in ['report.npz','report_metadata.parquet','report_recipes.json']:
                raise PermissionError('Report inputs are forbidden')
    sys.addaudithook(audit)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,Dataset
    from pipeline.inference_r5 import read_query_csv
    from pipeline.relative_grid_r6_v2 import relative_grid_v2
    from pipeline.runner_r9 import restore_neural
    from pipeline.training_r9 import predict_loader
    from pipeline.trees_ensemble_r9 import PhysicsLGBM
    torch.set_num_threads(4)
    ids,db,query=read_query_csv(inside(a.input),expected_rows=1500)
    if query['raw_query_csv_sha256']!=receipt['query']['raw_query_csv_sha256'] or canonical(ids)!=receipt['ID_order_sha256']:
        raise ValueError('Query bytes or row order changed')
    with np.load(folder/'query_member_predictions.npz',allow_pickle=False) as z:
        saved={k:z[k] for k in z.files}
    if saved['ID'].astype(str).tolist()!=ids:raise ValueError('Saved member query order differs')
    if saved['share'].shape!=(1500,5) or saved['score'].shape!=(1500,5):raise ValueError('Saved member dimensions differ')
    with (folder/'submission.csv').open(newline='') as f:
        rows=list(csv.reader(f))
    if (rows[0]!=['ID','share_added_vessel','score_added_vessel'] or len(rows)!=1501
            or [r[0] for r in rows[1:]]!=ids or len(set(ids))!=1500):raise ValueError('CSV schema/ID order differs')
    csv_values=np.asarray([[float(x) for x in r[1:]] for r in rows[1:]],dtype=np.float64)
    if not np.isfinite(csv_values).all() or ((csv_values<0)|(csv_values>1)).any():raise ValueError('Invalid CSV predictions')
    hand={};manual_error={};arithmetic_rounding={}
    for col,key in enumerate(['share','score']):
        matrix=np.asarray(saved[key],dtype=np.float64)
        if not np.isfinite(matrix).all() or ((matrix<0)|(matrix>1)).any():raise ValueError('Invalid member predictions')
        full=np.asarray([math.fsum(float(row[i])*policy['weights'][i] for i in range(5)) for row in matrix])
        # fsum and BLAS can differ by machine-epsilon at a float32 midpoint.
        # Verify their float64 error bound independently, then require exact
        # reconstruction of the production expression and its CSV roundtrip.
        production=matrix@np.asarray(policy['weights'],dtype=np.float64)
        bound=10*np.finfo(np.float64).eps*np.sum(np.abs(matrix*np.asarray(policy['weights'])),axis=1)+np.finfo(np.float64).tiny
        if (np.abs(full-production)>bound).any():raise ValueError('Weighted-sum arithmetic differs beyond rounding bound')
        manual32=np.clip(full,0,1).astype(np.float32);stored=saved['ensemble_'+key]
        if (np.abs(manual32-stored)>np.abs(np.spacing(stored))).any():raise ValueError('Independent fsum exceeds one float32 ULP')
        hand[key]=np.clip(production,0,1).astype(np.float32)
        if (not np.array_equal(hand[key],stored)
                or not np.array_equal(csv_values[:,col],hand[key].astype(np.float64))):
            raise ValueError('Manual fixed-weight combination differs from NPZ/CSV')
        arithmetic_rounding[key]={'fsum64_vs_BLAS64_max':float(np.max(np.abs(full-production))),
            'fsum32_vs_stored32_max':float(np.max(np.abs(manual32-stored))),
            'float32_midpoint_different_rows':int((manual32!=stored).sum()),
            'within_derived_float64_rounding_bound_and_one_float32_ULP':True,
            'production_expression_and_CSV_exact':True}
        manual_error[key]=float(np.max(np.abs(full-saved['ensemble_'+key].astype(float))))
    selected=np.asarray(INDICES);x,w=relative_grid_v2(db[selected])
    class Query(Dataset):
        def __len__(self):return len(x)
        def __getitem__(self,i):return {'x':x[i],'w':w[i]}
    replay={}
    for column,kind in enumerate(KINDS):
        model,summary=restore_neural(inside(plan['nn_dirs'][kind]),device='cpu',fixture=False,
                                     expected_identity=plan['binding']['selection_identity'])
        if summary['kind']!=kind or summary['train_identity']!=plan['binding']['train_identity']:
            raise ValueError('NN source binding differs')
        detail={}
        for batch in [1,7]:
            p=predict_loader(model,DataLoader(Query(),batch_size=batch,shuffle=False,num_workers=0),
                             'cpu',bf16=False,collect_targets=False)
            for key in ['share','score']:
                err=float(np.max(np.abs(p[key].astype(np.float64)-saved[key][selected,column].astype(np.float64))))
                detail[key+'_batch'+str(batch)]=err
                if not np.isfinite(err) or err>TOL:
                    print(json.dumps({'status':'FAILED_no_output_rewritten','kind':kind,'key':key,'batch':batch,'error':err,'tolerance':TOL}),flush=True)
                    raise ValueError('CPU query batch dependence exceeds fixed tolerance')
            gap=float(np.max(np.abs(p['share'].astype(float)-(p['cell_mask'].astype(float)*w).sum((1,2)))))
            if gap>TOL or not np.isfinite(p['cell_mask']).all() or (p['cell_mask'][w==0]!=0).any():
                raise ValueError('Physical mask/pooling validation failed')
            detail['physical_pool_batch'+str(batch)]=gap
        replay[kind]=detail;del model;gc.collect()
    job=inside(plan['tree_job']);tree=PhysicsLGBM.load(job/'models/lgbm_physics',expected_binding=plan['binding'])
    replay['lgbm_physics']={}
    for batch in [1,7]:
        p=tree.predict(db[selected],batch_size=batch)
        for key in ['share','score']:
            err=float(np.max(np.abs(p[key].astype(float)-saved[key][selected,4].astype(float))))
            if not np.isfinite(err) or err>TOL:raise ValueError('Tree batch replay differs')
            replay['lgbm_physics'][key+'_batch'+str(batch)]=err
    if omp()!=[lock['resolved_libomp_path']]:raise ValueError('Multiple runtime images loaded')
    verify(plan['source_pins'])
    if {n:sha(folder/n) for n in original}!=original or sha(inside(a.input))!=query['raw_query_csv_sha256']:
        raise ValueError('CSV/policy/source/query changed during independent verification')
    result={'format':'r9_fold0_all5_independent_inference_verification_v1','status':'passed',
        'created_utc':datetime.now(timezone.utc).isoformat(),'CSV_rows':1500,'CSV_columns':rows[0],
        'ID_order_exact_and_unique':True,'finite_and_unit_interval':True,
        'query_indices':INDICES,'replay_batch_sizes':[1,7],'fixed_atol':TOL,'rtol':0,
        'per_model_replay_actual_errors':replay,
        'maximum_share_score_replay_error':max(v for detail in replay.values() for k,v in detail.items() if not k.startswith('physical')),
        'fixed_weight_production_expression_exact_saved_and_CSV':True,
        'independent_math_fsum_rounding_diagnostics':arithmetic_rounding,
        'manual_float64_vs_saved_float32_error':manual_error,'weights':dict(zip(MEMBERS,policy['weights'])),
        'strict_completed_best_window_checkpoints':4,'strict_completed_Mac_tree':1,
        'all_model_source_pins_unchanged':True,'all_output_policy_query_SHA_unchanged':True,
        'original_file_sha256':original,'raw_query_csv_sha256':query['raw_query_csv_sha256'],
        'policy_payload_sha256':policy['policy_sha256'],'loaded_libomp':omp(),
        'verification_code_sha256':sha(__file__),'elapsed_seconds':time.perf_counter()-start,
        'new_fit_weight_adjustment_report_access_GPU_SSH_upload':False,
        'scope':'single Fold0 all-five pilot, not full-fivefold neural pipeline or OOF'}
    with destination.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','receipt':str(destination),'receipt_sha256':sha(destination),
        'max_replay_error':result['maximum_share_score_replay_error'],'seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='submissions/r9_fold0_all5_cpu_r1');p.add_argument('--input',default='data/raw/Test.csv')
    main(p.parse_args())
