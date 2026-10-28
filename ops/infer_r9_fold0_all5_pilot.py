"""Separate Mac CPU Fold0 all-five pilot; never a completed-fivefold R9 run.

freeze: choose one of THREE strictly-positive policies on ensemble_fit only;
write frozen weights/pins BEFORE gate; gate is comparison-only, never refit.
infer: verify that frozen state before opening the explicit query CSV.
No TRAIN fitting, report access, upload, calibration or query-population fit.
--self-test uses tiny numeric observations only and never opens actual Test.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
KINDS=('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer')
MEMBERS=KINDS+('lgbm_physics',)
CANDIDATES={'equal_all5':[.2,.2,.2,.2,.2],
            'tree60_nn10_each':[.1,.1,.1,.1,.6],
            'tree40_nn15_each':[.15,.15,.15,.15,.4]}
DEFAULT_OUTPUT='submissions/r9_fold0_all5_cpu_r1'
DEFAULT_CONFIG='configs/r9_full_training_v1.json'
DEFAULT_BUNDLE='data/synthetic/group_cv_minute_full_r9_v1'
DEFAULT_TREE='models/r9_mac_trees_r2/fold_00'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value,*,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
def utc():return datetime.now(timezone.utc).isoformat()
def local(path):
    p=Path(path);p=(ROOT/p if not p.is_absolute() else p).resolve()
    if ROOT not in p.parents:raise ValueError('Pilot inputs/outputs must stay inside this project')
    return p
def relative(path):return str(Path(path).relative_to(ROOT))
def verify_pins(pins):
    for name,digest in pins.items():
        p=local(name)
        if not p.is_file() or p.is_symlink() or sha(p)!=digest:raise ValueError('Frozen pilot input changed:'+name)


def environment(threads):
    if os.environ.get('KMP_DUPLICATE_LIB_OK'):raise ValueError('KMP_DUPLICATE_LIB_OK is forbidden')
    if sys.platform=='darwin' and '/opt/homebrew/opt/libomp/lib' not in os.environ.get('DYLD_LIBRARY_PATH','').split(':'):
        raise RuntimeError('Start a fresh process with DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib before importing Torch/LightGBM')
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name]=str(threads)


def default_nn_dirs():
    return {k:f'ops/server_backups/r9_gpu_r1/fold_00_{k}_completed_r1/restored/fold_00/models/{k}' for k in KINDS}


def check_summary(summary,kind,selection_identity,train_identity):
    if (summary.get('kind')!=kind or summary.get('status')!='completed'
            or summary.get('epochs_run')!=100 or summary.get('fixture_only') is not False
            or summary.get('stop_reason')!='completed_fixed_epochs'
            or summary.get('validation_identity')!=selection_identity
            or summary.get('train_identity')!=train_identity):
        raise ValueError('Completed Fold0 checkpoint/data binding differs:'+kind)


def inputs(nn_dirs,tree_job,bundle,config):
    from pipeline.runner_r9 import CODE_NAMES
    from pipeline.trees_ensemble_r9 import PhysicsLGBM
    if set(nn_dirs)!=set(KINDS):raise ValueError('Exactly four explicit completed neural directories required')
    cfg=read(config);receipt=read(tree_job/'receipt.json');binding=receipt['binding']
    if (receipt.get('status')!='completed' or receipt.get('fixture_only') is not False
            or receipt.get('executor')!='mac_cpu' or receipt.get('platform_system')!='Darwin'
            or receipt.get('fold')!=0 or (tree_job/'FAILED.json').exists()
            or canonical({k:v for k,v in receipt.items() if k!='receipt_sha256'})!=receipt.get('receipt_sha256')):
        raise ValueError('Completed native Mac Fold0 tree receipt required')
    if binding['config_file_sha256']!=sha(config) or binding['config_sha256']!=canonical(cfg):raise ValueError('Tree/config identity differs')
    manifest=read(bundle/'manifest.json');ready=read(bundle/'readiness.json')
    if (ready.get('ready_to_train') is not True or ready['manifest_sha256']!=sha(bundle/'manifest.json')
            or manifest['source_hash']!=binding['data_bundle_hash'] or ready['source_hash']!=manifest['source_hash']):
        raise ValueError('Prepared data identity differs')
    pins={relative(config):sha(config),relative(bundle/'manifest.json'):sha(bundle/'manifest.json'),
          relative(bundle/'readiness.json'):sha(bundle/'readiness.json'),
          relative(tree_job/'receipt.json'):sha(tree_job/'receipt.json')}
    if set(receipt['code_sha256'])!=set(CODE_NAMES):raise ValueError('Full R9 source closure required')
    for name,digest in receipt['code_sha256'].items():
        if sha(local(name))!=digest:raise ValueError('Frozen R9 source changed:'+name)
        pins[name]=digest
    for name,digest in receipt['artifact_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Escaping tree artifact')
        pins[relative(tree_job/name)]=digest
    for role in ['ensemble_fit','ensemble_gate']:
        for suffix in ['.npz','_metadata.parquet','_recipes.json']:
            name=f'fold_00/{role}{suffix}';pins[relative(bundle/name)]=manifest['file_sha256'][name]
    model_pins={}
    for kind in KINDS:
        folder=local(nn_dirs[kind]);summary=read(folder/'summary.json')
        check_summary(summary,kind,binding['selection_identity'],binding['train_identity'])
        if (folder/'FAILED.json').exists():raise ValueError('Failed neural artifact cannot be reused')
        pins[relative(folder/'summary.json')]=sha(folder/'summary.json')
        for name,digest in summary['artifact_sha256'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Escaping model artifact')
            pins[relative(folder/name)]=digest
        model_pins[kind]={'directory':relative(folder),'summary_sha256':sha(folder/'summary.json'),
                          'best_window_sha256':summary['artifact_sha256']['best_window.pt'],
                          'best_window_epoch':summary['best_window_epoch']}
    pins[relative(Path(__file__))]=sha(Path(__file__))
    for name in ['pipeline/inference_r5.py','pipeline/hierarchical_ensemble_r5.py']:pins[name]=sha(ROOT/name)
    verify_pins(pins)
    # Booster load performs schema/source/artifact checks; never fitting.
    tree=PhysicsLGBM.load(tree_job/'models/lgbm_physics',expected_binding=binding)
    return pins,model_pins,binding,tree


def predict_neural_db(folder,db,selection_identity,*,batch_size,threads):
    import numpy as np
    import torch
    from torch.utils.data import Dataset,DataLoader
    from pipeline.runner_r9 import restore_neural
    from pipeline.relative_grid_r6_v2 import relative_grid_v2
    from pipeline.training_r9 import predict_loader
    torch.set_num_threads(threads)
    x,w=relative_grid_v2(db)
    class Query(Dataset):
        def __len__(self):return len(x)
        def __getitem__(self,i):return {'x':x[i],'w':w[i]}
    model,summary=restore_neural(folder,device='cpu',fixture=False,expected_identity=selection_identity)
    begin=time.perf_counter()
    result=predict_loader(model,DataLoader(Query(),batch_size=batch_size,shuffle=False,num_workers=0),
                          device='cpu',bf16=False,collect_targets=False)
    gap=float(np.max(np.abs(result['share'].astype(float)-(result['cell_mask'].astype(float)*w).sum((1,2)))))
    if gap>3e-6:raise ValueError('CPU neural prediction bypasses physical pooling')
    elapsed=time.perf_counter()-begin
    del model;gc.collect()
    return {k:result[k] for k in ['share','score']},{'seconds':elapsed,'physical_pool_max_abs':gap,
           'best_window_epoch':summary['best_window_epoch'],'examples':len(db),'device':'cpu FP32'}


def predict_members(nn_dirs,tree,db,binding,*,batch_size,threads):
    out={};timings={}
    for kind in KINDS:
        out[kind],timings[kind]=predict_neural_db(local(nn_dirs[kind]),db,binding['selection_identity'],batch_size=batch_size,threads=threads)
    begin=time.perf_counter();out['lgbm_physics']=tree.predict(db,batch_size=batch_size)
    timings['lgbm_physics']={'seconds':time.perf_counter()-begin,'examples':len(db),'device':'Mac CPU'}
    return out,timings


def combine(predictions,weights):
    import numpy as np
    w=np.asarray(weights,dtype=np.float64)
    if w.shape!=(5,) or not np.isfinite(w).all() or (w<=0).any() or abs(w.sum()-1)>1e-12:raise ValueError('All five positive simplex weights required')
    result={}
    for key in ['share','score']:
        columns=[np.asarray(predictions[k][key],dtype=np.float64) for k in MEMBERS]
        if any(a.ndim!=1 or a.shape!=columns[0].shape or not np.isfinite(a).all() or ((a<0)|(a>1)).any() for a in columns):raise ValueError('Aligned bounded member predictions required')
        result[key]=np.clip(np.stack(columns,axis=1)@w,0,1).astype(np.float32)
    return result


def select_candidate(y,predictions):
    import numpy as np
    y=np.asarray(y,dtype=float);scores={}
    for name,w in CANDIDATES.items():
        p=combine(predictions,w)['share'].astype(float)
        if p.shape!=y.shape:raise ValueError('Fit target/prediction order differs')
        scores[name]=float(np.sqrt(np.mean((p-y)**2)))
    selected=min(CANDIDATES,key=lambda name:scores[name])
    return selected,list(CANDIDATES[selected]),scores


def metrics(y,pred):
    import numpy as np
    error=np.asarray(pred,dtype=float)-np.asarray(y,dtype=float)
    return {'n':len(error),'rmse':float(np.sqrt(np.mean(error**2))),
            'mae':float(np.abs(error).mean()),'bias':float(error.mean())}


def freeze(a):
    import numpy as np
    from pipeline.dataset_r9 import FixedR9Dataset
    output=local(a.output)
    if output.exists():raise FileExistsError('Fresh pilot output required, no partial retry')
    nn=read(local(a.nn_map)) if a.nn_map else default_nn_dirs();tree_job=local(a.tree_job);bundle=local(a.bundle);config=local(a.config)
    pins,modelpins,binding,tree=inputs(nn,tree_job,bundle,config)
    output.mkdir(parents=True)
    plan={'format':'r9_fold0_all5_pilot_prequery_plan_v1','created_utc':utc(),'member_order':list(MEMBERS),
          'candidates':CANDIDATES,'selection':'ensemble_fit500 RMSE only; exactly one candidate',
          'gate':'compare frozen winner vs tree-only and neural-only; no refit, no zero weights',
          'nn_dirs':nn,'tree_job':relative(tree_job),'bundle':relative(bundle),'config':relative(config),
          'model_pins':modelpins,'source_pins':pins,'binding':binding,'batch_size':a.batch_size,'threads':a.threads,
          'checkpoint':'strict best_window','report_accessed':False,'Test_opened':False,'fit_executed':False}
    write(output/'prequery_plan.json',plan,exclusive=True)
    try:
        fit=FixedR9Dataset(bundle,0,'ensemble_fit');raw=fit.raw_arrays()
        if len(fit)!=500:raise ValueError('Exact Fold0fit500 required')
        pred,timing=predict_members(nn,tree,raw['db'],binding,batch_size=a.batch_size,threads=a.threads)
        selected,weights,scores=select_candidate(raw['share'],pred)
        np.savez_compressed(output/'fit_member_predictions.npz',sample_id=raw['sample_id'],
             share=np.stack([pred[k]['share'] for k in MEMBERS],axis=1),score=np.stack([pred[k]['score'] for k in MEMBERS],axis=1))
        policy={'format':'r9_fold0_all5_pilot_policy_v1','created_utc':utc(),'member_order':list(MEMBERS),
                'selected_name':selected,'weights':weights,'candidate_fit_rmse':scores,
                'fit_identity':fit.identity,'fit_sample_id_sha256':canonical(raw['sample_id'].astype(str).tolist()),
                'prequery_plan_sha256':sha(output/'prequery_plan.json'),'source_pins':pins,
                'all5_strictly_positive':True,'affine':[1.,0.],'gate_refit_allowed':False,
                'reference_independent_cv':False,'production_fivefold_run':False,'report_accessed':False,
                'completed_individual_models_reused_without_claiming_complete_run_ledger':True,
                'Test_opened_before_weight_freeze':False,'official_submission':False}
        policy['policy_sha256']=canonical(policy);write(output/'frozen_policy.json',policy,exclusive=True)
        # Only now open gate labels; policy bytes are frozen and never rewritten.
        policy_file_sha=sha(output/'frozen_policy.json');gate=FixedR9Dataset(bundle,0,'ensemble_gate');gr=gate.raw_arrays()
        if len(gate)!=500 or set(raw['background_group'].astype(str))&set(gr['background_group'].astype(str)):raise ValueError('Exact source-disjoint gate500 required')
        gp,gt=predict_members(nn,tree,gr['db'],binding,batch_size=a.batch_size,threads=a.threads)
        winner=combine(gp,weights);nnmean=np.mean([gp[k]['share'] for k in KINDS],axis=0)
        comparisons={'all5_frozen':metrics(gr['share'],winner['share']),
                     'tree_only':metrics(gr['share'],gp['lgbm_physics']['share']),
                     'neural_only_equal':metrics(gr['share'],nnmean)}
        gate_result={'format':'r9_fold0_all5_gate_comparison_v1','metrics':comparisons,'gate_identity':gate.identity,
            'all5_worse_than_tree':comparisons['all5_frozen']['rmse']>comparisons['tree_only']['rmse'],
            'all5_worse_than_neural_mean':comparisons['all5_frozen']['rmse']>comparisons['neural_only_equal']['rmse'],
            'weights_changed_after_gate':False,'all5_positive_retained_even_if_worse':True,
            'gate_used_for_candidate_selection':False,'fit_timings':timing,'gate_timings':gt,'report_accessed':False}
        write(output/'gate_comparison.json',gate_result,exclusive=True)
        np.savez_compressed(output/'gate_member_predictions.npz',sample_id=gr['sample_id'],
            share=np.stack([gp[k]['share'] for k in MEMBERS],axis=1),score=np.stack([gp[k]['score'] for k in MEMBERS],axis=1))
        verify_pins(pins)
        if sha(output/'frozen_policy.json')!=policy_file_sha:raise ValueError('Weights changed after freeze')
        ready={'status':'ready_for_explicit_query_inference','frozen_policy_file_sha256':policy_file_sha,
               'prequery_plan_sha256':sha(output/'prequery_plan.json'),'gate_comparison_sha256':sha(output/'gate_comparison.json'),
               'all5_positive':True,'Test_opened':False,'report_opened':False,'fit_executed':False}
        write(output/'PREQUERY_READY.json',ready,exclusive=True)
        print(json.dumps({'status':ready['status'],'selected':selected,'weights':weights,'gate':comparisons}));return ready
    except BaseException as error:
        write(output/'FAILED.json',{'error_type':type(error).__name__,'error':str(error),'automatic_retry':False});raise


def infer(a):
    import numpy as np
    from pipeline.inference_r5 import read_query_csv
    output=local(a.output);ready=read(output/'PREQUERY_READY.json');plan=read(output/'prequery_plan.json');policy=read(output/'frozen_policy.json')
    if (output/'FAILED.json').exists() or (output/'submission.csv').exists() or (output/'QUERY_STARTED.json').exists():raise ValueError('Failed/completed query cannot be retried in place')
    if (sha(output/'prequery_plan.json')!=ready['prequery_plan_sha256'] or sha(output/'frozen_policy.json')!=ready['frozen_policy_file_sha256']
            or sha(output/'gate_comparison.json')!=ready['gate_comparison_sha256']
            or canonical({k:v for k,v in policy.items() if k!='policy_sha256'})!=policy['policy_sha256']):raise ValueError('Frozen pilot receipt differs')
    if (policy['prequery_plan_sha256']!=ready['prequery_plan_sha256'] or policy.get('member_order')!=list(MEMBERS)
            or policy['selected_name'] not in CANDIDATES or policy['weights']!=CANDIDATES[policy['selected_name']]):
        raise ValueError('Policy is not one of the three predeclared all-five candidates')
    verify_pins(plan['source_pins'])
    pins,_,binding,tree=inputs(plan['nn_dirs'],local(plan['tree_job']),local(plan['bundle']),local(plan['config']))
    if pins!=plan['source_pins'] or binding!=plan['binding']:raise ValueError('Model/source identity changed')
    # Query is not opened until all five completed model/policy identities pass.
    write(output/'QUERY_STARTED.json',{'utc':utc(),'policy_sha256':policy['policy_sha256'],
          'no_training_or_calibration':True,'all_model_pins_verified_before_query':True},exclusive=True)
    try:
        ids,db,query=read_query_csv(local(a.input),expected_rows=a.expected_rows)
        pred,timings=predict_members(plan['nn_dirs'],tree,db,binding,batch_size=a.batch_size,threads=a.threads)
        result=combine(pred,policy['weights']);csvpath=output/'submission.csv'
        with csvpath.open('x',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['ID','share_added_vessel','score_added_vessel'])
            writer.writerows(zip(ids,result['share'].tolist(),result['score'].tolist()))
        with csvpath.open(newline='') as stream:
            rows=list(csv.reader(stream))
        if [row[0] for row in rows[1:]]!=ids or len(rows)!=len(ids)+1:raise ValueError('OutputID order/coverage differs')
        parsed=np.asarray([[float(v) for v in row[1:]] for row in rows[1:]])
        if not np.array_equal(parsed[:,0],result['share'].astype(float)) or not np.array_equal(parsed[:,1],result['score'].astype(float)):raise ValueError('CSV float roundtrip differs')
        verify_pins(pins)
        np.savez_compressed(output/'query_member_predictions.npz',ID=np.asarray(ids),
            share=np.stack([pred[k]['share'] for k in MEMBERS],axis=1),score=np.stack([pred[k]['score'] for k in MEMBERS],axis=1),
            ensemble_share=result['share'],ensemble_score=result['score'])
        receipt={'status':'completed_local_CSV_only','query':query,'ID_order_sha256':canonical(ids),
            'policy_file_sha256':sha(output/'frozen_policy.json'),'output_sha256':sha(csvpath),'model_timings':timings,
            'all5_weights':dict(zip(MEMBERS,policy['weights'])),'all5_positive':True,
            'report_accessed':False,'training_or_calibration_executed':False,'upload_or_official_submission':False,
            'single_fold_pilot_not_full5fold_neural_inference':True,'CSV_roundtrip_exact':True,'additional_postprocessing':'[0,1] numerical clipping only'}
        write(output/'inference_receipt.json',receipt,exclusive=True);print(json.dumps(receipt));return receipt
    except BaseException as error:
        write(output/'FAILED.json',{'error_type':type(error).__name__,'error':str(error),'automatic_retry':False});raise


def self_test():
    import numpy as np
    from pipeline.inference_r5 import read_query_csv,FEATURE_COLUMNS
    rng=np.random.default_rng(17);db=rng.normal(90,3,(3,15,25)).astype(np.float32);db[0,:3]=np.nan
    ids=['id_z','id_a','id_m']
    with tempfile.TemporaryDirectory(prefix='r9_all5_numeric_') as tmp:
        p=Path(tmp)/'numeric_query.csv'
        with p.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['ID',*FEATURE_COLUMNS]);writer.writerows([[ids[i],*[str(v) if np.isfinite(v) else '' for v in db[i].ravel()]] for i in range(3)])
        actual,query,_=read_query_csv(p,expected_rows=3)
        assert actual==ids and np.array_equal(query,db,equal_nan=True)
        bad=Path(tmp)/'bad_numeric_header.csv';bad.write_text(p.read_text().replace('m01_TOL_31.6','wrong_band_name',1))
        try:read_query_csv(bad,expected_rows=3)
        except ValueError:wrong_header_rejected=True
        else:raise AssertionError('Wrong public header accepted')
        predictions={k:{'share':np.array([.1,.3,.8],np.float32)+i*.01,'score':np.array([.2,.4,.7],np.float32)+i*.01} for i,k in enumerate(MEMBERS)}
        y=np.array([.14,.34,.84]);selected,weights,scores=select_candidate(y,predictions)
        before=canonical(weights);combined=combine(predictions,weights)
        oracle=np.stack([predictions[k]['share'] for k in MEMBERS],axis=1).astype(float)@np.asarray(weights)
        assert np.array_equal(combined['share'],oracle.astype(np.float32))
        gate_y=np.zeros(3);comparison=metrics(gate_y,combined['share']);assert canonical(weights)==before
        rejected=0
        for bad in [[0,.25,.25,.25,.25],[.2,.2,.2,.2,.3]]:
            try:combine(predictions,bad)
            except ValueError:rejected+=1
        assert rejected==2
        from pipeline.models_r9 import make_r9_model
        from pipeline.relative_grid_r6_v2 import relative_grid_v2
        import torch
        torch.set_num_threads(2);x,w=relative_grid_v2(db);model=make_r9_model('bg_cnn',width=4,blocks=1).eval()
        with torch.no_grad():share,logit,q=model(torch.from_numpy(x),torch.from_numpy(w))
        assert torch.allclose(share,(q*torch.from_numpy(w)).sum((1,2)),atol=3e-6,rtol=0)
        pin=Path(tmp)/'pin.json';pin.write_text('{}');old=sha(pin);pin.write_text('{"changed":true}');assert sha(pin)!=old
    return {'status':'passed','numeric_fixture_only':True,'actual_Test_or_models_or_TRAIN_or_report_read':False,
        'fit_candidates_exact':CANDIDATES,'chosen_candidate':selected,'fit_scores':scores,
        'all5_positive':True,'weighted_share_oracle_exact':True,'gate_did_not_change_weights':True,
        'ID_order_and_375column_NaN_roundtrip_exact':True,'invalid_weights_rejected':2,
        'wrong_public_header_rejected':wrong_header_rejected,
        'tiny_actual_adapter_physical_pool_passed':True,'source_sha256':sha(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','infer','self-test'])
    p.add_argument('--nn-map');p.add_argument('--tree-job',default=DEFAULT_TREE);p.add_argument('--bundle',default=DEFAULT_BUNDLE)
    p.add_argument('--config',default=DEFAULT_CONFIG);p.add_argument('--output',default=DEFAULT_OUTPUT)
    p.add_argument('--input');p.add_argument('--expected-rows',type=int,default=1500)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--threads',type=int,default=2);p.add_argument('--test-output')
    a=p.parse_args()
    if a.action=='self-test':
        result=self_test()
        if a.test_output:write(local(a.test_output),result)
        print(json.dumps(result));return
    if a.threads<1 or a.batch_size<1:raise ValueError('Positive threads/batch required')
    environment(a.threads)
    if a.action=='freeze':
        if a.input:raise ValueError('freeze never accepts a query input')
        freeze(a)
    else:
        if not a.input:raise ValueError('Explicit query CSV required after freeze')
        infer(a)


if __name__=='__main__':main()
