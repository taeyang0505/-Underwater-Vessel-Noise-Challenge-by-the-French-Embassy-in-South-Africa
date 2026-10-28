"""Original raw R5 CNN, full FP32 Mac CPU inference on INNER only.

Weights/prior/policy are immutable. No fitting, postprocessing selection,
external data, Test/OUTER, GPU, SSH or submission. Saved CUDA BF16 predictions
are a comparison reference, not a requirement that FP32 be bitwise identical.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib.metadata,json,os,resource,sys,time,zipfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
BACKUP=ROOT/'ops/server_backups/r5_gpu_r1'
ARCHIVE='r5_snapshot_20261025T161450_550440Z_8aeccd97.zip'
ARCHIVE_SHA='12ec9d8bfb176a19c53891a3578ab32ffdacb4c1d99633a3f11b8a94c5a2fe2a'
CHECKPOINT_SHA='aa222697275306cf93e5778a707f7ee5fd75610ae0a0bf643d654854ee96d46f'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def array_sha(a):
    import numpy as np
    h=hashlib.sha256(json.dumps({'shape':list(a.shape),'dtype':str(a.dtype)},sort_keys=True,separators=(',',':')).encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()
def state_sha(model):
    return canonical({k:array_sha(v.detach().cpu().numpy()) for k,v in model.state_dict().items()})
def rss():return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
def write(path,x):
    with Path(path).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def run(report,output):
    report=Path(report).resolve();output=Path(output).resolve()
    if report.exists() or output.exists():raise FileExistsError('Fresh diagnostic report/directory required; no duplicate run or overwrite')
    output.mkdir(parents=True);start_all=time.perf_counter();stage='source_verification';times={}
    try:
        import numpy as np,pandas as pd,torch
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        from pipeline import neural_r5
        from pipeline.fold_data_r5 import FoldDataR5
        from pipeline.grid_inputs_r5 import load_shared_prior
        from pipeline.observation_augmentation_r5 import prepare_query_observations
        restored=BACKUP/'restored_first_cnn_v1';folder=restored/'fold_00';model_dir=folder/'preprocessing_pilot/raw/bg_cnn'
        archive=BACKUP/ARCHIVE
        if sha(archive)!=ARCHIVE_SHA:raise ValueError('Original verified snapshot ZIP SHA mismatch')
        sources={};t=time.perf_counter()
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise ValueError('Original ZIP CRC failure')
            manifest=json.loads(z.read('__snapshot_manifest.json'))
            for item in manifest['files']:
                p=restored/item['relative_path'];payload=z.read(item['archive_path'])
                if hashlib.sha256(payload).hexdigest()!=item['sha256'] or len(payload)!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Restored snapshot entry changed')
                sources[p.relative_to(ROOT).as_posix()]=item['sha256']
        context=json.loads((folder/'context.json').read_text());summary=json.loads((model_dir/'summary.json').read_text())
        if summary['status']!='completed' or summary['fixture_only'] is not False or summary['kind']!='bg_cnn' or summary['best_epoch']!=19:raise ValueError('Expected completed original raw CNN epoch19')
        if sha(model_dir/'best.pt')!=CHECKPOINT_SHA or summary['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Original checkpoint identity differs')
        if context['config_sha256']!=canonical(context['config']):raise ValueError('Original configuration changed')
        code={**context['code_sha256'],'ops/profile_r5_cnn_mac_inference.py':sha(__file__)}
        for p,h in code.items():
            if sha(ROOT/p)!=h:raise ValueError('Original code snapshot differs: '+p)
        data=ROOT/context['data_fold'];allowed={(data/n).resolve() for n in ['readiness_manifest.json','sources/manifest.json','train.npz','validation.npz','inner_validation_metadata.parquet']}
        data_root=(ROOT/'data').resolve()
        def check(p):
            q=Path(p).resolve()
            if q.is_relative_to(data_root) and q not in allowed:raise ValueError('INNER-only query guard: '+str(q))
            if q.suffix.lower() in {'.wav','.flac','.mp3'}:raise ValueError('Audio IO forbidden')
            return q
        def audit(event,a):
            if event in {'socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'}:raise RuntimeError('External action forbidden in CPU inference diagnostic')
            if event=='open' and isinstance(a[0],(str,bytes,os.PathLike)):
                q=check(os.fsdecode(a[0]))
                if q.is_relative_to(data_root) and a[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):raise ValueError('Source-data writes forbidden')
        sys.addaudithook(audit);nl,pr=np.load,pd.read_parquet
        def load(p,*a,**k):check(p);return nl(p,*a,**k)
        def parquet(p,*a,**k):check(p);return pr(p,*a,**k)
        np.load,pd.read_parquet=load,parquet
        f=FoldDataR5(data,42)
        if f.binding!=context['training_binding'] or summary['training_binding']!=f.binding:raise ValueError('Original TRAIN binding differs')
        prior_record=context['prior'];prior=load_shared_prior(folder/prior_record['filename'],expected_binding=f.binding,
            expected_file_sha256=prior_record['file_sha256'],expected_prior_sha256=prior_record['payload_sha256'])
        policy=summary['observation_policy']
        if policy!={'quantization_decimals':None,'endpoint_dropout_probability':0.}:raise ValueError('Original raw policy required')
        checkpoint=torch.load(model_dir/'best.pt',map_location='cpu',weights_only=True)
        model,receipt=neural_r5.make_neural('bg_cnn',context['config']['model_settings']['bg_cnn'])
        neural_r5.restore_into(model,checkpoint,kind='bg_cnn',binding=f.binding,prior_sha=prior_record['payload_sha256'],policy=policy)
        model=model.to(device='cpu',dtype=torch.float32).eval()
        if {str(p.dtype) for p in model.parameters()}!={'torch.float32'} or {str(p.device) for p in model.parameters()}!={'cpu'}:raise ValueError('Full FP32 CPU required')
        original_state=state_sha(model);times['hash_verify_prior_factory_checkpoint_restore_seconds']=time.perf_counter()-t
        stage='INNER_load';t=time.perf_counter();inner,metadata=f.load('inner');times['INNER_load_and_integrity_seconds']=time.perf_counter()-t
        inputs={str((data/n).relative_to(ROOT)):sha(data/n) for n in ['readiness_manifest.json','sources/manifest.json','train.npz','validation.npz','inner_validation_metadata.parquet']}
        if len(inner['db'])!=3000:raise ValueError('Original INNER3000 required')
        timing=[];reference128=None
        stage='CPU_prefix_timing'
        for n in [32,128,512]:
            neural_r5.predict_neural(model,inner['db'][:8],prior,f.binding,policy,device='cpu',batch_size=128)
            t=time.perf_counter();p=neural_r5.predict_neural(model,inner['db'][:n],prior,f.binding,policy,device='cpu',batch_size=128);elapsed=time.perf_counter()-t
            timing.append({'examples':n,'batch_size':128,'warmup_examples':8,'seconds':elapsed,'seconds_per_example':elapsed/n,
                'linear_Test1500_seconds_estimate':elapsed/n*1500,'linear_INNER3000_seconds_estimate':elapsed/n*3000})
            if n==128:reference128=p
            print(json.dumps({'timing':timing[-1],'peak_RSS_bytes':rss()}),flush=True)
        stage='CPU_batch_independence';batch_diffs={}
        for batch in [1,7,128]:
            p=neural_r5.predict_neural(model,inner['db'][:128],prior,f.binding,policy,device='cpu',batch_size=batch)
            diff={k:float(np.max(np.abs(p[k].astype(float)-reference128[k].astype(float)))) for k in p}
            if any(x>1e-5 for x in diff.values()):raise ValueError('Unexpected CPU query-batch dependence: '+str(diff))
            batch_diffs[str(batch)]=diff
        stage='full_INNER_CPU_FP32';t=time.perf_counter()
        cpu=neural_r5.predict_neural(model,inner['db'],prior,f.binding,policy,device='cpu',batch_size=128)
        times['full_INNER3000_FP32_predict_seconds']=time.perf_counter()-t
        ids=metadata.sample_id.astype(str).to_numpy(dtype=str)
        np.savez_compressed(output/'inner_cpu_predictions.npz',**cpu,sample_id=ids)
        with np.load(model_dir/'inner_predictions.npz',allow_pickle=False) as z:gpu={k:z[k] for k in ['share','score','cell_mask']}
        gpu_diff={}
        for k,p in cpu.items():
            if p.shape!=gpu[k].shape:raise ValueError('Saved GPU/CPU prediction shape differs')
            d=np.abs(p.astype(float)-gpu[k].astype(float));gpu_diff[k]={'maximum_absolute_difference':float(d.max()),
                'mean_absolute_difference':float(d.mean()),'p95_absolute_difference':float(np.quantile(d,.95))}
        y=inner['share'].astype(float);cpu_rmse=float(np.sqrt(np.mean((cpu['share'].astype(float)-y)**2)))
        gpu_rmse=float(np.sqrt(np.mean((gpu['share'].astype(float)-y)**2)))
        if abs(gpu_rmse-summary['best_inner_rmse'])>1e-12:raise ValueError('Stored GPU metric/INNER order mismatch')
        w=prepare_query_observations(inner['db'],quantization_decimals=None)['energy_weights']
        pool_error=float(np.max(np.abs((cpu['cell_mask'].astype(float)*w).sum((1,2))-cpu['share'])))
        if pool_error>3e-6:raise ValueError('CPU energy-pooling identity failed')
        if state_sha(model)!=original_state:raise ValueError('Model state changed during inference')
        for p,h in {**sources,**inputs,**code}.items():
            if sha(ROOT/p)!=h:raise ValueError('Frozen source/input changed during inference: '+p)
        if f.opened_roles!=['inner']:raise AssertionError('Unexpected numerical source role opened')
        result={'status':'passed original CNN FP32 Mac CPU inference diagnostic','diagnostic_only':True,
            'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_kind':'bg_cnn','checkpoint_epoch':19,
            'checkpoint_sha256':CHECKPOINT_SHA,'original_backup_archive_sha256':ARCHIVE_SHA,
            'original_backup_payload_files_verified':len(manifest['files']),'training_binding':f.binding,
            'context_sha256':sha(folder/'context.json'),'prior_record':prior_record,'observation_policy':policy,
            'model_factory_receipt':receipt,'model_state_unchanged':True,'timing_prefixes':timing,'timing_seconds':times,
            'estimation_limit':'Linear same-host CPU inference-only estimates include common feature/prior transforms. Actual Test is not read; timing varies with system load and input masks. Full INNER time is measured.',
            'CPU_batch_1_7_128_max_absolute_differences':batch_diffs,'CPU_batch_equivalence_atol':1e-5,'CPU_batch_equivalence_rtol':0,
            'saved_GPU_CUDA_BF16_vs_CPU_FP32':gpu_diff,
            'precision_comparison_limit':'Same FP32 checkpoint/policy/input binding; saved CUDA inference used BF16 autocast, while this run used FP32 CPU. Differences include precision/backend effects; their individual causes were not isolated by a new GPU experiment.',
            'INNER_metrics':{'examples':len(y),'CPU_FP32_RMSE':cpu_rmse,'saved_GPU_BF16_RMSE':gpu_rmse,
                'CPU_minus_GPU_RMSE':cpu_rmse-gpu_rmse,'CPU_MAE':float(np.mean(np.abs(cpu['share'].astype(float)-y))),
                'CPU_bias_pred_minus_truth':float(np.mean(cpu['share'].astype(float)-y))},
            'maximum_CPU_physical_pooling_residual':pool_error,
            'inner_identity':{'sample_id_order_sha256':canonical(ids.tolist()),'raw_db_sha256':array_sha(inner['db']),
                'raw_energy_weights_sha256':array_sha(w),'target_share_sha256':array_sha(inner['share'])},
            'prediction_file':str((output/'inner_cpu_predictions.npz').relative_to(ROOT)),
            'prediction_file_sha256':sha(output/'inner_cpu_predictions.npz'),
            'source_files_sha256':sources,'input_files_sha256':inputs,'code_sha256':code,
            'CPU_full_FP32':True,'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
            'peak_RSS_bytes':rss(),'parameters':sum(p.numel() for p in model.parameters()),
            'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
                'system':os.uname().sysname,'machine':os.uname().machine,'logical_CPUs':os.cpu_count(),
                'thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}},
            'labels_used_only_for_final_metrics_not_prediction_or_timing':True,
            'TRAIN_prior_refit_neural_training_postprocessing_selection_or_weight_update_performed':False,
            'Test_OUTER_GPU_SSH_external_transmission_or_submission_used':False,
            'elapsed_seconds':time.perf_counter()-start_all}
        write(report,result)
        print(json.dumps({'report':str(report),'metrics':result['INNER_metrics'],'timing_seconds':times,'peak_RSS_bytes':rss()},indent=2))
    except BaseException as e:
        if not report.exists():write(report,{'status':'failed CPU inference diagnostic','stage':stage,'error_type':type(e).__name__,
            'error':str(e),'completed_timings':times,'peak_RSS_bytes':rss(),'partial_predictions_reusable':False,
            'automatic_retry_performed':False,'Test_OUTER_GPU_SSH_submission_used':False})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,default=ROOT/'reports/R5_CNN_MAC_INFERENCE_PROFILE.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'reports/R5_CNN_MAC_INFERENCE_PROFILE')
    a=p.parse_args();run(a.report,a.output_dir)
