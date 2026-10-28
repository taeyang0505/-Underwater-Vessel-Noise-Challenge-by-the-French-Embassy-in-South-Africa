"""Predeclared physical-nuisance sensitivity of fixed epoch19 CNN, INNER only.

No fitting, selection, source edits, Test/OUTER/GPU/SSH or submission. This
stress study is not a reconstruction of the official hidden data generator.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
PREFIX=ROOT/'reports/R5_PHYSICAL_ROBUSTNESS_AUDIT'
SPEC=PREFIX.with_suffix('.specification.json')
SPEC_SHA='ef294c46c70bc97f161dc9f33d12654bb884ba87d1d82842cddd8f1270398a69'
PROFILE=ROOT/'reports/R5_CNN_MAC_INFERENCE_PROFILE.json'
PROFILE_SHA='9f91a8bf85c2b7ba33ea85368269073a2f6d5c53a499c87dc23f0f3b67ccce80'
CHECKPOINT_SHA='aa222697275306cf93e5778a707f7ee5fd75610ae0a0bf643d654854ee96d46f'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def array_sha(a):
    a=np.asarray(a);h=hashlib.sha256(json.dumps({'shape':list(a.shape),'dtype':str(a.dtype)},sort_keys=True,separators=(',',':')).encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()
def state_sha(model):return hashlib.sha256(json.dumps({k:array_sha(v.detach().cpu().numpy()) for k,v in model.state_dict().items()},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def weights(db):
    x=np.asarray(db,dtype='float64');o=np.isfinite(x);e=np.where(o,10**((x-np.nanmax(x,(1,2),keepdims=True))/10),0)
    if not (o.any(2).sum(1)>=3).all() or not (e[o]>0).all():raise ValueError('Invalid observed energy/support')
    return e/e.sum((1,2),keepdims=True)

def transform(db,q,y,scenario):
    x=np.array(db,copy=True);truth=np.array(q,copy=True);observed=np.isfinite(x);kind=scenario['kind'];applied=np.zeros(len(x),bool);protected=np.zeros(len(x),bool)
    if kind=='common_gain':x[observed]+=np.float32(scenario['gain_db']);applied[:]=True
    elif kind=='common_tilt':
        curve=scenario['tilt_db']*np.linspace(-1,1,25);x=np.where(observed,x.astype('float64')+curve[None,None,:],np.nan).astype('float32');applied[:]=True
    elif kind=='endpoint_drop':
        t=scenario['minute_index'];visible=observed[:,t].any(1);protected=visible&(observed.any(2).sum(1)-1<3);applied=visible&~protected;x[applied,t,:]=np.nan;truth[applied,t,:]=0
    elif kind!='identity':raise ValueError('Unknown fixed scenario')
    w=weights(x);recovered=(w*truth.astype('float64')).sum((1,2))
    target=np.asarray(y,dtype='float64').copy() if kind in {'identity','common_gain'} else recovered
    z=(target>0).astype('float32');err=float(np.max(abs(target-recovered)))
    if err>3e-6 or (truth[~np.isfinite(x)]!=0).any() or ((truth<0)|(truth>1)).any():raise ValueError('Transformed component/energy identity failed')
    if kind in {'identity','common_gain','common_tilt'} and not np.array_equal(truth,q):raise ValueError('Per-cell q changed under common response')
    return x,truth,target,z,w,{'applied_windows':int(applied.sum()),'protected_below3minutes':int(protected.sum()),'q_max_absolute_change':float(np.max(abs(truth-q))),'target_max_absolute_change':float(np.max(abs(target-y))),'target_mean_change':float(np.mean(target-y)),'physical_target_identity_max_absolute_error':err,'positive_to_zero_after_transform':int(np.count_nonzero((y>0)&(target==0))),'observed_minutes_min':int(np.isfinite(x).any(2).sum(1).min())}

def auc(y,p):
    y=np.asarray(y,bool);n1=int(y.sum());n0=len(y)-n1
    if n1==0 or n0==0:return None
    order=np.argsort(p,kind='mergesort');values=p[order];rank=np.empty(len(y),float);starts=np.r_[0,np.flatnonzero(values[1:]!=values[:-1])+1];ends=np.r_[starts[1:],len(y)]
    for lo,hi in zip(starts,ends):rank[order[lo:hi]]=(lo+1+hi)/2
    return float((rank[y].sum()-n1*(n1+1)/2)/(n1*n0))
def metrics(y,z,p,s):
    e=p.astype(float)-y;return {'n':len(y),'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(abs(e))),'bias_pred_minus_truth':float(e.mean()),'addition_presence_AUC':auc(z,s),'target_mean':float(y.mean()),'positive_rows':int(np.count_nonzero(z))}
def grouped(y,z,p,s,g):return {str(k):metrics(y[g==k],z[g==k],p[g==k],s[g==k]) for k in sorted(set(g.tolist()))}

def install_guard(folder):
    data=(ROOT/'data').resolve();allowed={(folder/n).resolve() for n in ['readiness_manifest.json','sources/manifest.json','train.npz','validation.npz','inner_validation_metadata.parquet']}
    def audit(event,args):
        if event in {'socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'}:raise RuntimeError('No external actions in CPU robustness audit')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();writing=args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
            if p.name.lower() in {'test.csv','outer_validation.npz','outer_validation_metadata.parquet'} or p.suffix.lower() in {'.wav','.flac','.mp3'}:raise ValueError('Test/OUTER/audio forbidden')
            if p.is_relative_to(data) and (p not in allowed or writing):raise ValueError('Only original INNER and integrity metadata permitted')
            if writing and (p.parent!=PREFIX.parent or not p.name.startswith(PREFIX.name)):raise ValueError('Only separate robustness audit outputs may be written')
    sys.addaudithook(audit)

def run():
    sys.dont_write_bytecode=True;start=time.perf_counter()
    if sha(SPEC)!=SPEC_SHA or sha(PROFILE)!=PROFILE_SHA:raise ValueError('Predeclared spec/profile changed')
    for suffix in ['.json','.md','.npz']:
        if PREFIX.with_suffix(suffix).exists():raise FileExistsError('No robustness rerun/overwrite')
    spec=json.loads(SPEC.read_text());profile=json.loads(PROFILE.read_text());folder=ROOT/'data/synthetic/group_cv_passage_r4_v1/fold_00';install_guard(folder)
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    if str(torch.__version__)!=profile['environment']['torch'] or np.__version__!=profile['environment']['numpy']:raise ValueError('Use the same fixed CPU runtime as the original profile')
    from pipeline.neural_r5 import make_neural,restore_into,predict_neural
    from pipeline.fold_data_r5 import FoldDataR5
    from pipeline.grid_inputs_r5 import load_shared_prior
    model_dir=ROOT/'ops/server_backups/r5_gpu_r1/restored_first_cnn_v1/fold_00/preprocessing_pilot/raw/bg_cnn';backup=model_dir.parents[2]
    context=json.loads((backup/'context.json').read_text());summary=json.loads((model_dir/'summary.json').read_text())
    if summary['status']!='completed' or summary['fixture_only'] is not False or summary['best_epoch']!=19 or sha(model_dir/'best.pt')!=CHECKPOINT_SHA:raise ValueError('Expected fixed completed original epoch19 CNN')
    snapshots={str(SPEC):SPEC_SHA,str(PROFILE):PROFILE_SHA,str(Path(__file__)):sha(__file__)}
    for key in ['source_files_sha256','input_files_sha256','code_sha256']:
        for name,h in profile[key].items():
            p=ROOT/name
            if sha(p)!=h:raise ValueError('Frozen source/input/code differs: '+name)
            snapshots[str(p)]=h
    f=FoldDataR5(folder,42)
    if f.binding!=profile['training_binding'] or context['training_binding']!=f.binding:raise ValueError('Fixed CNN/source binding mismatch')
    record=context['prior'];prior=load_shared_prior(backup/record['filename'],expected_binding=f.binding,expected_file_sha256=record['file_sha256'],expected_prior_sha256=record['payload_sha256'])
    checkpoint=torch.load(model_dir/'best.pt',map_location='cpu',weights_only=True);model,receipt=make_neural('bg_cnn',context['config']['model_settings']['bg_cnn']);observation=summary['observation_policy']
    if observation!={'quantization_decimals':None,'endpoint_dropout_probability':0.}:raise ValueError('Original raw observation policy required')
    restore_into(model,checkpoint,kind='bg_cnn',binding=f.binding,prior_sha=record['payload_sha256'],policy=observation);model=model.to(device='cpu',dtype=torch.float32).eval()
    before_state=state_sha(model);inner,meta=f.load('inner');db=inner['db'];q=inner['cell_share'];y=inner['share'].astype(float)
    for a in [db,q,y]:a.setflags(write=False)
    original_arrays={k:array_sha(a) for k,a in [('db',db),('q',q),('y',y)]};ids=meta.sample_id.astype(str).to_numpy(dtype=str);refs=meta.reference_group.astype(str).to_numpy();sites=meta.site.astype(str).to_numpy()
    if len(y)!=3000:raise ValueError('Original INNER3000 required')
    results=[];arrays={'sample_id':ids};base_pred=None;base_y=None;neural_fit=False
    predict_neural(model,db[:8],prior,f.binding,observation,device='cpu',batch_size=128)
    for scenario in spec['scenarios']:
        name=scenario['name'];t=time.perf_counter();print(json.dumps({'scenario':name,'status':'started','device':'cpu','fit':False}),flush=True)
        try:
            x,truth,target,z,w,audit=transform(db,q,y,scenario)
            pred=predict_neural(model,x,prior,f.binding,observation,device='cpu',batch_size=128)
            if state_sha(model)!=before_state:raise ValueError('Model state changed during stress inference')
            pool_error=float(np.max(abs((pred['cell_mask'].astype(float)*w).sum((1,2))-pred['share'])))
            if pool_error>3e-6:raise ValueError('Predicted physical pooling failed')
            row={'scenario':scenario,'status':'completed','metrics':metrics(target,z,pred['share'],pred['score']),'by_reference':grouped(target,z,pred['share'],pred['score'],refs),'by_site':grouped(target,z,pred['share'],pred['score'],sites),'transform_audit':audit,'predicted_physical_pooling_max_error':pool_error,'transformed_db_sha256':array_sha(x),'transformed_q_sha256':array_sha(truth),'transformed_target_sha256':array_sha(target),'prediction_cell_mask_sha256':array_sha(pred['cell_mask']),'seconds':time.perf_counter()-t}
            if name=='identity':
                base_pred=pred['share'].astype(float);base_y=target.copy()
                if abs(row['metrics']['RMSE']-profile['INNER_metrics']['CPU_FP32_RMSE'])>1e-12:raise ValueError('Original CPU identity metric failed exact replay')
            if base_pred is not None:
                dp=pred['share'].astype(float)-base_pred;dy=target-base_y;row['vs_identity']={'prediction_change_MAE':float(abs(dp).mean()),'prediction_change_max_abs':float(abs(dp).max()),'target_change_MAE':float(abs(dy).mean()),'change_tracking_residual_RMSE':float(np.sqrt(np.mean((dp-dy)**2))),'RMSE_difference':row['metrics']['RMSE']-profile['INNER_metrics']['CPU_FP32_RMSE']}
            for k,a in [('share',pred['share']),('score',pred['score']),('target',target),('presence',z)]:arrays[name+'__'+k]=a
            results.append(row);print(json.dumps({'scenario':name,'status':'completed','metrics':row['metrics'],'seconds':row['seconds']}),flush=True)
        except Exception as e:
            results.append({'scenario':scenario,'status':'failed','error_type':type(e).__name__,'error':str(e),'seconds':time.perf_counter()-t});print(json.dumps({'scenario':name,'status':'failed','error':str(e)}),flush=True)
    unchanged=True
    for path,h in snapshots.items():
        if sha(path)!=h:unchanged=False
    if original_arrays!={k:array_sha(a) for k,a in [('db',db),('q',q),('y',y)]}:unchanged=False
    status='completed' if unchanged and all(v['status']=='completed' for v in results) else 'completed_with_failed_checks'
    result={'format':'r5_physical_robustness_audit_v1','status':status,'created_at_utc':now(),'specification_sha256':SPEC_SHA,'code_sha256':{'ops/audit_r5_physical_robustness.py':sha(__file__),**profile['code_sha256']},'checkpoint_sha256':CHECKPOINT_SHA,'CPU_profile_sha256':PROFILE_SHA,'training_binding':f.binding,'source_files_sha256':profile['source_files_sha256'],'input_files_sha256':profile['input_files_sha256'],'original_arrays_sha256':original_arrays,'model_state_sha256_before':before_state,'model_state_sha256_after':state_sha(model),'all_frozen_source_and_original_arrays_unchanged':unchanged,'all_scenarios_reported':len(results)==len(spec['scenarios']),'scenarios':results,'CPU_threads':torch.get_num_threads(),'CPU_interop_threads':torch.get_num_interop_threads(),'batch_size':128,'precision':'CPU float32','fit_or_policy_selection_performed':False,'Test_OUTER_GPU_SSH_external_or_submission_used':False,'not_official_hidden_generator_reconstruction':True,'opened_native_roles':f.opened_roles,'elapsed_seconds':time.perf_counter()-start}
    np.savez_compressed(PREFIX.with_suffix('.npz'),**arrays);write_new(PREFIX.with_suffix('.json'),result)
    lines=['# Fixed-CNN physical robustness stress','', 'This is a predeclared own-INNER diagnostic, not a reconstruction of the official hidden generator. There was no fitting, policy selection or Test/OUTER/GPU use. All nine scenarios, including failures, are reported.','', '| Scenario | RMSE | MAE | Bias | AUC | RMSE minus identity |','|---|---:|---:|---:|---:|---:|']
    for row in results:
        name=row['scenario']['name']
        if row['status']!='completed':lines.append(f'| {name} | FAILED | | | | |');continue
        m=row['metrics'];a='undefined' if m['addition_presence_AUC'] is None else f'{m["addition_presence_AUC"]:.6f}';lines.append(f'| {name} | {m["RMSE"]:.6f} | {m["MAE"]:.6f} | {m["bias_pred_minus_truth"]:+.6f} | {a} | {row["vs_identity"]["RMSE_difference"]:+.6f} |')
    lines+=['','Common-gain cases preserve known q/y. Common-tilt cases preserve per-cell q but recompute pooled y after energy reweighting. Endpoint masks zero hidden q and recompute remaining y, including legitimate positive-to-zero transitions. Windows with fewer than three remaining observed minutes are protected by a rule independent of labels.','', 'Per-reference AUC is undefined when that group contains only positives or only zero controls. The JSON uses null, rather than inventing a numerical AUC. Every case uses the same fixed model, prior and CPU settings.','', 'Failure of these stress tests would establish sensitivity to the declared intervention on this synthetic population; it would not identify the actual hidden Test perturbation or explain a numerical fraction of the leaderboard error. Passing them would not establish external validity.','', '[Predeclared specification](R5_PHYSICAL_ROBUSTNESS_AUDIT.specification.json) · [Full metrics and source hashes](R5_PHYSICAL_ROBUSTNESS_AUDIT.json)','']
    with PREFIX.with_suffix('.md').open('x') as f:f.write('\n'.join(lines))
    print(json.dumps({'report':str(PREFIX.with_suffix('.json')),'status':status,'source_unchanged':unchanged,'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    return result

def self_test():
    db=np.full((3,15,25),40,dtype='float32');q=np.full_like(db,.3);y=(weights(db)*q).sum((1,2));db[1,3:]=np.nan;q[1,3:]=0;y=(weights(db)*q).sum((1,2))
    for g in [-12,-6,6,12]:
        x,qq,yy,z,w,a=transform(db,q,y,{'kind':'common_gain','gain_db':g});assert np.array_equal(qq,q) and np.array_equal(yy,y) and a['physical_target_identity_max_absolute_error']<1e-12
    for t in [-4,4]:
        _,qq,yy,_,_,a=transform(db,q,y,{'kind':'common_tilt','tilt_db':t});assert np.array_equal(qq,q) and a['physical_target_identity_max_absolute_error']==0
    _,qq,_,_,_,a=transform(db,q,y,{'kind':'endpoint_drop','minute_index':0});assert a['protected_below3minutes']==1 and np.array_equal(qq[1],q[1])
    return {'passed':True,'scope':'tiny synthetic physical transform fixture only','model_or_source_loaded':False,'code_sha256':sha(__file__)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:print(json.dumps(self_test(),indent=2))
    else:run()
if __name__=='__main__':main()
