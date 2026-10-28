"""Discard-only CUDA microbenchmarks on train/inner-validation CV inputs.

No models, fitted context, predictions or checkpoints are saved. Foundation
benchmarks are opt-in and use checksum-pinned local weights without downloads.
"""
from pathlib import Path
import argparse,gc,json,math,platform,time
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from pipeline.models import make_model
from pipeline.mixtures import fixed_features,cnn_channels,foundation_features
ROOT=Path(__file__).resolve().parents[1]
KINDS=('tabm','cnn','fdy','itransformer')
LIMITS=[
    'A measurement-derived planning upper allowance is not a guaranteed wall-clock upper bound.',
    '100 epochs assumes no early stopping; real runs may stop earlier.',
    'Per-step wall time includes NumPy batch indexing, synchronous CPU-to-device transfer, forward, backward, gradient clipping and Adam step.',
    'Full train plus inner-validation prediction every epoch is estimated from sampled evaluation batches; full-dataset prediction was not executed.',
    'CPU feature preparation is measured once; OS cache, CPU contention, memory pressure, loading, checkpoint IO, scheduling, other jobs and long-run thermal effects can change timing.',
    'No estimate here proves convergence, generalization or a global optimum; contrastive training is disabled in this baseline.',
    'Summed neural allowance covers one fold and one seed only; LightGBM, optional foundation context preparation and CV orchestration are separate.',
]

def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)

def memory(device):
    if device.type!='cuda':return {'peak_allocated_bytes':None,'peak_reserved_bytes':None}
    return {'peak_allocated_bytes':int(torch.cuda.max_memory_allocated(device)),
            'peak_reserved_bytes':int(torch.cuda.max_memory_reserved(device))}

def input_bytes(inputs):return sum(x.nbytes for x in inputs) if isinstance(inputs,tuple) else inputs.nbytes

def forward(model,kind,inputs,ids,device):
    if kind=='tabm':
        logits=model(torch.from_numpy(inputs[ids]).to(device));z=logits.sigmoid().mean(1)
        return z[:,0],logits,None
    c,w=inputs;weight=torch.from_numpy(w[ids]).to(device)
    pred,logit,mask=model(torch.from_numpy(c[ids]).to(device),weight)
    return pred,logit,(mask,weight)

def benchmark_neural(kind,train,inner,args,device,development_fixture=False):
    if device.type!='cuda' and not development_fixture:raise ValueError('Real preflight requires CUDA')
    if device.type=='cuda':gc.collect();torch.cuda.empty_cache()
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    version=getattr(args,'feature_version',1)
    t0=time.perf_counter();inputs=fixed_features(train['db'],version=version) if kind=='tabm' else cnn_channels(train['db'],version=version)
    feature_seconds=time.perf_counter()-t0
    t0=time.perf_counter();inner_features=fixed_features(inner['db'],version=version) if kind=='tabm' else cnn_channels(inner['db'],version=version)
    inner_feature_seconds=time.perf_counter()-t0
    feature_bytes=input_bytes(inputs)+input_bytes(inner_features);del inner_features
    width={'tabm':256,'cnn':32,'fdy':32,'itransformer':96}[kind]
    architecture={'architecture_version':3 if version==3 else 2,'dropout':.1,'model_width':width}
    if version==3:architecture.update({'n_features':1258} if kind=='tabm' else {'input_channels':6})
    model=make_model(kind,width=256,k=16,**architecture).to(device).train()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
    y=torch.from_numpy(train['share']).to(device);z=torch.from_numpy(train['presence']).to(device)
    q=torch.from_numpy(train['cell_share']).to(device) if kind!='tabm' else None
    batch=min(args.batch,len(y));durations=[];eval_durations=[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for i in range(args.warmup+args.steps):
        ids=rng.choice(len(y),batch,replace=False);sync(device);start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        pred,logit,cell=forward(model,kind,inputs,ids,device)
        if kind=='tabm':
            loss=F.mse_loss(logit[:,:,0].sigmoid(),y[ids,None].expand(-1,logit.shape[1]))+.03*F.binary_cross_entropy_with_logits(logit[:,:,1],z[ids,None].expand(-1,logit.shape[1]))
        else:
            mask,w=cell;loss=F.mse_loss(pred,y[ids])+.03*F.binary_cross_entropy_with_logits(logit,z[ids])+.03*((mask-q[ids]).square()*w).sum((1,2)).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite microbenchmark loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();sync(device)
        if i>=args.warmup:durations.append(time.perf_counter()-start)
    model.eval()
    with torch.inference_mode():
        for i in range(args.warmup+args.steps):
            ids=rng.choice(len(y),batch,replace=False);sync(device);start=time.perf_counter();forward(model,kind,inputs,ids,device);sync(device)
            if i>=args.warmup:eval_durations.append(time.perf_counter()-start)
    mem=memory(device);train_batches=math.ceil(len(y)/batch);eval_batches=math.ceil(len(y)/batch)+math.ceil(len(inner['share'])/batch)
    # Maximum observed batch + 25% planning headroom, not a strict bound.
    training_epoch_allowance=max(durations)*train_batches*1.25
    evaluation_epoch_allowance=max(eval_durations)*eval_batches*1.25
    result={'kind':kind,'status':'CPU development fixture only' if development_fixture else 'CUDA microbenchmark completed',
        'CUDA_measured':device.type=='cuda','random_initialization':True,'discarded_after_benchmark':True,'model_saved':False,
        'architecture':architecture,'feature_version':version,'parameters':sum(p.numel() for p in model.parameters()),'optimizer':'Adam','contrastive_weight':0.,
        'batch':batch,'warmup_steps':args.warmup,'measured_training_steps':args.steps,'measured_eval_steps':args.steps,
        'training_step_seconds':{'mean':float(np.mean(durations)),'median':float(np.median(durations)),'p95':float(np.quantile(durations,.95)),'max':max(durations)},
        'examples_per_second':batch/float(np.mean(durations)),'eval_step_seconds_mean':float(np.mean(eval_durations)),
        **mem,'feature_CPU_seconds':{'train':feature_seconds,'inner_validation':inner_feature_seconds},'feature_CPU_bytes':feature_bytes,
        'estimated_epoch_training_allowance_seconds':training_epoch_allowance,'estimated_epoch_train_plus_inner_prediction_allowance_seconds':evaluation_epoch_allowance,
        'estimated_100epoch_per_fold_allowance_seconds':100*(training_epoch_allowance+evaluation_epoch_allowance)+feature_seconds+inner_feature_seconds,
        'estimate_method':'100 * (ceil(ntrain/batch)*max_measured_train_step + [ceil(ntrain/batch)+ceil(ninner/batch)]*max_measured_eval_step) * 1.25 + one-time feature CPU time',
        'estimate_is_guaranteed_upper_bound':False,'development_fixture_estimate_not_for_GPU_rental':development_fixture}
    del model,optimizer,inputs,y,z,q
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return result

def benchmark_foundations(train,inner,args,device):
    from pipeline.cv_foundations import CHECKPOINTS,make_predictor,sha256
    results=[];n=min(args.foundation_context,len(train['share']));ids=np.random.default_rng(args.seed).choice(len(train['share']),n,replace=False)
    X=foundation_features(train['db'][ids],version=args.feature_version);y=train['share'][ids];query=foundation_features(inner['db'][:args.foundation_queries],version=args.feature_version)
    for kind in CHECKPOINTS:
        filename,expected=CHECKPOINTS[kind];checkpoint=ROOT/'models/pretrained'/filename
        if not checkpoint.is_file() or checkpoint.is_symlink() or sha256(checkpoint)!=expected:
            results.append({'kind':kind,'status':'not executed: pinned local checkpoint missing or changed','checkpoint_sha256':expected});continue
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
        spec=SimpleNamespace(kind=kind,members=4,threads=args.threads,seed=args.seed,device=str(device))
        sync(device);start=time.perf_counter();predictor=make_predictor(spec,X,y,checkpoint);sync(device);fit_seconds=time.perf_counter()-start
        sync(device);start=time.perf_counter();prediction=predictor(query);sync(device);query_seconds=time.perf_counter()-start
        if not np.isfinite(prediction).all():raise ValueError('Nonfinite foundation timing probe')
        results.append({'kind':kind,'status':'CUDA frozen-context benchmark completed','context_rows':n,'features':X.shape[1],'feature_version':args.feature_version,'members':4,'estimator_batch_size':1,'query_rows':len(query),'fit_seconds':fit_seconds,'query_seconds':query_seconds,**memory(device),'checkpoint_sha256':expected,'gradient_training':False,'model_or_predictions_saved':False})
        del predictor,prediction;gc.collect();torch.cuda.empty_cache()
    return results

def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'reports/gpu_preflight.json')
    p.add_argument('--device',default='cuda');p.add_argument('--batch',type=int,default=128);p.add_argument('--warmup',type=int,default=3);p.add_argument('--steps',type=int,default=20)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--feature-version',type=int,choices=[1,3],default=1)
    p.add_argument('--foundation',action='store_true');p.add_argument('--foundation-context',type=int,default=4096);p.add_argument('--foundation-queries',type=int,default=128)
    return p

def run(args):
    out=args.output.resolve()
    if out.suffix!='.json' or any(out.is_relative_to(ROOT/p) for p in ['data','models','pipeline']):raise ValueError('Output must be a separate JSON report, outside data/models/pipeline')
    if min(args.batch,args.warmup,args.steps,args.threads,args.foundation_context,args.foundation_queries)<1:raise ValueError('Benchmark budgets must be positive')
    report={'schema_version':1,'status':'not executed','CUDA_measured':False,'full_training_executed':False,'model_saved':False,
        'test_accessed':False,'outer_validation_accessed':False,'calibration_accessed':False,'sealed_accessed':False,
        'data_package_verified':False,'platform':platform.platform(),'torch_version':torch.__version__,
        'requested_settings':{'batch':args.batch,'warmup':args.warmup,'measured_steps':args.steps,'optimizer':'Adam','dropout':.1,'feature_version':args.feature_version,'contrastive_weight':0.,'max_epochs_for_estimate':100},
        'models':[],'foundation':{'status':'skipped; --foundation not enabled; no pinned weights opened'},'limitations':LIMITS}
    def save():out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,default=str))
    device=torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():
        report['status']='not executed: real CUDA unavailable or a CPU device was requested';report['exit_code']=2;save();return 2
    torch.set_num_threads(args.threads)
    report['device']={'name':torch.cuda.get_device_name(device),'total_memory_bytes':torch.cuda.get_device_properties(device).total_memory,'CUDA_runtime':torch.version.cuda}
    # This verifier opens train and inner validation only, including no outer byte hash.
    from pipeline.cv_foundations import verify_cv_package
    data,gate,verification=verify_cv_package(args.data_dir)
    if 'physics' in str(gate.get('version','')) and args.feature_version!=3:raise ValueError('Physics V3 package requires --feature-version 3')
    report['data_package_verified']=True;report['data_package_hash']=gate['data_package_hash'];report['fold']=gate['fold']
    report['verification']=verification;report['examples']={k:len(data[k]['share']) for k in ['train','validation']}
    code=0
    for kind in KINDS:
        try:report['models'].append(benchmark_neural(kind,data['train'],data['validation'],args,device))
        except torch.cuda.OutOfMemoryError:
            report['models'].append({'kind':kind,'status':'CUDA out of memory','CUDA_measured':False});gc.collect();torch.cuda.empty_cache();code=3
        save()
    if args.foundation:
        try:
            report['foundation']=benchmark_foundations(data['train'],data['validation'],args,device)
            if any('not executed' in row['status'] for row in report['foundation']):code=3
        except Exception as exc:
            report['foundation']={'status':'failed optional benchmark','error':{'type':type(exc).__name__,'message':str(exc)}};code=3
    report['CUDA_measured']=bool(report['models']) and all(m.get('CUDA_measured') is True for m in report['models'])
    report['estimated_100epoch_four_neural_models_per_fold_allowance_seconds']=sum(m.get('estimated_100epoch_per_fold_allowance_seconds',0) for m in report['models']) if report['CUDA_measured'] else None
    report['status']='completed discard-only CUDA preflight' if code==0 else 'partial preflight; inspect failed or missing candidates'
    report['exit_code']=code;save();return code

if __name__=='__main__':
    args=build_parser().parse_args()
    try:
        exit_code=run(args)
        print(('CUDA unavailable; no measurements executed. ' if exit_code==2 else 'Preflight finished. ')+'Report: '+str(args.output)+'; exit='+str(exit_code))
        raise SystemExit(exit_code)
    except Exception as exc:
        # A failed probe is never presented as a timing measurement.
        if args.output.suffix=='.json' and not any(args.output.resolve().is_relative_to(ROOT/p) for p in ['data','models','pipeline']):
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps({'schema_version':1,'status':'failed preflight; no usable CUDA estimate','CUDA_measured':False,'full_training_executed':False,'model_saved':False,'test_accessed':False,'outer_validation_accessed':False,'calibration_accessed':False,'sealed_accessed':False,'error':{'type':type(exc).__name__,'message':str(exc)},'exit_code':3},indent=2))
        raise
