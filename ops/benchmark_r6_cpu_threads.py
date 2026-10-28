"""Short sequential fresh-process synthetic CPU throughput fixture only.

No acoustic dataset, validation label, checkpoint or model performance is read.
Each synthetic model is discarded after two warmups and eight timed updates.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/R6_CPU_THREAD_HARDWARE_FIXTURE.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def case(threads):
    # The parent has set native thread environment before this fresh import.
    import numpy as np
    import torch
    import resource
    sys.path.insert(0,str(ROOT))
    from pipeline.relative_models_r6_v2 import TinyMaskCNNv2
    torch.set_num_threads(threads);torch.set_num_interop_threads(1);torch.manual_seed(20261026)
    model=TinyMaskCNNv2(input_channels=8,width=32,blocks=3,dropout=.1,global_context=True).train()
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
    x=torch.randn(128,8,15,25);w=torch.rand(128,15,25);w=w/w.sum((1,2),keepdim=True)
    qtrue=torch.rand(128,15,25);y=(qtrue*w).sum((1,2));z=(torch.rand(128)>.2).float()
    durations=[]
    for i in range(10):
        start=time.perf_counter();opt.zero_grad(set_to_none=True);pred,logit,q=model(x,w)
        loss=((pred-y).square()+.25*((q-qtrue).square()*w).sum((1,2))+.05*torch.nn.functional.binary_cross_entropy_with_logits(logit,z,reduction='none')).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Synthetic nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        elapsed=time.perf_counter()-start
        if i>=2:durations.append(elapsed)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'threads':threads,'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'warmup_updates':2,'timed_updates':8,'batch_size':128,'median_step_seconds':float(np.median(durations)),
        'mean_step_seconds':float(np.mean(durations)),'minimum_step_seconds':float(np.min(durations)),
        'maximum_step_seconds':float(np.max(durations)),'all_step_seconds':durations,
        'examples_per_second_from_median':128/float(np.median(durations)),
        'peak_process_RSS_bytes':int(rss if sys.platform=='darwin' else rss*1024),
        'parameters':sum(p.numel() for p in model.parameters()),'torch_version':torch.__version__,
        'numpy_version':np.__version__,'all_losses_and_gradients_finite':True,
        'no_dataset_or_checkpoint_opened':True,'no_weights_saved':True}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=int,choices=[1,2,4]);a=p.parse_args()
    if a.case is not None:
        print(json.dumps(case(a.case)));return
    if OUT.exists():raise FileExistsError('Fresh hardware fixture output required')
    rows=[]
    for threads in [1,2,4]:
        env=os.environ.copy();env['OMP_NUM_THREADS']=str(threads);env['MKL_NUM_THREADS']=str(threads);env['OPENBLAS_NUM_THREADS']='1'
        proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--case',str(threads)],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
        rows.append(json.loads(proc.stdout));print(json.dumps({'completed_threads':threads,'median_step_seconds':rows[-1]['median_step_seconds']}),flush=True)
    best=min(rows,key=lambda v:v['median_step_seconds'])
    report={'status':'PASS','scope':'synthetic CPU hardware fixture, not source training or validation performance',
        'sequential_fresh_processes':True,'settings':{'input_shape':[128,8,15,25],'width':32,'blocks':3,'global_context':True,'dropout':.1,
            'optimizer':'Adam','learning_rate':.001,'weight_decay':.0003,'gradient_clip':1.,'loss_weights':{'share':1.,'cell':.25,'presence':.05},'seed':20261026},
        'measurements':rows,'fastest_single_process_threads':best['threads'],
        'concurrency_not_measured':True,'data_or_model_performance_read':False,
        'limitations':['Eight timed updates are a short operational sample, not an epoch or full training benchmark.',
            'No concurrent two-worker speedup is measured. CPU contention, system memory pressure and thermal state can change timings.',
            'Synthetic repeated batch excludes input preparation, real INNER evaluation, hashing and checkpoint overhead.',
            'Thread changes can change floating-point trajectories; no bitwise-prefix claim relative to previous runs.'],
        'source_sha256':{'ops/benchmark_r6_cpu_threads.py':sha(__file__),'pipeline/relative_models_r6_v2.py':sha(ROOT/'pipeline/relative_models_r6_v2.py')}}
    with OUT.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','fastest_single_process_threads':best['threads'],'report':str(OUT)},indent=2))

if __name__=='__main__':main()
