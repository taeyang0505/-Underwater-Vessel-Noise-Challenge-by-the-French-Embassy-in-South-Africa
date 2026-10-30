"""Detached sequential CPU8 R16 CatBoost supervisor. No Test/final policy."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline.catboost_r16_cpu8_production import (RUN,OPS,DATA,BENCHMARK_CACHE,pins,sha,write,
    timestamp,check_plan,verify_completed,fit_pair_cpu8,disk_guard)


def plan():
    if (OPS/'PLAN.json').exists():raise FileExistsError('Plan already frozen')
    if RUN.exists():raise FileExistsError('Fresh production run required')
    disk_guard();RUN.mkdir(parents=True);OPS.mkdir(parents=True,exist_ok=True)
    p={'format':'r16_catboost_cpu8_production_plan_v1','status':'fixed_before_training',
        'created_at_kst':timestamp(),'CPU_threads':8,'maximum_trees':1500,'early_stopping_rounds':40,
        'source_sha256':pins(),'data_manifest_sha256':sha(DATA/'manifest.json'),
        'benchmark_cache_manifest_sha256':sha(BENCHMARK_CACHE/'manifest.json'),
        'jobs':[{'fold':f,'regime':r} for f in range(5) for r in ('D0','H0')],
        'feature_counts':{'regression':5160,'classification':2033},
        'checkpoint_selection':'ownfoldCV1000','ensemble_fit':'sharedlocalfit2400/400backgrounds afterbothfits',
        'memory_guard_GiB':12,'minimum_free_disk_GiB':3,'concurrency':1,
        'source_data_splits_unchanged':True,'original_CPU2_defaults_unchanged':True,
        'resume_scope':'completedpairs only, exact SHA/source/data; no failedpartial reuse',
        'Test_used':False,'locked_report_used':False,'CSV_created':False}
    write(OPS/'PLAN.json',p);write(OPS/'state.json',{'phase':'planned','plan_sha256':sha(OPS/'PLAN.json'),
        'updated_at_kst':timestamp(),'completed_pairs':[]});return p


def supervise():
    p=check_plan();lock=OPS/'supervisor.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    completed=[];began=time.perf_counter()
    try:
        for job in p['jobs']:
            f,r=job['fold'],job['regime'];folder=RUN/r/f'fold_{f:02d}'
            if (folder/'summary.json').exists():
                s=verify_completed(folder,p)
                if s.get('disposable_smoke_only'):raise ValueError('Smoke cannot resume production')
                completed.append({**job,'summary_sha256':sha(folder/'summary.json')});continue
            if folder.exists():raise FileExistsError('Incomplete pair exists; explicit fresh run required')
            if (RUN/'PAUSE_AFTER_PAIR').exists():
                write(OPS/'state.json',{'phase':'paused_at_pair_boundary','completed_pairs':completed,
                    'next_job':job,'updated_at_kst':timestamp(),'explicit_resume_required':True});return 20
            disk_guard();write(OPS/'state.json',{'phase':'training','current_job':job,
                'completed_pairs':completed,'updated_at_kst':timestamp(),'pid':os.getpid()})
            with (OPS/f'{r}_fold_{f:02d}.log').open('a') as log:
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--fold',str(f),'--regime',r],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                write(OPS/'worker.json',{'pid':child.pid,'job':job,'updated_at_kst':timestamp()})
                code=child.wait()
            if code!=0:raise RuntimeError(f'Worker{r}/Fold{f} exited{code}; inspectlog, noautomaticrestart')
            s=verify_completed(folder,p);completed.append({**job,'summary_sha256':sha(folder/'summary.json')})
            write(OPS/'state.json',{'phase':'pair_completed','last_job':job,'completed_pairs':completed,
                'updated_at_kst':timestamp(),'seconds':time.perf_counter()-began})
        result={'status':'completed','phase':'completed','completed_pairs':completed,'pair_count':10,
            'regressor_count':10,'classifier_count':10,'source_sha256':p['source_sha256'],
            'data_manifest_sha256':p['data_manifest_sha256'],'seconds':time.perf_counter()-began,
            'updated_at_kst':timestamp(),'Test_used':False,'CSV_created':False}
        write(RUN/'ALL_PAIRS_COMPLETED.json',result);write(OPS/'state.json',result);return 0
    except Exception as exc:
        write(OPS/'state.json',{'phase':'failed','error':f'{type(exc).__name__}: {exc}',
            'completed_pairs':completed,'updated_at_kst':timestamp(),'automatic_restart':False});raise
    finally:
        lock.unlink(missing_ok=True)


def launch():
    p=check_plan()
    if (OPS/'supervisor.lock').exists():raise RuntimeError('Supervisor lock exists; do not duplicate')
    with (OPS/'driver.log').open('a') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'supervise'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'phase':'launched','supervisor_pid':child.pid,'updated_at_kst':timestamp(),
        'plan_sha256':sha(OPS/'PLAN.json'),'detached':True,'CPU_threads':8,'Test_used':False}
    write(OPS/'LAUNCHED.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    for name in ('plan','launch','supervise','status'):sub.add_parser(name)
    w=sub.add_parser('worker');w.add_argument('--fold',type=int,required=True);w.add_argument('--regime',choices=('D0','H0'),required=True)
    w=sub.add_parser('smoke');w.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.action=='plan':print(json.dumps(plan(),ensure_ascii=False),flush=True)
    elif a.action=='launch':launch()
    elif a.action=='supervise':sys.exit(supervise())
    elif a.action=='status':print((OPS/'state.json').read_text())
    elif a.action=='smoke':fit_pair_cpu8(a.output,0,'D0',smoke_rows=256,smoke_iterations=4)
    else:fit_pair_cpu8(RUN/a.regime/f'fold_{a.fold:02d}',a.fold,a.regime)
