"""Run one frozen CV job and persist health/convergence observations.

Monitoring never changes model settings or reads challenge inputs. This lives
outside the frozen project so its progress writes cannot change code identity.
"""
from pathlib import Path
import ast,json,os,re,subprocess,sys,time

BASE=Path('/workspace/underwater_vessel_noise_r1')
PROJECT=BASE/'project';OPS=BASE/'ops';PY=BASE/'venv/bin/python'
RUN=PROJECT/'models/group_cv_physics_v2_r1_adam'


def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def proc_alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def cgroup_memory():
    root=Path('/sys/fs/cgroup/memory')
    out={}
    for name,field in [('memory.usage_in_bytes','used_bytes'),('memory.limit_in_bytes','limit_bytes'),
        ('memory.max_usage_in_bytes','peak_bytes'),('memory.failcnt','allocation_failure_count')]:
        p=root/name
        if p.exists():
            try:out[field]=int(p.read_text().strip())
            except ValueError:pass
    return out


def process_rss(pid):
    """Include the driver and its active candidate subprocesses."""
    pending=[int(pid)];seen=set();total=0
    while pending:
        current=pending.pop()
        if current in seen:continue
        seen.add(current)
        try:
            status=Path(f'/proc/{current}/status').read_text()
            match=re.search(r'^VmRSS:\s+(\d+)\s+kB',status,re.M)
            if match:total+=int(match.group(1))*1024
            children=Path(f'/proc/{current}/task/{current}/children')
            pending.extend(int(v) for v in children.read_text().split())
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return {'process_count':len(seen),'rss_sum_bytes':total}


def observations():
    latest={};alerts=[]
    if RUN.exists():
        for log in RUN.rglob('*training.log'):
            lines=log.read_text(errors='replace').splitlines()[-500:]
            for line in lines:
                if re.search(r'CUDA out of memory|Traceback \(most recent call last\)|Nonfinite training loss|Nonfinite validation',line,re.I):
                    alerts.append({'path':str(log.relative_to(PROJECT)),'event':line[-500:]})
                match=re.match(r'^(cnn|fdy|itransformer|tabm) (\{.*\})$',line)
                if match:
                    try:row=ast.literal_eval(match.group(2))
                    except (ValueError,SyntaxError):continue
                    if not isinstance(row,dict):continue
                    key=str(log.parent.relative_to(RUN))+'/'+match.group(1)
                    latest[key]={**row,'kind':match.group(1),'log':str(log.relative_to(PROJECT))}
        for manifest in RUN.rglob('run_manifest.json'):
            try:m=json.loads(manifest.read_text())
            except (ValueError,OSError):continue
            if str(m.get('status','')).startswith('failed'):
                alerts.append({'path':str(manifest.relative_to(PROJECT)),'event':m.get('error',m['status'])})
    for row in latest.values():
        if 'train_rmse' in row and 'validation_rmse' in row:
            row['generalization_gap']=row['validation_rmse']-row['train_rmse']
            row['note']='Interpret this source-generalization gap with the different frozen evaluation recipe distribution; it is not automatically proof of overfitting.'
    completed=[]
    if RUN.exists():
        completed=[str(p.parent.relative_to(RUN)) for p in RUN.glob('fold_*/outer_metrics.json')]
    try:
        gpu=subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10).stdout.strip()
    except Exception as exc:gpu=type(exc).__name__
    return {'unix_time':time.time(),'latest_neural_epochs':latest,'alerts':alerts,
        'completed_folds':completed,'gpu_csv':gpu,'cgroup_memory':cgroup_memory()}


def main():
    OPS.mkdir(exist_ok=True)
    state_path=OPS/'training_state.json'
    if state_path.exists():
        previous=json.loads(state_path.read_text())
        if previous.get('pid') and proc_alive(previous['pid']):raise RuntimeError('An existing training process is alive; never launch a duplicate')
        raise FileExistsError('A previous run state exists; inspect it before starting another job')
    if RUN.exists():raise FileExistsError('Training output exists; explicit reviewed resume or a fresh output is required')
    config=PROJECT/'configs/group_cv_physics_training.json'
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    command=[str(PY),'-u','-m','pipeline.train_group_cv','--config',str(config),'--device','cuda','--execute']
    state={'status':'starting','command':command,'project':str(PROJECT),'run_directory':str(RUN),'started_unix':time.time(),
        'supervisor_pid':os.getpid(),'automatic_hyperparameter_changes':False,'outer_used_for_tuning':False}
    atomic(state_path,state)
    with (OPS/'training_driver.log').open('x') as log:
        process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running',pid=process.pid);atomic(state_path,state)
        while process.poll() is None:
            obs=observations();obs['training_pid']=process.pid
            obs['training_process_tree']=process_rss(process.pid)
            atomic(OPS/'training_health.json',obs)
            with (OPS/'health_history.jsonl').open('a') as f:f.write(json.dumps(obs,allow_nan=False)+'\n')
            time.sleep(30)
        state.update(status='completed' if process.returncode==0 and (RUN/'cv_report.json').exists() else 'failed',
            exit_code=process.returncode,finished_unix=time.time())
        atomic(state_path,state);atomic(OPS/'training_health.json',observations())
    print(json.dumps(state,indent=2))


if __name__=='__main__':main()
