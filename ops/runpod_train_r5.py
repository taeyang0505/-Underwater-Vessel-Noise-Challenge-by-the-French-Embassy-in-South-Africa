"""Supervise exactly one explicitly requested, fresh R5 GPU runner process.

Writes 30-second observations and verified completed-artifact backups. No
automatic restarts, parameter changes, Pod control, Test IO or submission.
No app/scheduler automation is created by this ordinary server process.
"""
from pathlib import Path
import argparse, ast, fcntl, hashlib, json, os, platform, re, shutil, subprocess, sys, tempfile, time
from backup_r5_completed import snapshot, sha, canonical, encode

def atomic(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as f:f.write(encode(value));f.flush();os.fsync(f.fileno())
    temp.replace(path)

def json_value(text):
    def bad(value):raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(text,parse_constant=bad)
def read_json(path):return json_value(Path(path).read_text())

def process_start(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,ValueError,IndexError):return None

def alive(pid,start=None):
    try:os.kill(int(pid),0)
    except (ProcessLookupError,ValueError,TypeError):return False
    except PermissionError:return True
    actual=process_start(pid)
    return start is None or actual==str(start)

def running_r5_drivers():
    found=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:parts=p.read_bytes().split(b'\0')
        except OSError:continue
        if b'pipeline.runner_r5' in parts and b'fit' in parts:found.append(int(p.parent.name))
    return found

def memory_directory(root):
    root=Path(root);out={'path':str(root)}
    specs={'memory.current':'used_bytes','memory.max':'limit_bytes','memory.peak':'peak_bytes',
        'memory.usage_in_bytes':'used_bytes','memory.limit_in_bytes':'limit_bytes',
        'memory.max_usage_in_bytes':'peak_bytes','memory.failcnt':'allocation_failure_count'}
    for name,field in specs.items():
        p=root/name
        if p.exists():
            try:
                v=p.read_text().strip();out[field]=None if v=='max' else int(v)
            except (OSError,ValueError):pass
    events=root/'memory.events'
    if events.exists():
        try:out['events']={k:int(v) for k,v in (line.split() for line in events.read_text().splitlines())}
        except (OSError,ValueError):pass
    return out if len(out)>1 else None

def cgroup_memory():
    """Resolve v1/v2 memberships, with namespace-root/common-mount fallbacks."""
    candidates=[];members={}
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            _,controllers,subpath=line.split(':',2)
            for controller in (controllers.split(',') if controllers else ['unified']):members[controller]=subpath
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            left,right=line.split(' - ',1);fields=left.split();fs,*rest=right.split()
            if fs not in {'cgroup','cgroup2'}:continue
            mountroot=fields[3];mount=Path(fields[4]);controller='unified' if fs=='cgroup2' else 'memory'
            if controller not in members:continue
            if fs=='cgroup' and 'memory' not in right.split()[-1].split(',') and mount.name!='memory':continue
            relative=members[controller]
            if mountroot!='/' and (relative==mountroot or relative.startswith(mountroot.rstrip('/')+'/')):relative=relative[len(mountroot):]
            candidates.extend([mount/relative.lstrip('/'),mount])
    except (OSError,ValueError,IndexError):pass
    candidates.extend([Path('/sys/fs/cgroup/memory'),Path('/sys/fs/cgroup')]);out=[];seen=set()
    for p in candidates:
        if str(p) in seen:continue
        seen.add(str(p));record=memory_directory(p)
        if record:out.append(record)
    return out

def process_rss(pid):
    pending=[int(pid)];seen=set();total=0
    while pending:
        current=pending.pop()
        if current in seen:continue
        seen.add(current)
        try:
            status=Path(f'/proc/{current}/status').read_text();m=re.search(r'^VmRSS:\s+(\d+)\s+kB',status,re.M)
            if m:total+=int(m.group(1))*1024
            pending.extend(int(v) for v in Path(f'/proc/{current}/task/{current}/children').read_text().split())
        except (OSError,ValueError):pass
    return {'process_count':len(seen),'rss_sum_bytes':total}

def gpu_health():
    command=['nvidia-smi','--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader,nounits']
    try:
        r=subprocess.run(command,capture_output=True,text=True,timeout=10)
        return {'returncode':r.returncode,'csv':r.stdout.strip(),'stderr':r.stderr[-500:]}
    except (OSError,subprocess.TimeoutExpired) as e:return {'unavailable':type(e).__name__}

def tail(path,limit=131072):
    with Path(path).open('rb') as f:
        f.seek(0,2);size=f.tell();f.seek(max(0,size-limit));return f.read().decode('utf-8',errors='replace')

ALERT_PATTERN=re.compile(r'CUDA out of memory|OutOfMemoryError|Traceback \(most recent call last\)|FloatingPointError|nonfinite|\b(?:NaN|Infinity|-Infinity)\b|(?:^|[\s:=])[-+]?inf(?:[\s,;}]|$)',re.I)

def epoch_observations(run):
    epochs={};completed=[];failures=[]
    if not run.exists():return epochs,completed,failures
    for p in run.rglob('epochs.jsonl'):
        if p.is_symlink():continue
        try:
            lines=tail(p).splitlines();row=None
            for line in reversed(lines):
                try:row=json_value(line);break
                except ValueError:continue
            if isinstance(row,dict):
                k=str(p.parent.relative_to(run));epochs[k]=row
                if 'train_rmse' in row and 'inner_rmse' in row:epochs[k]['inner_minus_train_rmse']=row['inner_rmse']-row['train_rmse']
        except OSError:pass
    for p in run.rglob('summary.json'):
        try:
            v=read_json(p)
            if v.get('status')=='completed':completed.append({'path':str(p.parent.relative_to(run)),'kind':v.get('kind'),'best_epoch':v.get('best_epoch')})
        except (OSError,ValueError):pass
    for p in run.rglob('FAILED.json'):
        try:failures.append({'path':str(p.relative_to(run)),'failure':read_json(p)})
        except (OSError,ValueError):pass
    return epochs,completed,failures

def environment_receipt(python,project,env):
    # Explicit allowlist only: no environment dump, tokens, keys or credentials.
    code="""import importlib.metadata as m,json,sys,platform
names=['torch','numpy','scipy','pandas','pyarrow','lightgbm','safetensors','transformers','tokenizers']
versions={}
for name in names:
    try: versions[name]=m.version(name)
    except m.PackageNotFoundError: versions[name]=None
print(json.dumps({'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':versions,'CUDA_probe_scope':'nvidia-smi health only; runner initializes CUDA'}))
"""
    r=subprocess.run([str(python),'-c',code],cwd=project,env=env,capture_output=True,text=True,timeout=30,check=True)
    obj=json.loads(r.stdout);obj['supervisor_platform']=platform.platform();return obj

def validate_inputs(base,config_arg,python_arg,preflight_arg,run_name,folds):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',run_name) or '..' in run_name:raise ValueError('Simple fresh run name required')
    if not folds or len(set(folds))!=len(folds) or any(type(v) is not int or v not in range(5) for v in folds):raise ValueError('Explicit unique folds0..4 required')
    base=Path(base).resolve();project=base/'project'
    def resolve(v):
        p=Path(v);return p.resolve() if p.is_absolute() else (base/p).resolve()
    config=resolve(config_arg)
    if not config.is_relative_to(project/'configs') or not config.is_file():raise ValueError('Config must be a saved PROJECT/configs file')
    python=resolve(python_arg) if python_arg else base/'venv/bin/python'
    if not python.is_file():raise FileNotFoundError('Explicit existing Python/venv required: '+str(python))
    proof=resolve(preflight_arg) if preflight_arg else project/'reports/R5_TRAINING_PREFLIGHT.json'
    cfg=read_json(config);preflight=read_json(proof)
    if (cfg.get('training',{}).get('epochs'),cfg.get('training',{}).get('patience'),cfg.get('training',{}).get('optimizer'))!=(100,7,'adam'):raise ValueError('User contract requires maximum100 epochs/patience7/Adam')
    if preflight.get('ready_for_gpu_pilot') is not True or preflight.get('config_sha256')!=canonical(cfg):raise ValueError('Current matching completed CPU preflight required')
    tree=ast.parse((project/'pipeline/runner_r5.py').read_text());expected=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CODE_NAMES' for t in node.targets):expected=ast.literal_eval(node.value)
    if not isinstance(expected,list) or not expected or set(preflight.get('code_sha256',{}))!=set(expected):raise ValueError('Exact runner source SHA closure required')
    for name,h in preflight.get('code_sha256',{}).items():
        p=(project/name)
        if not p.resolve().is_relative_to(project) or p.is_symlink() or sha(p)!=h:raise ValueError('Frozen preflight code changed: '+name)
    if not preflight.get('code_sha256'):raise ValueError('Frozen source SHA closure required')
    for name,v in preflight.get('fixture_reports',{}).items():
        if Path(name).name!=name or sha(project/'reports'/name)!=v['sha256']:raise ValueError('Frozen fixture receipt changed')
    return base,project,config,python,proof,cfg,preflight

def supervise(a):
    base,project,config,python,proof,cfg,preflight=validate_inputs(a.base,a.config,a.python,a.preflight_report,a.run_name,a.folds)
    ops=base/'ops';ops.mkdir(exist_ok=True);job=ops/a.run_name;run=project/'models'/a.run_name
    if job.exists() or run.exists():raise FileExistsError('Fresh RUN and OPS job directories required; no partial reuse or automatic resume')
    with (ops/'.r5_training.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another R5 supervisor holds the global lock')
        active_path=ops/'active_training_r5.json'
        if active_path.exists():
            old=read_json(active_path)
            if old.get('pid') and alive(old['pid'],old.get('process_start_ticks')):raise RuntimeError('Existing R5 training process is alive; duplicate launch forbidden')
        existing=running_r5_drivers()
        if existing:raise RuntimeError('An R5 fit driver is already alive: '+str(existing))
        job.mkdir();config_copy=job/'config.json';shutil.copyfile(config,config_copy);shutil.copyfile(proof,job/'preflight_report.json')
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        env['PYTHONPATH']=str(project)+os.pathsep+env.get('PYTHONPATH','')
        environment=environment_receipt(python,project,env);atomic(job/'environment_receipt.json',environment)
        command=[str(python),'-u','-m','pipeline.runner_r5','fit','--config',str(config),'--output',str(run),'--preflight-report',str(proof),'--folds',*map(str,a.folds)]
        startup={'format':'r5_training_startup_v1','created_unix':time.time(),'project':str(project),'run_directory':str(run),'requested_folds':a.folds,
            'config_path':str(config),'config_file_sha256':sha(config),'config_sha256':canonical(cfg),'preflight_file_sha256':sha(proof),
            'source_code_sha256':preflight['code_sha256'],'pinned_public_checkpoints':preflight.get('public_weights'),
            'fold_bindings':preflight.get('folds'),'environment_receipt_sha256':sha(job/'environment_receipt.json'),
            'supervisor_sha256':sha(Path(__file__)),'backup_helper_sha256':sha(Path(__file__).with_name('backup_r5_completed.py')),
            'command':command,'training_contract':{'maximum_epochs':100,'early_stopping_patience':7,'optimizer':'adam'},
            'automatic_restart':False,'automatic_hyperparameter_change':False,'Test_accessed_by_supervisor':False,'official_submission':False}
        atomic(job/'startup_receipt.json',startup)
        state={'status':'starting','supervisor_pid':os.getpid(),'started_unix':time.time(),'run_directory':str(run),'ops_directory':str(job),'command':command,'config_sha256':canonical(cfg),'requested_folds':a.folds,'automatic_restart':False,'automatic_hyperparameter_change':False,'official_submission':False}
        atomic(job/'training_state.json',state);alerts_seen=set();log_cursor=0
        with (job/'training_driver.log').open('x') as log:
            process=subprocess.Popen(command,cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='running',pid=process.pid,process_start_ticks=process_start(process.pid));atomic(job/'training_state.json',state);atomic(active_path,state)
            try:
                while True:
                    epochs,completed,failures=epoch_observations(run);alerts=[]
                    with (job/'training_driver.log').open('r',errors='replace') as source:
                        source.seek(log_cursor)
                        for line in source:
                            if ALERT_PATTERN.search(line):
                                event=line.strip()[-1000:];key=hashlib.sha256(event.encode()).hexdigest()
                                if key not in alerts_seen:alerts.append(event);alerts_seen.add(key)
                        log_cursor=source.tell()
                    try:backup=snapshot(run,job/'backups',job)
                    except Exception as e:backup={'status':'failed','error':type(e).__name__,'message':str(e)};alerts.append('Completed artifact backup failed: '+str(e))
                    obs={'unix_time':time.time(),'training_pid':process.pid,'process_returncode':process.poll(),'latest_epochs':epochs,'completed_summaries':completed,'failed_artifacts':failures,'new_log_alerts':alerts,'gpu':gpu_health(),'cgroup_memory':cgroup_memory(),'process_tree':process_rss(process.pid),'backup':backup,'automatic_intervention_executed':False}
                    atomic(job/'training_health.json',obs)
                    with (job/'health_history.jsonl').open('a') as f:f.write(json.dumps(obs,allow_nan=False)+'\n')
                    if alerts:
                        with (job/'alerts.jsonl').open('a') as f:
                            for alert in alerts:f.write(json.dumps({'unix_time':time.time(),'event':alert})+'\n')
                    if process.poll() is not None:break
                    deadline=time.monotonic()+30
                    while time.monotonic()<deadline and process.poll() is None:time.sleep(min(1,max(0,deadline-time.monotonic())))
                final=None
                try:final=read_json(run/'run_state.json')
                except (OSError,ValueError):pass
                expected='completed' if len(a.folds)==5 else 'completed_requested_folds'
                good=process.returncode==0 and final is not None and final.get('status')==expected and final.get('config_sha256')==state['config_sha256'] and sorted(v['fold'] for v in final.get('folds',[]))==sorted(a.folds)
                state.update(status='completed' if good else 'failed',exit_code=process.returncode,finished_unix=time.time(),runner_status=None if final is None else final.get('status'),final_backup=backup,log_alert_count=len(alerts_seen))
                atomic(job/'training_state.json',state);atomic(active_path,state)
            except BaseException as e:
                # Do not kill/restart an already running fit on a monitor error.
                state.update(status='supervisor_interrupted',monitor_error=type(e).__name__,child_alive=process.poll() is None,finished_unix=time.time())
                atomic(job/'training_state.json',state);atomic(active_path,state);raise
        print(json.dumps(state,indent=2,allow_nan=False));return 0 if state['status']=='completed' else 2

def self_test():
    with tempfile.TemporaryDirectory(prefix='r5_supervisor_fixture_') as tmp:
        p=Path(tmp);(p/'memory.current').write_text('1234');(p/'memory.max').write_text('max');(p/'memory.peak').write_text('2000');(p/'memory.events').write_text('oom 0\noom_kill 0\n')
        assert memory_directory(p)=={'path':str(p),'used_bytes':1234,'limit_bytes':None,'peak_bytes':2000,'events':{'oom':0,'oom_kill':0}}
        for name in ['memory.current','memory.max','memory.peak','memory.events']:(p/name).unlink()
        (p/'memory.usage_in_bytes').write_text('2500');(p/'memory.limit_in_bytes').write_text('4000');(p/'memory.failcnt').write_text('3')
        assert memory_directory(p)['used_bytes']==2500 and memory_directory(p)['allocation_failure_count']==3
        assert all(ALERT_PATTERN.search(v) for v in ['CUDA out of memory','FloatingPointError: Nonfinite loss','{"loss": NaN}','loss: inf'])
        assert not ALERT_PATTERN.search('{"epoch":0,"inner_rmse":0.3,"train_rmse":0.25}')
        run=p/'run';m=run/'fold_00/models/bg_cnn';m.mkdir(parents=True);(m/'epochs.jsonl').write_text('{"epoch":0,"train_rmse":0.25,"inner_rmse":0.3}\n')
        epochs,_,_=epoch_observations(run);assert abs(epochs['fold_00/models/bg_cnn']['inner_minus_train_rmse']-.05)<1e-12
        # Saved config/source preflight validation uses only fake local files.
        base=p/'base';project=base/'project';(project/'configs').mkdir(parents=True);(project/'pipeline').mkdir();(project/'reports').mkdir()
        source=project/'pipeline/runner_r5.py';source.write_text("CODE_NAMES=['pipeline/runner_r5.py']\n")
        cfg={'training':{'epochs':100,'patience':7,'optimizer':'adam'}};config=project/'configs/c.json';config.write_bytes(encode(cfg))
        proof=project/'reports/p.json';data={'ready_for_gpu_pilot':True,'config_sha256':canonical(cfg),'code_sha256':{'pipeline/runner_r5.py':sha(source)}};proof.write_bytes(encode(data))
        validate_inputs(base,'project/configs/c.json',sys.executable,'project/reports/p.json','fresh_fixture',[0,1,2,3,4])
        for mutation in ['source','epochs','folds']:
            if mutation=='source':source.write_text("CODE_NAMES=['pipeline/runner_r5.py']\n# changed\n")
            if mutation=='epochs':source.write_text("CODE_NAMES=['pipeline/runner_r5.py']\n");cfg['training']['epochs']=99;config.write_bytes(encode(cfg))
            try:validate_inputs(base,'project/configs/c.json',sys.executable,'project/reports/p.json','fresh_fixture',[0,0] if mutation=='folds' else [0])
            except ValueError:pass
            else:raise AssertionError('Invalid source/config/fold policy accepted')
    return {'passed':True,'scope':'local synthetic numeric/JSON/cgroup fixture only','GPU_or_training_started':False,'checks':['cgroup v1/v2 counters and unlimited bound','OOM/NaN/Inf error detection','finite epoch parse and gap','startup source/config/training-contract/fold guards','no remote access']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path);p.add_argument('--config');p.add_argument('--run-name');p.add_argument('--folds',type=int,nargs='+')
    p.add_argument('--python');p.add_argument('--preflight-report');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:print(json.dumps(self_test(),indent=2));return
    if a.base is None or a.config is None or a.run_name is None or not a.folds:p.error('Explicit --base --config --run-name --folds required')
    raise SystemExit(supervise(a))

if __name__=='__main__':main()
