"""Run one frozen CV job and persist health/convergence observations.

Monitoring never changes model settings or reads challenge inputs. This lives
outside the frozen project so its progress writes cannot change code identity.
"""
from pathlib import Path
import argparse,ast,copy,hashlib,json,os,re,subprocess,sys,time

BASE=Path('/workspace/underwater_vessel_noise_r4')
PROJECT=BASE/'project';OPS=BASE/'ops';PY=BASE/'venv/bin/python'
RUN=PROJECT/'models/group_cv_passage_r4_adam'


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
        for log in RUN.rglob('*.log'):
            lines=log.read_text(errors='replace').splitlines()[-500:]
            for line in lines:
                if re.search(r'CUDA out of memory|Traceback \(most recent call last\)|Nonfinite training loss|Nonfinite validation|Query dependence detected|Fresh context cannot reproduce',line,re.I):
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
            row['note']='R4 uses the same declared passage recipe across splits with disjoint source groups. Interpret this gap with source transfer and fitting behavior; the same recipe does not imply identical source distributions.'
    completed=[]
    if RUN.exists():
        completed=[str(p.parent.relative_to(RUN)) for p in RUN.glob('fold_*/outer_metrics.json')]
    try:
        gpu=subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10).stdout.strip()
    except Exception as exc:gpu=type(exc).__name__
    return {'unix_time':time.time(),'latest_neural_epochs':latest,'alerts':alerts,
        'completed_folds':completed,'gpu_csv':gpu,'cgroup_memory':cgroup_memory()}


def launch():
    OPS.mkdir(exist_ok=True)
    state_path=OPS/'training_state.json'
    if state_path.exists():
        previous=json.loads(state_path.read_text())
        if previous.get('pid') and proc_alive(previous['pid']):raise RuntimeError('An existing training process is alive; never launch a duplicate')
        raise FileExistsError('A previous run state exists; inspect it before starting another job')
    if RUN.exists():raise FileExistsError('Training output exists; explicit reviewed resume or a fresh output is required')
    config=PROJECT/'configs/group_cv_passage_training_r4.json'
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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_r4_configuration(config):
    required_models=['lgbm_retrieval','cnn','fdy','itransformer']
    if config.get('models')!=required_models or config.get('foundation_models')!={}:raise ValueError('R4 is exactly four candidates with no foundation models')
    if config.get('seeds')!=[42] or config.get('protocol')!='source_group_nested5_vtuad_passage_r4':raise ValueError('R4 seed/protocol differs')
    if config.get('cv_dir')!='data/synthetic/group_cv_passage_r4_v1' or config.get('output_dir')!='models/group_cv_passage_r4_adam':raise ValueError('R4 paths differ')
    expected={'epochs':100,'patience':7,'optimizer':'adam','dropout':.1,'scheduler':'plateau','feature_version':3,'cnn_background_prior':True,'cnn_head_mode':'direct_mask','cnn_prior_residual_weight':0.,'cell_loss_weight':.25,'presence_loss_weight':.05,'contrastive_weight':0.,'domain_augmentation':'none'}
    for key,value in expected.items():
        if config.get('training',{}).get(key)!=value:raise ValueError('Frozen R4 training setting differs: '+key)
    safety=config.get('ensemble_safety_r4',{})
    expected_safety={'enabled':True,'ridge_lambda':.05,'epsilon_rmse':1e-4,'require_crossfit_and_fullfit_improvement':True,'outer_gate_changes_predictions':False,'affine':[1.,0.]}
    for key,value in expected_safety.items():
        if safety.get(key)!=value:raise ValueError('Frozen R4 ensemble setting differs: '+key)
    if safety.get('candidate_models')!=required_models or safety.get('fallback_models')!=['cnn','fdy','itransformer']:raise ValueError('R4 fallback/candidate model order differs')
    return config


def checked_preflight(expected_config_sha=None):
    config_path=PROJECT/'configs/group_cv_passage_training_r4.json'
    config=validate_r4_configuration(json.loads(config_path.read_text()))
    if expected_config_sha is not None and digest(config_path)!=expected_config_sha:raise ValueError('Configuration SHA differs from explicit launch identity')
    if PROJECT.parent!=BASE or BASE.name!='underwater_vessel_noise_r4':raise ValueError('R4 must stay separate from previous releases')
    cv_dir=PROJECT/config['cv_dir'];cv=json.loads((cv_dir/'cv_manifest.json').read_text())
    if cv.get('ready_to_train') is not True or cv.get('n_splits')!=5 or len(cv.get('folds',[]))!=5:raise ValueError('A complete R4 five-fold dataset is required')
    readiness={}
    for fold in cv['folds']:
        data=PROJECT/fold['data_dir']
        if not data.resolve().is_relative_to(cv_dir.resolve()):raise ValueError('R4 fold leaves its dataset root')
        p=data/'readiness_manifest.json';gate=json.loads(p.read_text())
        if gate.get('version')!=config['expected_dataset_gate_version'] or gate.get('ready_to_train') is not True:raise ValueError('Fold is not a ready R4 passage package')
        if gate.get('source_hash')!=fold.get('source_hash') or gate.get('data_package_hash')!=fold.get('data_package_hash'):raise ValueError('R4 CV/fold source or package differs')
        for flag in ['source_checks_passed','reference_group_and_duplicate_isolation_passed','background_week_and_purge_isolation_passed','sealed_holdout_locked']:
            if gate.get(flag) is not True:raise ValueError('Required R4 quality gate failed: '+flag)
        if gate.get('challenge_test_accessed') is not False:raise ValueError('Challenge input provenance is forbidden')
        readiness[str(fold['fold'])]=digest(p)
    script='import json; from pathlib import Path; from pipeline.train_group_cv import preflight; p=Path('+repr(str(config_path))+'); c=json.loads(p.read_text()); _,r=preflight(c,Path('+repr(str(cv_dir))+')); print(json.dumps(r,allow_nan=False))'
    result=subprocess.run([str(PY),'-c',script],cwd=PROJECT,capture_output=True,text=True,check=False)
    if result.returncode:raise RuntimeError('Core R4 preflight failed: '+result.stderr[-4000:]+result.stdout[-4000:])
    receipt=json.loads(result.stdout)
    if receipt.get('ready') is not True or receipt.get('full_training_executed') is not False:raise ValueError('Core preflight did not return a nontraining ready receipt')
    return {'status':'ready; preflight only','training_started':False,'config_sha256':digest(config_path),'cv_manifest_sha256':digest(cv_dir/'cv_manifest.json'),'fold_readiness_sha256':readiness,'core_preflight':receipt,'code_sha256':{p.name:digest(p) for p in (PROJECT/'pipeline').glob('*.py')},'supervisor_sha256':digest(__file__)}


def self_test():
    local=Path(__file__).resolve().parents[1]/'configs/group_cv_passage_training_r4.json'
    p=local if local.exists() else PROJECT/'configs/group_cv_passage_training_r4.json'
    config=json.loads(p.read_text());validate_r4_configuration(config)
    bad=[]
    for key,value in [('epochs',8),('optimizer','adamw'),('cnn_head_mode','soft_prior_residual'),('cell_loss_weight',.03),('cnn_prior_residual_weight',.001)]:
        item=copy.deepcopy(config);item['training'][key]=value;bad.append(item)
    item=copy.deepcopy(config);item['models'].append('tabicl');bad.append(item)
    item=copy.deepcopy(config);item['ensemble_safety_r4']['outer_gate_changes_predictions']=True;bad.append(item)
    for item in bad:
        try:validate_r4_configuration(item)
        except ValueError:pass
        else:raise AssertionError('Invalid R4 configuration was accepted')
    print(json.dumps({'status':'PASS','invalid_configurations_rejected':len(bad),'training_started':False,'GPU_accessed':False,'remote_accessed':False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Start exactly one explicitly authorized fresh R4 run after all gates pass')
    parser.add_argument('--expected-config-sha256',help='Mandatory SHA identity for --execute')
    parser.add_argument('--self-test',action='store_true',help='Numerical-free local configuration rejection checks; no server/GPU use')
    args=parser.parse_args()
    if args.self_test:
        if args.execute:parser.error('--self-test cannot execute training')
        self_test();return
    if args.execute and (not args.expected_config_sha256 or not re.fullmatch('[0-9a-f]{64}',args.expected_config_sha256)):parser.error('--execute requires the explicit 64-digit configuration SHA')
    try:
        receipt=checked_preflight(args.expected_config_sha256)
        if not args.execute:
            print(json.dumps(receipt,indent=2));return
        if (OPS/'training_state.json').exists() or RUN.exists():raise FileExistsError('Existing R4 state/output cannot be overwritten or automatically resumed')
        OPS.mkdir(parents=True,exist_ok=True)
        receipt.update(status='ready; explicit launch requested',full_training_executed=False)
        with (OPS/'launch_preflight_receipt.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
        launch()
    except Exception as exc:
        print(json.dumps({'status':'R4 launch/preflight failed; preserve state','error_type':type(exc).__name__,'error':str(exc),'automatic_restart':False}),file=sys.stderr)
        raise


if __name__=='__main__':main()
