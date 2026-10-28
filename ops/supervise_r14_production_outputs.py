"""One continuous authorized batch: wait for model barriers, then local outputs.

Never starts/restarts training, changes policies, submits files, or schedules
future tasks. Failure is retained for review rather than retried silently.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PY='/Users/hongtaeyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
TREE=ROOT/'models/r14_production_trees_r1'
NN=ROOT/'models/r14_production_nn_fold0_r1'
OUT=ROOT/'models/r14_production_outputs_r1'
STATE=ROOT/'ops/r14_production_state.json'
QA=ROOT/'reports/R14_PRODUCTION_OUTPUTS_INDEPENDENT_QA.json'
DRIVER=ROOT/'ops/run_r14_production_outputs.py'
KST=ZoneInfo('Asia/Seoul')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def maybe(p):return read(p) if Path(p).is_file() else None
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
    if p==STATE:
        from ops.render_r14_production_status import render
        render(v)
def now():return datetime.now(KST).isoformat()
def health():
    result={}
    for name,launch,key,barrier in (
      ('Tree',TREE/'LAUNCH_RECEIPT.json','driver_pid',TREE/'ALL_FIVEFOLD_TREES_FROZEN.json'),
      ('NN',ROOT/'ops/r14_production_nn_launch_r1.json','PID',NN/'ALL_FOLD0_MODELS_FROZEN.json')):
        p=read(launch)[key]
        try:os.kill(p,0);alive=True
        except ProcessLookupError:alive=False
        except PermissionError:alive=None
        result[name]={'PID':p,'process_alive':alive,'all_models_barrier_present':barrier.is_file()}
        if alive is False and not barrier.is_file():raise RuntimeError(name+' training process ended without complete barrier; no restart')
    return result
def validate_qa():
    q=read(QA)
    if q['status']!='passed':raise ValueError('Output independent QA required')
    if not {'ops/run_r14_production_outputs.py','pipeline/ensemble_r14_production.py'}<=set(q['source_code_sha256']):raise ValueError('Missing reviewed output code')
    for p,h in q['source_code_sha256'].items():
        if h!=sha(ROOT/p):raise ValueError('Output code differs from reviewed version')
    return {str(QA.relative_to(ROOT)):sha(QA),**q['source_code_sha256']}
def validate_done(path):
    d=read(path)
    if d['status']!='completed':raise ValueError('Invalid completed output receipt')
    for p,h in d['source_pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('Completed output source changed:'+p)
    receipts=d.get('local_CSVs',[d.get('receipt')])
    if not receipts or any(not isinstance(r,dict) for r in receipts):raise ValueError('Missing completed CSV receipts')
    for r in receipts:
        output=ROOT/r['path']
        if r['official_upload'] or sha(output)!=r['output_sha256'] or read(output.with_suffix('.manifest.json'))!=r:
            raise ValueError('Completed CSV or manifest changed:'+str(output))
    return d
def snapshot(stages,active=None):
    config=read(ROOT/'configs/r14_production_mac_v1.json')
    return {'status':'running','updated_KST':now(),'production_recipe_approved':True,
      'approval_sha256':sha(ROOT/config['approval_path']),'training_settings':{'Adam':True,'fast3_epochs':100,'TFGrid_epochs':50,'early_stopping':False,'input_channels':14},
      'tree_progress':maybe(TREE/'progress.json'),'neural_run_state':maybe(NN/'run_state.json'),
      'neural_current_batch':maybe(NN/'progress.json'),
      'neural_models':{k:maybe(NN/k/'progress.json') for k in config['models']},
      'stages':stages,'active_output':active,'free_disk_GiB':shutil.disk_usage(ROOT).free/2**30,
      'official_upload':False,'other_neural_folds':'1..4 deferred until first requested local output is available',
      'old_diagnostic_status_rewritten':False}
def main():
    lock=ROOT/'ops/r14_production_output_supervisor.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    stages={};pins={};started=now()
    tasks=[('tree-only',TREE/'ALL_FIVEFOLD_TREES_FROZEN.json',OUT/'TREE_ONLY_OUTPUTS_COMPLETED.json'),
      ('hybrid-fast3',NN/'FAST_MODELS_FROZEN.json',OUT/'hybrid_fast3/COMPLETED.json'),
      ('hybrid-full4',NN/'ALL_FOLD0_MODELS_FROZEN.json',OUT/'hybrid_full4/COMPLETED.json')]
    active=None
    try:
        pins=validate_qa()
        while len(stages)<len(tasks):
            validate_qa()
            live=health()
            for failed in (TREE/'FAILED.json',NN/'FAILED.json'):
                if failed.exists():raise RuntimeError('Training failed; no restart: '+str(failed.relative_to(ROOT)))
            write(STATE,{**snapshot(stages,active),'started_KST':started,'output_QA_pins':pins,'training_processes':live})
            for action,barrier,done in tasks:
                if action in stages:continue
                if done.exists():
                    validate_done(done)
                    stages[action]={'status':'completed','receipt':str(done.relative_to(ROOT)),'receipt_sha256':sha(done),'observed_KST':now()};continue
                if not barrier.exists() or not (TREE/'ALL_FIVEFOLD_TREES_FROZEN.json').exists():continue
                if shutil.disk_usage(ROOT).free<2*2**30:raise RuntimeError('Under2GiB free: retain training and defer additional output work')
                active=action;log=ROOT/('ops/r14_output_'+action+'.log')
                if log.exists():raise FileExistsError('Existing output attempt is never silently reused: '+str(log))
                env=os.environ.copy();env.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT)+':'+str(ROOT/'ops/r14_production_runtime_py')+':/private/tmp/underwater-ml-deps:/private/tmp/underwater-eda-deps',
                  'DYLD_LIBRARY_PATH':'/opt/homebrew/opt/libomp/lib','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2'})
                with log.open('x') as f:
                    process=subprocess.Popen([PY,str(DRIVER),action],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        write(STATE,{**snapshot(stages,active),'started_KST':started,'output_QA_pins':pins,'output_PID':process.pid})
                        time.sleep(15)
                if process.returncode or not done.exists():raise RuntimeError(action+' failed; inspect '+str(log.relative_to(ROOT)))
                validate_done(done)
                stages[action]={'status':'completed','receipt':str(done.relative_to(ROOT)),'receipt_sha256':sha(done),'completed_KST':now()};active=None
                print(json.dumps({'stage_completed':action,'KST':now()},ensure_ascii=False),flush=True)
            if len(stages)<len(tasks):time.sleep(15)
        for _,_,done in tasks:validate_done(done)
        write(STATE,{**snapshot(stages),'status':'completed','started_KST':started,'completed_KST':now(),'output_QA_pins':pins})
    except BaseException as e:
        write(STATE,{**snapshot(stages,active),'status':'failed','error':str(e),'started_KST':started,'automatic_retry':False})
        raise
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
