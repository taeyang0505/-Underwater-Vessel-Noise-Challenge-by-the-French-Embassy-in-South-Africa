"""Run the approved Mac Fold0 budget once, then a separately pinned finalizer.

This is an execution supervisor, not a periodic desktop automation. It never
retries failures, changes training settings, launches Fold1-4, or uploads.
"""
from pathlib import Path
import fcntl, hashlib, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / 'ops/r16_production_neural_r1'
RUN = ROOT / 'models/r16_production_nn_fold0_r1'
PYTHON = Path('/Users/hongtaeyang/Documents/Codex/2026-10-27/new-chat/.venv/bin/python')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)

def main():
    OPS.mkdir(parents=True,exist_ok=True)
    lock=(OPS/'writer.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OPS/'STARTED.json').exists():raise RuntimeError('A fresh supervisor is required; no silent replay')
    if RUN.exists():raise RuntimeError('Neural output already exists; never restart a partial run')
    tree=read(ROOT/'models/r16_production_trees_r1/ALL_FIVEFOLD_TREES_FROZEN.json')
    if tree.get('status')!='completed' or tree.get('all_models_frozen')is not True:raise RuntimeError('Tree5 must finish first')
    for name,h in tree['artifact_sha256'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Tree registry artifact changed: '+name)
    qa=read(ROOT/'reports/R16_NEURAL_FINAL_INDEPENDENT_QA.json')
    if not str(qa.get('status','')).startswith('passed'):raise RuntimeError('Independent neural QA missing')
    cmd=[str(PYTHON),'-m','ops.train_r16_production_nn','train',
         '--config','configs/r16_neural_mac_v1.json',
         '--manifest','data/synthetic/r16_production_v1/fold_00/manifest.json',
         '--fold','0','--preflight','reports/R16_NEURAL_ACTUAL_PREFLIGHT_r3.json',
         '--cache-root','models/r16_production_feature_cache_r2',
         '--output','models/r16_production_nn_fold0_r1']
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'.:ops/r14_production_runtime_py',
         'PYTORCH_ENABLE_MPS_FALLBACK':'0','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
         'PYTHONUNBUFFERED':'1'}
    log=(OPS/'training.log').open('ab',buffering=0)
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    started=time.time()
    base={'supervisor_PID':os.getpid(),'training_PID':child.pid,'command':cmd,
          'started_unix':started,'fold':0,'architectures':4,'epochs_each':100,
          'NN_Fold1_to4_authorized_for_this_execution':False,'official_upload':False}
    write(OPS/'STARTED.json',base)
    while child.poll() is None:
        progress=read(RUN/'progress.json') if (RUN/'progress.json').exists() else None
        write(OPS/'state.json',{**base,'phase':'neural_training','elapsed_seconds':time.time()-started,'progress':progress})
        time.sleep(15)
    awake.wait(timeout=30);log.close()
    if child.returncode!=0 or not(RUN/'ALL_COMPLETED.json').exists():
        phase='paused' if (RUN/'PAUSED.json').exists() else 'failed'
        write(OPS/'state.json',{**base,'phase':phase,'exit_code':child.returncode,'no_automatic_retry':True});return
    write(OPS/'state.json',{**base,'phase':'neural_complete_waiting_pinned_finalizer','elapsed_seconds':time.time()-started})
    # Finalization may be prepared while this long training job runs. Only a
    # source-pinned, independently checked contract may trigger it.
    contract=OPS/'FINALIZER_READY.json'
    if not contract.exists():return
    plan=read(contract)
    if plan.get('status')!='ready' or plan.get('official_upload')is not False:raise RuntimeError('Finalizer contract invalid')
    for name,h in plan['artifact_sha256'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT)or sha(path)!=h:raise RuntimeError('Finalizer artifact changed')
    command=plan['command']
    if not isinstance(command,list)or not command:raise RuntimeError('Explicit argv required')
    with (OPS/'finalization.log').open('ab',buffering=0) as output:
        proc=subprocess.Popen(command,cwd=ROOT,env={**os.environ,**plan.get('environment',{})},stdout=output,stderr=subprocess.STDOUT)
        write(OPS/'state.json',{**base,'phase':'frozen_evaluation_and_candidate','finalizer_PID':proc.pid})
        code=proc.wait()
    write(OPS/'state.json',{**base,'phase':'finalization_finished' if code==0 else 'finalization_failed',
          'finalizer_exit_code':code,'official_upload':False,'elapsed_seconds':time.time()-started})

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(OPS/'SUPERVISOR_FAILED.json',{'error':repr(e),'no_automatic_retry':True});raise
