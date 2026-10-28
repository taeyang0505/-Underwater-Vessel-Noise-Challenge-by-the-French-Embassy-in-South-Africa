#!/usr/bin/env python3
"""Single-shot operator supervisor for an explicitly approved fresh R9 run.

No retry, resume, model edits, tree fitting, transport, pod control or signals.
This operator file is outside the immutable model/data source closure.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def atomic(path,value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def ticks(pid):
    s=Path('/proc')/str(pid)/'stat'
    value=s.read_text();return int(value[value.rfind(')')+2:].split()[19])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True)
    a=p.parse_args();base=Path(a.base).resolve();project=base/'project';python=base/'venv/bin/python'
    if str(base)!='/workspace/underwater_vessel_noise_r9_r1':raise ValueError('Explicit new approved server base required')
    read=lambda name:json.loads((base/name).read_text())
    deploy=read('artifacts/deployment_verification.json')
    if deploy.get('status')!='passed' or deploy['archive_sha256']!='d3d9919bfb27b1fb3e7dfc841efadd0af88c410499d72da78d02e506648450a0':
        raise ValueError('Full package verification required')
    pf=json.loads((project/'reports/R9_GPU_PREFLIGHT.json').read_text())
    probe=json.loads((project/'reports/R9_CUDA_FULL_MODEL_PREFLIGHT.json').read_text())
    if (pf.get('ready_for_production_training') is not True or probe.get('ready_for_production') is not True
            or probe.get('status')!='passed' or len(probe['cases'])!=4
            or any(not c['all_gradients_parameters_finite'] or c['Adam_steps']!=1 or c['batch_size']!=128 for c in probe['cases'])
            or pf['config_file_sha256']!='9962b8a5c41708b9f8a51cb77cd0aa3295969b830077e240f2329b1f6c64679a'
            or pf['data_bundle_hash']!='9788ec783be139d51a45a49ec3b22539f10a8b9eff85f3acb501127ae82a4092'):
        raise ValueError('Actual CUDA/model/data gate failed')
    for n,h in pf['code_sha256'].items():
        if sha(project/n)!=h:raise ValueError('Production code changed: '+n)
    output=project/'models/group_cv_minute_full_r9_v1'
    if output.exists():raise FileExistsError('Output exists; no partial reuse or resume')
    claim=base/'nn_supervisor_claim.json'
    state={'format':'r9_neural_operator_supervisor_v1','status':'initializing',
           'supervisor_pid':os.getpid(),'supervisor_start_ticks':ticks(os.getpid()),
           'operator_script_sha256':sha(__file__),'base':str(base),'project':str(project),
           'archive_sha256':deploy['archive_sha256'],'config_file_sha256':pf['config_file_sha256'],
           'data_bundle_hash':pf['data_bundle_hash'],
           'CUDA_preflight_sha256':sha(project/'reports/R9_CUDA_FULL_MODEL_PREFLIGHT.json'),
           'structural_preflight_sha256':sha(project/'reports/R9_GPU_PREFLIGHT.json'),
           'environment_sha256':sha(base/'artifacts/environment_frozen.json'),
           'requested_folds':[0,1,2,3,4],'requested_models':['bg_cnn','tfgrid_fdy_conformer','fdy','itransformer'],
           'epochs':100,'early_stopping':False,'tree_fit_on_server':False,
           'retry_or_resume_allowed':False,'started_unix':time.time(),
           'started_utc':datetime.now(timezone.utc).isoformat(),'output':str(output)}
    with claim.open('x') as f:json.dump(state,f,indent=2);f.write('\n')
    (base/'logs').mkdir(exist_ok=True)
    command=[str(python),'-u','-m','ops.run_r9_training','fit','--preflight','reports/R9_GPU_PREFLIGHT.json']
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(project),
         'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    log=base/'logs/nn_training.log'
    with log.open('x') as stream:
        child=subprocess.Popen(command,cwd=project,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    state.update(status='running',child_pid=child.pid,child_start_ticks=ticks(child.pid),command=command,stdout_log=str(log))
    atomic(base/'nn_supervisor_state.json',state)
    print(json.dumps(state),flush=True)
    while child.poll() is None:
        state['last_alive_unix']=time.time();atomic(base/'nn_supervisor_state.json',state);time.sleep(10)
    code=child.returncode;state.update(exit_code=code,finished_unix=time.time(),
          status='awaiting_mac_tree' if code==0 else 'failed')
    completed=output/'neural_stage_completed.json'
    if code==0:
        if not completed.is_file() or json.loads(completed.read_text()).get('neural_stage_complete') is not True:
            state.update(status='failed_missing_neural_receipt',exit_code=1);code=1
        else:state['neural_completed_receipt_sha256']=sha(completed)
    atomic(base/'nn_supervisor_state.json',state);print(json.dumps(state),flush=True)
    raise SystemExit(code)


if __name__=='__main__':main()
