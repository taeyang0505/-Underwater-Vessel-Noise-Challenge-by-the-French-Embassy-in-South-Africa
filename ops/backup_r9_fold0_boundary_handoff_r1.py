#!/usr/bin/env python3
"""Bounded read-only guard wait, then independent completed Fold0 handoff.

Never signals a process or edits an original run/ledger. A missing original
iTransformer ledger step remains missing; this operator emits its own
explicit model-completion producer receipt after independent verification.
"""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
from ops.backup_r9_completed_cnn import REMOTE

ROOT=Path(__file__).resolve().parents[1]
GUARD_SHA='04985407f260200abc7fc5102dfed63e7fbf77a6070ad728a661da3166d4113c'

WAIT=r'''
guard_file=base/'ops/fold0_boundary_pause_r1/status.json'
guard_code=base/'ops/r9_fold_boundary_guard.py'
assert sha(guard_code)=='04985407f260200abc7fc5102dfed63e7fbf77a6070ad728a661da3166d4113c'
deadline=time.monotonic()+2400
while True:
 guard=json.loads(guard_file.read_text())
 assert guard['guard_source_sha256']=='04985407f260200abc7fc5102dfed63e7fbf77a6070ad728a661da3166d4113c'
 assert guard['operator_owner']['pid']==23667 and guard['operator_owner']['startticks']==87938062
 assert guard['expected']['pid']==2257 and guard['expected']['startticks']==86168117
 assert guard['expected']['supervisor']==2256 and guard['expected']['supervisor_startticks']==86168115
 print('R9_WAIT_STATUS '+json.dumps({'status':guard['status'],'updated_utc':guard['updated_utc'],'time_unix':time.time()}),flush=True)
 if guard['status']=='paused_after_last_model_summary':break
 if guard['status'].startswith('failed'):raise RuntimeError('Guard integrity/stop failure: '+json.dumps(guard))
 if guard['status']!='armed':raise RuntimeError('Unexpected guard state')
 if time.monotonic()>deadline:raise TimeoutError('40minute read-only guard wait expired')
 time.sleep(20)
assert guard['Fold0_all_four_model_training_complete'] is True
assert guard['later_fold_model_directories']==[]
assert guard['original_code_or_ledger_modified'] is False
def proc(p):
 text=(Path('/proc')/str(p)/'stat').read_text();v=text[text.rfind(')')+2:].split()
 return {'pid':p,'state':v[0],'pgid':int(v[2]),'startticks':int(v[19])}
def assert_stopped():
 current=[]
 for pid,start in [(2257,86168117),(2256,86168115)]:
  v=proc(pid);assert v['startticks']==start and v['state'] in 'Tt';current.append(v)
 assert proc(2257)['pgid']==2257
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:v=proc(int(p.name))
  except (FileNotFoundError,ProcessLookupError):continue
  if v['pgid']!=2257 or v['state'] in 'ZX':continue
  assert v['state'] in 'Tt',v
  for t in (p/'task').iterdir():
   text=(t/'stat').read_text();assert text[text.rfind(')')+2:].split()[0] in 'TtZX'
  current.append(v)
 for f in range(1,5):assert not (run/f'fold_{f:02d}/models').exists()
 return current
stopped_processes_independent=assert_stopped()
identity=json.loads((run/'execution_identity.json').read_text())
assert identity['config_file_sha256']=='9962b8a5c41708b9f8a51cb77cd0aa3295969b830077e240f2329b1f6c64679a'
assert identity['data_bundle_hash']=='9788ec783be139d51a45a49ec3b22539f10a8b9eff85f3acb501127ae82a4092'
assert sha(project/'configs/r9_full_training_v1.json')==identity['config_file_sha256']
all4_summary_sha={}
for kind in ('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer'):
 source=fold/'models'/kind;sm=json.loads((source/'summary.json').read_text())
 assert sm['status']=='completed' and sm['epochs_run']==100 and sm['fixture_only'] is False
 assert sm['kind']==kind and sm['stop_reason']=='completed_fixed_epochs'
 for n,h in sm['artifact_sha256'].items():assert sha(source/n)==h,(kind,n)
 for n,h in sm['code_sha256'].items():assert identity['code_sha256'][n]==h and sha(project/n)==h
 all4_summary_sha[kind]=sha(source/'summary.json')
 assert all4_summary_sha[kind]==guard['models'][kind]['summary_sha256']
'''

METADATA=r'''
independent_model_sha={p.relative_to(model).as_posix():sha(p) for p in model.rglob('*') if p.is_file()}
original_it_ledger_present=step is not None
original_neural_receipt_exists=(fold/'neural_completed.json').exists()
original_root_neural_receipt_exists=(run/'neural_stage_completed.json').exists()
'''

PAYLOAD=r'''
for kind in ('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer'):
 payload['handoff_metadata/all4_model_summaries/'+kind+'.json']=fold/'models'/kind/'summary.json'
for f in range(1,5):payload[f'run_metadata/fold_{f:02d}_context.json']=run/f'fold_{f:02d}/context.json'
extra_sources={'guard_status.json':guard_file,'environment_frozen.json':base/'artifacts/environment_frozen.json',
 'deployment_verification.json':base/'artifacts/deployment_verification.json',
 'run_state.json':run/'run_state.json','RUNNING.lock':run/'RUNNING.lock',
 'nn_supervisor_state.json':base/'nn_supervisor_state.json',
 'boundary_guard_deployment.json':base/'ops/fold0_boundary_guard_deployment.json',
 'neural_completed.json':fold/'neural_completed.json',
 'root_neural_stage_completed.json':run/'neural_stage_completed.json',
 'nn_training_stdout.log':base/'logs/nn_training.log',
 'data_bundle_manifest.json':project/'data/synthetic/group_cv_minute_full_r9_v1/manifest.json',
 'data_bundle_readiness.json':project/'data/synthetic/group_cv_minute_full_r9_v1/readiness.json'}
snapshot_directory=backup/'metadata_snapshots';snapshot_directory.mkdir()
for name,source in extra_sources.items():
 if source.is_file():
  target=snapshot_directory/name;target.write_bytes(source.read_bytes())
  payload['handoff_metadata/'+name]=target
payload['handoff_metadata/frozen_config.json']=project/'configs/r9_full_training_v1.json'
payload['operator_code/r9_fold_boundary_guard.py']=guard_code
assert_stopped()
'''

AUDIT=r'''
audit.update(format='r9_independent_completed_model_boundary_handoff_v1',
 independently_verified_model='itransformer',original_train_itransformer_ledger_step_present=original_it_ledger_present,
 original_fold_neural_completed_receipt_exists=original_neural_receipt_exists,
 original_root_neural_stage_completed_receipt_exists=original_root_neural_receipt_exists,
 original_ledger_was_not_edited=True,original_missing_stage_receipts_were_not_fabricated=True,
 all_four_Fold0_models_completed_independently=True,all5_fold_run_completed=False,
 all4_original_summary_sha256=all4_summary_sha,guard_status_sha256=sha(guard_file),
 guard_source_sha256=sha(guard_code),current_stopped_processes=stopped_processes_independent,
 later_fold_models_exist=False,fresh_version_fold_import_required=True,resume_authorized=False,
 previously_backed_up_model_weights_duplicated=False,
 completed_model_file_sha256=independent_model_sha)
'''


def remote_program():
    s=REMOTE.replace('bg_cnn','itransformer').replace('completed_r1','boundary_handoff_r1')
    def replace(old,new,count=1):
        nonlocal s
        if s.count(old)!=count:raise ValueError('Remote audit template contract changed: '+old)
        s=s.replace(old,new)
    replace("ledger_bytes=(fold/'artifact_ledger.json').read_bytes()",WAIT+"\nledger_bytes=(fold/'artifact_ledger.json').read_bytes()")
    replace("step=ledger['steps']['train/itransformer']","step=ledger.get('steps',{}).get('train/itransformer')")
    replace("assert step['status']=='completed' and summary['status']=='completed'","assert summary['status']=='completed'\nif step is not None:assert step['status']=='completed'")
    replace("assert actual==set(step['artifact_sha256'])","assert step is None or actual==set(step['artifact_sha256'])")
    replace("for n,h in step['artifact_sha256'].items():assert sha(fold/n)==h,n", "for n,h in (step['artifact_sha256'] if step else {}).items():assert sha(fold/n)==h,n",count=2)
    replace("backup=base/'backups/r9_fold00_itransformer_boundary_handoff_r1'",METADATA+"\nbackup=base/'backups/r9_fold00_itransformer_boundary_handoff_r1'")
    replace("audit={'format':'r9_completed_CNN_backup_v1'",PAYLOAD+"\naudit={'format':'r9_completed_CNN_backup_v1'")
    replace("manifest=backup/'backup_manifest.json'",AUDIT+"\nmanifest=backup/'backup_manifest.json'")
    replace("receipt={'status':'passed'", "assert_stopped()\nfor n,h in independent_model_sha.items():assert sha(model/n)==h,n\nreceipt={'status':'passed'")
    compile(s,'remote-boundary-handoff','exec')
    return s


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    start=time.monotonic();dest=ROOT/'ops/server_backups/r9_gpu_r1/fold_00_itransformer_boundary_handoff_r1'
    if dest.exists():raise FileExistsError('Fresh immutable handoff backup required')
    dest.mkdir(parents=True)
    state_path=ROOT/'ops/r9_fold0_handoff_backup_state.json'
    if state_path.exists():raise FileExistsError('Existing operator job must not be duplicated')
    raw=Path(tempfile.mkstemp(prefix='r9_boundary_handoff_transport_',suffix='.stdout',dir='/private/tmp')[1])
    command=['ssh','-tt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=accept-new',
             '-o','UserKnownHostsFile=/private/tmp/underwater_runpod_known_hosts',
             '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6',
             'j5u5bv4ewso40i-644111f6@ssh.runpod.io']
    stdin="stty -echo -onlcr\n/workspace/underwater_vessel_noise_r9_r1/venv/bin/python -u - <<'R9PY'\n"+remote_program()+"\nR9PY\nr9_backup_status=$?\nexit \"$r9_backup_status\"\n"
    state={'status':'waiting_guard','maximum_guard_wait_seconds':2400,'job_code_sha256':sha(__file__),
           'guard_sha256':GUARD_SHA,'started_unix':time.time(),'transport_log':str(raw),
           'original_run_or_processes_modified':False,'backup_dir':str(dest)}
    atomic(state_path,state)
    print(json.dumps(state),flush=True)
    try:
        with raw.open('wb') as f:
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=f,stderr=subprocess.STDOUT)
            child.stdin.write(stdin.encode());child.stdin.close()
            state['transport_pid']=child.pid;atomic(state_path,state)
            offset=0;carry=b'';stream_started=False
            while child.poll() is None:
                time.sleep(5)
                if not stream_started:
                    with raw.open('rb') as r:r.seek(offset);new=r.read();offset=r.tell()
                    carry+=new
                    lines=carry.split(b'\n');carry=lines.pop()
                    for line in lines:
                        v=line.strip()
                        if v.startswith(b'R9_WAIT_STATUS '):
                            status=json.loads(v[len(b'R9_WAIT_STATUS '):]);state['last_guard_status']=status
                            if status['status']=='paused_after_last_model_summary':state['status']='paused_verified_preparing_archive'
                        elif v==b'R9_BACKUP_DATA_BEGIN':stream_started=True;state['status']='receiving_completed_handoff'
                state['transport_received_bytes']=raw.stat().st_size;atomic(state_path,state)
        receipt=None;started=False;finished=False;archive=dest/'r9_fold00_itransformer_boundary_handoff_r1.tar.gz'
        with raw.open('rb') as f,archive.open('xb') as out:
            for line in f:
                v=line.strip()
                if v.startswith(b'R9_BACKUP_RECEIPT '):receipt=json.loads(v[len(b'R9_BACKUP_RECEIPT '):])
                elif v==b'R9_BACKUP_DATA_BEGIN':started=True
                elif v==b'R9_BACKUP_DATA_END':finished=True;break
                elif started:out.write(base64.b64decode(v,validate=True))
        if not receipt or not started or not finished or child.returncode!=0:
            raise RuntimeError('Guard/independent verification/transport failed; inspect '+str(raw))
        if sha(archive)!=receipt['archive_sha256'] or archive.stat().st_size!=receipt['archive_bytes']:
            raise ValueError('Transferred handoff archive SHA/size differs')
        extracted=dest/'restored';extracted.mkdir()
        with tarfile.open(archive,'r:gz') as t:
            entries=t.getmembers()
            for m in entries:
                if not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts:raise ValueError('Unsafe archive entry')
            t.extractall(extracted,filter='data')
        manifest=json.loads((extracted/'backup_manifest.json').read_text())
        if sha(extracted/'backup_manifest.json')!=receipt['manifest_sha256']:raise ValueError('Manifest differs')
        for n,h in manifest['file_sha256'].items():
            if sha(extracted/n)!=h:raise ValueError('Handoff artifact SHA differs: '+n)
        if len(entries)!=len(manifest['file_sha256'])+1:raise ValueError('Untracked archive entry')
        import numpy as np
        import torch
        from pipeline.models_r9 import make_r9_model
        torch.set_num_threads(1)
        folder=extracted/'fold_00/models/itransformer';summary=json.loads((folder/'summary.json').read_text())
        checks=[]
        for p in sorted(folder.rglob('*.pt')):
            saved=torch.load(p,map_location='cpu',weights_only=True)
            model=make_r9_model(saved['kind'],**saved['architecture']['model_settings'])
            if model.architecture()!=summary['architecture']:raise ValueError('Architecture differs')
            model.load_state_dict(saved['state_dict'],strict=True)
            if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite checkpoint weights')
            checks.append({'checkpoint':p.relative_to(folder).as_posix(),'epoch':saved['epoch'],'sha256':sha(p),
                           'strict_full_model_restore':True,'parameters_finite':True})
            del model,saved
        if len(checks)!=13:raise ValueError('All13 completed checkpoints required')
        with np.load(folder/'best_window_validation_predictions.npz',allow_pickle=False) as z:
            rmse=float(np.sqrt(np.mean((z['share'].astype(np.float64)-z['target_share'].astype(np.float64))**2)))
            if abs(rmse-summary['best_window_rmse'])>1e-12:raise ValueError('Saved selection RMSE differs')
        result={'format':'r9_independent_boundary_handoff_local_verification_v1','status':'passed',
                'remote_receipt':receipt,'archive_and_all_entry_SHA_verified':True,'entries':len(entries),
                'restored':str(extracted),'local_CPU_strict_restore_checkpoints':checks,
                'best_window_epoch':summary['best_window_epoch'],'best_window_rmse':rmse,
                'best_macro_epoch':summary['best_macro_epoch'],'best_macro_rmse':summary['best_site_macro_rmse'],
                'last_rmse':summary['last_validation_metrics']['rmse'],
                'all4_original_summary_sha256':manifest['all4_original_summary_sha256'],
                'original_train_itransformer_ledger_step_present':manifest['original_train_itransformer_ledger_step_present'],
                'original_fold_neural_completed_receipt_exists':manifest['original_fold_neural_completed_receipt_exists'],
                'original_root_neural_stage_completed_receipt_exists':manifest['original_root_neural_stage_completed_receipt_exists'],
                'original_ledger_edited_or_missing_receipts_fabricated':False,
                'all_four_Fold0_models_completed_independently':True,'all5_fold_run_completed':False,
                'future_combination_requires_fresh_version_verified_import':True,
                'previous_three_model_weights_duplicated':False,'signals_sent_by_this_job':False,
                'source_config_or_processes_changed':False,'model_fit_or_new_predictions_or_Test_report_evaluation':False,
                'elapsed_seconds':time.monotonic()-start}
        (dest/'collection_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
        report=ROOT/'reports/R9_FOLD0_BOUNDARY_HANDOFF_BACKUP.json'
        if report.exists():raise FileExistsError('Completed handoff report immutable')
        report.write_text(json.dumps(result,indent=2)+'\n')
        state.update(status='completed_verified_handoff',report=str(report),report_sha256=sha(report),
                     archive_sha256=receipt['archive_sha256'],elapsed_seconds=result['elapsed_seconds'])
        atomic(state_path,state);print(json.dumps(state),flush=True)
    except Exception as error:
        state.update(status='failed_no_signal_no_resume',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-start)
        atomic(state_path,state);print(json.dumps(state),flush=True);raise


if __name__=='__main__':main()
