#!/usr/bin/env python3
"""Audit and copy only the completed R9 Fold0 CNN over an SSH gateway.

The remote archive is outside the running project. No process signalling,
training, predictions, report/Test access, or active-model file access.
"""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
REMOTE=r'''
from pathlib import Path
import base64,hashlib,json,math,sys,tarfile,time
import numpy as np
import torch
torch.set_num_threads(1)
base=Path('/workspace/underwater_vessel_noise_r9_r1')
project=base/'project';run=project/'models/group_cv_minute_full_r9_v1'
fold=run/'fold_00';model=fold/'models/bg_cnn'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
ledger_bytes=(fold/'artifact_ledger.json').read_bytes()
ledger=json.loads(ledger_bytes);summary=json.loads((model/'summary.json').read_text())
step=ledger['steps']['train/bg_cnn']
assert step['status']=='completed' and summary['status']=='completed'
assert summary['fixture_only'] is False and summary['epochs_run']==100
assert summary['stop_reason']=='completed_fixed_epochs' and summary['early_stopping_enabled'] is False
assert not (model/'FAILED.json').exists()
assert summary['settings']['epochs']==100 and summary['settings']['optimizer']=='adam'
assert summary['architecture']['model_settings']['dropout']==.1
actual={p.relative_to(fold).as_posix() for p in model.rglob('*') if p.is_file()}
assert actual==set(step['artifact_sha256'])
for n,h in step['artifact_sha256'].items():assert sha(fold/n)==h,n
for n,h in summary['artifact_sha256'].items():assert sha(model/n)==h,n
for n,h in ledger['code_sha256'].items():assert sha(project/n)==h,n
for n,h in summary['code_sha256'].items():assert sha(project/n)==h,n
history=[json.loads(v) for v in (model/'epochs.jsonl').read_text().splitlines()]
assert len(history)==100 and [v['epoch'] for v in history]==list(range(1,101))
window=min(history,key=lambda v:v['validation']['rmse'])
macro=min(history,key=lambda v:v['validation']['site_macro_rmse'])
assert window['epoch']==summary['best_window_epoch']
assert macro['epoch']==summary['best_macro_epoch']
assert window['validation']['rmse']==summary['best_window_rmse']
assert macro['validation']['site_macro_rmse']==summary['best_site_macro_rmse']
assert history[-1]['validation']==summary['last_validation_metrics']
assert summary['optimizer_steps']==61700 and history[-1]['optimizer_steps']==61700
checkpoints={}
for path in sorted(model.rglob('*.pt')):
 ck=torch.load(path,map_location='cpu',weights_only=True)
 for key in ['kind','architecture','settings','input_fingerprint','train_identity','validation_identity','code_sha256','fixture_only']:
  assert ck[key]==summary[key],(path.name,key)
 assert ck['validation_input_sha256']==summary['validation_input_sha256']
 assert ck['validation_metrics']==history[ck['epoch']-1]['validation']
 assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
 if path.name=='best_window.pt':assert ck['epoch']==window['epoch']
 if path.name=='best_macro.pt':assert ck['epoch']==macro['epoch']
 if path.name=='last.pt':
  assert ck['epoch']==100
  assert set(int(s['step'].item()) for s in ck['optimizer_state_dict']['state'].values())=={61700}
  assert 'scheduler_state_dict' in ck and 'cuda_rng_state_all' in ck
 if path.parent.name=='checkpoints':assert ck['epoch']==int(path.stem.split('_')[1])
 checkpoints[path.relative_to(model).as_posix()]={'sha256':sha(path),'epoch':ck['epoch'],'parameters_finite':True}
 del ck
assert len(checkpoints)==13
with np.load(model/'best_window_validation_predictions.npz',allow_pickle=False) as z:
 assert len(z['share'])==1000 and len(set(z['sample_id'].astype(str)))==1000
 assert np.isfinite(z['share']).all() and np.isfinite(z['score']).all()
 assert (z['share']>=0).all() and (z['share']<=1).all()
 assert (z['score']>=0).all() and (z['score']<=1).all()
 keys=z.files
backup=base/'backups/r9_fold00_bg_cnn_completed_r1'
assert not backup.exists();backup.mkdir(parents=True)
snapshot=backup/'fold_00_ledger_snapshot.json';snapshot.write_bytes(ledger_bytes)
payload={}
for p in sorted(model.rglob('*')):
 if p.is_file():payload['fold_00/models/bg_cnn/'+p.relative_to(model).as_posix()]=p
payload['run_metadata/execution_identity.json']=run/'execution_identity.json'
payload['run_metadata/fold_00_context.json']=fold/'context.json'
payload['run_metadata/fold_00_ledger_snapshot.json']=snapshot
for n in ledger['code_sha256']:payload['frozen_code/'+n]=project/n
audit={'format':'r9_completed_CNN_backup_v1','status':'passed','fold':0,'kind':'bg_cnn',
 'source_run':str(run),'ledger_snapshot_sha256':sha(snapshot),
 'summary_sha256':sha(model/'summary.json'),'epochs':100,'optimizer_steps':61700,
 'best_window_epoch':window['epoch'],'best_window_rmse':summary['best_window_rmse'],
 'best_macro_epoch':macro['epoch'],'best_macro_rmse':summary['best_site_macro_rmse'],
 'last_rmse':summary['last_validation_metrics']['rmse'],'checkpoint_checks':checkpoints,
 'prediction_keys':keys,'selection_rows':1000,'checkpoint_count':13,
 'file_sha256':{n:sha(p) for n,p in payload.items()},
 'file_bytes':{n:p.stat().st_size for n,p in payload.items()},
 'active_TFGrid_files_read':False,'model_fit_or_predictions_or_Test_report_access':False,
 'source_files_or_processes_changed':False,'ledger_is_timepoint_snapshot_not_completed_run':True}
manifest=backup/'backup_manifest.json';manifest.write_text(json.dumps(audit,indent=2)+'\n')
archive=backup/'r9_fold00_bg_cnn_completed_r1.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
 for n,p in payload.items():t.add(p,arcname=n,recursive=False)
 t.add(manifest,arcname='backup_manifest.json',recursive=False)
for n,h in step['artifact_sha256'].items():assert sha(fold/n)==h,n
receipt={'status':'passed','archive':str(archive),'archive_sha256':sha(archive),
 'archive_bytes':archive.stat().st_size,'payload_entries':len(payload),
 'manifest_sha256':sha(manifest),'summary_sha256':audit['summary_sha256'],
 'best_window_epoch':window['epoch'],'best_window_rmse':summary['best_window_rmse'],
 'best_macro_epoch':macro['epoch'],'last_rmse':audit['last_rmse']}
(backup/'remote_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('R9_BACKUP_RECEIPT '+json.dumps(receipt),flush=True)
print('R9_BACKUP_DATA_BEGIN',flush=True)
with archive.open('rb') as f:
 for b in iter(lambda:f.read(3*65536),b''):
  sys.stdout.buffer.write(base64.b64encode(b)+b'\n');sys.stdout.buffer.flush()
print('R9_BACKUP_DATA_END',flush=True)
'''


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def main():
    start=time.monotonic();dest=ROOT/'ops/server_backups/r9_gpu_r1/fold_00_bg_cnn_completed_r1'
    if dest.exists():raise FileExistsError('Fresh immutable backup destination required')
    dest.mkdir(parents=True)
    raw=Path(tempfile.mkstemp(prefix='r9_backup_transport_',suffix='.stdout',dir='/private/tmp')[1])
    command=['ssh','-tt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=accept-new',
             '-o','UserKnownHostsFile=/private/tmp/underwater_runpod_known_hosts',
             '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6',
             'j5u5bv4ewso40i-644111f6@ssh.runpod.io']
    stdin="stty -echo -onlcr\n/workspace/underwater_vessel_noise_r9_r1/venv/bin/python -u - <<'R9PY'\n"+REMOTE+"\nR9PY\nexit\n"
    with raw.open('wb') as f:
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=f,stderr=subprocess.STDOUT)
        print(json.dumps({'phase':'remote_audit_then_gateway_transfer','pid':child.pid,'transport_stdout_file':str(raw)}),flush=True)
        child.communicate(stdin.encode(),timeout=900)
    receipt=None;started=False;finished=False;archive=dest/'r9_fold00_bg_cnn_completed_r1.tar.gz'
    with raw.open('rb') as f,archive.open('xb') as out:
        for line in f:
            s=line.strip()
            if s.startswith(b'R9_BACKUP_RECEIPT '):receipt=json.loads(s[len(b'R9_BACKUP_RECEIPT '):])
            elif s==b'R9_BACKUP_DATA_BEGIN':started=True
            elif s==b'R9_BACKUP_DATA_END':finished=True;break
            elif started:out.write(base64.b64decode(s,validate=True))
    if not receipt or not started or not finished or child.returncode!=0:
        raise RuntimeError('Incomplete remote audit/transfer; transport log retained at '+str(raw))
    if sha(archive)!=receipt['archive_sha256'] or archive.stat().st_size!=receipt['archive_bytes']:
        raise ValueError('Transferred archive SHA/size differs')
    extracted=dest/'restored';extracted.mkdir()
    with tarfile.open(archive,'r:gz') as t:
        entries=t.getmembers()
        for m in entries:
            if not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts:
                raise ValueError('Nonregular/unsafe archive entry')
        t.extractall(extracted,filter='data')
    m=json.loads((extracted/'backup_manifest.json').read_text())
    if sha(extracted/'backup_manifest.json')!=receipt['manifest_sha256']:raise ValueError('Manifest SHA differs')
    for n,h in m['file_sha256'].items():
        if sha(extracted/n)!=h:raise ValueError('Restored payload SHA differs: '+n)
    if len(entries)!=len(m['file_sha256'])+1:raise ValueError('Unexpected archive entry')
    result={'format':'r9_completed_model_local_backup_verification_v1','status':'passed',
            'remote_receipt':receipt,'archive_sha256_verified':True,'all_entry_sha256_verified':True,
            'entries':len(entries),'restored':str(extracted),'source_summary_sha256':m['summary_sha256'],
            'checkpoint_count':m['checkpoint_count'],'all_checkpoint_metadata_and_finite_weights_remote_verified':True,
            '100_epochs_and_strict_best_minima_verified':True,
            'source_model_and_config_unmodified':True,'active_model_untouched':True,
            'elapsed_seconds':time.monotonic()-start,'transport_log':str(raw)}
    (dest/'collection_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    out=ROOT/'reports/R9_FOLD0_CNN_COMPLETED_BACKUP.json'
    if out.exists():raise FileExistsError('Immutable report exists')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','report':str(out),'report_sha256':sha(out),
                      'backup':str(dest),'archive_bytes':receipt['archive_bytes'],
                      'archive_sha256':receipt['archive_sha256'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
