#!/usr/bin/env python3
"""Back up completed Fold0 TFGrid/FDY, excluding CNN and active iTransformer.

Uses the same audited gateway transport/archive checks as the first CNN
backup. Remote changes are limited to new /base/backups folders. No signal,
training, evaluation, source edit, or active-model file access is performed.
"""
from __future__ import annotations
import base64
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

from ops.backup_r9_completed_cnn import REMOTE

ROOT=Path(__file__).resolve().parents[1]
KINDS=('tfgrid_fdy_conformer','fdy')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def collect(kind):
    start=time.monotonic()
    if kind not in KINDS:raise ValueError('Only newly completed Fold0 TFGrid/FDY authorized')
    dest=ROOT/'ops/server_backups/r9_gpu_r1'/f'fold_00_{kind}_completed_r1'
    if dest.exists():raise FileExistsError('Existing completed or partial local backup is not overwritten')
    dest.mkdir(parents=True)
    remote=REMOTE.replace('bg_cnn',kind).replace('r9_completed_CNN_backup_v1','r9_completed_neural_backup_v1')
    compile(remote,'remote-'+kind,'exec')
    raw=Path(tempfile.mkstemp(prefix='r9_'+kind+'_transport_',suffix='.stdout',dir='/private/tmp')[1])
    command=['ssh','-tt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=accept-new',
             '-o','UserKnownHostsFile=/private/tmp/underwater_runpod_known_hosts',
             '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6',
             'j5u5bv4ewso40i-644111f6@ssh.runpod.io']
    stdin="stty -echo -onlcr\n/workspace/underwater_vessel_noise_r9_r1/venv/bin/python -u - <<'R9PY'\n"+remote+"\nR9PY\nexit\n"
    with raw.open('wb') as f:
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=f,stderr=subprocess.STDOUT)
        print(json.dumps({'phase':'completed_model_audit_and_transfer','kind':kind,'transport_pid':child.pid,
                          'transport_stdout_file':str(raw)}),flush=True)
        child.communicate(stdin.encode(),timeout=1200)
    receipt=None;started=False;finished=False
    archive=dest/f'r9_fold00_{kind}_completed_r1.tar.gz'
    with raw.open('rb') as f,archive.open('xb') as out:
        for line in f:
            value=line.strip()
            if value.startswith(b'R9_BACKUP_RECEIPT '):receipt=json.loads(value[len(b'R9_BACKUP_RECEIPT '):])
            elif value==b'R9_BACKUP_DATA_BEGIN':started=True
            elif value==b'R9_BACKUP_DATA_END':finished=True;break
            elif started:out.write(base64.b64decode(value,validate=True))
    if not receipt or not started or not finished or child.returncode!=0:
        raise RuntimeError('Incomplete remote audit/transfer: '+str(raw))
    if sha(archive)!=receipt['archive_sha256'] or archive.stat().st_size!=receipt['archive_bytes']:
        raise ValueError('Transport SHA/size differs')
    import tarfile
    extracted=dest/'restored';extracted.mkdir()
    with tarfile.open(archive,'r:gz') as t:
        entries=t.getmembers()
        for m in entries:
            if not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts:
                raise ValueError('Unsafe archive entry')
        t.extractall(extracted,filter='data')
    manifest=json.loads((extracted/'backup_manifest.json').read_text())
    if manifest['kind']!=kind or manifest['epochs']!=100:raise ValueError('Completed model identity differs')
    if sha(extracted/'backup_manifest.json')!=receipt['manifest_sha256']:raise ValueError('Backup manifest SHA differs')
    for name,h in manifest['file_sha256'].items():
        if sha(extracted/name)!=h:raise ValueError('Entry SHA differs: '+name)
    if len(entries)!=len(manifest['file_sha256'])+1:raise ValueError('Unexpected archive entry')
    # Independent local strict architecture reconstruction, with no forward
    # inference and without loading source/evaluation observation arrays.
    import numpy as np
    import torch
    from pipeline.models_r9 import make_r9_model
    torch.set_num_threads(1)
    folder=extracted/f'fold_00/models/{kind}'
    summary=json.loads((folder/'summary.json').read_text())
    history=[json.loads(v) for v in (folder/'epochs.jsonl').read_text().splitlines()]
    for name,h in summary['code_sha256'].items():
        if sha(ROOT/name)!=h or sha(extracted/'frozen_code'/name)!=h:raise ValueError('Checkpoint code differs')
    if len(history)!=100 or any(v['validation_input_sha256']!=summary['validation_input_sha256'] for v in history):
        raise ValueError('Fixed validation input evidence differs')
    checks=[]
    for path in sorted(folder.rglob('*.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['kind']!=kind:raise ValueError('Checkpoint kind differs')
        model=make_r9_model(kind,**saved['architecture']['model_settings'])
        if model.architecture()!=summary['architecture'] or model.architecture()!=saved['architecture']:
            raise ValueError('Strict architecture differs')
        model.load_state_dict(saved['state_dict'],strict=True)
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite weights')
        if saved['validation_metrics']!=history[saved['epoch']-1]['validation']:raise ValueError('Epoch metrics differ')
        if path.name=='last.pt':
            if saved['epoch']!=100 or set(int(s['step'].item()) for s in saved['optimizer_state_dict']['state'].values())!={61700}:
                raise ValueError('Final epoch/optimizer steps differ')
        checks.append({'checkpoint':path.relative_to(folder).as_posix(),'epoch':saved['epoch'],
                       'sha256':sha(path),'strict_full_model_restore':True,'parameters_finite':True})
        del model,saved
    if len(checks)!=13:raise ValueError('Expected best2/last/10periodic checkpoints')
    with np.load(folder/'best_window_validation_predictions.npz',allow_pickle=False) as z:
        rmse=float(np.sqrt(np.mean((z['share'].astype(np.float64)-z['target_share'].astype(np.float64))**2)))
        if abs(rmse-summary['best_window_rmse'])>1e-12 or len(z['share'])!=1000 or len(set(z['sample_id']))!=1000:
            raise ValueError('Stored selection predictions/RMSE differ')
        if not all(str(v).startswith('r9_f00_selection_e001_') for v in z['sample_id']):raise ValueError('Wrong prediction role')
    result={'format':'r9_completed_model_local_backup_verification_v1','status':'passed','kind':kind,'fold':0,
            'remote_receipt':receipt,'archive_sha256_verified':True,'all_entry_sha256_verified':True,
            'entries':len(entries),'restored':str(extracted),'epochs':100,'optimizer_steps':61700,
            'parameters':summary['architecture']['total_parameters'],'local_CPU_strict_restore_checkpoints':checks,
            'best_window_epoch':summary['best_window_epoch'],'best_window_rmse':rmse,
            'best_macro_epoch':summary['best_macro_epoch'],'best_macro_rmse':summary['best_site_macro_rmse'],
            'last_rmse':summary['last_validation_metrics']['rmse'],
            'source_summary_sha256':sha(folder/'summary.json'),
            'context_sha256':sha(extracted/'run_metadata/fold_00_context.json'),
            'no_active_model_files_or_processes_changed':True,'no_source_or_Test_report_arrays_read':True,
            'model_training_or_new_predictions_performed':False,
            'elapsed_seconds':time.monotonic()-start,'transport_log':str(raw)}
    (dest/'collection_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'phase':'completed_model_backup_verified','kind':kind,'archive_bytes':receipt['archive_bytes'],
                      'archive_sha256':receipt['archive_sha256'],'entries':len(entries),'checkpoint_count':len(checks),
                      'best_epoch':summary['best_window_epoch'],'best_RMSE':rmse,'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    gc.collect()
    return result


def main():
    results=[]
    for kind in KINDS:results.append(collect(kind))
    out=ROOT/'reports/R9_FOLD0_ADDITIONAL_MODELS_BACKUP.json'
    if out.exists():raise FileExistsError('Immutable report already exists')
    report={'status':'passed','models':results,'previous_CNN_backup_modified_or_duplicated':False,
            'active_iTransformer_files_or_processes_touched':False,'source_or_config_modified':False,
            'Test_or_report_source_arrays_read':False,'gateway_session_used':'independent subprocesses; root51993 not used'}
    out.write_text(json.dumps(report,indent=2)+'\n')
    out.with_suffix('.md').write_text('# R9 Fold0 TFGrid·FDY 완료 백업\n\n'
        '**PASS.** 새로 완료된 TFGrid와FDY만 독립 gateway로 Mac에 회수했다. 기존 CNN 백업과 활성 iTransformer는 변경하지 않았다. '
        '각100epoch·61,700업데이트, best-window/best-macro/last 및10epoch 간격10개(총13개) 체크포인트, 학습로그/summary/selection예측/원장snapshot/동결코드를 보존했다. '
        'archive 및42개 entry SHA, 로컬 기본전체모델 strict restore/finite weights, 저장된selection예측의RMSE 재계산을 확인했다. 새 학습·추론·Test/report 평가는 없다.\n\n'
        + '\n'.join(f"- {r['kind']}: best-window epoch{r['best_window_epoch']}, RMSE {r['best_window_rmse']:.10f}; last100 {r['last_rmse']:.10f}." for r in results)
        +'\n\n이것은 두 완료 모델의 보존 결과이며 전체5fold 완료 또는 공식 성능을 뜻하지 않는다.\n\n'
        +f'[수치·파일 경로·원장·SHA]({out.name})\n')
    print(json.dumps({'status':'passed','report':str(out),'sha256':sha(out),'models':len(results)}),flush=True)


if __name__=='__main__':main()
