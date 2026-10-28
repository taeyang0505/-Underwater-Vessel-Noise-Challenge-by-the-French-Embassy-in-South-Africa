"""Queued continuation after the first Fold0 CSV is validated, never before."""
from pathlib import Path
import argparse
import csv
import datetime
import fcntl
import html
import json
import math
import os
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.r16_multisupport_data import OUT, KINDS, REGIMES, read, write, sha
from pipeline import training_r16_multisupport as original
from pipeline.training_r16_multisupport_fivefold import MODELS, OPS, codes, jobs, folder_for, validate_completed

CPU = '/Users/hongtaeyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
MPS = '/Users/hongtaeyang/Documents/Codex/2026-10-27/new-chat/.venv/bin/python'
PARENT_OPS = ROOT / 'ops/r16_multisupport_full_r1'
PARENT_CSV = ROOT / 'submissions/r16_multisupport_full_r1/submission_r16.csv'


def verify_csv(path, expected_sha, expected_rows=1500):
    if sha(path) != expected_sha:
        raise ValueError('Submission SHA differs')
    with path.open(newline='') as f:
        rows = list(csv.reader(f))
    if rows[0] != ['ID', 'share_added_vessel', 'score_added_vessel']:
        raise ValueError('Submission columns differ')
    values = rows[1:]
    if len(values) != expected_rows or len({r[0] for r in values}) != expected_rows:
        raise ValueError('Submission unique row count differs')
    if any(len(r) != 3 or not r[0] or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1
                                       for v in r[1:]) for r in values):
        raise ValueError('Invalid submission values')
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
            'sha256': expected_sha, 'rows': expected_rows, 'verified': True}


def handoff():
    state = read(PARENT_OPS / 'state.json')
    done = original.MODELS / 'final/COMPLETED.json'
    if state['phase'] != 'completed' or not done.exists():
        raise ValueError('Fold0 run and its final CSV must finish first')
    r = read(done)
    if r['status'] != 'completed' or r['official_upload'] is not False:
        raise ValueError('Invalid Fold0 receipt')
    csv_receipt = verify_csv(PARENT_CSV, r['submission']['sha256'])
    from ops.finalize_r16_multisupport_full import prerequisites
    pins = prerequisites()
    original.verify(pins)
    for k in KINDS:
        for regime in REGIMES:
            validate_completed(folder_for(0, regime, k), 0, regime, k)
    return {'status': 'verified', 'initial_CSV': csv_receipt,
            'completion_receipt_sha256': sha(done), 'completed_model_artifact_sha256': pins,
            'does_not_reuse_partial_models': True, 'official_upload': False}


def plan():
    OPS.mkdir(parents=True, exist_ok=True)
    path = OPS / 'PLAN.json'
    if path.exists():
        raise FileExistsError('Plan already frozen')
    write(path, {'status': 'authorized_queued',
          'authorization': 'User: create Fold0 candidate CSV first, then train all remaining folds through4',
          'start_condition': 'Parent supervisor completed + CSV SHA/schema/1500 rows + all Fold0 artifacts verified',
          'new_jobs': [{'fold': f, 'kind': k, 'regime': r} for f, k, r in jobs()],
          'model_epochs': 3200, 'epochs_per_new_job': 100, 'optimizer': 'Adam', 'early_stopping': False,
          'warm_start': False, 'model_or_recipe_hyperparameters_changed': False,
          'D0_H0_weights': [0.5, 0.5], 'Tree_minimum_regression_weight': 0.5,
          'each_NN_architecture_maximum': 0.25, 'score_share_weight': 0,
          'cache_strategy': 'one fold at a time; delete only verified derived cache after all8 jobs complete',
          'initial_CSV_retained': str(PARENT_CSV.relative_to(ROOT)),
          'full5fold_CSV': 'submissions/r16_multisupport_fivefold_r1/submission_r16_5fold.csv',
          'data_manifest_sha256': sha(OUT / 'manifest.json'), 'code_sha256': codes(),
          'no_automatic_retry': True, 'official_upload': False})


def status(value):
    value = {**value, 'updated_kst': datetime.datetime.now().astimezone().isoformat(), 'official_upload': False}
    write(OPS / 'state.json', value)
    report = ROOT / 'reports/r16_multisupport_fivefold_r1'
    report.mkdir(parents=True, exist_ok=True)
    text = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>R16 전체5Fold 후속 실행</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:36px auto}pre{white-space:pre-wrap}</style><h1>R16 Fold0 CSV → Fold1~4 전체 학습</h1><p>현재 Fold0 실행을 변경하지 않고, 첫 CSV 검증 후 신경망32회(4모델×2조건×4Fold)를 순차 학습합니다. 각100epoch/Adam/noES, 독립 초기화. 공식 업로드 없음.</p><pre>'
    text += html.escape(json.dumps(value, ensure_ascii=False, indent=2)) + '</pre>'
    text += '<p><a href="../../submissions/r16_multisupport_full_r1/submission_r16.csv">첫 Fold0 CSV (완료 후)</a> · <a href="../../submissions/r16_multisupport_fivefold_r1/submission_r16_5fold.csv">전체5Fold CSV (완료 후)</a></p>'
    (report / 'START.html').write_text(text)


def run_stage(name, args, mps=False):
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
           'PYTORCH_ENABLE_MPS_FALLBACK': '0', 'DYLD_LIBRARY_PATH': '/opt/homebrew/opt/libomp/lib',
           'PYTHONPATH': '.:ops/r14_production_runtime_py' if mps else '.:/private/tmp/underwater-ml-deps:/private/tmp/underwater-eda-deps'}
    with (OPS / (name + '.log')).open('xb', buffering=0) as log:
        process = subprocess.Popen([MPS if mps else CPU, str(Path(__file__).resolve()), *args],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            progress = read(MODELS / 'progress.json') if (MODELS / 'progress.json').exists() else None
            status({'phase': name, 'PID': os.getpid(), 'child_PID': process.pid, 'progress': progress})
            time.sleep(15)
    if process.returncode:
        raise RuntimeError(f'{name} failed, exit={process.returncode}; no retry; see {name}.log')


def supervise():
    OPS.mkdir(parents=True, exist_ok=True)
    with (OPS / 'writer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OPS / 'STARTED.json').exists() or MODELS.exists():
            raise FileExistsError('Fresh continuation only')
        p = read(OPS / 'PLAN.json')
        if p['code_sha256'] != codes() or p['data_manifest_sha256'] != sha(OUT / 'manifest.json'):
            raise ValueError('Queued plan source/data changed')
        qa = read(OPS / 'CONNECTION_VERIFICATION.json')
        if qa['status'] != 'passed' or qa['code_sha256'] != codes():
            raise ValueError('Verified continuation wiring required')
        write(OPS / 'STARTED.json', {'PID': os.getpid(), 'status': 'waiting_for_fold0_CSV',
              'started_kst': datetime.datetime.now().astimezone().isoformat(), 'official_upload': False})
        keepawake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            stale = 0
            while True:
                original.verify(p['code_sha256'])
                parent = read(PARENT_OPS / 'state.json')
                if parent['phase'] in ('failed', 'paused') or (PARENT_OPS / 'FAILED.json').exists():
                    raise RuntimeError('Parent failed/paused; no follow-up training started')
                if parent['phase'] == 'completed':
                    break
                stale = stale + 1 if time.time() - (PARENT_OPS / 'state.json').stat().st_mtime > 180 else 0
                if stale >= 4:
                    raise RuntimeError('Parent heartbeat stale across four checks; no duplicate/retry')
                status({'phase': 'waiting_for_fold0_CSV', 'PID': os.getpid(), 'parent': parent})
                time.sleep(30)
            receipt = handoff()
            write(OPS / 'FOLD0_HANDOFF.json', receipt)
            MODELS.mkdir()
            for fold in range(1, 5):
                run_stage(f'fold{fold}_cache', ['cache', '--fold', str(fold)])
                for kind in KINDS:
                    for regime in REGIMES:
                        run_stage(f'fold{fold}_{regime}_{kind}', ['train', '--fold', str(fold),
                                  '--kind', kind, '--regime', regime], mps=True)
                run_stage(f'fold{fold}_release_cache', ['release', '--fold', str(fold)])
            write(MODELS / 'ALL_NEURAL_COMPLETED.json', {'status': 'completed', 'new_folds': [1, 2, 3, 4],
                  'fold0_source': str(original.MODELS.relative_to(ROOT)), 'code_sha256': codes(),
                  'data_manifest_sha256': sha(OUT / 'manifest.json')})
            run_stage('finalize', ['finalize'])
            verify_csv(PARENT_CSV, receipt['initial_CSV']['sha256'])
            status({'phase': 'completed', 'submission': 'submissions/r16_multisupport_fivefold_r1/submission_r16_5fold.csv'})
        except BaseException as e:
            failure = {'phase': 'failed', 'error': repr(e), 'traceback': traceback.format_exc(), 'no_automatic_retry': True}
            write(OPS / 'FAILED.json', failure); status(failure)
            raise
        finally:
            keepawake.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('plan', 'supervise', 'cache', 'train', 'release', 'finalize'))
    parser.add_argument('--fold', type=int, choices=range(1, 5))
    parser.add_argument('--kind', choices=KINDS)
    parser.add_argument('--regime', choices=REGIMES)
    args = parser.parse_args()
    if args.stage == 'plan': plan()
    elif args.stage == 'supervise': supervise()
    elif args.stage == 'cache':
        from pipeline.training_r16_multisupport_fivefold import cache_fold
        cache_fold(args.fold)
    elif args.stage == 'train':
        from pipeline.training_r16_multisupport_fivefold import train_one
        train_one(args.fold, args.kind, args.regime)
    elif args.stage == 'release':
        from pipeline.training_r16_multisupport_fivefold import clean_cache
        clean_cache(args.fold)
    else:
        from ops.finalize_r16_multisupport_fivefold import finalize
        finalize()


if __name__ == '__main__':
    main()
