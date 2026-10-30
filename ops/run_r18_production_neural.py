#!/usr/bin/env python3
"""R18 eight-job supervisor / explicit preflight / epoch-boundary resume CLI."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import training_r18_production_neural as r18


def supervise():
    r18.OPS.mkdir(parents=True, exist_ok=True)
    with (r18.OPS / 'supervisor.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('An R18 neural supervisor is already active')
        plan = r18.plan_read()
        jobs = [(family, regime, task) for family in r18.FAMILIES
                for regime in r18.REGIMES for task in r18.TASKS]
        for number, (family, regime, task) in enumerate(jobs, 1):
            folder = r18.RUN / family / regime / task
            if (folder / 'summary.json').exists():
                continue
            if (r18.OPS / 'PAUSE_AFTER_EPOCH').exists():
                return 20
            pf = r18.OPS / f'PREFLIGHT_{family}.json'
            commands = []
            if not pf.exists():
                commands.append(('preflight', [sys.executable, str(Path(__file__)), 'preflight', '--family', family]))
            commands.append(('train', [sys.executable, str(Path(__file__)), 'train', '--family', family,
                                      '--regime', regime, '--task', task]))
            for phase, cmd in commands:
                log_path = r18.OPS / f'{family}_{regime}_{task}_{phase}.log'
                with log_path.open('a', buffering=1) as log:
                    child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, start_new_session=True)
                    while child.poll() is None:
                        r18.write(r18.OPS / 'training_health.json', {
                            'status': 'running', 'time_unix': time.time(), 'supervisor_pid': os.getpid(),
                            'child_pid': child.pid, 'job_number': number, 'job_total': len(jobs),
                            'family': family, 'regime': regime, 'task': task, 'phase': phase,
                            'log_path': str(log_path.relative_to(ROOT)),
                            'data_manifest_sha256': plan['data_manifest_sha256']})
                        time.sleep(15)
                    code = child.returncode
                if code:
                    r18.write(r18.OPS / 'training_health.json', {
                        'status': 'paused' if code == 20 else 'failed', 'time_unix': time.time(),
                        'family': family, 'regime': regime, 'task': task, 'phase': phase,
                        'exit_code': code, 'log_path': str(log_path.relative_to(ROOT)),
                        'resume': 'same all command; epoch-boundary state is preserved'})
                    return code
        summaries = {f'{f}/{r}/{t}': r18.read(r18.RUN / f / r / t / 'summary.json')
                     for f, r, t in jobs}
        result = {'status': 'completed', 'jobs': summaries,
                  'time_unix': time.time(), 'Test_used': False,
                  'data_manifest_sha256': plan['data_manifest_sha256'],
                  'source_sha256': plan['source_sha256'],
                  'next': 'frozen checkpoints are ready for fresh_fit/gate/report/Test inference'}
        r18.write(r18.OPS / 'ALL_NEURAL_COMPLETED.json', result)
        r18.write(r18.REPORT / 'ALL_NEURAL_COMPLETED.json', result)
        r18.write(r18.OPS / 'training_health.json', {'status': 'completed', 'time_unix': time.time(), 'jobs': 8})
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('plan', 'preflight', 'train', 'all', 'predict', 'status'))
    p.add_argument('--manifest', default=str(r18.DATA))
    p.add_argument('--device', choices=('cuda', 'mps', 'cpu'), default='cuda')
    p.add_argument('--microbatch', type=int, default=16)
    p.add_argument('--effective-batch', type=int, default=64)
    p.add_argument('--family', choices=r18.FAMILIES)
    p.add_argument('--regime', choices=r18.REGIMES)
    p.add_argument('--task', choices=r18.TASKS)
    p.add_argument('--input')
    p.add_argument('--output')
    p.add_argument('--masks', action='store_true')
    a = p.parse_args()
    if a.action == 'plan':
        value = r18.freeze_plan(a.manifest, device=a.device, microbatch=a.microbatch,
                                effective_batch=a.effective_batch)
    elif a.action == 'preflight':
        if not a.family:
            p.error('--family required')
        value = r18.preflight(a.family)
    elif a.action == 'train':
        if not all((a.family, a.regime, a.task)):
            p.error('--family --regime --task required')
        folder = r18.RUN / a.family / a.regime / a.task
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'job.lock').open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('This exact R18 training job is already active')
            return r18.train_job(a.family, a.regime, a.task)
    elif a.action == 'all':
        return supervise()
    elif a.action == 'predict':
        if not all((a.family, a.regime, a.task, a.input, a.output)):
            p.error('--family --regime --task --input --output required')
        value = r18.predict_npz(a.family, a.regime, a.task, a.input, a.output,
                               device=a.device, masks=a.masks)
    else:
        value = {name: r18.read(r18.OPS / name) if (r18.OPS / name).exists() else None
                 for name in ('training_health.json', 'progress.json', 'ALL_NEURAL_COMPLETED.json')}
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        r18.write(r18.OPS / 'LATEST_ERROR.json', {'error': type(error).__name__,
                  'message': str(error), 'time_unix': time.time(),
                  'traceback': traceback.format_exc()})
        raise
