"""Read-only R16 controller supervision and SHA-verified completed-job backups.

This helper never starts/stops/restarts training or edits its files. Its health,
index and ZIP outputs are written exclusively under the explicit --ops-root.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time
import zipfile

RUN_REL = Path('models/r16_pretrained_cuda_bf16_r1/both')
TRAIN_OPS_REL = Path('ops/r16_pretrained_cuda_bf16_r1')
META = '__metadata__/manifest.json'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        # A zombie cannot continue training even though kill(pid, 0) succeeds.
        stat = Path(f'/proc/{pid}/stat')
        return not (stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z')
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def gpu_health():
    try:
        process = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw',
             '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=8)
        return {'available': process.returncode == 0,
                'fields': ['name', 'utilization_percent', 'memory_used_MiB', 'memory_total_MiB', 'power_W'],
                'rows': process.stdout.strip().splitlines(),
                'error': process.stderr.strip()[:500] if process.returncode else None}
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {'available': False, 'error': type(error).__name__}


def memory_health():
    result = {}
    for name in ('memory.current', 'memory.max', 'memory.peak', 'memory.events'):
        path = Path('/sys/fs/cgroup') / name
        if path.is_file():
            value = path.read_text().strip()
            result[name] = int(value) if value.isdigit() else value
    if not result:
        for name in ('memory.usage_in_bytes', 'memory.limit_in_bytes', 'memory.max_usage_in_bytes', 'memory.failcnt'):
            path = Path('/sys/fs/cgroup/memory') / name
            if path.is_file():
                result[name] = path.read_text().strip()
    return result


def job_manifest(root, folder):
    root, folder = root.resolve(), folder.resolve()
    if not folder.is_relative_to((root / RUN_REL).resolve()):
        raise ValueError('Job escaped approved run')
    summary_path = folder / 'summary.json'
    summary = read(summary_path)
    if not summary or summary.get('status') != 'completed' or (folder / 'FAILED.json').exists():
        raise ValueError('Only completed unfailed jobs may be backed up')
    expected = summary.get('artifact_sha256')
    if not isinstance(expected, dict) or not {'best.pt', 'PLAN.json', 'epochs.jsonl', 'selection.npz'} <= set(expected):
        raise ValueError('Incomplete completed-job SHA ledger')
    files = {}
    for name, digest in {**expected, 'summary.json': sha(summary_path)}.items():
        path = folder / name
        if Path(name).name != name or path.is_symlink() or path.resolve().parent != folder or not path.is_file():
            raise ValueError('Unsafe or missing artifact path')
        actual = sha(path)
        if actual != digest:
            raise ValueError('Completed artifact SHA differs: ' + name)
        files[str(path.relative_to(root))] = {'sha256': actual, 'bytes': path.stat().st_size}
    return {'format': 'r16_completed_job_backup_v1', 'job': str(folder.relative_to(root / RUN_REL)),
            'summary_sha256': sha(summary_path), 'files': files}, summary


def verify_zip(path, expected_manifest=None):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Backup ZIP CRC failed')
        manifest = json.loads(archive.read(META))
        if expected_manifest is not None and manifest != expected_manifest:
            raise ValueError('Backup manifest differs from completed source ledger')
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate backup entries')
        if set(archive.namelist()) != set(manifest['files']) | {META}:
            raise ValueError('Backup entry closure differs')
        for name, record in manifest['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe backup entry')
            h, size = hashlib.sha256(), 0
            with archive.open(name) as stream:
                for part in iter(lambda: stream.read(2**20), b''):
                    h.update(part)
                    size += len(part)
            if h.hexdigest() != record['sha256'] or size != record['bytes']:
                raise ValueError('Backup entry SHA/size differs: ' + name)
    return {'sha256': sha(path), 'bytes': Path(path).stat().st_size,
            'entry_count': len(manifest['files']) + 1, 'CRC_verified': True, 'entry_SHA_verified': True}


def backup_completed(root, folder, out):
    manifest, summary = job_manifest(root, folder)
    name = manifest['job'].replace('/', '_') + '_' + manifest['summary_sha256'][:16] + '.zip'
    destination = out / 'backups' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_name(destination.name + '.tmp')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr(META, json.dumps(manifest, ensure_ascii=False, sort_keys=True))
            for relative in manifest['files']:
                archive.write(root / relative, arcname=relative)
        verify_zip(temporary, manifest)
        temporary.replace(destination)
    receipt = verify_zip(destination, manifest)
    return {**receipt, 'path': str(destination), 'job': manifest['job'],
            'summary_sha256': manifest['summary_sha256'], 'verified_at': now(),
            'best_epoch': summary.get('best_epoch'), 'epochs_run': summary.get('epochs_run'),
            'best_cv': summary.get('best_cv'), 'stop_reason': summary.get('stop_reason')}


def scan(root, out, controller_pid):
    train_ops = root / TRAIN_OPS_REL
    state = read(train_ops / 'state.json', {})
    process = read(train_ops / 'process.json', {})
    index = read(out / 'backup_index.json', {'format': 'r16_completed_backup_index_v1', 'jobs': {}})
    completed, present, errors = [], [], []
    for folder in sorted((root / RUN_REL).glob('*/*/*')):
        if not folder.is_dir():
            continue
        key = str(folder.relative_to(root / RUN_REL))
        summary = read(folder / 'summary.json', {})
        present.append({'job': key, 'status': summary.get('status', 'partial_or_in_progress')})
        if summary.get('status') != 'completed':
            continue
        try:
            receipt = backup_completed(root, folder, out)
            previous = index['jobs'].get(key)
            if previous and previous['summary_sha256'] != receipt['summary_sha256']:
                raise ValueError('Previously backed-up completed summary changed')
            index['jobs'][key] = receipt
            completed.append(receipt)
        except Exception as error:
            errors.append({'job': key, 'error': repr(error)})
    index['updated_at'] = now()
    write(out / 'backup_index.json', index)
    controller_alive = alive(controller_pid)
    phase = ('monitoring' if controller_alive else
             'completed' if state.get('phase') == 'all_selected_neural_completed' and len(completed) == 12 else
             'controller_exited_without_full_completion')
    health = {'updated_at': now(), 'phase': phase, 'controller_pid': controller_pid,
              'controller_alive': controller_alive, 'worker_pid': process.get('worker_pid'),
              'worker_alive': alive(process.get('worker_pid')), 'training_state': state,
              'latest_epoch': read(train_ops / 'progress.json', {}),
              'training_failure': read(train_ops / 'LAST_FAILURE.json'),
              'GPU': gpu_health(), 'cgroup_memory': memory_health(),
              'present_jobs': present, 'completed_jobs': completed, 'completed_job_count': len(completed),
              'backup_errors': errors, 'training_or_plan_modified': False,
              'automatic_restart': False, 'official_upload': False}
    write(out / 'health.json', health)
    return health


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--controller-pid', type=int, required=True)
    parser.add_argument('--ops-root', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.controller_pid <= 0 or args.interval < 1:
        parser.error('Positive PID and interval >= 1 required')
    root, out = args.project_root.resolve(), args.ops_root.resolve()
    if not root.is_dir() or out.is_relative_to(root / 'models') or out == root / TRAIN_OPS_REL:
        parser.error('Existing project and separate supervision output directory required')
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            health = scan(root, out, args.controller_pid)
            print(json.dumps({k: health[k] for k in ('updated_at', 'phase', 'completed_job_count', 'backup_errors')}, ensure_ascii=False), flush=True)
            if args.once or not health['controller_alive']:
                write(out / 'SUPERVISOR_EXIT.json', {k: health[k] for k in
                      ('updated_at', 'phase', 'controller_alive', 'completed_job_count', 'backup_errors')})
                return 0 if health['phase'] == 'completed' or args.once else 1
            time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
