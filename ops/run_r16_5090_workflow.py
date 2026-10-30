"""One-shot approved R16 CUDA training, frozen finalization and local ZIP backup.

Operational wrapper only: no model/data/plan edits, retry, Pod control or upload.
The existing immutable training runner and finalizer enforce their own gates.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN_REL = Path('models/r16_pretrained_cuda_bf16_r1')
PLAN_REL = Path('ops/r16_pretrained_cuda_bf16_r1/RUN_PLAN.json')
CSV_REL = Path('submissions/r16_pretrained_cuda_bf16_r1/both/submission_r16_pretrained_cuda_bf16.csv')
META = '__metadata__/manifest.json'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_paths(root, out, python):
    if not out.is_absolute() or not python.is_absolute():
        raise ValueError('--ops-root and --python must be absolute paths')
    # Keep venv/bin/python as the invoked executable. Resolving its symlink to
    # the system interpreter silently drops the virtual environment packages.
    out = out.resolve()
    python = Path(os.path.abspath(os.fspath(python)))
    if not root.is_dir() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('Existing project and executable Python required')
    if out.is_relative_to(root):
        raise ValueError('Operational outputs must be outside the frozen project')
    return out, python


def validate_scope(plan):
    expected = {'folds': [0], 'total_jobs': 12, 'variants': ['both'],
                'regimes': ['D0', 'H0'], 'tasks': ['regression', 'classification'],
                'full_5fold_neural_training': False, 'scratch_folds1to4_cancelled': True}
    for name, value in expected.items():
        if plan.get(name) != value:
            raise ValueError('Approved one-split masked scope differs: ' + name)
    if (set(plan.get('families', [])) != {'dasheng_base', 'dinov2_small', 'timemoe'}
            or len(plan.get('families', [])) != 3):
        raise ValueError('Exactly the three approved pretrained model families required')


def final_backup(root, out, plan_hash):
    completed = root / RUN_REL / 'both/final/COMPLETED.json'
    if read(completed).get('status') != 'completed' or not (root / CSV_REL).is_file():
        raise ValueError('Verified completed finalizer and submission file required')
    if sha(root / PLAN_REL) != plan_hash:
        raise ValueError('Immutable run plan changed during workflow')
    sources = {}
    for folder in (root / RUN_REL, root / PLAN_REL.parent):
        for path in sorted(folder.rglob('*')):
            if path.is_symlink():
                raise ValueError('Symlink in final artifact closure')
            if path.is_file():
                sources[str(path.relative_to(root))] = path
    sources[str(CSV_REL)] = root / CSV_REL
    for name in ('controller.log', 'finalizer.log'):
        sources['__workflow__/' + name] = out / name
    manifest = {'format': 'r16_complete_workflow_backup_v1', 'created_at': now(),
                'run_plan_sha256': plan_hash, 'official_upload': False,
                'files': {name: {'sha256': sha(path), 'bytes': path.stat().st_size}
                          for name, path in sources.items()}}
    destination = out / 'r16_pretrained_cuda_bf16_completed.zip'
    temporary = destination.with_name(destination.name + '.tmp')
    if destination.exists() or temporary.exists():
        raise FileExistsError('Fresh final archive required; no partial reuse')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.writestr(META, json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        for name, path in sources.items():
            archive.write(path, arcname=name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('Final ZIP CRC failed')
        if (len(archive.namelist()) != len(set(archive.namelist()))
                or set(archive.namelist()) != set(manifest['files']) | {META}
                or json.loads(archive.read(META)) != manifest):
            raise ValueError('Final ZIP entry/manifest closure differs')
        for name, expected in manifest['files'].items():
            h, size = hashlib.sha256(), 0
            with archive.open(name) as stream:
                for part in iter(lambda: stream.read(2**20), b''):
                    h.update(part)
                    size += len(part)
            if h.hexdigest() != expected['sha256'] or size != expected['bytes']:
                raise ValueError('Final ZIP entry SHA/size differs: ' + name)
    temporary.replace(destination)
    receipt = {'status': 'verified', 'path': str(destination), 'sha256': sha(destination),
               'bytes': destination.stat().st_size, 'entry_count': len(manifest['files']) + 1,
               'CRC_verified': True, 'entry_SHA_verified': True,
               'run_plan_sha256': plan_hash, 'submission_sha256': sha(root / CSV_REL)}
    write(out / 'final_backup_receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ops-root', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    out, python = validate_paths(ROOT, args.ops_root, args.python)
    plan_path = ROOT / PLAN_REL
    plan = read(plan_path)
    validate_scope(plan)
    plan_hash = sha(plan_path)
    out.mkdir(parents=True, exist_ok=True)
    state_path, lock = out / 'workflow_state.json', out / 'workflow.lock'
    if state_path.exists():
        raise FileExistsError('Existing workflow state requires explicit inspection, never automatic retry')
    for name in ('controller.log', 'supervisor.log', 'finalizer.log'):
        if (out / name).exists():
            raise FileExistsError('Fresh workflow logs required before launching any child')
    for name in ('run_r16_pretrained_cuda_bf16.py', 'supervise_r16_pretrained_cuda_outputs.py',
                 'finalize_r16_pretrained_cuda_bf16.py'):
        if not (ROOT / 'ops' / name).is_file():
            raise FileNotFoundError('Required existing workflow component absent: ' + name)
    descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    state = {'created_at': now(), 'updated_at': now(), 'phase': 'launching_controller',
             'wrapper_pid': os.getpid(), 'project_root': str(ROOT),
             'run_plan_sha256': plan_hash, 'total_jobs': 12, 'folds': [0],
             'automatic_retry': False, 'official_upload': False, 'Pod_actions': False}
    supervisor_log = None
    try:
        write(state_path, state)
        command = [str(python), str(ROOT / 'ops/run_r16_pretrained_cuda_bf16.py'), 'run']
        with (out / 'controller.log').open('x') as controller_log:
            controller = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                          stdout=controller_log, stderr=subprocess.STDOUT)
            state.update(phase='training', controller_pid=controller.pid, updated_at=now())
            write(state_path, state)
            supervisor_log = (out / 'supervisor.log').open('x')
            supervisor = subprocess.Popen(
                [str(python), str(ROOT / 'ops/supervise_r16_pretrained_cuda_outputs.py'),
                 '--project-root', str(ROOT), '--controller-pid', str(os.getpid()),
                 '--ops-root', str(out / 'supervision'), '--interval', '30'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=supervisor_log,
                stderr=subprocess.STDOUT, start_new_session=True)
            state.update(supervisor_pid=supervisor.pid, updated_at=now())
            write(state_path, state)
            code = controller.wait()
        state.update(controller_exit_code=code, updated_at=now())
        if code:
            state['phase'] = 'training_failed_or_paused'
            write(state_path, state)
            return code
        if sha(plan_path) != plan_hash:
            raise ValueError('Immutable plan changed before finalization')
        state.update(phase='finalizing', updated_at=now())
        write(state_path, state)
        with (out / 'finalizer.log').open('x') as finalizer_log:
            finalizer = subprocess.Popen(
                [str(python), str(ROOT / 'ops/finalize_r16_pretrained_cuda_bf16.py'),
                 'run', '--variant', 'both', '--device', 'cuda', '--batch', '32',
                 '--catboost-root', 'models/r16_catboost_peft_cpu8_r1'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=finalizer_log, stderr=subprocess.STDOUT)
            state.update(finalizer_pid=finalizer.pid, updated_at=now())
            write(state_path, state)
            code = finalizer.wait()
        state.update(finalizer_exit_code=code, updated_at=now())
        if code:
            state['phase'] = 'finalization_failed'
            write(state_path, state)
            return code
        state.update(phase='backing_up_completed_artifacts', updated_at=now())
        write(state_path, state)
        receipt = final_backup(ROOT, out, plan_hash)
        state.update(phase='completed', updated_at=now(), final_backup=receipt,
                     submission_path=str(ROOT / CSV_REL))
        write(state_path, state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        return 0
    except BaseException as error:
        state.update(phase='workflow_failed', updated_at=now(), error=repr(error),
                     traceback=traceback.format_exc())
        write(state_path, state)
        raise
    finally:
        if supervisor_log is not None:
            supervisor_log.close()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
