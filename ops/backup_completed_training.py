"""Snapshot only SHA-verified completed ledger artifacts; never launch a job.

Run on the pod, then download the returned ZIP from ops/backups. Only stdlib
is needed. Source RUN files, training settings and vendor weights are untouched.
"""
from pathlib import Path, PurePosixPath
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import tempfile
import uuid
import zipfile

DEFAULT_RUN = Path('/workspace/underwater_vessel_noise_r1/project/models/group_cv_physics_v2_r1_adam')
DEFAULT_BACKUPS = Path(__file__).resolve().parent / 'backups'
MANIFEST = '__snapshot_manifest.json'
FORBIDDEN = {'pretrained', 'raw', 'external', 'data', 'calibration', 'sealed_holdout',
             'noaa_sanctsound_holdout', '__pycache__'}


class BackupError(ValueError):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''): h.update(block)
    return h.hexdigest()


def encode(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode()


def relative_name(name):
    if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
        raise BackupError('Invalid artifact path')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or p.as_posix() != name or re.match(r'^[A-Za-z]:', name):
        raise BackupError('Artifact path escape or noncanonical path: ' + name)
    if FORBIDDEN.intersection(v.lower() for v in p.parts) or p.name.lower() in {'test.csv', 'sample_submission.csv', 'calibration.npz', 'sealed_holdout.npz', 'domain_holdout.npz'}:
        raise BackupError('Forbidden source/vendor/Test payload: ' + name)
    if p.suffix.lower() in {'.wav', '.flac', '.mp3', '.ckpt', '.safetensors'}:
        raise BackupError('Raw/vendor payload is excluded: ' + name)
    return name


def regular_path(root, name):
    relative_name(name)
    path = root
    for part in PurePosixPath(name).parts:
        path = path / part
        if path.is_symlink(): raise BackupError('Symlink artifact path rejected: ' + name)
    if not path.resolve().is_relative_to(root): raise BackupError('Artifact leaves RUN: ' + name)
    if not path.is_file(): raise FileNotFoundError(name)
    return path


def read_json(root, name):
    path = regular_path(root, name)
    data = path.read_bytes()
    obj = json.loads(data)
    if not isinstance(obj, dict): raise BackupError('Expected JSON object: ' + name)
    return obj, data


def collect(run):
    files, virtual, ledger_views, skipped = {}, {}, [], []
    completed_steps = 0
    for folder in sorted(run.glob('fold_*')):
        if folder.is_symlink(): raise BackupError('Symlink fold rejected: ' + folder.name)
        if not re.fullmatch(r'fold_\d+', folder.name) or not folder.is_dir(): continue
        ledger_name = folder.name + '/artifact_ledger.json'
        if (folder / 'artifact_ledger.json').is_symlink(): raise BackupError('Symlink ledger rejected: ' + ledger_name)
        if not (folder / 'artifact_ledger.json').exists(): continue
        try:
            ledger, raw = read_json(run, ledger_name)
        except (json.JSONDecodeError, FileNotFoundError):
            skipped.append({'fold': folder.name, 'reason': 'ledger incomplete or absent'}); continue
        selected = {}
        for label, entries in ledger.items():
            if not isinstance(label, str) or not isinstance(entries, dict) or not entries:
                raise BackupError('Malformed completed-step ledger entry')
            checked, reason = {}, None
            for name, expected in entries.items():
                name = relative_name(name)
                if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
                    raise BackupError('Invalid expected artifact SHA')
                rel = folder.name + '/' + name
                try:
                    path = regular_path(run, rel)
                    if digest(path) != expected: reason = 'SHA mismatch: ' + rel; break
                except FileNotFoundError:
                    reason = 'recorded file missing: ' + rel; break
                checked[rel] = expected
            if reason is not None:
                skipped.append({'fold': folder.name, 'step': label, 'reason': reason}); continue
            for rel, expected in checked.items():
                if rel in files and files[rel] != expected: raise BackupError('Conflicting ledger hashes: ' + rel)
                files[rel] = expected
            selected[label] = entries
            completed_steps += 1
        # A filtered ledger states exactly which completed steps this ZIP can
        # supply. Source ledger growth is allowed, existing selected steps are
        # checked again before committing the snapshot.
        if selected:
            virtual[ledger_name] = encode(selected)
            ledger_views.append({'path': ledger_name, 'source_sha256': hashlib.sha256(raw).hexdigest(), 'selected': selected})
    for name in ['execution_identity.json', 'cv_report.json']:
        if not (run / name).exists(): continue
        try:
            _, raw = read_json(run, name)
        except (json.JSONDecodeError, FileNotFoundError):
            skipped.append({'path': name, 'reason': 'global JSON incomplete'}); continue
        files[name] = hashlib.sha256(raw).hexdigest()
    return files, virtual, ledger_views, skipped, completed_steps


def snapshot(run, backup_dir):
    if run.is_symlink(): raise BackupError('Symlink RUN is rejected')
    if not run.exists():
        return {'status': 'noop', 'reason': 'RUN does not exist; no training job started', 'run': str(run)}
    run = run.resolve()
    if not run.is_dir(): raise BackupError('RUN must be a directory')
    if backup_dir.is_symlink(): raise BackupError('Symlink backup directory is rejected')
    backup_dir = backup_dir.resolve()
    if backup_dir.is_relative_to(run): raise BackupError('Backups must remain outside source RUN')
    backup_dir.mkdir(parents=True, exist_ok=True)
    with (backup_dir / '.backup.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'noop', 'reason': 'Another backup process already holds the lock'}
        files, virtual, ledgers, skipped, steps = collect(run)
        if not files:
            return {'status': 'noop', 'reason': 'No verified completed artifacts/global JSON yet', 'skipped': skipped}
        entries = [{'relative_path': name, 'archive_path': name,
                    'sha256': expected, 'bytes': regular_path(run, name).stat().st_size,
                    'origin': 'completed_ledger' if name.startswith('fold_') else 'global_json'}
                   for name, expected in sorted(files.items())]
        entries += [{'relative_path': name, 'archive_path': name, 'sha256': hashlib.sha256(raw).hexdigest(),
                     'bytes': len(raw), 'origin': 'filtered_completed_ledger'} for name, raw in sorted(virtual.items())]
        if len({e['archive_path'] for e in entries}) != len(entries): raise BackupError('Duplicate snapshot path')
        created = datetime.datetime.now(datetime.timezone.utc)
        stamp = created.strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]
        final = backup_dir / ('snapshot_' + stamp + '.zip')
        partial = final.with_suffix('.zip.partial')
        metadata = {'format': 'completed_training_snapshot_v1', 'created_utc': created.isoformat(),
            'run_directory_hint': str(run), 'paths_relative_to': 'RUN', 'files': entries,
            'completed_steps': steps, 'skipped': skipped,
            'source_ledgers': [{'path': item['path'], 'source_sha256': item['source_sha256'],
                                'included_steps': list(item['selected'])} for item in ledgers],
            'restoration': 'Extract listed archive_path entries into a fresh local RUN directory. Review skipped steps before any resume; a partial snapshot is not an automatically resumable complete job.',
            'unrecorded_in_progress_files_excluded': True, 'vendor_weights_or_source_data_included': False,
            'training_started_by_backup': False}
        try:
            with partial.open('xb') as handle:
                with zipfile.ZipFile(handle, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
                    for entry in entries:
                        name = entry['archive_path']
                        if name in virtual:
                            archive.writestr(name, virtual[name]); continue
                        h = hashlib.sha256(); count = 0
                        with regular_path(run, name).open('rb') as src, archive.open(name, 'w', force_zip64=True) as dst:
                            for block in iter(lambda: src.read(2 ** 20), b''):
                                dst.write(block); h.update(block); count += len(block)
                        if count != entry['bytes'] or h.hexdigest() != entry['sha256']:
                            raise BackupError('Artifact changed while copying: ' + name)
                    archive.writestr(MANIFEST, encode(metadata))
                handle.flush(); os.fsync(handle.fileno())
            with zipfile.ZipFile(partial) as archive:
                if archive.testzip() is not None: raise BackupError('ZIP CRC mismatch')
                for entry in entries:
                    h = hashlib.sha256()
                    with archive.open(entry['archive_path']) as src:
                        for block in iter(lambda: src.read(2 ** 20), b''): h.update(block)
                    if h.hexdigest() != entry['sha256']: raise BackupError('ZIP entry SHA mismatch')
            for name, expected in files.items():
                if digest(regular_path(run, name)) != expected: raise BackupError('Artifact changed before commit: ' + name)
            for old in ledgers:
                current, _ = read_json(run, old['path'])
                if any(current.get(label) != values for label, values in old['selected'].items()):
                    raise BackupError('Previously completed ledger steps changed before commit')
            # Atomic exclusive publication; never replace an earlier snapshot.
            os.link(partial, final); partial.unlink()
        except BaseException:
            partial.unlink(missing_ok=True); raise
        return {'status': 'completed' if not skipped else 'completed_with_skips', 'path': str(final),
                'bytes': final.stat().st_size, 'sha256': digest(final), 'files': len(entries),
                'completed_steps': steps, 'skipped': skipped, 'training_started': False}


def self_test():
    with tempfile.TemporaryDirectory(prefix='completed-backup-fixture-') as tmp:
        base = Path(tmp); run = base / 'run'; out = base / 'backups'
        assert snapshot(run, out)['status'] == 'noop' and not out.exists()
        folder = run / 'fold_00'; (folder / 'seed_42').mkdir(parents=True)
        artifact = folder / 'seed_42/cnn.pt'; artifact.write_bytes(b'completed tensor fixture')
        partial = folder / 'seed_42/unrecorded_in_progress.pt'; partial.write_bytes(b'not complete')
        ledger = {'task_inner_seed_42': {'seed_42/cnn.pt': digest(artifact)}}
        ledger_file = folder / 'artifact_ledger.json'; ledger_file.write_bytes(encode(ledger))
        (run / 'execution_identity.json').write_bytes(encode({'fixture': True}))
        result = snapshot(run, out); assert result['status'] == 'completed'
        restore = base / 'restore'; restore.mkdir()
        with zipfile.ZipFile(result['path']) as z:
            m = json.loads(z.read(MANIFEST)); assert partial.relative_to(run).as_posix() not in z.namelist()
            for entry in m['files']:
                name = entry['relative_path']; relative_name(name)
                target = restore / name; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(entry['archive_path']))
                assert digest(target) == entry['sha256']
        assert (restore / artifact.relative_to(run)).read_bytes() == artifact.read_bytes()
        artifact.write_bytes(b'changed after ledger commit')
        changed = snapshot(run, out); assert changed['status'] == 'completed_with_skips'
        with zipfile.ZipFile(changed['path']) as z: assert artifact.relative_to(run).as_posix() not in z.namelist()
        assert changed['skipped']
        for invalid in ['../outside.pt', '/absolute.pt', 'seed_42/../../outside.pt', 'pretrained/vendor.pt', 'Test.csv']:
            try: relative_name(invalid)
            except BackupError: pass
            else: raise AssertionError('Unsafe path accepted')
        link = folder / 'seed_42/link.pt'; link.symlink_to(artifact)
        ledger_file.write_bytes(encode({'bad_link': {'seed_42/link.pt': digest(artifact)}}))
        try: snapshot(run, out)
        except BackupError: pass
        else: raise AssertionError('Symlink accepted')
    return {'self_test_passed': True, 'checks': ['RUN-absent noop', 'completed ledger roundtrip and entry SHA',
        'unrecorded in-progress file excluded', 'changed-file step rejected', 'path escape/vendor/Test rejected',
        'symlink rejected', 'source RUN not modified by backup'], 'real_training_or_server_access': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    p.add_argument('--backup-dir', type=Path, default=DEFAULT_BACKUPS)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    try:
        result = self_test() if a.self_test else snapshot(a.run_dir, a.backup_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (BackupError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({'status': 'failed', 'error': type(exc).__name__, 'message': str(exc),
                          'source_modified': False, 'training_started': False}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == '__main__': main()
