"""Private r1 prebuilt-data handoff. Default is a read-only inventory plan.

Run --create only after the final dataset, code and documents are frozen.
The original packager/archives are never replaced. No command trains a model.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tempfile, zipfile
from urllib.parse import unquote
from prepare_cv_delivery import CHECKPOINTS, sha256, json_bytes, fetch_pretrained
from prepare_cv_delivery import safe_name as legacy_safe_name

ROOT = Path(__file__).resolve().parent
CV = Path('data/synthetic/group_cv_physics_v2_r1')
PARENT = Path('data/synthetic/group_cv_nested5_pilot_v1/cv_manifest.json')
DEFAULT_ZIP = Path('delivery/group_cv_physics_v2_r1_gpu.zip')
DELIVERY_MANIFEST = Path('delivery/group_cv_physics_v2_r1_delivery_manifest.json')
DOCUMENT = Path('DATASET_CONSTRUCTION_PREPROCESSING_REPORT.md')
PLOT_NAMES = ['01_dataset_scale', '02_nested_group_cv', '03_training_distributions', '04_known_component_example']
REQUIRED_QA = ['reports/physics_sampler_v3_fixture.json', 'reports/physics_model_v3_verification.json',
               'reports/physics_cv_runner_verification.json', 'reports/group_cv_physics_data_audit.json',
               'reports/group_cv_physics_data_verification.json']


def safe_name(name):
    result = legacy_safe_name(name)
    p = Path(result)
    if any(part.lower() in {'raw', 'calibration', 'sealed_holdout', 'noaa_sanctsound_holdout'} for part in p.parts):
        raise ValueError('Reserved payload path: ' + result)
    if p.suffix.lower() in {'.pkl', '.pickle', '.joblib', '.pth', '.bin', '.safetensors'}:
        raise ValueError('Fitted model/cache binary is excluded: ' + result)
    if p.suffix.lower() in {'.npy', '.npz'} and not p.is_relative_to(CV):
        raise ValueError('Only r1 prebuilt arrays/order files are allowed: ' + result)
    return result


def assert_baseline(config):
    expected = {'epochs': 60, 'patience': 7, 'optimizer': 'adam', 'feature_version': 3}
    if config.get('cv_dir') != CV.as_posix() or any(config.get('training', {}).get(k) != v for k, v in expected.items()):
        raise ValueError('r1 baseline must use the frozen physics dataset, Adam/60 epochs/patience7/feature3')


def local_links(text):
    return re.findall(r'!?\[[^\]]*\]\(([^)\n]+)\)', text)


def portable_document(data):
    text = data.decode('utf-8')
    # Normalize workspace-absolute links only; retain the referenced file's
    # project-relative location rather than silently substituting older plots.
    for prefix in {str(ROOT) + '/', str(ROOT).replace('.nosync', '') + '/'}:
        text = text.replace(prefix, '')
    if CV.name not in text:
        raise ValueError('Dataset document has not been updated to r1')
    return text.encode('utf-8')


def check_document_links(text, included):
    for target in local_links(text):
        target = target.strip('<>')
        if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', target) or target.startswith('#'):
            continue
        name = unquote(target.split('#', 1)[0])
        if name and (Path(name).is_absolute() or '..' in Path(name).parts or name not in included):
            raise ValueError('Dataset document has a missing/nonportable local link: ' + target)


def source_provenance(dependencies):
    import pandas as pd
    paths = [CV / 'reference_outer_assignment.parquet', Path('data/processed/external_clip_manifest.parquet'),
             Path('external/expansion_research/native_reference_adapter.parquet')]
    for p in paths: dependencies[p.as_posix()] = sha256(ROOT / p)
    refs, old, adapter = [pd.read_parquet(ROOT / p) for p in paths]
    keys = set(zip(refs.source, refs.clip_id))
    old = old[[key in keys for key in zip(old.source, old.clip_id)]]
    adapter = adapter[[key in keys for key in zip(adapter.source, adapter.clip_id)]]
    if not set(old.split) <= {'train', 'validation'} or not set(adapter.split) <= {'train', 'validation'} or adapter.sealed_holdout_locked.any():
        raise ValueError('Reserved source records cannot be exported')
    if len(old) + len(adapter) != len(refs): raise ValueError('Missing CV-admitted source provenance')
    return {'source_pool': 'prior train+validation only, regrouped by connected sources',
        'old_external_cv_records': json.loads(old.to_json(orient='records', date_format='iso')),
        'native_cv_records': json.loads(adapter.to_json(orient='records', date_format='iso')),
        'raw_audio_included': False, 'original_source_arrays_included': False,
        'calibration_or_sealed_records_included': False,
        'rebuild_limitation': 'Exact regeneration needs the original source workspace; this handoff supports prebuilt training only.'}


def inventory():
    cv = json.loads((ROOT / CV / 'cv_manifest.json').read_text())
    if cv.get('ready_to_train') is not True or cv.get('n_splits') != 5 or len(cv.get('folds', [])) != 5:
        raise ValueError('The complete independently verified r1 five-fold package is not ready')
    for name in ['challenge_test_accessed', 'sealed_arrays_accessed']:
        if cv.get(name) is not False: raise ValueError('Unsafe or unspecified CV provenance: ' + name)
    config = json.loads((ROOT / 'configs/group_cv_physics_training.json').read_text()); assert_baseline(config)
    required = [Path('prepare_physics_cv_delivery.py'), Path('prepare_cv_delivery.py'), Path('run_group_cv_gpu.sh'),
        Path('requirements_gpu_cv.txt'), Path('requirements_tested.json'), Path('delivery/GPU_PHYSICS_R1_README.txt'),
        Path('external_data_declaration_DRAFT.txt'), Path('external/vtuad_v1.3.2/dataset_card.md'),
        Path('external/vtuad_v1.3.2/license_evidence_v131.md'), Path('external/hearmyship_v1/datacite.json'), PARENT,
        Path('reports/dataset_document_external_facts.json'), Path('reports/data_quality_boundary_audit.json'),
        Path('reports/group_cv_physics_data_facts.json')]
    files, dependencies = set(), {}
    def add(path):
        path = Path(path); name = safe_name(path); actual = ROOT / path
        if actual.is_symlink() or not actual.is_file() or not actual.resolve().is_relative_to(ROOT):
            raise ValueError('Expected a regular project file: ' + name)
        files.add(path)
    # All root policy/verification scripts and pipeline modules preserve the
    # exact import/subprocess/code-identity contracts after relocation.
    for folder, pattern in [('pipeline', '*.py'), ('configs', 'group_cv*.json')]:
        for p in (ROOT / folder).glob(pattern): add(p.relative_to(ROOT))
    for pattern in ['*.py', '*.sh']:
        for p in ROOT.glob(pattern): add(p.relative_to(ROOT))
    for p in (ROOT / CV).rglob('*'):
        if p.is_file(): add(p.relative_to(ROOT))
    for p in required: add(p)
    for name in REQUIRED_QA:
        add(name); report = json.loads((ROOT / name).read_text())
        passed = report.get('passed') is True or report.get('status') == 'passed' or report.get('ready_to_train') is True
        if not passed: raise ValueError('Required QA report has not passed: ' + name)
        if 'data_audit' in name or 'data_verification' in name:
            if report.get('source_hash') != cv['source_hash']: raise ValueError('QA belongs to a different data version: ' + name)
    for name in PLOT_NAMES:
        for ext in ['png', 'svg']: add(Path('figures') / (name + '.' + ext))
    evidence = ROOT / 'evidence'
    if not evidence.is_dir() or not any(evidence.iterdir()): raise ValueError('Portable document evidence has not been generated')
    for p in evidence.rglob('*'):
        if p.is_file(): add(p.relative_to(ROOT))
    if (evidence / 'cv_manifest.json').exists():
        if json.loads((evidence / 'cv_manifest.json').read_text()).get('source_hash') != cv['source_hash']:
            raise ValueError('Document evidence still refers to another CV source version')
    for spec in CHECKPOINTS.values():
        add(spec['provenance']); meta = json.loads((ROOT / spec['provenance']).read_text())
        if any(meta[k] != spec[k] for k in ['revision', 'sha256', 'bytes']): raise ValueError('Pinned vendor provenance differs')
    for dep in ['ci01_01', 'fk01_01', 'gr01_01', 'hi01_01']:
        add(Path('external/expansion_research') / dep / ('SanctSound_' + dep.upper() + '.json'))
    if sha256(ROOT / PARENT) != cv.get('parent_cv_manifest_sha256'):
        raise ValueError('Parent CV metadata fingerprint differs')
    for fold in cv['folds']:
        folder = Path(fold['data_dir'])
        if folder.is_absolute() or not folder.is_relative_to(CV): raise ValueError('Nonportable fold path')
        gate = json.loads((ROOT / folder / 'readiness_manifest.json').read_text())
        if gate.get('ready_to_train') is not True or gate.get('source_hash') != fold['source_hash'] or gate.get('parent_cv_source_hash') != cv['source_hash']:
            raise ValueError('Fold readiness/source binding differs')
        if gate.get('version') != 'quality_v2_physics_group_cv_prebuilt': raise ValueError('Expected physics r1 fold package')
        if gate.get('physics_config', {}).get('max_minute_snr_db') != 20.: raise ValueError('The r1 minute20dB policy is missing')
        if hashlib.sha256(json.dumps(gate['file_sha256'], sort_keys=True).encode()).hexdigest() != gate['data_package_hash']:
            raise ValueError('Fold package identity differs')
        source = Path(gate['source_version_dir'])
        if source.is_absolute() or not source.is_relative_to(folder): raise ValueError('Nonportable source-version path')
        if sha256(ROOT / source / 'manifest.json') != gate['source_version_manifest_sha256']: raise ValueError('Source manifest changed')
        for name, expected in gate['file_sha256'].items():
            if Path(name).name != name or sha256(ROOT / folder / name) != expected: raise ValueError('Changed fold file: ' + name)
            if folder / name not in files: raise ValueError('Missing fold payload')
        for name, expected in gate['generator_code_sha256'].items():
            if Path(name) not in files or sha256(ROOT / name) != expected: raise ValueError('Frozen generator code changed: ' + name)
    dependencies[DOCUMENT.as_posix()] = sha256(ROOT / DOCUMENT)
    virtual = {DOCUMENT.as_posix(): portable_document((ROOT / DOCUMENT).read_bytes()),
               'delivery/provenance/physics_cv_r1_sources.json': json_bytes(source_provenance(dependencies))}
    entries = [{'path': p.as_posix(), 'bytes': (ROOT / p).stat().st_size, 'sha256': sha256(ROOT / p)} for p in sorted(files)]
    entries += [{'path': safe_name(name), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for name, data in virtual.items()]
    included = {e['path'] for e in entries}
    if len(included) != len(entries): raise ValueError('Duplicate delivery entry')
    check_document_links(virtual[DOCUMENT.as_posix()].decode(), included)
    manifest = {'format': 'private_physics_cv_r1_handoff_v1', 'cv_directory': CV.as_posix(),
        'cv_source_hash': cv['source_hash'], 'cv_manifest_sha256': sha256(ROOT / CV / 'cv_manifest.json'),
        'files': sorted(entries, key=lambda e: e['path']), 'source_snapshot_dependencies': dependencies,
        'pretrained_downloads': CHECKPOINTS, 'pretrained_weights_included': False,
        'raw_audio_included': False, 'challenge_test_included': False, 'calibration_and_sealed_payloads_included': False,
        'previous_task_fitted_models_included': False, 'original_source_arrays_included': False,
        'training_executed_by_packager': False, 'gpu_execution_measured': False,
        'purpose': 'Private prebuilt-data GPU handoff; not a public data release'}
    return manifest, virtual


def build(output, manifest_path):
    output = (ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    local_manifest = ROOT / manifest_path
    safe_name(manifest_path)
    if output.suffix != '.zip' or output.exists() or output.with_suffix('.zip.partial').exists() or local_manifest.exists():
        raise FileExistsError('Choose fresh ZIP and manifest paths; existing deliveries are immutable')
    manifest, virtual = inventory(); output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix('.zip.partial')
    try:
        with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
            for entry in manifest['files']:
                name = entry['path']; data = virtual[name] if name in virtual else (ROOT / name).read_bytes()
                if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                    raise ValueError('Source changed while snapshotting: ' + name)
                info = zipfile.ZipInfo(name); info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100755 if name.endswith('.sh') else 0o100644) << 16
                z.writestr(info, data, compresslevel=4)
            z.writestr(manifest_path.as_posix(), json_bytes(manifest), compress_type=zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(partial) as z:
            if z.testzip() is not None: raise ValueError('ZIP CRC verification failed')
            for entry in manifest['files']:
                if hashlib.sha256(z.read(entry['path'])).hexdigest() != entry['sha256']: raise ValueError('Archived SHA mismatch')
        for entry in manifest['files']:
            if entry['path'] not in virtual and sha256(ROOT / entry['path']) != entry['sha256']:
                raise ValueError('Source changed before commit; retry after freeze: ' + entry['path'])
        for name, expected in manifest['source_snapshot_dependencies'].items():
            if sha256(ROOT / name) != expected: raise ValueError('Snapshot dependency changed: ' + name)
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True); raise
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    with local_manifest.open('xb') as f: f.write(json_bytes(manifest))
    receipt = {'path': str(output), 'bytes': output.stat().st_size, 'sha256': sha256(output),
               'files': len(manifest['files']), 'cv_source_hash': manifest['cv_source_hash']}
    with output.with_suffix('.zip.sha256.json').open('xb') as f: f.write(json_bytes(receipt))
    print(json.dumps(receipt, indent=2))


def verify_delivery(manifest_path):
    manifest = json.loads((ROOT / manifest_path).read_text())
    if manifest.get('format') != 'private_physics_cv_r1_handoff_v1': raise ValueError('Wrong delivery format')
    included = set()
    for entry in manifest['files']:
        name = safe_name(entry['path']); path = ROOT / name
        if name in included or path.is_symlink() or not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError('Transferred file missing/changed/duplicated: ' + name)
        included.add(name)
    check_document_links((ROOT / DOCUMENT).read_text(), included)
    cv = json.loads((ROOT / CV / 'cv_manifest.json').read_text())
    if sha256(ROOT / CV / 'cv_manifest.json') != manifest['cv_manifest_sha256'] or cv['source_hash'] != manifest['cv_source_hash']:
        raise ValueError('Transferred CV identity differs')
    assert_baseline(json.loads((ROOT / 'configs/group_cv_physics_training.json').read_text()))
    print(json.dumps({'verified': True, 'files': len(included), 'cv_source_hash': manifest['cv_source_hash'], 'model_training_executed': False}, indent=2))


def portable_check(archive, manifest_path):
    archive = archive.resolve(); result = {'archive_sha256': sha256(archive), 'training_executed': False}
    before = {k: sha256(ROOT / 'models/pretrained' / v['filename']) for k, v in CHECKPOINTS.items()}
    if any(before[k] != spec['sha256'] for k, spec in CHECKPOINTS.items()): raise ValueError('Original pinned checkpoint differs')
    with tempfile.TemporaryDirectory(prefix='physics-r1-portable-') as tmp:
        destination = Path(tmp)
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                name = safe_name(entry.filename)
                if (entry.external_attr >> 16) & 0o170000 == 0o120000: raise ValueError('Archive symlinks are excluded')
                target = destination / name; target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(entry) as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
        for spec in CHECKPOINTS.values():
            # Borrow original inodes for READ-ONLY operations; never chmod or
            # overwrite them, since inode permissions are shared by hardlinks.
            target = destination / 'models/pretrained' / spec['filename']; target.parent.mkdir(parents=True, exist_ok=True)
            os.link(ROOT / 'models/pretrained' / spec['filename'], target)
        commands = [[sys.executable, 'prepare_physics_cv_delivery.py', '--verify-delivery', '--manifest', manifest_path.as_posix()],
                    [sys.executable, '-m', 'pipeline.train_group_cv']]
        for label, command in zip(['delivery_verification', 'default_preflight'], commands):
            done = subprocess.run(command, cwd=destination, text=True, capture_output=True)
            if done.returncode: raise RuntimeError(label + ' failed: ' + (done.stdout + done.stderr)[-8000:])
            parsed = json.loads(done.stdout)
            result[label] = {k: parsed[k] for k in ['verified', 'ready', 'files', 'n_splits', 'oof_examples', 'full_training_executed'] if k in parsed}
        result['source_arrays_absent'] = not (destination / 'data/processed').exists()
    result['original_weights_unchanged'] = all(sha256(ROOT / 'models/pretrained' / spec['filename']) == before[k] for k, spec in CHECKPOINTS.items())
    if not result['original_weights_unchanged']: raise ValueError('Original vendor weight changed')
    report = archive.with_suffix('.portable_verification.json')
    with report.open('xb') as f: f.write(json_bytes(result))
    print(json.dumps(result, indent=2))


def main(a):
    if a.fetch_pretrained: fetch_pretrained(); return
    if a.verify_delivery: verify_delivery(a.manifest); return
    if a.portable_check: portable_check(a.portable_check, a.manifest); return
    if a.create: build(a.output, a.manifest); return
    manifest, _ = inventory()
    print(json.dumps({'action': 'plan only; no ZIP created', 'output': str(a.output),
        'files': len(manifest['files']), 'payload_bytes_before_zip': sum(e['bytes'] for e in manifest['files']),
        'cv_source_hash': manifest['cv_source_hash'], 'pretrained_weights_excluded_bytes': sum(v['bytes'] for v in CHECKPOINTS.values())}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--create', action='store_true'); actions.add_argument('--fetch-pretrained', action='store_true')
    actions.add_argument('--verify-delivery', action='store_true'); actions.add_argument('--portable-check', type=Path)
    parser.add_argument('--output', type=Path, default=DEFAULT_ZIP)
    parser.add_argument('--manifest', type=Path, default=DELIVERY_MANIFEST)
    main(parser.parse_args())
