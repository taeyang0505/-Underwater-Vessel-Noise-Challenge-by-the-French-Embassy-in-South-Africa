"""Plan/build a private, portable CV handoff; default action never creates ZIP.

Packaging snapshots the current audited code only when --create is explicitly
requested. --fetch-pretrained uses fixed official revisions and fixed hashes;
it does not resolve a branch/tag, fit a model, or access challenge data.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, urllib.request, zipfile

ROOT=Path(__file__).resolve().parent
CV=Path('data/synthetic/group_cv_nested5_pilot_v1')
DEFAULT_ZIP=Path('delivery/group_cv_gpu_handoff_v1.zip')
DELIVERY_MANIFEST=Path('delivery/group_cv_delivery_manifest.json')
CHECKPOINTS={
    'tabicl':{
        'filename':'tabicl-regressor-v2-20260212.ckpt',
        'revision':'4dcd344ece2c00be9e831fdd35bed57b5ad83e19',
        'sha256':'0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a',
        'bytes':114324594,'licence':'BSD-3-Clause',
        'url':'https://huggingface.co/jingang/TabICL/resolve/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-regressor-v2-20260212.ckpt',
        'provenance':'models/pretrained/tabicl_provenance.json'},
    'kumo_large':{
        'filename':'kumo_large_regressor.pt',
        'revision':'3c3e10bbdb590ace29e7026847f92db3c603096d',
        'sha256':'745e0e5d97e4256d8a24cb61c2eecbd1128da6c83938c196b7589ea6f62e7421',
        'bytes':862909259,'licence':'OpenMDW-1.1',
        'url':'https://huggingface.co/nvidia/Kumo-Tabular/resolve/3c3e10bbdb590ace29e7026847f92db3c603096d/large/regressor.pt',
        'provenance':'models/pretrained/kumo_provenance.json'}
}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def safe_name(name):
    path=Path(name)
    if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe archive path: '+str(name))
    forbidden={'raw','noaa_sanctsound_holdout','sealed_holdout','calibration','__pycache__'}
    if forbidden.intersection(path.parts):raise ValueError('Excluded directory: '+str(name))
    if path.name.lower() in {'test.csv','sample_submission.csv','calibration.npz','domain_holdout.npz','sealed_holdout.npz'}:
        raise ValueError('Excluded payload: '+str(name))
    if path.suffix.lower() in {'.wav','.flac','.mp3','.pt','.ckpt','.zip','.pyc'}:
        raise ValueError('Raw audio, model weights and nested archives are excluded: '+str(name))
    return path.as_posix()

def json_bytes(value):return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode()

def source_provenance():
    # Packaging runs in the source workspace. Only CV-admitted source records
    # are exported, rather than the full adapter containing reserved splits.
    import pandas as pd
    refs=pd.read_parquet(ROOT/CV/'reference_outer_assignment.parquet')
    keys=set(zip(refs.source,refs.clip_id))
    old=pd.read_parquet(ROOT/'data/processed/external_clip_manifest.parquet')
    old=old[[key in keys for key in zip(old.source,old.clip_id)]]
    adapter=pd.read_parquet(ROOT/'external/expansion_research/native_reference_adapter.parquet')
    adapter=adapter[[key in keys for key in zip(adapter.source,adapter.clip_id)]]
    assert set(old.split)<=set(['train','validation'])
    assert set(adapter.split)<=set(['train','validation']) and not adapter.sealed_holdout_locked.any()
    return {'source_pool':'prior train + validation only',
        'challenge_test_included':False,'raw_audio_included':False,'calibration_or_sealed_records_included':False,
        'old_external_cv_records':json.loads(old.to_json(orient='records',date_format='iso')),
        'native_cv_records':json.loads(adapter.to_json(orient='records',date_format='iso')),
        'original_arrays_included':False,
        'rebuild_limitation':'Recipes are retained, but exact source-level generation needs the original SA and external band banks plus pinned excerpts in the source workspace.'}

def inventory():
    cv=json.loads((ROOT/CV/'cv_manifest.json').read_text())
    if cv.get('ready_to_train') is not True or cv.get('n_splits')!=5:
        raise ValueError('Completed frozen five-fold CV package is required')
    if cv.get('challenge_test_accessed') is not False or cv.get('sealed_arrays_accessed') is not False:
        raise ValueError('Unexpected CV source provenance')
    files=set()
    def add(path):
        path=Path(path);name=safe_name(path)
        actual=ROOT/path
        if actual.is_symlink() or not actual.is_file():raise ValueError('Expected regular source file: '+name)
        if not actual.resolve().is_relative_to(ROOT):raise ValueError('Source resolves outside project: '+name)
        files.add(path)
    # Include all Python source modules so runtime code identities, static
    # imports and subprocess modules remain available after relocation.
    for f in (ROOT/'pipeline').glob('*.py'):add(f.relative_to(ROOT))
    for f in (ROOT/'configs').glob('group_cv*.json'):add(f.relative_to(ROOT))
    for f in (ROOT/CV).rglob('*'):
        if f.is_file():add(f.relative_to(ROOT))
    required=[Path('prepare_cv_delivery.py'),Path('run_group_cv_gpu.sh'),Path('requirements_gpu_cv.txt'),
        Path('requirements_tested.json'),Path('delivery/GPU_GROUP_CV_README.txt'),
        Path('reports/group_cv_data_audit.json'),Path('reports/group_cv_data_verification.json'),
        Path('reports/group_cv_runner_verification.json'),Path('reports/training_policy_v2_verification.json'),
        Path('reports/training_cv_package_preflight.json'),Path('reports/contrastive_design.json'),
        Path('reports/training_readiness.json'),Path('reports/training_readiness.html'),
        Path('reports/data_quality_report.html'),Path('reports/gpu_preflight_fixture_verification.json'),
        Path('reports/dataset_review_facts.json'),Path('reports/dataset_document_external_facts.json'),
        Path('reports/dataset_document_verification.json'),Path('reports/dataset_document_data_quality_factual_audit.json'),
        Path('DATASET_CONSTRUCTION_PREPROCESSING_REPORT.md'),
        Path('reports/data_quality_boundary_audit.json'),Path('external_data_declaration_DRAFT.txt'),
        Path('external/vtuad_v1.3.2/dataset_card.md'),Path('external/vtuad_v1.3.2/license_evidence_v131.md'),
        Path('external/hearmyship_v1/datacite.json')]
    for x in required:add(x)
    for pattern in ['*.png','*.svg']:
        for f in (ROOT/'reports/dataset_figures').glob(pattern):add(f.relative_to(ROOT))
    for spec in CHECKPOINTS.values():
        add(spec['provenance']);meta=json.loads((ROOT/spec['provenance']).read_text())
        if any(meta[key]!=spec[key] for key in ['revision','sha256','bytes']):raise ValueError('Pinned provenance differs')
    for deployment in ['ci01_01','fk01_01','gr01_01','hi01_01']:
        add(Path('external/expansion_research')/deployment/f'SanctSound_{deployment.upper()}.json')
    for fold in cv['folds']:
        data=Path(fold['data_dir'])
        if data.is_absolute() or not data.is_relative_to(CV):raise ValueError('Fold must have a portable project-relative path')
        gate=json.loads((ROOT/data/'readiness_manifest.json').read_text())
        if gate.get('ready_to_train') is not True or gate.get('source_hash')!=fold['source_hash']:
            raise ValueError('Fold readiness does not match CV index')
        if gate.get('parent_cv_source_hash')!=cv['source_hash']:raise ValueError('CV source binding differs')
        if hashlib.sha256(json.dumps(gate['file_sha256'],sort_keys=True).encode()).hexdigest()!=gate['data_package_hash']:
            raise ValueError('Fold package identity differs')
        source=Path(gate['source_version_dir'])
        if source.is_absolute() or not source.is_relative_to(data):raise ValueError('Source manifest path is not portable')
        if sha256(ROOT/source/'manifest.json')!=gate['source_version_manifest_sha256']:
            raise ValueError('Source-version manifest changed')
        for name,h in gate['file_sha256'].items():
            if Path(name).name!=name or sha256(ROOT/data/name)!=h:raise ValueError('Fold payload changed: '+name)
            assert data/name in files
        for name,h in gate['generator_code_sha256'].items():
            if sha256(ROOT/name)!=h:raise ValueError('Generator code changed: '+name)
            assert Path(name) in files
    virtual={'delivery/provenance/cv_source_provenance.json':json_bytes(source_provenance())}
    entries=[{'path':p.as_posix(),'bytes':(ROOT/p).stat().st_size,'sha256':sha256(ROOT/p)} for p in sorted(files)]
    entries += [{'path':safe_name(name),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()} for name,blob in virtual.items()]
    manifest={'format':'private_group_cv_handoff_v1','cv_source_hash':cv['source_hash'],
        'cv_manifest_sha256':sha256(ROOT/CV/'cv_manifest.json'),'files':sorted(entries,key=lambda x:x['path']),
        'pretrained_weights_included':False,'pretrained_downloads':CHECKPOINTS,
        'raw_audio_included':False,'challenge_test_included':False,
        'calibration_and_sealed_payloads_included':False,'previous_task_fitted_models_included':False,
        'original_source_arrays_included':False,'training_executed_by_packager':False,
        'gpu_execution_measured':False,'purpose':'Private GPU training handoff; not a public data release'}
    return manifest,virtual

def build(output):
    output=(ROOT/output).resolve() if not output.is_absolute() else output.resolve()
    if output==ROOT/'delivery/gpu_handoff.zip':raise ValueError('The existing V1 archive must not be replaced')
    if output.exists() or output.with_suffix('.zip.partial').exists():raise FileExistsError('Use a new delivery path')
    manifest,virtual=inventory();output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_suffix('.zip.partial')
    try:
        with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for item in manifest['files']:
                name=item['path'];data=virtual[name] if name in virtual else (ROOT/name).read_bytes()
                if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
                    raise ValueError('File changed while snapshotting: '+name)
                info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_DEFLATED
                info.external_attr=(0o100755 if name.endswith('.sh') else 0o100644)<<16
                z.writestr(info,data,compresslevel=4)
            z.writestr(DELIVERY_MANIFEST.as_posix(),json_bytes(manifest),compress_type=zipfile.ZIP_DEFLATED)
        # Verify the completed archive payload before publishing its filename.
        with zipfile.ZipFile(partial) as z:
            if z.testzip() is not None:raise ValueError('ZIP CRC verification failed')
            for item in manifest['files']:
                if hashlib.sha256(z.read(item['path'])).hexdigest()!=item['sha256']:
                    raise ValueError('Archived file SHA-256 mismatch')
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True);raise
    (ROOT/DELIVERY_MANIFEST).write_bytes(json_bytes(manifest))
    receipt={'path':str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
        'bytes':output.stat().st_size,'sha256':sha256(output),'files':len(manifest['files'])}
    output.with_suffix('.zip.sha256.json').write_bytes(json_bytes(receipt));print(json.dumps(receipt,indent=2))

def verify_delivery():
    manifest=json.loads((ROOT/DELIVERY_MANIFEST).read_text())
    for entry in manifest['files']:
        name=safe_name(entry['path']);path=ROOT/name
        if path.is_symlink() or not path.is_file() or sha256(path)!=entry['sha256']:
            raise ValueError('Transferred file mismatch: '+name)
    print(json.dumps({'verified':True,'files':len(manifest['files']),
        'cv_source_hash':manifest['cv_source_hash'],'model_training_executed':False},indent=2))

def fetch_pretrained():
    for name,spec in CHECKPOINTS.items():
        target=ROOT/'models/pretrained'/spec['filename'];target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if target.is_symlink() or target.stat().st_size!=spec['bytes'] or sha256(target)!=spec['sha256']:
                raise ValueError('Existing checkpoint differs; preserve it and use a clean destination: '+str(target))
            print(name,'already verified',spec['sha256'],flush=True);continue
        partial=target.with_name(target.name+'.partial')
        if partial.exists():raise FileExistsError('A previous partial download exists: '+str(partial))
        h=hashlib.sha256();size=0
        try:
            with urllib.request.urlopen(spec['url'],timeout=120) as src,partial.open('xb') as dst:
                while data:=src.read(2**20):dst.write(data);h.update(data);size+=len(data)
            if h.hexdigest()!=spec['sha256'] or size!=spec['bytes']:raise ValueError('Official pinned download checksum/size mismatch: '+name)
            partial.replace(target)
        except BaseException:
            partial.unlink(missing_ok=True);raise
        print(name,'downloaded and verified',size,spec['sha256'],flush=True)

def main(a):
    if a.fetch_pretrained:fetch_pretrained();return
    if a.verify_delivery:verify_delivery();return
    if a.create:build(a.output);return
    manifest,_=inventory()
    print(json.dumps({'action':'plan only; no ZIP created','output':str(a.output),
        'files':len(manifest['files']),'payload_bytes_before_zip':sum(x['bytes'] for x in manifest['files']),
        'cv_source_hash':manifest['cv_source_hash'],'pretrained_weights_excluded_bytes':sum(x['bytes'] for x in CHECKPOINTS.values()),
        'raw_test_calibration_sealed_payloads_included':False},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--create',action='store_true');actions.add_argument('--fetch-pretrained',action='store_true')
    actions.add_argument('--verify-delivery',action='store_true');parser.add_argument('--output',type=Path,default=DEFAULT_ZIP)
    main(parser.parse_args())

