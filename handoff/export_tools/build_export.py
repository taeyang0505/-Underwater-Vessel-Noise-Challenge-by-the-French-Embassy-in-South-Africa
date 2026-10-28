"""Read-only snapshot of a live research directory into a public handoff checkout.

Never traverses data/audio symlinks or rewrites the live project. All exclusions
are inventoried. Owned weight snapshots go to a separate GitHub release archive.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, re, shutil, tarfile

ROOT = Path('/Users/hongtaeyang/Documents/ChatGPT/대회/underwater_vessel_noise.nosync')
DEST = ROOT.parent / 'work/underwater-vessel-noise-handoff'
PACK = ROOT.parent / 'work/underwater-vessel-noise-packaging'
SKIP_PARTS = {'__pycache__', '.git', '.DS_Store', 'server_backups',
              'r14_production_runtime_py', 'pretrained', 'scratch', 'reference_materials'}
TEXT = {'.py', '.md', '.html', '.json', '.csv', '.txt', '.sh', '.toml', '.yaml', '.yml', '.svg'}
MEDIA = {'.png', '.pdf'}
ALLOWED_DIRS = {'pipeline','configs','ops','reports','audit','eda','eda_deep',
                'evidence','figures','submissions'}
SECRET = [
    ('private_key', re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----')),
    ('github_token', re.compile(r'\b(?:github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{30,})\b')),
    ('provider_key', re.compile(r'\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{32,}\b')),
    ('aws_access_key', re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
    ('signed_url', re.compile(r'https?://[^\s\"<>]+[?&](?:X-Amz-Signature|X-Goog-Signature|access_token|token)=[A-Za-z0-9_%.-]{20,}', re.I)),
]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--weights',action='store_true');args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True);PACK.mkdir(parents=True,exist_ok=True)
    copied=[];excluded=[];redacted=[];weights=[]
    for directory, dirs, files in os.walk(ROOT, followlinks=False):
        d=Path(directory); rel_d=d.relative_to(ROOT)
        if rel_d==Path('.'):
            for n in list(dirs):
                if n not in ALLOWED_DIRS:
                    excluded.append({'path':n+'/', 'reason':'raw/cache/vendor/large artifact or non-first-party directory; see local inventory'})
            dirs[:]=[n for n in dirs if n in ALLOWED_DIRS and not (d/n).is_symlink()]
        else:
            for n in list(dirs):
                if n in SKIP_PARTS or (d/n).is_symlink():
                    excluded.append({'path':str((d/n).relative_to(ROOT))+'/', 'reason':'vendor/cache/backup/symlink'})
            dirs[:]=[n for n in dirs if n not in SKIP_PARTS and not (d/n).is_symlink()]
        for name in sorted(files):
            f=d/name;rel=f.relative_to(ROOT);size=f.stat().st_size
            reason=None
            if f.is_symlink(): reason='symlink'
            elif name.startswith('.') or name in {'writer.lock'}:reason='hidden/runtime lock'
            elif f.suffix not in TEXT|MEDIA:reason='raw/synthetic arrays, opaque binary, workspace config, or live log'
            elif f.suffix in TEXT and size>2_000_000:reason='large recipe/prediction ledger; local inventory retained'
            elif f.suffix=='.png' and size>12_000_000:reason='large media'
            elif f.suffix=='.pdf' and size>15_000_000:reason='large media'
            elif name.lower() in {'credentials','secrets.json','.env'}:reason='credential filename'
            if reason:
                excluded.append({'path':str(rel),'bytes':size,'reason':reason});continue
            out=DEST/rel;out.parent.mkdir(parents=True,exist_ok=True)
            raw=f.read_bytes();original_sha=hashlib.sha256(raw).hexdigest()
            if f.suffix in TEXT:
                try:text=raw.decode('utf-8')
                except UnicodeDecodeError:
                    excluded.append({'path':str(rel),'bytes':size,'reason':'non UTF-8 text'});continue
                matches=[label for label,pattern in SECRET if pattern.search(text)]
                if matches:
                    # Never publish a suspected secret just because it appears in
                    # code, a saved response or an old report. Do not print it.
                    excluded.append({'path':str(rel),'bytes':size,'reason':'secret-pattern quarantine','patterns':matches});continue
                # Make documentation links portable without changing code or hashes.
                if f.suffix in {'.md','.html'}:
                    prefix=os.path.relpath(DEST,out.parent).replace(os.sep,'/')
                    edited=text.replace(str(ROOT)+'/',prefix+'/')
                    edited=edited.replace('/Users/hongtaeyang/Desktop/French_Embassy_Underwater_Vessel_Noise/',prefix+'/')
                    if edited!=text:
                        redacted.append({'path':str(rel),'reason':'local documentation links made relative','source_sha256':original_sha})
                    raw=edited.encode()
            out.write_bytes(raw)
            if f.suffix=='.sh':out.chmod(0o755)
            copied.append({'path':str(rel),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'source_sha256':original_sha})
    # Metadata only: do not pretend the public checkout contains the datasets.
    local_inventory=[]
    for base in ('data.nosync','external.nosync','models'):
        for directory, dirs, files in os.walk(ROOT/base,followlinks=False):
            dirs[:]=[n for n in dirs if n not in {'__pycache__'} and not (Path(directory)/n).is_symlink()]
            for name in files:
                f=Path(directory)/name
                if not f.is_symlink():local_inventory.append({'path':str(f.relative_to(ROOT)),'bytes':f.stat().st_size})
    dump(DEST/'handoff/LOCAL_ARTIFACT_INVENTORY.json',local_inventory)
    dump(DEST/'handoff/EXPORT_MANIFEST.json',{'snapshot_kst':datetime.datetime.now().astimezone().isoformat(),
        'source_root':str(ROOT),'copied':copied,'excluded':excluded,'documentation_rewrites':redacted,
        'raw_data_redistributed':False,'pretrained_vendor_weights_redistributed':False})
    # Small exact metadata under original data path, sufficient for inspection,
    # not sufficient to claim a reconstructed training dataset.
    for rel in ['data/synthetic/r16_multisupport_v1/manifest.json']:
        f=ROOT/rel
        if f.exists():
            out=DEST/'handoff/data_manifests'/Path(rel).name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,out)
    states={}
    for rel in ['ops/r16_multisupport_full_r1/state.json','ops/r16_multisupport_fivefold_r1/state.json',
                'models/r16_multisupport_full_r1/progress.json']:
        if (ROOT/rel).exists():states[rel]=json.loads((ROOT/rel).read_text())
    dump(DEST/'handoff/LIVE_STATE_SNAPSHOT.json',{'snapshot_kst':datetime.datetime.now().astimezone().isoformat(),'states':states,'snapshot_is_not_live':True})
    if args.weights:
        paths=[]
        for f in (ROOT/'models').rglob('*'):
            if not f.is_file() or any(n in SKIP_PARTS for n in f.relative_to(ROOT).parts):continue
            # Own trained checkpoints and native LightGBM model text only.
            if f.suffix=='.pt' or (f.suffix=='.txt' and f.stat().st_size>50_000):paths.append(f)
        # Best submitted R9 GPU neural weights were restored into this backup.
        backup=ROOT/'ops/server_backups/r9_gpu_r1/restored'
        if backup.exists():paths += list(backup.rglob('*.pt'))
        archive=PACK/'owned-trained-models-2026-10-28.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
            for f in sorted(set(paths)):
                rel=str(f.relative_to(ROOT))
                # Checkpoints are atomically replaced by training; opening once
                # snapshots one complete old/new inode, never a mixture.
                with f.open('rb') as stream:
                    info=tar.gettarinfo(str(f),arcname=rel)
                    info.size=os.fstat(stream.fileno()).st_size
                    data=stream.read()
                    import io
                    tar.addfile(info,io.BytesIO(data))
                weights.append({'path':rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                                'completion':'running snapshot; not a completed run' if 'r16_multisupport_full_r1/neural' in rel else 'consult corresponding summary/completion receipt'})
            ledger=json.dumps({'snapshot_kst':datetime.datetime.now().astimezone().isoformat(),'files':weights,
                               'trusted_source_only':'PyTorch checkpoint deserialization may execute code; only load trusted artifacts.',
                               'no_dataset_or_vendor_weights':True},ensure_ascii=False,indent=2).encode()
            import io
            info=tarfile.TarInfo('OWNED_MODEL_LEDGER.json');info.size=len(ledger);tar.addfile(info,io.BytesIO(ledger))
        dump(PACK/'owned-trained-models-2026-10-28.json',{'archive':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),'files':weights})
        shutil.copyfile(PACK/'owned-trained-models-2026-10-28.json',DEST/'handoff/OWNED_MODEL_RELEASE_MANIFEST.json')
    print(json.dumps({'files':len(copied),'bytes':sum(x['bytes'] for x in copied),'excluded':len(excluded),'rewritten_docs':len(redacted),
                      'quarantined':[x for x in excluded if x.get('reason')=='secret-pattern quarantine'],
                      'owned_model_count':len(weights)},ensure_ascii=False))

if __name__=='__main__':main()
