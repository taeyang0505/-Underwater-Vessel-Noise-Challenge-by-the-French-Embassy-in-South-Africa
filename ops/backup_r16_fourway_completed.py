"""Archive only a completed V1 four-way comparison; verify every ledger entry."""
from pathlib import Path
import argparse, datetime, hashlib, json, sys, zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.supervise_r16_pretrained_cuda_outputs import sha,verify_zip,write,META

def backup(destination):
    run=ROOT/'models/r16_v1_fourway_r1'
    done=run/'COMPLETED.json'
    if not done.is_file() or (run/'FAILED.json').exists():
        raise RuntimeError('Only completed, unfailed comparisons can be archived')
    complete=json.loads(done.read_text())
    if complete.get('status')!='completed':raise ValueError('Completion status differs')
    sources={}
    def add(relative,expected=None):
        path=ROOT/relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink() or not path.is_file():
            raise ValueError('Unsafe or missing artifact: '+relative)
        if not path.resolve().is_relative_to(ROOT):raise ValueError('Artifact escaped project')
        actual=sha(path)
        if expected and actual!=expected:raise ValueError('Ledger mismatch: '+relative)
        sources[relative]={'sha256':actual,'bytes':path.stat().st_size}
    for name,digest in complete['artifact_sha256'].items():add(name,digest)
    add(str(done.relative_to(ROOT)))
    protocol=run/'PROTOCOL.json'
    for name,digest in json.loads(protocol.read_text())['code_sha256'].items():add(name,digest)
    add('ops/r16_fourway_preregistration_r1.json')
    for path in sorted((ROOT/'reports/r16_v1_fourway_r1').glob('*')):
        if path.is_file():add(str(path.relative_to(ROOT)))
    manifest={'format':'r16_fourway_complete_backup_v1',
      'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'completion_sha256':sha(done),'files':sources,'official_upload':False}
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    archive=destination/('r16_v1_fourway_completed_'+sha(done)[:12]+'.zip')
    if not archive.exists():
        temporary=archive.with_suffix('.zip.tmp')
        if temporary.exists():raise FileExistsError('Partial archive requires explicit review')
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            z.writestr(META,json.dumps(manifest,sort_keys=True,ensure_ascii=False))
            for name in sources:z.write(ROOT/name,arcname=name)
        verify_zip(temporary,manifest);temporary.replace(archive)
    else:
        # Compare the immutable closure; creation timestamp may differ.
        with zipfile.ZipFile(archive) as z:existing=json.loads(z.read(META))
        if existing['files']!=sources or existing['completion_sha256']!=sha(done):
            raise ValueError('Existing archive has another closure')
    receipt={**verify_zip(archive),'path':str(archive),'completion_sha256':sha(done)}
    write(destination/'LATEST.json',receipt)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path)
    p.add_argument('--verify',type=Path)
    a=p.parse_args()
    if a.verify:result=verify_zip(a.verify)
    elif a.destination:result=backup(a.destination)
    else:p.error('--destination or --verify required')
    print(json.dumps(result,ensure_ascii=False))
