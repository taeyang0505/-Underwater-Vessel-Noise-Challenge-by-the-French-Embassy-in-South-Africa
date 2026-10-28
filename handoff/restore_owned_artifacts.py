"""Verify and extract a trusted release snapshot to a NEW directory only.

Never points at, overwrites, or resumes the live project. Uses no torch/pickle.
"""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, os, tarfile

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def verify(archive, manifest):
 if digest(archive)!=manifest['sha256'] or archive.stat().st_size!=manifest['bytes']:
  raise ValueError('Archive SHA/size differs; refuse extraction')
 expected={e['path']:e for e in manifest['files']}
 if len(expected)!=len(manifest['files']):raise ValueError('Duplicate ledger entries')
 seen=set()
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar:
   path=PurePosixPath(member.name)
   if path.is_absolute() or '..' in path.parts or '\\' in member.name or not member.isfile():
    raise ValueError('Unsafe archive path/type')
   if member.name in seen:raise ValueError('Duplicate archive member')
   seen.add(member.name)
   if member.name=='OWNED_MODEL_LEDGER.json':
    internal=json.load(tar.extractfile(member))
    if internal['files']!=manifest['files']:raise ValueError('Internal/external ledgers differ')
    continue
   if member.name not in expected:raise ValueError('Unlisted archive member')
   entry=expected[member.name];h=hashlib.sha256();stream=tar.extractfile(member)
   for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
   if member.size!=entry['bytes'] or h.hexdigest()!=entry['sha256']:raise ValueError('Entry checksum differs: '+member.name)
 if seen!=set(expected)|{'OWNED_MODEL_LEDGER.json'}:raise ValueError('Missing members')
 return {'verified_files':len(expected),'archive_sha256':manifest['sha256']}

def main():
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('OWNED_MODEL_RELEASE_MANIFEST.json'))
 p.add_argument('--destination',type=Path,help='New directory only; omit to verify without extracting')
 args=p.parse_args();manifest=json.loads(args.manifest.read_text());receipt=verify(args.archive,manifest)
 if args.destination:
  if args.destination.exists():p.error('Destination exists; refusing to overwrite anything')
  args.destination.mkdir(parents=True,exist_ok=False)
  with tarfile.open(args.archive,'r:gz') as tar:
   for member in tar:
    out=args.destination/member.name;out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as stream,tar.extractfile(member) as src:
     for b in iter(lambda:src.read(1<<20),b''):stream.write(b)
  receipt['extracted_to']=str(args.destination.resolve())
  receipt['training_started']=False
 print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
