"""SHA-checked mirror of R14 outputs to the user-requested Desktop project."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
DEST=Path('/Users/hongtaeyang/Desktop/French_Embassy_Underwater_Vessel_Noise')

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    files=set()
    for group in ('reports','ops','pipeline','configs','models'):
        for entry in (ROOT/group).iterdir():
            if 'r14' not in entry.name.lower() or entry.name=='R14_DESKTOP_DELIVERY.json':continue
            for p in ([entry] if entry.is_file() else entry.rglob('*')):
                if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
                if not p.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Foreign output')
                files.add(p.relative_to(ROOT))
    rows=[]
    for rel in sorted(files):
        p=ROOT/rel;d=DEST/rel;h=digest(p);d.parent.mkdir(parents=True,exist_ok=True)
        if not d.is_file() or digest(d)!=h:shutil.copy2(p,d)
        if digest(d)!=h:raise ValueError('Desktop copy SHA mismatch:'+str(rel))
        rows.append({'path':str(rel),'bytes':p.stat().st_size,'sha256':h})
    manifest={'status':'completed','source_root':str(ROOT),'destination_root':str(DEST),'files':len(rows),
        'bytes':sum(r['bytes'] for r in rows),'entries':rows,'unrelated_existing_files_removed':False}
    target=ROOT/'reports/R14_DESKTOP_DELIVERY.json'
    target.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    shutil.copy2(target,DEST/'reports/R14_DESKTOP_DELIVERY.json')
    print(json.dumps({'status':'completed','files':manifest['files'],'bytes':manifest['bytes'],'manifest_sha256':digest(target)},ensure_ascii=False))

if __name__=='__main__':main()
