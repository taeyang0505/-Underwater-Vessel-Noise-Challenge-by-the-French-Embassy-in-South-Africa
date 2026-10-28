"""Copy only new R10 artifacts to the user-authorized Desktop project."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=Path('/Users/hongtaeyang/Desktop/French_Embassy_Underwater_Vessel_Noise')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    manifest=ROOT/'reports/R10_DIAGNOSTICS_MANIFEST.json';d=json.loads(manifest.read_text())
    paths=list(d['files_sha256'])+['reports/R10_DIAGNOSTICS_MANIFEST.json']
    copied=[];same=[]
    for name in paths:
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe relative file')
        src=ROOT/name;dst=DEST/name
        expected=d['files_sha256'].get(name,sha(manifest))
        if sha(src)!=expected:raise ValueError('Source changed before delivery:'+name)
        if dst.exists():
            if sha(dst)!=expected:raise FileExistsError('Preserve differing existing Desktop file:'+name)
            same.append(name);continue
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        if sha(dst)!=expected:raise ValueError('Desktop copy SHA mismatch:'+name)
        copied.append(name)
    receipt={'authorized_destination':str(DEST),'files_copied':len(copied),'identical_existing':len(same),
             'all_SHA_verified':True,'differing_existing_overwritten':0,
             'manifest_sha256':sha(manifest),'files':paths,'bytes':sum((ROOT/p).stat().st_size for p in paths)}
    p=ROOT/'reports/R10_DESKTOP_DELIVERY.json';p.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    shutil.copy2(p,DEST/'reports/R10_DESKTOP_DELIVERY.json')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'},ensure_ascii=False))
if __name__=='__main__':main()
