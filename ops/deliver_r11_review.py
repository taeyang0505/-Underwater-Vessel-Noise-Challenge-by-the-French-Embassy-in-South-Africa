"""SHA-verified delivery of new R11 review artifacts; preserve user files."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=Path('/Users/hongtaeyang/Desktop/French_Embassy_Underwater_Vessel_Noise')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    manifest=ROOT/'reports/R11_DIAGNOSTICS_MANIFEST.json';d=json.loads(manifest.read_text())
    paths=list(d['files_sha256'])+['reports/R11_DIAGNOSTICS_MANIFEST.json'];copied=[];identical=[]
    for name in paths:
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Escaping relative path')
        src=ROOT/name;dst=DEST/name;expected=d['files_sha256'].get(name,sha(manifest))
        if sha(src)!=expected:raise ValueError('Source changed before delivery:'+name)
        if dst.exists():
            if sha(dst)!=expected:raise FileExistsError('Preserve differing Desktop file:'+name)
            identical.append(name);continue
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        if sha(dst)!=expected:raise ValueError('Destination SHA differs:'+name)
        copied.append(name)
    receipt={'destination':str(DEST),'files_copied':len(copied),'identical_existing':len(identical),
             'all_SHA_verified':True,'differing_existing_overwritten':0,'manifest_SHA256':sha(manifest),
             'bytes':sum((ROOT/name).stat().st_size for name in paths),'files':paths}
    p=ROOT/'reports/R11_DESKTOP_DELIVERY.json';p.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    shutil.copy2(p,DEST/'reports/R11_DESKTOP_DELIVERY.json')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'},ensure_ascii=False))
if __name__=='__main__':main()
