"""SHA-verified delivery to the user-authorized Desktop competition folder."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1]
DEST=Path('/Users/hongtaeyang/Desktop/French_Embassy_Underwater_Vessel_Noise')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    result=json.loads((ROOT/'reports/r13_cpu_correction_r1/RESULTS.json').read_text())
    if result['status']!='completed' or not (ROOT/'reports/R13_REVIEW_START.html').exists():raise ValueError('Actual completed CPU report required')
    files=set()
    roots=['data/synthetic/r13_cpu_correction_v1','data/synthetic/group_cv_physics_r13_v1','data/synthetic/group_cv_physics_r13_variants_v2',
        'models/r13_cpu_cnn_r1','models/r13_compact_trees_v1','reports/r13_cpu_correction_r1','reports/r13_score_snapshot_r1',
        'reports/r13_safety_feasibility_r1','reports/r13_figures','models/r13_dynamic_mix_r1','models/r13_dynamic_mix_r2','reports/r13_dynamic_mix_r1']
    for root in roots:files.update(p for p in (ROOT/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for folder,pattern in [('ops','*r13*.py'),('pipeline','*r13*.py'),('configs','r13*.json'),('reports','R13*'),('ops','r13*log')]:
        for path in (ROOT/folder).glob(pattern):
            if path.is_file():files.add(path)
            elif path.is_dir():files.update(p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files={p for p in files if p.name not in ['R13_DELIVERY_MANIFEST.json','R13_DESKTOP_DELIVERY.json']}
    entries={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    manifest={'scope':'R13 completed CPU correction and separately labelled fullscale preparation; no CUDA production completion claimed',
        'files':len(entries),'bytes':sum(p.stat().st_size for p in files),'file_sha256':entries}
    m=ROOT/'reports/R13_DELIVERY_MANIFEST.json';m.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    entries[str(m.relative_to(ROOT))]=sha(m);copied=identical=0
    for name,h in entries.items():
        src=ROOT/name;dst=DEST/name
        if sha(src)!=h:raise ValueError('Source changed:'+name)
        if dst.exists():
            if sha(dst)!=h:raise FileExistsError('Preserve differing Desktop file:'+name)
            identical+=1;continue
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        if sha(dst)!=h:raise ValueError('Desktop copy differs:'+name)
        copied+=1
    receipt={'destination':str(DEST),'copied_files':copied,'identical_existing':identical,'all_SHA_verified':True,
        'bytes':manifest['bytes'],'manifest_sha256':sha(m),'overwrites':0}
    p=ROOT/'reports/R13_DESKTOP_DELIVERY.json';p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');shutil.copy2(p,DEST/'reports'/p.name)
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
