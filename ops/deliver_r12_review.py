"""Deliver completed R12 evidence to the user-authorized Desktop project."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
DEST=Path('/Users/hongtaeyang/Desktop/French_Embassy_Underwater_Vessel_Noise')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    required=ROOT/'reports/r12_cpu_ablation_r1/RESULTS.json'
    if not required.is_file():raise ValueError('No completed evaluation; refuse completed delivery')
    files=set()
    for relative in ['data/synthetic/r12_cpu_ablation_v3','data/synthetic/r12_mask_matched_control_v1','data/synthetic/r12_mask_matched_control_v2','models/r12_shared_basis_v1','models/r12_cpu_cnn_r1','models/r12_compact_trees_v1','models/r12_mask_matched_control_r1','models/r12_mask_matched_control_r2','reports/r12_cpu_ablation_r1','reports/r12_mask_matched_evaluation_r2','reports/r12_spectral_probe','reports/r12_figures']:
        files.update(p for p in (ROOT/relative).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for folder,pattern in [('ops','*r12*.py'),('pipeline','*r12*.py'),('configs','r12*.json'),('reports','R12*'),('ops','r12*log')]:
        files.update(p for p in (ROOT/folder).glob(pattern) if p.is_file())
    files={p for p in files if p.name not in ['R12_DELIVERY_MANIFEST.json','R12_DESKTOP_DELIVERY.json']}
    entries={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    manifest={'scope':'completed R12 CPU ablation and diagnostics only; no new GPU production run','files':len(entries),'bytes':sum(p.stat().st_size for p in files),'file_sha256':entries}
    path=ROOT/'reports/R12_DELIVERY_MANIFEST.json';path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    entries[str(path.relative_to(ROOT))]=sha(path);copied=identical=0
    for name,h in entries.items():
        src=ROOT/name;dst=DEST/name
        if sha(src)!=h:raise ValueError('Source changed before delivery:'+name)
        if dst.exists():
            if sha(dst)!=h:raise FileExistsError('Preserve differing existing Desktop file:'+name)
            identical+=1;continue
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        if sha(dst)!=h:raise ValueError('Desktop SHA differs:'+name)
        copied+=1
    receipt={'destination':str(DEST),'files_copied':copied,'identical_existing':identical,'all_SHA_verified':True,'overwrite_existing_differences':0,'manifest_sha256':sha(path),'bytes':manifest['bytes']}
    p=ROOT/'reports/R12_DESKTOP_DELIVERY.json';p.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n');shutil.copy2(p,DEST/'reports'/p.name)
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
