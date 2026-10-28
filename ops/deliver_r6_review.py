"""Targeted R6 review handoff; preserve desktop user edits via prior manifest."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,subprocess,re
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);a=p.parse_args();dst=a.destination.resolve()
 if not dst.is_dir() or dst==ROOT:raise ValueError('Existing independent desktop folder required')
 result=json.loads((ROOT/'reports/R6V2_CPU_ABLATION_RESULTS.json').read_text())
 if result['status']!='completed':raise ValueError('Completed comparison required')
 baseline=json.loads((dst/'DESKTOP_PACKAGE_MANIFEST.json').read_text());old={v['path']:v['sha256'] for v in baseline['files']}
 # Add-only R6 deliverables plus the exact dependency closure bound by the
 # final manifest. Do not copy changing remote state or secrets.
 data=json.loads((ROOT/'reports/R6V2_DATASET_AUDIT.json').read_text())
 paths=set(data['freeze']['code_sha256'])|{'configs/r6_cpu_guardrails_v2.json','ops/deliver_r6_review.py','requirements-r6-cpu.txt','ops/local_alignment_audit.py'}
 for pattern in ['configs/r6*.json','pipeline/*r6*.py','ops/*r6*.py','reports/R6*.md','reports/R6*.json','reports/r6*.png','reports/r6*.svg','reports/r6*.json','ops/r6*.log']:
  paths.update(str(v.relative_to(ROOT)) for v in ROOT.glob(pattern) if v.is_file())
 for name in ['data/versions/source_split_r6_guardrails_v2','data/versions/hard_negatives_r6_v2',
              'data/versions/source_split_r6_cpu_v1','models/r6_cpu_guardrails_v2','models/r6_cpu_guardrails_v2_FAILED_before_first_fit_schema',
              'data/synthetic/r6_cpu_guardrails_v3_schemafix','data/synthetic/r6_cpu_guardrails_v2',
              'data/synthetic/r6_cpu_ablation_v1_FAILED_pre_fit_support_v1','reports/r6_guardrails_v2_figures']:
  paths.update(str(v.relative_to(ROOT)) for v in (ROOT/name).rglob('*') if v.is_file() and '__pycache__' not in v.parts)
 paths.update(str(v.relative_to(ROOT)) for v in (ROOT/'reports').glob('R6_REVIEW*.html'))
 for rel in sorted(paths):
  source,target=ROOT/rel,dst/rel
  if target.is_symlink():raise ValueError('Refuse to replace desktop symlink: '+rel)
  if target.exists() and sha(target)!=sha(source) and old.get(rel)!=sha(target):raise ValueError('Preserved user change; conflict needs review: '+rel)
 copied=0;size=0;receipt={}
 for rel in sorted(paths):
  source,target=ROOT/rel,dst/rel;h=sha(source);n=source.stat().st_size
  if not target.exists() or sha(target)!=h:
   target.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['/bin/cp','-cp',str(source),str(target)],check=True);copied+=1
  if sha(target)!=h:raise RuntimeError('Copy hash mismatch: '+rel)
  receipt[rel]={'sha256':h,'bytes':n};size+=n
 report=dst/'reports/R6_GUARDRAILS_AND_CPU_ABLATION_REPORT.md'
 links=[]
 for href in re.findall(r'\]\(([^)]+)\)',report.read_text()):
  if '://' in href or href.startswith('#'):continue
  path=(report.parent/href).resolve()
  if not path.exists():raise ValueError('Broken local report link: '+href)
  links.append(href)
 record={'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),'destination':str(dst),'verified_files':len(paths),
  'copied_or_updated_files':copied,'verified_bytes':size,'local_report_links_verified':len(links),'user_edits_preserved':True,
  'report':str(report),'report_sha256':sha(report),'file_hashes':receipt,'GPU_SSH_Test_submission':False}
 out=ROOT/'reports/R6_GUARDRAILS_DESKTOP_DELIVERY.json';out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 subprocess.run(['/bin/cp','-cp',str(out),str(dst/'reports'/out.name)],check=True)
 print(json.dumps({k:v for k,v in record.items() if k!='file_hashes'},ensure_ascii=False))
if __name__=='__main__':main()
