"""Deliver the completed fixed100 experiment without overwriting IDE edits."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,re,subprocess
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);a=p.parse_args();dst=a.destination.resolve()
 if dst==ROOT or not dst.is_dir():raise ValueError('Existing independent Desktop folder required')
 cfg=json.loads((ROOT/'configs/r6_fixed100_v1.json').read_text());run=ROOT/'models/r6_fixed100_v1'
 result=json.loads((run/'results.json').read_text());assert result['status']=='completed' and len(result['summaries'])==5
 assert all(v['epochs_run']==100 and len(v['epoch_checkpoint_sha256'])==100 for v in result['summaries'])
 old={v['path']:v['sha256'] for v in json.loads((dst/'DESKTOP_PACKAGE_MANIFEST.json').read_text())['files']}
 prior=dst/'reports/R6_GUARDRAILS_DESKTOP_DELIVERY.json'
 if prior.exists():old.update({n:v['sha256'] for n,v in json.loads(prior.read_text())['file_hashes'].items()})
 names={'configs/r6_fixed100_v1.json','pipeline/fixed100_r6.py','ops/run_r6_fixed100.py','ops/status_r6_fixed100.py',
 'ops/build_r6_fixed100_report.py','ops/deliver_r6_fixed100.py','ops/benchmark_r6_cpu_threads.py','ops/r6_fixed100_training.log','ops/r6_fixed100_state.json'}
 for pattern in ['ops/*fixed100*.py','reports/R6_FIXED100*','reports/R6_CPU_THREAD*',
 'reports/R6_PUBLIC_TRAINING*','reports/R6_TEST_GAP_CLAIMS_REVIEW.md']:
  names.update(str(v.relative_to(ROOT)) for v in ROOT.glob(pattern) if v.is_file())
 for folder in [run,ROOT/'reports/r6_fixed100_figures']:
  names.update(str(v.relative_to(ROOT)) for v in folder.rglob('*') if v.is_file())
 # Dependencies already delivered in the source-data handoff remain read-only.
 old_manifest=json.loads((ROOT/'data/synthetic/r6_cpu_guardrails_v3_schemafix/manifest.json').read_text())
 dependency_pins={**cfg['used_input_sha256'],**cfg['original_frozen_core17_sha256'],**old_manifest['freeze']['code_sha256'],
 'configs/r6_cpu_guardrails_v2.json':old_manifest['freeze']['config_sha256']}
 for n,h in dependency_pins.items():
  if not (dst/n).is_file() or sha(dst/n)!=h:raise ValueError('Desktop source dependency differs; preserved for review: '+n)
 records={};updated=0;size=0
 for n in sorted(names):
  source,target=ROOT/n,dst/n;digest=sha(source)
  if target.is_symlink():raise ValueError('Preserve Desktop symlink: '+n)
  if target.exists() and sha(target)!=digest and old.get(n)!=sha(target):raise ValueError('Preserve IDE edit: '+n)
 for n in sorted(names):
  source,target=ROOT/n,dst/n;digest=sha(source)
  if not target.exists() or sha(target)!=digest:
   target.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['/bin/cp','-cp',str(source),str(target)],check=True);updated+=1
  assert sha(target)==digest
  records[n]={'sha256':digest,'bytes':source.stat().st_size};size+=source.stat().st_size
 report=dst/'reports/R6_FIXED100_TRAINING_REPORT.md';links=[]
 for value in re.findall(r'\]\(([^)]+)\)',report.read_text()):
  if '://' in value or value.startswith('#'):continue
  if not (report.parent/value).exists():raise ValueError('Broken report link: '+value)
  links.append(value)
 rec={'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),'destination':str(dst),'files_verified':len(records),
 'copied_or_updated':updated,'bytes_verified':size,'epoch_checkpoints_verified':500,'local_report_links_verified':len(links),
 'user_edits_preserved':True,'report':str(report),'report_sha256':sha(report),'files':records,'GPU_Test_SSH_submission':False}
 out=ROOT/'reports/R6_FIXED100_DESKTOP_DELIVERY.json';out.write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n')
 subprocess.run(['/bin/cp','-cp',str(out),str(dst/'reports'/out.name)],check=True)
 print(json.dumps({k:v for k,v in rec.items() if k!='files'},ensure_ascii=False))
if __name__=='__main__':main()
