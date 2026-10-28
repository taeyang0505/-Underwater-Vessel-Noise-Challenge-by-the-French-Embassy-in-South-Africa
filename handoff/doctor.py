"""Read-only handoff doctor; no dependency installation, model loading or training."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, subprocess

CANONICAL=Path('/Users/hongtaeyang/Documents/ChatGPT/대회/underwater_vessel_noise.nosync')
STATES=('ops/r16_multisupport_full_r1/state.json','models/r16_multisupport_full_r1/progress.json',
        'ops/r16_multisupport_fivefold_r1/state.json')

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def inspect(root):
 result={'root':str(root),'checked_at':datetime.datetime.now().astimezone().isoformat(),
         'read_only':True,'canonical_mac_root':root==CANONICAL,'states':{},'data':{},'processes':{}}
 pids=set()
 for rel in STATES:
  p=root/rel
  if p.exists():
   try:
    value=json.loads(p.read_text()); result['states'][rel]={'mtime':datetime.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(),'value':value}
    pids.update(v for k,v in value.items() if k in ('PID','child_PID') and isinstance(v,int) and v>0)
   except (OSError,ValueError) as e:result['states'][rel]={'error':str(e)}
  else:result['states'][rel]={'missing':True,'meaning':'not restored or not yet produced; do not start duplicate run'}
 for pid in sorted(pids):
  # A live PID alone is insufficient: inspect its command and age as well.
  try:
   r=subprocess.run(['ps','-p',str(pid),'-o','pid=,ppid=,etime=,command='],capture_output=True,text=True)
   result['processes'][str(pid)]={'exists':True if r.returncode==0 else None,'ps':r.stdout.strip(),
                               'note':'No output can mean exited or access denied; inspect in an authorized terminal.'}
  except OSError as e:
   result['processes'][str(pid)]={'exists':None,'note':'Process inspection unavailable; not evidence of process death.','error':str(e)}
 for rel in ('data/raw/Test.csv','data/synthetic/r16_multisupport_v1/manifest.json',
             'models/r16_multisupport_full_r1/ALL_TREES_COMPLETED.json',
             'submissions/r16_multisupport_tree5_r1/submission_r16_tree5.csv',
             'submissions/r16_multisupport_full_r1/submission_r16.csv',
             'submissions/r16_multisupport_fivefold_r1/submission_r16_5fold.csv'):
  p=root/rel;result['data'][rel]={'exists':p.exists()}
  if p.is_file():result['data'][rel].update(bytes=p.stat().st_size,sha256=digest(p))
 for rel in ('ops/r16_multisupport_full_r1/FAILED.json','ops/r16_multisupport_fivefold_r1/FAILED.json'):
  p=root/rel
  if p.exists():result.setdefault('failures',{})[rel]=json.loads(p.read_text())
 stat=os.statvfs(root);result['disk_available_bytes']=stat.f_bavail*stat.f_frsize
 result['warning']='State files in a GitHub clone are snapshots, not proof a process is live. Never auto-start or auto-resume.'
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path)
 args=parser.parse_args();override=os.environ.get('UWVN_PROJECT_ROOT')
 root=args.root or (Path(override) if override else (CANONICAL if CANONICAL.is_dir() else Path(__file__).resolve().parents[1]))
 if not root.is_dir():parser.error('Project directory does not exist')
 print(json.dumps(inspect(root.resolve()),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
