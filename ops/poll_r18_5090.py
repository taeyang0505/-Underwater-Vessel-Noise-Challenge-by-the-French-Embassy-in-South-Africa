"""Read-only R18 remote status; never enumerate process arguments or credentials."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    state = json.loads((ROOT / 'ops/r18_production_remote_state.json').read_text())
    remote_code = r'''
from pathlib import Path
import json,subprocess
p=Path('/workspace/underwater_r18_production/project')
out={}
for name in ['ops/r18_production_r1/neural/progress.json',
             'ops/r18_production_r1/neural/training_health.json',
             'ops/r18_production_r1/neural/LATEST_ERROR.json',
             'ops/r18_production_r1/neural/ALL_NEURAL_COMPLETED.json',
             'ops/r18_production_r1/neural/PAUSED.json',
             'ops/r18_production_r1/neural/PREFLIGHT_dinov2_small.json',
             'ops/r18_production_r1/neural/PREFLIGHT_dasheng_base.json']:
 f=p/name
 if f.is_file():
  d=json.loads(f.read_text())
  if 'PREFLIGHT' in name:
   d={'status':d['status'],'models':{k:{n:v for n,v in z.items() if n in ['steady_seconds','rows_per_second','training_epoch_estimate_seconds','peak_cuda_allocated_bytes']} for k,z in d['models'].items()}}
  out[name]=d
out['completed_jobs']=[]
for f in sorted((p/'models/r18_production_r1/neural').glob('*/*/*/summary.json')):
 d=json.loads(f.read_text())
 out['completed_jobs'].append({k:d[k] for k in ['family','regime','task','best_epoch','epochs_run','stop_reason','clean_ES_best']})
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
print(json.dumps(out,ensure_ascii=False))
'''
    cmd = ['ssh', '-p', str(state['ssh_port']), '-o', 'BatchMode=yes', '-o',
           'ConnectTimeout=15', '-o', 'UserKnownHostsFile=' + state['known_hosts_file'],
           state['ssh_host'], state['remote_python'] + " - <<'PY'\n" + remote_code + '\nPY']
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=45, check=True)
    value = json.loads(result.stdout)
    dest = ROOT / 'reports/r18_production_r1/remote_status_latest.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(dest)
    print(json.dumps(value,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
