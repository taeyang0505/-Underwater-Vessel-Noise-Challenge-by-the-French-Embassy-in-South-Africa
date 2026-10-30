"""Copy immutable completed R18 jobs, with per-file SHA verification.

This helper never starts/stops training or the Pod. It reads only a fixed
allowlist of R18 model artifacts; no credentials or process arguments.
"""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / 'ops/r18_production_r1'
DEST = ROOT / 'models/r18_production_r1/neural'
ALLOW = ('best.pt', 'latest.pt', 'clean_ES_best.npz', 'PLAN.json',
         'epochs.jsonl', 'progress.json', 'FEATURE_RAM_RECEIPT.json', 'summary.json')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(p)


def snapshot():
    s = json.loads((ROOT / 'ops/r18_production_remote_state.json').read_text())
    opts = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o',
            'UserKnownHostsFile=' + s['known_hosts_file']]
    remote_code = r'''
from pathlib import Path
import json,hashlib
p=Path('/workspace/underwater_r18_production/project')
jobs=[]
for f in sorted((p/'models/r18_production_r1/neural').glob('*/*/*/summary.json')):
 d=json.loads(f.read_text())
 if d.get('status')!='completed':continue
 files={}
 for n in ('best.pt','latest.pt','clean_ES_best.npz','PLAN.json','epochs.jsonl','progress.json','FEATURE_RAM_RECEIPT.json','summary.json'):
  a=f.parent/n
  if not a.is_file() or a.is_symlink():raise RuntimeError('Missing/nonregular completed artifact')
  h=hashlib.sha256(a.read_bytes()).hexdigest()
  if n!='summary.json' and d['artifacts'].get(n)!=h:raise RuntimeError('Completed receipt SHA mismatch '+n)
  files[n]={'sha256':h,'bytes':a.stat().st_size}
 jobs.append({'family':d['family'],'regime':d['regime'],'task':d['task'],'files':files})
complete=p/'ops/r18_production_r1/neural/ALL_NEURAL_COMPLETED.json'
print(json.dumps({'jobs':jobs,'all_complete':complete.is_file()}))
'''
    cmd = ['ssh', '-p', str(s['ssh_port']), *opts, s['ssh_host'],
           s['remote_python'] + " - <<'PY'\n" + remote_code + '\nPY']
    response = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
    inventory = json.loads(response.stdout)
    saved = []
    for job in inventory['jobs']:
        family, regime, task = (job[k] for k in ('family', 'regime', 'task'))
        if family not in ('dinov2_small', 'dasheng_base') or regime not in ('D0', 'H0') or task not in ('regression', 'classification'):
            raise ValueError('Job outside fixed R18 scope')
        folder = DEST / family / regime / task
        folder.mkdir(parents=True, exist_ok=True)
        for name in ALLOW:
            record = job['files'][name]
            target = folder / name
            if target.is_file():
                if sha(target) != record['sha256']:
                    raise RuntimeError('Local completed artifact differs; refusing overwrite: ' + str(target))
                continue
            remote = s['remote_project'] + '/models/r18_production_r1/neural/' + '/'.join((family, regime, task, name))
            temp = target.with_name(name + '.download')
            subprocess.run(['scp', '-q', '-P', str(s['ssh_port']), *opts,
                            s['ssh_host'] + ':' + remote, str(temp)], check=True, timeout=180)
            if temp.stat().st_size != record['bytes'] or sha(temp) != record['sha256']:
                raise RuntimeError('Downloaded artifact failed byte/SHA check: ' + name)
            temp.replace(target)
        saved.append({'family': family, 'regime': regime, 'task': task,
                      'all_file_SHA256_verified': True, 'files': job['files']})
    result = {'updated_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'completed' if inventory['all_complete'] and len(saved) == 8 else 'watching',
              'completed_jobs_backed_up': len(saved), 'jobs': saved,
              'training_or_Pod_mutation': False}
    write(OPS / 'BACKUP_STATUS.json', result)
    print(json.dumps({'status': result['status'], 'completed_jobs_backed_up': len(saved)}), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    OPS.mkdir(parents=True, exist_ok=True)
    lock = (OPS / 'backup_watcher.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('R18 backup watcher already active; no second writer started.', flush=True)
        return
    write(OPS / 'BACKUP_WATCHER.json', {'pid': os.getpid(), 'action': 'read/copy completed R18 jobs only'})
    while True:
        try:
            result = snapshot()
            if args.once or result['status'] == 'completed':
                return
        except Exception as error:
            write(OPS / 'BACKUP_ERROR.json', {'updated_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'type': type(error).__name__, 'message': str(error)})
            if args.once:
                raise
        time.sleep(90)


if __name__ == '__main__':
    main()
