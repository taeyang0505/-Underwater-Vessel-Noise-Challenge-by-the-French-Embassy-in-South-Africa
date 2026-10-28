"""Deploy only the user-approved external Fold0 stop guard; no training edits."""
from pathlib import Path
import base64
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = (ROOT / 'ops/r9_fold_boundary_guard.py').read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    tests = json.loads((ROOT / 'reports/R9_FOLD_BOUNDARY_GUARD_VERIFICATION.json').read_text())
    if tests['status'] != 'passed' or tests['guard_source_sha256'] != digest:
        raise ValueError('Current guard source has no matching passed fixture receipt')
    remote = r'''
import base64,hashlib,json,os,pathlib,subprocess,time
base=pathlib.Path('/workspace/underwater_vessel_noise_r9_r1')
project=base/'project';py=base/'venv/bin/python';ops=base/'ops';ops.mkdir(exist_ok=True)
digest=EXPECTED_SHA
source=base64.b64decode(PAYLOAD)
assert hashlib.sha256(source).hexdigest()==digest
target=ops/'r9_fold_boundary_guard.py'
receipt=ops/'fold0_boundary_guard_deployment.json'
assert not receipt.exists(),'Existing deployment must be inspected, never duplicated'
assert not target.exists(),'Never overwrite an existing guard'
target.write_bytes(source)
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','PYTHONPATH':str(ops)}
test=subprocess.run([str(py),str(target),'--self-test','--test-output',str(ops/'fold0_boundary_guard_fixture.json')],env=env,capture_output=True,text=True,check=True)
result=json.loads(test.stdout.splitlines()[-1]);assert result['status']=='passed' and result['guard_source_sha256']==digest
smoke="""from r9_fold_boundary_guard import Inotify,IN_MOVED_TO
from pathlib import Path
import tempfile,time
with tempfile.TemporaryDirectory(prefix='r9_inotify_fixture_') as temp:
 p=Path(temp);w=Inotify(p)
 try:
  (p/'summary.json.tmp').write_text('{}');(p/'summary.json.tmp').replace(p/'summary.json')
  events=w.events(1.0)
  assert any(e['name']=='summary.json' and e['mask']&IN_MOVED_TO for e in events)
 finally:w.close()
print('Linux inotify fixture passed; no process signals')
"""
subprocess.run([str(py),'-c',smoke],env=env,capture_output=True,text=True,check=True)
out=ops/'fold0_boundary_pause_r1'
cmd=[str(py),'-u',str(target),'--base',str(base),'--project',str(project),'--run',str(project/'models/group_cv_minute_full_r9_v1'),'--config',str(project/'configs/r9_full_training_v1.json'),'--output',str(out),'--pid','2257','--startticks','86168117','--pgid','2257','--supervisor','2256','--supervisor-startticks','86168115','--config-sha256','9962b8a5c41708b9f8a51cb77cd0aa3295969b830077e240f2329b1f6c64679a','--data-hash','9788ec783be139d51a45a49ec3b22539f10a8b9eff85f3acb501127ae82a4092']
state={'status':'launching_external_guard','guard_sha256':digest,'Linux_fixture_passed':True,'Linux_inotify_fixture_passed':True,'frozen_training_code_changed':False,'command':cmd}
with receipt.open('x') as f:json.dump(state,f,indent=2)
log=ops/'fold0_boundary_guard.log'
with log.open('xb') as stream:
 child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
state['guard_pid']=child.pid
receipt.write_text(json.dumps(state,indent=2)+'\n')
deadline=time.monotonic()+30
while time.monotonic()<deadline:
 if (out/'status.json').exists():
  s=json.loads((out/'status.json').read_text())
  if s['status'] in ['armed','paused_after_last_model_summary']:
   state.update(status=s['status'],guard_status=s);break
  if s['status'].startswith('failed'):raise RuntimeError(json.dumps(s))
 if child.poll() is not None:raise RuntimeError('Guard exited before arming: '+log.read_text())
 time.sleep(.1)
else:raise TimeoutError('Guard arming not confirmed; inspect existing deployment before any retry')
receipt.write_text(json.dumps(state,indent=2)+'\n')
print('R9_GUARD_DEPLOYMENT '+json.dumps(state),flush=True)
'''
    remote = remote.replace('EXPECTED_SHA', repr(digest)).replace('PAYLOAD', repr(base64.b64encode(source).decode()))
    command = ['ssh','-tt','-o','BatchMode=yes','-o','ConnectTimeout=15',
               '-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile=/private/tmp/underwater_runpod_known_hosts',
               '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',
               'j5u5bv4ewso40i-644111f6@ssh.runpod.io']
    message = "stty -echo -onlcr\n/workspace/underwater_vessel_noise_r9_r1/venv/bin/python -u - <<'R9GUARD'\n"+remote+"\nR9GUARD\nexit\n"
    result = subprocess.run(command,input=message.encode(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=100)
    output = result.stdout.decode(errors='replace')
    values = [json.loads(line.split('R9_GUARD_DEPLOYMENT ',1)[1]) for line in output.splitlines() if line.startswith('R9_GUARD_DEPLOYMENT ')]
    if result.returncode or len(values) != 1:
        (ROOT/'ops/r9_guard_deployment_transport.log').write_text(output)
        raise RuntimeError('Deployment outcome requires inspection; no automatic retry. See ops/r9_guard_deployment_transport.log')
    value = values[0]
    (ROOT/'ops/r9_boundary_guard_deployment.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'status':value['status'],'guard_pid':value['guard_pid'],'guard_sha256':digest,'Linux_fixtures_passed':True,'local_receipt':'ops/r9_boundary_guard_deployment.json'}))


if __name__=='__main__':
    main()
