#!/usr/bin/env python3
"""Mac stdlib bridge: frozen remote development -> report -> query -> CSV.

Never changes training/finalizer source, restarts training, uploads a challenge
submission, or changes a Pod. Raw Test is read by the frozen local finalizer
only after decision+report receipts. All copies are hash-verified and immutable.
"""
from __future__ import annotations
import argparse,csv,fcntl,hashlib,json,math,os,shlex,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REL='reports/r18_production_r1/finalization'
DEV=('fresh_fit_D0','fresh_fit_H0','local_gate','domain_holdout','native12_VS_hf_high_negative','native12_VS_st_francis_dense_negative')
FAMILIES=('lightgbm','dinov2_small','dasheng_base')
os.environ.update({k:'2'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')})
class AlreadyRunning(RuntimeError):pass

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as stream:
        for b in iter(lambda:stream.read(2**20),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)

def safe_rel(name):
    p=Path(name)
    if p.is_absolute()or'..'in p.parts:raise ValueError('Project-relative path required')
    return str(p)

REMOTE_TRANSFER=r'''
import sys,json,hashlib,os
from pathlib import Path
root=Path(sys.argv[1]).resolve();action=sys.argv[2];rel=sys.argv[3]
p=(root/rel).resolve()
if not p.is_relative_to(root):raise ValueError('Unsafe path')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
if action=='stat':
 print(json.dumps({'exists':p.is_file(),'sha256':digest(p)if p.is_file()else None,'bytes':p.stat().st_size if p.is_file()else None}))
elif action=='get':sys.stdout.buffer.write(p.read_bytes())
elif action=='put':
 expected=sys.argv[4];data=sys.stdin.buffer.read()
 if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Transfer input SHA')
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if digest(p)!=expected:raise FileExistsError('Refuse overwrite '+str(p))
 else:
  t=p.with_name(p.name+'.bridge.tmp');t.write_bytes(data)
  if digest(t)!=expected:raise ValueError('Saved transfer SHA')
  t.replace(p)
 print(json.dumps({'sha256':digest(p),'bytes':p.stat().st_size,'immutable':True}))
'''

REMOTE_JSON_SNAPSHOT=r'''
import sys,json,hashlib,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();p=(root/sys.argv[2]).resolve()
if not p.is_relative_to(root):raise ValueError('Unsafe path')
# Status writers replace their files atomically. Read one version once and
# compute the payload/hash from those same bytes, instead of racing stat/get.
for attempt in range(5):
 try:data=p.read_bytes()
 except FileNotFoundError:
  print(json.dumps({'exists':False}));break
 try:
  text=data.decode('utf-8');json.loads(text)
 except (UnicodeDecodeError,json.JSONDecodeError):
  if attempt==4:raise
  time.sleep(0.02);continue
 print(json.dumps({'exists':True,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'raw_utf8':text}));break
'''

REMOTE_STEP=r'''
import sys,json,subprocess,os,fcntl,time
from pathlib import Path
root=Path(sys.argv[1]);label=sys.argv[2];cmd=json.loads(sys.argv[3]);folder=root/'reports/r18_production_r1/finalization/bridge_remote_steps'/label
folder.mkdir(parents=True,exist_ok=True)
lock=(folder/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
status=folder/'STATUS.json'
if status.exists():
 s=json.loads(status.read_text())
 if s.get('status')=='completed':print(json.dumps(s));sys.exit(0)
 if s.get('status')=='running':
  try:
   os.kill(s['pid'],0);stat=Path('/proc')/str(s['pid'])/'stat'
   alive=not(stat.exists()and stat.read_text().split(') ',1)[1].startswith('Z'))
  except ProcessLookupError:alive=False
  if alive:print(json.dumps(s));sys.exit(0)
runner="""
import sys,json,subprocess,os,fcntl,time,traceback
from pathlib import Path
folder=Path(sys.argv[1]);cmd=json.loads(sys.argv[2]);root=Path(sys.argv[3]);status=folder/'STATUS.json'
execution=(folder/'execution.lock').open('a');fcntl.flock(execution,fcntl.LOCK_EX|fcntl.LOCK_NB)
def write(v):
 t=status.with_name('STATUS.tmp');t.write_text(json.dumps(v,indent=2)+'\\n');t.replace(status)
write(dict(status='running',pid=os.getpid(),command=cmd,started_unix=time.time()))
try:
 with(folder/'phase.log').open('a')as log:result=subprocess.run(cmd,cwd=root,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
 write(dict(status='completed'if result.returncode==0 else'failed',pid=os.getpid(),returncode=result.returncode,command=cmd,finished_unix=time.time()))
except BaseException as exc:
 write(dict(status='failed',pid=os.getpid(),error=repr(exc),command=cmd));raise
"""
with(folder/'runner.log').open('a')as log:
 p=subprocess.Popen([sys.executable,'-c',runner,str(folder),json.dumps(cmd),str(root)],cwd=root,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=dict(status='running',pid=p.pid,command=cmd,started_unix=time.time());t=status.with_name('LAUNCH.tmp');t.write_text(json.dumps(s));t.replace(status)
print(json.dumps(s))
'''

class Bridge:
    def __init__(self,args):
        self.args=args;self.root=args.input_root.resolve();self.state=read(self.root/args.state)
        self.remote=self.state['remote_project'];self.rpython=self.state['remote_python']
        self.folder=self.root/'ops/r18_production_r1/finalization_bridge';self.folder.mkdir(parents=True,exist_ok=True)
        self.ssh=['ssh','-p',str(self.state['ssh_port']),'-o','BatchMode=yes','-o','ConnectTimeout=15',
            '-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+self.state['known_hosts_file'],
            '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3',self.state['ssh_host']]
        self.poll=max(5,min(args.poll_seconds,60));self.lock=None

    def event(self,phase,**values):
        obj={'phase':phase,'pid':os.getpid(),'unix_time':time.time(),**values};write(self.folder/'progress.json',obj)
        with(self.folder/'events.jsonl').open('a')as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
        print(json.dumps(obj,ensure_ascii=False),flush=True)

    def rpc(self,code,*args,input_bytes=None,json_result=True):
        command=shlex.join([self.rpython,'-c',code,*map(str,args)])
        for attempt in range(3):
            result=subprocess.run([*self.ssh,command],input=input_bytes,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
            if not result.returncode:break
            if result.returncode!=255 or attempt==2:raise RuntimeError('SSH command failed: '+result.stderr.decode(errors='replace')[-2000:])
            time.sleep((5,15)[attempt])
        return json.loads(result.stdout)if json_result else result.stdout

    def verify_remote_pins(self,pins):
        code="""import sys,json,hashlib\nfrom pathlib import Path\nroot=Path(sys.argv[1]);pins=json.loads(sys.argv[2]);result={}\nfor name,expected in pins.items():\n p=root/name;h=hashlib.sha256(p.read_bytes()).hexdigest()\n if h!=expected:raise ValueError('Frozen remote source differs: '+name)\n result[name]=h\nprint(json.dumps(result))"""
        result=self.rpc(code,self.remote,json.dumps(pins))
        if result!=pins:raise ValueError('Remote frozen source verification differs')

    def remote_json(self,rel):
        snapshot=self.rpc(REMOTE_JSON_SNAPSHOT,self.remote,safe_rel(rel))
        if not snapshot['exists']:return None
        data=snapshot['raw_utf8'].encode('utf-8')
        if len(data)!=snapshot['bytes']or hashlib.sha256(data).hexdigest()!=snapshot['sha256']:
            raise ValueError('Remote JSON snapshot integrity differs')
        return json.loads(data)

    def download(self,rel):
        rel=safe_rel(rel);p=self.root/rel;stat=self.rpc(REMOTE_TRANSFER,self.remote,'stat',rel)
        if not stat['exists']:raise FileNotFoundError('Missing remote '+rel)
        if p.exists()and sha(p)==stat['sha256']:return stat
        data=self.rpc(REMOTE_TRANSFER,self.remote,'get',rel,json_result=False)
        if hashlib.sha256(data).hexdigest()!=stat['sha256']:raise ValueError('Download SHA differs')
        if p.exists():
            # Same immutable NPZ can have equivalent local/remote provenance
            # receipts whose absolute model paths/creation timestamps differ.
            if rel.endswith('.json')and'/predictions/'in rel:
                old,new=read(p),json.loads(data)
                keys=('status','generation','family','dataset','n','input_sha256','prediction_sha256','feature_masking_at_inference','optimizer_steps','fitting_performed')
                if all(old.get(k)==new.get(k)for k in keys)and sorted(old['model_sha256'].values())==sorted(new['model_sha256'].values()):
                    return{**stat,'kept_equivalent_local_receipt':True}
            raise FileExistsError('Refuse overwrite local '+rel)
        p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.bridge.tmp');tmp.write_bytes(data)
        if sha(tmp)!=stat['sha256']:raise ValueError('Local saved SHA differs')
        tmp.replace(p);return stat

    def upload(self,rel):
        rel=safe_rel(rel);p=self.root/rel;digest=sha(p)
        result=self.rpc(REMOTE_TRANSFER,self.remote,'put',rel,digest,input_bytes=p.read_bytes())
        if result['sha256']!=digest:raise ValueError('Uploaded SHA differs')
        return result

    def local_phase(self,phase,stage=None):
        cmd=[str(self.root/'.venv_r17_eda/bin/python'),str(self.root/'ops/finalize_r18_production.py'),phase,'--input-root',str(self.root)]
        if stage:cmd+=['--stage',stage]
        log=self.folder/(phase+('_'+stage if stage else'')+'.log')
        self.event('local_'+phase,stage=stage)
        with log.open('a')as stream:subprocess.run(cmd,cwd=self.root,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT,check=True)
        return{'command':cmd,'log_sha256':sha(log)}

    def remote_phase(self,phase,stage=None):
        label=phase+('_'+stage if stage else'')
        cmd=[self.rpython,self.remote+'/ops/finalize_r18_production.py',phase,'--input-root',self.remote]
        if stage:cmd+=['--stage',stage]
        if phase=='collect-neural':cmd+=['--device','cuda','--batch',str(self.args.batch)]
        self.event('remote_'+label,status='launch_or_verified_resume')
        result=self.rpc(REMOTE_STEP,self.remote,label,json.dumps(cmd))
        while result['status']=='running':
            time.sleep(self.poll)
            result=self.remote_json(f'{REL}/bridge_remote_steps/{label}/STATUS.json')
            if result is None:raise RuntimeError('Remote step receipt disappeared')
        if result['status']!='completed':raise RuntimeError('Remote phase failed; outputs preserved: '+json.dumps(result))
        self.event('remote_'+label,status='completed');return result

    def download_predictions(self,stage):
        names=DEV if stage=='development'else('local_report',)if stage=='report'else('query',)
        receipts={}
        for generation in('old','new'):
            for family in FAMILIES:
                for name in names:
                    for ext in('npz','json'):
                        rel=f'{REL}/predictions/{generation}/{family}/{name}.{ext}';receipts[rel]=self.download(rel)
        write(self.folder/f'download_{stage}_SHA256.json',receipts)

    def run(self):
        self.lock=(self.folder/'bridge.lock').open('a')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise AlreadyRunning('Another R18 bridge process already holds lock')
        plan=read(self.root/f'{REL}/PLAN.json')
        pins={**plan['source_sha256'],**read(self.root/'ops/r18_production_r1/neural/RUN_PLAN.json')['source_sha256']}
        for name,h in pins.items():
            if sha(self.root/name)!=h:raise ValueError('Frozen source differs: '+name)
        self.verify_remote_pins(pins)
        self.event('bridge_started',PLAN_sha256=sha(self.root/f'{REL}/PLAN.json'),source_sha256=sha(__file__),Test_read=False)
        while True:
            done=self.remote_json(f'{REL}/orchestration/COMPLETED.json')
            if done and done.get('status')=='completed':break
            failed=self.remote_json(f'{REL}/orchestration/FAILED.json')
            if failed:raise RuntimeError('Remote development failed: '+json.dumps(failed))
            running=self.remote_json(f'{REL}/orchestration/progress.json')
            if running and running.get('pid'):
                code="""import sys,os,json\nfrom pathlib import Path\npid=int(sys.argv[1])\ntry:\n os.kill(pid,0);p=Path('/proc')/str(pid)/'stat';alive=not(p.exists()and p.read_text().split(') ',1)[1].startswith('Z'))\nexcept ProcessLookupError:alive=False\nprint(json.dumps({'alive':alive}))"""
                if not self.rpc(code,running['pid'])['alive']:raise RuntimeError('Remote completion orchestrator exited before receipt; no training restart')
            self.event('waiting_remote_development',report_and_Test_unread=True);time.sleep(self.poll)
        if self.remote_json(f'{REL}/DECISION_FREEZE.json')is None:self.remote_phase('select')
        # Copy a complete frozen policy and compact predictions; never weights.
        for name in('PLAN.json','selected_policy.json','DECISION_FREEZE.json','BLEND_METRICS_development.csv','INDIVIDUAL_METRICS_development.csv'):
            self.download(f'{REL}/{name}')
        self.download_predictions('development')
        self.local_phase('prepare-panels','report')
        for name in('inputs/local_report.npz','INPUTS_report.json'):self.upload(f'{REL}/{name}')
        # The legacy report collector reads the pinned original saved output.
        # It is deliberately deployed only after policy selection is frozen.
        self.upload('models/r16_v1_fourway_r2/predictions/local_report.npz')
        for phase in('collect-tree','collect-neural','collect-legacy-saved'):self.remote_phase(phase,'report')
        if self.remote_json(f'{REL}/REPORT_DONE.json')is None:self.remote_phase('report')
        self.download_predictions('report')
        for name in('REPORT_DONE.json','BLEND_METRICS_report.csv','INDIVIDUAL_METRICS_report.csv'):self.download(f'{REL}/{name}')
        # Local finalizer performs the first and only raw Test read now.
        query_receipt=self.root/f'{REL}/INPUTS_query.json'
        marker=self.folder/'QUERY_CAPTURE_STARTED.json'
        if not query_receipt.exists():
            if marker.exists():raise RuntimeError('A previous query capture started without its atomic receipt; refuse any raw Test reopen. Preserve partial artifacts for inspection.')
            write(marker,{'decision_sha256':sha(self.root/f'{REL}/DECISION_FREEZE.json'),'started_unix':time.time(),
                'raw_Test_capture_attempt':1,'automatic_raw_Test_retry':False})
        self.local_phase('prepare-query')
        query=read(self.root/f'{REL}/INPUTS_query.json')
        if query['raw_Test_open_count']!=1 or query['Test_used_for_fit'] or query['query_labels_used']:raise ValueError('Query single-read/fit contract')
        freeze=sha(self.root/f'{REL}/DECISION_FREEZE.json')
        if query['decision_sha256']!=freeze:raise ValueError('Query not bound to decision')
        for name in('inputs/query.npz','INPUTS_query.json'):self.upload(f'{REL}/{name}')
        self.upload('models/r16_v1_fourway_r2/query_predictions.npz')
        for phase in('collect-tree','collect-neural','collect-legacy-saved'):self.remote_phase(phase,'query')
        if self.remote_json(f'{REL}/COMPLETED.json')is None:self.remote_phase('export')
        self.download_predictions('query')
        for name in('COMPLETED.json','R18_PRODUCTION_COMPARISON.md'):self.download(f'{REL}/{name}')
        manifest_rel='submissions/r18_production_r1/candidate_manifest.json';self.download(manifest_rel)
        manifest=read(self.root/manifest_rel)
        if manifest['decision_sha256']!=freeze or manifest['official_upload']:raise ValueError('Export identity/action differs')
        ids=[]
        with(self.root/'submissions/r17_fixed_blends_20261030/submission_r17_B1_L60_DINO25_Dash15.csv').open(newline='')as f:
            r=csv.DictReader(f);ids=[row['ID']for row in r]
        verified=[]
        for candidate in manifest['candidates']:
            rel=candidate['path']
            if rel not in{f'submissions/r18_production_r1/submission_r18_{name}.csv'for name in('B1','B2','selected')}:raise ValueError('Unexpected candidate destination')
            self.download(rel);p=self.root/rel
            if sha(p)!=candidate['sha256']:raise ValueError('Candidate SHA differs')
            with p.open(newline='')as stream:
                reader=csv.DictReader(stream)
                if reader.fieldnames!=['ID','share_added_vessel','score_added_vessel']:raise ValueError('CSV columns')
                rows=list(reader)
            if len(rows)!=1500 or[r['ID']for r in rows]!=ids:raise ValueError('CSV count/ID/order differs')
            values=[float(r[k])for r in rows for k in('share_added_vessel','score_added_vessel')]
            if not all(math.isfinite(v)and 0<=v<=1 for v in values):raise ValueError('CSV finite/range')
            verified.append({'path':rel,'sha256':sha(p),'rows':len(rows)})
        for name,h in pins.items():
            if sha(self.root/name)!=h:raise ValueError('Frozen source changed during bridge')
        self.verify_remote_pins(pins)
        write(self.folder/'COMPLETED.json',{'status':'completed','decision_sha256':freeze,'candidates':verified,
            'raw_Test_open_count':1,'Test_used_for_fit':False,'official_upload':False,'Pod_changes':False,'source_unchanged':True,'finished_unix':time.time()})
        self.event('all_CSVs_retrieved_and_verified',candidates=verified)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-root',type=Path,default=ROOT)
    ap.add_argument('--state',default='ops/r18_production_remote_state.json')
    ap.add_argument('--poll-seconds',type=int,default=30)
    ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args();bridge=Bridge(args)
    if args.check_only:
        local=read(bridge.root/f'{REL}/PLAN.json');remote=bridge.remote_json(f'{REL}/PLAN.json')
        if remote!=local:raise ValueError('Local/remote registered plan differs')
        print(json.dumps({'status':'connection_and_plan_verified','remote':bridge.remote,'PLAN_sha256':sha(bridge.root/f'{REL}/PLAN.json')},ensure_ascii=False));return
    try:bridge.run()
    except AlreadyRunning as exc:
        print(json.dumps({'status':'already_running','duplicate_process_started':False,'reason':str(exc)}));return
    except BaseException as exc:
        write(bridge.folder/'FAILED.json',{'status':'failed','error':repr(exc),'pid':os.getpid(),'unix_time':time.time(),
            'outputs_preserved':True,'training_restarted':False,'automatic_overwrite':False})
        traceback.print_exc();raise

if __name__=='__main__':main()
