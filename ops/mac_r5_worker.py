"""Hash-pinned Mac CPU tree worker; no SSH/fit occurs in --self-test.

Real execution is explicit --once or --watch. Only controller metadata travels
to the Mac; raw arrays/audio remain in the existing local package. The seven
tree files travel back to an exclusive staging archive, then completion.json
is published last. Failure is terminal: no partial fit, retry or server fit.
"""
from pathlib import Path, PurePosixPath
import argparse, base64, contextlib, datetime, fcntl, hashlib, json, os, re
import shlex, shutil, signal, subprocess, sys, tempfile, time, uuid, zipfile

ROOT=Path(__file__).resolve().parents[1]
CONFIG_SHA='c6979cbe66e9da0ff840f30477a824d3f7e28fd1bc2e83295924e968fea8d75a'
PRODUCER_SHA='d1b6483759932662560505cd0f281e9239a6e0aafd780afbc1434375626148e7'
CORE17={'pipeline/'+n for n in ('runner_r5.py','neural_r5.py','lgbm_r5.py','fold_data_r5.py',
    'grid_inputs_r5.py','dynamic_passage_r5.py','observation_augmentation_r5.py',
    'physics_features_r5.py','hierarchical_ensemble_r5.py','hybrid_models_r5.py',
    'grid_models_r5.py','foundation_masks_r5.py','load_foundation_r5.py','mixtures.py',
    'models.py','background_prior_features.py','inference_r5.py')}
CORE19=CORE17|{'pipeline/train_lgbm.py','pipeline/prototypes.py'}
OPS={'ops/runner_r5_mac.py','ops/mac_lgbm_r5.py','ops/mac_r5_worker.py'}
TREE_FILES={'summary.json','share.txt','presence.txt','prototypes.npz','physics_pca.json',
    'inner_predictions.npz','producer_summary.json'}
CORE_TREE_FILES=TREE_FILES-{'summary.json','producer_summary.json'}
KNOWN_HOSTS='/private/tmp/underwater_runpod_known_hosts'
FORMAT='r5_mac_worker_v1'
REQUEST_KEYS={'format','job_id','config_sha256','training_binding','observation_name','observation_policy',
    'context_sha256','selection_sha256','context_source_path','selection_source_path','cnn_summaries',
    'prior_sha256','expected_core19_sha256','tree_settings','output_model_path','executor_plan_sha256','request_sha256'}

def canon(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def strict_json(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('Duplicate JSON key: '+k)
            result[k]=v
        return result
    def bad(value):raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)

def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    write_new(temp,value);os.replace(temp,path)

def safe_file(root,relative):
    p=PurePosixPath(relative)
    if not isinstance(relative,str) or not relative or '\\' in relative or p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe relative path')
    out=Path(root)
    if out.is_symlink():raise ValueError('Symlink root forbidden')
    for part in p.parts:
        out/=part
        if out.is_symlink():raise ValueError('Symlink file forbidden')
    if not out.is_file():raise ValueError('Missing regular file: '+relative)
    return out

def archive_check(path,files):
    if set(files)!=TREE_FILES:raise ValueError('Exactly seven tree files required')
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=7 or set(z.namelist())!=TREE_FILES:
            raise ValueError('ZIP names/duplicate entries/CRC invalid')
        if sum(i.file_size for i in z.infolist())>512*2**20:raise ValueError('Oversized tree archive')
        if z.testzip() is not None:raise ValueError('ZIP CRC invalid')
        for name in TREE_FILES:
            i=z.getinfo(name)
            if PurePosixPath(name).name!=name or ((i.external_attr>>16)&0o170000)==0o120000:
                raise ValueError('ZIP path or symlink forbidden')
            if hashlib.sha256(z.read(name)).hexdigest()!=files[name]:raise ValueError('ZIP member SHA mismatch')

# This program reads JSON/source metadata and commits the returned tree bundle.
# It never imports the project, numpy, LightGBM, Torch, or reads source arrays.
REMOTE_PROGRAM=r'''
import base64,hashlib,json,os,pathlib,re,sys,zipfile
a=json.load(sys.stdin);base=pathlib.Path(a['base']);run=a['run']
if not re.fullmatch(r'/workspace/[A-Za-z0-9_.-]+',str(base)) or not re.fullmatch(r'[A-Za-z0-9_-]+',run):raise ValueError('Unsafe base/run')
project=base/'project';rr=project/'models'/run;spool=base/'ops/mac_cpu_jobs'/run
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def regular(p):
 for q in [p,*p.parents]:
  if q.is_symlink():raise ValueError('Remote symlink forbidden')
 if not p.is_file():raise ValueError('Remote regular file required')
 return p
def job():
 f=a['fold']
 if type(f) is not int or not 0<=f<5:raise ValueError('Bad fold')
 return spool/('fold_%02d'%f)
def request(j):
 p=regular(j/'request.json')
 if digest(p)!=a['request_file_sha256']:raise ValueError('Remote request changed')
 r=json.loads(p.read_text())
 if r.get('request_sha256')!=a['request_sha256']:raise ValueError('Remote request identity changed')
 return r
op=a['action'];result={}
if op=='list':
 result={'jobs':[{'fold':f,'request':(spool/('fold_%02d'%f)/'request.json').is_file(),'completed':(spool/('fold_%02d'%f)/'completion.json').exists(),'failed':(spool/('fold_%02d'%f)/'FAILED.json').exists()} for f in range(5)]}
elif op=='read':
 p=pathlib.Path(a['path']);s=str(p)
 allowed=s==str(rr/'executor_plan.json')
 for f in range(5):
  jf=spool/('fold_%02d'%f);rf=rr/('fold_%02d'%f)
  allowed|=p==jf/'request.json'
  allowed|=p in [rf/'context.json',rf/'preprocessing_selection.json',rf/'shared_background_prior.json']
  allowed|=p in [rf/'preprocessing_pilot'/o/'bg_cnn/summary.json' for o in ['raw','resolution_01','resolution_01_endpoint']]
 if not allowed:raise ValueError('Only declared JSON metadata may be fetched')
 regular(p)
 if p.stat().st_size>4194304:raise ValueError('Oversized metadata')
 b=p.read_bytes();result={'data':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()}
elif op=='code':
 result={'hashes':{}}
 for name in a['names']:
  if not re.fullmatch(r'(pipeline|ops)/[A-Za-z0-9_]+\.py',name):raise ValueError('Only code metadata permitted')
  result['hashes'][name]=digest(regular(project/name))
elif op=='prepare':
 j=job();request(j)
 if (j/'completion.json').exists() or (j/'FAILED.json').exists():raise ValueError('Terminal remote job')
 stage=j/'staging';stage.mkdir(exist_ok=True)
 if stage.is_symlink() or (stage/'artifacts.zip').exists():raise ValueError('Existing remote stage')
 nonce=a['nonce']
 if not re.fullmatch(r'[0-9a-f]{32}',nonce):raise ValueError('Bad upload nonce')
 p=stage/('artifacts.'+nonce+'.zip.tmp')
 with p.open('xb'):pass
 result={'temporary_path':str(p)}
elif op=='commit':
 j=job();r=request(j);c=a['completion'];tmp=pathlib.Path(a['temporary_path']);stage=j/'staging'
 if tmp.parent!=stage or not re.fullmatch(r'artifacts\.[0-9a-f]{32}\.zip\.tmp',tmp.name):raise ValueError('Unsafe staging path')
 regular(tmp)
 if (j/'completion.json').exists() or (j/'FAILED.json').exists() or (stage/'artifacts.zip').exists():raise ValueError('Terminal/existing remote result')
 expected={'summary.json','share.txt','presence.txt','prototypes.npz','physics_pca.json','inner_predictions.npz','producer_summary.json'}
 if c.get('format')!='r5_mac_tree_completion_v1' or c.get('job_id')!=r['job_id'] or c.get('request_sha256')!=r['request_sha256'] or set(c.get('artifact_sha256',{}))!=expected:raise ValueError('Bad completion schema')
 if digest(tmp)!=c['archive_sha256']:raise ValueError('Uploaded ZIP SHA differs')
 with zipfile.ZipFile(tmp) as z:
  if len(z.namelist())!=7 or set(z.namelist())!=expected or sum(i.file_size for i in z.infolist())>536870912 or z.testzip() is not None:raise ValueError('ZIP CRC/entries invalid')
  for name,h in c['artifact_sha256'].items():
   if ((z.getinfo(name).external_attr>>16)&0o170000)==0o120000 or hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('ZIP member invalid')
  p=json.loads(z.read('producer_summary.json'))
  if hashlib.sha256(z.read('producer_summary.json')).hexdigest()!=c['producer_receipt_sha256'] or p.get('status')!='completed' or p.get('fixture_only') is not False or p.get('config_sha256')!=r['config_sha256'] or p.get('training_binding')!=r['training_binding'] or p.get('core_code_sha256')!=r['expected_core19_sha256'] or p.get('producer_environment')!=c['producer_environment']:raise ValueError('Producer/completion/request mismatch')
 with (stage/'commit.claim').open('x') as f:f.write(r['request_sha256'])
 request(j);os.rename(tmp,stage/'artifacts.zip')
 cp=j/'completion.json.tmp'
 with cp.open('x') as f:json.dump(c,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.link(cp,j/'completion.json');cp.unlink();result={'committed':True,'completion_sha256':digest(j/'completion.json')}
elif op=='fail':
 j=job();request(j)
 if (j/'completion.json').exists():raise ValueError('Cannot fail a committed job')
 p=j/'FAILED.json';temp=j/'FAILED.json.tmp'
 with temp.open('x') as f:json.dump(a['failure'],f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.link(temp,p);temp.unlink();result={'failed_recorded':True}
else:raise ValueError('Unknown metadata operation')
print(json.dumps(result,allow_nan=False))
'''

class NativeTransport:
    def __init__(self,host,port,base,run):
        if not re.fullmatch(r'root@[A-Za-z0-9.-]+',host) or not 1<=port<=65535:raise ValueError('Explicit safe SSH host/port required')
        self.host=host;self.port=port;self.base=base;self.run=run
        self.options=['-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+KNOWN_HOSTS]
    def rpc(self,action,**payload):
        data={'action':action,'base':self.base,'run':self.run,**payload}
        command=shlex.join(['python3','-c',REMOTE_PROGRAM])
        r=subprocess.run(['ssh','-p',str(self.port),*self.options,self.host,command],input=json.dumps(data).encode(),capture_output=True,check=False,timeout=60)
        if r.returncode:raise RuntimeError('SSH metadata operation failed: '+r.stderr.decode(errors='replace')[-3000:])
        return strict_json(r.stdout)
    def jobs(self):return self.rpc('list')['jobs']
    def read(self,path):
        r=self.rpc('read',path=path);b=base64.b64decode(r['data'],validate=True)
        if hashlib.sha256(b).hexdigest()!=r['sha256']:raise ValueError('Transport metadata SHA differs')
        return b
    def codes(self,names):return self.rpc('code',names=sorted(names))['hashes']
    def publish(self,fold,request,file_sha,archive,completion):
        nonce=uuid.uuid4().hex
        p=self.rpc('prepare',fold=fold,request_sha256=request['request_sha256'],request_file_sha256=file_sha,nonce=nonce)['temporary_path']
        r=subprocess.run(['scp','-P',str(self.port),*self.options,str(archive),self.host+':'+p],capture_output=True,check=False,timeout=300)
        if r.returncode:raise RuntimeError('SCP tree staging failed: '+r.stderr.decode(errors='replace')[-3000:])
        return self.rpc('commit',fold=fold,request_sha256=request['request_sha256'],request_file_sha256=file_sha,temporary_path=p,completion=completion)
    def fail(self,fold,request,file_sha,failure):
        return self.rpc('fail',fold=fold,request_sha256=request['request_sha256'],request_file_sha256=file_sha,failure=failure)

class NativeProducer:
    def __init__(self,project):self.project=Path(project)
    def _call(self,action,inbox,output,log):
        args=[sys.executable,str(self.project/'ops/mac_lgbm_r5.py'),action,'--context',str(inbox/'context.json'),
            '--selection',str(inbox/'preprocessing_selection.json'),'--output',str(output),
            '--receipt',str(output/'producer_summary.json'),'--expected-config-sha',CONFIG_SHA]
        env=os.environ.copy();env.update({k:'4' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
        env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(self.project)+os.pathsep+env.get('PYTHONPATH','')
        with Path(log).open('x') as handle:
            process=subprocess.Popen(args,cwd=self.project,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait()
            except BaseException:
                os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=15);raise
        if code:raise RuntimeError('Native producer '+action+' failed; exit='+str(code)+'; log='+str(log))
    def fit(self,inbox,output,job):self._call('fit',inbox,output,job/'fit.log')
    def verify(self,inbox,output,job):self._call('verify',inbox,output,job/'verify.log')

class Worker:
    def __init__(self,project,base,run,plan,transport,producer,*,fixture=False):
        self.project=Path(project).resolve();self.base=base;self.run=run;self.transport=transport;self.producer=producer;self.fixture=fixture
        if not re.fullmatch(r'/workspace/[A-Za-z0-9_.-]+',base) or not re.fullmatch(r'[A-Za-z0-9_-]+',run):raise ValueError('Unsafe base/run path')
        self.plan_path=Path(plan);self.plan=strict_json(self.plan_path.read_bytes());self.plan_sha=sha(self.plan_path)
        p=self.plan
        if p.get('format')!='r5_mac_executor_plan_v1' or p.get('tree_executor')!='mac_spool_v1' or p.get('user_approved') is not True or p.get('server_tree_fit_allowed') is not False or p.get('config_sha256')!=CONFIG_SHA:raise ValueError('Pinned approved executor plan required')
        codes=p.get('code_sha256',{})
        if not CORE19|OPS<=set(codes):raise ValueError('Plan must pin core19/controller/producer/worker')
        if not fixture and codes['ops/mac_lgbm_r5.py']!=PRODUCER_SHA:raise ValueError('Mac producer release SHA differs')
        self.pins=dict(codes);self.check_local()
        self.state_path=self.project/'ops/mac_cpu_worker_state.json'
        self.state=strict_json(self.state_path.read_bytes()) if self.state_path.exists() else {'format':FORMAT,'jobs':{}}
        if self.state.get('format')!=FORMAT:raise ValueError('Unknown worker state')
        if any(x.get('status') not in {'completed'} for x in self.state['jobs'].values()):raise ValueError('Previous failed/partial job requires manual review; no restart')
    def check_local(self):
        if sha(self.plan_path)!=self.plan_sha:raise ValueError('Executor plan changed')
        for n,h in self.pins.items():
            if sha(safe_file(self.project,n))!=h:raise ValueError('Local frozen source changed: '+n)
    def save_state(self):atomic(self.state_path,self.state)
    @contextlib.contextmanager
    def lock(self):
        p=self.project/'ops/mac_r5_worker.lock';p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('a+') as f:
            try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Another Mac worker is active; duplicate watcher forbidden')
            f.seek(0);f.truncate();f.write(json.dumps({'pid':os.getpid(),'run':self.run,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}));f.flush();os.fsync(f.fileno())
            try:yield
            finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)
    def validate_request(self,r,fold):
        if set(r)!=REQUEST_KEYS:raise ValueError('Only the fixed metadata request schema is allowed; no data regeneration flags')
        if r.get('format')!='r5_mac_tree_request_v1' or r.get('job_id')!=self.run+f'.fold{fold:02d}' or r.get('config_sha256')!=CONFIG_SHA or r.get('request_sha256')!=canon({k:v for k,v in r.items() if k!='request_sha256'}):raise ValueError('Immutable request format/job/config/SHA differs')
        if r.get('executor_plan_sha256')!=self.plan_sha:raise ValueError('Request executor plan differs')
        if r.get('expected_core19_sha256')!={n:self.pins[n] for n in CORE19}:raise ValueError('Request core19 release differs')
        b=r.get('training_binding',{})
        if set(b)!={'fold','seed','train_examples','source_hash','train_npz_sha256','data_package_hash'}:raise ValueError('Exact TRAIN binding fields required')
        if type(b.get('fold')) is not int or b['fold']!=fold or b.get('seed')!=42 or type(b.get('train_examples')) is not int or b['train_examples']<1:raise ValueError('Request fold/seed/TRAIN rows invalid')
        for name in ['source_hash','train_npz_sha256','data_package_hash']:
            if not re.fullmatch('[0-9a-f]{64}',str(b.get(name,''))):raise ValueError('Request data binding SHA invalid')
        remote_fold=PurePosixPath(self.base)/'project/models'/self.run/f'fold_{fold:02d}'
        if r.get('context_source_path')!=str(remote_fold/'context.json') or r.get('selection_source_path')!=str(remote_fold/'preprocessing_selection.json') or r.get('output_model_path')!=str(remote_fold/'models/lgbm_physics'):raise ValueError('Unexpected remote context/selection/model path')
        expected=[f'preprocessing_pilot/{n}/bg_cnn/summary.json' for n in ['raw','resolution_01','resolution_01_endpoint']]
        if [x.get('relative_path') for x in r.get('cnn_summaries',[])]!=expected:raise ValueError('Exactly original three CNN summaries required')
        if any(set(x)!={'relative_path','sha256'} or not re.fullmatch('[0-9a-f]{64}',str(x['sha256'])) for x in r['cnn_summaries']):raise ValueError('Exact CNN summary path/SHA metadata required')
        for k in ['context_sha256','selection_sha256','prior_sha256']:
            if not re.fullmatch('[0-9a-f]{64}',str(r.get(k,''))):raise ValueError('Request metadata SHA invalid')
        return remote_fold
    def fetch(self,r,fold,inbox):
        remote_fold=self.validate_request(r,fold)
        remote_plan=self.transport.read(str(PurePosixPath(self.base)/'project/models'/self.run/'executor_plan.json'))
        if strict_json(remote_plan)!=self.plan:raise ValueError('Remote executor plan semantic identity differs')
        if self.transport.codes(CORE19|{'ops/runner_r5_mac.py','ops/mac_lgbm_r5.py'})!={n:self.pins[n] for n in CORE19|{'ops/runner_r5_mac.py','ops/mac_lgbm_r5.py'}}:raise ValueError('Remote current code release differs')
        inbox.mkdir(parents=True,exist_ok=False)
        files={'context.json':r['context_sha256'],'preprocessing_selection.json':r['selection_sha256'],
            'shared_background_prior.json':r['prior_sha256'],**{x['relative_path']:x['sha256'] for x in r['cnn_summaries']}}
        for name,h in files.items():
            raw=self.transport.read(str(remote_fold/name))
            if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Fetched immutable metadata SHA differs: '+name)
            p=inbox/name;p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as handle:handle.write(raw)
        c=strict_json((inbox/'context.json').read_bytes());cfg=c['config'];s=strict_json((inbox/'preprocessing_selection.json').read_bytes())
        if canon(cfg)!=CONFIG_SHA or c.get('config_sha256')!=CONFIG_SHA or c.get('training_binding')!=r['training_binding'] or c.get('code_sha256')!={n:self.pins[n] for n in CORE17}:raise ValueError('Fetched context config/TRAIN/core17 differs')
        if c.get('data_fold')!=str(PurePosixPath(cfg['dataset'])/f'fold_{fold:02d}') or c.get('prior',{}).get('filename')!='shared_background_prior.json' or c['prior']['file_sha256']!=r['prior_sha256']:raise ValueError('Portable source/prior binding differs')
        if cfg.get('tree')!=r.get('tree_settings') or cfg['observation_candidates'].get(r.get('observation_name'))!=r.get('observation_policy'):raise ValueError('Frozen tree/observation settings differ')
        if s.get('selected')!=r['observation_name'] or s.get('training_binding')!=r['training_binding'] or s.get('fit_split')!='inner' or s.get('outer_or_Test_used') is not False:raise ValueError('CNN INNER selection binding differs')
        # Native producer independently verifies all candidate scores and summaries.
        write_new(inbox/'worker_source_copy_manifest.json',{'request_sha256':r['request_sha256'],'file_sha256':files,
            'remote_plan_copy_sha256':hashlib.sha256(remote_plan).hexdigest(),'executor_plan_identity_sha256':self.plan_sha})
    def output_check(self,r,output):
        files={n:sha(safe_file(output,n)) for n in TREE_FILES}
        p=strict_json((output/'producer_summary.json').read_bytes());s=strict_json((output/'summary.json').read_bytes())
        if p.get('format')!='mac_lgbm_r5_producer_v1' or p.get('status')!='completed' or p.get('fixture_only') is not self.fixture or p.get('receipt_payload_sha256')!=canon({k:v for k,v in p.items() if k!='receipt_payload_sha256'}):raise ValueError('Completed producer payload required')
        for k,expected in [('config_sha256',CONFIG_SHA),('context_sha256',r['context_sha256']),('selection_sha256',r['selection_sha256']),('training_binding',r['training_binding']),('selected_observation',r['observation_name']),('observation_policy',r['observation_policy']),('core_code_sha256',r['expected_core19_sha256']),('producer_script_sha256',self.pins['ops/mac_lgbm_r5.py'])]:
            if p.get(k)!=expected:raise ValueError('Producer/request differs: '+k)
        if p.get('output_summary_sha256')!=files['summary.json'] or p.get('artifact_sha256')!={n:files[n] for n in CORE_TREE_FILES} or s.get('artifact_sha256')!=p['artifact_sha256']:raise ValueError('Producer tree artifact declaration differs')
        if not self.fixture and (p.get('producer_environment',{}).get('system')!='Darwin' or p['producer_environment']['packages'].get('lightgbm')!='4.7.0'):raise ValueError('Native Mac LightGBM4.7 producer required')
        if p.get('data_and_core19_SHA_unchanged_after_fit') is not True or p.get('core_or_original_R5_run_modified') is not False or p.get('remote_transfer_SSH_GPU_OUTER_Test_or_ensemble_selection_performed') is not False or p.get('partial_resume_allowed') is not False or set(p.get('trained_array_roles',[]))!={'train','inner'}:raise ValueError('Producer data/scope/immutability declaration differs')
        return p,files
    def once(self):
        self.check_local();jobs=self.transport.jobs()
        for j in jobs:
            if j['request'] and j['failed']:raise RuntimeError('Remote failed job requires manual review')
        ready=[j for j in jobs if j['request'] and not j['completed']]
        if not ready:return False
        fold=min(j['fold'] for j in ready)
        if type(fold) is not int or fold not in range(5):raise ValueError('Invalid remote fold')
        path=str(PurePosixPath(self.base)/'ops/mac_cpu_jobs'/self.run/f'fold_{fold:02d}/request.json')
        raw=self.transport.read(path);r=strict_json(raw);self.validate_request(r,fold);request_file_sha=hashlib.sha256(raw).hexdigest();job_id=r['job_id']
        if job_id in self.state['jobs']:raise ValueError('Job already claimed; no duplicate/partial restart')
        job=self.project/'ops/mac_cpu_worker_jobs'/self.run/f'fold_{fold:02d}'
        output=self.project/'models/group_cv_r5_mac_lgbm_producers_r1'/self.run/f'fold_{fold:02d}'
        if job.exists() or output.exists() or output.with_name(output.name+'.claim.json').exists():raise FileExistsError('Local job/output already exists; no partial reuse')
        job.mkdir(parents=True)
        self.state['jobs'][job_id]={'status':'claimed','fold':fold,'request_sha256':r['request_sha256'],'request_file_sha256':request_file_sha,'job_path':str(job),'producer_output':str(output)};self.save_state()
        write_new(job/'claim.json',self.state['jobs'][job_id]);write_new(job/'request.json',r)
        try:
            inbox=job/'inbox';self.fetch(r,fold,inbox);self.check_local()
            if self.transport.read(path)!=raw:raise ValueError('Request changed before fit')
            self.state['jobs'][job_id]['status']='fitting';self.save_state()
            self.producer.fit(inbox,output,job);self.producer.verify(inbox,output,job)
            producer,files=self.output_check(r,output);self.check_local()
            archive=job/'artifacts.zip'
            with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
                for name in sorted(TREE_FILES):z.write(safe_file(output,name),arcname=name)
            archive_check(archive,files)
            completion={'format':'r5_mac_tree_completion_v1','job_id':job_id,'request_sha256':r['request_sha256'],
                'executor_plan_sha256':self.plan_sha,'producer_receipt_sha256':files['producer_summary.json'],
                'archive_sha256':sha(archive),'artifact_sha256':files,'producer_environment':producer['producer_environment']}
            write_new(job/'completion.json',completion)
            if self.transport.read(path)!=raw:raise ValueError('Request changed before commit')
            committed=self.transport.publish(fold,r,request_file_sha,archive,completion)
            if committed.get('committed') is not True:raise RuntimeError('Remote completion not committed')
            self.state['jobs'][job_id].update({'status':'completed','completion_sha256':committed.get('completion_sha256'),
                'archive_sha256':completion['archive_sha256'],'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});self.save_state()
            return True
        except BaseException as e:
            failure={'format':FORMAT,'status':'failed','job_id':job_id,'request_sha256':r['request_sha256'],
                'error_type':type(e).__name__,'error':str(e),'partial_resume_allowed':False,'automatic_retry':False}
            write_new(job/'FAILED.json',failure);self.state['jobs'][job_id].update({'status':'failed','error':str(e)});self.save_state()
            try:self.transport.fail(fold,r,request_file_sha,failure)
            except Exception as publish_error:atomic(job/'failure_publish_status.json',{'error':str(publish_error),'automatic_retry':False})
            raise

def self_test():
    """Filesystem-only mocked transport/producer; no actual ML object or SSH."""
    compile(REMOTE_PROGRAM,'remote_metadata_program','exec')
    with tempfile.TemporaryDirectory(prefix='r5_mac_worker_mock_') as temp:
        root=Path(temp);project=root/'project';remote=root/'remote';project.mkdir();remote.mkdir()
        cfg=strict_json((ROOT/'configs/r5_training_gpu_r1.json').read_bytes())
        assert canon(cfg)==CONFIG_SHA
        pins={}
        for n in CORE19|OPS:
            p=project/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# mock code identity only\n');pins[n]=sha(p)
        plan={'format':'r5_mac_executor_plan_v1','tree_executor':'mac_spool_v1','user_approved':True,'server_tree_fit_allowed':False,'config_sha256':CONFIG_SHA,'code_sha256':pins}
        planfile=root/'plan.json';write_new(planfile,plan);plan_sha=sha(planfile)
        base='/workspace/underwater_vessel_noise_r5_r1';run='group_cv_r5_mac_cpu_r1';folder=remote/'project/models'/run/'fold_00';folder.mkdir(parents=True)
        binding={'seed':42,'fold':0,'train_examples':64,'source_hash':'a'*64,'train_npz_sha256':'b'*64,'data_package_hash':'c'*64}
        prior={'mock':True};write_new(folder/'shared_background_prior.json',prior)
        context={'format':'r5_fold_context_v1','config':cfg,'config_sha256':CONFIG_SHA,'training_binding':binding,
            'data_fold':cfg['dataset']+'/fold_00','code_sha256':{n:pins[n] for n in CORE17},
            'prior':{'filename':'shared_background_prior.json','file_sha256':sha(folder/'shared_background_prior.json'),'payload_sha256':canon(prior)}}
        write_new(folder/'context.json',context);candidates=[]
        for i,name in enumerate(cfg['observation_candidates']):
            p=folder/'preprocessing_pilot'/name/'bg_cnn/summary.json';write_new(p,{'mock':True})
            candidates.append({'observation':name,'inner_rmse':.2+i*.01,'model_path':f'preprocessing_pilot/{name}/bg_cnn','summary_sha256':sha(p)})
        selection={'fit_split':'inner','outer_or_Test_used':False,'training_binding':binding,'selected':'raw','candidates':candidates};write_new(folder/'preprocessing_selection.json',selection)
        write_new(folder.parent/'executor_plan.json',plan)
        rf=PurePosixPath(base)/'project/models'/run/'fold_00'
        request={'format':'r5_mac_tree_request_v1','job_id':run+'.fold00','config_sha256':CONFIG_SHA,'training_binding':binding,
            'observation_name':'raw','observation_policy':cfg['observation_candidates']['raw'],'context_sha256':sha(folder/'context.json'),
            'selection_sha256':sha(folder/'preprocessing_selection.json'),'context_source_path':str(rf/'context.json'),
            'selection_source_path':str(rf/'preprocessing_selection.json'),'output_model_path':str(rf/'models/lgbm_physics'),
            'cnn_summaries':[{'relative_path':x['model_path']+'/summary.json','sha256':x['summary_sha256']} for x in candidates],
            'prior_sha256':sha(folder/'shared_background_prior.json'),'expected_core19_sha256':{n:pins[n] for n in CORE19},
            'tree_settings':cfg['tree'],'executor_plan_sha256':plan_sha};request['request_sha256']=canon(request)
        spool=remote/'ops/mac_cpu_jobs'/run/'fold_00';write_new(spool/'request.json',request)
        class MockTransport:
            def __init__(self):self.committed=False;self.ssh_calls=0
            def jobs(self):return [{'fold':0,'request':True,'completed':self.committed,'failed':False}]
            def read(self,p):return (remote/str(PurePosixPath(p).relative_to(base))).read_bytes()
            def codes(self,names):return {n:pins[n] for n in names}
            def publish(self,fold,r,rs,archive,c):
                assert sha(spool/'request.json')==rs and r['request_sha256']==request['request_sha256'];archive_check(archive,c['artifact_sha256'])
                (spool/'staging').mkdir();shutil.copyfile(archive,spool/'staging/artifacts.zip');write_new(spool/'completion.json',c);self.committed=True
                return {'committed':True,'completion_sha256':sha(spool/'completion.json')}
            def fail(self,*a):raise AssertionError('No failure expected in valid mock')
        class MockProducer:
            def __init__(self):self.fit_calls=0;self.verify_calls=0
            def fit(self,inbox,output,job):
                self.fit_calls+=1;output.mkdir(parents=True)
                for n in CORE_TREE_FILES:(output/n).write_bytes(b'mocked artifact; not an ML model\n')
                artifacts={n:sha(output/n) for n in CORE_TREE_FILES}
                write_new(output/'summary.json',{'artifact_sha256':artifacts})
                p={'format':'mac_lgbm_r5_producer_v1','status':'completed','fixture_only':True,'config_sha256':CONFIG_SHA,
                    'context_sha256':request['context_sha256'],'selection_sha256':request['selection_sha256'],'training_binding':binding,
                    'selected_observation':'raw','observation_policy':request['observation_policy'],'core_code_sha256':request['expected_core19_sha256'],
                    'producer_script_sha256':pins['ops/mac_lgbm_r5.py'],'output_summary_sha256':sha(output/'summary.json'),
                    'artifact_sha256':artifacts,'producer_environment':{'system':'mock','packages':{}},
                    'data_and_core19_SHA_unchanged_after_fit':True,'core_or_original_R5_run_modified':False,
                    'remote_transfer_SSH_GPU_OUTER_Test_or_ensemble_selection_performed':False,'partial_resume_allowed':False,'trained_array_roles':['train','inner']}
                p['receipt_payload_sha256']=canon(p);write_new(output/'producer_summary.json',p)
            def verify(self,*a):self.verify_calls+=1
        transport=MockTransport();producer=MockProducer();w=Worker(project,base,run,planfile,transport,producer,fixture=True)
        with w.lock():
            with (project/'ops/mac_r5_worker.lock').open('a+') as other:
                try:fcntl.flock(other.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:duplicate_guard=True
                else:raise AssertionError('Duplicate worker accepted')
            assert w.once() and not w.once()
        assert producer.fit_calls==producer.verify_calls==1 and transport.ssh_calls==0
        bad=dict(request,context_source_path='/etc/passwd');bad['request_sha256']=canon({k:v for k,v in bad.items() if k!='request_sha256'})
        try:w.validate_request(bad,0)
        except ValueError:unsafe_path=True
        else:raise AssertionError('Unsafe metadata path accepted')
        bad=dict(request,config_sha256='0'*64);bad['request_sha256']=canon({k:v for k,v in bad.items() if k!='request_sha256'})
        try:w.validate_request(bad,0)
        except ValueError:config_guard=True
        else:raise AssertionError('Changed config accepted')
        bad=dict(request,regenerate_data=True);bad['request_sha256']=canon({k:v for k,v in bad.items() if k!='request_sha256'})
        try:w.validate_request(bad,0)
        except ValueError:regeneration_guard=True
        else:raise AssertionError('Data regeneration flag accepted')
        archive=project/'ops/mac_cpu_worker_jobs'/run/'fold_00/artifacts.zip';files=strict_json((spool/'completion.json').read_bytes())['artifact_sha256']
        with zipfile.ZipFile(archive,'a') as z:z.writestr('../escape',b'x')
        try:archive_check(archive,files)
        except ValueError:zip_guard=True
        else:raise AssertionError('ZIP escape/extra entry accepted')
        w.state['jobs'][request['job_id']]['status']='failed';w.save_state()
        try:Worker(project,base,run,planfile,transport,producer,fixture=True)
        except ValueError:partial_guard=True
        else:raise AssertionError('Failed job auto-restart accepted')
        failed_project=root/'failed_project';failed_project.mkdir()
        for n in pins:
            p=failed_project/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(project/n,p)
        failed_remote=root/'failed_remote';shutil.copytree(remote,failed_remote)
        remote=failed_remote;spool=remote/'ops/mac_cpu_jobs'/run/'fold_00'
        (spool/'completion.json').unlink();shutil.rmtree(spool/'staging')
        class FailingTransport(MockTransport):
            def fail(self,fold,r,rs,failure):write_new(spool/'FAILED.json',failure);return {'failed_recorded':True}
        class FailingProducer(MockProducer):
            def fit(self,*a):raise RuntimeError('intentional mocked producer failure')
        bad_worker=Worker(failed_project,base,run,planfile,FailingTransport(),FailingProducer(),fixture=True)
        with bad_worker.lock():
            try:bad_worker.once()
            except RuntimeError as e:assert 'intentional mocked' in str(e)
            else:raise AssertionError('Producer failure did not stop worker')
        assert (spool/'FAILED.json').is_file()
        assert (failed_project/'ops/mac_cpu_worker_jobs'/run/'fold_00/FAILED.json').is_file()
        assert bad_worker.state['jobs'][request['job_id']]['status']=='failed'
        result={'status':'PASS','scope':'Mock APIs and temporary files only; not native producer fit or SSH',
            'mock_fit_calls':1,'mock_verify_calls':1,'SSH_GPU_real_TRAIN_INNER_OUTER_Test_opened':False,
            'native_training_executed':False,'seven_file_ZIP_CRC_SHA_completion_LAST':True,
            'duplicate_PID_guard':duplicate_guard,'unsafe_request_path_rejected':unsafe_path,'config_change_rejected':config_guard,
            'ZIP_extra_or_escape_rejected':zip_guard,'failed_partial_restart_rejected':partial_guard,
            'data_regeneration_request_rejected':regeneration_guard,'producer_failure_records_local_and_remote_FAILED_then_stops':True,
            'remote_program_syntax_compiles':True,'worker_sha256':sha(__file__)}
        print(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--once',action='store_true');mode.add_argument('--watch',action='store_true');mode.add_argument('--self-test',action='store_true')
    p.add_argument('--remote-host',default='root@209.170.80.132');p.add_argument('--port',type=int,default=13914)
    p.add_argument('--base',default='/workspace/underwater_vessel_noise_r5_r1');p.add_argument('--run-name',default='group_cv_r5_mac_cpu_r1')
    p.add_argument('--executor-plan',type=Path);a=p.parse_args()
    if a.self_test:self_test();return
    if a.executor_plan is None:p.error('--executor-plan is required for real execution')
    transport=NativeTransport(a.remote_host,a.port,a.base,a.run_name)
    worker=Worker(ROOT,a.base,a.run_name,a.executor_plan,transport,NativeProducer(ROOT))
    def terminate(signum,frame):raise KeyboardInterrupt('Worker termination requested; stop owned producer and record failure')
    signal.signal(signal.SIGTERM,terminate)
    with worker.lock():
        while True:
            worked=worker.once();print(json.dumps({'worker':FORMAT,'job_completed':worked,'watch':a.watch}),flush=True)
            if a.once:return
            time.sleep(60)

if __name__=='__main__':main()
