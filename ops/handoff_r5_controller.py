"""Prepare a model-boundary R5 controller handoff, never signal a GPU child.

inspect is read-only. Signal actions additionally require --execute-signals and
Linux pidfd support. Stop only the verified old launcher, let its direct neural
fit child finish, certify complete NN artifacts, then terminate that launcher
and its supervisor before authorizing a fresh Mac-only controller. No SIGKILL,
process-group signals, SSH, training, source/model mutation, or Pod control.

self-test uses fake process records/signals and synthetic JSON artifacts only.
It never examines or signals real processes, imports Torch, or starts training.
"""
from pathlib import Path
import argparse,ast,copy,datetime,hashlib,json,os,re,shutil,signal,sys,tempfile,time,zipfile

ROOT=Path(__file__).resolve().parents[1]
NEURAL={'bg_cnn','tfgrid_fdy_conformer','timemoe_public','dasheng','fdy','itransformer'}
EXTRA={'pipeline/train_lgbm.py':'704b77a00c89c636ada7881facb6deaf36afceaec35e7c4742b75478e650e42d',
       'pipeline/prototypes.py':'6349bd51e42b2c68986d44148644f63c9cfea5eae42a117dc86c5ca6da7b481a'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):
    def bad(x):raise ValueError('Nonfinite JSON constant: '+x)
    return json.loads(Path(path).read_text(),parse_constant=bad)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_new(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def safe(root,name):
    root=Path(root).resolve();p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Relative, nonescaping artifact path required')
    q=root
    for part in p.parts:
        q=q/part
        if q.is_symlink():raise ValueError('Symlink artifacts are forbidden')
    if not q.resolve().is_relative_to(root):raise ValueError('Escaping path')
    return q
def arg(argv,name):
    if argv.count(name)!=1 or argv.index(name)+1>=len(argv):raise ValueError('Exactly one '+name+' argument required')
    return argv[argv.index(name)+1]
def resolve(base,p):
    p=Path(p);return p.resolve() if p.is_absolute() else (Path(base)/p).resolve()
def names(path,name):
    tree=ast.parse(Path(path).read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
def dead(p):return p is None or p['state'] in {'Z','X','x'} and p.get('threads',1)<=1
def parse_stat(text):
    r=text.rsplit(')',1)[1].split()
    return {'state':r[0],'ppid':int(r[1]),'start_ticks':str(r[19]),
            'exit_code_raw':int(r[49]) if len(r)>49 else None}
def same(actual,expected,*,zombie=False):
    if actual is None or actual['pid']!=expected['pid'] or str(actual['start_ticks'])!=str(expected['start_ticks']):raise ValueError('Missing/reused process identity')
    if not (zombie and dead(actual)):
        for k in ['ppid','cmdline','cwd']:
            if actual.get(k)!=expected.get(k):raise ValueError('Foreign process identity: '+k)
    return actual

class LinuxProcesses:
    fixture=False
    def get(self,pid):
        p=Path('/proc')/str(int(pid))
        try:stat=parse_stat((p/'stat').read_text())
        except FileNotFoundError:return None
        status={}
        for line in (p/'status').read_text().splitlines():
            if ':' in line:
                k,v=line.split(':',1);status[k]=v.strip()
        try:cwd=os.readlink(p/'cwd')
        except FileNotFoundError:cwd=None
        return {'pid':int(pid),**stat,'threads':int(status.get('Threads','0')),
            'cmdline':[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v],
            'cwd':cwd,'signal_caught':int(status.get('SigCgt','0'),16),
            'signal_ignored':int(status.get('SigIgn','0'),16),
            'signal_pending':int(status.get('SigPnd','0'),16)|int(status.get('ShdPnd','0'),16)}
    def children(self,pid):
        found=set()
        for p in (Path('/proc')/str(pid)/'task').glob('*/children'):
            try:found.update(int(v) for v in p.read_text().split())
            except FileNotFoundError:pass
        out=[]
        for x in sorted(found):
            r=self.get(x)
            if r is not None:out.append(r)
        return out
    def all(self):
        out=[]
        for p in Path('/proc').glob('[0-9]*'):
            try:r=self.get(int(p.name))
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
            if r is not None:out.append(r)
        return out
    def send(self,identity,sig):
        if sys.platform!='linux' or not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
            raise RuntimeError('Signal actions require Linux pidfd; use inspect/prepare-only otherwise')
        fd=os.pidfd_open(identity['pid'],0)
        try:
            same(self.get(identity['pid']),identity)
            signal.pidfd_send_signal(fd,sig,None,0)
        finally:os.close(fd)
    def wait(self):time.sleep(.05)

class Handoff:
    def __init__(self,base,legacy,new,driver,supervisor,plan,*,processes=None,checkpoint_reader=None):
        self.base=Path(base).resolve();self.project=self.base/'project';self.proc=processes or LinuxProcesses()
        self.legacy=resolve(self.project/'models',legacy);self.new=resolve(self.project/'models',new)
        for p in [self.legacy,self.new]:
            if p.parent!=self.project/'models' or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',p.name):raise ValueError('Direct, named RUN under PROJECT/models required')
        if self.legacy==self.new:raise ValueError('New controller requires a different fresh RUN')
        self.driver=int(driver);self.supervisor=int(supervisor)
        if min(self.driver,self.supervisor)<2 or self.driver==self.supervisor:raise ValueError('Distinct positive controller/supervisor identities required')
        self.plan=Path(plan).resolve();self.ops=self.base/'ops'/self.legacy.name
        self.home=self.ops/('handoff_to_'+self.new.name)
        self.checkpoint_reader=checkpoint_reader or self._checkpoint
    @staticmethod
    def _checkpoint(path):
        import torch
        return torch.load(path,map_location='cpu',weights_only=True)
    def environment(self):
        startup=read(self.ops/'startup_receipt.json');state=read(self.ops/'training_state.json')
        cfg=read(startup['config_path']);plan=read(self.plan)
        if canonical(cfg)!=startup['config_sha256'] or sha(startup['config_path'])!=startup['config_file_sha256']:raise ValueError('Original launch configuration changed')
        if (cfg['training']['epochs'],cfg['training']['patience'],cfg['training']['optimizer'],cfg['seed'])!=(100,7,'adam',42):raise ValueError('Original training contract differs')
        if plan.get('format')!='r5_mac_executor_plan_v1' or plan.get('tree_executor')!='mac_spool_v1' or plan.get('user_approved') is not True or plan.get('server_tree_fit_allowed') is not False or plan.get('config_sha256')!=canonical(cfg):raise ValueError('Explicit matching Mac-only executor plan required')
        if not {'ops/runner_r5_mac.py','ops/mac_lgbm_r5.py'}.issubset(plan.get('code_sha256',{})):raise ValueError('New controller/producer source pins required')
        for p,h in plan.get('code_sha256',{}).items():
            if sha(safe(self.project,p))!=h:raise ValueError('Executor plan code changed: '+p)
        required=names(self.project/'pipeline/runner_r5.py','CODE_NAMES')
        if len(required)!=17 or set(startup['source_code_sha256'])!=set(required):raise ValueError('Exact original17 source closure required')
        code=dict(startup['source_code_sha256']);code.update(EXTRA)
        if len(code)!=19:raise ValueError('Original19 closure required')
        for p,h in code.items():
            if sha(safe(self.project,p))!=h:raise ValueError('Frozen source19 changed: '+p)
        if sha(self.project/'ops/runpod_train_r5.py')!=startup['supervisor_sha256']:raise ValueError('Original supervisor code changed')
        if state.get('pid')!=self.driver or state.get('supervisor_pid')!=self.supervisor:raise ValueError('PID arguments differ from original launch state')
        if state.get('status')!='running':raise ValueError('Original controller launch is not in its recorded running state')
        if resolve(self.project,arg(startup['command'],'--output'))!=self.legacy:raise ValueError('Original RUN differs from startup command')
        return {'startup':startup,'state':state,'config':cfg,'config_sha256':canonical(cfg),'code_sha256':code,
            'executor_plan_sha256':sha(self.plan),'startup_sha256':sha(self.ops/'startup_receipt.json')}
    def fresh(self):
        if self.new.exists():raise ValueError('New RUN already exists; no in-place handoff')
        for p in self.proc.all():
            if dead(p):continue
            argv=p['cmdline']
            if '--output' in argv and resolve(p.get('cwd') or self.project,arg(argv,'--output'))==self.new:raise ValueError('New controller already started')
    def context(self,fold,env):
        folder=safe(self.legacy,fold);p=folder/'context.json';c=read(p)
        if c.get('format')!='r5_fold_context_v1' or c['config']!=env['config'] or c['config_sha256']!=env['config_sha256'] or c['code_sha256']!={k:v for k,v in env['code_sha256'].items() if k not in EXTRA}:raise ValueError('Legacy context/config/source differs')
        prior=c['prior'];q=safe(folder,prior['filename']);obj=read(q)
        if sha(q)!=prior['file_sha256'] or obj.get('fit_split')!='train' or obj.get('training_binding')!=c['training_binding'] or obj.get('shared_prior_payload_sha256')!=prior['payload_sha256'] or canonical({k:v for k,v in obj.items() if k!='shared_prior_payload_sha256'})!=prior['payload_sha256']:raise ValueError('TRAIN prior provenance changed')
        return c,{'context_sha256':sha(p),'prior_file_sha256':sha(q),'prior_payload_sha256':prior['payload_sha256']}
    def selection(self,fold,env):
        p=safe(self.legacy,fold)/'preprocessing_selection.json';s=read(p);c,_=self.context(fold,env)
        if s.get('fit_split')!='inner' or s.get('outer_or_Test_used') is not False or s.get('training_binding')!=c['training_binding']:raise ValueError('Legacy CNN selection provenance differs')
        candidates=s['candidates']
        if [v['observation'] for v in candidates]!=list(env['config']['observation_candidates']):raise ValueError('Original preprocessing candidate order differs')
        for v in candidates:
            p=safe(self.legacy,fold)/v['model_path']/'summary.json';m=read(p)
            if sha(p)!=v['summary_sha256'] or m['best_inner_rmse']!=v['inner_rmse']:raise ValueError('CNN selection source score/SHA differs')
        if s['selected']!=min(candidates,key=lambda v:v['inner_rmse'])['observation']:raise ValueError('Legacy CNN winner differs')
        return s['selected']
    def ordered_progress(self,child,env):
        argv=child['cmdline']
        if 'pipeline.neural_r5' not in argv or 'fit' not in argv or 'pipeline.lgbm_r5' in argv:raise ValueError('Only a direct neural fit child may finish at this boundary')
        if argv[argv.index('pipeline.neural_r5')-1]!='-m' or argv[argv.index('pipeline.neural_r5')+1]!='fit':raise ValueError('Unexpected active neural command')
        output=resolve(child['cwd'],arg(argv,'--output'))
        if not output.is_relative_to(self.legacy):raise ValueError('Child output outside legacy RUN')
        relative=output.relative_to(self.legacy).as_posix();parts=Path(relative).parts;fold=parts[0]
        if not re.fullmatch(r'fold_0[0-4]',fold):raise ValueError('Unexpected active fold')
        if resolve(child['cwd'],arg(argv,'--context'))!=self.legacy/fold/'context.json':raise ValueError('Child context differs from output fold')
        current_kind=arg(argv,'--kind');observation=arg(argv,'--observation')
        requested=env['startup']['requested_folds'];number=int(fold[-2:])
        if number not in requested:raise ValueError('Active fold outside original request')
        ordered=[]
        for n in requested:
            f=f'fold_{n:02d}'
            for obs in env['config']['observation_candidates']:
                ordered.append((f'preprocessing_pilot/{obs}/bg_cnn','bg_cnn',obs,f))
            for kind in env['config']['core_models']+env['config']['comparisons']:
                if kind!='bg_cnn':ordered.append((f'models/{kind}',kind,None,f))
            if n==number:break
            done=read(self.legacy/f/'summary.json')
            if done.get('status')!='completed':raise ValueError('Prior fold is not completed')
        targets=[f'{f}/{r}' for r,k,o,f in ordered]
        if relative not in targets:raise ValueError('Active child is not an original sequential call')
        index=targets.index(relative);r,kind,obs,f=ordered[index]
        if kind not in NEURAL or kind!=current_kind:raise ValueError('Active child kind differs')
        if obs is None:obs=self.selection(f,env)
        if observation!=obs:raise ValueError('Active observation differs from canonical call')
        expected=[argv[0],'-m','pipeline.neural_r5','fit','--context',str(self.legacy/f/'context.json')]
        if len(parts)==4:expected+=['--kind',kind,'--observation',obs,'--output',str(output)]
        else:expected+=['--observation',obs,'--output',str(output),'--kind',kind]
        if argv!=expected:raise ValueError('Active NN argv differs from the exact immutable controller call')
        prior=[]
        for r,k,o,f in ordered[:index]:
            if k not in NEURAL:continue
            c,identity=self.context(f,env);p=safe(self.legacy,f+'/'+r)/'summary.json';m=read(p)
            if m.get('status')!='completed' or m.get('fixture_only') is not False or m.get('training_binding')!=c['training_binding']:raise ValueError('Preceding neural summary not complete')
            prior.append({'path':f+'/'+r,'summary_sha256':sha(p),**identity,
                'exit_code':0,'exit0_proof_kind':'sequential_parent_subprocess_run_check_true_progress'})
        return {'current_model_path':relative,'current_kind':kind,'current_observation':obs,
            'preceding_completed_neural_calls':prior,'ordered_original_call_prefix':targets[:index+1],
            'controller_source_sha256':env['code_sha256']['pipeline/runner_r5.py'],
            'proof':'The same verified original controller PID/start-ticks reached this direct child only after each preceding subprocess.run(check=True), summary read and code check returned successfully.'}
    def inspect(self):
        env=self.environment();self.fresh();d=self.proc.get(self.driver);s=self.proc.get(self.supervisor)
        if d is None or s is None or dead(d) or dead(s) or s['state'] in {'T','t'}:raise ValueError('Original live, independently running supervisor required')
        if d['cmdline']!=env['startup']['command'] or d['cwd']!=str(self.project) or d['ppid']!=self.supervisor or str(d['start_ticks'])!=str(env['state']['process_start_ticks']):raise ValueError('Foreign/reused legacy driver')
        supervisor_script=any(Path(v).name=='runpod_train_r5.py' and resolve(s['cwd'],v)==self.project/'ops/runpod_train_r5.py' for v in s['cmdline']) or 'ops.runpod_train_r5' in s['cmdline']
        if '--run-name' not in s['cmdline'] or arg(s['cmdline'],'--run-name')!=self.legacy.name or resolve(self.base,arg(s['cmdline'],'--base'))!=self.base or resolve(self.base,arg(s['cmdline'],'--config'))!=Path(env['startup']['config_path']).resolve() or not supervisor_script:raise ValueError('Foreign original supervisor')
        kids=self.proc.children(self.driver)
        if len(kids)!=1:raise ValueError('Exactly one direct active NN child required')
        child=kids[0]
        if child['ppid']!=self.driver or dead(child) or child['state'] in {'T','t'} or child['cwd']!=str(self.project):raise ValueError('Child is not an independently running direct NN fit')
        if not int(s['start_ticks'])<int(d['start_ticks'])<=int(child['start_ticks']) or child['cmdline'][0]!=d['cmdline'][0]:raise ValueError('Supervisor/driver/child ancestry or executable differs')
        if list(self.legacy.rglob('FAILED.json')):raise ValueError('Failed legacy artifacts cannot be handed off')
        progress=self.ordered_progress(child,env)
        return {'format':'r5_controller_handoff_inspection_v1','created_at_utc':stamp(),
            'base':str(self.base),'legacy_run_path':str(self.legacy),'new_run_path':str(self.new),
            'new_run_source':None,'config_sha256':env['config_sha256'],'executor_plan_sha256':env['executor_plan_sha256'],
            'startup_sha256':env['startup_sha256'],'source19_sha256':env['code_sha256'],
            'legacy_driver':d,'legacy_supervisor':s,'current_child':child,
            'parent_progress_receipt':{**progress,'controller_identity':d,'active_child_identity':child},
            'signals_sent':False,'newcontroller_start_allowed':False}
    def _prepared(self,allow_intent=False):
        p=self.home/'prepare.json'
        if allow_intent and not p.exists():p=self.home/'stop_intent.json'
        r=read(p);env=self.environment();self.fresh()
        if 'prepare_payload_sha256' in r and canonical({k:v for k,v in r.items() if k!='prepare_payload_sha256'})!=r['prepare_payload_sha256']:raise ValueError('Immutable prepare receipt changed')
        for k,v in [('legacy_run_path',str(self.legacy)),('new_run_path',str(self.new)),('config_sha256',env['config_sha256']),('executor_plan_sha256',env['executor_plan_sha256']),('source19_sha256',env['code_sha256']),('startup_sha256',env['startup_sha256'])]:
            if r.get(k)!=v:raise ValueError('Prepared handoff provenance changed: '+k)
        if r['legacy_driver']['pid']!=self.driver or r['legacy_supervisor']['pid']!=self.supervisor:raise ValueError('Prepared PID arguments differ')
        d=same(self.proc.get(self.driver),r['legacy_driver']);s=same(self.proc.get(self.supervisor),r['legacy_supervisor'])
        if d['state'] not in {'T','t'} or dead(s) or s['state'] in {'T','t'}:raise ValueError('Old launcher must remain stopped and supervisor running')
        if any((self.home/n).exists() for n in ['finalized.json','termination_intent.json','rollback.json']):raise ValueError('Finalization/rollback already started; no replay')
        return r,env
    def stop_launcher(self,execute=False):
        r=self.inspect()
        if not execute:return {'status':'prepare-only; add --execute-signals for verified old-launcher SIGSTOP','inspection':r,'signals_sent':False}
        if r['legacy_driver']['state'] in {'T','t'}:raise ValueError('Driver already stopped without this receipt')
        write_new(self.home/'stop_intent.json',r)
        same(self.proc.get(self.driver),r['legacy_driver']);same(self.proc.get(r['current_child']['pid']),r['current_child'])
        self.proc.send(r['legacy_driver'],signal.SIGSTOP)
        for _ in range(40):
            d=same(self.proc.get(self.driver),r['legacy_driver'])
            if d['state'] in {'T','t'}:break
            self.proc.wait()
        else:raise RuntimeError('Launcher stop not confirmed; use inspect before any recovery')
        child=same(self.proc.get(r['current_child']['pid']),r['current_child'],zombie=True)
        if child['state'] in {'T','t'} or dead(child) and child.get('exit_code_raw')!=0:raise ValueError('NN child stopped/failed after launcher stop; do not finalize')
        r.update(format='r5_controller_handoff_prepare_v1',status='launcher_stopped_child_unmodified',
            prepare_created_at_utc=stamp(),signals_sent=True,old_driver_stopped_or_dead=True,
            rollback_action='rollback --execute-signals only before termination_intent/finalized and while new RUN/controller is absent')
        r['prepare_payload_sha256']=canonical(r)
        write_new(self.home/'prepare.json',r);return r
    def validate_model(self,relative,env):
        model=safe(self.legacy,relative);parts=Path(relative).parts;fold=parts[0];c,prior=self.context(fold,env)
        if (model/'FAILED.json').exists():raise ValueError('Failed NN model')
        m=read(model/'summary.json');kind=m.get('kind')
        if m.get('status')!='completed' or m.get('fixture_only') is not False or m.get('outer_or_Test_accessed') is not False or kind not in NEURAL or m.get('training_binding')!=c['training_binding'] or m.get('shared_prior_sha256')!=c['prior']['payload_sha256']:raise ValueError('Production completed NN summary required')
        obs=parts[2] if len(parts)==4 and parts[1]=='preprocessing_pilot' else self.selection(fold,env)
        if m['observation_policy']!=env['config']['observation_candidates'][obs]:raise ValueError('Completed NN observation differs')
        expected=dict(env['config']['training'])
        if kind in {'timemoe_public','dasheng'}:expected.update(env['config']['foundation_training_overrides'])
        if m['settings']!=expected:raise ValueError('NN training contract differs')
        if sha(model/'best.pt')!=m['checkpoint_sha256'] or sha(model/'inner_predictions.npz')!=m['prediction_sha256']:raise ValueError('Completed NN checkpoint/prediction SHA mismatch')
        with zipfile.ZipFile(model/'inner_predictions.npz') as z:
            if z.testzip() is not None:raise ValueError('Inner prediction ZIP CRC failure')
        history=[json.loads(v,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite epoch'))) for v in (model/'epochs.jsonl').read_text().splitlines()]
        count=m['epochs_run'];best=m['best_epoch']
        if not 1<=best<=count<=100 or [v['epoch'] for v in history]!=list(range(1,count+1)):raise ValueError('Incomplete NN epoch history')
        if history[-1]['best_epoch']!=best or history[-1]['best_inner_rmse']!=m['best_inner_rmse']:raise ValueError('NN best epoch/metric differs')
        dr=safe(model,m['dynamic_source_receipts_file']);dynamic=read(dr);identity=m['dynamic_source_identity'];binding=c['training_binding']
        if not isinstance(identity,dict) or identity.get('binding')!=binding or dynamic['source_identity']!=identity:raise ValueError('NN dynamic source binding differs')
        gate_path=resolve(self.project,c['data_fold'])/'readiness_manifest.json';gate=read(gate_path)
        if sha(gate_path)!=identity['readiness_sha256'] or identity['source_version_manifest_sha256']!=gate['source_version_manifest_sha256'] or identity['training_order_sha256']!=gate['file_sha256']['training_background_order.npy'] or identity['reference_slice_sha256']!=gate['partitions']['train']['reference_segment_slice_sha256']:raise ValueError('NN dynamic source identity differs from frozen TRAIN gate')
        if gate.get('ready_to_train') is not True or gate.get('source_hash')!=binding['source_hash'] or gate.get('data_package_hash')!=binding['data_package_hash'] or gate['file_sha256'].get('train.npz')!=binding['train_npz_sha256']:raise ValueError('NN TRAIN gate/package identity differs')
        source_path=resolve(self.project,gate['source_version_dir'])/'manifest.json';source=read(source_path)
        if sha(source_path)!=gate['source_version_manifest_sha256'] or source['source_hash']!=binding['source_hash']:raise ValueError('NN TRAIN source manifest changed')
        expected_banks=source.get('numeric_bank_sha256')
        if expected_banks is None:
            parent=self.project/f'data/synthetic/group_cv_physics_v2_r1/fold_{binding["fold"]:02d}/sources'
            old=read(parent/'manifest.json')
            if sha(parent/'background_manifest.parquet')!=source['parent_background_manifest_sha256']:raise ValueError('Parent background metadata identity differs')
            rv=self.project/'data/versions/vtuad_reference_r4_v1/manifest.json'
            if sha(rv)!=source['reference_version_manifest_sha256']:raise ValueError('VT source-version identity differs')
            expected_banks={f'data/processed/{n}':old['frozen_v1_hashes'][n] for n in ['sa_windows_db.npy','external_segments_power.npy']}
            if expected_banks['data/processed/external_segments_power.npy']!=read(rv)['bank_sha256_from_frozen_R2_source_manifest']:raise ValueError('VT reference bank hash supplier differs')
        if identity.get('numeric_bank_sha256')!=expected_banks:raise ValueError('NN dynamic numerical-bank identity differs')
        if set(dynamic['completed_epochs'])!={str(i) for i in range(1,count+1)}:raise ValueError('Partial/missing completed epoch receipts')
        for row in history:
            r=dynamic['completed_epochs'][str(row['epoch'])]
            if r!=row['dynamic_epoch_source_receipt'] or r.get('clean_primary_once') is not True or r.get('examples')!=binding['train_examples'] or row['train_examples']!=binding['train_examples'] or r.get('source_identity_sha256')!=canonical(identity):raise ValueError('Incomplete TRAIN source coverage')
        cp=self.checkpoint_reader(model/'best.pt')
        ck_names={'pipeline/'+n for n in names(self.project/'pipeline/neural_r5.py','CHECKPOINT_CODE_NAMES')}
        if cp.get('format')!='r5_neural_checkpoint_v1' or cp.get('kind')!=kind or cp.get('training_binding')!=binding or cp.get('shared_prior_sha256')!=c['prior']['payload_sha256'] or cp.get('observation_policy')!=m['observation_policy'] or cp.get('training_settings')!=m['settings'] or cp.get('epoch')!=best or cp.get('receipt')!=m['receipt']:raise ValueError('NN checkpoint metadata differs')
        if set(cp.get('model_code_sha256',{}))!=ck_names:raise ValueError('NN checkpoint preprocessing11 closure differs')
        for p,h in cp['model_code_sha256'].items():
            if sha(safe(self.project,p))!=h:raise ValueError('NN checkpoint source changed')
        if cp.get('dynamic_source_identity')!=identity or cp.get('best_epoch_source_receipt')!=dynamic['completed_epochs'][str(best)]:raise ValueError('Checkpoint best epoch coverage differs')
        if cp.get('weights',{}).get('adapter_only')!=(kind in {'timemoe_public','dasheng'}):raise ValueError('NN weight storage kind differs')
        return {'summary_sha256':sha(model/'summary.json'),'exit_code':0,'kind':kind,**prior,
            'artifact_sha256':{n:sha(model/n) for n in ['summary.json','best.pt','inner_predictions.npz','epochs.jsonl',m['dynamic_source_receipts_file']]}}
    def _default_term(self,p):
        term=1<<(signal.SIGTERM-1)
        if (p.get('signal_caught',0)|p.get('signal_ignored',0))&term:raise ValueError('SIGTERM handler/ignore prevents guaranteed boundary termination')
    def _wait_dead(self,expected):
        for _ in range(100):
            p=self.proc.get(expected['pid'])
            if p is None:return {'pid':expected['pid'],'state':'absent','start_ticks':expected['start_ticks']}
            same(p,expected,zombie=True)
            if dead(p):return p
            self.proc.wait()
        raise RuntimeError('Controller termination not confirmed; no new controller may start')
    def finalize(self,execute=False):
        r,env=self._prepared();child=same(self.proc.get(r['current_child']['pid']),r['current_child'],zombie=True)
        if not dead(child) or child['state'] not in {'Z','X','x'} or child.get('exit_code_raw')!=0:raise ValueError('Same captured NN child must have kernel wait-status0 and no live threads; live/unknown/nonzero is not success')
        progress=r['parent_progress_receipt'];models={}
        for old in progress['preceding_completed_neural_calls']:
            cert=self.validate_model(old['path'],env)
            if cert['summary_sha256']!=old['summary_sha256'] or cert['context_sha256']!=old['context_sha256'] or cert['prior_file_sha256']!=old['prior_file_sha256']:raise ValueError('Preceding NN parent-progress evidence changed')
            cert['exit0_proof_kind']=old['exit0_proof_kind'];models[old['path']]=cert
        current=progress['current_model_path'];cert=self.validate_model(current,env)
        cert['exit0_proof_kind']='captured_same_PID_startticks_zombie_kernel_wait_status0';models[current]=cert
        discovered=[]
        for p in self.legacy.glob('fold_*/**/summary.json'):
            if re.fullmatch(r'fold_0[0-4]',p.parent.name):continue
            m=read(p)
            if m.get('status')=='completed' and m.get('kind') in NEURAL:discovered.append(p.parent.relative_to(self.legacy).as_posix())
        if set(discovered)!=set(models):raise ValueError('Uncertified/extra completed NN models; no silent duplicate training or summary-only exit claims')
        d=same(self.proc.get(self.driver),r['legacy_driver']);s=same(self.proc.get(self.supervisor),r['legacy_supervisor']);self._default_term(d);self._default_term(s)
        if d['state'] not in {'T','t'}:raise ValueError('Old driver resumed during artifact audit')
        for rel,record in models.items():
            for n,h in record['artifact_sha256'].items():
                if sha(safe(self.legacy,rel)/n)!=h:raise ValueError('Completed NN changed during final audit')
        again=self.environment();self.fresh()
        if again['code_sha256']!=env['code_sha256'] or again['executor_plan_sha256']!=env['executor_plan_sha256'] or again['startup_sha256']!=env['startup_sha256']:raise ValueError('Source/executor identity changed during final audit')
        if not execute:return {'status':'verified complete boundary; no signals sent','completed_models':models,'newcontroller_start_allowed':False,'required_action':'finalize --execute-signals'}
        write_new(self.home/'termination_intent.json',{'created_at_utc':stamp(),'prepare_sha256':sha(self.home/'prepare.json'),'completed_models':models,'no_GPU_child_signal':True})
        self.proc.send(r['legacy_driver'],signal.SIGTERM)
        pending=same(self.proc.get(self.driver),r['legacy_driver'])
        if pending['state'] not in {'T','t'} or not pending.get('signal_pending',0)&(1<<(signal.SIGTERM-1)):raise RuntimeError('Stopped launcher pending TERM not confirmed; do not CONT or launch another controller')
        self.proc.send(r['legacy_driver'],signal.SIGCONT)
        driver_end=self._wait_dead(r['legacy_driver'])
        sup=self.proc.get(self.supervisor)
        if sup is not None:
            same(sup,r['legacy_supervisor'],zombie=True)
            if not dead(sup):self._default_term(sup);self.proc.send(r['legacy_supervisor'],signal.SIGTERM)
        supervisor_end=self._wait_dead(r['legacy_supervisor']);self.fresh()
        final={'format':'r5_controller_handoff_v1','created_at_utc':stamp(),
            'legacy_run_path':str(self.legacy),'new_run_path':str(self.new),'new_run_source':None,
            'config_sha256':env['config_sha256'],'executor_plan_sha256':env['executor_plan_sha256'],
            'legacy_driver_pid':self.driver,'legacy_driver_start_ticks':r['legacy_driver']['start_ticks'],
            'legacy_supervisor_pid':self.supervisor,'legacy_supervisor_start_ticks':r['legacy_supervisor']['start_ticks'],
            'current_child_pid':child['pid'],'current_child_start_ticks':child['start_ticks'],
            'child_exit_code_raw':child['exit_code_raw'],'child_finished_exit0':True,
            'complete_summary_sha256':models[current]['summary_sha256'],'completed_models':models,
            'old_driver_stopped_or_dead':True,'old_driver_dead_confirmed':True,'old_supervisor_dead_confirmed':True,
            'newcontroller_start_allowed':True,'old_driver_final_state':driver_end,'old_supervisor_final_state':supervisor_end,
            'prepare_sha256':sha(self.home/'prepare.json'),'parent_progress_receipt':progress,
            'source19_sha256':env['code_sha256'],'original_RUN_data_model_code_modified':False,
            'GPU_child_signals_sent':False,'SIGKILL_processgroup_SSH_Pod_actions_used':False}
        final['receipt_payload_sha256']=canonical(final);write_new(self.home/'finalized.json',final);return final
    def rollback(self,execute=False):
        r,_=self._prepared(allow_intent=True)
        if not execute:return {'status':'rollback is possible before finalization/new RUN; no signals sent','required_action':'rollback --execute-signals'}
        self.proc.send(r['legacy_driver'],signal.SIGCONT)
        record={'format':'r5_controller_handoff_rollback_v1','created_at_utc':stamp(),
            'stop_identity_receipt_sha256':sha(self.home/('prepare.json' if (self.home/'prepare.json').exists() else 'stop_intent.json')),'old_driver_resumed':True,'newcontroller_start_allowed':False,
            'warning':'Original controller will continue its original model schedule; no Mac-only controller may start.'}
        write_new(self.home/'rollback.json',record);return record

    def report(self):
        """Recovery inspection also works after a controller has exited."""
        if not (self.home/'stop_intent.json').exists():return self.inspect()
        result={'format':'r5_controller_handoff_status_v1','handoff_directory':str(self.home),
            'driver':self.proc.get(self.driver),'supervisor':self.proc.get(self.supervisor),
            'new_run_exists':self.new.exists(),'signals_sent_by_inspect':False,
            'receipts':{n:sha(self.home/n) for n in ['stop_intent.json','prepare.json','termination_intent.json','finalized.json','rollback.json'] if (self.home/n).exists()}}
        if (self.home/'finalized.json').exists():result['finalized']=read(self.home/'finalized.json')
        else:result['newcontroller_start_allowed']=False
        result['manual_review_required']=bool((self.home/'termination_intent.json').exists() and not (self.home/'finalized.json').exists())
        return result

class FakeProcesses:
    fixture=True
    def __init__(self,records):self.records=copy.deepcopy(records);self.signals=[]
    def get(self,pid):return copy.deepcopy(self.records.get(pid))
    def children(self,pid):return [self.get(p) for p,r in self.records.items() if r['ppid']==pid and not r.get('reaped')]
    def all(self):return [self.get(p) for p in self.records]
    def send(self,identity,sig):
        same(self.get(identity['pid']),identity);p=self.records[identity['pid']];self.signals.append([p['pid'],int(sig)])
        if sig==signal.SIGSTOP:p['state']='T'
        elif sig==signal.SIGTERM:
            if p['state']=='T':p['signal_pending']=1<<(signal.SIGTERM-1)
            else:p.update(state='Z',exit_code_raw=int(signal.SIGTERM),cmdline=[],cwd=None)
        elif sig==signal.SIGCONT:
            if p.get('signal_pending',0)&(1<<(signal.SIGTERM-1)):p.update(state='Z',exit_code_raw=int(signal.SIGTERM),cmdline=[],cwd=None)
            else:p['state']='S'
        else:raise AssertionError('Unexpected mock signal')
    def wait(self):pass

def _fixture(base):
    project=base/'project';(project/'models/legacy/fold_00').mkdir(parents=True)
    for p in names(ROOT/'pipeline/runner_r5.py','CODE_NAMES')+list(EXTRA)+['ops/runpod_train_r5.py','ops/runner_r5_mac.py','ops/mac_lgbm_r5.py']:
        dst=project/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,dst)
    cfg=read(ROOT/'configs/r5_training_gpu_r1.json');config=project/'configs/c.json';write_new(config,cfg)
    core={p:sha(project/p) for p in names(project/'pipeline/runner_r5.py','CODE_NAMES')}
    legacy=project/'models/legacy';fold=legacy/'fold_00';op=base/'ops/legacy';op.mkdir(parents=True)
    binding={'fold':0,'seed':42,'train_examples':8,'train_npz_sha256':'1'*64,'data_package_hash':'2'*64,'source_hash':'3'*64}
    prior={'fit_split':'train','training_binding':binding};prior['shared_prior_payload_sha256']=canonical(prior)
    write_new(fold/'shared_background_prior.json',prior)
    context={'format':'r5_fold_context_v1','config':cfg,'config_sha256':canonical(cfg),'training_binding':binding,
        'code_sha256':core,'data_fold':'data/fold_00','prior':{'filename':'shared_background_prior.json',
        'file_sha256':sha(fold/'shared_background_prior.json'),'payload_sha256':prior['shared_prior_payload_sha256']}}
    write_new(fold/'context.json',context)
    source_folder=project/'data/fold_00/sources'
    banks={'fakeTRAINbank':'7'*64}
    write_new(source_folder/'manifest.json',{'source_hash':binding['source_hash'],'numeric_bank_sha256':banks})
    gate={'ready_to_train':True,'source_hash':binding['source_hash'],'data_package_hash':binding['data_package_hash'],
        'source_version_dir':str(source_folder),'source_version_manifest_sha256':sha(source_folder/'manifest.json'),
        'file_sha256':{'training_background_order.npy':'5'*64,'train.npz':binding['train_npz_sha256']},
        'partitions':{'train':{'reference_segment_slice_sha256':{'fakeVT':'6'*64}}}}
    write_new(project/'data/fold_00/readiness_manifest.json',gate)
    identity={'binding':binding,'readiness_sha256':sha(project/'data/fold_00/readiness_manifest.json'),
        'source_version_manifest_sha256':gate['source_version_manifest_sha256'],'training_order_sha256':'5'*64,'reference_slice_sha256':{'fakeVT':'6'*64},
        'numeric_bank_sha256':banks}
    receipt={'epoch':1,'examples':8,'clean_primary_once':True,'source_identity_sha256':canonical(identity)}
    def completed(obs):
        model=fold/'preprocessing_pilot'/obs/'bg_cnn';model.mkdir(parents=True,exist_ok=True)
        policy=cfg['observation_candidates'][obs];model_receipt={'kind':'bg_cnn','numeric_fixture_only':True}
        cp={'format':'r5_neural_checkpoint_v1','kind':'bg_cnn','training_binding':binding,
            'shared_prior_sha256':prior['shared_prior_payload_sha256'],'observation_policy':policy,
            'training_settings':cfg['training'],'epoch':1,'receipt':model_receipt,
            'dynamic_source_identity':identity,'best_epoch_source_receipt':receipt,
            'weights':{'adapter_only':False,'state_dict':{'synthetic_fixture':[]}},
            'model_code_sha256':{'pipeline/'+n:sha(project/'pipeline'/n) for n in names(project/'pipeline/neural_r5.py','CHECKPOINT_CODE_NAMES')}}
        write_new(model/'best.pt',cp)
        with zipfile.ZipFile(model/'inner_predictions.npz','w') as z:z.writestr('fixture.txt','synthetic prediction container only')
        row={'epoch':1,'train_examples':8,'best_epoch':1,'best_inner_rmse':.2,'dynamic_epoch_source_receipt':receipt}
        (model/'epochs.jsonl').write_text(json.dumps(row)+'\n')
        write_new(model/'dynamic_source_receipts.json',{'source_identity':identity,'completed_epochs':{'1':receipt}})
        write_new(model/'summary.json',{'status':'completed','kind':'bg_cnn','fixture_only':False,
            'outer_or_Test_accessed':False,'training_binding':binding,'shared_prior_sha256':prior['shared_prior_payload_sha256'],
            'observation_policy':policy,'settings':cfg['training'],'receipt':model_receipt,
            'checkpoint_sha256':sha(model/'best.pt'),'prediction_sha256':sha(model/'inner_predictions.npz'),
            'epochs_run':1,'best_epoch':1,'best_inner_rmse':.2,
            'dynamic_source_identity':identity,'dynamic_source_receipts_file':'dynamic_source_receipts.json'})
    completed('raw')
    command=['/fake/python','-u','-m','pipeline.runner_r5','fit','--config',str(config),'--output',str(legacy),'--folds','0','1','2','3','4']
    startup={'command':command,'requested_folds':[0,1,2,3,4],'config_path':str(config),
        'config_sha256':canonical(cfg),'config_file_sha256':sha(config),'source_code_sha256':core,
        'supervisor_sha256':sha(project/'ops/runpod_train_r5.py')}
    write_new(op/'startup_receipt.json',startup)
    write_new(op/'training_state.json',{'status':'running','pid':200,'supervisor_pid':100,'process_start_ticks':'20'})
    plan=base/'executor_plan.json';write_new(plan,{'format':'r5_mac_executor_plan_v1','tree_executor':'mac_spool_v1',
        'user_approved':True,'server_tree_fit_allowed':False,'config_sha256':canonical(cfg),
        'code_sha256':{p:sha(project/p) for p in ['ops/runner_r5_mac.py','ops/mac_lgbm_r5.py']}})
    def process(pid,ppid,start,cmd):return {'pid':pid,'ppid':ppid,'start_ticks':str(start),'cmdline':cmd,
        'cwd':str(project),'state':'S','exit_code_raw':0,'signal_caught':0,'signal_ignored':0,'signal_pending':0}
    child_command=['/fake/python','-m','pipeline.neural_r5','fit','--context',str(fold/'context.json'),
        '--kind','bg_cnn','--observation','resolution_01','--output',str(fold/'preprocessing_pilot/resolution_01/bg_cnn')]
    records={100:process(100,1,10,['/fake/python',str(project/'ops/runpod_train_r5.py'),'--base',str(base),'--config',str(config),'--run-name','legacy']),
        200:process(200,100,20,command),300:process(300,200,30,child_command)}
    proc=FakeProcesses(records);tool=Handoff(base,'legacy','fresh_mac',200,100,plan,processes=proc,checkpoint_reader=read)
    return tool,proc,completed

def self_test():
    checks=[]
    fields=['0']*50;fields[0]='Z';fields[1]='200';fields[19]='1234';fields[49]='256'
    assert parse_stat('300 (synthetic (comm)) '+' '.join(fields))=={'state':'Z','ppid':200,'start_ticks':'1234','exit_code_raw':256}
    assert not dead({'state':'Z','threads':2})
    checks.append('Linux stat start/exit field offsets and zombie live-thread rejection')
    with tempfile.TemporaryDirectory(prefix='r5-handoff-mock-',dir='/private/tmp') as t:
        tool,proc,complete=_fixture(Path(t)/'main')
        inspected=tool.inspect();assert not proc.signals
        assert len(inspected['parent_progress_receipt']['preceding_completed_neural_calls'])==1
        checks.append('raw exit0 is explicit verified sequential-parent progress, not summary alone')
        dry=tool.stop_launcher();assert dry['signals_sent'] is False and not proc.signals
        prepared=tool.stop_launcher(execute=True);assert proc.signals==[[200,int(signal.SIGSTOP)]] and proc.get(300)['state']=='S'
        try:tool.finalize(execute=True)
        except ValueError:checks.append('active NN child blocks finalization without further signals')
        else:raise AssertionError('Active child accepted')
        complete('resolution_01');proc.records[300].update(state='Z',exit_code_raw=0,cmdline=[],cwd=None)
        before={p:sha(p) for p in tool.legacy.rglob('*') if p.is_file()}
        boundary=tool.finalize();assert len(boundary['completed_models'])==2 and len(proc.signals)==1
        final=tool.finalize(execute=True)
        assert final['newcontroller_start_allowed'] and final['old_driver_dead_confirmed'] and final['old_supervisor_dead_confirmed']
        assert len(final['completed_models'])==2 and all(x['exit_code']==0 for x in final['completed_models'].values())
        assert proc.signals==[[200,int(signal.SIGSTOP)],[200,int(signal.SIGTERM)],[200,int(signal.SIGCONT)],[100,int(signal.SIGTERM)]]
        assert all(pid!=300 for pid,sig in proc.signals) and all(sha(p)==h for p,h in before.items())
        checks+=['TERM pending verified before CONT; only old driver and then old supervisor signaled',
            'same-PID/start zombie kernel wait-status0 required','both completed NN models certified',
            'original model/context/code/data files unchanged','new RUN absent; new controller only after both old controllers inactive']
    for mutation in ['driver_reuse','non_neural_child','wrong_parent','nonzero_child','term_handler','new_run_blocks_rollback']:
        with tempfile.TemporaryDirectory(prefix='r5-handoff-negative-',dir='/private/tmp') as t:
            tool,proc,complete=_fixture(Path(t))
            if mutation=='driver_reuse':proc.records[200]['start_ticks']='999'
            elif mutation=='non_neural_child':proc.records[300]['cmdline'][2]='pipeline.lgbm_r5'
            elif mutation=='wrong_parent':proc.records[300]['ppid']=777
            if mutation in {'driver_reuse','non_neural_child','wrong_parent'}:
                try:tool.stop_launcher(execute=True)
                except ValueError:assert not proc.signals;checks.append(mutation+' rejected before signals')
                else:raise AssertionError('Unsafe process accepted')
                continue
            tool.stop_launcher(execute=True)
            if mutation=='new_run_blocks_rollback':
                tool.new.mkdir()
                try:tool.rollback(execute=True)
                except ValueError:checks.append(mutation)
                else:raise AssertionError('Rollback accepted with new RUN')
                continue
            complete('resolution_01');proc.records[300].update(state='Z',exit_code_raw=256 if mutation=='nonzero_child' else 0,cmdline=[],cwd=None)
            if mutation=='term_handler':proc.records[200]['signal_caught']=1<<(signal.SIGTERM-1)
            try:tool.finalize(execute=True)
            except ValueError:assert len(proc.signals)==1;checks.append(mutation+' rejected before termination')
            else:raise AssertionError('Unsafe boundary accepted')
    with tempfile.TemporaryDirectory(prefix='r5-handoff-rollback-',dir='/private/tmp') as t:
        tool,proc,_=_fixture(Path(t));tool.stop_launcher(execute=True);tool.rollback(execute=True)
        assert proc.signals==[[200,int(signal.SIGSTOP)],[200,int(signal.SIGCONT)]]
        checks.append('explicit pre-finalization rollback resumes only the original launcher')
    return {'passed':True,'scope':'synthetic proc records/signals and JSON checkpoint metadata only',
        'real_proc_read_real_signals_SSH_GPU_or_training_executed':False,'Torch_imported':False,
        'checks':checks,'receipt_format':'r5_controller_handoff_v1'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',nargs='?',default='inspect',choices=['inspect','stop-launcher','finalize','rollback'])
    p.add_argument('--base',type=Path);p.add_argument('--legacy-run');p.add_argument('--new-run')
    p.add_argument('--driver-pid',type=int);p.add_argument('--supervisor-pid',type=int);p.add_argument('--executor-plan',type=Path)
    p.add_argument('--execute-signals',action='store_true');p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:result=self_test()
    else:
        if any(getattr(a,k) is None for k in ['base','legacy_run','new_run','driver_pid','supervisor_pid','executor_plan']):p.error('Explicit base/RUN/PID/plan arguments required')
        h=Handoff(a.base,a.legacy_run,a.new_run,a.driver_pid,a.supervisor_pid,a.executor_plan)
        if a.action=='inspect':result=h.report()
        else:
            # Lock only our separate OPS handoff directory; never model/data RUN.
            if not a.execute_signals:
                result=getattr(h,{'stop-launcher':'stop_launcher','finalize':'finalize','rollback':'rollback'}[a.action])(False)
            else:
                import fcntl
                h.home.mkdir(parents=True,exist_ok=True)
                with (h.home/'.operation.lock').open('a') as f:
                    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    result=getattr(h,{'stop-launcher':'stop_launcher','finalize':'finalize','rollback':'rollback'}[a.action])(True)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
