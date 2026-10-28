"""Explicit verified recovery from an interrupted process, no automatic retry.

Preserves the frozen training source. Restores the last fully committed epoch
including Adam, cosine scheduler, CPU/MPS/Python/NumPy RNG and best artifacts.
"""
from pathlib import Path
import argparse,ast,copy,datetime,fcntl,json,os,shutil,subprocess,sys,time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline.training_r16_multisupport import MODELS,OUT,CACHE,codes,sha,read,write,verify
OPS=ROOT/'ops/r16_multisupport_resume_20261028_r1'
PARENT=ROOT/'ops/r16_multisupport_full_r1'
CPU='/Users/hongtaeyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
# Same installed torch2.14.1, MPS accelerator retained. The former Documents
# venv blocks on dataless iCloud-managed package reads; do not modify it.
MPS=CPU

def now():return datetime.datetime.now().astimezone().isoformat()
def source_pins():
 p=codes()
 for n in ('ops/continue_r16_multisupport_verified.py','ops/resume_r16_multisupport_lifecycle.py','pipeline/training_r16_production.py'):
  p[n]=sha(ROOT/n)
 return p
def inactive():
 s=read(PARENT/'state.json')
 for key in ('PID','child_PID'):
  pid=s.get(key)
  if pid:
   try:os.kill(pid,0)
   except ProcessLookupError:pass
   else:raise RuntimeError(f'Stored training process still present: {pid}')
 if (PARENT/'FAILED.json').exists() or (MODELS/'PAUSED.json').exists() or (MODELS/'PAUSE_AFTER_EPOCH').exists():raise ValueError('Explicit failed/paused marker requires separate resolution')
 if s['phase']!='neural':raise ValueError('Only interrupted neural phase supported')

def fixture(device):
 import torch,numpy as np,random
 from pipeline.training_r16_production import rng_state,restore_rng,cpu_state
 torch.set_num_threads(2);torch.manual_seed(1729);np.random.seed(1729);random.seed(1729)
 if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable, no fallback')
 def build():
  m=torch.nn.Sequential(torch.nn.Linear(5,11),torch.nn.Dropout(.1),torch.nn.GELU(),torch.nn.Linear(11,1)).to(device)
  o=torch.optim.Adam(m.parameters(),lr=.001,weight_decay=.0003)
  s=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=100,eta_min=1e-5)
  return m,o,s
 x=torch.arange(40,dtype=torch.float32,device=device).reshape(8,5)/40;y=x.sum(1,keepdim=True)/5
 def step(m,o,s):
  m.train();o.zero_grad(set_to_none=True);loss=((m(x)-y)**2).mean();loss.backward();o.step();s.step();return float(loss.detach().cpu())
 m,o,s=build();step(m,o,s)
 ck=copy.deepcopy(cpu_state({'m':m.state_dict(),'o':o.state_dict(),'s':s.state_dict(),'rng':rng_state(device)}))
 uninterrupted=step(m,o,s);expected=cpu_state(m.state_dict())
 mm,oo,ss=build();mm.load_state_dict(ck['m']);oo.load_state_dict(ck['o']);ss.load_state_dict(ck['s']);restore_rng(ck['rng'],device)
 resumed=step(mm,oo,ss)
 delta=max(float(abs(expected[k]-cpu_state(mm.state_dict())[k]).max()) for k in expected)
 if delta>1e-7 or abs(uninterrupted-resumed)>1e-7 or ss.state_dict()!=s.state_dict():raise ValueError('Checkpoint continuation fixture differs')
 return {'device':device,'parameter_max_abs_difference':delta,'loss_abs_difference':abs(uninterrupted-resumed),
    'scheduler_equal':True,'restores_Adam_and_all_RNG':True,'fixture_only':True}

def verify_recovery():
 import torch,math
 from ops.continue_r16_multisupport_verified import validate_partial
 from pipeline.models_r16_multisupport import MultiSupportModel
 inactive();pins=source_pins()
 # Training minibatch/epoch algorithm is byte-for-byte the original AST.
 old=ast.parse((ROOT/'pipeline/training_r16_multisupport.py').read_text())
 new=ast.parse((ROOT/'ops/continue_r16_multisupport_verified.py').read_text())
 def epoch_body(tree,name):
  fn=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name==name)
  loops=[v for v in ast.walk(fn) if isinstance(v,ast.For) and isinstance(v.target,ast.Name) and v.target.id=='epoch']
  return ast.dump(ast.Module(body=loops[0].body,type_ignores=[]),include_attributes=False)
 assert epoch_body(old,'train_neural')==epoch_body(new,'continue_neural_verified')
 f=MODELS/'neural/D0/bg_cnn';plan=read(f/'PLAN.json');model=MultiSupportModel('bg_cnn')
 ck=validate_partial(f,'bg_cnn','D0','mps',64,model.architecture())
 # Check actual saved production predictions on the unchanged feature cache
 # before accepting the environment relocation. No training/evaluation panels.
 import numpy as np
 from pipeline.training_r16_multisupport import cache_read
 va=cache_read(CACHE/'shared_selection/selection');parity={}
 for name,key in [('best_regression.pt','share'),('best_classifier.pt','classifier_prob')]:
  best=torch.load(f/name,map_location='cpu',weights_only=True);model.load_state_dict(best['state_dict']);model.to('mps').eval()
  saved=np.load(f/('selection_regression.npz' if key=='share' else 'selection_classification.npz'),allow_pickle=False)[key][:64]
  with torch.no_grad():
   xx=torch.tensor(np.asarray(va['x'][:64]),device='mps');ww=torch.tensor(np.asarray(va['w'][:64]),device='mps')
   pred,cl,_=model(xx,ww);values=(pred if key=='share' else cl.sigmoid()).cpu().numpy()
  delta=float(abs(values-saved).max());parity[name]=delta
  if delta>2e-5:raise ValueError('Existing MPS checkpoint/prediction replay differs after environment relocation')
 model.cpu();del model;torch.mps.empty_cache()
 tree=read(MODELS/'ALL_TREES_COMPLETED.json')
 assert tree['code_sha256']==codes() and tree['data_manifest_sha256']==sha(OUT/'manifest.json')
 for r in ('D0','H0'):
  for fold in range(5):
   folder=MODELS/'trees'/r/f'fold_{fold:02d}';sm=read(folder/'summary.json')
   for n,h in sm['artifact_sha256'].items():assert sha(folder/n)==h
 for manifest in CACHE.glob('*/*/manifest.json'):
  m=read(manifest);assert m['code_sha256']==codes() and m['data_manifest_sha256']==sha(OUT/'manifest.json')
  for n,h in m['artifact_sha256'].items():assert sha(manifest.parent/n)==h
 out={'status':'verified','created_kst':now(),'original_training_process_missing':True,'interruption_reason':'unknown; no Python exception recorded',
  'epoch_checkpoint':ck['epoch'],'resume_at_epoch':ck['epoch']+1,'optimizer_steps':ck['optimizer_steps'],
  'checkpoint_sha256':sha(f/'latest.pt'),'epoch_log_sha256':sha(f/'epochs.jsonl'),
  'original_training_epoch_algorithm_AST_identical':True,'all_completed_Trees_and_feature_caches_SHA_verified':True,
  'fixtures':[fixture('cpu'),fixture('mps')],'code_sha256':pins,'data_manifest_sha256':sha(OUT/'manifest.json'),
  'runtime':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'interpreter':sys.executable,
    'previous_interpreter':'/Users/hongtaeyang/Documents/Codex/2026-10-27/new-chat/.venv/bin/python',
    'reason':'former venv hangs reading cloud-managed dataless package files; originals untouched',
    'MPS_accelerator_retained':True,'existing_best_checkpoint_prediction_replay_max_abs':parity},
  'original_code_and_recipe_changed':False,'uncommitted_epoch_steps_discarded':True,
  'MPS_cross_process_bitwise_production_guarantee':False,'automatic_retry':False,'official_upload':False}
 verify(pins);write(OPS/'VERIFIED.json',out);print(json.dumps(out,ensure_ascii=False,indent=2))

def supervise():
 from ops.run_r16_multisupport_full import status
 OPS.mkdir(parents=True,exist_ok=True)
 with (PARENT/'writer.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);inactive()
  v=read(OPS/'VERIFIED.json');verify(v['code_sha256'])
  if v['data_manifest_sha256']!=sha(OUT/'manifest.json') or v['checkpoint_sha256']!=sha(MODELS/'neural/D0/bg_cnn/latest.pt'):raise ValueError('Recovery preflight stale')
  if (OPS/'STARTED.json').exists():raise FileExistsError('One explicit recovery only; no silent retry')
  snapshot=OPS/'snapshot_epoch61';snapshot.mkdir()
  for n in ('latest.pt','best_regression.pt','best_classifier.pt','epochs.jsonl','selection_regression.npz','selection_classification.npz','PLAN.json'):
   src=MODELS/'neural/D0/bg_cnn'/n;shutil.copy2(src,snapshot/n);assert sha(src)==sha(snapshot/n)
  write(OPS/'PREVIOUS_STATE.json',read(PARENT/'state.json'))
  write(OPS/'SNAPSHOT.json',{p.name:sha(p) for p in snapshot.iterdir()})
  write(OPS/'STARTED.json',{'started_kst':now(),'PID':os.getpid(),'verification_sha256':sha(OPS/'VERIFIED.json'),
   'authorization':'Current user explicitly instructs completion of first R16 training, followed by four audits; verified recovery preserves fixed training contract',
   'automatic_retry':False,'official_upload':False})
  keep=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  try:
   for stage,py in [('neural',MPS),('finalize',CPU)]:
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
      'PYTORCH_ENABLE_MPS_FALLBACK':'0','DYLD_LIBRARY_PATH':'/opt/homebrew/opt/libomp/lib',
      'PYTHONPATH':'.:/private/tmp/underwater-ml-deps:/private/tmp/underwater-eda-deps'}
    command=[py,str(ROOT/'ops/resume_r16_multisupport_lifecycle.py'),'neural'] if stage=='neural' else [py,str(ROOT/'ops/run_r16_multisupport_full.py'),'finalize']
    with (OPS/(stage+'.log')).open('xb',buffering=0) as log:
     p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
     while p.poll() is None:
      status({'phase':stage,'PID':os.getpid(),'child_PID':p.pid,'progress':read(MODELS/'progress.json'),
       'verified_recovery_receipt':str((OPS/'VERIFIED.json').relative_to(ROOT)),'official_upload':False})
      time.sleep(15)
    if p.returncode:raise RuntimeError(f'Recovery {stage} exited {p.returncode}; no retry')
   status({'phase':'completed','completed_kst':now(),'verified_recovery':True,'official_upload':False})
   write(OPS/'COMPLETED.json',{'completed_kst':now(),'official_upload':False})
  except BaseException as e:
   write(OPS/'FAILED.json',{'error':repr(e),'no_automatic_retry':True});status({'phase':'failed','error':repr(e),'recovery':True,'official_upload':False});raise
  finally:keep.terminate()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=('verify','supervise','neural'));a=p.parse_args()
 if a.action=='verify':verify_recovery()
 elif a.action=='supervise':supervise()
 else:
  from ops.continue_r16_multisupport_verified import continue_neural_verified
  if continue_neural_verified() is False:sys.exit(20)
