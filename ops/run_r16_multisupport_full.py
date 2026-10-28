"""One supervised paired-regime full run; no automatic failure restart/upload."""
from pathlib import Path
import argparse,datetime,fcntl,html,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.r16_multisupport_data import OUT,read,write,sha
from pipeline.training_r16_multisupport import MODELS
OPS=ROOT/'ops/r16_multisupport_full_r1'
CPU='/Users/hongtaeyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
MPS='/Users/hongtaeyang/Documents/Codex/2026-10-27/new-chat/.venv/bin/python'
def status(value):
 write(OPS/'state.json',value)
 report=ROOT/'reports/r16_multisupport_full_r1';report.mkdir(parents=True,exist_ok=True)
 text='<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>R16 본체 결합 실행</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:40px auto;padding:24px;background:#f6f8fc;color:#203049}pre{background:white;padding:20px;white-space:pre-wrap}a{color:#1259a7}</style><h1>R16 D0F1+H0 본체 실행</h1><p>장시간/HF 파일럿 RMSE0.117553(기준0.11) 미달을 명시한 진행 승인. D0/H0 각각 Tree5-Fold + 신경망4종 Fold0, 각100epoch/Adam/noES. 회귀·독립 분류 전문가 결합은 각각50:50 고정.</p><p>기존 Tree5와 CNN epoch27을 보존하며 새17채널 실행을 사용합니다. 원천 분리·24h purge 유지. 공식 업로드 없음.</p><pre>'+html.escape(json.dumps(value,ensure_ascii=False,indent=2))+'</pre><p><a href="index.html">완료 시 결과 보고서</a> · <a href="R16_MULTISUPPORT_FULL_RESULTS.md">완료 시 전체 MD</a> · <a href="../../submissions/r16_multisupport_full_r1/submission_r16.csv">완료 시 제출 후보 CSV</a></p></html>'
 (report/'START.html').write_text(text)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('all','build','trees','cache','preflight','neural','finalize'));args=parser.parse_args()
 if args.stage!='all':
  if args.stage=='build':
   from pipeline.r16_multisupport_data import build
   build()
  elif args.stage=='finalize':
   from ops.finalize_r16_multisupport_full import finalize
   finalize()
  else:
   from pipeline.training_r16_multisupport import train_trees,prepare_cache,preflight,train_neural
   result={'trees':train_trees,'cache':prepare_cache,'preflight':preflight,'neural':train_neural}[args.stage]()
   if result is False:sys.exit(20)
  return
 OPS.mkdir(parents=True,exist_ok=True);lock=(OPS/'writer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (OPS/'STARTED.json').exists()or MODELS.exists()or OUT.exists():raise FileExistsError('Fresh version only; no silent replay of a partial run')
 start=time.time();write(OPS/'STARTED.json',{'status':'started','PID':os.getpid(),'started_kst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
  'authorization':'User explicitly selected proceed with disclosed native12/HF RMSE miss .117553 vs .11',
  'regime_blend':{'regression':[.5,.5],'independent_classification':[.5,.5]},'Tree_folds_each_regime':5,'NN_architectures_each_regime':4,'NN_folds':[0],
  'NN_epochs':100,'NN_optimizer':'Adam','NN_early_stopping':False,'Tree_minimum_regression_weight':.5,'each_NN_architecture_maximum':.25,
  'old_checkpoint_resume':False,'official_upload':False,'no_automatic_retry':True})
 keepawake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 try:
  for stage in('build','trees','cache','preflight','neural','finalize'):
   py=MPS if stage in('preflight','neural')else CPU
   env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','PYTORCH_ENABLE_MPS_FALLBACK':'0',
    'PYTHONPATH':'.:ops/r14_production_runtime_py'if py==MPS else'.:/private/tmp/underwater-ml-deps:/private/tmp/underwater-eda-deps','DYLD_LIBRARY_PATH':'/opt/homebrew/opt/libomp/lib'}
   command=[py,str(Path(__file__).resolve()),stage]
   with(OPS/(stage+'.log')).open('xb',buffering=0)as log:
    proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    while proc.poll()is None:
     progress=read(MODELS/'progress.json')if(MODELS/'progress.json').exists()else(read(OUT/'progress.json')if(OUT/'progress.json').exists()else None)
     status({'phase':stage,'PID':os.getpid(),'child_PID':proc.pid,'elapsed_seconds':time.time()-start,'progress':progress,'official_upload':False})
     time.sleep(15)
   if proc.returncode!=0:raise RuntimeError(f'{stage} exited {proc.returncode}; see {OPS/(stage+".log")}')
  status({'phase':'completed','elapsed_seconds':time.time()-start,'submission':'submissions/r16_multisupport_full_r1/submission_r16.csv','official_upload':False})
 except BaseException as e:
  failure={'phase':'failed','error':repr(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-start,'no_automatic_retry':True}
  write(OPS/'FAILED.json',failure);status(failure);raise
 finally:
  keepawake.terminate()
if __name__=='__main__':main()
