"""V2 explicit CLI. Default plan never starts model training or Test inference."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',nargs='?',default='plan',choices=('plan','preflight','train','run-conv'))
    p.add_argument('--regime',choices=('D0','H0'))
    p.add_argument('--task',choices=('regression','classification'))
    a=p.parse_args()
    from pipeline.training_r16_version2 import OPS,RUN,make_plan,check_preflight,train_job,REGIMES,TASKS
    from pipeline.r16_multisupport_data import guard,write
    sys.addaudithook(guard)
    if a.command=='preflight':
        # Inspect passed disposable receipts only. Benchmark execution belongs
        # to ops/preflight_r16_v2_mps.py and does not imply completed training.
        pf=check_preflight();print(json.dumps({'status':'passed','device':pf['device'],'torch':pf['torch'],'training_started':False}));return 0
    if a.command=='plan':
        plan=make_plan();print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    if a.command=='train':
        if not a.regime or not a.task:p.error('Explicit --regime and --task required')
        return train_job(a.regime,a.task)
    # Explicit controller command only. Never skip/reuse a partial V2 job.
    lock=OPS/'controller.lock';OPS.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        for regime in REGIMES:
            for task in TASKS:
                code=train_job(regime,task)
                if code:return code
        write(OPS/'ALL_CONV_COMPLETED.json',{'status':'completed','jobs':4,'folds':[0],
            'Dasheng_TimeMoE_retrained':False,'finalization_pending':True,'Test_inference_performed':False})
        return 0
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':raise SystemExit(main())
