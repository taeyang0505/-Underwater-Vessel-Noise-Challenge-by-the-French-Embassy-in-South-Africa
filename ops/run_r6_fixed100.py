"""Fresh fixed100 CNN run. No Test, old cross-panel evaluation or GPU access."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,subprocess,sys,time,concurrent.futures,threading
import numpy as np
from pipeline.cpu_ablation_r6_v2 import sha,save_json
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/r6_fixed100_v1.json'
RUN=ROOT/'models/r6_fixed100_v1'
DATA=ROOT/'data/synthetic/r6_cpu_guardrails_v3_schemafix'
CODE=['pipeline/fixed100_r6.py','ops/run_r6_fixed100.py']


def check_inputs(cfg):
    old=json.loads((DATA/'manifest.json').read_text())
    if sha(DATA/'manifest.json')!=cfg['data_manifest_sha256']:raise ValueError('Stored data manifest changed')
    if sha(ROOT/'configs/r6_cpu_guardrails_v2.json')!=old['freeze']['config_sha256']:raise ValueError('Original data recipe changed')
    for n,h in old['freeze']['code_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Original core changed: '+n)
    for n,h in cfg['used_input_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Input changed: '+n)
    for n,h in cfg['original_frozen_core17_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Original frozen R5 core changed: '+n)
    return old


def load(name):
    with np.load(DATA/name,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def worker(index):
    cfg=json.loads(CONFIG.read_text());check_inputs(cfg);freeze=json.loads((RUN/'freeze.json').read_text())
    if sha(CONFIG)!=freeze['config_sha256'] or any(sha(ROOT/n)!=h for n,h in freeze['new_code_sha256'].items()):raise ValueError('New run code/config changed')
    if not 0<=index<len(cfg['CNN_arms']):raise ValueError('Explicit valid arm required')
    from pipeline.fixed100_r6 import fit_fixed100
    arm=cfg['CNN_arms'][index]
    def progress(row):save_json(RUN/(arm['name']+'.progress.json'),row)
    return fit_fixed100(load('train_'+arm['generator']+'.npz'),load('inner_'+arm['generator']+'.npz'),arm,cfg,RUN/arm['name'],progress)


def fit_all():
    cfg=json.loads(CONFIG.read_text());check_inputs(cfg)
    if RUN.exists():raise FileExistsError('Fresh run only; prior or failed run cannot be resumed')
    RUN.mkdir(parents=True);freeze={'config_sha256':sha(CONFIG),'new_code_sha256':{n:sha(ROOT/n) for n in CODE},'data_manifest_sha256':cfg['data_manifest_sha256']}
    save_json(RUN/'freeze.json',freeze);start=time.monotonic()
    children={};child_lock=threading.Lock()
    def run_one(index):
        arm=cfg['CNN_arms'][index];env=dict(os.environ)
        for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[k]=str(cfg['neural']['threads'])
        with (RUN/(arm['name']+'.driver.log')).open('x') as f:
            command=[sys.executable,'-u','-m','ops.run_r6_fixed100','worker','--index',str(index)]
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            with child_lock:children[index]=child
            code=child.wait()
            if code:raise subprocess.CalledProcessError(code,command)
        return json.loads((RUN/arm['name']/'summary.json').read_text())
    summaries={}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['execution']['maximum_parallel_arms']) as pool:
            jobs={pool.submit(run_one,i):i for i in range(len(cfg['CNN_arms']))}
            while jobs:
                done,_=concurrent.futures.wait(jobs,timeout=10,return_when=concurrent.futures.FIRST_COMPLETED)
                for job in done:
                    idx=jobs.pop(job)
                    try:s=job.result()
                    except BaseException as error:
                        # Report immediately, then stop only subprocesses this
                        # runner itself created. No external jobs are touched.
                        save_json(RUN/'FAILED.json',{'error':str(error),'arm':cfg['CNN_arms'][idx]['name'],'partial_reusable':False})
                        save_json(RUN/'progress.json',{'phase':'failed','arm':cfg['CNN_arms'][idx]['name'],'remaining_own_children_stopping':True})
                        with child_lock:
                            for process in children.values():
                                if process.poll() is None:process.terminate()
                        for future in jobs:future.cancel()
                        raise
                    summaries[idx]=s
                    print(json.dumps({'completed':s['arm']['name'],'epochs':s['epochs_run'],'best_window_rmse':s['best_inner_window']['metrics']['rmse'],'best_window_epoch':s['best_inner_window']['epoch']}),flush=True)
                save_json(RUN/'progress.json',{'phase':'fitting','completed_arms':len(summaries),'total_arms':len(cfg['CNN_arms']),
                    'running_or_pending':[cfg['CNN_arms'][i]['name'] for i in jobs.values()],'elapsed_seconds':time.monotonic()-start,'heldout_evaluated':False})
        check_inputs(cfg)
        if sha(CONFIG)!=freeze['config_sha256'] or any(sha(ROOT/n)!=h for n,h in freeze['new_code_sha256'].items()):raise ValueError('Code/config changed during training')
        out={'status':'completed','config_sha256':sha(CONFIG),'data_manifest_sha256':cfg['data_manifest_sha256'],'former_ES_run':'models/r6_cpu_guardrails_v2',
             'summaries':[summaries[i] for i in range(len(cfg['CNN_arms']))],'elapsed_seconds':time.monotonic()-start,
             'scope':'five fixed CPU CNN arms100epochs each; no global optimum guarantee','heldout_evaluated':False,'Test_GPU_or_submission_used':False,
             'old_heldout_panels_already_seen_and_not_blind':True,'training_end_utc':datetime.now(timezone.utc).isoformat()}
        save_json(RUN/'results.json',out);save_json(ROOT/'reports/R6_FIXED100_RESULTS.json',out)
        save_json(RUN/'all_models_frozen.json',{'status':'frozen','config_sha256':sha(CONFIG),'model_summary_sha256':{s['arm']['name']:sha(RUN/s['arm']['name']/'summary.json') for s in out['summaries']},'heldout_evaluated':False})
        save_json(RUN/'progress.json',{'phase':'completed','completed_arms':5,'epochs_per_arm':100,'heldout_evaluated':False})
    except BaseException as e:
        save_json(RUN/'FAILED.json',{'error':str(e),'type':type(e).__name__,'partial_reusable':False});raise


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','worker']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.action=='fit':fit_all()
    elif a.index is None:raise ValueError('Worker index required')
    else:worker(a.index)
if __name__=='__main__':main()
