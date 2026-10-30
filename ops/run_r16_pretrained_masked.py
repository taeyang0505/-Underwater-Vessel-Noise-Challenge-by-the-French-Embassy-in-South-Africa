"""Explicit CLI for the new public-pretrained pipeline. No scratch continuation.

No command resumes a partial job automatically. Completed jobs can be skipped
only after checksum verification with --continue-after-completed.
"""
from pathlib import Path
import argparse, datetime, json, os, subprocess, sys, traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

def main():
    from pipeline.training_r16_pretrained_masked import (OPS,RUN,OUT,FAMILIES,TASKS,REGIMES,
        branch,prepare,preflight,train_job,source_pins,VARIANTS,selection_split)
    from pipeline.r16_multisupport_data import guard,read,write,sha
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','preflight','plan','train','run'))
    p.add_argument('--family',choices=FAMILIES)
    p.add_argument('--variant',choices=VARIANTS,default='both')
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['off','both'])
    p.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    p.add_argument('--regime',choices=REGIMES)
    p.add_argument('--task',choices=TASKS)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--device',choices=('cpu','mps','cuda'),default='mps')
    p.add_argument('--continue-after-completed',action='store_true')
    a=p.parse_args();sys.addaudithook(guard);OPS.mkdir(parents=True,exist_ok=True)
    if a.command=='prepare':prepare();return 0
    if a.command=='preflight':
        if not a.family or a.batch<1:p.error('--family and positive --batch required')
        preflight(a.family,a.batch,a.device,a.variant);return 0
    if a.command=='plan':
        import torch, numpy as np
        prepare();pins=source_pins()
        families=list(dict.fromkeys(a.families))
        for family in families:
            for variant in a.variants:
                pf=read(OPS/f'PREFLIGHT_{family}_{variant}.json')
                if pf['status']!='passed' or pf['source_sha256']!=pins:raise ValueError('Current real masking preflight required '+family+'/'+variant)
                if pf['local_fit_split_sha256']!=sha(OPS/'LOCAL_FIT_SPLIT.json'):raise ValueError('Preflight split differs')
        plan={'format':'r16_public_pretrained_plan_v1','status':'fixed_before_training',
            'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'families':families,'variants':list(dict.fromkeys(a.variants)),
            'masking_variants':{v:__import__('pipeline.r16_feature_masking',fromlist=['FeatureMaskingConfig']).FeatureMaskingConfig.for_variant(v).state_dict() for v in a.variants},
            'local_fit_split_sha256':sha(OPS/'LOCAL_FIT_SPLIT.json'),'regimes':list(REGIMES),'tasks':list(TASKS),'folds':[0],
            'warmup_epochs':3,'stage2_max_epochs':25,'patience':7,
            'optimizer':'Adam','warmup_learning_rate':.001,'stage2_learning_rate':.0002,
            'stage2_minimum_learning_rate':.00001,'schedule':'cosine','weight_decay':.001,
            'dropout':.1,'LoRA_rank':8,'effective_batch':64,'gradient_clip':1.,
            'source_sha256':pins,'data_manifest_sha256':sha(OUT/'manifest.json'),
            'checkpoint_selection':'local_fit checkpoint subset:199bg1194rows; no overlap with new ensemble subset',
            'ensemble_fit':'local_fit ensemble subset:201bg1206rows; legacyLightGBM historically saw all400 for ES',
            'regression_classifier_trainable_parameters_shared':False,
            'frozen_base_still_differentiates_inputs':True,'silent_CPU_fallback':False,
            'LightGBM_regression_weight_minimum':.5,'NN_architecture_maximum':.25,
            'D0_H0_weights':[.5,.5],'classifier_score_share_weight':0.,
            'scratch_folds1to4_cancelled':True,'public_pretrained_transfer_benefit':'not yet measured',
            'runtime':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__},
            'Test_used':False}
        dest=OPS/'RUN_PLAN.json'
        if dest.exists():
            old=read(dest)
            if old['source_sha256']!=pins or old['families']!=families or old['variants']!=list(dict.fromkeys(a.variants)):raise FileExistsError('Immutable run plan exists')
        else:write(dest,plan)
        print(json.dumps(plan,ensure_ascii=False),flush=True);return 0
    if a.command=='train':
        if not a.family or not a.regime or not a.task:p.error('--family --regime --task required')
        plan=read(OPS/'RUN_PLAN.json')
        if a.family not in plan['families']:raise ValueError('Family outside fixed run plan')
        if a.variant not in plan['variants']:raise ValueError('Variant outside fixed run plan')
        return train_job(a.family,a.regime,a.task,a.variant)
    plan=read(OPS/'RUN_PLAN.json');jobs=[]
    for variant in plan['variants']:
        for family in plan['families']:
            for regime in REGIMES:
                for task in TASKS:jobs.append((variant,family,regime,task))
    lock=OPS/'controller.lock'
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        for variant,family,regime,task in jobs:
            folder=RUN/variant/family/regime/task
            if folder.exists():
                if not a.continue_after_completed:raise FileExistsError('Existing job needs explicit completed-only continuation')
                s=read(folder/'summary.json')
                if s['status']!='completed' or s['source_sha256']!=plan['source_sha256']:raise ValueError('Partial/stale job cannot be reused')
                for name,h in s['artifact_sha256'].items():
                    if sha(folder/name)!=h:raise ValueError('Completed job artifact changed')
                continue
            write(OPS/'state.json',{'phase':'training','variant':variant,'family':family,'regime':regime,'task':task,
                'controller_pid':os.getpid(),'run_plan_sha256':sha(OPS/'RUN_PLAN.json')})
            cmd=[sys.executable,str(Path(__file__)),'train','--variant',variant,'--family',family,'--regime',regime,'--task',task]
            with (OPS/f'{variant}_{family}_{regime}_{task}.log').open('a') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                write(OPS/'process.json',{'controller_pid':os.getpid(),'worker_pid':child.pid,'job':[variant,family,regime,task]})
                code=child.wait()
            if code==20:
                write(OPS/'state.json',{'phase':'paused_after_epoch','job':[variant,family,regime,task]});return 20
            if code:raise RuntimeError(f'Job {family}/{regime}/{task} exited {code}; no retry')
        write(OPS/'state.json',{'phase':'all_selected_neural_completed','families':plan['families'],
            'finalization_pending':True,'official_submission':False})
        return 0
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        from pipeline.r16_multisupport_data import write
        write(ROOT/'ops/r16_pretrained_masking_r1/LAST_FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc(),
            'automatic_retry':False})
        raise
