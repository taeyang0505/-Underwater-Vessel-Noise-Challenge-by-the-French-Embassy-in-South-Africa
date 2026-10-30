"""Fresh CUDA BF16, masked-only Fold0 controller: 3 families x D/H x 2 tasks.

Preparation and preflight never train a production checkpoint or read Test.
A real CUDA preflight for every family and a completed explicit Mac control
report are required before the immutable 12-job production plan is created.
No incomplete job is resumed or retried automatically.
"""
from pathlib import Path
import argparse, datetime, json, os, subprocess, sys, traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')


def completed_control(path):
    from pipeline.r16_multisupport_data import read, sha
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('Explicit project-contained completed Mac control report required')
    record = read(path)
    if record.get('status') != 'completed':
        raise ValueError('Mac comparison is not completed')
    # Exact task/variant coverage must be proved by the control, not inferred
    # from an MPS one-step or throughput receipt.
    required = {('regression', 'off'), ('regression', 'both'),
                ('classification', 'off'), ('classification', 'both')}
    actual = {(job.get('task'), job.get('variant')) for job in record.get('jobs', [])
              if job.get('status') == 'completed'}
    if actual != required or len(record.get('jobs',[]))!=4:
        raise ValueError('Completed Mac control must cover four task x masking jobs')
    plan_path=ROOT/'ops'/path.parent.name/'PLAN.json'
    if not plan_path.is_file() or sha(plan_path)!=record.get('plan_sha256'):
        raise ValueError('Mac control plan identity differs')
    plan=read(plan_path)
    if (plan.get('device')!='mps' or plan.get('Test_used') is not False
        or plan.get('locked_panel_used') is not False
        or plan.get('comparison_used_for_epochs') is not False):
        raise ValueError('Mac control is not the authorized independent small comparison')
    job_receipts={}
    for job in record['jobs']:
        folder=(ROOT/job['folder']).resolve()
        if not folder.is_relative_to(ROOT/'models'):
            raise ValueError('Mac control artifact folder escaped project model storage')
        if read(folder/'summary.json')!=job or job.get('plan_sha256')!=sha(plan_path):
            raise ValueError('Mac completed job summary/plan differs')
        if (job.get('epochs_run')!=8 or job.get('frozen_base_unchanged') is not True
            or job.get('small_control_not_production') is not True
            or job.get('Test_used') is not False or job.get('locked_panels_used') is not False):
            raise ValueError('Incomplete or unsafe Mac control job')
        required_files={'PLAN.json','best.pt','epochs.jsonl','comparison.npz','checkpoint_best.npz'}
        if not required_files<=set(job.get('artifact_sha256',{})):
            raise ValueError('Mac control required artifacts missing')
        for name,digest in job['artifact_sha256'].items():
            candidate=(folder/name).resolve()
            if candidate.parent!=folder or sha(candidate)!=digest:
                raise ValueError('Mac control artifact SHA/path differs')
        job_receipts[job['task']+'/'+job['variant']]=sha(folder/'summary.json')
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
            'plan_sha256':sha(plan_path),'completed_job_summary_sha256':job_receipts,
            'not_proof_for_Dasheng_or_TimeMoE_masking_benefit': True}


def fixed_contract():
    from pipeline.training_r16_pretrained_cuda_bf16 import FAMILIES, TASKS, REGIMES, MICROBATCH, NUMERICAL_POLICY
    from pipeline.r16_feature_masking import FeatureMaskingConfig
    return {'format':'r16_pretrained_cuda_bf16_12job_plan_v1','status':'fixed_before_training',
            'families':list(FAMILIES),'variants':['both'],'regimes':list(REGIMES),'tasks':list(TASKS),
            'folds':[0],'total_jobs':12,'numerical_policy':NUMERICAL_POLICY,'microbatch':MICROBATCH,
            'masking_variants':{'both':FeatureMaskingConfig.for_variant('both').state_dict()},
            'effective_batch':64,'warmup_epochs':3,'stage2_max_epochs':25,'patience':7,
            'optimizer':'Adam','warmup_learning_rate':.001,'stage2_learning_rate':.0002,
            'stage2_minimum_learning_rate':.00001,'schedule':'cosine','weight_decay':.001,
            'dropout':.1,'LoRA_rank':8,'gradient_clip':1.,
            'LightGBM_regression_weight_minimum':.5,'NN_architecture_maximum':.25,
            'D0_H0_weights':[.5,.5],'classifier_score_share_weight':0.,
            'regression_classifier_trainable_parameters_shared':False,
            'scratch_folds1to4_cancelled':True,'Test_used':False,'full_5fold_neural_training':False}


def validate_plan(plan):
    from pipeline.training_r16_pretrained_cuda_bf16 import OPS, OUT, FAMILIES, TASKS, MICROBATCH, NUMERICAL_POLICY, source_pins
    from pipeline.r16_multisupport_data import read, sha
    for key,value in fixed_contract().items():
        if plan.get(key)!=value:raise ValueError('Fixed production contract differs: '+key)
    if plan.get('source_sha256')!=source_pins() or plan.get('data_manifest_sha256')!=sha(OUT/'manifest.json'):
        raise ValueError('Frozen CUDA plan source/data mismatch')
    if plan.get('local_fit_split_sha256')!=sha(OPS/'LOCAL_FIT_SPLIT.json'):
        raise ValueError('Fixed local_fit partition changed')
    for family in FAMILIES:
        path=OPS/f'PREFLIGHT_{family}_both.json';record=read(path)
        if sha(path)!=plan.get('preflight_sha256',{}).get(family):
            raise ValueError('Preflight receipt changed after plan freeze')
        if (record.get('status')!='passed' or record.get('source_sha256')!=source_pins()
            or record.get('data_manifest_sha256')!=plan['data_manifest_sha256']
            or record.get('local_fit_split_sha256')!=plan['local_fit_split_sha256']
            or record.get('numerical_policy')!=NUMERICAL_POLICY):
            raise ValueError('Actual GPU preflight no longer binds current run')
        for task in TASKS:
            branch=record.get('models',{}).get(task,{})
            if branch.get('passed') is not True or branch.get('device')!='cuda' or branch.get('microbatch')!=MICROBATCH[family]:
                raise ValueError('Missing actual GPU task/microbatch verification')
    receipt=plan['Mac_small_control']
    if completed_control(ROOT/receipt['path'])!=receipt:
        raise ValueError('Mac comparison report changed after plan freeze')


def main():
    from pipeline.training_r16_pretrained_cuda_bf16 import (
        OPS, RUN, OUT, FAMILIES, TASKS, REGIMES, MICROBATCH, NUMERICAL_POLICY,
        prepare, preflight, train_job, source_pins, selection_split)
    from pipeline.r16_feature_masking import FeatureMaskingConfig
    from pipeline.r16_multisupport_data import guard, read, write, sha
    from pipeline.r16_cuda_precision import require_cuda_bf16
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'preflight', 'plan', 'train', 'run'))
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--regime', choices=REGIMES)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--mac-control-report', type=Path)
    parser.add_argument('--continue-after-completed', action='store_true')
    args = parser.parse_args()
    def audit(event, values):
        guard(event, values)
        if event in ('socket.connect', 'socket.getaddrinfo'):
            raise PermissionError('Offline production runner: public weights must already be local')
    sys.addaudithook(audit)
    OPS.mkdir(parents=True, exist_ok=True)
    if args.command == 'prepare':
        prepare(); return 0
    if args.command == 'preflight':
        if not args.family: parser.error('--family required')
        preflight(args.family); return 0
    if args.command == 'plan':
        if args.mac_control_report is None: parser.error('--mac-control-report required')
        control = completed_control(args.mac_control_report)
        hardware = require_cuda_bf16(); prepare(); pins = source_pins()
        preflights = {}
        for family in FAMILIES:
            path = OPS/f'PREFLIGHT_{family}_both.json'; record = read(path)
            if (record.get('status') != 'passed' or record.get('source_sha256') != pins
                or record.get('numerical_policy') != NUMERICAL_POLICY
                or record.get('data_manifest_sha256') != sha(OUT/'manifest.json')
                or record.get('local_fit_split_sha256') != sha(OPS/'LOCAL_FIT_SPLIT.json')):
                raise ValueError('Current real CUDA BF16 preflight required: '+family)
            preflights[family] = sha(path)
        plan = {**fixed_contract(),
            'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'families': list(FAMILIES), 'variants': ['both'], 'regimes': list(REGIMES),
            'tasks': list(TASKS), 'folds': [0], 'total_jobs': 12,
            'masking_variants': {'both': FeatureMaskingConfig.for_variant('both').state_dict()},
            'numerical_policy': NUMERICAL_POLICY, 'microbatch': MICROBATCH,
            'effective_batch': 64, 'warmup_epochs': 3, 'stage2_max_epochs': 25, 'patience': 7,
            'optimizer': 'Adam', 'warmup_learning_rate': .001, 'stage2_learning_rate': .0002,
            'stage2_minimum_learning_rate': .00001, 'schedule': 'cosine', 'weight_decay': .001,
            'dropout': .1, 'LoRA_rank': 8, 'gradient_clip': 1., 'source_sha256': pins,
            'data_manifest_sha256': sha(OUT/'manifest.json'),
            'local_fit_split_sha256': sha(OPS/'LOCAL_FIT_SPLIT.json'),
            'checkpoint_selection': '199 source backgrounds /1194rows; ensemble role disjoint',
            'ensemble_fit': '201 source backgrounds /1206rows; legacyTree ES reuse disclosed',
            'LightGBM_regression_weight_minimum': .5, 'NN_architecture_maximum': .25,
            'D0_H0_weights': [.5,.5], 'classifier_score_share_weight': 0.,
            'regression_classifier_trainable_parameters_shared': False,
            'scratch_folds1to4_cancelled': True, 'preflight_sha256': preflights,
            'Mac_small_control': control, 'hardware': hardware, 'Test_used': False,
            'production_full_epoch_time_not_yet_observed': True,
            'full_5fold_neural_training': False}
        dest = OPS/'RUN_PLAN.json'
        if dest.exists():
            old = read(dest)
            plan['created_at'] = old.get('created_at')
            if plan != old: raise FileExistsError('Immutable CUDA plan already exists and differs')
        else: write(dest, plan)
        print(json.dumps(plan,ensure_ascii=False),flush=True); return 0
    plan = read(OPS/'RUN_PLAN.json')
    validate_plan(plan)
    require_cuda_bf16()
    if args.command == 'train':
        if not args.family or not args.regime or not args.task:
            parser.error('--family --regime --task required')
        return train_job(args.family,args.regime,args.task,'both')
    lock = OPS/'controller.lock'
    descriptor = os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode()); os.close(descriptor)
    try:
        for family in FAMILIES:
            for regime in REGIMES:
                for task in TASKS:
                    folder = RUN/'both'/family/regime/task
                    if folder.exists():
                        if not args.continue_after_completed:
                            raise FileExistsError('Existing job requires explicit completed-only continuation')
                        record = read(folder/'summary.json')
                        if (record.get('status') != 'completed' or record.get('source_sha256') != source_pins()
                            or record.get('numerical_policy') != NUMERICAL_POLICY):
                            raise ValueError('Partial/failed/changed job may not be reused')
                        for name, digest in record['artifact_sha256'].items():
                            if sha(folder/name) != digest: raise ValueError('Completed artifact changed')
                        continue
                    write(OPS/'state.json',{'phase':'training','family':family,'regime':regime,
                        'task':task,'variant':'both','controller_pid':os.getpid(),
                        'run_plan_sha256':sha(OPS/'RUN_PLAN.json')})
                    command = [sys.executable,str(Path(__file__)),'train','--family',family,
                               '--regime',regime,'--task',task]
                    with (OPS/f'{family}_{regime}_{task}.log').open('a') as log:
                        child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                                 env=os.environ.copy())
                        write(OPS/'process.json',{'controller_pid':os.getpid(),'worker_pid':child.pid,
                                                'job':[family,regime,task]})
                        code = child.wait()
                    if code == 20:
                        write(OPS/'state.json',{'phase':'paused_after_epoch','job':[family,regime,task]})
                        return 20
                    if code: raise RuntimeError(f'{family}/{regime}/{task} failed; no automatic retry')
        write(OPS/'state.json',{'phase':'all_selected_neural_completed','total_jobs':12,
                              'finalization_pending':True,'official_submission':False})
        return 0
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        from pipeline.r16_multisupport_data import write
        write(ROOT/'ops/r16_pretrained_cuda_bf16_r1/LAST_FAILURE.json',
              {'error':repr(error),'traceback':traceback.format_exc(),'automatic_retry':False})
        raise
