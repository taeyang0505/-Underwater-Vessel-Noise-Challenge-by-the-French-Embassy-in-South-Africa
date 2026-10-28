"""Explicit R13 stages; no SSH, fitting on Test or automatic query reading.

Only the separate explicit infer command opens a query CSV, after verifying
all five completed frozen folds. All training/preparation commands exclude it.

Examples (GPU server only after a real CUDA preflight):
  python -m ops.run_r13_training preflight --output reports/R13_GPU_PREFLIGHT.json
  python -m ops.run_r13_training fit --preflight reports/R13_GPU_PREFLIGHT.json

CPU self-test uses only synthetic arrays and one epoch/full-size model fixtures.
The packaging plan lists dependencies; it neither copies nor downloads data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.runner_r13 import (ROOT, NEURAL, CODE_NAMES, sha, canonical, atomic_json,
                               _validate_config, ProductionData, run_training,
                               restore_neural, fit_mac_trees, import_mac_trees,
                               freeze_ensembles, evaluate_reports, _verify_global)


def verify_run(folder, *, allow_fixture=False):
    """Verify every completed step artifact and model summary; never load data."""
    folder=Path(folder).resolve()
    if (folder/'FAILED.json').exists() or not (folder/'results.json').is_file():
        raise ValueError('Completed unfailed run required')
    identity=json.loads((folder/'execution_identity.json').read_text())
    result=json.loads((folder/'results.json').read_text())
    state=json.loads((folder/'run_state.json').read_text())
    if (result.get('status')!='completed' or state.get('status')!='completed'
            or state.get('results_sha256')!=sha(folder/'results.json')
            or result.get('fixture_only')!=identity.get('fixture_only')
            or result.get('config_sha256')!=identity.get('config_sha256')):
        raise ValueError('Run completion/binding differs')
    if identity['fixture_only'] and not allow_fixture:raise ValueError('Fixture cannot pass a production verification')
    if canonical(identity['config'])!=identity['config_sha256']:raise ValueError('Config payload changed')
    if not identity['fixture_only']:
        rp=folder/'runtime_policy.json'
        if not rp.is_file() or sha(rp)!=identity.get('runtime_policy_sha256'):
            raise ValueError('Runtime recipe file missing/changed')
        cp=folder/'source_config.json'
        if not cp.is_file() or sha(cp)!=identity.get('config_file_sha256'):
            raise ValueError('Portable original config byte copy missing/changed')
        from pipeline.runtime_policy_r13 import load_runtime_policy
        bundle=ROOT/identity['prepared_bundle_dir']
        load_runtime_policy(rp,cp,json.loads((bundle/'manifest.json').read_text()),
            identity['code_sha256'],ROOT,require_approved=True)
    for name,digest in identity['code_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Code changed: '+name)
    global_ledger=json.loads((folder/'run_artifact_ledger.json').read_text())
    if (sha(folder/'run_artifact_ledger.json')!=state.get('run_artifact_ledger_sha256')
            or global_ledger.get('format')!='r13_completed_run_artifact_ledger_v1'
            or global_ledger.get('complete') is not True or global_ledger['code_sha256']!=identity['code_sha256']):
        raise ValueError('Completed root artifact ledger differs')
    for name,digest in global_ledger['artifact_sha256'].items():
        rel=Path(name);p=folder/rel
        if rel.is_absolute() or '..' in rel.parts or p.is_symlink() or not p.is_file() or sha(p)!=digest:
            raise ValueError('Completed run artifact changed: '+name)
    actual_global={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if actual_global!=set(global_ledger['artifact_sha256'])|{'run_artifact_ledger.json','run_state.json'}:
        raise ValueError('Untracked completed-run artifact')
    artifacts=0;ledgers={}
    for fold in result['folds']:
        fp=folder/f'fold_{fold:02d}';ledger=json.loads((fp/'artifact_ledger.json').read_text())
        if ledger.get('binding',{}).get('fold')!=fold or ledger['binding']['config_sha256']!=identity['config_sha256']:
            raise ValueError('Fold ledger binding differs')
        expected=['context',*['train/'+k for k in NEURAL],'neural_completed',
                  'train/lgbm_physics','ensemble_frozen','report_completed']
        if list(ledger['steps'])!=expected or ledger['code_sha256']!=identity['code_sha256']:
            raise ValueError('Incomplete ordered stage ledger')
        covered=set()
        for step in ledger['steps'].values():
            if step.get('status')!='completed':raise ValueError('Partial ledger step')
            for name,digest in step['artifact_sha256'].items():
                rel=Path(name)
                if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe artifact path')
                p=fp/rel
                if p.is_symlink() or not p.is_file() or sha(p)!=digest:raise ValueError('Completed artifact changed: '+name)
                covered.add(name);artifacts+=1
        actual={p.relative_to(fp).as_posix() for p in fp.rglob('*') if p.is_file()}
        if actual-covered-{'artifact_ledger.json','progress.json'}:
            raise ValueError('Untracked fold artifact: '+str(sorted(actual-covered)))
        ledgers[str(fold)]=sha(fp/'artifact_ledger.json')
    _verify_global(folder,result['folds'],identity['code_sha256'])
    return {'status':'verified','fixture_only':identity['fixture_only'],'folds':result['folds'],
            'artifact_entries_verified':artifacts,'fold_ledger_sha256':ledgers,
            'results_sha256':sha(folder/'results.json'),'config_sha256':identity['config_sha256'],
            'model_or_dataset_fitting_performed':False,'Test_accessed':False}


def preflight(config_path,bundle_dir,*,device='cuda',runtime_policy=None,cuda_benchmark=None):
    """Source/fixed-file structural preflight; report arrays stay unopened."""
    config_path=Path(config_path).resolve();config=_validate_config(json.loads(config_path.read_text()))
    backend=ProductionData(config_path,bundle_dir or ROOT/config['data_output'],runtime_policy_path=runtime_policy)
    closure={n:sha(ROOT/n) for n in CODE_NAMES}
    files={}
    for fold in range(5):
        # Constructor metadata validates roles; report remains hash-only until its
        # later real summary/policy gate. No report label distribution inspection.
        for role in ['selection']:
            ds=backend.fixed(fold,role)
            if len(ds)!=config['fixed_examples'][role]:raise ValueError('Fixed role count differs')
        # Both numerical variants were prepared before recipe resolution. Stream
        # bytes for integrity without loading report arrays or targets.
        for name,digest in backend.manifest['file_sha256'].items():
            p=backend.bundle_dir/name
            if not p.is_file() or sha(p)!=digest:raise ValueError('Prepared artifact changed: '+name)
            files[name]=digest
    available=torch.cuda.is_available()
    cuda=None
    if device=='cuda' and available:
        prop=torch.cuda.get_device_properties(0)
        cuda={'device_name':prop.name,'device_uuid':str(getattr(prop,'uuid','unavailable')),'total_memory_bytes':prop.total_memory,
              'bf16_supported':torch.cuda.is_bf16_supported(),'torch_cuda':torch.version.cuda}
    benchmark_ok=False;benchmark_sha=None
    if cuda_benchmark is not None:
        b=json.loads(Path(cuda_benchmark).read_text());benchmark_sha=sha(cuda_benchmark)
        if (b.get('format')!='r13_actual_CUDA1epoch_benchmark_v1' or b.get('status')!='passed'
            or b.get('config_sha256')!=sha(config_path) or b.get('runtime_policy_sha256')!=backend.runtime_file_sha256
            or b.get('data_hash')!=backend.bundle_hash or b.get('code_sha256')!=closure
            or b.get('all_four_models_one_complete_epoch') is not True
            or [r['kind'] for r in b['model_results']]!=list(NEURAL)
            or any(r['rows']!=len(backend.train(0)) or r['batch_size']!=config['training']['batch_size'] for r in b['model_results'])):
            raise ValueError('Foreign/incomplete CUDA1epoch measurement receipt')
        if cuda is None or any(b['hardware'][k]!=cuda[k] for k in ('device_name','device_uuid','total_memory_bytes','torch_cuda')):
            raise ValueError('CUDA hardware changed after the measured full epoch')
        benchmark_ok=True
    approved=backend.runtime_policy is not None and backend.runtime_policy['production_recipe_approved'] is True
    ready=device=='cuda' and available and approved and benchmark_ok
    return {'format':'r13_training_preflight_v1','status':'passed' if ready else 'structural_only_launch_blocked',
        'ready_for_production_training':bool(ready),'device':device,'CUDA':cuda,'config_file_sha256':sha(config_path),
        'config_sha256':canonical(config),'code_sha256':closure,'data_bundle_hash':backend.bundle_hash,
        'runtime_policy_sha256':backend.runtime_file_sha256,'production_recipe_approved':bool(approved),
        'actual_CUDA1epoch_benchmark_sha256':benchmark_sha,'actual_CUDA1epoch_verified':benchmark_ok,
        'prepared_artifact_sha256':files,'report_opened':False,'model_fit_performed':False,
        'report_access_scope':'SHA byte streams only; report arrays/targets not loaded',
        'preflight_scope':'source/fixed-file integrity, explicit recipe approval and same-hardware full-CUDA1epoch receipt; absent receipt blocks launch',
        'reference_independent_cv':False,'all509_reference_pool_shared':True,'Test_or_submission_accessed':False}


def packaging_plan(config_path,bundle_dir=None,runtime_policy=None):
    config_path=Path(config_path).resolve();c=_validate_config(json.loads(config_path.read_text()))
    # R13Sources verifies these provenance bytes even though it does not open
    # their historical numeric arrays. A portable bundle must retain the exact
    # metadata files, not replace their pinned manifests with shortened copies.
    provenance={}
    for source in [c['background_version'],c['reference_version']]:
        m=json.loads((ROOT/source/'manifest.json').read_text())
        for name,digest in m['input_sha256'].items():
            if name in provenance and provenance[name]!=digest:raise ValueError('Conflicting source hash pins')
            if sha(ROOT/name)!=digest:raise ValueError('Source provenance changed: '+name)
            provenance[name]=digest
    prepared=Path(bundle_dir or ROOT/c['data_output']).resolve()
    from pipeline.full_data_r13 import validate_full_bundle
    bundle=validate_full_bundle(prepared)
    codes=set(CODE_NAMES)|set(bundle['code_sha256'])|{n for n in provenance if n.endswith('.py')}
    return {'format':'r13_training_packaging_plan_v1','config':str(config_path.relative_to(ROOT)),
        'config_file_sha256':sha(config_path),'code_files':sorted(codes),
        'code_file_sha256':{n:sha(ROOT/n) for n in sorted(codes)},
        'required_additional_provenance_file_sha256':provenance,
        'prepared_bundle':str(prepared.relative_to(ROOT)),'prepared_bundle_hash':bundle['data_hash'],
        'prepared_file_sha256':bundle['file_sha256'],'runtime_policy':str(runtime_policy) if runtime_policy else None,
        'runtime_policy_sha256':sha(runtime_policy) if runtime_policy else None,'background_version':c['background_version'],
        'CPU_recipe_evidence':json.loads(Path(runtime_policy).read_text())['CPU_evidence_file'] if runtime_policy else None,
        'DataLoader':{'workers':c['training']['num_workers'],'start_method':'spawn','prefetch_batches_per_worker':2,
            'CPU_parallel_speedup_measured':False,'CUDA_end_to_end_timing_measured':False},
        'reference_version':c['reference_version'],
        'required_banks':['data/processed/sa_windows_db.npy','data/processed/external_segments_power.npy'],
        'full_bank_scope':'whole immutable files retain original SHA; runtime statistical reads remain restricted to admitted R13 source indices',
        'dependency_policy':'install the project-pinned Python packages for the target Linux CUDA host; no Mac runtime copy',
        'required_packages':['numpy','pandas','pyarrow','torch','scikit-learn','lightgbm'],
        'excluded':['Test.csv','official submission files','old fitted checkpoints','raw WAV archives','credentials','SSH keys'],
        'fresh_output':c['run_output'],'GPU_default':'CUDA; production CPU100-epoch fitting rejected',
        'explicit_stages':['fit: all requested CUDA neural models, then exit awaiting Mac trees',
            'tree --executor mac_cpu: separate fresh Mac jobs; no transport or server fitting',
            'import-tree: completed receipt/artifact/source pins and selection prediction replay, no fit',
            'ensemble: all folds fit/gate policies then global immutable freeze',
            'report: all requested folds frozen before any report labels or predictions open'],
        'Mac_tree_parallelism':'needs source/config/prepared bundle, not CUDA model weights; may run while later CUDA folds fit',
        'Mac_transport':'explicit file transfer is a later operator action; this tool performs none',
        'tree_cpu':{'maximum_trees':1500,'ES_rounds':40,'features':5252,'maximum_training_rows':100000},
        'ensemble':'all singles plus equal-all/equal-neural on fit; one gate candidate; report after global freeze',
        'frozen_inference':'equal average of all five completed frozen fold policies, no query-cohort fitting; runner has no Test reader',
        'no_download_or_copy_executed':True,'no_Test_reader':True}


class _NumericDataset(Dataset):
    def __init__(self,role,n=10):
        self.role=role;self.n=n;self.epoch=1
        self.identity={'fixture_only':True,'role':role,'rows':n,'seed':91}
    def __len__(self):return self.n
    def set_epoch(self,epoch):
        if self.role!='train':raise AssertionError('Fixed role cannot advance epoch')
        self.epoch=epoch
    def raw_item(self,index):
        r=np.random.default_rng(91+index);b=r.uniform(.5,1.5,(15,25))
        a=np.exp(np.linspace(.5,-.5,25))[None,:]*np.maximum(0,np.sin(np.linspace(-.2,3.4,15)))[:,None]
        a=a*(.3+.03*self.epoch) if index%2 else np.zeros_like(a)
        mask=np.ones((15,25),bool)
        if index==0:mask[:3]=False
        b=np.where(mask,b,0);a=np.where(mask,a,0);e=a+b
        db=np.full((15,25),np.nan,dtype=np.float32);db[mask]=np.round(100+10*np.log10(e[mask]),1)
        q=np.divide(a,e,out=np.zeros_like(e),where=mask).astype('float32')
        y=np.float32(a.sum()/e.sum())
        return {'db':db,'share':y,'presence':np.float32(y>0),'cell_share':q,
            'site':'numeric_A' if index%3 else 'numeric_B','sample_id':self.role+'_'+str(index),
            'background_group':self.role+'_group'+str(index%3)}
    def raw_arrays(self):
        rows=[self.raw_item(i) for i in range(len(self))]
        return {k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    def __getitem__(self,index):
        from pipeline.relative_grid_r6_v2 import relative_grid_v2
        raw=self.raw_item(index);x,w=relative_grid_v2(raw['db'][None]);x=np.pad(x,((0,0),(0,3),(0,0),(0,0)))
        return {**{k:raw[k] for k in ['share','presence','cell_share','site','sample_id']},
                'x':x[0],'w':w[0],'loss_weight':np.float32(1)}


class _NumericBackend:
    def __init__(self,config_path):
        self.config_file_sha256=sha(config_path);self.bundle_hash='numeric-fixture-bundle'
        self.source_hash='numeric-fixture-source';self.report_open_count=0;self.roles=[]
    def train(self,fold):return _NumericDataset('train')
    def fixed(self,fold,role,*,report_gate=None):
        if role=='report':
            if report_gate is None:raise ValueError('Report requires a frozen model/policy gate')
            g=json.loads(Path(report_gate).read_text())
            if not g['all_models_frozen'] or not g['ensemble_policy_frozen']:raise ValueError('Incomplete freeze')
            for p,d in g['model_summary_files'].items():
                if sha(p)!=d:raise ValueError('Summary changed before report')
            if sha(g['ensemble_policy_file'])!=g['ensemble_policy_sha256']:raise ValueError('Policy changed before report')
            self.report_open_count+=1
        self.roles.append(role);return _NumericDataset(role)
    def model_settings(self,kind):return {}
    def make_tree(self,fold,config,fixture=False):
        from pipeline.trees_ensemble_r13 import PhysicsLGBM
        return PhysicsLGBM(config['tree'],seed=config['seed']+fold*10007,include_shape=False)
    def train_groups(self,fold):return ['train_group'+str(i) for i in range(3)]
    def report_gate(self,fold,summaries,policy,folder):
        global_gate=folder.parent/'all_requested_models_and_policies_frozen.json'
        if not global_gate.is_file():raise ValueError('Global freeze must precede fixture report access')
        g=json.loads(global_gate.read_text())
        assert g['all_requested_models_and_policies_frozen'] and not g['report_opened_before_this_gate']
        return {'fold':fold,'all_models_frozen':True,'ensemble_policy_frozen':True,
            'model_summary_files':{str(p):sha(p) for p in summaries.values()},
            'ensemble_policy_file':str(policy),'ensemble_policy_sha256':sha(policy),'data_bundle_hash':self.bundle_hash}


def self_test(output):
    """Actual one-step full-size Adam/tiny-tree fits on numeric arrays only; never source data."""
    output=Path(output)
    if output.exists():raise FileExistsError('Fresh verification report required')
    source_roots=[(ROOT/'data').resolve(),(ROOT/'data.nosync').resolve()]
    def no_source(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if any(p==r or r in p.parents for r in source_roots):raise PermissionError('Numeric fixture forbids all source dataset I/O')
    sys.addaudithook(no_source)
    start=time.perf_counter();temp=Path(tempfile.mkdtemp(prefix='r13-runner-numeric-',dir='/private/tmp'))
    c=json.loads((ROOT/'configs/r13_full_training_v1.json').read_text())
    c['training'].update(batch_size=10,num_workers=0,threads=1)
    c['tree'].update(maximum_trees=4,early_stopping_patience=2,threads=1,minimum_leaf_rows=2)
    cfg=temp/'numeric_config.json';atomic_json(cfg,c);backend=_NumericBackend(cfg)
    try:run_training(cfg,output=temp/'forbidden_production_CPU',device='cpu')
    except ValueError:production_cpu_rejected=True
    else:raise AssertionError('Production CPU neural training accepted')
    assert not (temp/'forbidden_production_CPU').exists()
    try:run_training(cfg,output=temp/'forbidden_production_subset',folds=[0],device='cuda',runtime_policy='unused')
    except ValueError:production_subset_rejected=True
    else:raise AssertionError('Production subset admitted before the fivefold global report barrier')
    assert not (temp/'forbidden_production_subset').exists()
    try:backend.fixed(0,'report')
    except ValueError:report_gate_rejected=True
    else:raise AssertionError('Ungated report opened')
    result=run_training(cfg,output=temp/'run',folds=[0,1],device='cpu',backend=backend,fixture=True)
    assert backend.report_open_count==2
    verification=verify_run(temp/'run',allow_fixture=True)
    assert result['status']=='completed' and result['fixture_only']
    for kind in NEURAL:
        model,s=restore_neural(temp/'run'/'fold_00'/'models'/kind,device='cpu',fixture=True)
        assert s['epochs_run']==1 and s['optimizer_steps']==1
        del model
    try:run_training(cfg,output=temp/'run',folds=[0,1],device='cpu',backend=backend,fixture=True)
    except FileExistsError:reuse_rejected=True
    else:raise AssertionError('Completed run reuse accepted')
    # Corrupt one already-ledgered prediction. Verification must reject without
    # reading source data or retraining. Restore original fixture bytes after it.
    p=temp/'run'/'fold_00'/'report_predictions.npz';before=p.read_bytes();p.write_bytes(before+b'changed')
    try:verify_run(temp/'run',allow_fixture=True)
    except ValueError:changed_artifact_rejected=True
    else:raise AssertionError('Changed completed artifact accepted')
    p.write_bytes(before);assert verify_run(temp/'run',allow_fixture=True)['status']=='verified'
    # A synthetic second report attempt must fail before a second report opens.
    before_count=backend.report_open_count
    try:evaluate_reports(temp/'run',config_path=cfg,backend=backend,device='cpu',fixture=True)
    except (ValueError,FileExistsError):duplicate_report_rejected=True
    else:raise AssertionError('One-shot report was repeated')
    assert backend.report_open_count==before_count
    assert verify_run(temp/'run',allow_fixture=True)['status']=='verified'
    report={'format':'r13_runner_numeric_fixture_v1','status':'passed','fixture_only':True,
        'code_sha256':{n:sha(ROOT/n) for n in CODE_NAMES if (ROOT/n).is_file()},
        'actual_Adam_steps_per_model_per_fold':1,'actual_neural_models':list(NEURAL),'actual_tree_max_rounds':4,
        'numeric_folds':[0,1],
        'completed_stage_order':list(json.loads((temp/'run'/'fold_00'/'artifact_ledger.json').read_text())['steps']),
        'report_opened_exactly_once_per_fold_after_global_policy_freeze':backend.report_open_count==2,
        'ungated_report_rejected':report_gate_rejected,'fresh_output_reuse_rejected':reuse_rejected,
        'production_CPU_neural_fit_rejected_before_output_creation':production_cpu_rejected,
        'production_subset_rejected_before_source_or_output_access':production_subset_rejected,
        'global_all_fold_model_and_policy_freeze_before_report':True,
        'explicit_separate_tree_job_and_import_replay':True,
        'duplicate_report_rejected_before_dataset_open':duplicate_report_rejected,
        'changed_ledger_artifact_rejected':changed_artifact_rejected,'verification':verification,
        'source_data_Test_OUTER_GPU_SSH_accessed':False,'production100_training_executed':False,
        'temporary_outputs_must_never_be_used_as_trained_models':str(temp),
        'elapsed_seconds':time.perf_counter()-start}
    atomic_json(output,report)
    print(json.dumps({'status':'passed','report':str(output),'sha256':sha(output),'seconds':report['elapsed_seconds']}))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for cmd in ['preflight','fit','package-plan','tree','import-tree','ensemble','report']:
        q=sub.add_parser(cmd);q.add_argument('--config',default='configs/r13_full_training_v1.json')
        q.add_argument('--bundle-dir',default='data/synthetic/group_cv_physics_r13_variants_v2')
        if cmd in ['preflight','fit','tree','package-plan']:q.add_argument('--runtime-policy')
        if cmd=='preflight':
            q.add_argument('--output',required=True);q.add_argument('--device',choices=['cpu','cuda'],default='cuda')
            q.add_argument('--cuda-benchmark')
        elif cmd=='fit':
            q.add_argument('--output');q.add_argument('--folds',nargs='+',type=int);q.add_argument('--preflight',required=True)
            q.add_argument('--cuda-benchmark',required=True)
        elif cmd=='tree':
            q.add_argument('--output',required=True);q.add_argument('--folds',nargs='+',type=int)
            q.add_argument('--executor',choices=['mac_cpu'],required=True)
        elif cmd=='import-tree':q.add_argument('--run-dir',required=True);q.add_argument('--jobs-dir',required=True)
        elif cmd in ['ensemble','report']:q.add_argument('--run-dir',required=True)
        elif cmd=='package-plan':q.add_argument('--output')
    q=sub.add_parser('verify');q.add_argument('run_dir');q.add_argument('--output');q.add_argument('--allow-fixture',action='store_true')
    q=sub.add_parser('self-test');q.add_argument('--output',default='reports/R13_RUNNER_VERIFICATION.json')
    q=sub.add_parser('freeze-runtime');q.add_argument('--config',default='configs/r13_full_training_v1.json')
    q.add_argument('--bundle-dir',default='data/synthetic/group_cv_physics_r13_variants_v2')
    q.add_argument('--cpu-evidence',required=True);q.add_argument('--output',required=True)
    q.add_argument('--variant',choices=['baseline','highshare'],required=True);q.add_argument('--shape',choices=[0,1],type=int,required=True)
    q=sub.add_parser('cuda-benchmark');q.add_argument('--config',default='configs/r13_full_training_v1.json')
    q.add_argument('--bundle-dir',default='data/synthetic/group_cv_physics_r13_variants_v2')
    q.add_argument('--runtime-policy',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('infer');q.add_argument('--run-dir',required=True);q.add_argument('--input',required=True)
    q.add_argument('--output',required=True);q.add_argument('--device',choices=['cpu','cuda'],default='cpu');q.add_argument('--batch-size',type=int,default=128)
    a=p.parse_args()
    if a.command=='self-test':self_test(a.output);return
    if a.command=='freeze-runtime':
        from pipeline.runtime_policy_r13 import freeze_runtime_policy
        result=freeze_runtime_policy(a.config,a.bundle_dir,a.cpu_evidence,a.output,
            data_variant=a.variant,include_shape=bool(a.shape),code_sha256={n:sha(ROOT/n) for n in CODE_NAMES},production_approved=False)
    elif a.command=='cuda-benchmark':
        from ops.benchmark_r13_cuda_epoch import benchmark
        result=benchmark(a.config,a.bundle_dir,a.runtime_policy,a.output)
    elif a.command=='infer':
        from pipeline.inference_r13 import infer_csv
        result=infer_csv(a.run_dir,a.input,a.output,device=a.device,batch_size=a.batch_size)
    elif a.command=='verify':
        result=verify_run(a.run_dir,allow_fixture=a.allow_fixture)
        if a.output:atomic_json(a.output,result)
    elif a.command=='package-plan':
        result=packaging_plan(a.config,a.bundle_dir,a.runtime_policy)
        if a.output:atomic_json(a.output,result)
    elif a.command=='preflight':
        result=preflight(a.config,a.bundle_dir,device=a.device,runtime_policy=a.runtime_policy,cuda_benchmark=a.cuda_benchmark);atomic_json(a.output,result)
    elif a.command=='tree':result=fit_mac_trees(a.config,bundle_dir=a.bundle_dir,output=a.output,folds=a.folds,runtime_policy=a.runtime_policy)
    elif a.command=='import-tree':result=import_mac_trees(a.run_dir,a.jobs_dir,config_path=a.config,bundle_dir=a.bundle_dir)
    elif a.command=='ensemble':result=freeze_ensembles(a.run_dir,config_path=a.config,bundle_dir=a.bundle_dir)
    elif a.command=='report':result=evaluate_reports(a.run_dir,config_path=a.config,bundle_dir=a.bundle_dir)
    else:
        before=json.loads(Path(a.preflight).read_text())
        if before.get('format')!='r13_training_preflight_v1' or before.get('ready_for_production_training') is not True:
            raise ValueError('Actual successful CUDA preflight required before production fit')
        now=preflight(a.config,a.bundle_dir,device='cuda',runtime_policy=a.runtime_policy,cuda_benchmark=a.cuda_benchmark)
        if before!=now:raise ValueError('Config/code/data/hardware changed after CUDA preflight')
        result=run_training(a.config,bundle_dir=a.bundle_dir,output=a.output,folds=a.folds,device='cuda',runtime_policy=a.runtime_policy,cuda_benchmark=a.cuda_benchmark)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
