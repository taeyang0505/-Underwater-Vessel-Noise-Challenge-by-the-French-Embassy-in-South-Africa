"""Fresh R9 background-CV orchestration; no challenge-query reader.

Production is CUDA-only for neural fitting. A separately labelled two-epoch
numeric fixture exercises the same artifact/ensemble/report plumbing on CPU.
Selection chooses checkpoints; fit/gate choose an ensemble; report opens last.
All509 reference recordings are shared across TRAIN folds, so this is not
reference-independent cross validation or an official-Test estimate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
NEURAL = ('bg_cnn', 'tfgrid_fdy_conformer', 'fdy', 'itransformer')
ROLES = ('selection', 'ensemble_fit', 'ensemble_gate', 'report')
CODE_NAMES = (
    'pipeline/runner_r9.py', 'ops/run_r9_training.py',
    'pipeline/dataset_r9.py', 'pipeline/sampler_r9.py',
    'pipeline/reference_pool_r9.py', 'pipeline/background_cv_r9.py',
    'pipeline/source_split_r6.py', 'pipeline/robust_sampler_r6.py',
    'pipeline/hard_negatives_r6_v2.py',
    'pipeline/models_r9.py', 'pipeline/training_r9.py',
    'pipeline/trees_ensemble_r9.py', 'pipeline/cpu_ablation_r6_v2.py',
    'pipeline/relative_grid_r6.py', 'pipeline/relative_grid_r6_v2.py',
    'pipeline/physics_features_r6_v2.py', 'pipeline/relative_models_r6_v2.py',
    'pipeline/hybrid_models_r5.py', 'pipeline/grid_models_r5.py', 'pipeline/models.py',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''): h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def _plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _identity(dataset):
    value = dataset.identity() if callable(dataset.identity) else dataset.identity
    if not isinstance(value, dict): raise ValueError('Explicit dataset identity dictionary required')
    return _plain(value)


def _data_unchanged(backend):
    if hasattr(backend,'assert_unchanged'):backend.assert_unchanged()


def _regular_files(folder):
    paths = []
    for p in sorted(Path(folder).rglob('*')):
        if p.is_symlink(): raise ValueError('Symlink artifacts are forbidden: '+str(p))
        if p.is_file() and p.suffix != '.tmp': paths.append(p)
    return paths


def _check_code(code, root=ROOT):
    for name, digest in code.items():
        if sha(root/name) != digest: raise ValueError('Code changed during run: '+name)


def _ledger(folder, steps, name, files, *, code, binding):
    _check_code(code)
    if name in steps: raise ValueError('Completed stage cannot execute twice')
    artifacts = {}
    for p in files:
        p = Path(p)
        if p.is_symlink() or not p.is_file(): raise ValueError('Regular completed artifact required')
        rel = p.relative_to(folder).as_posix()
        if rel in {'artifact_ledger.json', 'progress.json', 'FAILED.json'}: continue
        artifacts[rel] = sha(p)
    steps[name] = {'status': 'completed', 'artifact_sha256': artifacts,
                   'completed_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(folder/'artifact_ledger.json', {
        'format': 'r9_fold_artifact_ledger_v1', 'binding': binding,
        'code_sha256': code, 'steps': steps, 'partial_reuse_allowed': False})
    atomic_json(folder/'progress.json', {'status': 'running', 'last_completed_stage': name,
                                      'completed_stages': list(steps), 'fold': binding['fold']})


def _validate_config(config):
    if config.get('version') != 'r9_full_training_v1': raise ValueError('Explicit R9 config required')
    if config.get('folds') != list(range(5)): raise ValueError('Production config must declare all5 background folds')
    if config.get('model_order') != [*NEURAL, 'lgbm_physics']: raise ValueError('R9 five-member policy differs')
    if config.get('fixed_examples') != dict(selection=1000, ensemble_fit=500, ensemble_gate=500, report=1000):
        raise ValueError('Fixed selection/fit/gate/report counts differ')
    t = config['training']
    if (t['epochs'] != 100 or t['early_stopping'] is not False or t['optimizer'].lower() != 'adam'
            or t['dropout'] != .1 or t['monitor'] != 'window_rmse'):
        raise ValueError('R9 production requires100/noES/Adam/dropout.1/window checkpoint selection')
    g = config['generator']
    if (g['minute_first_fraction'] != 1. or g['time_warp'] is not False
            or g['quantization_decimals'] != 1 or g['parameters_from_Test_statistics'] is not False):
        raise ValueError('Predeclared minute-first/no-Test-parameter generator required')
    if config['ensemble']['epsilon_rmse'] != .0001 or config['ensemble']['report_role_for_selection'] is not False:
        raise ValueError('Fixed ensemble gate and untouched report required')
    if (config['ensemble']['candidates']!=['every single model','all-model equal average','neural-only equal average']
            or config.get('inference',{}).get('fold_aggregation')!='equal arithmetic mean over all five complete fold policies'
            or config.get('inference',{}).get('Test_fitting_or_calibration') is not False
            or config.get('execution',{}).get('tree_executor')!='Mac CPU in separately verified stage; no implicit server tree fitting'):
        raise ValueError('Explicit finite candidates, Mac tree executor and equal-fold inference policy required')
    if config.get('official_submission') is not False: raise ValueError('Runner never submits predictions')
    return config


def _settings(config, fold, *, fixture=False):
    t = config['training']
    return dict(epochs=2 if fixture else 100, fixture_only=fixture,
                optimizer='adam', early_stopping_enabled=False,
                learning_rate=t['learning_rate'], weight_decay=t['weight_decay'],
                gradient_clip=t['gradient_clip'], cell_loss_weight=t['cell_loss_weight'],
                presence_loss_weight=t['presence_loss_weight'], batch_size=t['batch_size'],
                evaluation_batch_size=t.get('evaluation_batch_size', t['batch_size']),
                num_workers=t['num_workers'], threads=t['threads'], seed=config['seed']+fold*10007,
                bf16=not fixture, monitor='window_rmse', site_balanced_loss=t['site_balanced_loss'])


def _raw(dataset):
    """Materialize only the explicitly supplied role; never opens another role."""
    if hasattr(dataset, 'raw_arrays'):
        data = dataset.raw_arrays()
    else:
        # Do not retain tens of thousands of recipe dictionaries for tree fit.
        # Numeric arrays are preallocated; string identities stay small lists.
        first = dataset.raw_item(0);n=len(dataset)
        data={'db':np.empty((n,15,25),np.float32),'share':np.empty(n,np.float32),
              'presence':np.empty(n,np.float32),'site':[],'sample_id':[],'background_group':[]}
        if 'cell_share' in first:data['cell_share']=np.empty((n,15,25),np.float32)
        for i in range(n):
            row=first if i==0 else dataset.raw_item(i)
            for key in data:
                if key in ['site','sample_id','background_group']:data[key].append(str(row[key]))
                else:data[key][i]=row[key]
    required = {'db', 'share', 'presence', 'sample_id', 'site', 'background_group'}
    if not required <= set(data): raise ValueError('Raw role API missing: '+str(required-set(data)))
    n = len(dataset)
    out = {k: np.asarray(v) for k, v in data.items()}
    if out['db'].shape != (n,15,25) or any(out[k].shape != (n,) for k in required-{'db'}):
        raise ValueError('Role raw arrays/order schema differs')
    if any(v.dtype.kind == 'O' for v in out.values()): raise ValueError('No-pickle numeric/Unicode arrays required')
    if len(set(out['sample_id'].astype(str))) != n: raise ValueError('Unique sample IDs required')
    if not np.isfinite(out['share']).all() or not np.isfinite(out['presence']).all(): raise ValueError('Finite known labels required')
    return out


def _save_predictions(path, pred, raw):
    keep = {k: np.asarray(pred[k]) for k in ['share','score','cell_mask'] if k in pred}
    keep.update(sample_id=raw['sample_id'].astype(str), background_group=raw['background_group'].astype(str))
    for k, v in keep.items():
        if v.dtype.kind == 'O': raise ValueError('Object predictions forbidden')
        if v.dtype.kind in 'fc' and not np.isfinite(v).all(): raise FloatingPointError('Nonfinite '+k)
        if len(v) != len(raw['share']): raise ValueError('Prediction row count differs')
    if not {'share','score'} <= set(keep): raise ValueError('Share and diagnostic score required')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream: np.savez_compressed(stream, **keep)
    tmp.replace(path)


def _metrics(pred, raw):
    from pipeline.training_r9 import aggregate_metrics
    return aggregate_metrics(dict(share=pred['share'], score=pred['score'],
        target_share=raw['share'], target_presence=raw['presence'], site=raw['site'],
        site_metadata_available=True))


def restore_neural(folder, *, device='cuda', fixture=False, expected_identity=None):
    """Restore exactly the completed strict best-window checkpoint, no resize."""
    from pipeline.models_r9 import make_r9_model
    from pipeline.training_r9 import CODE_NAMES as engine_code
    folder = Path(folder); summary = json.loads((folder/'summary.json').read_text())
    if (summary.get('status') != 'completed' or (folder/'FAILED.json').exists()
            or summary.get('fixture_only') != fixture or summary.get('epochs_run') != (2 if fixture else 100)):
        raise ValueError('Only completed, correctly labelled fixed-epoch models may restore')
    for name, digest in summary['artifact_sha256'].items():
        p = folder/name
        if p.is_symlink() or not p.is_file() or sha(p) != digest: raise ValueError('Checkpoint artifact changed: '+name)
    if set(summary['code_sha256']) != set(engine_code): raise ValueError('Neural code closure differs')
    _check_code(summary['code_sha256'])
    if expected_identity is not None and summary['validation_identity'] != expected_identity:
        raise ValueError('Selection data identity differs from saved checkpoint')
    saved = torch.load(folder/'best_window.pt', map_location='cpu', weights_only=True)
    for key in ['kind','settings','architecture','input_fingerprint','train_identity','validation_identity','code_sha256','fixture_only']:
        if saved[key] != summary[key]: raise ValueError('Checkpoint/summary binding differs: '+key)
    if saved['epoch'] != summary['best_window_epoch']: raise ValueError('Best-window epoch differs')
    model = make_r9_model(saved['kind'], **saved['architecture']['model_settings'])
    if model.architecture() != saved['architecture']: raise ValueError('Architecture reconstruction differs')
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval(), summary


def _predict_nn(folder, dataset, raw, *, settings, device, fixture):
    from pipeline.training_r9 import predict_loader
    model, _ = restore_neural(folder, device=device, fixture=fixture)
    loader = DataLoader(dataset, batch_size=settings['evaluation_batch_size'], shuffle=False,
                        num_workers=0, pin_memory=str(device).startswith('cuda'))
    pred = predict_loader(model, loader, device, bf16=settings['bf16'], collect_targets=False)
    if len(pred['share']) != len(raw['share']): raise ValueError('Prediction coverage differs')
    del model; gc.collect()
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    return pred


def _binding(fold,config,backend,train,selection,fixture):
    return {'fold':fold,'config_sha256':canonical(config),'config_file_sha256':backend.config_file_sha256,
        'data_bundle_hash':backend.bundle_hash,'source_hash':backend.source_hash,
        'train_identity':_identity(train),'selection_identity':_identity(selection),
        'reference_independent_cv':False,'all509_references_shared':not fixture,'fixture_only':fixture}


def _create_context(folder,fold,config,backend,code,fixture):
    folder.mkdir();train=backend.train(fold);selection=backend.fixed(fold,'selection')
    if not fixture and len(selection)!=config['fixed_examples']['selection']:raise ValueError('Fixed selection count differs')
    binding=_binding(fold,config,backend,train,selection,fixture)
    atomic_json(folder/'context.json',{'format':'r9_fold_context_v1','binding':binding,'code_sha256':code,'config':config})
    _ledger(folder,{},'context',[folder/'context.json'],code=code,binding=binding)
    return binding


def _verify_ledger(folder,code,*,required=None):
    value=json.loads((folder/'artifact_ledger.json').read_text())
    if (folder/'FAILED.json').exists() or value['code_sha256']!=code:raise ValueError('Failed fold or changed code closure')
    if required and not set(required)<=set(value['steps']):raise ValueError('Required completed stages are missing')
    for stage in value['steps'].values():
        if stage.get('status')!='completed':raise ValueError('Incomplete stage')
        for name,digest in stage['artifact_sha256'].items():
            rel=Path(name);p=folder/rel
            if rel.is_absolute() or '..' in rel.parts or p.is_symlink() or not p.is_file() or sha(p)!=digest:
                raise ValueError('Completed artifact changed: '+name)
    _check_code(code)
    return value


def _nn_fold(folder,fold,config,backend,code,*,device,fixture):
    from pipeline.training_r9 import train_neural
    ledger=_verify_ledger(folder,code,required=['context']);binding=ledger['binding'];steps=ledger['steps']
    train=backend.train(fold);selection=backend.fixed(fold,'selection')
    if _binding(fold,config,backend,train,selection,fixture)!=binding:raise ValueError('Source/selection identity changed')
    settings=_settings(config,fold,fixture=fixture)
    for kind in NEURAL:
        result=train_neural(kind,train,selection,dict(settings,model_settings=backend.model_settings(kind)),
                            folder/'models'/kind,device=device)
        if result['epochs_run']!=(2 if fixture else 100):raise ValueError('Incomplete fixed neural epochs')
        _ledger(folder,steps,'train/'+kind,_regular_files(folder/'models'/kind),code=code,binding=binding)
        gc.collect()
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    atomic_json(folder/'neural_completed.json',{'status':'completed','fold':fold,'binding':binding,
        'tree_executor':'separate Mac CPU job; no tree fit on CUDA execution host',
        'model_summary_sha256':{k:sha(folder/'models'/k/'summary.json') for k in NEURAL}})
    _ledger(folder,steps,'neural_completed',[folder/'neural_completed.json'],code=code,binding=binding)


def _tree_job(folder,fold,config,backend,code,*,fixture=False):
    """Independent fresh Mac CPU job; no NN checkpoint is needed or copied."""
    from pipeline.trees_ensemble_r9 import PhysicsLGBM
    if not fixture and platform.system()!='Darwin':raise ValueError('Explicit mac_cpu executor requires macOS; server tree fit is forbidden')
    folder.mkdir();train=backend.train(fold);selection=backend.fixed(fold,'selection');train.set_epoch(1)
    if len(train)>config['tree']['maximum_training_rows']:raise ValueError('Full epoch1 exceeds cap; never silently subsample')
    binding=_binding(fold,config,backend,train,selection,fixture)
    atomic_json(folder/'context.json',{'format':'r9_mac_tree_context_v1','binding':binding,'code_sha256':code})
    tr=_raw(train);sr=_raw(selection)
    if set(tr['background_group'].astype(str))&set(sr['background_group'].astype(str)):
        raise ValueError('TRAIN/selection groups overlap')
    np.savez_compressed(folder/'tree_epoch1_train.npz',**{k:tr[k] for k in ['db','share','presence','sample_id']})
    atomic_json(folder/'tree_epoch1_manifest.json',{'epoch':1,'rows':len(train),'identity':_identity(train),
        'npz_sha256':sha(folder/'tree_epoch1_train.npz'),'no_automatic_subsampling':True,
        'order':'deterministic epoch1 schedule-index order; each eligible primary background once'})
    tree=PhysicsLGBM(config['tree'],seed=_settings(config,fold,fixture=fixture)['seed'])
    tree.fit(dict(tr,role='train'),dict(sr,role='selection'),folder/'models'/'lgbm_physics',binding,fixture=fixture)
    files=_regular_files(folder)
    receipt={'format':'r9_mac_tree_job_v1','status':'completed','executor':'numeric_fixture_cpu' if fixture else 'mac_cpu',
        'platform_system':platform.system(),'fixture_only':fixture,'fold':fold,'binding':binding,'code_sha256':code,
        'producer_environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
                                'lightgbm':tree.summary['lightgbm_version']},
        'artifact_sha256':{p.relative_to(folder).as_posix():sha(p) for p in files},
        'selection_only_for_tree_early_stopping':True,'report_or_Test_accessed':False,'transport_performed':False}
    receipt['receipt_sha256']=canonical(receipt);atomic_json(folder/'receipt.json',receipt)
    return receipt


def _import_tree(folder,job,code,backend,*,fixture=False):
    """Hash/replay verified completed Mac tree only. This function never fits."""
    from pipeline.trees_ensemble_r9 import PhysicsLGBM
    import shutil
    ledger=_verify_ledger(folder,code,required=['neural_completed']);binding=ledger['binding'];steps=ledger['steps']
    if 'train/lgbm_physics' in steps:raise ValueError('Completed tree import cannot execute twice')
    receipt=json.loads((job/'receipt.json').read_text())
    if (receipt.get('format')!='r9_mac_tree_job_v1' or receipt.get('status')!='completed'
            or receipt.get('fixture_only')!=fixture or receipt.get('binding')!=binding
            or receipt.get('code_sha256')!=code or (job/'FAILED.json').exists()
            or canonical({k:v for k,v in receipt.items() if k!='receipt_sha256'})!=receipt['receipt_sha256']):
        raise ValueError('Completed Mac tree receipt/source binding differs')
    if not fixture and (receipt['executor']!='mac_cpu' or receipt['platform_system']!='Darwin'):
        raise ValueError('Production tree must have explicit Mac CPU provenance')
    for name,digest in receipt['artifact_sha256'].items():
        rel=Path(name);p=job/rel
        if rel.is_absolute() or '..' in rel.parts or p.is_symlink() or not p.is_file() or sha(p)!=digest:
            raise ValueError('Mac job artifact changed: '+name)
    actual={p.relative_to(job).as_posix() for p in _regular_files(job)}
    if actual!=set(receipt['artifact_sha256'])|{'receipt.json'}:raise ValueError('Untracked Mac job artifacts')
    tree=PhysicsLGBM.load(job/'models'/'lgbm_physics',expected_binding=binding,allow_fixture=fixture)
    selection=backend.fixed(binding['fold'],'selection');raw=_raw(selection)
    if _identity(selection)!=binding['selection_identity']:raise ValueError('Selection source identity differs')
    pred=tree.predict(raw['db']);replay={}
    with np.load(job/'models'/'lgbm_physics'/'inner_predictions.npz',allow_pickle=False) as saved:
        for key in ['share','score','score_classifier']:
            delta=float(np.max(np.abs(pred[key].astype(float)-saved[key].astype(float))))
            if delta>1e-6:raise ValueError('Mac tree selection replay differs: '+key)
            replay[key]=delta
    # No write to original Mac job; copy into a fresh target subtree only.
    target=folder/'models'/'lgbm_physics'
    if target.exists() or (folder/'mac_tree_job').exists():raise FileExistsError('Fresh tree import paths required')
    shutil.copytree(job/'models'/'lgbm_physics',target)
    evidence=folder/'mac_tree_job';evidence.mkdir()
    for name in ['receipt.json','context.json','tree_epoch1_manifest.json']:
        shutil.copyfile(job/name,evidence/name)
    atomic_json(evidence/'import_receipt.json',{'source_receipt_file_sha256':sha(job/'receipt.json'),
        'source_receipt_payload_sha256':receipt['receipt_sha256'],'selection_replay_max_absolute_delta':replay,
        'replay_atol':1e-6,'fit_performed':False,'executor':'mac_cpu' if not fixture else 'numeric_fixture_cpu',
        'source_tree_epoch1_npz_sha256':receipt['artifact_sha256']['tree_epoch1_train.npz']})
    _ledger(folder,steps,'train/lgbm_physics',_regular_files(target)+_regular_files(evidence),code=code,binding=binding)


def _summary_paths(folder):return {k:folder/'models'/k/'summary.json' for k in [*NEURAL,'lgbm_physics']}


def _ensemble_fold(folder,fold,config,backend,code,*,device,fixture):
    from pipeline.trees_ensemble_r9 import PhysicsLGBM,fit_safe_ensemble,freeze_ensemble
    ledger=_verify_ledger(folder,code,required=['neural_completed','train/lgbm_physics']);binding=ledger['binding'];steps=ledger['steps']
    if 'ensemble_frozen' in steps:raise ValueError('Frozen ensemble cannot be selected twice')
    summaries=_summary_paths(folder);settings=_settings(config,fold,fixture=fixture)
    tree=PhysicsLGBM.load(folder/'models'/'lgbm_physics',expected_binding=binding,allow_fixture=fixture)
    selection_raw=_raw(backend.fixed(fold,'selection'))
    # The train source-group list is metadata, not labels or predictor features.
    train_groups=set(backend.train_groups(fold))
    known=train_groups|set(selection_raw['background_group'].astype(str))
    role_raw={};role_pred={};identities={}
    for role in ['ensemble_fit','ensemble_gate']:
        ds=backend.fixed(fold,role);raw=_raw(ds)
        if not fixture and len(ds)!=config['fixed_examples'][role]:raise ValueError('Fixed role count differs: '+role)
        if known&set(raw['background_group'].astype(str)):raise ValueError('Training/selection and ensemble groups overlap')
        identities[role]=_identity(ds);role_raw[role]=raw;role_pred[role]={}
        for kind in NEURAL:
            pred=_predict_nn(folder/'models'/kind,ds,raw,settings=settings,device=device,fixture=fixture)
            role_pred[role][kind]=pred;_save_predictions(folder/'predictions'/role/(kind+'.npz'),pred,raw)
        pred=tree.predict(raw['db']);role_pred[role]['lgbm_physics']=pred
        _save_predictions(folder/'predictions'/role/'lgbm_physics.npz',pred,raw)
    nf,ng=(len(role_raw[r]['share']) for r in ['ensemble_fit','ensemble_gate'])
    joined={k:{v:np.concatenate([role_pred[r][k][v] for r in ['ensemble_fit','ensemble_gate']])
               for v in ['share','score']} for k in [*NEURAL,'lgbm_physics']}
    groups=np.concatenate([role_raw[r]['background_group'].astype(str) for r in ['ensemble_fit','ensemble_gate']])
    y=np.concatenate([role_raw[r]['share'] for r in ['ensemble_fit','ensemble_gate']])
    policy=fit_safe_ensemble(y,joined,groups,fit_indices=np.arange(nf),gate_indices=np.arange(nf,nf+ng),
                             epsilon_rmse=config['ensemble']['epsilon_rmse'])
    policy.update(training_binding=binding,frozen_before_report=True,
        model_summary_sha256={k:sha(p) for k,p in summaries.items()},role_identities=identities)
    policy['policy_sha256']=canonical({k:v for k,v in policy.items() if k!='policy_sha256'})
    freeze_ensemble(policy,folder/'ensemble_policy.json')
    atomic_json(folder/'all_models_frozen.json',{'all_models_frozen':True,'fold':fold,'binding':binding,
        'model_summary_sha256':{k:sha(p) for k,p in summaries.items()}})
    _ledger(folder,steps,'ensemble_frozen',_regular_files(folder/'predictions')+
        [folder/'all_models_frozen.json',folder/'ensemble_policy.json'],code=code,binding=binding)


def _freeze_global(run,folds,code):
    models={};policies={};ledgers={}
    for fold in folds:
        folder=run/f'fold_{fold:02d}';_verify_ledger(folder,code,required=['ensemble_frozen'])
        for p in _regular_files(folder/'models'):models[p.relative_to(run).as_posix()]=sha(p)
        policies[(folder/'ensemble_policy.json').relative_to(run).as_posix()]=sha(folder/'ensemble_policy.json')
        ledgers[str(fold)]=sha(folder/'artifact_ledger.json')
    value={'format':'r9_global_models_and_policies_frozen_v1','all_requested_models_and_policies_frozen':True,
        'folds':folds,'model_artifact_sha256':models,'ensemble_policy_sha256':policies,
        'prefit_fold_ledger_sha256':ledgers,'report_opened_before_this_gate':False,'code_sha256':code}
    path=run/'all_requested_models_and_policies_frozen.json'
    if path.exists():raise FileExistsError('Global freeze is immutable')
    atomic_json(path,value);return value


def _verify_global(run,folds,code):
    path=run/'all_requested_models_and_policies_frozen.json';g=json.loads(path.read_text())
    if (g.get('format')!='r9_global_models_and_policies_frozen_v1'
            or g.get('all_requested_models_and_policies_frozen') is not True or g['folds']!=folds
            or g['code_sha256']!=code):raise ValueError('All requested folds must freeze before any report opens')
    for key in ['model_artifact_sha256','ensemble_policy_sha256']:
        for name,digest in g[key].items():
            p=run/name
            if p.is_symlink() or not p.is_file() or sha(p)!=digest:raise ValueError('Global model/policy artifact changed')
    _check_code(code);return g


def _report_fold(run,folder,fold,folds,config,backend,code,*,device,fixture):
    from pipeline.trees_ensemble_r9 import PhysicsLGBM,apply_frozen_ensemble
    _verify_global(run,folds,code)
    ledger=_verify_ledger(folder,code,required=['ensemble_frozen']);binding=ledger['binding'];steps=ledger['steps']
    if 'report_completed' in steps:raise ValueError('Report is one-shot and cannot be evaluated twice')
    summaries=_summary_paths(folder);policy_path=folder/'ensemble_policy.json';policy=json.loads(policy_path.read_text())
    gate=backend.report_gate(fold,summaries,policy_path,folder)
    gate_path=folder/'report_access_gate.json';atomic_json(gate_path,gate)
    report=backend.fixed(fold,'report',report_gate=gate_path);rr=_raw(report);rp={}
    if not fixture and len(report)!=config['fixed_examples']['report']:raise ValueError('Fixed report count differs')
    previous=set(backend.train_groups(fold))
    for role in ['selection','ensemble_fit','ensemble_gate']:
        previous|=set(_raw(backend.fixed(fold,role))['background_group'].astype(str))
    if previous&set(rr['background_group'].astype(str)):raise ValueError('Report source group overlaps a selection/fitting role')
    settings=_settings(config,fold,fixture=fixture)
    for kind in NEURAL:
        p=_predict_nn(folder/'models'/kind,report,rr,settings=settings,device=device,fixture=fixture)
        rp[kind]=p;_save_predictions(folder/'predictions'/'report'/(kind+'.npz'),p,rr)
    tree=PhysicsLGBM.load(folder/'models'/'lgbm_physics',expected_binding=binding,allow_fixture=fixture)
    rp['lgbm_physics']=tree.predict(rr['db'])
    _save_predictions(folder/'predictions'/'report'/'lgbm_physics.npz',rp['lgbm_physics'],rr)
    ep=apply_frozen_ensemble(policy,rp);_save_predictions(folder/'report_predictions.npz',ep,rr)
    np.savez_compressed(folder/'report_known_targets.npz',**{k:rr[k] for k in ['share','presence','site','sample_id']})
    metrics={k:_metrics(v,rr) for k,v in rp.items()};metrics['frozen_ensemble']=_metrics(ep,rr)
    atomic_json(folder/'report_metrics.json',metrics)
    result={'status':'completed','fold':fold,'binding':binding,'fixture_only':fixture,
        'model_summary_files':{k:str(p.relative_to(folder)) for k,p in summaries.items()},
        'ensemble_policy_sha256':sha(policy_path),'report_metrics':metrics,'report_identity':_identity(report),
        'global_freeze_file_sha256':sha(run/'all_requested_models_and_policies_frozen.json'),
        'report_used_for_selection':False,'reference_independent_cv':False,'no_Test_or_official_submission':True}
    atomic_json(folder/'summary.json',result)
    _ledger(folder,steps,'report_completed',[gate_path,folder/'report_predictions.npz',folder/'report_known_targets.npz',
        folder/'report_metrics.json',folder/'summary.json']+_regular_files(folder/'predictions'/'report'),code=code,binding=binding)
    atomic_json(folder/'progress.json',{'status':'completed','fold':fold,'completed_stages':list(steps)})
    return result


class ProductionData:
    def __init__(self,config_path,bundle_dir,*,verify_banks=True):
        from pipeline.dataset_r9 import R9Sources,validate_bundle
        self.config_path=Path(config_path).resolve();self.bundle_dir=Path(bundle_dir).resolve()
        self.config_file_sha256=sha(self.config_path)
        self.sources=R9Sources(config_path=self.config_path,project_root=ROOT,verify_banks=verify_banks)
        manifest=validate_bundle(self.bundle_dir,project_root=ROOT,verify_all=True)
        ready=json.loads((self.bundle_dir/'readiness.json').read_text())
        if (ready.get('ready_to_train') is not True or ready.get('manifest_sha256')!=sha(self.bundle_dir/'manifest.json')
                or ready.get('source_hash')!=manifest.get('source_hash')):
            raise ValueError('Prepared dataset readiness/source/manifest gate differs')
        if manifest.get('version')!='r9_minute_full_dataset_v1':raise ValueError('Prepared R9 dataset version differs')
        if manifest.get('config_sha256')!=self.config_file_sha256:raise ValueError('Prepared config file SHA differs')
        if (manifest['background_source_hash']!=self.sources.bg_manifest['source_hash']
                or manifest['reference_source_hash']!=self.sources.ref_manifest['source_hash']):
            raise ValueError('Prepared source version differs from dynamic TRAIN sources')
        for name,digest in manifest['file_sha256'].items():
            rel=Path(name);p=self.bundle_dir/rel
            if rel.is_absolute() or '..' in rel.parts or p.is_symlink() or not p.is_file() or sha(p)!=digest:
                raise ValueError('Prepared fixed artifact changed: '+name)
        self.bundle_hash=manifest['source_hash'];self.source_hash=self.sources.source_hash
        self.manifest=manifest
    def train(self,fold):
        from pipeline.dataset_r9 import DynamicR9Dataset
        return DynamicR9Dataset(self.config_path,fold,epoch=1,sources=self.sources,verify_banks=False)
    def fixed(self,fold,role,*,report_gate=None):
        from pipeline.dataset_r9 import FixedR9Dataset
        return FixedR9Dataset(self.bundle_dir,fold,role,report_gate=report_gate)
    def model_settings(self,kind):return {'dropout':.1}
    def assert_unchanged(self):
        from pipeline.dataset_r9 import validate_bundle
        self.sources.assert_unchanged(verify_banks=True)
        manifest=validate_bundle(self.bundle_dir,project_root=ROOT,verify_all=True)
        if manifest['source_hash']!=self.bundle_hash:raise ValueError('Prepared source changed during execution')
    def train_groups(self,fold):
        import pandas as pd
        config=json.loads(self.config_path.read_text())
        meta=pd.read_parquet(ROOT/config['background_version']/f'fold_{fold:02d}'/'train_manifest.parquet',columns=['background_group'])
        return sorted(set(meta['background_group'].astype(str)))
    def report_gate(self,fold,summaries,policy,folder):
        return {'fold':fold,'all_models_frozen':True,'ensemble_policy_frozen':True,
            'model_summary_files':{str(p.relative_to(ROOT)):sha(p) for p in summaries.values()},
            'ensemble_policy_file':str(policy.relative_to(ROOT)),
            'ensemble_policy_sha256':sha(policy),'data_bundle_hash':self.bundle_hash}


def _folds(value):
    if (not value or any(type(v) is not int or v not in range(5) for v in value)
            or len(set(value))!=len(value)):raise ValueError('Explicit unique folds0..4 required')
    return value


def _stage_load(run,config_path,bundle_dir,backend,fixture):
    if not fixture and backend is not None:raise ValueError('Production dataset backend injection is forbidden')
    run=Path(run).resolve();identity=json.loads((run/'execution_identity.json').read_text())
    if (run/'FAILED.json').exists() or identity['fixture_only']!=fixture:raise ValueError('Failed/fixture run cannot continue as production')
    cfg=Path(config_path or identity['config_path']).resolve();config=_validate_config(json.loads(cfg.read_text()))
    if sha(cfg)!=identity['config_file_sha256'] or canonical(config)!=identity['config_sha256']:
        raise ValueError('Run config changed between explicit stages')
    _check_code(identity['code_sha256'])
    if backend is None:backend=ProductionData(cfg,bundle_dir or ROOT/config['data_output'])
    if backend.bundle_hash!=identity['data_bundle_hash'] or backend.source_hash!=identity['source_hash']:
        raise ValueError('Prepared data/source identity changed between stages')
    return run,identity,config,backend


class _Stage:
    """One writer per healthy run; failed or interrupted output is never resumed."""
    def __init__(self,run,name,*,cuda=False,fixture=False):
        self.run=Path(run);self.name=name;self.cuda=cuda;self.fixture=fixture;self.handle=None
    def __enter__(self):
        if (self.run/'FAILED.json').exists():raise ValueError('Failed run reuse forbidden')
        if self.cuda and not self.fixture:
            import fcntl
            self.handle=(ROOT/'models'/'.r9_cuda_writer.lock').open('a+')
            try:fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                self.handle.close();raise RuntimeError('Another R9 CUDA stage is active')
        try:
            with (self.run/'RUNNING.lock').open('x') as stream:
                stream.write(json.dumps({'pid':os.getpid(),'stage':self.name})+'\n')
        except BaseException:
            if self.handle:self.handle.close()
            raise
        atomic_json(self.run/'run_state.json',{'status':'running','stage':self.name})
        return self
    def __exit__(self,kind,error,tb):
        if error is not None:
            atomic_json(self.run/'FAILED.json',{'status':'failed','stage':self.name,'error_type':kind.__name__,
                'error':str(error),'partial_reuse_allowed':False,'created_utc':datetime.now(timezone.utc).isoformat()})
            atomic_json(self.run/'run_state.json',{'status':'failed','stage':self.name,'partial_reuse_allowed':False})
        (self.run/'RUNNING.lock').unlink(missing_ok=True)
        if self.handle:self.handle.close()


def run_training(config_path='configs/r9_full_training_v1.json',*,bundle_dir=None,output=None,
                 folds=None,device='cuda',backend=None,fixture=False):
    """Stage1: all requested CUDA NN fits, then stop for explicit Mac tree jobs.

    No production invocation calls LightGBM.fit. Only the synthetic self-test
    continues through the remaining stages on its numeric CPU backend.
    """
    start=time.perf_counter();config_path=Path(config_path).resolve()
    config=_validate_config(json.loads(config_path.read_text()))
    selected=_folds(config['folds'] if folds is None else folds)
    if fixture:
        if backend is None or device!='cpu':raise ValueError('Numeric fixture requires explicit synthetic backend and CPU')
    elif backend is not None or device!='cuda' or not torch.cuda.is_available():
        raise ValueError('Production neural execution requires available CUDA and native dataset backend')
    out=Path(output or ROOT/config['run_output']).resolve()
    if not fixture and (ROOT/'models').resolve() not in out.parents:raise ValueError('Production output must be a new project models directory')
    if out.exists():raise FileExistsError('Fresh output required; completed/failed/partial reuse is forbidden')
    code={name:sha(ROOT/name) for name in CODE_NAMES if (ROOT/name).is_file()}
    if not fixture and set(code)!=set(CODE_NAMES):raise ValueError('Production source closure incomplete')
    if backend is None:backend=ProductionData(config_path,bundle_dir or ROOT/config['data_output'])
    out.mkdir(parents=True)
    identity={'format':'r9_execution_identity_v1','config':config,'config_sha256':canonical(config),
        'config_file_sha256':sha(config_path),'config_path':str(config_path),
        'code_sha256':code,'folds':selected,'device':device,
        'fixture_only':fixture,'pid':os.getpid(),'created_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
        'data_bundle_hash':backend.bundle_hash,'source_hash':backend.source_hash,
        'resume_allowed':False,'report_role_for_selection':False,'reference_independent_cv':False}
    atomic_json(out/'execution_identity.json',identity)
    with _Stage(out,'nn',cuda=True,fixture=fixture):
        # All contexts exist first, allowing separate Mac jobs to run while later
        # CUDA folds execute. Mac jobs never mutate this NN run or its contexts.
        for fold in selected:_create_context(out/f'fold_{fold:02d}',fold,config,backend,code,fixture)
        for fold in selected:_nn_fold(out/f'fold_{fold:02d}',fold,config,backend,code,device=device,fixture=fixture)
        _data_unchanged(backend)
        receipt={'status':'awaiting_mac_tree','folds':selected,'fixture_only':fixture,
            'config_sha256':identity['config_sha256'],'elapsed_seconds':time.perf_counter()-start,
            'neural_stage_complete':True,'server_tree_fit_performed':False,
            'fold_ledger_sha256':{str(f):sha(out/f'fold_{f:02d}'/'artifact_ledger.json') for f in selected}}
        atomic_json(out/'neural_stage_completed.json',receipt);atomic_json(out/'run_state.json',receipt)
    if not fixture:return receipt
    jobs=out.parent/'numeric_mac_tree_jobs'
    fit_mac_trees(config_path,bundle_dir=bundle_dir,output=jobs,folds=selected,backend=backend,fixture=True)
    import_mac_trees(out,jobs,config_path=config_path,backend=backend,fixture=True)
    freeze_ensembles(out,config_path=config_path,backend=backend,device='cpu',fixture=True)
    return evaluate_reports(out,config_path=config_path,backend=backend,device='cpu',fixture=True)


def fit_mac_trees(config_path='configs/r9_full_training_v1.json',*,bundle_dir=None,output,
                  folds=None,backend=None,fixture=False):
    """Stage2 producer: fresh Mac CPU trees, independent of CUDA NN weights."""
    config_path=Path(config_path).resolve();config=_validate_config(json.loads(config_path.read_text()))
    if not fixture and platform.system()!='Darwin':raise ValueError('Mac CPU tree executor required; no server fallback')
    if not fixture and backend is not None:raise ValueError('Production backend injection forbidden')
    selected=_folds(config['folds'] if folds is None else folds);out=Path(output).resolve()
    if out.exists():raise FileExistsError('Fresh Mac job output required')
    if backend is None:backend=ProductionData(config_path,bundle_dir or ROOT/config['data_output'])
    code={n:sha(ROOT/n) for n in CODE_NAMES if (ROOT/n).is_file()}
    out.mkdir(parents=True)
    with _Stage(out,'mac_cpu_tree',fixture=fixture):
        receipts=[]
        for fold in selected:
            started=time.perf_counter()
            atomic_json(out/'progress.json',{'status':'running','executor':'mac_cpu' if not fixture else 'numeric_fixture_cpu',
                'fold':fold,'completed_folds':[r['fold'] for r in receipts],'threads':config['tree']['threads']})
            print(json.dumps({'stage':'mac_cpu_tree','fold':fold,'status':'started'}),flush=True)
            receipts.append(_tree_job(out/f'fold_{fold:02d}',fold,config,backend,code,fixture=fixture))
            gc.collect()
            atomic_json(out/'progress.json',{'status':'running','last_completed_fold':fold,
                'completed_folds':[r['fold'] for r in receipts],'last_fold_elapsed_seconds':time.perf_counter()-started})
            print(json.dumps({'stage':'mac_cpu_tree','fold':fold,'status':'completed',
                              'elapsed_seconds':time.perf_counter()-started}),flush=True)
        _data_unchanged(backend)
        result={'format':'r9_mac_tree_jobs_v1','status':'completed','folds':selected,'fixture_only':fixture,
            'config_sha256':canonical(config),'source_hash':backend.source_hash,
            'receipt_file_sha256':{str(f):sha(out/f'fold_{f:02d}'/'receipt.json') for f in selected},
            'GPU_or_Test_accessed':False,'transport_performed':False}
        atomic_json(out/'jobs_completed.json',result);atomic_json(out/'run_state.json',{'status':'completed'})
        atomic_json(out/'progress.json',{'status':'completed','completed_folds':selected})
    return result


def import_mac_trees(run,jobs,*,config_path=None,bundle_dir=None,backend=None,fixture=False):
    """Stage2 consumer: completed tree import and selection replay, never fit."""
    run,identity,config,backend=_stage_load(run,config_path,bundle_dir,backend,fixture)
    if (run/'trees_imported.json').exists():raise FileExistsError('Completed Mac tree import is immutable')
    completed=json.loads((run/'neural_stage_completed.json').read_text())
    if completed.get('neural_stage_complete') is not True:raise ValueError('All requested neural folds must complete before importing')
    jobs=Path(jobs).resolve();jc=json.loads((jobs/'jobs_completed.json').read_text())
    if (jobs/'FAILED.json').exists() or jc.get('status')!='completed' or jc['folds']!=identity['folds']:
        raise ValueError('Complete Mac jobs matching the requested folds required')
    for f,d in jc['receipt_file_sha256'].items():
        if sha(jobs/f'fold_{int(f):02d}'/'receipt.json')!=d:raise ValueError('Mac completion receipt changed')
    with _Stage(run,'import_mac_trees',fixture=fixture):
        for fold in identity['folds']:_import_tree(run/f'fold_{fold:02d}',jobs/f'fold_{fold:02d}',identity['code_sha256'],backend,fixture=fixture)
        _data_unchanged(backend)
        atomic_json(run/'trees_imported.json',{'status':'completed','executor':'mac_cpu' if not fixture else 'numeric_fixture_cpu',
            'fit_performed_by_consumer':False,'source_jobs_manifest_sha256':sha(jobs/'jobs_completed.json')})
        atomic_json(run/'run_state.json',{'status':'awaiting_ensemble','folds':identity['folds']})
    return {'status':'awaiting_ensemble','tree_fit_performed':False}


def freeze_ensembles(run,*,config_path=None,bundle_dir=None,backend=None,device='cuda',fixture=False):
    """Stage3: fit/gate policies, then global freeze; report remains unopened."""
    run,identity,config,backend=_stage_load(run,config_path,bundle_dir,backend,fixture)
    if (run/'all_requested_models_and_policies_frozen.json').exists():raise FileExistsError('Global policy freeze is immutable')
    if not (run/'trees_imported.json').exists():raise ValueError('Verified Mac trees required before ensemble stage')
    if not fixture and (device!='cuda' or not torch.cuda.is_available()):raise ValueError('Available CUDA required for neural query stage')
    with _Stage(run,'ensemble',cuda=True,fixture=fixture):
        for fold in identity['folds']:_ensemble_fold(run/f'fold_{fold:02d}',fold,config,backend,identity['code_sha256'],device=device,fixture=fixture)
        _data_unchanged(backend)
        _freeze_global(run,identity['folds'],identity['code_sha256'])
        atomic_json(run/'run_state.json',{'status':'awaiting_report','folds':identity['folds'],
            'global_freeze_sha256':sha(run/'all_requested_models_and_policies_frozen.json'),'report_opened':False})
    return {'status':'awaiting_report','report_opened':False}


def evaluate_reports(run,*,config_path=None,bundle_dir=None,backend=None,device='cuda',fixture=False):
    """Stage4: one-shot report evaluation after all requested folds freeze."""
    from pipeline.training_r9 import aggregate_metrics
    run,identity,config,backend=_stage_load(run,config_path,bundle_dir,backend,fixture)
    if (run/'results.json').exists():raise FileExistsError('Completed one-shot report cannot be repeated')
    if not fixture and (device!='cuda' or not torch.cuda.is_available()):raise ValueError('Available CUDA required for neural query stage')
    _verify_global(run,identity['folds'],identity['code_sha256'])
    with _Stage(run,'report',cuda=True,fixture=fixture):
        summaries=[_report_fold(run,run/f'fold_{f:02d}',f,identity['folds'],config,backend,identity['code_sha256'],device=device,fixture=fixture)
                   for f in identity['folds']]
        _data_unchanged(backend)
        pooled={k:[] for k in ['share','score','target_share','target_presence','site']}
        for f in identity['folds']:
            fp=run/f'fold_{f:02d}'
            with np.load(fp/'report_predictions.npz',allow_pickle=False) as p,np.load(fp/'report_known_targets.npz',allow_pickle=False) as y:
                for k in ['share','score']:pooled[k].append(p[k])
                for k,v in [('target_share','share'),('target_presence','presence'),('site','site')]:pooled[k].append(y[v])
        pm={k:np.concatenate(v) for k,v in pooled.items()};pm['site_metadata_available']=True
        result={'status':'completed','folds':identity['folds'],'summaries':summaries,'fixture_only':fixture,
            'config_sha256':identity['config_sha256'],'data_bundle_hash':identity['data_bundle_hash'],
            'pooled_report_metrics':aggregate_metrics(pm),
            'fold_macro_report_rmse':float(np.mean([s['report_metrics']['frozen_ensemble']['rmse'] for s in summaries])),
            'all_requested_folds_completed':True,'all5_background_folds_completed':identity['folds']==list(range(5)),
            'global_model_policy_freeze_before_any_report':True,'report_used_for_selection':False,
            'reference_independent_cv':False,'official_Test_or_submission_accessed':False,
            'inference_contract':'equal mean of the five frozen fold policies; no cohort fitting',
            'elapsed_since_nn_start_seconds':time.time()-__import__('datetime').datetime.fromisoformat(identity['created_utc']).timestamp()}
        atomic_json(run/'results.json',result)
        # The final root ledger pins every stable result and nested fold file.
        # RUNNING.lock and the final run_state are operational, not model inputs.
        artifacts={p.relative_to(run).as_posix():sha(p) for p in _regular_files(run)
                   if p.relative_to(run).as_posix() not in ['RUNNING.lock','run_state.json','run_artifact_ledger.json']}
        atomic_json(run/'run_artifact_ledger.json',{'format':'r9_completed_run_artifact_ledger_v1',
            'config_sha256':identity['config_sha256'],'code_sha256':identity['code_sha256'],
            'artifact_sha256':artifacts,'fixture_only':fixture,'complete':True})
        atomic_json(run/'run_state.json',{'status':'completed','folds':identity['folds'],
            'config_sha256':identity['config_sha256'],'results_sha256':sha(run/'results.json'),
            'run_artifact_ledger_sha256':sha(run/'run_artifact_ledger.json'),'fixture_only':fixture})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/r9_full_training_v1.json')
    p.add_argument('--bundle-dir');p.add_argument('--output');p.add_argument('--folds',type=int,nargs='+')
    p.add_argument('--device',choices=['cuda'],default='cuda')
    a=p.parse_args();result=run_training(a.config,bundle_dir=a.bundle_dir,output=a.output,folds=a.folds,device=a.device)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}))


if __name__=='__main__':main()
