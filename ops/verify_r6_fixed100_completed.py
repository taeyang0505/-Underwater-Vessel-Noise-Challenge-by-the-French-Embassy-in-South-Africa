"""Read-only completed-run audit; no fitting, inference or source-array decoding.

Prepare now, execute only after the parent confirms all five arms completed.
Declared TRAIN/INNER files are streamed solely for SHA256 verification.
No np.load/data loader/prediction call is used. All 500 saved model states
are loaded as weights_only on CPU, checked for finite values and shapes.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def require(condition,message):
    if not condition:raise ValueError(message)


def compare_hashes(mapping,label):
    for name,expected in mapping.items():
        require(sha(ROOT/name)==expected,label+' changed: '+name)
    return dict(mapping)


def close(a,b,what):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=1e-10,
            what+' mismatch')


def finite_state(value,torch,label):
    if isinstance(value,torch.Tensor):
        require(torch.isfinite(value).all().item(),label+' contains nonfinite tensor')
    elif isinstance(value,dict):
        for k,v in value.items():finite_state(v,torch,label+'/'+str(k))
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):finite_state(v,torch,label+'/'+str(i))
    elif isinstance(value,float):require(math.isfinite(value),label+' contains nonfinite scalar')


def verify(run,config):
    started=time.perf_counter()
    require(not (run/'FAILED.json').exists(),'Run has FAILED.json; not a completed release')
    result=read_json(run/'results.json');gate=read_json(run/'all_models_frozen.json')
    progress=read_json(run/'progress.json');freeze=read_json(run/'freeze.json');cfg=read_json(config)
    require(result['status']=='completed' and gate['status']=='frozen' and progress['phase']=='completed',
            'Parent completion/freeze artifacts are required before auditing weights')
    require(sha(config)==freeze['config_sha256']==result['config_sha256']==gate['config_sha256'],
            'Config differs from frozen execution')
    new_code=compare_hashes(freeze['new_code_sha256'],'Fixed100 code')
    require(len(cfg['CNN_arms'])==5 and len({a['name'] for a in cfg['CNN_arms']})==5,'Exactly five unique arms required')
    arms=cfg['CNN_arms'];names={a['name'] for a in arms}
    require(set(gate['model_summary_sha256'])==names,'Frozen summary arm set differs from config')
    require(len(result['summaries'])==5 and {s['arm']['name'] for s in result['summaries']}==names,
            'Final result summary arm set mismatch')
    require(cfg['neural']['epochs']==100 and cfg['neural']['early_stopping_enabled'] is False,
            'Exactly100/no-ES config required')
    require(not result['heldout_evaluated'] and not gate['heldout_evaluated'] and not result['Test_GPU_or_submission_used'],
            'Completion scope differs from the no-heldout release')
    require(result['old_heldout_panels_already_seen_and_not_blind'] is True,'Old diagnostic-panel status missing')
    require(len(cfg['original_frozen_core17_sha256'])==17,'Original 17-code closure missing')
    core17=compare_hashes(cfg['original_frozen_core17_sha256'],'Original R5 core17')
    used_inputs=compare_hashes(cfg['used_input_sha256'],'Declared input bytes')
    data_manifest=ROOT/'data/synthetic/r6_cpu_guardrails_v3_schemafix/manifest.json'
    require(sha(data_manifest)==cfg['data_manifest_sha256']==result['data_manifest_sha256']==freeze['data_manifest_sha256'],
            'Original data manifest changed')
    data=read_json(data_manifest)
    require(sha(ROOT/'configs/r6_cpu_guardrails_v2.json')==data['freeze']['config_sha256'],'Original generator config changed')
    old_code=compare_hashes(data['freeze']['code_sha256'],'Original R6 dataset code')
    import torch
    from pipeline.relative_models_r6_v2 import TinyMaskCNNv2
    torch.set_num_threads(1)
    verified=[];total_weights=0
    for arm in arms:
        name=arm['name'];folder=run/name;s=read_json(folder/'summary.json')
        require(sha(folder/'summary.json')==gate['model_summary_sha256'][name],'Frozen arm summary changed: '+name)
        require(s==next(v for v in result['summaries'] if v['arm']['name']==name),'Final result disagrees with summary: '+name)
        require(s['status']=='completed' and s['arm']==arm and not s['early_stopping_enabled']
                and s['epochs_run']==100 and s['stop_reason']=='completed_100_epochs' and not s['heldout_evaluated'],
                'Arm completion contract failed: '+name)
        n=int(data['diagnostics']['train_'+arm['generator']]['examples']);batch=cfg['neural']['batch_size']
        require(n==8192 and batch==128,'Production sample/batch contract differs from expected8192/128')
        steps_per_epoch=math.ceil(n/batch);expected_steps=100*steps_per_epoch
        require(expected_steps==6400 and s['optimizer_steps']==expected_steps,'Optimizer total step count mismatch: '+name)
        hist=[json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
        require(len(hist)==100 and [r['epoch'] for r in hist]==list(range(1,101)),'Epoch sequence incomplete: '+name)
        for epoch,row in enumerate(hist,1):
            require(row['optimizer_steps']==epoch*steps_per_epoch,'Per-epoch optimizer steps differ: '+name)
            for field in ['train_rmse','train_loss','inner_rmse','inner_site_macro_rmse','learning_rate','next_learning_rate','elapsed_seconds']:
                require(math.isfinite(float(row[field])),'Nonfinite history '+name+'/'+field)
        we=min(range(100),key=lambda i:hist[i]['inner_rmse'])+1
        me=min(range(100),key=lambda i:hist[i]['inner_site_macro_rmse'])+1
        for i,row in enumerate(hist):
            prefix=hist[:i+1];wbest=min(range(len(prefix)),key=lambda j:prefix[j]['inner_rmse'])+1
            mbest=min(range(len(prefix)),key=lambda j:prefix[j]['inner_site_macro_rmse'])+1
            require(row['best_window_epoch']==wbest and row['best_macro_epoch']==mbest,'Strict prefix minima mismatch: '+name)
            close(row['best_inner_window_rmse'],prefix[wbest-1]['inner_rmse'],'Window prefix metric')
            close(row['best_inner_site_macro_rmse'],prefix[mbest-1]['inner_site_macro_rmse'],'Macro prefix metric')
        mc={'input_channels':8,'width':cfg['neural']['width'],'blocks':cfg['neural']['blocks'],
            'dropout':cfg['neural']['dropout'],'global_context':arm.get('global_context',True)}
        require(s['model_config']==mc,'Model configuration mismatch: '+name)
        model=TinyMaskCNNv2(**mc);expected=model.state_dict();schema={k:(tuple(v.shape),v.dtype) for k,v in expected.items()}
        require(s['parameters']==sum(p.numel() for p in model.parameters()),'Model parameter count mismatch: '+name)
        expected_files={f'epoch_{i:03d}.pt' for i in range(1,101)}
        actual={p.name for p in (folder/'checkpoints').iterdir()}
        require(actual==expected_files and set(s['epoch_checkpoint_sha256'])==expected_files,'Epoch checkpoint coverage mismatch: '+name)
        for epoch in range(1,101):
            path=folder/'checkpoints'/f'epoch_{epoch:03d}.pt'
            require(sha(path)==s['epoch_checkpoint_sha256'][path.name],'Checkpoint bytes changed: '+str(path))
            ck=torch.load(path,map_location='cpu',weights_only=True)
            require(ck['epoch']==epoch and ck['arm']==arm and ck['seed']==cfg['seed'] and ck['model_config']==mc,
                    'Checkpoint metadata mismatch: '+str(path))
            state=ck['state_dict'];require(set(state)==set(schema),'State key set mismatch: '+str(path))
            for k,v in state.items():
                require(isinstance(v,torch.Tensor) and (tuple(v.shape),v.dtype)==schema[k],'Tensor schema mismatch: '+str(path)+'/'+k)
                require(torch.isfinite(v).all().item(),'Nonfinite model weight: '+str(path)+'/'+k)
            close(ck['inner_window_rmse'],hist[epoch-1]['inner_rmse'],'Checkpoint/history window metric')
            close(ck['inner_site_macro_rmse'],hist[epoch-1]['inner_site_macro_rmse'],'Checkpoint/history macro metric')
            total_weights+=1;del ck,state
        alias_epochs={'best_window.pt':we,'best_macro.pt':me,'last.pt':100}
        for alias,epoch in alias_epochs.items():
            require(sha(folder/alias)==sha(folder/'checkpoints'/f'epoch_{epoch:03d}.pt'),'Alias not exact selected epoch: '+name+'/'+alias)
        require(s['best_inner_window']['epoch']==we and s['best_inner_macro']['epoch']==me and s['final_epoch']['epoch']==100,
                'Selected summary epochs mismatch: '+name)
        close(s['best_inner_window']['metrics']['rmse'],hist[we-1]['inner_rmse'],'Replayed best window metric')
        close(s['best_inner_macro']['metrics']['site_macro_rmse'],hist[me-1]['inner_site_macro_rmse'],'Replayed best macro metric')
        close(s['final_epoch']['metrics']['rmse'],hist[-1]['inner_rmse'],'Replayed last window metric')
        for path,h in s['artifact_sha256'].items():require(sha(folder/path)==h,'Final artifact changed: '+name+'/'+path)
        if s['prior_sha256'] is None:require(not (folder/'legacy_train_prior.json').exists(),'Unexpected prior: '+name)
        else:require(sha(folder/'legacy_train_prior.json')==s['prior_sha256'],'TRAIN prior changed: '+name)
        final=torch.load(folder/'final_training_state.pt',map_location='cpu',weights_only=True)
        require(final['epoch']==100 and final['optimizer_steps']==expected_steps,'Final optimizer epoch/step mismatch: '+name)
        finite_state(final['optimizer_state_dict'],torch,name+'/optimizer_state')
        finite_state(final['torch_rng_state'],torch,name+'/torch_rng_state')
        # ReduceLROnPlateau deliberately stores mode_worse=+inf for mode='min'.
        # This is a sentinel, not a nonfinite learned weight or optimizer buffer.
        sched_state=final['scheduler_state_dict']
        require(sched_state['mode']=='min' and sched_state['mode_worse']==float('inf'),
                'Unexpected scheduler sentinel policy: '+name)
        finite_state({k:v for k,v in sched_state.items() if k!='mode_worse'},torch,name+'/scheduler_state')
        opt=final['optimizer_state_dict'];ids=[p for g in opt['param_groups'] for p in g['params']]
        require(len(ids)==len(set(ids))==len(list(model.parameters())) and set(opt['state'])==set(ids),'Optimizer state coverage mismatch: '+name)
        require(all(float(v['step'])==expected_steps for v in opt['state'].values()),'Per-parameter Adam step mismatch: '+name)
        require(final['scheduler_state_dict']['last_epoch']==100,'Scheduler epoch count mismatch: '+name)
        # Independently replay the diagnostic-only shadow ES7 history accounting.
        shadow_best=float('inf');shadow_epoch=0;stale=0;stop=None
        for row in hist:
            if stop is not None:break
            if row['inner_site_macro_rmse']<shadow_best-1e-5:
                shadow_best=row['inner_site_macro_rmse'];shadow_epoch=row['epoch'];stale=0
            else:stale+=1
            if stale>=7:stop=row['epoch']
        require(s['shadow_ES7']['stop_epoch']==stop and s['shadow_ES7']['best_macro_epoch']==shadow_epoch
                and s['shadow_ES7']['stopping_used_for_training'] is False,'Shadow ES accounting mismatch: '+name)
        close(s['shadow_ES7']['best_macro_rmse'],shadow_best,'Shadow ES best metric')
        verified.append({'arm':name,'summary_sha256':sha(folder/'summary.json'),'epoch_rows':100,'finite_epoch_weights_loaded':100,
                         'optimizer_steps':expected_steps,'best_window_epoch':we,'best_macro_epoch':me,'last_epoch':100,
                         'strict_first_tie_minimum_aliases_match':True,'state_keys_shapes_dtype_finite':True,
                         'shadow_ES7_stop_epoch':stop,'parameter_count':s['parameters'],
                         'final_training_state_sha256':sha(folder/'final_training_state.pt')})
    require(total_weights==500,'Expected500 verified epoch weights')
    # Repeat immutable-source checks after loading the complete artifact bank.
    compare_hashes(core17,'Original core17');compare_hashes(used_inputs,'Declared inputs')
    compare_hashes(new_code,'Fixed100 code');compare_hashes(old_code,'Original R6 code')
    require(sha(config)==freeze['config_sha256'],'Config changed during verification')
    return {'format':'r6_fixed100_independent_completion_verification_v1','status':'passed',
            'created_utc':datetime.now(timezone.utc).isoformat(),'run':str(run.relative_to(ROOT)),
            'five_completed_arms':len(verified),'epochs_per_arm':100,'epoch_weights_loaded':total_weights,
            'total_epoch_history_rows':500,'optimizer_steps_per_arm':6400,'optimizer_steps_all_arms':32000,
            'fitting_or_inference_or_metric_reevaluation_performed':False,
            'source_array_values_decoded':False,'only_declared_TRAIN_INNER_bytes_hashed':True,
            'Test_old_crosspanels_OUTER_GPU_SSH_used':False,'heldout_evaluated':False,
            'old_crosspanels_already_seen_not_blind':True,'global_minimum_proven':False,
            'config_sha256':sha(config),'run_freeze_sha256':sha(run/'freeze.json'),
            'results_sha256':sha(run/'results.json'),'model_freeze_sha256':sha(run/'all_models_frozen.json'),
            'original_core17_sha256':core17,'declared_input_sha256':used_inputs,
            'new_code_sha256':new_code,'dataset_generator_code_sha256':old_code,
            'arms':verified,'verifier_code_sha256':sha(__file__),'elapsed_seconds':time.perf_counter()-started}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-completed',action='store_true',help='Run only after explicit parent completion signal')
    p.add_argument('--run',type=Path,default=ROOT/'models/r6_fixed100_v1')
    p.add_argument('--config',type=Path,default=ROOT/'configs/r6_fixed100_v1.json')
    p.add_argument('--output',type=Path,default=ROOT/'reports/R6_FIXED100_COMPLETION_VERIFICATION.json')
    a=p.parse_args()
    if not a.verify_completed:p.error('--verify-completed is required after completion; preparation never audits active weights')
    if a.output.exists():raise FileExistsError('Fresh postflight output required')
    try:report=verify(a.run.resolve(),a.config.resolve())
    except Exception as exc:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps({'status':'failed','error':str(exc),'error_type':type(exc).__name__,
            'verifier_code_sha256':sha(__file__),'no_fitting_inference_or_source_array_decode':True},indent=2)+'\n')
        raise
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'report':str(a.output),'report_sha256':sha(a.output),
                      'arms':report['five_completed_arms'],'epoch_weights':report['epoch_weights_loaded'],
                      'optimizer_steps_per_arm':report['optimizer_steps_per_arm'],'elapsed_seconds':report['elapsed_seconds']}))


if __name__=='__main__':main()
