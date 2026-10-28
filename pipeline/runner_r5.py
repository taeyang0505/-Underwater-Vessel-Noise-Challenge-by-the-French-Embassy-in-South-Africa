"""Nested source CV orchestrator; fresh fits, INNER selection, frozen OUTER.

No SSH, GPU rental, Test execution or official submission is performed here.
Running `fit` requires a prepared CUDA host and an explicit command.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
from pipeline.grid_inputs_r5 import (canonical_sha256,file_sha256,fit_shared_prior,save_shared_prior)
from pipeline.fold_data_r5 import FoldDataR5

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/r5_training_integration.json'
FIXTURE_NAMES=['R5_SHARED_INPUT_DYNAMIC_FIXTURE.json','R5_OBSERVATION_AUGMENTATION_FIXTURE.json',
    'R5_PHYSICS_FEATURES_FIXTURE.json','R5_NEURAL_RUNNER_FIXTURE.json','R5_LGBM_RUNNER_FIXTURE.json',
    'R5_ENSEMBLE_FIXTURE.json','R5_RUNNER_ORCHESTRATION_FIXTURE.json','R5_FROZEN_INFERENCE_FIXTURE.json']
CODE_NAMES=['pipeline/runner_r5.py','pipeline/neural_r5.py','pipeline/lgbm_r5.py',
    'pipeline/fold_data_r5.py','pipeline/grid_inputs_r5.py','pipeline/dynamic_passage_r5.py',
    'pipeline/observation_augmentation_r5.py','pipeline/physics_features_r5.py',
    'pipeline/hierarchical_ensemble_r5.py','pipeline/hybrid_models_r5.py',
    'pipeline/grid_models_r5.py','pipeline/foundation_masks_r5.py','pipeline/load_foundation_r5.py',
    'pipeline/mixtures.py','pipeline/models.py','pipeline/background_prior_features.py','pipeline/inference_r5.py']


def validate_config(config):
    if config.get('version')!='r5_training_integration_v1' or config.get('protocol')!='source_group_nested5_vtuad_passage_r4':raise ValueError('Unsupported R5 config')
    if set(config['core_models'])!={'bg_cnn','tfgrid_fdy_conformer','lgbm_physics','timemoe_public','dasheng'}:raise ValueError('R5 candidate registry changed')
    t=config['training']
    if (t['epochs'],t['patience'],t['optimizer'])!=(100,7,'adam'):raise ValueError('User neural training contract changed')
    if t['contrastive_weight']!=0:raise ValueError('Contrastive objective requires a separately validated experiment')
    if config['ensemble'].get('ridge')!=.05 or config['ensemble'].get('epsilon_rmse')!=1e-4:
        raise ValueError('INNER ensemble helper uses the predeclared ridge.05/epsilon1e-4')
    if config['site_policy']!='diagnostic slices and macro-site RMSE; no Test site-proportion fitting; no weighted resampling':raise ValueError('Undeclared site-risk tuning')
    if config.get('no_Test_fitting') is not True or config.get('no_official_submission') is not True:raise ValueError('Inference/submission scope must remain separate')
    return canonical_sha256(config)


def _call(module,arguments):
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4'
    subprocess.run([sys.executable,'-m',module,*map(str,arguments)],cwd=ROOT,env=env,check=True)


def _read_prediction(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def _verify_code(context):
    for name,h in context['code_sha256'].items():
        if file_sha256(ROOT/name)!=h:raise ValueError('Code changed during run: '+name)


def _metrics(arrays,metadata,prediction):
    from pipeline.metrics import evaluate_regression,quality_flags_from_metadata
    return evaluate_regression(arrays['share'],prediction['share'],arrays['presence'],prediction['score'],
        db=arrays['db'],groups={'site':metadata.site.to_numpy(),
                              'background_source':metadata.background_group.to_numpy(),
                              'reference_combination':metadata.reference_group.to_numpy()},
        quality_flags=quality_flags_from_metadata(metadata))


def _ensemble_prediction(policy,branches,w,binding,role):
    from pipeline.hierarchical_ensemble_r5 import apply_policy
    shares={k:v['share'] for k,v in branches.items()}
    masks={k:v['cell_mask'] for k,v in branches.items() if 'cell_mask' in v}
    share=apply_policy(policy,shares,evaluation_split=role,binding=binding,neural_masks=masks,energy_weights=w)
    # Presence is diagnostic controlled-addition evidence, not natural vessel truth.
    weights=policy['weights']
    score=sum(float(value)*branches[name]['score'] for name,value in weights.items())
    return {'share':np.asarray(share,dtype='float32'),'score':np.clip(score,0,1).astype('float32')}


def fit_fold(config,number,output):
    """One full nested experiment, never resumes failed or partial directories."""
    from pipeline.hierarchical_ensemble_r5 import select_inner_policy
    from pipeline.observation_augmentation_r5 import prepare_query_observations
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh fold directory required')
    output.mkdir(parents=True);begin=time.monotonic()
    folder=Path(config['dataset'])/f'fold_{number:02d}';f=FoldDataR5(folder,config['seed'])
    train,train_meta=f.load('train');inner,inner_meta=f.load('inner')
    prior=fit_shared_prior(train,f.binding,split='train')
    prior_record=save_shared_prior(output/'shared_background_prior.json',prior,expected_binding=f.binding)
    context={'format':'r5_fold_context_v1','data_fold':str(folder),'training_binding':f.binding,
             'config':config,'config_sha256':validate_config(config),'prior':prior_record,
             'code_sha256':{name:file_sha256(ROOT/name) for name in CODE_NAMES}}
    context_path=output/'context.json';context_path.write_text(json.dumps(context,indent=2)+'\n')
    del train,train_meta
    paths={};candidates={};pilot=[]
    try:
        # Three fixed robustness candidates, using only this fold's INNER.
        for observation,policy in config['observation_candidates'].items():
            path=output/'preprocessing_pilot'/observation/'bg_cnn'
            _call('pipeline.neural_r5',['fit','--context',context_path,'--kind','bg_cnn','--observation',observation,'--output',path])
            _verify_code(context);summary=json.loads((path/'summary.json').read_text())
            pilot.append({'observation':observation,'inner_rmse':summary['best_inner_rmse'],
                          'model_path':str(path.relative_to(output)),'summary_sha256':file_sha256(path/'summary.json')})
        winner=min(pilot,key=lambda x:x['inner_rmse']);observation=winner['observation']
        paths['bg_cnn']=output/winner['model_path']
        (output/'preprocessing_selection.json').write_text(json.dumps({'fit_split':'inner','candidates':pilot,
            'selected':observation,'outer_or_Test_used':False,'training_binding':f.binding},indent=2)+'\n')
        for kind in [k for k in config['core_models']+config['comparisons'] if k!='bg_cnn']:
            path=output/'models'/kind
            module='pipeline.lgbm_r5' if kind=='lgbm_physics' else 'pipeline.neural_r5'
            arguments=['fit','--context',context_path,'--observation',observation,'--output',path]
            if kind!='lgbm_physics':arguments+=['--kind',kind]
            _call(module,arguments);_verify_code(context);paths[kind]=path
        for kind,path in paths.items():candidates[kind]=_read_prediction(path/'inner_predictions.npz')
        weights=prepare_query_observations(inner['db'],quantization_decimals=None)['energy_weights']
        policy=select_inner_policy(inner['share'],{k:v['share'] for k,v in candidates.items()},
            inner_meta.reference_group.to_numpy(),fit_split='inner',binding=f.binding,
            explicit_exclusions={},enable_logit_mask_candidate=config['ensemble']['enable_logit_mask_candidate'],
            neural_masks={k:v['cell_mask'] for k,v in candidates.items() if 'cell_mask' in v},energy_weights=weights)
        if policy.get('training_binding')!=f.binding or policy.get('frozen_before_outer') is not True:raise ValueError('Ensemble policy failed freeze contract')
        policy_path=output/'frozen_ensemble_policy.json';policy_path.write_text(json.dumps(policy,indent=2,allow_nan=False)+'\n')
        inner_ensemble=_ensemble_prediction(policy,candidates,weights,f.binding,'inner')
        inner_metrics={k:_metrics(inner,inner_meta,p) for k,p in candidates.items()}
        inner_metrics['ensemble']=_metrics(inner,inner_meta,inner_ensemble)
        (output/'inner_metrics.json').write_text(json.dumps(inner_metrics,indent=2,allow_nan=False)+'\n')
        # Only now may any OUTER acoustic values/labels be opened.
        outer,outer_meta=f.load('outer',frozen_policy=policy);outer_predictions={}
        for kind,path in paths.items():
            destination=output/'outer_predictions'/f'{kind}.npz'
            arguments=['predict','--context',context_path,'--observation',observation,'--model-dir',path,
                       '--role','outer','--frozen-policy',policy_path,'--output',destination]
            module='pipeline.lgbm_r5' if kind=='lgbm_physics' else 'pipeline.neural_r5'
            if kind!='lgbm_physics':arguments+=['--kind',kind]
            _call(module,arguments);_verify_code(context)
            outer_predictions[kind]=_read_prediction(destination)
        w=prepare_query_observations(outer['db'],quantization_decimals=None)['energy_weights']
        ensemble=_ensemble_prediction(policy,outer_predictions,w,f.binding,'outer')
        outer_metrics={k:_metrics(outer,outer_meta,p) for k,p in outer_predictions.items()}
        outer_metrics['ensemble']=_metrics(outer,outer_meta,ensemble)
        (output/'outer_metrics.json').write_text(json.dumps(outer_metrics,indent=2,allow_nan=False)+'\n')
        np.savez_compressed(output/'outer_oof.npz',share=ensemble['share'],score=ensemble['score'],
            truth=outer['share'],presence=outer['presence'],sample_id=outer_meta.sample_id.to_numpy(dtype=str),
            boundary_artifact_candidate=outer_meta.boundary_artifact_candidate.to_numpy(dtype=bool),
            urn_8321_nominal_warning=outer_meta.urn_8321_nominal_warning.to_numpy(dtype=bool),
            site=outer_meta.site.to_numpy(dtype=str),background_group=outer_meta.background_group.to_numpy(dtype=str),
            reference_group=outer_meta.reference_group.to_numpy(dtype=str))
        summary={'status':'completed','fold':number,'training_binding':f.binding,
            'selected_observation':observation,'models':{k:str(p.relative_to(output)) for k,p in paths.items()},
            'frozen_policy_sha256':file_sha256(policy_path),'context_sha256':file_sha256(context_path),
            'inner_rmse':inner_metrics['ensemble']['rmse'],'outer_rmse':outer_metrics['ensemble']['rmse'],
            'outer_rmse_excluding_boundary':outer_metrics['ensemble'].get('oof_rmse_excluding_boundary'),
            'outer_changed_policy':False,'Test_accessed':False,'official_submission':False,
            'elapsed_seconds':time.monotonic()-begin}
        summary['artifact_sha256']={p.relative_to(output).as_posix():file_sha256(p) for p in output.rglob('*') if p.is_file() and p.suffix in {'.pt','.json','.jsonl','.npz','.txt'}}
        (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        return summary
    except BaseException as exc:
        (output/'FAILED.json').write_text(json.dumps({'status':'failed','type':type(exc).__name__,'error':str(exc),
            'partial_run_reusable':False,'Test_accessed':False},indent=2)+'\n')
        raise


def preflight(config):
    """CPU integrity checks, no model fitting, GPU contact or Test read."""
    from pipeline.load_foundation_r5 import verify_public_checkpoint
    result={'config_sha256':validate_config(config),'folds':[],
            'code_sha256':{name:file_sha256(ROOT/name) for name in CODE_NAMES}}
    for i in range(5):
        f=FoldDataR5(Path(config['dataset'])/f'fold_{i:02d}',config['seed'])
        train,metadata=f.load('train');inner,im=f.load('inner')
        result['folds'].append({'fold':i,'train_examples':len(train['share']),
            'inner_examples':len(inner['share']),'binding':f.binding,
            'train_sites':metadata.site.value_counts().astype(int).to_dict(),
            'inner_sites':im.site.value_counts().astype(int).to_dict(),
            'outer_arrays_opened':False})
    result['public_weights']={kind:verify_public_checkpoint(kind) for kind in ['timemoe_public','dasheng']}
    fixture_reports={}
    for name in FIXTURE_NAMES:
        path=ROOT/'reports'/name;report=json.loads(path.read_text())
        if report.get('passed') is not True and not str(report.get('status','')).lower().startswith('pass'):
            raise ValueError('CPU integration fixture has not passed: '+name)
        for source,h in report.get('code_sha256',{}).items():
            if source.startswith(('pipeline/','ops/')) and file_sha256(ROOT/source)!=h:raise ValueError('Fixture code receipt is stale: '+source)
        fixture_reports[name]={'sha256':file_sha256(path),'status':report.get('status',report.get('passed'))}
    result['fixture_reports']=fixture_reports
    result.update({'data_and_public_weight_integrity':'passed','GPU_used':False,'Test_accessed':False,
                   'runner_GPU_execution':'not_yet_measured','GPU_rented_or_started':False,
                   'ready_for_gpu_pilot':True,'production_GPU_verified':False})
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','fit'])
    parser.add_argument('--config',type=Path,default=CONFIG);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-report',type=Path,default=ROOT/'reports/R5_TRAINING_PREFLIGHT.json')
    parser.add_argument('--folds',type=int,nargs='+',default=[0,1,2,3,4]);a=parser.parse_args()
    config=json.loads(a.config.read_text());validate_config(config)
    if a.action=='preflight':
        report=preflight(config);a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'preflight':'passed','report':str(a.output)}));return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU required for production fit; wait for user-provided access')
    proof=json.loads(a.preflight_report.read_text())
    if proof.get('ready_for_gpu_pilot') is not True or proof.get('config_sha256')!=canonical_sha256(config):raise ValueError('A current completed CPU preflight is required')
    for name,h in proof['code_sha256'].items():
        if file_sha256(ROOT/name)!=h:raise ValueError('CPU preflight source snapshot is stale: '+name)
    for name,receipt in proof['fixture_reports'].items():
        if file_sha256(ROOT/'reports'/name)!=receipt['sha256']:raise ValueError('CPU preflight fixture receipt changed: '+name)
    if len(set(a.folds))!=len(a.folds) or any(i not in range(5) for i in a.folds):raise ValueError('Unique folds0..4 required')
    if a.output.exists():raise FileExistsError('Fresh run directory required')
    a.output=a.output.resolve();a.output.mkdir(parents=True);summaries=[]
    for i in a.folds:
        summaries.append(fit_fold(config,i,a.output/f'fold_{i:02d}'))
        (a.output/'run_state.json').write_text(json.dumps({'status':'running','config_sha256':canonical_sha256(config),'completed_folds':summaries},indent=2)+'\n')
    final={'status':'completed' if len(summaries)==5 else 'completed_requested_folds','config_sha256':canonical_sha256(config),'folds':summaries,
           'outer_selected_any_policy':False,'Test_accessed':False,'official_submission':False}
    if len(summaries)==5:
        from pipeline.metrics import evaluate_regression
        parts=[_read_prediction(a.output/f'fold_{i:02d}'/'outer_oof.npz') for i in a.folds]
        final['pooled_oof']=evaluate_regression(np.concatenate([p['truth'] for p in parts]),np.concatenate([p['share'] for p in parts]),
            np.concatenate([p['presence'] for p in parts]),np.concatenate([p['score'] for p in parts]),
            groups={'site':np.concatenate([p['site'] for p in parts]),'background_source':np.concatenate([p['background_group'] for p in parts]),
                    'reference_combination':np.concatenate([p['reference_group'] for p in parts])},
            quality_flags={name:np.concatenate([p[name] for p in parts]) for name in ['boundary_artifact_candidate','urn_8321_nominal_warning']})
    (a.output/'run_state.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
