"""R5 neural fit/predict primitives with physical pooling and Adam.

The GPU launch is a separate explicit action. CPU fixtures may inject a small
model/data stream; production never substitutes random foundation weights.
"""
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch import nn
from pipeline.grid_inputs_r5 import (shared_grid_inputs, canonical_sha256,
                                    file_sha256)

ROOT=Path(__file__).resolve().parents[1]
FOUNDATIONS={'timemoe_public','dasheng'}
CHECKPOINT_CODE_NAMES=('neural_r5.py','hybrid_models_r5.py','grid_models_r5.py','models.py','foundation_masks_r5.py',
    'grid_inputs_r5.py','mixtures.py','background_prior_features.py','observation_augmentation_r5.py',
    'load_foundation_r5.py','dynamic_passage_r5.py')


def grid_inputs(db,prior,binding,observation_policy):
    from pipeline.observation_augmentation_r5 import prepare_query_observations
    obs=prepare_query_observations(db,quantization_decimals=observation_policy['quantization_decimals'])
    # Every learned input channel sees feature_db. The nonlearned physical
    # pooling weights retain raw observed energy, so the exact q/y is unchanged.
    grid,_=shared_grid_inputs(obs['feature_db'],prior,expected_binding=binding)
    return grid,obs['energy_weights']


def make_neural(kind,settings,*,cache_dir=None):
    if kind in FOUNDATIONS:
        from pipeline.load_foundation_r5 import load_public_foundation,DEFAULT_CACHE
        model,receipt=load_public_foundation(kind,cache_dir or DEFAULT_CACHE,dtype='float32')
    elif kind in {'bg_cnn','tfgrid_fdy_conformer'}:
        from pipeline.hybrid_models_r5 import make_hybrid_model
        model=make_hybrid_model(kind,**settings);receipt={'model_config':model.model_config,'pretrained':False}
    elif kind in {'fdy','itransformer'}:
        from pipeline.models import FDYMaskCRNN,InvertedMaskTransformer
        # Historical make_model(version3) deliberately insists on six inputs.
        # Preserve that contract and construct new, explicit R5 comparisons.
        model=(FDYMaskCRNN if kind=='fdy' else InvertedMaskTransformer)(input_channels=8,**settings)
        receipt={'model_config':settings,'input_channels':8,'pretrained':False}
    else:raise ValueError('Unknown R5 neural candidate: '+kind)
    receipt.update({'kind':kind,'total_parameters':sum(p.numel() for p in model.parameters()),
                    'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
    return model,receipt


def physical_multitask_loss(outputs,targets,energy_weights,*,cell_weight=.25,presence_weight=.05):
    p,s,q=outputs;y,z,truth=targets
    if p.ndim!=1 or s.shape!=p.shape or q.shape!=energy_weights.shape or truth.shape!=q.shape or y.shape!=p.shape or z.shape!=p.shape:
        raise ValueError('Loss tensor alignment failed')
    if not all(torch.isfinite(a).all() for a in [p,s,q,y,z,truth,energy_weights]):raise FloatingPointError('Nonfinite prediction/target')
    if (q<0).any() or (q>1).any() or (energy_weights<0).any():raise ValueError('Invalid physical fractions/weights')
    if not torch.allclose(energy_weights.sum((1,2)),torch.ones_like(p),atol=3e-6,rtol=0):raise ValueError('Energy weights do not sum to one')
    if not torch.allclose(p.float(),(q.float()*energy_weights.float()).sum((1,2)),atol=3e-6,rtol=0):raise ValueError('Scalar head bypasses physical energy pooling')
    share=(p.float()-y.float()).square()
    cell=((q.float()-truth.float()).square()*energy_weights.float()).sum((1,2))
    presence=torch.nn.functional.binary_cross_entropy_with_logits(s.float(),z.float(),reduction='none')
    return share+cell_weight*cell+presence_weight*presence,{'share_mse':share,'weighted_cell_mse':cell,'presence_bce':presence}


@torch.no_grad()
def predict_neural(model,db,prior,binding,policy,*,device='cpu',batch_size=128):
    device=torch.device(device);model.eval();out={k:[] for k in ['share','score','cell_mask']}
    if batch_size<1:raise ValueError('Positive prediction batch required')
    for start in range(0,len(db),batch_size):
        grid,w=grid_inputs(db[start:start+batch_size],prior,binding,policy)
        x=torch.from_numpy(grid).to(device);weights=torch.from_numpy(w).to(device)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            p,s,q=model(x,weights)
        for k,v in [('share',p),('score',s.sigmoid()),('cell_mask',q)]:
            value=v.float().cpu().numpy()
            if not np.isfinite(value).all():raise FloatingPointError('Nonfinite inference: '+k)
            out[k].append(value)
    if not out['share']:raise ValueError('Cannot predict an empty query batch')
    return {k:np.concatenate(v) for k,v in out.items()}


def _checkpoint(model,kind,receipt,binding,prior_sha,policy,settings,epoch,*,dynamic_source=None):
    if kind in FOUNDATIONS:
        weights={'adapter_only':True,'adapter':model.adapter_checkpoint()}
    else:weights={'adapter_only':False,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
    return {'format':'r5_neural_checkpoint_v1','kind':kind,'weights':weights,'receipt':receipt,
            'training_binding':binding,'shared_prior_sha256':prior_sha,'observation_policy':policy,
            'training_settings':settings,'epoch':epoch,
            'dynamic_source_identity':None if dynamic_source is None else dynamic_source.identity,
            'best_epoch_source_receipt':None if dynamic_source is None else dynamic_source.completed_epochs[epoch],
            'model_code_sha256':{str(p.relative_to(ROOT)):file_sha256(p) for p in [ROOT/'pipeline'/name for name in CHECKPOINT_CODE_NAMES]}}


def restore_into(model,checkpoint,*,kind,binding,prior_sha,policy):
    if checkpoint.get('format')!='r5_neural_checkpoint_v1' or checkpoint.get('kind')!=kind:raise ValueError('Checkpoint kind/format mismatch')
    if checkpoint.get('training_binding')!=binding or checkpoint.get('shared_prior_sha256')!=prior_sha or checkpoint.get('observation_policy')!=policy:
        raise ValueError('Checkpoint data/prior/preprocessing binding mismatch')
    if set(checkpoint.get('model_code_sha256',{}))!={'pipeline/'+name for name in CHECKPOINT_CODE_NAMES}:
        raise ValueError('Complete checkpoint model/preprocessing code closure required')
    for name,h in checkpoint['model_code_sha256'].items():
        if file_sha256(ROOT/name)!=h:raise ValueError('Frozen model code changed: '+name)
    state=checkpoint['weights']
    if kind in FOUNDATIONS:
        if state.get('adapter_only') is not True:raise ValueError('Foundation checkpoints must contain adapters only')
        model.load_adapter_checkpoint(state['adapter'])
    else:
        if state.get('adapter_only') is not False:raise ValueError('Unexpected neural checkpoint type')
        model.load_state_dict(state['state_dict'],strict=True)
    return model


def fit_neural(model,kind,receipt,epoch_batches,inner,prior,binding,policy,settings,output,*,device='cpu',site_weights_by_background=None,fixture=False,dynamic_source=None):
    """Fit on TRAIN streams, early-stop on fixed INNER, never read OUTER.

    Sum losses over microbatches and normalize gradients by actual accumulated
    rows, including the final short update. Training risk defaults unweighted;
    optional site weights must come from the same TRAIN background metadata.
    """
    from pipeline.observation_augmentation_r5 import augment_training_observations,EndpointDropoutPolicy
    output=Path(output)
    if output.exists():raise FileExistsError('Fresh model output required; partial runs are not resumed')
    output.mkdir(parents=True)
    device=torch.device(device)
    if not fixture and device.type!='cuda':raise ValueError('Production fit awaits an explicitly provided CUDA GPU; CPU is fixture-only')
    if not fixture and (dynamic_source is None or dynamic_source.binding!=binding):raise ValueError('Production fit requires a bound dynamic TRAIN source')
    if settings.get('optimizer')!='adam' or settings['epochs']!=100 or settings['patience']!=7:
        if not fixture:raise ValueError('Production neural policy is Adam, max100, patience7')
    seed=binding['seed'];torch.manual_seed(seed);np.random.seed(seed)
    model=model.to(device);params=[p for p in model.parameters() if p.requires_grad]
    if not params:raise ValueError('No trainable parameters')
    optimizer=torch.optim.Adam(params,lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2,min_lr=1e-6)
    effective=int(settings['effective_batch']);min_delta=float(settings.get('minimum_improvement',1e-5))
    if effective<1:raise ValueError('Positive effective batch required')
    best=float('inf');best_epoch=0;stale=0;history=[];start_time=time.monotonic()
    prior_sha=prior['shared_prior_payload_sha256'];checkpoint_file=output/'best.pt'
    dropout=EndpointDropoutPolicy(probability=policy['endpoint_dropout_probability'],max_drop=2,min_observed_minutes=3)
    try:
        for epoch in range(1,int(settings['epochs'])+1):
            model.train();optimizer.zero_grad(set_to_none=True);seen=0;pending=0;updates=0
            totals={k:0. for k in ['loss','share_mse','weighted_cell_mse','presence_bce']};grad_max=0.
            augmentation={k:0 for k in ['requested_events','applied_events','minimum_support_protection_skips','positive_to_zero_after_removing_known_support']}
            for batch in epoch_batches(epoch):
                changed,audit=augment_training_observations(batch['db'],batch['cell_share'],epoch=epoch,
                    sample_indices=batch['background_index'],seed=seed,policy=dropout,split='train',
                    quantization_decimals=policy['quantization_decimals'])
                grid,_=shared_grid_inputs(changed['feature_db'],prior,expected_binding=binding)
                w=changed['energy_weights'];size=len(w)
                for name in augmentation:augmentation[name]+=audit[name]
                x=torch.from_numpy(grid).to(device);weights=torch.from_numpy(w).to(device)
                targets=tuple(torch.from_numpy(np.asarray(changed[k],dtype='float32')).to(device) for k in ['share','presence','cell_share'])
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    result=model(x,weights)
                    losses,parts=physical_multitask_loss(result,targets,weights,cell_weight=settings['cell_loss_weight'],presence_weight=settings['presence_loss_weight'])
                if site_weights_by_background is None:loss_weight=torch.ones_like(losses)
                else:
                    loss_weight=torch.tensor([site_weights_by_background[int(i)] for i in batch['background_index']],device=device)
                (losses*loss_weight).sum().backward();pending+=size;seen+=size
                totals['loss']+=float(losses.detach().sum())
                for k,v in parts.items():totals[k]+=float(v.detach().sum())
                if pending>=effective:
                    for p in params:
                        if p.grad is not None:p.grad.div_(pending)
                    norm=torch.nn.utils.clip_grad_norm_(params,float(settings['gradient_clip']),error_if_nonfinite=True)
                    grad_max=max(grad_max,float(norm));optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1;pending=0
            if pending:
                for p in params:
                    if p.grad is not None:p.grad.div_(pending)
                norm=torch.nn.utils.clip_grad_norm_(params,float(settings['gradient_clip']),error_if_nonfinite=True)
                grad_max=max(grad_max,float(norm));optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
            if not seen:raise RuntimeError('TRAIN stream empty')
            source_receipt=None if dynamic_source is None else dynamic_source.completed_epochs.get(epoch)
            if dynamic_source is not None and (not source_receipt or not source_receipt['clean_primary_once'] or source_receipt['examples']!=seen):
                raise ValueError('Dynamic epoch did not complete every clean primary exactly once')
            pred=predict_neural(model,inner['db'],prior,binding,policy,device=device,batch_size=settings['evaluation_batch'])
            inner_rmse=float(np.sqrt(np.mean((pred['share'].astype(float)-inner['share'])**2)))
            if not np.isfinite(inner_rmse):raise FloatingPointError('INNER RMSE nonfinite')
            improved=inner_rmse<best-min_delta
            if improved:
                best=inner_rmse;best_epoch=epoch;stale=0
                temporary=output/'best.tmp.pt'
                torch.save(_checkpoint(model,kind,receipt,binding,prior_sha,policy,settings,epoch,dynamic_source=dynamic_source),temporary)
                temporary.replace(checkpoint_file)
            else:stale+=1
            row={'epoch':epoch,'train_examples':seen,'optimizer_updates':updates,
                 **{k:v/seen for k,v in totals.items()},'train_rmse':float(np.sqrt(totals['share_mse']/seen)),
                 'inner_rmse':inner_rmse,'best_inner_rmse':best,'best_epoch':best_epoch,
                 'patience_used':stale,'learning_rate':optimizer.param_groups[0]['lr'],
                 'maximum_unclipped_gradient_norm':grad_max,'elapsed_seconds':time.monotonic()-start_time,
                 'observation_augmentation':augmentation,'dynamic_epoch_source_receipt':source_receipt}
            history.append(row)
            with (output/'epochs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps({'model':kind,**row}),flush=True)
            scheduler.step(inner_rmse)
            if stale>=int(settings['patience']):break
        checkpoint=torch.load(checkpoint_file,map_location=device,weights_only=True)
        restore_into(model,checkpoint,kind=kind,binding=binding,prior_sha=prior_sha,policy=policy)
        prediction=predict_neural(model,inner['db'],prior,binding,policy,device=device,batch_size=settings['evaluation_batch'])
        np.savez_compressed(output/'inner_predictions.npz',**prediction)
        summary={'status':'completed','kind':kind,'best_epoch':best_epoch,'epochs_run':len(history),
                 'best_inner_rmse':best,'training_binding':binding,'shared_prior_sha256':prior_sha,
                 'observation_policy':policy,'settings':settings,'receipt':receipt,
                 'checkpoint_sha256':file_sha256(checkpoint_file),'prediction_sha256':file_sha256(output/'inner_predictions.npz'),
                 'stop_reason':'patience' if stale>=int(settings['patience']) else 'maximum_epochs',
                 'device':str(device),'fixture_only':fixture,'outer_or_Test_accessed':False,
                 'dynamic_source_identity':None if dynamic_source is None else dynamic_source.identity,
                 'dynamic_source_receipts_file':None if dynamic_source is None else 'dynamic_source_receipts.json',
                 'elapsed_seconds':time.monotonic()-start_time}
        (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        return summary,prediction
    except BaseException as exc:
        (output/'FAILED.json').write_text(json.dumps({'status':'failed','type':type(exc).__name__,'error':str(exc),'partial_run_reusable':False,'outer_or_Test_accessed':False},indent=2)+'\n')
        raise
    finally:
        if dynamic_source is not None:
            (output/'dynamic_source_receipts.json').write_text(json.dumps({'source_identity':dynamic_source.identity,
                'completed_epochs':dynamic_source.completed_epochs,'scope':'Completed epochs only; interrupted epochs are not certified'},indent=2,allow_nan=False)+'\n')


def main():
    import argparse
    from pipeline.fold_data_r5 import FoldDataR5
    from pipeline.grid_inputs_r5 import load_shared_prior
    from pipeline.dynamic_passage_r5 import DynamicPassageEpochSource
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['fit','predict'])
    parser.add_argument('--context',required=True,type=Path);parser.add_argument('--kind',required=True)
    parser.add_argument('--observation',required=True);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device',default='cuda');parser.add_argument('--role',choices=['inner','outer'],default='inner')
    parser.add_argument('--model-dir',type=Path);parser.add_argument('--frozen-policy',type=Path)
    a=parser.parse_args();context=json.loads(a.context.read_text());config=context['config']
    if canonical_sha256(config)!=context['config_sha256']:raise ValueError('Run configuration snapshot changed')
    for name,h in context['code_sha256'].items():
        if file_sha256(ROOT/name)!=h:raise ValueError('Run code snapshot changed: '+name)
    fold=FoldDataR5(context['data_fold'],config['seed']);binding=fold.binding
    if binding!=context['training_binding']:raise ValueError('Run data binding differs')
    prior_path=a.context.parent/context['prior']['filename']
    prior=load_shared_prior(prior_path,expected_binding=binding,expected_file_sha256=context['prior']['file_sha256'],expected_prior_sha256=context['prior']['payload_sha256'])
    policy=config['observation_candidates'][a.observation]
    torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    model,receipt=make_neural(a.kind,config['model_settings'][a.kind])
    settings=dict(config['training'])
    if a.kind in FOUNDATIONS:settings.update(config['foundation_training_overrides'])
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA GPU is not available; no CPU production fallback')
    if a.action=='fit':
        inner,_=fold.load('inner')
        source=DynamicPassageEpochSource(context['data_fold'],config['seed'],expected_binding=binding)
        stream=lambda epoch:source.iter_batches(epoch,settings['microbatch'],with_grid=False)
        fit_neural(model,a.kind,receipt,stream,inner,prior,binding,policy,settings,a.output,device=device,dynamic_source=source)
    else:
        if a.model_dir is None:raise ValueError('Explicit completed model directory required')
        summary=json.loads((a.model_dir/'summary.json').read_text())
        checkpoint_path=a.model_dir/'best.pt'
        if summary.get('status')!='completed' or summary.get('fixture_only') is not False or file_sha256(checkpoint_path)!=summary['checkpoint_sha256']:
            raise ValueError('A completed production checkpoint with matching SHA is required')
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        restore_into(model,checkpoint,kind=a.kind,binding=binding,prior_sha=prior['shared_prior_payload_sha256'],policy=policy)
        frozen=json.loads(a.frozen_policy.read_text()) if a.frozen_policy else None
        query,_=fold.load(a.role,frozen_policy=frozen)
        prediction=predict_neural(model.to(device),query['db'],prior,binding,policy,device=device,batch_size=settings['evaluation_batch'])
        if a.output.exists():raise FileExistsError('Prediction output already exists')
        a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**prediction)


if __name__=='__main__':main()
