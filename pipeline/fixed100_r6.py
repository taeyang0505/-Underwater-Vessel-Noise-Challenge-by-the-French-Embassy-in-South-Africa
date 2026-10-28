"""Fixed-duration CPU retraining; no early stopping and no query-data input."""
from pathlib import Path
import json,math,time,resource,sys
import numpy as np
from pipeline.cpu_ablation_r6_v2 import sha,save_json,grid,prior_for,predict_cnn,metrics


def fit_fixed100(train,inner,arm,cfg,folder,progress=None):
    import torch
    from pipeline.relative_models_r6_v2 import TinyMaskCNNv2
    folder=Path(folder)
    if folder.exists():raise FileExistsError('Fresh arm output required; no partial-run reuse')
    s=cfg['neural'];epochs=s['epochs']
    if epochs!=100 or s['early_stopping_enabled'] is not False or s['optimizer']!='adam':raise ValueError('Exactly100 epochs, Adam, no ES required')
    folder.mkdir(parents=True);(folder/'checkpoints').mkdir();start=time.monotonic();seed=cfg['seed']
    torch.set_num_threads(s['threads']);torch.manual_seed(seed)
    prior=prior_for(train,arm['channels'],seed)
    if prior is not None:save_json(folder/'legacy_train_prior.json',prior)
    tx,tw=grid(train['db'],arm['channels'],prior);vx,vw=grid(inner['db'],arm['channels'],prior)
    mc={'input_channels':tx.shape[1],'width':s['width'],'blocks':s['blocks'],'dropout':s['dropout'],'global_context':arm.get('global_context',True)}
    model=TinyMaskCNNv2(**mc);opt=torch.optim.Adam(model.parameters(),lr=s['learning_rate'],weight_decay=s['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=2,min_lr=1e-6)
    yt=torch.from_numpy(train['share'].astype('float32'));qt=torch.from_numpy(train['cell_share'].astype('float32'));zt=torch.from_numpy(train['presence'].astype('float32'))
    best_window=float('inf');best_macro=float('inf');window_epoch=macro_epoch=0;hist=[];steps=0
    shadow_best=float('inf');shadow_stale=0;shadow_epoch=0;shadow_stop=None
    save_json(folder/'started.json',{'arm':arm,'model_config':mc,'seed':seed,'device':'CPU','epochs_required':100,'early_stopping_enabled':False})
    for epoch in range(1,101):
        model.train();order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(tx));ss=total=0.;count=0
        for begin in range(0,len(order),s['batch_size']):
            ix=order[begin:begin+s['batch_size']];opt.zero_grad(set_to_none=True);wt=torch.from_numpy(tw[ix])
            y,z,q=model(torch.from_numpy(tx[ix]),wt);share=(y-yt[ix]).square();cell=((q-qt[ix]).square()*wt).sum((1,2))
            presence=torch.nn.functional.binary_cross_entropy_with_logits(z,zt[ix],reduction='none')
            loss=(share+s['cell_loss_weight']*cell+s['presence_loss_weight']*presence).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; not an early stop')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),s['gradient_clip'],error_if_nonfinite=True);opt.step();steps+=1
            ss+=float(share.detach().sum());total+=float(loss.detach())*len(ix);count+=len(ix)
        pred=predict_cnn(model,vx,vw,s['batch_size']);m=metrics(inner,pred);window=m['rmse'];macro=m['site_macro_rmse']
        if not math.isfinite(window) or not math.isfinite(macro):raise FloatingPointError('Nonfinite INNER metric')
        # Strict minima, no min_delta exclusion: retain the first exact tie.
        if window<best_window:best_window=window;window_epoch=epoch
        if macro<best_macro:best_macro=macro;macro_epoch=epoch
        # Counterfactual accounting only. It NEVER controls optimization.
        if shadow_stop is None:
            if macro<shadow_best-1e-5:shadow_best=macro;shadow_epoch=epoch;shadow_stale=0
            else:shadow_stale+=1
            if shadow_stale>=7:shadow_stop=epoch
        lr_used=opt.param_groups[0]['lr'];scheduler.step(macro)
        ck={'state_dict':model.state_dict(),'model_config':mc,'epoch':epoch,'arm':arm,'seed':seed,
            'inner_window_rmse':window,'inner_site_macro_rmse':macro,'selection_inputs':'own INNER only'}
        target=folder/'checkpoints'/f'epoch_{epoch:03d}.pt';tmp=target.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(target)
        row={'arm':arm['name'],'epoch':epoch,'train_rmse':float(np.sqrt(ss/count)),'train_loss':total/count,
             'inner_rmse':window,'inner_site_macro_rmse':macro,'best_inner_window_rmse':best_window,'best_window_epoch':window_epoch,
             'best_inner_site_macro_rmse':best_macro,'best_macro_epoch':macro_epoch,'learning_rate':lr_used,
             'next_learning_rate':opt.param_groups[0]['lr'],'optimizer_steps':steps,'elapsed_seconds':time.monotonic()-start,
             'shadow_ES7_stop_epoch':shadow_stop,'shadow_ES7_best_macro_epoch':shadow_epoch,'shadow_ES7_stale_epochs':shadow_stale,
             'process_peak_RSS_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)}
        hist.append(row)
        with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if progress:progress(row)
        print(json.dumps(row,allow_nan=False),flush=True)
        # No break on patience, LR plateau or any validation score.
    from shutil import copyfile
    for name,epoch in [('best_window.pt',window_epoch),('best_macro.pt',macro_epoch),('last.pt',100)]:
        copyfile(folder/'checkpoints'/f'epoch_{epoch:03d}.pt',folder/name)
    # Full state is kept for provenance; this runner never resumes partial fits.
    torch.save({'epoch':100,'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
                'torch_rng_state':torch.get_rng_state(),'optimizer_steps':steps},folder/'final_training_state.pt')
    selected={}
    for label,file,epoch in [('window','best_window.pt',window_epoch),('macro','best_macro.pt',macro_epoch),('last','last.pt',100)]:
        ck=torch.load(folder/file,map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict'])
        p=predict_cnn(model,vx,vw,s['batch_size']);selected[label]={'epoch':epoch,'metrics':metrics(inner,p),'checkpoint':file}
        np.savez_compressed(folder/f'{label}_inner_predictions.npz',**p)
    ck=torch.load(folder/'best_window.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict'])
    train_eval=metrics(train,predict_cnn(model,tx,tw,s['batch_size']))
    outputs=['best_window.pt','best_macro.pt','last.pt','final_training_state.pt','window_inner_predictions.npz','macro_inner_predictions.npz','last_inner_predictions.npz']
    summary={'status':'completed','arm':arm,'model_config':mc,'parameters':sum(p.numel() for p in model.parameters()),
        'epochs_run':len(hist),'optimizer_steps':steps,'early_stopping_enabled':False,'stop_reason':'completed_100_epochs',
        'primary_selection':'strict lowest own INNER window RMSE; first exact tie',
        'best_inner_window':selected['window'],'best_inner_macro':selected['macro'],'final_epoch':selected['last'],
        'best_window_TRAIN_eval':train_eval,'shadow_ES7':{'stop_epoch':shadow_stop,'best_macro_epoch':shadow_epoch,'best_macro_rmse':shadow_best,
            'stopping_used_for_training':False,'minimum_improvement':1e-5},
        'device':'Mac CPU FP32','threads':s['threads'],'elapsed_seconds':time.monotonic()-start,
        'process_peak_RSS_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
        'artifact_sha256':{n:sha(folder/n) for n in outputs},'epoch_checkpoint_sha256':{p.name:sha(p) for p in sorted((folder/'checkpoints').glob('*.pt'))},
        'prior_sha256':None if prior is None else sha(folder/'legacy_train_prior.json'),'heldout_evaluated':False,
        'scope':'same source pilot retraining; not full5fold, not global minimum guarantee'}
    if summary['epochs_run']!=100 or len(summary['epoch_checkpoint_sha256'])!=100:raise RuntimeError('100epoch artifact coverage failed')
    save_json(folder/'summary.json',summary);return summary
