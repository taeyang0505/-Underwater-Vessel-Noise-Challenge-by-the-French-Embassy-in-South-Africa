"""Small, explicitly CPU-only R6 mechanism study. No challenge query I/O."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()


def save_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def metrics(data,pred):
    from sklearn.metrics import roc_auc_score
    y=np.asarray(data['share'],float);p=np.asarray(pred['share'],float)
    e=p-y;sites=np.asarray(data['site']).astype(str)
    basic=lambda ix:{'n':int(ix.sum()),'rmse':float(np.sqrt(np.mean(e[ix]**2))),'mae':float(np.abs(e[ix]).mean()),'bias':float(e[ix].mean())}
    slices={s:basic(sites==s) for s in sorted(set(sites))}
    out={'n':len(y),'rmse':float(np.sqrt(np.mean(e**2))),'site_macro_rmse':float(np.mean([v['rmse'] for v in slices.values()])),
         'mae':float(np.abs(e).mean()),'bias':float(e.mean()),'site_slices':slices}
    z=np.asarray(data['presence']);score=np.asarray(pred['score'],float)
    out['presence_auc']=float(roc_auc_score(z,score)) if len(np.unique(z))==2 else None
    out['presence_auc_share_only']=float(roc_auc_score(z,p)) if len(np.unique(z))==2 else None
    if 'score_classifier' in pred:out['presence_auc_classifier_only']=float(roc_auc_score(z,pred['score_classifier'])) if len(np.unique(z))==2 else None
    if 'family' in data:
        fam=np.asarray(data['family']).astype(str);out['family_slices']={s:basic(fam==s) for s in sorted(set(fam))}
    masks={'zero':y==0,'positive_low_to_.1':(y>0)&(y<=.1),'.1_to_.4':(y>.1)&(y<=.4),'.4_to_1':y>.4}
    out['target_slices']={s:basic(ix) if ix.any() else {'n':0} for s,ix in masks.items()}
    return out


def grid(db,mode,prior=None):
    from pipeline.relative_grid_r6_v2 import relative_grid_v2,quantized_db
    if mode=='relative8':return relative_grid_v2(db)
    if mode!='legacy8' or prior is None:raise ValueError('Explicit legacy TRAIN prior required')
    from pipeline.mixtures import hybrid_cnn_channels
    # Both controls use the same fixed quantization and rounded-energy pooling.
    x,_=hybrid_cnn_channels(quantized_db(db).astype('float32'),prior)
    _,w=relative_grid_v2(db)
    return x,w


def prior_for(train,mode,seed):
    if mode!='legacy8':return None
    from pipeline.background_prior_features import fit_background_prior
    # Original precision is needed to recover TRAIN B=(1-q)E correctly.
    return fit_background_prior(train['unrounded_db'],train['cell_share'],seed=seed,clusters=32)


def predict_cnn(model,x,w,batch=128):
    import torch
    model.eval();out={'share':[],'score':[],'cell_mask':[]}
    with torch.inference_mode():
        for start in range(0,len(x),batch):
            y,z,q=model(torch.from_numpy(x[start:start+batch]),torch.from_numpy(w[start:start+batch]))
            out['share'].append(y.numpy());out['score'].append(z.sigmoid().numpy());out['cell_mask'].append(q.numpy())
    return {k:np.concatenate(v) for k,v in out.items()}


def fit_cnn(train,inner,arm,cfg,folder,progress=None):
    import torch
    from pipeline.relative_models_r6_v2 import TinyMaskCNNv2
    folder=Path(folder)
    if folder.exists():raise FileExistsError('No partial checkpoint reuse: '+str(folder))
    folder.mkdir(parents=True);start=time.monotonic();s=cfg['neural'];seed=int(cfg['seed'])
    torch.set_num_threads(s['threads']);torch.manual_seed(seed)
    prior=prior_for(train,arm['channels'],seed)
    if prior is not None:save_json(folder/'legacy_train_prior.json',prior)
    tx,tw=grid(train['db'],arm['channels'],prior);vx,vw=grid(inner['db'],arm['channels'],prior)
    model_config={'input_channels':tx.shape[1],'width':s['width'],'blocks':s['blocks'],'dropout':s['dropout'],'global_context':arm.get('global_context',True)}
    model=TinyMaskCNNv2(**model_config)
    opt=torch.optim.Adam(model.parameters(),lr=s['learning_rate'],weight_decay=s['weight_decay'])
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=2,min_lr=1e-6)
    best=float('inf');stale=0;best_epoch=0;hist=[]
    yt=torch.from_numpy(train['share'].astype('float32'));qt=torch.from_numpy(train['cell_share'].astype('float32'));zt=torch.from_numpy(train['presence'].astype('float32'))
    for epoch in range(1,s['epochs']+1):
        model.train();order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(tx));ss=0.;total=0.;count=0
        for start_i in range(0,len(order),s['batch_size']):
            ix=order[start_i:start_i+s['batch_size']];opt.zero_grad(set_to_none=True)
            wt=torch.from_numpy(tw[ix]);p,z,q=model(torch.from_numpy(tx[ix]),wt)
            share=(p-yt[ix]).square();cell=((q-qt[ix]).square()*wt).sum((1,2));presence=torch.nn.functional.binary_cross_entropy_with_logits(z,zt[ix],reduction='none')
            loss=(share+s['cell_loss_weight']*cell+s['presence_loss_weight']*presence).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),s['gradient_clip'],error_if_nonfinite=True);opt.step()
            ss+=float(share.detach().sum());total+=float(loss.detach())*len(ix);count+=len(ix)
        pred=predict_cnn(model,vx,vw,s['batch_size']);m=metrics(inner,pred);score=m['site_macro_rmse']
        if score<best-s['minimum_improvement']:
            best=score;stale=0;best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'model_config':model_config,'epoch':epoch,'arm':arm,'seed':seed},folder/'best.tmp.pt');(folder/'best.tmp.pt').replace(folder/'best.pt')
        else:stale+=1
        row={'arm':arm['name'],'epoch':epoch,'train_rmse':float(np.sqrt(ss/count)),'train_loss':total/count,'inner_rmse':m['rmse'],'inner_site_macro_rmse':score,
             'best_inner_site_macro_rmse':best,'best_epoch':best_epoch,'patience_used':stale,'learning_rate':opt.param_groups[0]['lr'],'elapsed_seconds':time.monotonic()-start}
        hist.append(row)
        with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if progress:progress(row)
        print(json.dumps(row,allow_nan=False),flush=True)
        sched.step(score)
        if stale>=s['early_stopping_patience']:break
    ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict'])
    pred=predict_cnn(model,vx,vw,s['batch_size']);np.savez_compressed(folder/'inner_predictions.npz',**pred)
    summary={'status':'completed','arm':arm,'model_config':model_config,'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,'epochs_run':len(hist),
             'selection':'inner_site_macro_rmse','best_inner_metrics':metrics(inner,pred),'stop_reason':'patience' if stale>=s['early_stopping_patience'] else 'maximum_epochs',
             'device':'Mac CPU FP32','pilot_only':True,'heldout_evaluated':False,'elapsed_seconds':time.monotonic()-start,'checkpoint_sha256':sha(folder/'best.pt'),
             'prior_sha256':None if prior is None else sha(folder/'legacy_train_prior.json')}
    save_json(folder/'summary.json',summary)
    return summary


def tree_features(db,mode,prior):
    from pipeline.physics_features_r6_v2 import physics_features_r6_v2
    from pipeline.relative_grid_r6_v2 import quantized_db
    db=quantized_db(db)
    x,w=grid(db,mode,prior)
    # No IDs/labels/prior fitting at transform time. Channel-statistics are local.
    stats=np.concatenate([x.mean(2).reshape(len(x),-1),x.std(2).reshape(len(x),-1),x.max(2).reshape(len(x),-1),x.min(2).reshape(len(x),-1)],1)
    return np.concatenate([x.reshape(len(x),-1),stats,physics_features_r6_v2(db)],1).astype('float32')


def fit_tree(train,inner,arm,cfg,folder,progress=None):
    from lightgbm import LGBMRegressor,LGBMClassifier,early_stopping,log_evaluation
    folder=Path(folder)
    if folder.exists():raise FileExistsError('No partial tree reuse')
    folder.mkdir(parents=True);start=time.monotonic();s=cfg['tree'];seed=int(cfg['seed'])
    prior=prior_for(train,arm['channels'],seed)
    if prior is not None:save_json(folder/'legacy_train_prior.json',prior)
    x=tree_features(train['db'],arm['channels'],prior);v=tree_features(inner['db'],arm['channels'],prior)
    groups=np.asarray(inner['site']).astype(str)
    def macro_metric(y,p):
        e=np.clip(np.asarray(p),0,1)-y
        return 'site_macro_rmse',float(np.mean([np.sqrt(np.mean(e[groups==g]**2)) for g in sorted(set(groups))])),False
    args={'n_estimators':s['maximum_trees'],'learning_rate':s['learning_rate'],'num_leaves':s['num_leaves'],'min_child_samples':s['minimum_leaf_rows'],
          'max_bin':127,'n_jobs':s['threads'],'random_state':seed,'verbosity':-1,'colsample_bytree':.85,'subsample':.85,'subsample_freq':1}
    reg=LGBMRegressor(objective='regression_l2',metric='None',**args)
    reg.fit(x,train['share'],eval_set=[(v,inner['share'])],eval_metric=macro_metric,callbacks=[early_stopping(s['early_stopping_patience'],verbose=False),log_evaluation(0)])
    clf=LGBMClassifier(objective='binary',**args)
    clf.fit(x,train['presence'],eval_set=[(v,inner['presence'])],eval_metric='binary_logloss',callbacks=[early_stopping(s['early_stopping_patience'],verbose=False),log_evaluation(0)])
    reg.booster_.save_model(str(folder/'share.txt'));clf.booster_.save_model(str(folder/'presence.txt'))
    p=np.clip(reg.predict(v),0,1).astype('float32');score=clf.predict_proba(v)[:,1].astype('float32')
    pred={'share':p,'score':.5*p+.5*score,'score_classifier':score}
    np.savez_compressed(folder/'inner_predictions.npz',**pred)
    summary={'status':'completed','arm':arm,'n_features':x.shape[1],'best_iteration':int(reg.best_iteration_),'presence_best_iteration':int(clf.best_iteration_),
             'maximum_trees':s['maximum_trees'],'selection':'inner_site_macro_rmse','best_inner_metrics':metrics(inner,pred),'pilot_only':True,'heldout_evaluated':False,
             'device':'Mac CPU','elapsed_seconds':time.monotonic()-start,'model_sha256':{n:sha(folder/n) for n in ['share.txt','presence.txt']},
             'prior_sha256':None if prior is None else sha(folder/'legacy_train_prior.json'),'presence_selection':'INNER binary_logloss'}
    save_json(folder/'summary.json',summary)
    if progress:progress({'arm':arm['name'],'status':'completed','inner_rmse':summary['best_inner_metrics']['rmse'],'inner_site_macro_rmse':summary['best_inner_metrics']['site_macro_rmse']})
    print(json.dumps(summary,allow_nan=False),flush=True)
    return summary


def evaluate_frozen(folder,data,kind):
    folder=Path(folder);summary=json.loads((folder/'summary.json').read_text());arm=summary['arm']
    if summary.get('prior_sha256') is not None:
        if sha(folder/'legacy_train_prior.json')!=summary['prior_sha256']:raise ValueError('TRAIN prior changed')
        prior=json.loads((folder/'legacy_train_prior.json').read_text())
    else:
        if (folder/'legacy_train_prior.json').exists():raise ValueError('Unexpected prior for relative-only arm')
        prior=None
    if kind=='cnn':
        import torch
        from pipeline.relative_models_r6_v2 import TinyMaskCNNv2
        if sha(folder/'best.pt')!=summary['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model=TinyMaskCNNv2(**ck['model_config']);model.load_state_dict(ck['state_dict'])
        x,w=grid(data['db'],arm['channels'],prior);p=predict_cnn(model,x,w)
    else:
        from lightgbm import Booster
        for n,h in summary['model_sha256'].items():
            if sha(folder/n)!=h:raise ValueError('Tree changed')
        x=tree_features(data['db'],arm['channels'],prior)
        share=np.clip(Booster(model_file=str(folder/'share.txt')).predict(x,num_threads=4),0,1).astype('float32');score=Booster(model_file=str(folder/'presence.txt')).predict(x,num_threads=4).astype('float32')
        p={'share':share,'score':.5*share+.5*score,'score_classifier':score}
    return metrics(data,p),p
