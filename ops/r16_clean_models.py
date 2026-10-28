"""Small, TRAIN-only cross-operator learning experiment. Never opens Test.

All architecture/hyperparameters/ensemble constraints are fixed here before
the new panels are evaluated. Production R9/R14 sources remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from pipeline.relative_grid_r6_v2 import relative_grid_v2
from pipeline.features_r12 import compact_relative_features
from pipeline.relative_models_r6_v2 import TinyMaskCNNv2

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026102716
EPOCHS = 100
TEMPORAL_CHANNELS = (0, 1, 2, 3, 6)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    q.replace(p)


def safe_panel(d):
    x,w = relative_grid_v2(d['db'])
    y = np.asarray(d['share'], dtype=np.float32)
    q = np.asarray(d['cell_share'], dtype=np.float32)
    z = np.asarray(d['presence'], dtype=np.int64)
    assert x.shape == (len(y),8,15,25) and q.shape == w.shape
    assert np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(q).all() and np.isfinite(y).all()
    assert np.array_equal(z, y>0) and np.all(q[w==0] == 0)
    assert np.max(np.abs((q*w).sum((1,2))-y)) < .0058
    return x,w,y,q,z


def temporal_features(db):
    """Independent classification: no profile, energy weights or PCA.

    Fixed per-band temporal statistics and correlations are invariant to a
    static band dB offset when rounding commutes. This is not invariance to
    changing only the physical background component before mixing.
    """
    x,w = relative_grid_v2(db)
    obs = w > 0
    count = obs.sum(1)
    cols = []
    for c in TEMPORAL_CHANNELS:
        a = x[:,c].astype(np.float64)
        mean = np.divide(a.sum(1), count, out=np.zeros_like(count,dtype=float), where=count>0)
        var = np.divide(np.where(obs,(a-mean[:,None])**2,0).sum(1),count,out=np.zeros_like(mean),where=count>0)
        mx = np.where(count>0,np.max(np.where(obs,a,-np.inf),axis=1),0)
        cols.extend([mean,np.sqrt(np.maximum(var,0)),mx])
    # Pairwise temporal correlations use relative channel0 and only common
    # observed minutes. No across-window fitted centring or scale is used.
    a = x[:,0].astype(np.float64)
    for f in range(25):
        for g in range(f+1,25):
            m = obs[:,:,f] & obs[:,:,g]
            n = m.sum(1)
            u,v = a[:,:,f],a[:,:,g]
            mu=np.divide(np.where(m,u,0).sum(1),n,out=np.zeros(len(a)),where=n>0)
            mv=np.divide(np.where(m,v,0).sum(1),n,out=np.zeros(len(a)),where=n>0)
            uu=np.where(m,u-mu[:,None],0);vv=np.where(m,v-mv[:,None],0)
            denom=np.sqrt((uu*uu).sum(1)*(vv*vv).sum(1))
            corr=np.divide((uu*vv).sum(1),denom,out=np.zeros(len(a)),where=denom>1e-12)
            cols.append(corr[:,None])
    cols.append(count.astype(float)/15.)
    result=np.concatenate(cols,axis=1).astype(np.float32)
    assert np.isfinite(result).all()
    return result


class SmallDecoupledCNN(nn.Module):
    """8-channel energy mask regressor plus a separate temporal classifier.

    No parameter is shared between the regression and classification paths.
    The BCE gradient cannot change the cell-mask estimator.
    """
    def __init__(self):
        super().__init__()
        self.reg = TinyMaskCNNv2(input_channels=8,width=24,blocks=2,dropout=.1,global_context=True)
        self.clf = nn.Sequential(nn.Conv2d(6,16,3,padding=1),nn.GELU(),nn.Dropout2d(.1),
                                 nn.Conv2d(16,16,3,padding=1),nn.GELU())
        self.clf_head=nn.Sequential(nn.Linear(32,16),nn.GELU(),nn.Dropout(.1),nn.Linear(16,1))

    def forward(self,x,w):
        h=self.reg.body(x)
        pooled=torch.cat([h[:,:,:,a:b].mean(3) for a,b in [(10,20),(20,23),(15,23)]],1)
        context=torch.nn.functional.gelu(self.reg.context_projection(pooled))*torch.sigmoid(self.reg.context_gate(pooled))
        q=self.reg.mask_head(h+context.unsqueeze(3))[:,0].sigmoid()
        obs=w>0;q=torch.where(obs,q,torch.zeros_like(q))
        p=(q*w).sum((1,2))
        # The mask channel supplies missingness, never normalized energy.
        h=self.clf(x[:,[0,1,2,3,6,5]])
        om=obs[:,None].to(h.dtype);n=om.sum((2,3)).clamp_min(1)
        avg=(h*om).sum((2,3))/n
        mx=h.masked_fill(~obs[:,None],-torch.inf).amax((2,3))
        logit=self.clf_head(torch.cat([avg,mx],1))[:,0]
        return p,q,logit


def metrics(y,p,clf,site=None):
    y=np.asarray(y,float);p=np.asarray(p,float);clf=np.asarray(clf,float);z=y>0;e=p-y
    out={'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),
         'bias':float(e.mean()),'target_mean':float(y.mean()),'prediction_mean':float(p.mean()),
         'target_rms':float(np.sqrt(np.mean(y*y))),'prediction_rms':float(np.sqrt(np.mean(p*p))),
         'auc_share':float(roc_auc_score(z,p)) if len(np.unique(z))==2 else None,
         'auc_classifier':float(roc_auc_score(z,clf)) if len(np.unique(z))==2 else None,
         'auc_75share_25clf':float(roc_auc_score(z,.75*p+.25*clf)) if len(np.unique(z))==2 else None,
         'auc_25share_75clf':float(roc_auc_score(z,.25*p+.75*clf)) if len(np.unique(z))==2 else None,
         'classifier_ge099_fraction':float(np.mean(clf>=.99)),
         'classifier_le001_fraction':float(np.mean(clf<=.01))}
    if site is not None:
        s=np.asarray(site).astype(str)
        out['site_slices']={k:metrics(y[s==k],p[s==k],clf[s==k]) for k in sorted(set(s))}
        out['site_macro_rmse']=float(np.mean([v['rmse'] for v in out['site_slices'].values()]))
    return out


@torch.no_grad()
def nn_predict(model,x,w,batch=256):
    model.eval();p=[];clf=[]
    for i in range(0,len(x),batch):
        pp,_,ll=model(torch.from_numpy(x[i:i+batch]).contiguous(memory_format=torch.channels_last),torch.from_numpy(w[i:i+batch]))
        p.append(pp.numpy());clf.append(ll.sigmoid().numpy())
    return np.concatenate(p),np.concatenate(clf)


def train_arm(train, fit, output, arm):
    folder=Path(output)/arm;folder.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(SEED)
    tx,tw,y,q,z=safe_panel(train);vx,vw,vy,_,vz=safe_panel(fit)
    regf=compact_relative_features(train['db'],include_shape=False)
    clf=temporal_features(train['db']);vr=compact_relative_features(fit['db'],include_shape=False);vc=temporal_features(fit['db'])
    common=dict(n_estimators=600,learning_rate=.04,num_leaves=23,max_depth=-1,min_child_samples=30,
                reg_lambda=2.,feature_fraction=.85,n_jobs=2,verbosity=-1,random_state=SEED,
                deterministic=True,force_col_wise=True)
    reg=lgb.LGBMRegressor(objective='regression',**common)
    classifier=lgb.LGBMClassifier(objective='binary',**common)
    reg.fit(regf,y,eval_set=[(vr,vy)],eval_metric='rmse',callbacks=[lgb.early_stopping(40,verbose=False)])
    classifier.fit(clf,z,eval_set=[(vc,vz)],eval_metric='binary_logloss',callbacks=[lgb.early_stopping(40,verbose=False)])
    reg.booster_.save_model(str(folder/'tree_regressor.txt'));classifier.booster_.save_model(str(folder/'tree_classifier.txt'))
    tree_p=np.clip(reg.predict(vr),0,1);tree_c=classifier.predict_proba(vc)[:,1]
    np.savez_compressed(folder/'fit_tree_predictions.npz',share=tree_p,classifier_prob=tree_c)
    write(folder/'TREE_COMPLETED.json',{'arm':arm,'regressor_features':regf.shape[1],'classifier_features':clf.shape[1],
          'regressor_best_iteration':reg.best_iteration_,'classifier_best_iteration':classifier.best_iteration_,
          'metrics':metrics(vy,tree_p,tree_c),'elapsed_seconds':time.perf_counter()-began})
    print(json.dumps({'event':'tree_done','arm':arm,'seconds':time.perf_counter()-began}),flush=True)
    model=SmallDecoupledCNN().to(memory_format=torch.channels_last)
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=1e-5)
    xt=torch.from_numpy(tx);wt=torch.from_numpy(tw);yt=torch.from_numpy(y);qt=torch.from_numpy(q);zt=torch.from_numpy(z.astype(np.float32))
    best=float('inf');best_epoch=0;best_auc=-1.;best_clf_epoch=0
    for epoch in range(1,EPOCHS+1):
        model.train();order=np.random.default_rng(np.random.SeedSequence([SEED,epoch])).permutation(len(y))
        train_sse=0;total=0
        for start in range(0,len(order),128):
            ix=order[start:start+128];opt.zero_grad(set_to_none=True)
            p,mask,logit=model(xt[ix].contiguous(memory_format=torch.channels_last),wt[ix])
            mse=(p-yt[ix]).square()
            regression=mse+.25*((mask-qt[ix]).square()*wt[ix]).sum((1,2))
            classification=nn.functional.binary_cross_entropy_with_logits(logit,zt[ix],reduction='none')
            loss=regression.mean()+classification.mean()
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.reg.parameters(),1.,error_if_nonfinite=True)
            nn.utils.clip_grad_norm_(list(model.clf.parameters())+list(model.clf_head.parameters()),1.,error_if_nonfinite=True)
            opt.step()
            train_sse+=float(mse.detach().sum());total+=len(ix)
        pp,cc=nn_predict(model,vx,vw);mm=metrics(vy,pp,cc)
        if mm['rmse']<best:
            best=mm['rmse'];best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'epoch':epoch,'arm':arm},folder/'best_regression.pt')
        if mm['auc_classifier']>best_auc:
            best_auc=mm['auc_classifier'];best_clf_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'epoch':epoch,'arm':arm},folder/'best_classifier.pt')
        rec={'epoch':epoch,'train_online_rmse':float(np.sqrt(train_sse/total)),
             'fit_rmse':mm['rmse'],'fit_classifier_auc':mm['auc_classifier'],'best_regression_epoch':best_epoch,
             'best_regression_rmse':best,'best_classifier_epoch':best_clf_epoch,'best_classifier_auc':best_auc,
             'elapsed_seconds':time.perf_counter()-began,'learning_rate':opt.param_groups[0]['lr']}
        with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        write(folder/'progress.json',rec)
        if epoch==1 or epoch%10==0:print(json.dumps({'event':'epoch','arm':arm,**rec}),flush=True)
        scheduler.step()
    # Independent paths select their own checkpoint using only local_fit.
    model.load_state_dict(torch.load(folder/'best_regression.pt',map_location='cpu',weights_only=True)['state_dict'])
    pp,_=nn_predict(model,vx,vw)
    model.load_state_dict(torch.load(folder/'best_classifier.pt',map_location='cpu',weights_only=True)['state_dict'])
    _,cc=nn_predict(model,vx,vw)
    np.savez_compressed(folder/'fit_nn_predictions.npz',share=pp,classifier_prob=cc)
    result={'status':'completed','arm':arm,'epochs':EPOCHS,'early_stopping':False,'optimizer':'Adam',
            'dropout':.1,'parameters_total':sum(p.numel() for p in model.parameters()),
            'regression_parameters':sum(p.numel() for p in model.reg.parameters()),
            'classification_parameters':sum(p.numel() for n,p in model.named_parameters() if not n.startswith('reg.')),
            'best_regression_epoch':best_epoch,'best_classifier_epoch':best_clf_epoch,
            'fit_nn_metrics':metrics(vy,pp,cc),'fit_tree_metrics':metrics(vy,tree_p,tree_c),
            'fit_fixed60tree40nn_metrics':metrics(vy,.6*tree_p+.4*pp,.5*tree_c+.5*cc),
            'elapsed_seconds':time.perf_counter()-began,'Test_used':False,'report_or_gate_read':False,
            'checkpoint_sha256':{p.name:sha(p) for p in folder.glob('*.pt')},
            'tree_sha256':{p.name:sha(p) for p in folder.glob('*.txt')}}
    write(folder/'COMPLETED.json',result)
    return result


def load_arm(folder):
    folder=Path(folder);model=SmallDecoupledCNN().to(memory_format=torch.channels_last)
    reg=lgb.Booster(model_file=str(folder/'tree_regressor.txt'));clf=lgb.Booster(model_file=str(folder/'tree_classifier.txt'))
    return model,reg,clf


def predict_arm(folder,panel):
    folder=Path(folder);model,reg,clf=load_arm(folder)
    x,w,_,_,_=safe_panel(panel)
    model.load_state_dict(torch.load(folder/'best_regression.pt',map_location='cpu',weights_only=True)['state_dict'])
    nn_p,_=nn_predict(model,x,w)
    model.load_state_dict(torch.load(folder/'best_classifier.pt',map_location='cpu',weights_only=True)['state_dict'])
    _,nn_c=nn_predict(model,x,w)
    tree_p=np.clip(reg.predict(compact_relative_features(panel['db'],include_shape=False),num_threads=2),0,1)
    tree_c=clf.predict(temporal_features(panel['db']),num_threads=2)
    return {'tree_share':tree_p,'tree_classifier':tree_c,'nn_share':nn_p,'nn_classifier':nn_c,
            'balanced_share':.6*tree_p+.4*nn_p,'balanced_classifier':.5*tree_c+.5*nn_c}


def concatenate(panels):
    keys=('db','share','cell_share','presence','site')
    result={k:np.concatenate([d[k] for d in panels]) for k in keys if k in panels[0]}
    return result


def fixture():
    rng=np.random.default_rng(SEED);db=np.round(rng.normal(80,6,(4,15,25)),1);db[0,0]=np.nan
    shifts=np.round(rng.normal(0,8,25),1)[None,None]
    a=temporal_features(db);b=temporal_features(db+shifts)
    assert np.allclose(a,b,atol=2e-6,rtol=0)
    model=SmallDecoupledCNN();x,w=relative_grid_v2(db)
    p,q,l=model(torch.from_numpy(x),torch.from_numpy(w))
    assert torch.allclose(p,(q*torch.from_numpy(w)).sum((1,2)))
    for parameter in model.parameters():parameter.grad=None
    l.sum().backward()
    assert all(parameter.grad is None for parameter in model.reg.parameters())
    return {'status':'passed','temporal_band_offset_max_diff':float(abs(a-b).max()),
            'classifier_to_regression_gradient_exists':False,
            'parameter_count':sum(p.numel() for p in model.parameters()),'code_sha256':sha(__file__)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['fixture','train'])
    parser.add_argument('--manifest');parser.add_argument('--output',default='models/r16_clean_pilot_r1')
    args=parser.parse_args()
    if args.command=='fixture':print(json.dumps(fixture()))
    else:
        from ops.r16_forward_data import load_training,load_panel
        out=ROOT/args.output;out.mkdir(exist_ok=False)
        plan={'seed':SEED,'epochs':EPOCHS,'NN_early_stopping':False,'optimizer':'Adam','regressor_features':'R6v2 relative8, compact466 tree; no PCA/LFanchor',
              'classifier_features':'temporal-only, no w/profile/PCA/anchor; separate NN parameters',
              'arms':['control','mixed'],'selection':'all local_fit operators equal weight, no gate/report access during fitting',
              'fixed_small_ensemble':{'tree_regressor':.6,'NN_regressor':.4,'tree_classifier':.5,'NN_classifier':.5},
              'Test_used':False,'code_sha256':sha(__file__),'manifest_sha256':sha(args.manifest)}
        write(out/'PLAN.json',plan)
        operators=list(json.loads(Path(args.manifest).read_text())['panels']['local_fit'])
        fit=concatenate([load_panel(args.manifest,'local_fit',op) for op in operators])
        results={}
        for arm in plan['arms']:
            train=load_training(args.manifest,arm=arm)
            results[arm]=train_arm(train,fit,out,arm)
        write(out/'ALL_COMPLETED.json',{'status':'all_small_arms_completed','results':results,'plan_sha256':sha(out/'PLAN.json')})
