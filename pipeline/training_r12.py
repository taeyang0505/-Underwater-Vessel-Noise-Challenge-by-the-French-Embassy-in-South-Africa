"""R12 small CPU ablation: physical mask objective, fixed100, no Test I/O.

The legacy contrast objective exists ONLY as one explicitly labelled control.
All candidates report share as score; max-min contrast is diagnostic only.
"""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from pipeline.relative_models_r6_v2 import TinyMaskCNNv2,fixed_presence_score

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)

class R12MaskCNN(TinyMaskCNNv2):
    """Same27k CPU body across arms; unused shape channels are zero padded."""
    def __init__(self):super().__init__(input_channels=11,width=24,blocks=2,dropout=.1,global_context=True)
    def add_global_context(self,h):
        # No context summary above nominal12kHz Nyquist. Original R9 unchanged.
        pooled=torch.cat([h[:,:,:,a:b].mean(3) for a,b in [(10,20),(20,23),(15,23)]],1)
        return h+(torch.nn.functional.gelu(self.context_projection(pooled))*torch.sigmoid(self.context_gate(pooled))).unsqueeze(3)
    def forward(self,x,w):
        h=self.add_global_context(self.body(x));q=self.mask_head(h)[:,0].sigmoid()
        q=torch.where(w>0,q,torch.zeros_like(q));p=(w*q).sum((1,2))
        return p,q

def objective(p,q,w,y,truth,*,legacy_control=False):
    share=(p-y).square();cell=((q-truth).square()*w).sum((1,2))
    loss=share+.25*cell
    if legacy_control:
        score,_,_=fixed_presence_score(q,w)
        loss=loss+.05*nn.functional.binary_cross_entropy(score.clamp(1e-6,1-1e-6),(y>0).float(),reduction='none')
    return loss,share,cell

def channels(db,basis,include_shape):
    from pipeline.features_r12 import transform_channels
    x,w=transform_channels(db,basis=basis,include_shape=include_shape)
    if x.shape[1]==8:x=np.pad(x,((0,0),(0,3),(0,0),(0,0)))
    if x.shape[1]!=11:raise ValueError('Exactly eleven channels, shape-off zero padded')
    return x,w

@torch.no_grad()
def predict(model,x,w,batch=256,legacy_score=False):
    model.eval();parts={k:[] for k in ['share','score','contrast','legacy_score','cell_mask']}
    for i in range(0,len(x),batch):
        xx=x[i:i+batch].contiguous(memory_format=torch.channels_last);ww=w[i:i+batch]
        p,q=model(xx,ww);old,_,c=fixed_presence_score(q,ww)
        for key,v in [('share',p),('score',old if legacy_score else p),('contrast',c),('legacy_score',old),('cell_mask',q)]:parts[key].append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in parts.items()}

def metrics(y,pred,site=None):
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y,float);p=np.asarray(pred['share'],float);e=p-y;z=y>0
    out={'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(abs(e).mean()),'bias':float(e.mean()),
         'prediction_mean':float(p.mean()),'prediction_rms':float(np.sqrt(np.mean(p*p))),
         'target_mean':float(y.mean()),'target_rms':float(np.sqrt(np.mean(y*y))),
         'zero_n':int((~z).sum()),'zero_prediction_mean':float(p[~z].mean()) if (~z).any() else None}
    for name in ['share','score','legacy_score']:
        if name in pred:out['auc_'+name]=float(roc_auc_score(z,pred[name])) if len(np.unique(z))==2 else None
    if 'contrast' in pred:out['contrast_ge099_fraction']=float(np.mean(pred['contrast']>=.99))
    if site is not None:
        site=np.asarray(site).astype(str);out['site_slices']={s:{'n':int((site==s).sum()),'rmse':float(np.sqrt(np.mean(e[site==s]**2))),'bias':float(e[site==s].mean())} for s in sorted(set(site))}
        out['site_macro_rmse']=float(np.mean([v['rmse'] for v in out['site_slices'].values()]))
    out['target_slices']={}
    for name,mask in [('zero',~z),('positive_to_.2',z&(y<=.2)),('above_.2',y>.2)]:
        out['target_slices'][name]={'n':int(mask.sum()),'rmse':float(np.sqrt(np.mean(e[mask]**2))) if mask.any() else None,'bias':float(e[mask].mean()) if mask.any() else None,'SSE':float(np.sum(e[mask]**2))}
    return out

def fit(train,val,basis,arm,outdir,*,threads=4,batch=128):
    """fit uses TRAIN+validation_fit only. No gate/eval reads here."""
    folder=Path(outdir);folder.mkdir(parents=True,exist_ok=False);beg=time.perf_counter()
    torch.set_num_threads(threads);torch.manual_seed(2026102712)
    x,w=channels(train['db'],basis,arm['shape']);v,vw=channels(val['db'],basis,arm['shape'])
    x=torch.from_numpy(x);w=torch.from_numpy(w);v=torch.from_numpy(v);vw=torch.from_numpy(vw)
    y=torch.from_numpy(np.asarray(train['share'],np.float32));q=torch.from_numpy(np.asarray(train['cell_share'],np.float32))
    if not all(torch.isfinite(t).all() for t in [x,w,y,q,v,vw]):raise ValueError('Nonfinite train/validation tensors')
    if ((q<0)|(q>1)).any() or ((y<0)|(y>1)).any() or (q[w==0]!=0).any():raise ValueError('Invalid truth q/share or missing truth')
    vy=np.asarray(val['share'],float)
    if not np.isfinite(vy).all() or np.any((vy<0)|(vy>1)):raise ValueError('Invalid validation label')
    if not torch.allclose(w.sum((1,2)),torch.ones(len(w)),atol=3e-6,rtol=0):raise ValueError('Invalid weights')
    if float(abs((w*q).sum((1,2))-y).max())>.0058:raise ValueError('Rounded pooling bound exceeded')
    model=R12MaskCNN().to(memory_format=torch.channels_last)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=1e-5)
    best=float('inf');best_epoch=0
    write(folder/'STARTED.json',{'arm':arm,'epochs':100,'optimizer':'Adam','early_stopping':False,'parameters':sum(t.numel() for t in model.parameters()),'score':'share_only_except_legacy_control','fixture':False})
    for epoch in range(1,101):
        model.train();order=np.random.default_rng(np.random.SeedSequence([2026102712,epoch])).permutation(len(x));ss=total=0.;count=0
        for start in range(0,len(order),batch):
            ix=order[start:start+batch];xx=x[ix].contiguous(memory_format=torch.channels_last);ww=w[ix]
            optimizer.zero_grad(set_to_none=True);p,mask=model(xx,ww)
            rows,mse,_=objective(p,mask,ww,y[ix],q[ix],legacy_control=arm.get('legacy_control',False));loss=rows.mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            ss+=float(mse.detach().sum());total+=float(rows.detach().sum());count+=len(ix)
        pred=predict(model,v,vw,batch,legacy_score=arm.get('legacy_control',False));m=metrics(val['share'],pred)
        if m['rmse']<best:
            best=m['rmse'];best_epoch=epoch
            temp=folder/'best.tmp.pt';torch.save({'state_dict':model.state_dict(),'arm':arm,'epoch':epoch,'parameters':sum(t.numel() for t in model.parameters())},temp);temp.replace(folder/'best.pt')
        row={'epoch':epoch,'train_online_rmse':float(np.sqrt(ss/count)),'train_loss':total/count,'validation_fit_rmse':m['rmse'],
             'validation_fit_auc_share':m['auc_share'],'validation_contrast_ge099':m['contrast_ge099_fraction'],
             'best_rmse':best,'best_epoch':best_epoch,'learning_rate':optimizer.param_groups[0]['lr'],'elapsed_seconds':time.perf_counter()-beg}
        with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        write(folder/'progress.json',row)
        if epoch%10==0 or epoch==1:print(json.dumps({'arm':arm['name'],**row}),flush=True)
        scheduler.step()
    ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict'])
    pred=predict(model,v,vw,batch,legacy_score=arm.get('legacy_control',False));np.savez_compressed(folder/'validation_fit_predictions.npz',**pred)
    summary={'format':'r12_small_physics_cnn_v1','status':'completed','arm':arm,'epochs_run':100,'best_epoch':best_epoch,
             'early_stopping':False,'optimizer':'Adam','dropout':.1,'learning_rate_schedule':'cosine .001->.00001, fixed100',
             'parameters':sum(t.numel() for t in model.parameters()),'validation_fit_metrics':metrics(val['share'],pred,val.get('site')),
             'checkpoint_sha256':sha(folder/'best.pt'),'elapsed_seconds':time.perf_counter()-beg,'device':'Mac CPU FP32',
             'Test_read':False,'gate_or_eval_read':False,'score':'legacy_mix' if arm.get('legacy_control') else 'share',
             'code_sha256':{str(Path(__file__).name):sha(__file__)}}
    write(folder/'summary.json',summary);return summary
