"""Fresh R14 baseline-only anchor ablation, fixed100 Adam; no Test I/O.

Two CNNs: unweighted and 1+2*y row-weighted physical MSE objective.
Two Tree regressors use same weights, one independently unweighted classifier.
Only TRAIN/validation_fit are materialized; all models freeze before panels.
"""
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='models/r14_anchor_cpu_r1');p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    from ops.train_r12_lightgbm import environment,guard,array_sha
    environment(a.threads);sys.addaudithook(guard)
    import numpy as np,torch,lightgbm as lgb
    from torch import nn
    from pipeline.dataset_r13 import load_r13_arrays,validate_r13_bundle
    from pipeline.dataset_r12 import load_r12_arrays,validate_r12_bundle
    from pipeline.features_r12 import load_spectral_basis
    from pipeline.features_r14_anchor import channels,tree_features
    from pipeline.training_r12 import sha,write,predict,metrics,objective
    from pipeline.relative_models_r6_v2 import TinyMaskCNNv2
    class Model(TinyMaskCNNv2):
        def __init__(self):super().__init__(input_channels=14,width=24,blocks=2,dropout=.1,global_context=True)
        def add_global_context(self,h):
            v=torch.cat([h[:,:,:,s:e].mean(3) for s,e in ((10,20),(20,23),(15,23))],1)
            return h+(nn.functional.gelu(self.context_projection(v))*torch.sigmoid(self.context_gate(v))).unsqueeze(3)
        def forward(self,x,w):
            q=self.mask_head(self.add_global_context(self.body(x)))[:,0].sigmoid();q=torch.where(w>0,q,torch.zeros_like(q))
            return (w*q).sum((1,2)),q
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(a.threads)
    data=ROOT/'data/synthetic/r13_cpu_correction_v1';m=validate_r13_bundle(data)
    orig=ROOT/'data/synthetic/r12_cpu_ablation_v3';pm=validate_r12_bundle(orig)
    train=load_r13_arrays(data,'baseline');val=load_r12_arrays(orig,'L0_H0','validation_fit')
    bp=ROOT/'models/r12_shared_basis_v1/basis.npz';basis=load_spectral_basis(bp,expected_training_binding=pm['basis_training_binding'])
    subset=np.load(orig/'cnn_subset_indices.npy',allow_pickle=False)
    paths=[Path(__file__),ROOT/'pipeline/features_r14_anchor.py',ROOT/'pipeline/training_r12.py',ROOT/'pipeline/features_r12.py',data/'manifest.json',orig/'manifest.json',bp,orig/'cnn_subset_indices.npy']
    pins={str(v.relative_to(ROOT)):sha(v) for v in paths}
    plan={'format':'r14_anchor_cpu_plan_v1','data_hash':m['data_hash'],'data_arm':'baseline','TRAIN_npz_sha256':m['arms']['baseline']['sha256'],
      'source_pins':pins,'CNN_train_rows':len(subset),'Tree_train_rows':len(train['share']),'CNN_epochs':100,'CNN_early_stopping':False,
      'optimizer':'Adam','lr':.001,'dropout':.1,'row_weight_candidates':[0,2],'loss':'(1+c*y)*(share_MSE+.25*weighted_cell_MSE)',
      'Tree_features':880,'Tree_max':1500,'Tree_ES':40,'classifier_training':'unweighted baseline only; independent of regressor targets/weights',
      'validation_role':'validation_fit1000 only','Test_or_panels_read':False,'development_exposed':True,
      'anchor_caveat':'Redundant contrasts from existing profile; LF is not certified pure background; additional TRAIN prior can introduce site bias.'}
    write(out/'PRE_FIT_FREEZE.json',plan)
    # Verify gain invariance and no target dependence before any fit.
    probe=train['db'][:16];aa,ww=channels(probe,basis);bb,vv=channels(probe+20,basis)
    diff=float(np.max(abs(aa-bb)));wd=float(np.max(abs(ww-vv)))
    if diff>3e-6 or wd>3e-6:raise ValueError('Postquantization gain invariance failed')
    write(out/'FEATURE_PREFLIGHT.json',{'gain20db_max_channel_error':diff,'weight_error':wd,'shape':list(aa.shape),'finite':bool(np.isfinite(aa).all())})
    started=time.perf_counter();complete={}
    try:
        xx,ww=channels(train['db'][subset],basis);vx,vw=channels(val['db'],basis)
        x=torch.from_numpy(xx);w=torch.from_numpy(ww);v=torch.from_numpy(vx);vw=torch.from_numpy(vw)
        y=torch.from_numpy(train['share'][subset].astype(np.float32));q=torch.from_numpy(train['cell_share'][subset].astype(np.float32))
        for c in (0,2):
            name='anchor_c'+str(c);folder=out/name;folder.mkdir();torch.manual_seed(2026102712)
            model=Model().to(memory_format=torch.channels_last);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
            sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=1e-5)
            best=float('inf');bestep=0;t0=time.perf_counter()
            for epoch in range(1,101):
                model.train();idx=np.random.default_rng(np.random.SeedSequence([2026102712,epoch])).permutation(len(x));sse=total=0.
                for s in range(0,len(idx),128):
                    ix=idx[s:s+128];opt.zero_grad(set_to_none=True);pred,mask=model(x[ix].contiguous(memory_format=torch.channels_last),w[ix])
                    row,share,_=objective(pred,mask,w[ix],y[ix],q[ix]);weight=1+c*y[ix]
                    loss=(weight*row).sum()/weight.sum()
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();sse+=float(share.detach().sum());total+=len(ix)
                pv=predict(model,v,vw,128);metric=metrics(val['share'],pv)
                if metric['rmse']<best:
                    best=metric['rmse'];bestep=epoch;torch.save({'state_dict':model.state_dict(),'epoch':epoch,'input_channels':14,'row_c':c},folder/'best.pt')
                row={'epoch':epoch,'train_online_rmse':float(np.sqrt(sse/total)),'validation_fit_rmse':metric['rmse'],'best_rmse':best,'best_epoch':bestep,'elapsed_seconds':time.perf_counter()-t0}
                with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                write(out/'progress.json',{'stage':'cnn','arm':name,**row})
                if epoch%10==0 or epoch==1:print(json.dumps({'arm':name,**row}),flush=True)
                sched.step()
            model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True)['state_dict']);pv=predict(model,v,vw,128)
            np.savez_compressed(folder/'validation_fit_predictions.npz',**pv,target_share=val['share'])
            summary={'status':'completed','epochs_run':100,'optimizer':'Adam','early_stopping':False,'dropout':.1,'best_epoch':bestep,'best_fit_rmse':best,
              'parameters':sum(p.numel() for p in model.parameters()),'row_weight_c':c,'source_binding':plan,'elapsed_seconds':time.perf_counter()-t0}
            write(folder/'summary.json',summary);complete[name]={'summary_sha256':sha(folder/'summary.json'),'checkpoint_sha256':sha(folder/'best.pt')}
        tx=tree_features(train['db'],basis);tv=tree_features(val['db'],basis)
        args=dict(n_estimators=1500,learning_rate=.035,num_leaves=31,min_child_samples=50,max_bin=127,n_jobs=a.threads,random_state=2026102712,verbosity=-1,
          colsample_bytree=.85,subsample=.85,subsample_freq=1,deterministic=True,force_col_wise=True,data_random_seed=2026102712,feature_fraction_seed=2026102712,bagging_seed=2026102712)
        folder=out/'tree';folder.mkdir();raw={}
        for c in (0,2):
            reg=lgb.LGBMRegressor(objective='regression_l2',metric='None',**args)
            reg.fit(tx,train['share'],sample_weight=1+c*train['share'],eval_set=[(tv,val['share'])],
              eval_metric=lambda yy,pp:('clipped_rmse',float(np.sqrt(np.mean((yy-np.clip(pp,0,1))**2))),False),
              callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(0)])
            reg.booster_.save_model(str(folder/('share_c'+str(c)+'.txt')));raw['share_c'+str(c)]=np.clip(reg.predict(tv),0,1).astype(np.float32)
            complete['tree_c'+str(c)]={'best_iteration':reg.best_iteration_,'model_sha256':sha(folder/('share_c'+str(c)+'.txt'))}
        clf=lgb.LGBMClassifier(objective='binary',metric='binary_logloss',**args)
        clf.fit(tx,train['presence'],eval_set=[(tv,val['presence'])],callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(0)])
        clf.booster_.save_model(str(folder/'presence.txt'));raw['classifier']=clf.predict_proba(tv)[:,1].astype(np.float32)
        np.savez_compressed(folder/'validation_fit_predictions.npz',**raw,target_share=val['share'])
        complete['classifier']={'best_iteration':clf.best_iteration_,'model_sha256':sha(folder/'presence.txt')}
        for path,h in pins.items():
            if sha(ROOT/path)!=h:raise ValueError('Frozen source changed:'+path)
        write(out/'ALL_MODELS_FROZEN.json',{'status':'completed','models':complete,'plan':plan,'Test_or_panels_read':False,'elapsed_seconds':time.perf_counter()-started})
        print(json.dumps({'status':'completed','models':complete,'elapsed_seconds':time.perf_counter()-started}),flush=True)
    except BaseException as e:
        write(out/'FAILED.json',{'error':repr(e),'partial_reuse_allowed':False});raise

if __name__=='__main__':main()
