"""Explicit candidate training. No AutoML package and no access to Test.csv."""
from pathlib import Path
import argparse,json,time,hashlib,os,copy,platform,subprocess,sys
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import mean_squared_error,roc_auc_score
from pipeline.mixtures import MixtureSampler,fixed_features,cnn_channels
from pipeline.models import make_model
ROOT=Path(__file__).resolve().parents[1]

def rmse(y,p):return float(np.sqrt(mean_squared_error(y,p)))

def materialize(split,n,seed,folder):
    sampler=MixtureSampler(split,seed=seed)
    arrays=[[],[],[],[]];recipes=[]
    for start in range(0,n,256):
        result=sampler.draw(min(256,n-start))
        for i in range(4):arrays[i].append(result[i])
        recipe=result[4];recipe['background_index']=recipe['background_index'].tolist();recipes.append(recipe)
    data={k:np.concatenate(v) for k,v in zip(['db','share','presence','cell_share'],arrays)}
    np.savez_compressed(folder/f'{split}.npz',**data)
    (folder/f'{split}_recipes.json').write_text(json.dumps({'seed':seed,'batch_size':256,'batches':recipes}))
    print('Materialized',split,n,flush=True)
    return data

@torch.inference_mode()
def predict_deep(model,kind,data,device,batch=256):
    model.eval();shares=[];scores=[]
    inputs=fixed_features(data['db']) if kind=='tabm' else cnn_channels(data['db'])
    for start in range(0,len(data['share']),batch):
        sl=slice(start,start+batch)
        if kind=='tabm':
            z=model(torch.from_numpy(inputs[sl]).to(device)).sigmoid().mean(1)
            share,score=z[:,0],z[:,1]
        else:
            c,w=inputs
            share,logit,_=model(torch.from_numpy(c[sl]).to(device),torch.from_numpy(w[sl]).to(device))
            score=logit.sigmoid()
        shares.append(share.cpu().numpy());scores.append(score.cpu().numpy())
    return np.concatenate(shares),np.concatenate(scores)

def train_deep(kind,train,val,device,args,folder):
    torch.manual_seed(args.seed);torch.set_num_threads(args.threads)
    model=make_model(kind,width=args.width,k=args.k)
    model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
    tx=fixed_features(train['db']) if kind=='tabm' else cnn_channels(train['db'])
    y=torch.from_numpy(train['share']).to(device);z=torch.from_numpy(train['presence']).to(device)
    cell=torch.from_numpy(train['cell_share']).to(device) if kind!='tabm' else None
    best=float('inf');state=None;history=[];patience=0
    rng=np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train();losses=[];t0=time.time()
        for ids in np.array_split(rng.permutation(len(y)),max(1,int(np.ceil(len(y)/args.batch)))):
            opt.zero_grad(set_to_none=True)
            if kind=='tabm':
                logits=model(torch.from_numpy(tx[ids]).to(device))
                # Official TabM recipe: mean of member losses, not loss of the ensemble mean.
                loss=F.mse_loss(logits[:,:,0].sigmoid(),y[ids,None].expand(-1,logits.shape[1]))
                loss+=.03*F.binary_cross_entropy_with_logits(logits[:,:,1],z[ids,None].expand(-1,logits.shape[1]))
            else:
                c,w=tx;tw=torch.from_numpy(w[ids]).to(device)
                p,logit,mask=model(torch.from_numpy(c[ids]).to(device),tw)
                loss=F.mse_loss(p,y[ids])+.03*F.binary_cross_entropy_with_logits(logit,z[ids])
                loss+=.03*((mask-cell[ids]).square()*tw).sum((1,2)).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step();losses.append(float(loss.detach().cpu()))
        pred,score=predict_deep(model,kind,val,device)
        value=rmse(val['share'],pred)
        h={'epoch':epoch+1,'loss':float(np.mean(losses)),'validation_rmse':value,'seconds':time.time()-t0}
        history.append(h);print(kind,h,flush=True)
        if value<best:
            best=value;state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()});patience=0
        else:patience+=1
        if patience>=5:break
    model.load_state_dict(state)
    torch.save({'state_dict':state,'kind':kind,'width':args.width,'k':args.k,'seed':args.seed,
                'n_features':1254,'parameters':sum(p.numel() for p in model.parameters())},folder/f'{kind}.pt')
    (folder/f'{kind}_history.json').write_text(json.dumps(history,indent=2))
    return model

def main(args):
    torch.set_num_threads(args.threads)
    code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'pipeline').glob('*.py')}
    manifest=json.loads((ROOT/'data/processed/external_preparation_manifest.json').read_text())
    run=f'{args.tag}_{manifest["source_hash"][:8]}_seed{args.seed}'
    folder=ROOT/'models'/run;folder.mkdir(parents=True,exist_ok=True)
    datafolder=ROOT/'data/synthetic'/run;datafolder.mkdir(parents=True,exist_ok=True)
    data={}
    for i,(split,n) in enumerate([('train',args.n_train),('validation',args.n_val),('calibration',args.n_val),('domain_holdout',args.n_val)]):
        p=datafolder/f'{split}.npz'
        sample_seed=args.seed if split=='train' else args.eval_seed+1009*i
        data[split]=dict(np.load(p)) if p.exists() else materialize(split,n,sample_seed,datafolder)
        if len(data[split]['share'])!=n:raise ValueError('Cached size differs; use a new --tag for this experiment')
    t0=time.time();results={};val=data['validation']
    predictions={s:{} for s in data if s!='train'}
    baseline=float(data['train']['share'].mean())
    for split in predictions:
        results[f'constant_{split}']={'rmse':rmse(data[split]['share'],np.full(len(data[split]['share']),baseline))}
    for lgbm_kind in ['lgbm','lgbm_physics','lgbm_retrieval']:
        if lgbm_kind not in args.models:continue
        # Fresh interpreter avoids incompatible bundled Torch / LightGBM OpenMP runtimes on macOS.
        subprocess.run([sys.executable,'-m','pipeline.train_lgbm',str(datafolder),str(folder),
                        '--trees',str(args.trees),'--threads',str(args.threads),'--seed',str(args.seed),
                        '--variant',lgbm_kind.removeprefix('lgbm_') if lgbm_kind!='lgbm' else 'base'],check=True)
        for split in predictions:
            pred=np.load(folder/f'{lgbm_kind}_{split}.npz')
            predictions[split][lgbm_kind]=(pred['share'],pred['score'])
    device=torch.device(args.device)
    for kind in ['tabm','cnn','fdy','itransformer']:
        if kind not in args.models:continue
        model=train_deep(kind,data['train'],val,device,args,folder)
        for split in predictions:predictions[split][kind]=predict_deep(model,kind,data[split],device)
        del model
    for split,members in predictions.items():
        out={'share_true':data[split]['share'],'presence_true':data[split]['presence']}
        for kind,(p,s) in members.items():
            results[f'{kind}_{split}']={'rmse':rmse(data[split]['share'],p),'score_1_minus_rmse':1-rmse(data[split]['share'],p),
                                      'addition_auc':float(roc_auc_score(data[split]['presence'],s))}
            out[kind+'_share']=p;out[kind+'_score']=s
        np.savez_compressed(folder/f'{split}_predictions.npz',**out)
    info={'run':run,'arguments':vars(args),'external_manifest':manifest,'results':results,
          'code_sha256_at_start':code_hashes,
          'background_preparation':json.loads((ROOT/'data/processed/sa_preparation_manifest.json').read_text()),
          'elapsed_seconds':time.time()-t0,'platform':platform.platform(),'test_accessed':False,
          'data_folder':str(datafolder),'status':'source-disjoint synthetic experiment; not an official leaderboard result'}
    (folder/'run_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--tag',default='pilot');p.add_argument('--n-train',type=int,default=12000)
    p.add_argument('--n-val',type=int,default=2000);p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--models',nargs='+',default=['lgbm','tabm','cnn']);p.add_argument('--device',default='cpu')
    p.add_argument('--batch',type=int,default=128);p.add_argument('--width',type=int,default=256)
    p.add_argument('--k',type=int,default=16);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-seed',type=int,default=20261024)
    p.add_argument('--trees',type=int,default=700);p.add_argument('--threads',type=int,default=6)
    main(p.parse_args())
