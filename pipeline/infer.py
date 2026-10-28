"""Frozen, per-window competition inference. No fitting or test-wide calibration."""
from pathlib import Path
import argparse,json,sys,subprocess,tempfile,hashlib
import numpy as np
import pandas as pd
import torch
from pipeline.models import make_model
from pipeline.mixtures import fixed_features,cnn_channels,hybrid_cnn_channels
from pipeline.checkpoint_prior import attach_checkpoint_prior
from pipeline.prepare_sa import COMMON

@torch.inference_mode()
def predict(db,run,policy,device):
    if 'member_run_names' in policy:
        sub={k:v for k,v in policy.items() if k!='member_run_names'};sub['affine']=[1.,0.]
        values=[predict(db,run.parent/name,sub,device) for name in policy['member_run_names']]
        a,b=policy['affine']
        return np.clip(a*np.mean([x[0] for x in values],0)+b,0,1),np.mean([x[1] for x in values],0)
    predictions={};kinds=policy['models']
    base_run=run.parent/policy['base_run_name'] if 'base_run_name' in policy else run
    tree=[k for k in kinds if k.startswith('lgbm')]
    if tree:
        with tempfile.TemporaryDirectory(prefix='underwater-infer-') as tmp:
            np.save(Path(tmp)/'input.npy',db)
            subprocess.run([sys.executable,'-m','pipeline.predict_lgbm',str(base_run),str(Path(tmp)/'input.npy'),
                str(Path(tmp)/'pred.npz'),'--kinds',*tree],check=True)
            predictions.update(dict(np.load(Path(tmp)/'pred.npz')))
    for k in kinds:
        if k.startswith('lgbm') or k in ['tabicl','kumo_large']:continue
        saved=torch.load(base_run/(k+'.pt'),map_location='cpu',weights_only=True)
        model=make_model(k,width=saved['width'],k=saved['k'],**saved.get('architecture',{})).to(device)
        attach_checkpoint_prior(model,saved,base_run)
        model.load_state_dict(saved['state_dict']);model.eval();shares=[];scores=[]
        version=int(saved.get('feature_version',getattr(model,'feature_version',1)))
        if version!=getattr(model,'feature_version',1):raise ValueError('Checkpoint feature version differs from architecture')
        for start in range(0,len(db),256):
            part=db[start:start+256]
            if k=='tabm':
                z=model(torch.from_numpy(fixed_features(part,version=version)).to(device)).sigmoid().mean(1)
                y,s=z[:,0],z[:,1]
            else:
                c,w=hybrid_cnn_channels(part,model.background_prior) if getattr(model,'background_prior',None) is not None else cnn_channels(part,version=version)
                y,logit,_=model(torch.from_numpy(c).to(device),torch.from_numpy(w).to(device));s=logit.sigmoid()
            shares.append(y.cpu().numpy());scores.append(s.cpu().numpy())
        predictions[k+'_share']=np.concatenate(shares);predictions[k+'_score']=np.concatenate(scores)
    foundation=[k for k in kinds if k in ['tabicl','kumo_large']]
    if foundation:
        with tempfile.TemporaryDirectory(prefix='underwater-foundation-infer-') as tmp:
            np.save(Path(tmp)/'input.npy',db)
            subprocess.run([sys.executable,'-m','pipeline.predict_foundation',str(run),str(Path(tmp)/'input.npy'),
                str(Path(tmp)/'pred.npz'),'--kinds',*foundation,'--device',str(device)],check=True)
            predictions.update(dict(np.load(Path(tmp)/'pred.npz')))
    w=np.array(policy['weights']);raw=np.stack([predictions[k+'_share'] for k in kinds],1)@w
    score=np.stack([predictions[k+'_score'] for k in kinds],1)@w
    a,b=policy['affine'];return np.clip(a*raw+b,0,1),np.clip(score,0,1)

def main(a):
    torch.set_num_threads(a.threads)
    # Hash frozen policy BEFORE opening the inference file.
    policy_bytes=(a.run/'postprocess.json').read_bytes();policy=json.loads(policy_bytes)
    test=pd.read_csv(a.input);cols=[f'm{i:02d}_{c}' for i in range(1,16) for c in COMMON]
    assert test.ID.is_unique and all(c in test for c in cols)
    db=test[cols].to_numpy(np.float32).reshape(-1,15,25)
    assert np.isfinite(db).any((1,2)).all()
    y,s=predict(db,a.run,policy,torch.device(a.device))
    result=pd.DataFrame({'ID':test.ID,'share_added_vessel':y,'score_added_vessel':s})
    assert result.iloc[:,1:].apply(lambda c:np.isfinite(c)&c.between(0,1)).all().all()
    a.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output,index=False)
    (a.output.with_suffix('.manifest.json')).write_text(json.dumps({'rows':len(result),'run':str(a.run),
        'frozen_policy_sha256':hashlib.sha256(policy_bytes).hexdigest(),'fit_on_test':False,
        'submitted_to_zindi':False,'status':'local candidate; GPU multi-seed validation pending'},indent=2))
    print('Saved',len(result),'frozen predictions to',a.output)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4);main(p.parse_args())
