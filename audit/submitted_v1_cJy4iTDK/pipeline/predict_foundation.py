"""Frozen foundation inference: fit context preprocessing on synthetic TRAIN only."""
from pathlib import Path
import argparse,json,gc,time,hashlib
import numpy as np
import torch
from pipeline.mixtures import foundation_features
ROOT=Path(__file__).resolve().parents[1]

def predict_one(kind,db,spec,device):
    model_run=ROOT/'models'/spec['model_run_name']
    metadata=json.loads((model_run/'manifest.json').read_text())
    source=ROOT/'data/synthetic'/spec['synthetic_dataset_name']/'train.npz'
    tr=dict(np.load(source));n=int(spec['context_rows']);members=int(spec['members'])
    ids=np.array(metadata.get('context_indices',np.random.default_rng(42).choice(len(tr['share']),n,replace=False)))
    assert len(ids)==n and metadata['parameters']>0
    X=foundation_features(tr['db'][ids]);y=tr['share'][ids];query=foundation_features(db)
    probe_data=dict(np.load(source.parent/'validation.npz'))
    probe=foundation_features(probe_data['db'][:2]);del probe_data
    expected_key='kumo_share' if kind=='kumo_large' else kind+'_share'
    expected=np.load(model_run/'validation_predictions.npz')[expected_key][:2]
    tolerance=2e-3 if device.startswith('cuda') else 3e-5
    del tr
    t=time.perf_counter()
    if kind=='tabicl':
        from tabicl import TabICLRegressor
        m=TabICLRegressor(n_estimators=members,batch_size=1,kv_cache='repr',norm_methods=['none','power'],
            model_path=ROOT/'models/pretrained/tabicl-regressor-v2-20260212.ckpt',allow_auto_download=False,
            device=device,use_amp='auto',use_fa3=False,offload_mode='auto',n_jobs=4,random_state=42)
        m.fit(X,y)
        assert np.allclose(np.clip(m.predict(probe),0,1),expected,atol=tolerance),kind+' context reconstruction differs from frozen validation'
        pred=np.clip(m.predict(query),0,1)
    elif kind=='kumo_large':
        import sdm
        m=sdm.models.KumoTabular(task='regression',size='large',pretrained=False,device='cpu')
        state=torch.load(ROOT/'models/pretrained/kumo_large_regressor.pt',map_location='cpu',weights_only=True)
        m.models[sdm.Task.regression].load_state_dict(state,assign=True);del state
        m.to(device);m.eval()
        amp=torch.autocast(device,dtype=torch.float16,enabled=device.startswith('cuda'))
        with torch.no_grad(),amp:
            m.fit(torch.from_numpy(X).to(device),torch.from_numpy(y[:,None]).to(device),
                num_estimators=members,estimator_batch_size=1,generator=torch.Generator(device=device).manual_seed(42))
            check=np.clip(m.predict(torch.from_numpy(probe).to(device)).numerical.mean(-1).cpu().numpy(),0,1)
            assert np.allclose(check,expected,atol=tolerance),kind+' context reconstruction differs from frozen validation'
            pred=np.clip(m.predict(torch.from_numpy(query).to(device)).numerical.mean(-1).cpu().numpy(),0,1)
    else:raise ValueError(kind)
    print(kind,'frozen context prediction',len(pred),'rows',round(time.perf_counter()-t,2),'seconds',flush=True)
    del m;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return pred

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--kinds',nargs='+',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    specs=json.loads((a.run/'foundation_specs.json').read_text());db=np.load(a.input);out={}
    for k in a.kinds:
        pred=predict_one(k,db,specs[k],a.device);out[k+'_share']=pred
        # Diagnostic monotone evidence proxy; not a calibrated probability of natural vessels.
        out[k+'_score']=pred.copy()
    np.savez_compressed(a.output,**out)
