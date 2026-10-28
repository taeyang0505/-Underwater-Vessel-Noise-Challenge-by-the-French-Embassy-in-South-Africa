"""Fit ensemble on validation, affine calibration on separate calibration only."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.optimize import minimize

def metric(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def fit_weights(P,y):
    k=P.shape[1];u=np.full(k,1/k)
    r=minimize(lambda w:np.mean((P@w-y)**2)+1e-4*np.sum((w-u)**2),u,
               bounds=[(0,1)]*k,constraints={'type':'eq','fun':lambda w:w.sum()-1},
               method='SLSQP',options={'ftol':1e-12,'maxiter':300})
    if not r.success:raise RuntimeError(r.message)
    return r.x
def affine(x,y):
    r=minimize(lambda ab:np.mean((np.clip(ab[0]*x+ab[1],0,1)-y)**2),[1.,0.],
               bounds=[(0,3),(-.5,.5)],method='L-BFGS-B')
    return r.x
def main(a):
    data={s:dict(np.load(a.run/f'{s}_predictions.npz')) for s in ['validation','calibration','domain_holdout']}
    kinds=sorted(k.removesuffix('_share') for k in data['validation'] if k.endswith('_share'))
    P=lambda s:np.stack([data[s][k+'_share'] for k in kinds],1)
    w=fit_weights(P('validation'),data['validation']['share_true'])
    c=P('calibration')@w;y=data['calibration']['share_true']
    # Deterministic cross-fit calibration prevents selecting by its own fitted residual.
    idx=np.arange(len(y))%2;cv=np.empty_like(c)
    for j in [0,1]:
        ab=affine(c[idx!=j],y[idx!=j]);cv[idx==j]=np.clip(ab[0]*c[idx==j]+ab[1],0,1)
    use=metric(y,cv)+1e-4<metric(y,c)
    ab=affine(c,y) if use else np.array([1.,0.])
    results={}
    for s,d in data.items():
        pred=P(s)@w;cal=np.clip(ab[0]*pred+ab[1],0,1)
        results[s]={'ensemble_rmse':metric(d['share_true'],pred),'calibrated_rmse':metric(d['share_true'],cal),
                    'zero_addition_mean':float(cal[d['share_true']==0].mean())}
    info={'models':kinds,'weights':w.tolist(),'affine':ab.tolist(),'affine_selected_by_calibration_crossfit':bool(use),
          'calibration_cv_rmse':metric(y,cv),'results':results,'test_accessed':False,
          'policy':'weights from validation; affine from calibration; domain holdout reported once, not used to choose weights'}
    if (a.run/'members.json').exists():info['member_run_names']=json.loads((a.run/'members.json').read_text())['member_run_names']
    if (a.run/'foundation_specs.json').exists():
        info['foundation_specs']=json.loads((a.run/'foundation_specs.json').read_text())
        info['base_run_name']=json.loads((a.run/'candidate_merge_manifest.json').read_text())['base_run_name']
        info['score_policy']='Foundation diagnostic score is bounded share proxy, not calibrated presence probability.'
    (a.run/'postprocess.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);main(p.parse_args())
