"""Matched-context, matched-feature comparison for a foundation estimator."""
from pathlib import Path
import argparse,json
import numpy as np
from lightgbm import LGBMRegressor,early_stopping
from pipeline.mixtures import foundation_features
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=int,default=1024);p.add_argument('--n-eval',type=int,default=256)
    a=p.parse_args();data=ROOT/'data/synthetic/fullsources_cpu_a6d7f3a5_seed42'
    tr=dict(np.load(data/'train.npz'));rng=np.random.default_rng(42);ids=rng.choice(len(tr['share']),min(a.context,len(tr['share'])),replace=False)
    ev={s:dict(np.load(data/f'{s}.npz')) for s in ['validation','calibration']}
    X=foundation_features(tr['db'][ids]);V=foundation_features(ev['validation']['db'][:a.n_eval])
    m=LGBMRegressor(objective='regression_l2',n_estimators=800,learning_rate=.035,num_leaves=31,min_child_samples=20,
        max_bin=127,n_jobs=4,random_state=42,verbosity=-1)
    m.fit(X,tr['share'][ids],eval_X=V,eval_y=ev['validation']['share'][:a.n_eval],callbacks=[early_stopping(40,verbose=False)])
    result={'context_rows':len(ids),'features':X.shape[1],'n_eval':a.n_eval,'results':{},'test_accessed':False}
    for s,d in ev.items():
        pred=np.clip(m.predict(foundation_features(d['db'][:a.n_eval])),0,1);y=d['share'][:a.n_eval]
        result['results'][s]={'rmse':float(np.sqrt(np.mean((pred-y)**2)))}
    out=ROOT/f'models/lgbm_foundation_features_context{a.context}';out.mkdir(exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
