"""Explicit CPU LightGBM learner, isolated from Torch's OpenMP runtime."""
from pathlib import Path
import argparse,json
import numpy as np
from lightgbm import LGBMRegressor,LGBMClassifier,early_stopping,log_evaluation
from pipeline.mixtures import fixed_features,physics_features
from pipeline.prototypes import build as build_prototypes,features as prototype_features

def main(a):
    data={s:dict(np.load(a.data/f'{s}.npz')) for s in ['train','validation','calibration','domain_holdout']}
    kind='lgbm' if a.variant=='base' else 'lgbm_'+a.variant
    prototype_path=a.output/'prototypes.npz'
    if a.variant=='retrieval':
        info=build_prototypes(prototype_path)
        (a.output/'prototype_manifest.json').write_text(json.dumps(info,indent=2))
    def transform(db):
        parts=[fixed_features(db)]
        if a.variant!='base':parts.append(physics_features(db))
        if a.variant=='retrieval':parts.append(prototype_features(db,prototype_path))
        return np.concatenate(parts,1)
    X=transform(data['train']['db']);V=transform(data['validation']['db'])
    reg=LGBMRegressor(objective='regression_l2',n_estimators=a.trees,learning_rate=.035,num_leaves=31,
                      min_child_samples=50,colsample_bytree=.85,subsample=.85,subsample_freq=1,
                      max_bin=127,n_jobs=a.threads,random_state=a.seed,verbosity=-1)
    reg.fit(X,data['train']['share'],eval_X=V,eval_y=data['validation']['share'],
            callbacks=[early_stopping(40,verbose=False),log_evaluation(100)])
    clf=LGBMClassifier(n_estimators=min(300,a.trees),num_leaves=15,max_bin=127,min_child_samples=50,
                       n_jobs=a.threads,random_state=a.seed,verbosity=-1)
    clf.fit(X,data['train']['presence'])
    reg.booster_.save_model(str(a.output/(kind+'_share.txt')));clf.booster_.save_model(str(a.output/(kind+'_presence.txt')))
    for s,d in data.items():
        if s=='train':continue
        x=transform(d['db'])
        np.savez_compressed(a.output/f'{kind}_{s}.npz',share=np.clip(reg.predict(x),0,1),score=clf.predict_proba(x)[:,1])
    print('LightGBM complete',reg.best_iteration_,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--trees',type=int,default=700);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--variant',choices=['base','physics','retrieval'],default='base');main(p.parse_args())
