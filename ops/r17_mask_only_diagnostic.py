"""Mask-only source diagnostic; no acoustic values, Test, or model training."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops.r17_audit_common import load,jsave,sha,panel_data
from sklearn.metrics import roc_auc_score
OUT=ROOT/'reports/r17_clean_pipeline_r1/mask_only'

def patterns(db):
    # Pack all375 cells, rather than assuming that a minute has all25 bands.
    p=np.packbits(np.isfinite(db).reshape(len(db),-1),axis=1)
    return [v.tobytes().hex() for v in p]

def mi(keys,z):
    t=pd.crosstab(keys,np.asarray(z,int));p=t.to_numpy(float)/len(z)
    independent=p.sum(1)[:,None]*p.sum(0)[None];ix=p>0
    return max(0.,float(np.sum(p[ix]*np.log2(p[ix]/independent[ix]))))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=load(ROOT/'data/synthetic/r16_multisupport_v1/fold_00/train_D0.npz')
    new=load(ROOT/'data/synthetic/r17_clean_v1/train_D0.npz')
    rows=[];preds=[]
    for label,tr in [('old_R16_train',old),('new_R17_paired_train',new)]:
        k=patterns(tr['db']);z=tr['presence'];prior=float(z.mean())
        grouped=pd.DataFrame({'O':k,'z':z}).groupby('O').z.agg(['mean','count'])
        grouped.to_csv(OUT/(label+'_pattern_target_rates.csv'))
        rows.append({'dataset':label,'n':len(z),'unique_full_O_patterns':len(grouped),
                     'MI_full_O_presence_bits':mi(k,z),
                     'maximum_pattern_positive_rate_deviation':float(abs(grouped['mean']-prior).max())})
        mapping=grouped['mean'].to_dict()
        for name in ('local_gate','local_report','domain_holdout'):
            d=panel_data(name);kk=patterns(d['db']);p=np.array([mapping.get(x,prior) for x in kk])
            preds.append({'train_definition':label,'panel':name,'n':len(p),
                          'mask_only_auc':float(roc_auc_score(d['share']>0,p)),
                          'unseen_pattern_fraction':float(np.mean([x not in mapping for x in kk]))})
    pd.DataFrame(rows).to_csv(OUT/'full_O_MI.csv',index=False)
    pd.DataFrame(preds).to_csv(OUT/'mask_only_lookup_auc.csv',index=False)
    jsave(OUT/'COMPLETED.json',{'status':'completed','acoustic_values_used':False,'new_estimator_training':False,
         'definition':'Training empirical P(z=1|full 375-cell O); unseen mask uses training prior',
         'Test_used':False,'script_sha256':sha(__file__)})
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(preds).to_string(index=False))

if __name__=='__main__':main()
