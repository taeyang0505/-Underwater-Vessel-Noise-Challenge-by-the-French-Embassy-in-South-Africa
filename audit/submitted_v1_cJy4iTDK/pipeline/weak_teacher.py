"""Externally supervised AIS-scenario proxy. This does not create ground truth."""
from pathlib import Path
import json,argparse,hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,brier_score_loss
from pipeline.mixtures import row_features
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=ROOT/'data/processed';out=ROOT/'models/weak_teacher';out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_parquet(p/'external_clip_manifest.parquet')
    bank=np.load(p/'external_segments_power.npy',mmap_mode='r')
    x=[];y=[];fold=[];group=[]
    # Same ONC instrument for positive/control training reduces source-recorder confounding.
    # Ambient controls and AIS labels remain weak, not audible-vessel truth.
    for r in meta[meta.source.eq('VTUAD_v1.3.2')].itertuples():
        if r.split not in ['train','validation','domain_holdout']:continue
        positive=r.archive_role=='vessel_passage' and str(r.weak_class).lower() not in ['background','unknown','nan']
        negative=r.archive_role=='ambient_control'
        if not (positive or negative):continue
        s=np.array(bank[int(r.segment_start):int(r.segment_start+r.segment_count)],dtype=np.float64)
        # Complete one-minute means only. No synthetic extension in this teacher.
        s=s[:len(s)//30*30].reshape(-1,30,25).mean(1)
        if not len(s):continue
        x.append(row_features(10*np.log10(np.maximum(s,1e-30))))
        y.extend([int(positive)]*len(s));fold.extend([r.split]*len(s));group.extend([r.group_key]*len(s))
    X=np.concatenate(x);y=np.array(y);fold=np.array(fold);group=np.array(group)
    tr=fold=='train';counts=pd.Series(group[tr]).value_counts()
    weight=np.array([1/counts[g] for g in group[tr]])
    for cls in [0,1]:weight[y[tr]==cls]/=weight[y[tr]==cls].sum()
    weight*=len(weight)/2
    models=[HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,l2_regularization=2,random_state=42),
            ExtraTreesClassifier(n_estimators=180,min_samples_leaf=8,max_features=.8,n_jobs=4,random_state=43)]
    for m in models:m.fit(X[tr],y[tr],sample_weight=weight)
    report={'kind':'AIS-scenario weak proxy, NOT verified natural-vessel labels',
            'fit_rows':int(tr.sum()),'fit_source':'VTUAD only; positive AIS classes vs dedicated ambient controls',
            'external_manifest':json.loads((p/'external_preparation_manifest.json').read_text()),'metrics':{},'test_accessed':False}
    for split in ['validation','domain_holdout']:
        q=fold==split
        if q.any() and len(np.unique(y[q]))==2:
            prob=np.mean([m.predict_proba(X[q])[:,1] for m in models],0)
            report['metrics'][split]={'n_minutes':int(q.sum()),'groups':len(np.unique(group[q])),
                'auc':float(roc_auc_score(y[q],prob)),
                'balanced_accuracy':float(balanced_accuracy_score(y[q],prob>.5)),
                'brier':float(brier_score_loss(y[q],prob))}
    joblib.dump(models,out/'models.joblib')
    rows=np.load(p/'sa_rows_db.npy',mmap_mode='r');manifest=pd.read_parquet(p/'sa_row_manifest.parquet')
    raw=np.empty(len(rows),dtype=np.float32);unc=np.empty_like(raw)
    for start in range(0,len(rows),32768):
        stop=min(start+32768,len(rows));features=row_features(rows[start:stop])
        pred=np.stack([m.predict_proba(features)[:,1] for m in models])
        raw[start:stop]=pred.mean(0);unc[start:stop]=pred.std(0)
        if start%(32768*8)==0:print('Scored original rows',stop,'/',len(rows),flush=True)
    quality=manifest['sentinel_flatline'].to_numpy() if 'sentinel_flatline' in manifest else np.ptp(rows,axis=1)==0
    invalid=quality|~np.isfinite(rows).all(1)
    proxy=raw.copy();proxy[invalid]=np.nan
    label=np.full(len(rows),'uncertain',dtype=object)
    label[(raw>=.95)&(unc<=.05)&~invalid]='weak_vessel_scenario'
    label[(raw<=.05)&(unc<=.05)&~invalid]='weak_ambient_scenario'
    label[invalid]='invalid_measurement'
    passed=bool(report['metrics']) and all(v['auc']>=.65 for v in report['metrics'].values())
    if not passed:label[~invalid]='uncertain_teacher_failed_validation'
    result=pd.DataFrame({'row_id':manifest.row_id,'teacher_score_raw_audit':raw,
        'weak_vessel_scenario_proxy':proxy,'teacher_disagreement':unc,'weak_label_status':label,
        'is_verified_ground_truth':False,'teacher_version':'external_vtuad_v1'})
    assert len(result)==1715779 and result.row_id.is_unique
    result.to_parquet(p/'sa_all_rows_weak_labels.parquet',index=False,compression='zstd')
    report['all_rows_scored']=len(result);report['invalid_measurements_kept']=int(invalid.sum())
    report['weak_label_counts']=result.weak_label_status.value_counts().to_dict()
    report['approved_for_pseudo_label_training']=passed
    report['warning']='These are uncalibrated cross-domain model proxies. Do not use as evaluation truth or as share_added_vessel labels.'
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
