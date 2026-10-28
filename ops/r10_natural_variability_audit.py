"""Frozen Fold0 model response to natural selection backgrounds, known A=0.

No natural vessel-absence claim. Diagnostic panels are not TRAIN hard
negatives. No fitting, weight change, Test/report input or external transfer.
"""
from __future__ import annotations
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r10_diagnostics/natural_variability_r1'
SEED=2026102605
LF=np.arange(13)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,v):
    with Path(p).open('x') as f:f.write(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def array_sha(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(json.dumps({'shape':list(a.shape),'dtype':str(a.dtype)},sort_keys=True,separators=(',',':')).encode()+a.tobytes()).hexdigest()


def lf_std(db):
    """Population temporal SD of linear-summed LF power expressed in dB.

    Per-minute max subtraction is algebraic log-sum stabilization; add it
    back before temporal SD. There is no fitted power floor or rounding.
    """
    x=np.asarray(db,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(15,25) or not np.isfinite(x).all():
        raise ValueError('Complete finite15-minute source windows required')
    low=x[:,:,LF];top=low.max(axis=2)
    series=top+10*np.log10(np.power(10.,(low-top[:,:,None])/10).sum(axis=2))
    return series.std(axis=1,ddof=0)


def choose(frame,n,role,idcolumn='base_window_index'):
    keys=[hashlib.sha256(f'R10|{SEED}|{role}|{g}|{v}'.encode()).hexdigest()
          for g,v in zip(frame.background_group.astype(str),frame[idcolumn].astype(str))]
    return frame.iloc[np.argsort(np.asarray(keys),kind='stable')[:n]].copy().reset_index(drop=True)


def auc(z,p):
    z=np.asarray(z,bool);p=np.asarray(p,float)
    if not z.any() or z.all():return None
    idx=np.argsort(p,kind='stable');s=p[idx];rank=np.empty(len(p),float);i=0
    while i<len(p):
        j=i+1
        while j<len(p) and s[j]==s[i]:j+=1
        rank[idx[i:j]]=(i+1+j)/2;i=j
    return float((rank[z].sum()-z.sum()*(z.sum()+1)/2)/(z.sum()*(~z).sum()))


def zero_metrics(pred):
    p=np.asarray(pred['share'],float);s=np.asarray(pred['score'],float)
    if not np.isfinite(p).all() or not np.isfinite(s).all() or ((p<0)|(p>1)|(s<0)|(s>1)).any():
        raise ValueError('Finite bounded frozen predictions required')
    return {'n':len(p),'target_known_added_share':0.,'target_known_added_presence':0,
        'mean_share':float(p.mean()),'bias_pred_minus_known_A0':float(p.mean()),
        'share_RMS_RMSE_against_A0':float(np.sqrt(np.mean(p*p))),
        'share_q90':float(np.quantile(p,.9,method='linear')),
        'share_gt_.03_rate':float((p>.03).mean()),'share_gt_.10_rate':float((p>.10).mean()),
        'share_gt_.20_rate':float((p>.20).mean()),
        'diagnostic_score_ge_.5_known_addition_false_positive_rate':float((s>=.5).mean())}


def main():
    from ops.infer_r9_fold0_all5_pilot import (environment,verify_pins,inputs,predict_members,combine,MEMBERS)
    start=time.perf_counter()
    if OUT.exists():raise FileExistsError('Fresh diagnostic directory required')
    pilot=ROOT/'submissions/r9_fold0_all5_cpu_r1';plan0=json.loads((pilot/'prequery_plan.json').read_text())
    policy=json.loads((pilot/'frozen_policy.json').read_text())
    if canonical({k:v for k,v in policy.items() if k!='policy_sha256'})!=policy['policy_sha256']:
        raise ValueError('Frozen policy changed')
    verify_pins(plan0['source_pins'])
    centres=10.**(1.5+.1*np.arange(25));assert np.array_equal(np.flatnonzero(centres<=501.2),LF)
    # Persist the complete selection/statistic/sampling/metric protocol before
    # opening any source numeric window or computing quantiles/predictions.
    OUT.mkdir(parents=True)
    specification={'format':'r10_natural_variability_predeclared_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'seed':SEED,'fold':0,'source_role':'selection source pool only','required_observed_complete_minutes':15,
        'LF_indices':LF.tolist(),'actual_LF_centres_hz':centres[LF].tolist(),
        'LF_statistic':'std_ddof0 over15times of10log10(sum(linear band energies indices0..12)); source minuteQC rawdb, no extra rounding',
        'high':'strictly greater than q90 of all complete15-minute selection source windows; quantile linear',
        'control':'strictly below q50 of same subset; quantile linear',
        'selection':'stable SHA256 rank R10|seed|role|background_group|base_window_index, without replacement',
        'maximum_each_negative_panel':500,'shortfall':'all available high candidates; matched control count; no retries',
        'positive_panel':'existing savedselection presence>0 andNobs15; max500 stable source-hash rank using sample_id',
        'cross_panel_duplicates':'exclude positive base_window_indices from both negative candidate pools; high/control disjoint',
        'known_negative_semantics':'no added component A=0; existing natural vessels are not assumed absent',
        'diagnostic_score_threshold_ge':.5,'share_thresholds_strict_gt':[.03,.10,.20],
        'conditional_AUC':'same fixed syntheticpositive panel versus each naturalA0 panel; score and share',
        'model_members':list(MEMBERS)+['all5_frozen'],'all5_weights':policy['weights'],
        'model_selection_or_hyperparameter_tuning':False,'precision':'all panels recomputed CPUFP32; no saved GPU BF16 score mixing',
        'NN_threads':2,'tree_threads':'frozen tree API retains4; sequential after NN inference',
        'not_original_TRAIN_hard_negative_candidates':True,'source_pins':plan0['source_pins'],
        'frozen_policy_file_sha256':sha(pilot/'frozen_policy.json'),'audit_code_sha256':sha(__file__),
        'Test_report_newfit_external_transmission':False}
    save(OUT/'plan.json',specification)
    def guard(event,args):
        if event=='socket.connect':raise PermissionError('Network forbidden')
        if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
            name=Path(args[0]).name.lower()
            if name in ['test.csv','report.npz','report_metadata.parquet','report_recipes.json']:
                raise PermissionError('Test/report inputs forbidden')
    sys.addaudithook(guard);environment(2)
    # Numeric oracle: exact constant/common-level trajectories and a direct
    # unfloored linear-sum expression, all independent of source observations.
    oracle=np.zeros((3,15,25),np.float64);oracle[1]+=np.arange(15)[None,:,None][0]
    oracle[2,:,:13]=np.linspace(-10,10,15)[:,None];values=lf_std(oracle)
    expect=np.asarray([0.,np.arange(15).std(ddof=0),np.linspace(-10,10,15).std(ddof=0)])
    oracle_error=float(np.max(abs(values-expect)));assert oracle_error<1e-12
    direct=10*np.log10(np.power(10.,oracle[:,:,:13]/10).sum(2))
    assert np.max(abs(lf_std(oracle)-direct.std(1)))<1e-12
    assert abs(auc([0,0,1,1],[.1,.1,.9,.9])-1)<1e-12
    assert abs(auc([0,1],[.5,.5])-.5)<1e-12
    from pipeline.dataset_r9 import R9Sources,FixedR9Dataset
    sources=R9Sources(ROOT/'configs/r9_full_training_v1.json',project_root=ROOT,verify_banks=True)
    source_meta=sources.role_metadata(0,'selection');train_meta=sources.role_metadata(0,'train')
    assert not set(source_meta.background_group.astype(str))&set(train_meta.background_group.astype(str))
    assert not set(source_meta.base_window_index)&set(train_meta.base_window_index)
    frame=source_meta[source_meta.qc_observed_minutes==15].copy().reset_index(drop=True)
    db=np.stack([sources.background(row) for row in frame.to_dict('records')]);assert np.isfinite(db).all()
    variability=lf_std(db);frame['LF_temporal_std_db']=variability
    q90=float(np.quantile(variability,.9,method='linear'));q50=float(np.quantile(variability,.5,method='linear'))
    fixed=FixedR9Dataset(ROOT/plan0['bundle'],0,'selection');raw=fixed.raw_arrays()
    obs=np.isfinite(raw['db']).all(2).sum(1);positive_indices=np.flatnonzero((raw['presence']>0)&(obs==15))
    positive_frame=pd.DataFrame({k:raw[k][positive_indices] for k in ['sample_id','base_window_index','background_group','site','reference_index']})
    positive_frame['fixed_row_index']=positive_indices
    positive=choose(positive_frame,min(500,len(positive_frame)),'synthetic_positive','sample_id')
    if not len(positive):raise ValueError('No eligible fixed positive panel')
    excluded=set(positive.base_window_index)
    high_all=frame[(frame.LF_temporal_std_db>q90)&~frame.base_window_index.isin(excluded)]
    control_all=frame[(frame.LF_temporal_std_db<q50)&~frame.base_window_index.isin(excluded)]
    count=min(500,len(high_all),len(control_all))
    if not count:raise ValueError('Nonempty matched natural panels required')
    high=choose(high_all,count,'natural_high');control=choose(control_all,count,'natural_control')
    ids=[set(x.base_window_index) for x in [high,control,positive]]
    assert all(len(x)==len(set(x.base_window_index)) for x in [high,control,positive])
    assert not any(ids[i]&ids[j] for i in range(3) for j in range(i))
    frames={'natural_high':high,'natural_control':control,'synthetic_positive':positive}
    for name,meta in frames.items():meta.to_parquet(OUT/(name+'_membership.parquet'),index=False)
    selection_stats={'source_pool_rows':len(source_meta),'complete15_subset_rows':len(frame),
        'source_pool_groups':int(source_meta.background_group.nunique()),'LF_std_q90_db':q90,'LF_std_q50_db':q50,
        'high_candidates_after_positive_background_exclusion':len(high_all),
        'control_candidates_after_positive_background_exclusion':len(control_all),
        'negative_panel_rows_each':count,'eligible_syntheticpositive_rows':len(positive_frame),'positive_panel_rows':len(positive),
        'panels':{name:{'n':len(meta),'unique_backgrounds':int(meta.base_window_index.nunique()),
            'source_groups':int(meta.background_group.nunique()),'site_counts':{str(k):int(v) for k,v in meta.site.value_counts().items()}}
                  for name,meta in frames.items()},
        'TRAIN_background_group_and_window_overlap':0,'cross_panel_base_window_overlap':0,
        'purge_inherited_from':'data/versions/background_cv_r9/fold_00 selection/train manifests; recorded sensor/export-clock24h support purge',
        'source_manifest_sha256':sha(sources.bg_dir/'fold_00/selection_manifest.parquet'),
        'train_manifest_sha256':sha(sources.bg_dir/'fold_00/train_manifest.parquet'),
        'fixed_selection_identity':fixed.identity,'numeric_oracle_max_error':oracle_error,
        'panel_membership_sha256':{name:sha(OUT/(name+'_membership.parquet')) for name in frames}}
    save(OUT/'panel_selection.json',selection_stats)
    pins,_,binding,tree=inputs(plan0['nn_dirs'],ROOT/plan0['tree_job'],ROOT/plan0['bundle'],ROOT/plan0['config'])
    if pins!=plan0['source_pins'] or binding!=plan0['binding']:raise ValueError('Frozen model binding changed')
    panels={'natural_high':np.stack([sources.background(row) for row in high.to_dict('records')]),
            'natural_control':np.stack([sources.background(row) for row in control.to_dict('records')]),
            'synthetic_positive':raw['db'][positive.fixed_row_index.to_numpy(np.int64)]}
    allpred={};timings={}
    for name,observations in panels.items():
        pred,timing=predict_members(plan0['nn_dirs'],tree,observations,binding,batch_size=128,threads=2)
        pred['all5_frozen']=combine(pred,policy['weights']);allpred[name]=pred;timings[name]=timing
        np.savez_compressed(OUT/(name+'_predictions.npz'),
            base_window_index=frames[name].base_window_index.to_numpy(np.int64),
            share=np.stack([pred[k]['share'] for k in list(MEMBERS)+['all5_frozen']],axis=1),
            score=np.stack([pred[k]['score'] for k in list(MEMBERS)+['all5_frozen']],axis=1))
        print(json.dumps({'panel':name,'rows':len(observations),'status':'predicted'}),flush=True)
    metrics={}
    for name in ['natural_high','natural_control']:
        metrics[name]={}
        for kind in list(MEMBERS)+['all5_frozen']:
            v=zero_metrics(allpred[name][kind]);pos=allpred['synthetic_positive'][kind]
            labels=np.r_[np.ones(len(pos['share']),bool),np.zeros(count,bool)]
            v.update(conditional_score_AUC=auc(labels,np.r_[pos['score'],allpred[name][kind]['score']]),
                conditional_share_AUC=auc(labels,np.r_[pos['share'],allpred[name][kind]['share']]))
            metrics[name][kind]=v
    sources.assert_unchanged(verify_banks=True);verify_pins(pins)
    if sha(pilot/'frozen_policy.json')!=specification['frozen_policy_file_sha256']:raise ValueError('Policy changed')
    result={'format':'r10_natural_variability_audit_v1','status':'completed','plan_sha256':sha(OUT/'plan.json'),
        'panel_selection_sha256':sha(OUT/'panel_selection.json'),'selection':selection_stats,'metrics':metrics,
        'timings':timings,'member_order':list(MEMBERS)+['all5_frozen'],'frozen_weights':policy['weights'],
        'input_array_sha256':{k:array_sha(v) for k,v in panels.items()},
        'prediction_npz_sha256':{n:sha(OUT/(n+'_predictions.npz')) for n in panels},
        'source_pins_unchanged':True,'no_new_model_fit_source_sampler_policy_change_Test_report_or_external_transmission':True,
        'limitations':['A=0 is known absence of a controlled added component, not absence of natural vessel sound.',
            'These are selection-source diagnostic slices, not independent report or officialTest estimates.',
            'The positive panel is synthetic and was in the same selection role used for model checkpoint selection.',
            'Panels share source groups/sites within selection; marginal LF variability differences are not a causal LF-only effect.',
            'All509 references are shared across R9 folds/roles; conditionalAUC is not natural-vessel ground-truth AUC.',
            'This sampling rule is separate from the existing TRAIN hard-negative candidate procedure.'],
        'elapsed_seconds':time.perf_counter()-start,'audit_code_sha256':sha(__file__)}
    save(OUT/'results.json',result)
    fields=['panel','model',*next(iter(metrics['natural_high'].values())).keys()]
    with (OUT/'metrics.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for panel,models in metrics.items():
            for kind,v in models.items():w.writerow({'panel':panel,'model':kind,**v})
    print(json.dumps({'status':'completed','rows_each_A0':count,'positive_n':len(positive),
        'results_sha256':sha(OUT/'results.json'),'metrics':metrics,'seconds':result['elapsed_seconds']},indent=2))


if __name__=='__main__':main()
