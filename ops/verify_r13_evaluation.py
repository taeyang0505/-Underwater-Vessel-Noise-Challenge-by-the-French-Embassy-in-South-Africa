"""Independent R13 saved-output QA: no training, inference, Test or GPU.

Point AUC uses a direct pair oracle. Cluster CIs use expanded float64 rows
and searchsorted tie credit, independent of the producer's weighted-bin
bootstrap. Original R12/R13 source, policy and results remain immutable.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARMS=('baselineP0','baselineP1','highshareP0','highshareP1');LAMBDAS=(0.,.25,.5,1.)
EXPECTED_EVAL_SHA='14ba4d5d09c73d85a2054c8b2a8f1262fda20ecdff109903a7c2e8f24a7457c1'
REPS=500;SEED=2026102712

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def token(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def local(p):
    p=Path(p);p=ROOT/p if not p.is_absolute() else p
    if p.is_symlink():raise ValueError('Symlinkinputforbidden')
    p=p.resolve()
    if ROOT not in p.parents:raise ValueError('Project-containedpathrequired')
    return p
def pins(mapping):
    for n,h in mapping.items():
        if sha(local(n))!=h:raise AssertionError('Changedpinnedfile:'+n)
def close(a,b,label,tol=2e-10):
    if a is None or b is None:
        if a is not None or b is not None:raise AssertionError('Nullmismatch:'+label)
        return 0.
    d=abs(float(a)-float(b))
    if d>tol:raise AssertionError(f'{label}: {a} != {b}; delta{d}')
    return d
def parse(s):return None if s is None or str(s).strip()=='' else float(s)

def rank_auc(y,score):
    """Expanded-row rank oracle for bootstrap; no weighted-bin code."""
    import numpy as np
    z=np.asarray(y)>0;s=np.asarray(score,float);p=s[z];n=np.sort(s[~z])
    if not len(p) or not len(n):return None
    return float(np.sum(np.searchsorted(n,p,'left')+np.searchsorted(n,p,'right'))/(2*len(p)*len(n)))

def expanded_ci(name,y,groups,key,selected,other):
    import numpy as np
    from ops.verify_r12_evaluation_v2 import numeric_metrics
    a=numeric_metrics(y,*selected);b=numeric_metrics(y,*other);metrics=('rmse','auc_share','auc_score');values={k:[] for k in metrics}
    n=0 if groups is None else len(np.unique(np.asarray(groups).astype(str)))
    if n>=2:
        _,inverse=np.unique(np.asarray(groups).astype(str),return_inverse=True)
        rng=np.random.default_rng(np.random.SeedSequence([SEED,int(hashlib.sha256(name.encode()).hexdigest()[:8],16)]))
        yy=np.asarray(y,float);sa=[np.asarray(v,float) for v in selected];sb=[np.asarray(v,float) for v in other]
        for _ in range(REPS):
            weights=np.bincount(rng.integers(n,size=n),minlength=n)[inverse];ix=np.repeat(np.arange(len(y)),weights);target=yy[ix]
            values['rmse'].append(float(np.sqrt(np.mean((sa[0][ix]-target)**2))-np.sqrt(np.mean((sb[0][ix]-target)**2))))
            for j,m in [(0,'auc_share'),(1,'auc_score')]:
                u,v=rank_auc(target,sa[j][ix]),rank_auc(target,sb[j][ix])
                if u is not None and v is not None:values[m].append(u-v)
    return {m:{'point_difference':a[m]-b[m] if a[m] is not None and b[m] is not None else None,
        'lo95':float(np.quantile(values[m],.025)) if values[m] else None,'hi95':float(np.quantile(values[m],.975)) if values[m] else None,
        'n_clusters':n,'cluster_key':key,'repetitions':REPS,'valid_repetitions':len(values[m])} for m in metrics}

def source_db(name,p):
    """Approved completed-evaluation source db only, for weight oracle."""
    import numpy as np
    from ops.verify_r12_evaluation_v2 import R9,R10,R11
    def one(path):
        with np.load(path,allow_pickle=False) as z:return z['db'].copy()
    if name.startswith('R12_'):
        role=name[4:];m=read(local(p['data_dir'])/'manifest.json');e=m['common_validation'][role] if role!='high_snr_long' else m['high_snr_long']
        return one(local(p['data_dir'])/e['path'])
    if name.startswith('R9_'):return one(R9/(name[3:]+'.npz'))
    if name.startswith('R10_'):return one(R10/name[4:]/'samples.npz')
    if name.startswith('R11_'):
        a,b=name[4:].split('_VS_');return np.concatenate([one(R11/(a+'_samples.npz')),one(R11/(b+'_samples.npz'))])
    raise ValueError('Unapprovedpanel')

def source_groups(name,p):
    import numpy as np,pandas as pd
    from ops.verify_r12_evaluation_v2 import source_panel,R10
    y,ids,groups,key,sites=source_panel(name,p);audit=None
    if name.startswith('R10_unseen__'):
        meta=pd.read_parquet(R10/'unseen_source_manifest.parquet')
        groups=(meta.sensor_group.astype(str)+'__'+meta.recording_week_start.astype(str)).to_numpy();key='sensor_group_x_recording_week_start'
        # Assess, do not change, the declared block definition. Adjacent weeks
        # are not automatically separated by the24h split purge.
        times=pd.to_datetime(meta.start_export_clock).astype('int64').to_numpy()/1e9;pairs=[]
        for sensor,sub in meta.groupby('sensor_group',sort=True):
            ix=np.asarray(sub.index);labels=groups[ix];unique=np.unique(labels)
            for i,a in enumerate(unique):
                aa=times[ix[labels==a]]
                for b in unique[i+1:]:
                    bb=np.sort(times[ix[labels==b]]);positions=np.searchsorted(bb,aa)
                    nearest=np.minimum(abs(aa-bb[np.clip(positions,0,len(bb)-1)]),abs(aa-bb[np.clip(positions-1,0,len(bb)-1)]))
                    gap=max(float(nearest.min())-900,0.)
                    pairs.append({'sensor_group':str(sensor),'group_a':str(a),'group_b':str(b),'minimum_window_support_gap_seconds':gap})
        audit={'cluster_key':key,'clusters':len(np.unique(groups)),'source_base_IDs':int(meta.base_window_index.nunique()),
            'same_sensor_between_week_group_minimum_support_gap_seconds':min([v['minimum_window_support_gap_seconds'] for v in pairs],default=None),
            'week_group_pairs_below24h':sum(v['minimum_window_support_gap_seconds']<86400 for v in pairs),
            'group_gap_pairs':pairs,'clock':'original export clock, not independently verified UTC',
            'interpretation':'Sensor×weekconnectsavailablemetadataforR13CI;doesnotitselfguaranteeindependentor24h-separatedbootstrapblocks. OriginalR12nullCIsremainunchanged.'}
    return y,ids,groups,key,sites,audit

def extra_metrics(y,rawshare,rawscore,site,model,clf=None,contrast=None):
    import numpy as np
    from ops.verify_r12_evaluation_v2 import numeric_metrics,pair_auc
    yy=np.asarray(y,float);p=np.asarray(rawshare,float);s=np.asarray(rawscore,float);z=yy>0;err=p-yy
    m=numeric_metrics(yy,p,s,site,mixed=model=='R9_frozen_all5')
    # Reproduce the producer's float32 slicing/mean for this one field and
    # retain a separate float64 oracle with a declared precision budget.
    high=np.asarray(y,np.float32)>.2;low=(np.asarray(y,np.float32)>0)&(np.asarray(y,np.float32)<=.03)
    m.update(auc_classifier=pair_auc(yy,clf) if clf is not None else None,
        contrast_ge099_fraction=float(np.mean(contrast>=.99)) if contrast is not None else None,
        score_brier=float(np.mean((s-z.astype(float))**2)),score_ge099=float(np.mean(s>=.99)),
        zero_prediction_mean=float(np.mean(np.asarray(rawshare)[~z])) if (~z).any() else None,
        high_y_bias=float(err[high].mean()) if high.any() else None,
        high_y_rmse=float(np.sqrt(np.mean(err[high]**2))) if high.any() else None,
        low_positive_n=int(low.sum()),classifier_ge099=float(np.mean(clf>=.99)) if clf is not None else None)
    use=low|~z;m['low_positive_AUC']=pair_auc(yy[use],s[use]) if low.any() and (~z).any() else None
    doublemean=float(p[~z].mean()) if (~z).any() else None
    return m,doublemean

def model_checks(c,t,data_manifest):
    import numpy as np,torch
    result={};expected=26953
    for arm in ARMS:
        info=c['models'][arm];sp=local(info.get('summary_path',info.get('reused_summary_path')));s=read(sp);ckpath=local(info['checkpoint_path'])
        if sha(sp)!=info.get('summary_sha256',info.get('reused_summary_sha256')) or sha(ckpath)!=info['checkpoint_sha256']:raise AssertionError('CNNsummary/checkpointchanged')
        h=[json.loads(v) for v in (sp.parent/'epochs.jsonl').read_text().splitlines()];values=np.asarray([v['validation_fit_rmse'] for v in h],float)
        if [v['epoch'] for v in h]!=list(range(1,101)) or not np.isfinite(values).all():raise AssertionError('CNNfixed100historydiffers')
        best=int(np.argmin(values))+1;ck=torch.load(ckpath,map_location='cpu',weights_only=True)
        if (s['epochs_run']!=100 or s['early_stopping'] is not False or s['optimizer']!='Adam' or s['dropout']!=.1 or s['parameters']!=expected
            or s['best_epoch']!=best or ck['epoch']!=best or ck['arm']!=s['arm'] or ck['parameters']!=expected
            or sum(v.numel() for v in ck['state_dict'].values())!=expected):raise AssertionError('CNNfirstmin/architecture/metadatafailed')
        if not all(torch.isfinite(v).all() for v in ck['state_dict'].values()):raise AssertionError('NonfiniteCNNweights')
        if s['arm']['shape']!=arm.endswith('1'):raise AssertionError('CNNshapevariantdiffers')
        ti=t['models'][arm];ts=local(ti['summary_path']);tree=read(ts);b=tree['training_binding'];shape=arm.endswith('1');data_arm='baseline' if arm.startswith('baseline') else 'highshare'
        if sha(ts)!=ti['summary_sha256'] or tree.get('fixture_only') is not False:raise AssertionError('Treecompletiondiffers')
        if (b['R13_data_hash']!=data_manifest['data_hash'] or b['data_arm']!=data_arm or b['include_shape']!=shape
            or b['TRAIN_npz_sha256']!=data_manifest['arms'][data_arm]['sha256']
            or tree['feature_count']!=(673 if shape else 466) or tree['train_rows']!=12000 or tree['validation_fit_rows']!=1000):raise AssertionError('Treeinputbindingfailed')
        basis_sha=sha(ROOT/'models/r12_shared_basis_v1/basis.npz')
        if (b['R13_manifest_sha256']!=sha(ROOT/'data/synthetic/r13_cpu_correction_v1/manifest.json')
            or b['basis_sha256']!=(basis_sha if shape else None)
            or b['basis_training_binding']!=(data_manifest['basis_training_binding'] if shape else None)
            or tree['settings']['maximum_trees']!=1500 or tree['settings']['early_stopping_patience']!=40
            or tree['settings']['threads']!=2 or tree['seed']!=2026102712):raise AssertionError('TreePCA/trainingpolicybindingfailed')
        for n,v in tree['artifact_sha256'].items():
            if sha(ts.parent/n)!=v:raise AssertionError('Treeartifactchanged')
        for n,v in tree['source_code_sha256'].items():
            if sha(local(n))!=v:raise AssertionError('Treefeaturecodechanged')
        result[arm]={'CNN_checkpoint_sha256':sha(ckpath),'CNN_best_epoch_first_min':best,'CNN_parameter_tensor_count':expected,
            'CNN_reused_completed_control':arm=='baselineP1','tree_summary_sha256':sha(ts),'tree_features':tree['feature_count'],
            'regressor_sha256':sha(ts.parent/'share.txt'),'classifier_sha256':sha(ts.parent/'presence.txt')}
    return result

def verify(a):
    import numpy as np
    from ops.verify_r12_evaluation_v2 import read as _,numeric_metrics,pair_auc,crc
    out=local(a.results).parent;r=read(local(a.results));freeze=read(out/'FROZEN_BEFORE_GATE.json');policy=read(out/'selected_policy.json')
    if r.get('status')!='completed' or r.get('format')!='r13_cpu_evaluation_v1' or (out/'FAILED.json').exists():raise AssertionError('CompletedunfailedR13evaluationrequired')
    if sha(ROOT/'ops/evaluate_r13_cpu.py')!=EXPECTED_EVAL_SHA:raise AssertionError('Expectedfrozenevaluatorchanged')
    if token({k:v for k,v in freeze.items() if k!='freeze_sha256'})!=freeze['freeze_sha256']:raise AssertionError('GlobalfreezecontentsSHAfailed')
    if (policy!=r['selected_policy'] or token({k:v for k,v in policy.items() if k!='policy_sha256'})!=policy['policy_sha256']
        or policy['freeze_sha256']!=freeze['freeze_sha256'] or policy['regression_weights']!={'CNN':.5,'tree':.5}):raise AssertionError('Policybindingfailed')
    if r.get('no_Test_or_GPU_or_refit') is not True or r.get('raw_masks_weights_classifiers_saved') is not True:raise AssertionError('Scopeclaimsfailed')
    pins(r['source_pins']);pins(freeze['source_pins'])
    if r['source_pins']!=freeze['source_pins']:raise AssertionError('Sourcepincensuschanged')
    for n,v in r['output_sha256'].items():
        if Path(n).is_absolute() or '..' in Path(n).parts or sha(out/n)!=v:raise AssertionError('Outputartifactchanged')
    config=read(ROOT/'configs/r13_cpu_correction_v1.json');data=read(ROOT/'data/synthetic/r13_cpu_correction_v1/manifest.json')
    if config['training_arms']!=list(ARMS) or config['score_candidates_classifier_weight']!=list(LAMBDAS) or data['data_hash']!=freeze['data_hash']:raise AssertionError('Prospectiveconfig/dataidentitychanged')
    c=read(ROOT/'models/r13_cpu_cnn_r1/ALL_CNN_FROZEN.json');t=read(ROOT/'models/r13_compact_trees_v1/ALL_TREES_FROZEN.json')
    if (any(v.get('status')!='completed' or v.get('all_models_frozen') is not True for v in (c,t))
        or set(c['models'])!=set(ARMS) or set(t['models'])!=set(ARMS)):raise AssertionError('All8logicalmodelsnotcomplete')
    models=model_checks(c,t,data)
    with np.load(ROOT/'models/r12_shared_basis_v1/basis.npz',allow_pickle=False) as z:
        basis=json.loads(z['metadata_json'].item());basis['mean_db']=z['mean_db'].astype(float).tolist();basis['components']=z['components'].astype(float).tolist()
    if (token({k:v for k,v in basis.items() if k!='basis_sha256'})!=basis['basis_sha256'] or basis['fit_split']!='train'
        or basis['source_kind']!='released_sa_background' or basis['n_components']!=3
        or basis['training_binding']!=data['basis_training_binding'] or basis['unique_background_source_count']!=2000):raise AssertionError('SharedTRAIN-onlyPCAstatefailed')
    oldfreeze=read(ROOT/'reports/R12_ALL17_GLOBAL_FREEZE.json');p=read(local(oldfreeze['plan_file']))
    with (out/'results.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    rm={(v['panel'],v['model']):v for v in rows}
    if len(rm)!=len(rows) or len(rows)!=458 or r['metric_rows']!=458:raise AssertionError('Expected458uniqueR13metricrows')
    files={v.stem:v for v in (out/'predictions').glob('*.npz')}
    expectedpanels=['R12_validation_gate','R12_eval','R12_high_snr_long','R9_selection','R9_ensemble_gate']+['R10_'+n for n in ('unseen__r9_complete','seen__mean_scale_only','unseen__combined_stress','seen__combined_stress')]+['R11_'+u+'_VS_'+v for u in ('full_smooth12','full_smooth13','full_smooth15','native_long15') for v in ('st_francis_dense','hf_high')]
    if set(files)!=set(expectedpanels):raise AssertionError('Expected17panelpaths')
    maxmetric=maxpool=maxw=maxcontrast=maxzerodtype=0.;checks=0;groupsmeta={};allpred={};panels={};zero_precision=[]
    orderbase=[v for arm in ARMS for v in ['CNN_'+arm,'Tree_'+arm]+['Blend50_'+arm+'_lambda'+str(w) for w in LAMBDAS]]+['R13_selected','R12_matched_share_only']
    for name,path in files.items():
        crc(path)
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        order=arrays['model_order'].astype(str).tolist();expected=orderbase+([] if name=='R12_validation_gate' else ['R9_frozen_all5'])
        if order!=expected:raise AssertionError('R13modelcolumnorderchanged')
        y,ids,groups,key,sites,ga=source_groups(name,p)
        if not np.array_equal(arrays['sample_id'].astype(str),ids) or not np.array_equal(arrays['target_share'],y):raise AssertionError('SourcesampleID/targetorderdiffers')
        if ga is not None:groupsmeta[name]=ga
        panels[name]=(y,groups,key);pred={v:(arrays['share'][:,j],arrays['score'][:,j]) for j,v in enumerate(order)};allpred[name]=pred
        w=arrays['observed_weights'];db=source_db(name,p);qdb=np.round(db.astype(float),1);mask=np.isfinite(qdb)
        energy=np.where(mask,10**((qdb-np.nanmax(qdb,(1,2),keepdims=True))/10),0.);oracle=(energy/energy.sum((1,2),keepdims=True)).astype(np.float32)
        wd=float(abs(w-oracle).max());maxw=max(maxw,wd)
        if wd>3e-7 or not np.isfinite(w).all() or (w<0).any() or np.max(abs(w.sum((1,2))-1))>3e-6:raise AssertionError('Roundedphysicalweightoraclefailed')
        for arm in ARMS:
            q=arrays[arm+'_cell_mask'];clf=arrays[arm+'_classifier'];contrast=arrays[arm+'_contrast']
            if q.shape!=w.shape or not np.isfinite(q).all() or ((q<0)|(q>1)).any() or np.any(q[w==0]!=0):raise AssertionError('Savedqbounds/missingmaskfailed')
            delta=float(abs((w.astype(float)*q).sum((1,2))-pred['CNN_'+arm][0]).max());maxpool=max(maxpool,delta)
            if delta>3e-6:raise AssertionError('Independentqphysicalpoolfailed')
            qm=q[:,:,10:23];om=w[:,:,10:23]>0;floor=np.min(np.where(om,qm,np.inf),1,keepdims=True);floor=np.where(om.any(1,keepdims=True),floor,0.)
            co=np.max(np.where(om,np.maximum(qm-floor,0),0),(1,2));cd=float(abs(co-contrast).max());maxcontrast=max(maxcontrast,cd)
            if cd>2e-7:raise AssertionError('Contrastoraclefailed')
            if not np.isfinite(clf).all() or ((clf<0)|(clf>1)).any():raise AssertionError('Classifierarrayinvalid')
            if not np.array_equal(pred['CNN_'+arm][0],pred['CNN_'+arm][1]):raise AssertionError('CNNshareonlyscorefailed')
            blend=(.5*pred['CNN_'+arm][0].astype(float)+.5*pred['Tree_'+arm][0].astype(float)).astype(np.float32)
            for lam in LAMBDAS:
                a0,b0=pred['Blend50_'+arm+'_lambda'+str(lam)];score=((1-lam)*blend.astype(float)+lam*clf.astype(float)).astype(np.float32)
                if not np.array_equal(a0,blend) or not np.array_equal(b0,score):raise AssertionError('Fixedregression/scoremixturefailed')
            treescore=(.5*pred['Tree_'+arm][0].astype(float)+.5*clf.astype(float)).astype(np.float32)
            if not np.array_equal(pred['Tree_'+arm][1],treescore):raise AssertionError('Legacytreediagnosticscorefailed')
        chosen='Blend50_'+policy['selected_arm']+'_lambda'+str(policy['score_classifier_weight'])
        if any(not np.array_equal(pred['R13_selected'][j],pred[chosen][j]) for j in (0,1)):raise AssertionError('Selectedaliasdiffers')
        original=ROOT/'reports/r12_mask_matched_evaluation_r2/predictions'/(name+'.npz')
        with np.load(original,allow_pickle=False) as z:
            oi=z['model_order'].astype(str).tolist()
            for label,old in [('R12_matched_share_only','Blend50_L1HM1P1'),('R9_frozen_all5','R9_frozen_all5')]:
                if label not in pred:continue
                j=oi.index(old)
                if not np.array_equal(pred[label][0],z['share'][:,j]) or not np.array_equal(pred[label][1],z['score'][:,j]):raise AssertionError('Storedoldcomparatorchanged')
        for model,(share,score) in pred.items():
            clf=contrast=None
            if model.startswith('CNN_'):contrast=arrays[model[4:]+'_contrast']
            elif model.startswith('Tree_'):clf=arrays[model[5:]+'_classifier']
            elif model.startswith('Blend50_'):clf=arrays[model[len('Blend50_'):].split('_lambda')[0]+'_classifier']
            elif model=='R13_selected':clf=arrays[policy['selected_arm']+'_classifier']
            oracle,doublemean=extra_metrics(y,share,score,sites,model,clf,contrast);row=rm[(name,model)]
            if row['development_not_blind']!='True':raise AssertionError('Falseblindevaluationclaim')
            for field,value in oracle.items():maxmetric=max(maxmetric,close(parse(row[field]),value,name+model+field));checks+=1
            if doublemean is not None:
                zerr=close(parse(row['zero_prediction_mean']),doublemean,name+model+'zeroDoubleMean',2e-7);maxzerodtype=max(maxzerodtype,zerr)
            zero_precision.append({'panel':name,'model':model,'producer_zero_mean':oracle['zero_prediction_mean'],'independent_float64_zero_mean':doublemean})
    gy=panels['R12_validation_gate'][0];gp=allpred['R12_validation_gate'];gate_rmse={arm:float(np.sqrt(np.mean((gp['Blend50_'+arm+'_lambda0.0'][0].astype(float)-gy.astype(float))**2))) for arm in ARMS}
    winner=min(ARMS,key=lambda v:gate_rmse[v]);gate_auc={str(l):pair_auc(gy,gp['Blend50_'+winner+'_lambda'+str(l)][1]) for l in LAMBDAS};lam=max(LAMBDAS,key=lambda v:gate_auc[str(v)])
    if policy['selected_arm']!=winner or policy['score_classifier_weight']!=lam:raise AssertionError('Gateargmin/argmaxfirsttieoraclefailed')
    for arm,v in gate_rmse.items():close(policy['gate_rmse'][arm],v,'policyRMSE',1e-12)
    for l,v in gate_auc.items():close(policy['selected_arm_gate_score_AUC'][l],v,'policyAUC',1e-12)
    with (out/'paired_cluster_CI.csv').open(newline='') as f:cir=list(csv.DictReader(f))
    cm={(v['panel'],v['comparison'],v['metric']):v for v in cir}
    if len(cm)!=len(cir) or len(cir)!=99:raise AssertionError('Expected99CIrows')
    maxci=0.;cichecks=0;numeric=0;missing=0
    for name,pred in allpred.items():
        y,groups,key=panels[name]
        for other in ['R12_matched_share_only','R9_frozen_all5']:
            if other not in pred:continue
            expected=expanded_ci(name,y,groups,key,pred['R13_selected'],pred[other])
            for metric,values in expected.items():
                row=cm[(name,'selected_minus_'+other,metric)];numeric+=int(values['lo95'] is not None);missing+=int(values['lo95'] is None)
                for field,value in values.items():
                    if field=='cluster_key':
                        if (row[field] or None)!=value:raise AssertionError('CIclusterkeychanged')
                    else:maxci=max(maxci,close(parse(row[field]),value,name+other+metric+field,2e-9));cichecks+=1
    return {'format':'r13_saved_evaluation_independent_qa_v1','status':'passed','result_sha256':sha(local(a.results)),
        'prediction_panels':17,'metric_rows':458,'scalar_metrics_checked':checks,'scalar_metric_max_error':maxmetric,
        'gate_regression_first_argmin':winner,'gate_presence_first_argmax_lambda':lam,'policy_file_sha256':sha(out/'selected_policy.json'),
        'physical_weight_oracle_max_error':maxw,'predicted_q_weighted_share_pool_max_error':maxpool,'contrast_oracle_max_error':maxcontrast,
        'raw_classifier_and_contrast_metrics_independently_covered':True,'all_fixed_regression_score_mixtures_exact':True,
        'zero_mean_precision':{'producer_float32_mean_exactly_replayed':True,'independent_float64_comparison_abs_tolerance':2e-7,
            'max_float32_vs_float64_zero_mean_delta':maxzerodtype,'applies_only_to_field':'zero_prediction_mean','details':zero_precision},
        'CI_rows':99,'CI_numeric_intervals':numeric,'CI_missing_intervals':missing,'CI_scalar_checks':cichecks,'CI_max_error':maxci,
        'CI_oracle':'expandedfloat64bootstraprowsplussearchsortedtieAUC;independentoftheproducerweightedbinmethod',
        'new_unseen_cluster_audits':groupsmeta,'models':models,'source_pins_verified':r['source_pins'],
        'logical_CNN_models':4,'CNN_new_fits':3,'completed_control_reused':1,'fresh_tree_fits':4,
        'source_model_results_policy_unchanged':True,'new_training_model_inference_Test_GPU_or_external_actions':False,
        'limitations':['Development-exposedpanels/sharedreferences;notblindOOFandnotofficialLBforecast.',
            'Gateisreusedformultipledevelopmentsteps;conditionalCIsdonotremoveselection/multiple-comparisonoptimism.',
            'Sensor×weekbootstrapclustersneednotbe24h-separated;reportedmetadata-gapdiagnosticsdonotchangeCIs.',
            'PresenceAUC/BrierandphysicalenergyRMSEaredifferenttargets;score-onlychangesdonotimproveregression.'],
        'verifier_sha256':sha(__file__)}

def fixture():
    import numpy as np
    from ops.verify_r12_evaluation_v2 import pair_auc
    rng=np.random.default_rng(1313);y=np.array([0,0,.03,.1,.2,.6],np.float32);s=np.round(rng.random(6),1).astype(np.float32);g=np.array(['s0w0']*2+['s0w1']*2+['s1w0']*2)
    close(pair_auc(y,s),rank_auc(y,s),'pair/rankAUCfixture',1e-12)
    ci=expanded_ci('fixture',y,g,'sensor_week',(s,s),(s,s))
    if any(v['point_difference']!=0 or v['lo95']!=0 or v['hi95']!=0 for v in ci.values()):raise AssertionError('ExpandedidentityCIfailed')
    # Verify float32 zero mean and independent float64 are recorded separately.
    m,double=extra_metrics(y,s,s,np.array(['a']*6),'fixture')
    if m['zero_prediction_mean']!=float(np.mean(s[:2])):raise AssertionError('Producerfloat32meanfixture')
    q=np.full((3,15,25),.3,np.float32);w=np.full_like(q,1/375);p=(q.astype(float)*w).sum((1,2));assert np.max(abs(p-.3))<1e-6
    return {'status':'passed','numeric_only':True,'pairwise_vs_rank_AUC_oracle':True,'expanded500_cluster_identity_CI_zero':True,
        'zero_mean_producerdtype_vs_float64_separate':True,'physical_pool_fixture':True,'actual_data_models_or_evaluation_read':False,'verifier_sha256':sha(__file__)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',default='reports/r13_cpu_correction_r1/RESULTS.json');p.add_argument('--output-prefix',default='reports/R13_EVALUATION_INDEPENDENT_QA')
    p.add_argument('--self-test',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--timeout',type=int,default=7200);a=p.parse_args()
    from ops.verify_r12_evaluation_v2 import guard
    sys.addaudithook(guard)
    if a.self_test:print(json.dumps(fixture()));return
    path=local(a.results);started=time.monotonic()
    def complete():
        try:return read(path).get('status')=='completed'
        except (FileNotFoundError,json.JSONDecodeError):return False
    while not complete():
        if (path.parent/'FAILED.json').exists():raise RuntimeError('Evaluationfailed;preservepartial')
        if not a.watch:print(json.dumps({'status':'pending_complete_evaluation','actual_arrays_or_models_read':False}));raise SystemExit(3)
        if time.monotonic()-started>a.timeout:raise TimeoutError('Boundedwaitexpired')
        time.sleep(30)
    prefix=local(a.output_prefix);jp=Path(str(prefix)+'.json');mp=Path(str(prefix)+'.md')
    if jp.exists() or mp.exists():raise FileExistsError('PreservepriorQAoutput')
    try:r=verify(a)
    except BaseException as e:
        jp.write_text(json.dumps({'status':'failed','type':type(e).__name__,'error':str(e),'no_refit_or_new_inference':True},indent=2)+'\n');raise
    jp.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    text=['# R13 평가 독립 검산','',f"PASS: 17개 패널·458개 지표 행, {r['scalar_metrics_checked']}개 scalar와99개 CI 행을 저장 배열에서 검산했습니다.",
        f"Gate RMSE 첫 argmin은 {r['gate_regression_first_argmin']}, 해당 recipe의 presence AUC 첫 argmax λ는 {r['gate_presence_first_argmax_lambda']}로 정책과 일치합니다.",
        f"반올림 dB에서 독립 계산한 물리 가중치 오차 최대 {r['physical_weight_oracle_max_error']:.3g}, Σwq와 CNN share 오차 {r['predicted_q_weighted_share_pool_max_error']:.3g}, contrast 오차 {r['contrast_oracle_max_error']:.3g}입니다. q 범위와 결측 q=0도 확인했습니다.",
        'Raw classifier·contrast가 저장되어 해당 AUC·포화·Brier까지 검산했습니다. CNN4개는3개 신규 fit과1개 완료 control 재사용이며 tree4개는 새 fit입니다. checkpoint·100epochs·첫 best epoch·PCA·source hash도 확인했습니다.',
        f"zero_prediction_mean의 원래 float32 mean은 정확 재현하고 float64 oracle를 별도 기록했습니다. 최대 차이 {r['zero_mean_precision']['max_float32_vs_float64_zero_mean_delta']:.3g}, 이 필드만 절대허용치2×10⁻⁷입니다. 다른 점추정과 AUC는 기준을 바꾸지 않았습니다.",
        f"CI는 expanded float64 row와 searchsorted tie AUC로 독립 재계산했습니다. 수치 구간 {r['CI_numeric_intervals']}행, 미계산 {r['CI_missing_intervals']}행, 최대 차이 {r['CI_max_error']:.3g}입니다.",
        '', '## Unseen CI와 해석 한계','',
        'R13 unseen CI는 sensor×recording_week_start를 명시 연결한 새 결과입니다. 주간 cluster 사이가 자동으로24시간 이상 떨어지거나 독립이라는 뜻은 아닙니다. 실제 export-clock support gap과24시간 미만인 pair 수를 JSON에 기록했고 CI를 수정하지 않았습니다. 원 R12 null CI도 그대로입니다.',
        '같은 참조와 이미 노출된 개발 패널을 사용했습니다. 새 blind OOF, 공식 LB 예측, 인과 원인의 완전한 증명이 아닙니다. Gate 반복 사용과 다중 비교의 낙관은 남으며 presence Brier/AUC와 에너지 RMSE를 구분해야 합니다.',
        '', '[전체 수치·군집 gap·모델 SHA]('+jp.name+')']
    mp.write_text('\n'.join(text)+'\n');print(json.dumps({'status':'passed','QA':str(jp),'sha256':sha(jp)}))

if __name__=='__main__':main()
