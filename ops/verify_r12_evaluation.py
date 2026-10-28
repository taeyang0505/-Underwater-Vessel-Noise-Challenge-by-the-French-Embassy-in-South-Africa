"""Independent saved-output R12 QA; no fitting, inference or Test access.

The production evaluator is never imported. Read scalar targets, approved
source metadata, completed artifacts and saved share/score predictions only.
Classifier/contrast arrays are not saved, so their exact scalar metrics are
outside numerical coverage. Optional waiting merely waits for final RESULTS.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARMS=tuple(f'L{l}H{h}P{s}' for l in (0,1) for h in (0,1) for s in (0,1))
CNN=('legacy_contrast_control',)+ARMS
MODELS=tuple('CNN_'+n for n in CNN)+tuple('Tree_'+n for n in ARMS)+tuple('Blend50_'+n for n in ARMS)
R11POS=('full_smooth12','full_smooth13','full_smooth15','native_long15')
R11NEG=('st_francis_dense','hf_high')
R10NAMES=('unseen__r9_complete','seen__mean_scale_only','unseen__combined_stress','seen__combined_stress')
R11=ROOT/'reports/r11_diagnostics/full_support_hf_r1'
R10=ROOT/'reports/r10_diagnostics/factorial_transfer_r2'
R9=ROOT/'data/synthetic/group_cv_minute_full_r9_v1/fold_00'
REPS=500;SEED=2026102712

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def token(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def local(p):
    a=Path(p);a=(ROOT/a if not a.is_absolute() else a)
    if a.is_symlink():raise ValueError('Symlinkinputforbidden:'+str(a))
    a=a.resolve()
    if ROOT not in a.parents:raise ValueError('Project-containedpathrequired')
    return a
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0]).replace('\\','/').lower()
        if Path(p).name in ('test.csv','sample_submission.csv','report.npz','outer_validation.npz') or '/sealed/' in p:
            raise PermissionError('Test/unusedR9report/oldOUTER/sealedforbidden')
    if event in ('socket.connect','subprocess.Popen','os.system'):raise PermissionError('Noexternalorprocessactions')
def verify_map(mapping):
    for name,h in mapping.items():
        if sha(local(name))!=h:raise AssertionError('Changedpinnedfile:'+name)
def crc(path):
    with zipfile.ZipFile(path) as z:
        bad=z.testzip()
        if bad is not None:raise AssertionError('NPZCRCfailed:'+bad)

def pair_auc(y,s):
    """Direct positive-negative pair oracle, including exact tie credit."""
    import numpy as np
    z=np.asarray(y)>0;s=np.asarray(s,float);p=s[z];n=s[~z]
    if not len(p) or not len(n):return None
    wins=0.
    for start in range(0,len(p),128):
        a=p[start:start+128,None]
        wins+=float((a>n[None,:]).sum())+.5*float((a==n[None,:]).sum())
    return wins/(len(p)*len(n))

def numeric_metrics(y,share,score,site=None,mixed=False):
    import numpy as np
    y=np.asarray(y,float);p=np.asarray(share,float);s=np.asarray(score,float);e=p-y;hi=y>.2;sse=float((e*e).sum())
    out={'n':len(y),'positive_n':int((y>0).sum()),'zero_n':int((y==0).sum()),
        'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(abs(e).mean()),'bias':float(e.mean()),
        'prediction_mean':float(p.mean()),'prediction_rms':float(np.sqrt(np.mean(p*p))),
        'target_mean':float(y.mean()),'target_rms':float(np.sqrt(np.mean(y*y))),
        'auc_share':pair_auc(y,p),'auc_score':pair_auc(y,s),'high_y_n':int(hi.sum()),
        'high_y_SSE':float((e[hi]**2).sum()),'high_y_SSE_fraction':float((e[hi]**2).sum()/sse) if sse else None,
        'site_macro_rmse':None,'component_mixed_ge099_fraction':float(np.mean(2*s-p>=.99)) if mixed else None}
    if site is not None:
        labels=np.asarray(site).astype(str)
        out['site_macro_rmse']=float(np.mean([np.sqrt(np.mean(e[labels==v]**2)) for v in np.unique(labels)]))
    return out

def weighted_auc_prepare(y,s):
    import numpy as np
    z=np.asarray(y)>0;_,inverse=np.unique(np.asarray(s,float),return_inverse=True)
    def calculate(weights):
        a=np.bincount(inverse,weights=np.asarray(weights)*z);b=np.bincount(inverse,weights=np.asarray(weights)*(~z))
        if not a.sum() or not b.sum():return None
        return float(np.sum(a*(np.cumsum(b)-.5*b))/(a.sum()*b.sum()))
    return calculate

def cluster_ci(name,y,groups,key,selected,other):
    """Independent weighted-cluster bootstrap; never expand duplicate rows."""
    import numpy as np
    pointa=numeric_metrics(y,selected[0],selected[1]);pointb=numeric_metrics(y,other[0],other[1])
    metrics=('rmse','auc_share','auc_score');n=0 if groups is None else len(np.unique(np.asarray(groups).astype(str)))
    values={m:[] for m in metrics}
    if n>=2:
        _,inverse=np.unique(np.asarray(groups).astype(str),return_inverse=True)
        rng=np.random.default_rng(np.random.SeedSequence([SEED,int(hashlib.sha256(name.encode()).hexdigest()[:8],16)]))
        ea=(np.asarray(selected[0],float)-y)**2;eb=(np.asarray(other[0],float)-y)**2
        af=[weighted_auc_prepare(y,v) for v in selected];bf=[weighted_auc_prepare(y,v) for v in other]
        for _ in range(REPS):
            w=np.bincount(rng.integers(n,size=n),minlength=n)[inverse]
            values['rmse'].append(float(np.sqrt(np.sum(w*ea)/w.sum())-np.sqrt(np.sum(w*eb)/w.sum())))
            for j,m in [(0,'auc_share'),(1,'auc_score')]:
                a,b=af[j](w),bf[j](w)
                if a is not None and b is not None:values[m].append(a-b)
    return {m:{'point_difference':pointa[m]-pointb[m] if pointa[m] is not None and pointb[m] is not None else None,
        'lo95':float(np.quantile(values[m],.025)) if values[m] else None,
        'hi95':float(np.quantile(values[m],.975)) if values[m] else None,
        'n_clusters':n,'cluster_key':key,'repetitions':REPS,'valid_repetitions':len(values[m])} for m in metrics}

def source_panel(name,p):
    """Only target/ID members and metadata; never read source db members."""
    import numpy as np,pandas as pd
    def one(label,npz_path,metadata):
        with np.load(npz_path,allow_pickle=False) as z:
            y=z['share'].astype(np.float32);ids=z['sample_id'].astype(str) if 'sample_id' in z.files else np.asarray([label+'_'+str(i) for i in range(len(y))])
            source=z['source_id'].copy() if 'source_id' in z.files else None
        meta=pd.read_parquet(metadata)
        if len(meta)!=len(y):raise AssertionError('Sourcemetadatarowcountdiffers:'+label)
        if 'sample_id' in meta and not np.array_equal(ids,meta.sample_id.astype(str).to_numpy()):raise AssertionError('SourceIDorderdiffers:'+label)
        groups=meta.background_group.astype(str).to_numpy() if 'background_group' in meta else source
        key='background_group' if 'background_group' in meta else 'source_id_fallback' if source is not None else None
        sites=meta.site.astype(str).to_numpy() if 'site' in meta else None
        return y,ids,groups,key,sites
    if name.startswith('R12_'):
        role=name[4:];m=read(local(p['data_dir'])/'manifest.json');e=m['common_validation'][role] if role!='high_snr_long' else m['high_snr_long']
        return one(name,local(p['data_dir'])/e['path'],local(p['data_dir'])/e['metadata_path'])
    if name.startswith('R9_'):
        role=name[3:];return one(name,R9/(role+'.npz'),R9/(role+'_metadata.parquet'))
    if name.startswith('R10_'):
        panel=name[4:];return one(name,R10/panel/'samples.npz',R10/(panel.split('__')[0]+'_source_manifest.parquet'))
    if name.startswith('R11_') and '_VS_' in name:
        positive,negative=name[4:].split('_VS_');a=one('R11_'+positive,R11/(positive+'_samples.npz'),R11/'positive_background_manifest.parquet')
        b=one('R11_'+negative,R11/(negative+'_samples.npz'),R11/(negative+'_background_manifest.parquet'))
        if a[3]!=b[3]:raise AssertionError('CombinedR11clusterdefinitiondiffers')
        return (np.r_[a[0],b[0]],np.r_[a[1],b[1]],np.r_[a[2],b[2]],a[3],np.r_[a[4],b[4]])
    raise ValueError('Unapprovedevaluationpanel:'+name)

def saved_R9_baseline(name,ids):
    """Independent replay of stored old predictions, never a model forward."""
    import numpy as np
    pilot=ROOT/'submissions/r9_fold0_all5_cpu_r1'
    if name=='R9_selection':
        with np.load(ROOT/'reports/r10_diagnostics/temporal_permutation_r1/original_CPU_predictions.npz',allow_pickle=False) as z:
            oldids=z['sample_id'].astype(str);lookup={v:i for i,v in enumerate(oldids)};ix=np.asarray([lookup[v] for v in ids]);col=z['member_order'].astype(str).tolist().index('frozen_all5')
            return z['share'][ix,col],z['score'][ix,col]
    if name=='R9_ensemble_gate':
        weights=np.asarray(read(pilot/'frozen_policy.json')['weights'],float)
        with np.load(pilot/'gate_member_predictions.npz',allow_pickle=False) as z:
            if not np.array_equal(ids,z['sample_id'].astype(str)):raise AssertionError('OldR9gateIDmismatch')
            return tuple(np.clip(z[k].astype(float)@weights,0,1).astype(np.float32) for k in ('share','score'))
    if name.startswith('R10_'):
        with np.load(R10/name[4:]/'predictions.npz',allow_pickle=False) as z:return z['frozen_all5_share'].copy(),z['frozen_all5_score'].copy()
    if name.startswith('R11_'):
        a,b=name[4:].split('_VS_');values=[]
        for k in ('share','score'):
            with np.load(R11/(a+'_predictions.npz'),allow_pickle=False) as x,np.load(R11/(b+'_predictions.npz'),allow_pickle=False) as y:
                values.append(np.r_[x['all5_frozen_'+k],y['all5_frozen_'+k]])
        return tuple(values)
    return None

def verify_models(p,f):
    """Checkpoint deserialization only; no factories/forward/model inference."""
    import numpy as np,torch
    cnn=local(p['cnn_root']);tree=local(p['tree_root']);m=read(local(p['data_dir'])/'manifest.json');basis=sha(local(p['basis_path']))
    cf=read(cnn/'ALL_CNN_FROZEN.json');tf=read(tree/'ALL_TREES_FROZEN.json')
    if set(cf['summary_sha256'])!=set(CNN) or set(tf['summary_sha256'])!=set(ARMS):raise AssertionError('Incomplete9CNN8treefreeze')
    models={};closure={}
    for kind,names,folder,cert in [('CNN',CNN,cnn,cf),('tree',ARMS,tree,tf)]:
        for name in names:
            d=folder/name;s=read(d/'summary.json')
            if s.get('status')!='completed' or (d/'FAILED.json').exists() or sha(d/'summary.json')!=cert['summary_sha256'][name]:raise AssertionError('Incompleteorchangedmodel:'+name)
            if f['model_summary_sha256'].get(str((d/'summary.json').relative_to(ROOT)))!=sha(d/'summary.json'):raise AssertionError('Modelnotinglobalfreeze:'+name)
            if kind=='CNN':
                hist=[json.loads(v) for v in (d/'epochs.jsonl').read_text().splitlines()]
                if [v['epoch'] for v in hist]!=list(range(1,101)):raise AssertionError('Incomplete100epochhistory:'+name)
                values=np.asarray([v['validation_fit_rmse'] for v in hist],float)
                if not np.isfinite(values).all():raise AssertionError('Nonfinitehistory')
                best=int(np.argmin(values))+1
                if (s['epochs_run']!=100 or s['early_stopping'] is not False or s['optimizer']!='Adam'
                    or s['dropout']!=.1 or s['parameters']!=26953 or best!=s['best_epoch']):raise AssertionError('CNNpolicy/bestepochdiffers:'+name)
                if abs(s['validation_fit_metrics']['rmse']-values[best-1])>1e-10:raise AssertionError('Savedbestmetricdiffers')
                for i,row in enumerate(hist):
                    if row['best_epoch']!=int(np.argmin(values[:i+1]))+1 or abs(row['best_rmse']-float(values[:i+1].min()))>1e-12:
                        raise AssertionError('Runningbesthistorydiffers:'+name)
                    expected_lr=1e-5+(.001-1e-5)*(1+np.cos(np.pi*i/100))/2
                    if abs(row['learning_rate']-expected_lr)>1e-12:raise AssertionError('Fixed100cosinehistorydiffers')
                if sha(d/'best.pt')!=s['checkpoint_sha256']:raise AssertionError('CNNcheckpointSHAdiffers')
                ck=torch.load(d/'best.pt',map_location='cpu',weights_only=True);state=ck['state_dict']
                count=sum(int(v.numel()) for v in state.values())
                if count!=26953 or ck['parameters']!=26953 or ck['epoch']!=best or ck['arm']!=s['arm']:raise AssertionError('Checkpointtensorcountorbestmetadata')
                if not all(torch.isfinite(v).all() for v in state.values()):raise AssertionError('Nonfinitecheckpoint')
                models['CNN_'+name]={'epochs':100,'strict_first_min_epoch':best,'independent_state_tensor_count':count,'checkpoint_sha256':sha(d/'best.pt')}
            else:
                arm='L'+name[1]+'_H'+name[3];shape=name.endswith('1');b=s['training_binding']
                expected={'data_hash':m['data_hash'],'data_manifest_sha256':sha(local(p['data_dir'])/'manifest.json'),'arm':arm,'include_shape':shape,
                    'TRAIN_npz_sha256':m['arms'][arm]['train']['sha256'],'validation_fit_npz_sha256':m['arms'][arm]['validation_fit']['sha256'],
                    'basis_sha256':basis if shape else None,'basis_training_binding':m['basis_training_binding'] if shape else None}
                if s.get('fixture_only') is not False or any(b.get(k)!=v for k,v in expected.items()):raise AssertionError('Treebindingdiffers:'+name)
                if s['feature_count']!=(673 if shape else 466) or s['train_rows']!=12000 or s['validation_fit_rows']!=1000:raise AssertionError('Treecountsdiffer')
                if s['settings']['maximum_trees']!=1500 or s['settings']['early_stopping_patience']!=40 or s['settings']['threads']!=2:raise AssertionError('Treepolicychanged')
                for n,h in s['artifact_sha256'].items():
                    if Path(n).is_absolute() or '..' in Path(n).parts or sha(d/n)!=h:raise AssertionError('Changedtreeartifact')
                verify_map(s['source_code_sha256'])
                receipt=read(d/'FROZEN_BEFORE_HELDOUT.json')
                if receipt['summary_sha256']!=sha(d/'summary.json') or receipt['binding_sha256']!=token(b):raise AssertionError('Treefreezechanged')
                if shape and sha(d/'spectral_basis.npz')!=basis:raise AssertionError('TreebasisnotsharedpureTRAIN')
                models['Tree_'+name]={'feature_count':s['feature_count'],'regressor_best_round':s['best_iteration'],
                    'classifier_best_round':s['classifier_best_iteration'],'share_sha256':sha(d/'share.txt'),'classifier_sha256':sha(d/'presence.txt')}
            closure[str((d/'summary.json').relative_to(ROOT))]=sha(d/'summary.json')
    with np.load(cnn/'training_subset.npz',allow_pickle=False) as z:
        ix=z['row_indices']
        if len(ix)!=6000 or len(np.unique(ix))!=6000 or np.any((ix<0)|(ix>=12000)):raise AssertionError('FrozenCNNsubsetdiffers')
        if not np.array_equal(ix,np.load(local(p['data_dir'])/'cnn_subset_indices.npy',allow_pickle=False)):raise AssertionError('CNNsubsetnotdatasetfrozenindices')
    crc(local(p['basis_path']))
    with np.load(local(p['basis_path']),allow_pickle=False) as z:
        state=json.loads(z['metadata_json'].item());state['mean_db']=z['mean_db'].astype(float).tolist();state['components']=z['components'].astype(float).tolist()
    if (token({k:v for k,v in state.items() if k!='basis_sha256'})!=state['basis_sha256'] or state['fit_split']!='train'
        or state['source_kind']!='released_sa_background' or state['n_components']!=3
        or state['training_binding']!=m['basis_training_binding'] or state['unique_background_source_count']!=2000):raise AssertionError('SharedPCAprovenancechanged')
    return models,closure

def number(text):
    if text is None or str(text).strip()=='':return None
    return float(text)
def close(actual,expected,label,tolerance=2e-10):
    if actual is None or expected is None:
        if actual is not None or expected is not None:raise AssertionError('Undefinedmismatch:'+label)
        return 0.
    d=abs(float(actual)-float(expected))
    if d>tolerance:raise AssertionError(f'{label}: {actual} != {expected}; delta{d}')
    return d

def verify(args):
    import numpy as np
    resultpath=local(args.results);out=resultpath.parent;r=read(resultpath)
    if r.get('format')!='r12_full_ablation_evaluation_v1' or r.get('status')!='completed' or (out/'FAILED.json').exists():raise AssertionError('Completeunfailedevaluationrequired')
    if (r.get('all_models_frozen_before_gate') is not True or r.get('policy_frozen_before_locked_eval') is not True
        or r.get('development_not_blind') is not True or r.get('Test_or_training_or_threshold_weight_refit') is not False
        or r.get('performance_or_GPU_or_LB_improvement_guaranteed') is not False):raise AssertionError('Evaluationclaimboundarychanged')
    for name,h in r['output_sha256'].items():
        q=Path(name)
        if q.is_absolute() or '..' in q.parts or sha(out/q)!=h:raise AssertionError('Changedoutputartifact:'+name)
    gp=local(args.global_freeze);f=read(gp)
    if sha(gp)!=r['global_freeze_file_sha256'] or f.get('all_models_frozen')is not True or f.get('number_models')!=17 or len(f['model_summary_sha256'])!=17:raise AssertionError('Global17bindingdiffers')
    if token({k:v for k,v in f.items() if k!='freeze_sha256'})!=f['freeze_sha256']:raise AssertionError('GlobalcontentsSHAchanged')
    p=read(local(f['plan_file']))
    if (sha(local(f['plan_file']))!=f['plan_file_sha256'] or p['plan_sha256']!=f['plan_sha256']
        or token({k:v for k,v in p.items() if k!='plan_sha256'})!=p['plan_sha256']):raise AssertionError('Predeclaredplanchanged')
    verify_map(p['source_file_sha256']);verify_map(f['model_summary_sha256']);verify_map(f['artifact_sha256'])
    dm=read(local(p['data_dir'])/'manifest.json')
    if (f['data_hash']!=dm['data_hash'] or p['data_hash']!=dm['data_hash'] or f['data_manifest_sha256']!=sha(local(p['data_dir'])/'manifest.json')
        or f['basis_sha256']!=sha(local(p['basis_path']))):raise AssertionError('Globaldata/basisidentitydiffers')
    models,modelpins=verify_models(p,f)
    policy=read(out/'selected_policy.json')
    if (policy!=r['selected_policy'] or token({k:v for k,v in policy.items() if k!='policy_sha256'})!=policy['policy_sha256']
        or policy['global_freeze_sha256']!=f['freeze_sha256'] or policy['weights']!={'CNN':.5,'tree':.5}
        or policy['score_policy']!='share' or policy['weight_threshold_calibration_fitting'] is not False):raise AssertionError('Selectedpolicychanged')
    expectedpanels=['R12_validation_gate','R12_eval','R12_high_snr_long','R9_selection','R9_ensemble_gate']+['R10_'+n for n in R10NAMES]+['R11_'+a+'_VS_'+b for a in R11POS for b in R11NEG]
    npzpaths={v.stem:v for v in (out/'predictions').glob('*.npz')}
    if set(npzpaths)!=set(expectedpanels):raise AssertionError('Missingorextrapredictionpanels')
    with (out/r['results_csv']).open(newline='') as stream:rows=list(csv.DictReader(stream))
    rowmap={(v['panel'],v['model']):v for v in rows}
    if len(rowmap)!=len(rows) or len(rows)!=441:raise AssertionError('Expected441uniquemetricrows;gate25others26')
    allpred={};panels={};maxmetric=0.;maxblend=0.;scalarchecks=0;classifier_unverified=contrast_unverified=0;old_baseline_checks=0
    for name,path in npzpaths.items():
        crc(path)
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'sample_id','target_share','model_order','share','score'}:raise AssertionError('Savedpredictionfieldschanged')
            ids=z['sample_id'].astype(str);y=z['target_share'].astype(float);order=z['model_order'].astype(str).tolist();a=z['share'].copy();b=z['score'].copy()
        expected=list(MODELS)+([] if name=='R12_validation_gate' else ['R9_frozen_all5'])
        if order!=expected or a.shape!=b.shape or a.shape!=(len(y),len(order)) or len(set(ids))!=len(ids):raise AssertionError('PredictioncolumnorIDalignmentchanged')
        if not np.isfinite(np.r_[a.ravel(),b.ravel(),y]).all() or ((a<0)|(a>1)).any() or ((b<0)|(b>1)).any():raise AssertionError('Nonfinite/outofboundprediction')
        yy,sourceids,groups,key,sites=source_panel(name,p)
        if not np.array_equal(ids,sourceids) or not np.array_equal(y,yy.astype(float)):raise AssertionError('Predictionsdifferfromapprovedsourcetarget/IDorder:'+name)
        panels[name]=(y,groups,key);pred={model:(a[:,i],b[:,i]) for i,model in enumerate(order)};allpred[name]=pred
        for model,(share,score) in pred.items():
            actual=numeric_metrics(y,share,score,sites,mixed=model=='R9_frozen_all5');reported=rowmap.get((name,model))
            if reported is None:raise AssertionError('Missingmetricrow')
            if reported['development_not_blind']!='True':raise AssertionError('Panelmislabelledblind')
            expected_stage='gate_selection' if name=='R12_validation_gate' else 'post_policy_locked' if name in ('R12_eval','R12_high_snr_long') else 'exposed_development'
            if reported['stage']!=expected_stage:raise AssertionError('Panelstagechanged')
            for field,value in actual.items():
                maxmetric=max(maxmetric,close(number(reported[field]),value,name+model+field));scalarchecks+=1
            for field in ('auc_classifier','contrast_ge099_fraction'):
                x=number(reported[field])
                if x is not None and not 0<=x<=1:raise AssertionError('Invalidunreproduciblemetricrange')
                classifier_unverified+=int(field=='auc_classifier' and x is not None);contrast_unverified+=int(field=='contrast_ge099_fraction' and x is not None)
            if model.startswith('CNN_') and model!='CNN_legacy_contrast_control' and not np.array_equal(share,score):raise AssertionError('CandidateCNNscoremustbeshare')
        for arm in ARMS:
            s=(.5*pred['CNN_'+arm][0].astype(float)+.5*pred['Tree_'+arm][0].astype(float)).astype(np.float32)
            q=pred['Blend50_'+arm]
            if not np.array_equal(s,q[0]) or not np.array_equal(q[0],q[1]):raise AssertionError('Fixed50:50blenddiffers')
            maxblend=max(maxblend,float(abs(s-q[0]).max()))
        old=saved_R9_baseline(name,ids)
        if old is not None:
            if 'R9_frozen_all5' not in pred or any(not np.array_equal(old[i],pred['R9_frozen_all5'][i]) for i in (0,1)):raise AssertionError('ReusedR9baselinecolumn/orderchanged')
            old_baseline_checks+=1
    gate=allpred['R12_validation_gate'];gy=panels['R12_validation_gate'][0]
    scores={arm:float(np.sqrt(np.mean((gate['Blend50_'+arm][0].astype(float)-gy)**2))) for arm in ARMS}
    winner=min(ARMS,key=lambda v:scores[v])
    if winner!=policy['selected_arm'] or policy['baseline_arm']!='L0H0P0':raise AssertionError('Policyisnotstrictfirstgateargmin')
    for arm,v in scores.items():close(v,policy['gate_metrics'][arm],'gatepolicy'+arm,1e-12)
    if policy['selected_improves_gate_baseline']!=(scores[winner]<scores['L0H0P0']):raise AssertionError('Improvementflagwrong')
    with (out/r['paired_cluster_CI_csv']).open(newline='') as stream:cirows=list(csv.DictReader(stream))
    cmap={(v['panel'],v['comparison'],v['metric']):v for v in cirows}
    if len(cmap)!=len(cirows) or len(cirows)!=99:raise AssertionError('Expected99uniqueCIs')
    maxci=0.;cichecks=0
    for name,pred in allpred.items():
        y,groups,key=panels[name];selected=pred['Blend50_'+winner]
        for other,label in [('Blend50_L0H0P0','selected_minus_base_Blend50'),('R9_frozen_all5','selected_minus_R9_frozen_all5')]:
            if other not in pred:continue
            expected=cluster_ci(name,y,groups,key,selected,pred[other])
            for metric,v in expected.items():
                row=cmap[(name,label,metric)]
                for field,value in v.items():
                    if field=='cluster_key':
                        if (row[field] or None)!=value:raise AssertionError('CIclusterdefinitiondiffers')
                    else:maxci=max(maxci,close(number(row[field]),value,name+label+metric+field,2e-9));cichecks+=1
    report={'format':'r12_evaluation_independent_qa_v1','status':'passed','evaluation_results_sha256':sha(resultpath),
        'global_freeze_sha256':sha(gp),'selected_policy_file_sha256':sha(out/'selected_policy.json'),
        'prediction_panels':17,'metric_rows':441,'gate_models':25,'other_panel_models':26,
        'independent_scalar_checks':scalarchecks,'scalar_metric_max_abs_error':maxmetric,
        'AUC_oracle':'directpositive-negativepairswithexacttiehalfcredit;notproductionrankimplementation',
        'fixed50_50_blend_max_error':maxblend,'strict_first_gate_argmin':winner,'gate_rmse_by_arm':scores,
        'CI_rows_independently_checked':99,'CI_scalar_checks':cichecks,'CI_max_abs_error':maxci,
        'CI_oracle':'equivalentcluster-multiplicityweightedRMSEandweightedtieblockAUC;noexpandedrows',
        'models':models,'model_summary_sha256':modelpins,
        'old_R9_baseline_saved_output_replays':old_baseline_checks,
        'source_target_ID_metadata_order_and_all_output_SHA_verified':True,
        'coverage_limits':{'classifier_AUC_cells_not_exactly_recomputed':classifier_unverified,
            'contrast_saturation_cells_not_exactly_recomputed':contrast_unverified,
            'reason':'Onlyshare/scorearepersisted;rawclassifierandcontrastarraysarenotsaved. Boundsandprovenanceverified;no new inference used.',
            'neural_cell_pool_not_reexecuted':'Noq/x/warrayspersistedinevalNPZ;productionruntimecheckisprovenanceonly.'},
        'limitations':['Developmentpanelsandsharedreferences;notblindtestorindependentOOF.',
            'ConditionalclusterCIsdonotremovegateselectionoptimism,multiplecomparisonsorreferencedependence.',
            'Changedpresence-scoredefinitiondoesnotbyitselfprovelearningimprovement.',
            'NoofficialLBforecastorGPUimprovementguarantee.'],
        'new_model_fitting_inference_Test_GPU_SSH_external_actions':False,'verifier_sha256':sha(__file__)}
    return report

def fixture():
    import numpy as np
    rng=np.random.default_rng(1212);count=0
    for n in (6,17,31):
        y=np.r_[np.zeros(n//3),np.ones(n-n//3)];s=np.round(rng.uniform(0,1,n),1)
        for _ in range(12):
            w=rng.integers(0,4,n);ix=np.repeat(np.arange(n),w)
            if not len(ix):continue
            a=pair_auc(y[ix],s[ix]);b=weighted_auc_prepare(y,s)(w);close(a,b,'weightedpairwisefixture',1e-12);count+=1
    assert pair_auc(np.zeros(3),np.ones(3)) is None
    y=np.array([0,0,.1,.2,.3,.6]);s=np.array([.1,.1,.1,.3,.3,.7]);g=np.array(['a','a','b','b','c','c'])
    ci=cluster_ci('fixture',y,g,'group',(s,s),(s,s))
    if any(v['point_difference']!=0 or v['lo95']!=0 or v['hi95']!=0 for v in ci.values()):raise AssertionError('IdentityCIisnotzero')
    return {'status':'passed','numeric_only':True,'weighted_AUC_vs_expanded_pairwise_cases':count,'identity500cluster_CI_exact_zero':True,
        'actual_source_or_model_or_evaluation_files_opened':False,'verifier_sha256':sha(__file__)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',default='reports/r12_cpu_ablation_r1/RESULTS.json')
    p.add_argument('--global-freeze',default='reports/R12_ALL17_GLOBAL_FREEZE.json');p.add_argument('--output-prefix',default='reports/R12_EVALUATION_INDEPENDENT_QA')
    p.add_argument('--self-test',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--poll',type=int,default=30);p.add_argument('--timeout',type=int,default=7200)
    a=p.parse_args();sys.addaudithook(guard)
    if a.self_test:print(json.dumps(fixture()));return
    result=local(a.results);started=time.monotonic()
    def completed_result():
        if not result.exists():return False
        try:return read(result).get('status')=='completed'
        except (json.JSONDecodeError,OSError):return False
    # The evaluator writes RESULTS with exclusive creation rather than atomic
    # rename. Wait for a complete parse, not merely file existence.
    while not completed_result():
        if not a.watch:print(json.dumps({'status':'pending_completed_evaluation','fitting_or_inference':False}));raise SystemExit(3)
        if (result.parent/'FAILED.json').exists():raise RuntimeError('Evaluationfailed;preservepartialoutputs')
        if time.monotonic()-started>a.timeout:raise TimeoutError('Boundedwaitexpired;norestartornewinference')
        time.sleep(min(max(a.poll,1),60))
    prefix=local(a.output_prefix);jp=Path(str(prefix)+'.json');md=Path(str(prefix)+'.md')
    if jp.exists() or md.exists():raise FileExistsError('PreservepriorQA;freshprefixrequired')
    try:
        report=verify(a)
    except BaseException as e:
        jp.write_text(json.dumps({'status':'failed','error_type':type(e).__name__,'error':str(e),'no_refit_or_new_inference':True},indent=2)+'\n');raise
    jp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lim=report['coverage_limits'];text=['# R12 평가 독립 검산','',
        f"상태: PASS. 저장된 17개 패널·441개 모델별 행을 독립 재계산했습니다. Gate는 25모델, 나머지 16패널은 각각26모델입니다.",
        f"RMSE·MAE·bias·평균·RMS·share/score AUC·target-tail SSE·site macro를 {report['independent_scalar_checks']}회 검산했습니다. 최대 차이 {report['scalar_metric_max_abs_error']:.3g}.",
        f"직접 positive/negative 쌍과 tie 0.5를 세는 AUC oracle을 썼습니다. 99개 CI도 별도 weighted-cluster 알고리즘으로 재계산했고 최대 차이는 {report['CI_max_abs_error']:.3g}입니다.",
        f"선택 정책은 gate의 사전 고정 8개 50:50 recipe 중 첫 argmin({report['strict_first_gate_argmin']})과 일치합니다. 정책·17개 summary·체크포인트·PCA·데이터·예측 파일 SHA도 확인했습니다.",
        'CNN 9개는100epochs·Adam·dropout0.1·엄격한 첫 최저 epoch·26,953개 저장 tensor 원소를 확인했습니다. Tree8개는466/673특징·TRAIN12,000·1500/ES40·TRAIN-only basis binding을 확인했습니다.',
        '', '## 검증하지 못한 항목','',
        f"분류기 AUC {lim['classifier_AUC_cells_not_exactly_recomputed']}개와 contrast saturation {lim['contrast_saturation_cells_not_exactly_recomputed']}개는 원배열이 NPZ에 저장되지 않아 exact 재계산하지 못했습니다. 값의 범위와 원천 hash만 확인했습니다. 이 수치를 검증했다고 주장하지 않습니다.",
        'q/x/w가 평가 NPZ에 없으므로 cell-mask pooling을 다시 실행하지 않았습니다. 모델 추론·학습·Test 접근도 하지 않았습니다.',
        '', '## 해석 한계','',
        '패널과 참조는 개발 과정에서 이미 노출됐습니다. 독립 OOF나 blind Test가 아닙니다. CI는 gate 선택 낙관·여러 비교·참조 공유 의존성을 제거하지 않습니다. 점수 정의 변경과 실제 학습 개선을 구분해야 하며 공식 LB나 GPU 성능 향상을 보장하지 않습니다.',
        '', '[전체 검산·checkpoint SHA]('+jp.name+')']
    md.write_text('\n'.join(text)+'\n');print(json.dumps({'status':'passed','QA':str(jp),'sha256':sha(jp)}))

if __name__=='__main__':main()
