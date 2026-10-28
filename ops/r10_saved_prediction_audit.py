"""R10 saved-only Fold0 diagnostics. No inference, fit, policy edits or Test.

Selection NN predictions were produced with CUDA BF16 operators; fit/gate
NN predictions were produced on CPU FP32. This analysis keeps those scopes
separate. Known labels describe observed added-recording energy, not verified
natural-vessel presence. All slices are descriptive development diagnostics.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from pipeline.physics_features_r6_v2 import physics_features_r6_v2,feature_specification
from pipeline.relative_grid_r6_v2 import relative_grid_v2
from pipeline.trees_ensemble_r9 import array_sha

ROOT=Path(__file__).resolve().parents[1]
ROLES=('selection','ensemble_fit','ensemble_gate')
MEMBERS=('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer','lgbm_physics')
BUNDLE=ROOT/'data/synthetic/group_cv_minute_full_r9_v1'
PILOT=ROOT/'submissions/r9_fold0_all5_cpu_r1'
TREE=ROOT/'models/r9_mac_trees_r2/fold_00'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def auc(y,score,w=None):
    y=np.asarray(y,bool);s=np.asarray(score,float)
    if not len(y):return None
    w=np.ones(len(y)) if w is None else np.asarray(w,float)
    keep=np.isfinite(s)&np.isfinite(w)&(w>0);y,s,w=y[keep],s[keep],w[keep]
    if not len(y):return None
    pos=float(w[y].sum());neg=float(w[~y].sum())
    if pos==0 or neg==0:return None
    order=np.argsort(s,kind='stable');s,y,w=s[order],y[order],w[order]
    starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]
    pw=np.add.reduceat(w*y,starts);nw=np.add.reduceat(w*(~y),starts)
    return float(np.sum(pw*(np.cumsum(nw)-nw+.5*nw))/(pos*neg))


def align(ids,saved_ids):
    a=list(map(str,ids));b=list(map(str,saved_ids))
    if len(set(a))!=len(a) or len(set(b))!=len(b) or set(a)!=set(b):raise ValueError('Unique exact sample ID sets required')
    ix={v:i for i,v in enumerate(b)};return np.asarray([ix[v] for v in a])


def metrics(frame,p,s,c,mask=None,total_sse=None):
    sel=np.ones(len(frame),bool) if mask is None else np.asarray(mask,bool)
    f=frame.loc[sel];y=f.y.to_numpy();p,s,c=p[sel],s[sel],c[sel]
    explicit=f.explicit_no_source.to_numpy(bool);positive=y>0;contrast=explicit|positive
    if not len(f):return {'n':0,'RMSE':None,'MAE':None,'bias':None,'SSE':0.,'SSE_fraction_of_role':0.,'AUC_score':None,'AUC_share':None,'AUC_component':None,'AUC_score_explicit_neg_vs_positive':None,'AUC_share_explicit_neg_vs_positive':None,'AUC_component_explicit_neg_vs_positive':None}
    err=p-y;sse=float(err@err)
    return {'n':len(f),'positive':int(positive.sum()),'explicit_no_source':int(explicit.sum()),
            'mask_induced_zero':int((~explicit&~positive).sum()),'background_groups':int(f.background_group.nunique()),
            'y_mean':float(y.mean()),'prediction_mean':float(p.mean()),'RMSE':float(np.sqrt(np.mean(err**2))),
            'MAE':float(np.abs(err).mean()),'bias':float(err.mean()),'SSE':sse,
            'SSE_fraction_of_role':None if total_sse is None or total_sse==0 else sse/total_sse,
            'AUC_score':auc(positive,s),'AUC_share':auc(positive,p),'AUC_component':auc(positive,c),
            'AUC_score_explicit_neg_vs_positive':auc(positive[contrast],s[contrast]),
            'AUC_share_explicit_neg_vs_positive':auc(positive[contrast],p[contrast]),
            'AUC_component_explicit_neg_vs_positive':auc(positive[contrast],c[contrast])}


def bootstrap(frame,p,s,c,iterations,seed):
    groups,inv=np.unique(frame.background_group.astype(str),return_inverse=True);rng=np.random.default_rng(seed)
    y=frame.y.to_numpy(float);positive=y>0;ex=frame.explicit_no_source.to_numpy(bool)|positive
    vals={k:[] for k in ['RMSE','bias','AUC_score','AUC_share','AUC_component','AUC_score_explicit_neg_vs_positive','AUC_share_explicit_neg_vs_positive']}
    for _ in range(iterations):
        mult=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups));w=mult[inv].astype(float)
        vals['RMSE'].append(float(np.sqrt(np.average((p-y)**2,weights=w))))
        vals['bias'].append(float(np.average(p-y,weights=w)))
        for k,v in [('AUC_score',s),('AUC_share',p),('AUC_component',c)]:vals[k].append(auc(positive,v,w))
        vals['AUC_score_explicit_neg_vs_positive'].append(auc(positive[ex],s[ex],w[ex]))
        vals['AUC_share_explicit_neg_vs_positive'].append(auc(positive[ex],p[ex],w[ex]))
    out=[]
    for k,v in vals.items():
        valid=[a for a in v if a is not None]
        out.append({'metric':k,'iterations':iterations,'valid_replicates':len(valid),'source_groups':len(groups),
                    'ci_lower_025':None if not valid else float(np.quantile(valid,.025)),
                    'ci_upper_975':None if not valid else float(np.quantile(valid,.975))})
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='reports/r10_diagnostics/saved_predictions')
    parser.add_argument('--bootstrap',type=int,default=200)
    a=parser.parse_args();start=time.monotonic();out=ROOT/a.output
    if out.exists():raise FileExistsError('Fresh diagnostic output required')
    if not 0<=a.bootstrap<=200:raise ValueError('Predeclared bounded bootstrap0..200')
    out.mkdir(parents=True)
    nn={k:Path(v) for k,v in json.loads((ROOT/'ops/r9_fold0_pilot_nn_map.json').read_text()).items()}
    policy=json.loads((PILOT/'frozen_policy.json').read_text())
    if list(MEMBERS)!=policy['member_order'] or canonical({k:v for k,v in policy.items() if k!='policy_sha256'})!=policy['policy_sha256']:
        raise ValueError('Frozen policy canonical integrity failed')
    weights=np.asarray(policy['weights'],float)
    if weights.shape!=(5,) or (weights<=0).any() or abs(weights.sum()-1)>1e-12:raise ValueError('Frozen all5 weights invalid')
    inputs=[ROOT/'ops/r9_fold0_pilot_nn_map.json',PILOT/'frozen_policy.json',PILOT/'PREQUERY_READY.json',PILOT/'gate_comparison.json',
            PILOT/'fit_member_predictions.npz',PILOT/'gate_member_predictions.npz',BUNDLE/'manifest.json',
            ROOT/'data/versions/vtuad_full_r9/reference_manifest.parquet',TREE/'receipt.json',TREE/'models/lgbm_physics/summary.json',TREE/'models/lgbm_physics/inner_predictions.npz']
    for role in ROLES:
        inputs += [BUNDLE/f'fold_00/{role}.npz',BUNDLE/f'fold_00/{role}_metadata.parquet',BUNDLE/f'fold_00/{role}_recipes.json',
                   ROOT/f'data/versions/background_cv_r9/fold_00/{role}_manifest.parquet']
    for folder in nn.values():inputs += [folder/'summary.json',folder/'best_window_validation_predictions.npz']
    inputs += [ROOT/n for n in ['pipeline/models_r9.py','pipeline/relative_models_r6_v2.py','pipeline/trees_ensemble_r9.py','pipeline/physics_features_r6_v2.py','pipeline/relative_grid_r6.py','pipeline/relative_grid_r6_v2.py']]
    pins={str(p.resolve().relative_to(ROOT)):sha(p) for p in inputs}
    plan={'format':'r10_saved_prediction_audit_plan_v1','input_sha256':pins,'roles':list(ROLES),'members':list(MEMBERS)+['all5_frozen'],
          'frozen_weights':weights.tolist(),'bootstrap_background_group_replicates':a.bootstrap,
          'predeclared_Nobs_slices':['5','>=13','15'],'positive_bins':['(0,.03]','(.03,.08]','(.08,.2]','>.2'],
          'AUC_labels':'observed added-recording share>0; plus explicit-no-source negatives only contrast excluding mask-induced zeros',
          'feature_AUC_signs':['Nobs','-Nobs','HF_bump','-HF_bump','safe_bump','-safe_bump'],
          'bump_definition':feature_specification(),
          'component_recovery':'2*score-share; NN temporal q contrast / tree classifier; floating tolerance3e-6; raw and bounded values preserved',
          'selection_precision':'saved GPU BF16 neural operators with FP32 pooling; native CPU LightGBM',
          'fit_gate_precision':'saved CPU FP32 neural predictions and native CPU LightGBM',
          'no_new_predictions_fitting_policy_changes_Test_report_server_or_transmission':True,
          'source_script_sha256':sha(__file__)}
    write(out/'analysis_plan.json',plan)
    # Fail closed on accidental opening of excluded observation/prediction files.
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        name=Path(args[0]).name.lower()
        if name in {'test.csv','report.npz','report_metadata.parquet','report_recipes.json','query_member_predictions.npz','raw_test_predictions.npz','submission.csv'}:
            raise PermissionError('Excluded file opened by diagnostic: '+name)
    sys.addaudithook(guard)
    refs=pd.read_parquet(ROOT/'data/versions/vtuad_full_r9/reference_manifest.parquet').set_index('reference_row_index')
    bundle=json.loads((BUNDLE/'manifest.json').read_text())
    tree_summary=json.loads((TREE/'models/lgbm_physics/summary.json').read_text())
    gate_comparison=json.loads((PILOT/'gate_comparison.json').read_text())
    ready=json.loads((PILOT/'PREQUERY_READY.json').read_text())
    if sha(PILOT/'frozen_policy.json')!=ready['frozen_policy_file_sha256'] or sha(PILOT/'gate_comparison.json')!=ready['gate_comparison_sha256']:
        raise ValueError('Original policy/gate witness differs')
    overall=[];slices=[];contrasts=[];features=[];boots=[];component_checks=[];alignments=[];weak_auc=[];rows_saved=[]
    for role_index,role in enumerate(ROLES):
        path=BUNDLE/f'fold_00/{role}.npz'
        if sha(path)!=bundle['file_sha256'][f'fold_00/{role}.npz']:raise ValueError('Role source SHA differs')
        with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        meta=pd.read_parquet(BUNDLE/f'fold_00/{role}_metadata.parquet');recipes=json.loads((BUNDLE/f'fold_00/{role}_recipes.json').read_text())
        source=pd.read_parquet(ROOT/f'data/versions/background_cv_r9/fold_00/{role}_manifest.parquet')
        ids=data['sample_id'].astype(str);n=len(ids)
        if n!=(1000 if role=='selection' else 500) or len(recipes)!=n or not np.array_equal(ids,meta.sample_id.astype(str)):
            raise ValueError('Exact role counts/order differs')
        for k in ['base_window_index','reference_index','explicit_no_source','background_group','site']:
            if not np.array_equal(data[k],meta[k].to_numpy()):raise ValueError('NPZ/metadata differs: '+k)
        if [r['sample_id'] for r in recipes]!=ids.tolist():raise ValueError('Recipe ID order differs')
        frame=meta.merge(source[['base_window_index','recorder','sensor_group']],on='base_window_index',how='left',validate='one_to_one')
        if frame.recorder.isna().any():raise ValueError('Missing exact source metadata')
        y=data['share'].astype(float);q=data['cell_share'];mask=np.isfinite(data['db']);Nobs=mask.all(2).sum(1)
        if not np.array_equal(mask.all(2),mask.any(2)) or not np.array_equal(data['presence'],y>0):raise ValueError('Mask/target identity differs')
        if not np.isfinite(q).all() or (q[~mask]!=0).any() or (y[data['explicit_no_source']]!=0).any():raise ValueError('Known component identity differs')
        frame['y']=y;frame['Nobs']=Nobs;frame['mask_induced_zero']=(~data['explicit_no_source'])&(y==0)
        frame['route_actual']=[r['trajectory']['family'] for r in recipes]
        frame['native_minutes']=[int(r['trajectory']['native_minutes']) for r in recipes]
        frame['selected_minutes']=[int(r['trajectory'].get('selected_minutes',0)) for r in recipes]
        frame['geometric_overlap_minutes']=[int(r['trajectory'].get('geometric_overlap_minutes',0)) for r in recipes]
        frame['known_added_active_minutes']=(q>0).any(2).sum(1)
        frame['observed_support_coverage']=frame.known_added_active_minutes/Nobs
        frame['weak_class']=[str(refs.loc[int(i),'weak_class']) if i>=0 else '__no_reference__' for i in data['reference_index']]
        frame['reference_clip_id']=[str(refs.loc[int(i),'clip_id']) if i>=0 else '__no_reference__' for i in data['reference_index']]
        frame['weak_background_archive']=[bool(refs.loc[int(i),'is_weak_background_label_in_vessel_archive']) if i>=0 else False for i in data['reference_index']]
        frame['target_partition']=np.select([data['explicit_no_source'],frame.mask_induced_zero,y<=.03,y<=.08,y<=.2],
                                           ['explicit_no_source','mask_induced_zero','(0,.03]','(.03,.08]','(.08,.2]'],default='>.2')
        f=physics_features_r6_v2(data['db']);frame['HF_bump_db']=f[:,2];frame['safe_bump_db']=f[:,3]
        prediction={};cells={};exact_components={}
        if role=='selection':
            for kind in MEMBERS[:4]:
                summary=json.loads((nn[kind]/'summary.json').read_text());file=nn[kind]/'best_window_validation_predictions.npz'
                if sha(file)!=summary['artifact_sha256']['best_window_validation_predictions.npz']:raise ValueError('Saved neural prediction changed')
                with np.load(file,allow_pickle=False) as z:saved={k:z[k] for k in z.files}
                order=align(ids,saved['sample_id'])
                if not np.array_equal(saved['target_share'][order],data['share']) or not np.array_equal(saved['site'][order],data['site']):raise ValueError('Saved labels/order mismatch')
                prediction[kind]={'share':saved['share'][order].astype(float),'score':saved['score'][order].astype(float)}
                cells[kind]=saved['cell_mask'][order]
                region=cells[kind][:,:,10:23];observed=mask[:,:,10:23]
                floor=np.min(np.where(observed,region,np.inf),axis=1,keepdims=True)
                exact_components[kind]=np.max(np.where(observed,np.maximum(region-floor,0),0),axis=(1,2)).astype(float)
                if abs(float(np.sqrt(np.mean((prediction[kind]['share']-y)**2)))-summary['best_window_rmse'])>1e-12:
                    raise ValueError('Saved neural best RMSE mismatch')
            input_pin=tree_summary['input_sha256']['inner']
            for k,arr in [('db',data['db']),('share',y),('presence',data['presence'].astype(float))]:
                if array_sha(arr)!=input_pin[k]:raise ValueError('Tree original ordered input witness differs')
            file=TREE/'models/lgbm_physics/inner_predictions.npz'
            if sha(file)!=tree_summary['artifact_sha256']['inner_predictions.npz']:raise ValueError('Tree prediction changed')
            with np.load(file,allow_pickle=False) as z:
                prediction['lgbm_physics']={'share':z['share'].astype(float),'score':z['score'].astype(float)}
                exact_components['lgbm_physics']=z['score_classifier'].astype(float)
            alignments.append({'role':role,'neural':'exact IDs/target/site arrays; saved-summary SHA and RMSE agree',
                               'tree':'No saved IDs; original ordered db/y/z SHA + selection binding + artifact SHA witness'})
        else:
            file=PILOT/('fit_member_predictions.npz' if role=='ensemble_fit' else 'gate_member_predictions.npz')
            with np.load(file,allow_pickle=False) as z:
                order=align(ids,z['sample_id'])
                if z['share'].shape!=(n,5) or z['score'].shape!=(n,5):raise ValueError('Saved member matrix shape differs')
                for j,kind in enumerate(MEMBERS):prediction[kind]={'share':z['share'][order,j].astype(float),'score':z['score'][order,j].astype(float)}
            alignments.append({'role':role,'all_members':'exact unique IDs aligned; frozen matrix member order; role data SHA; frozen fit/gate scalar metrics checked'})
        for key in ['share','score']:
            prediction.setdefault('all5_frozen',{})[key]=(np.stack([prediction[k][key] for k in MEMBERS],1)@weights).astype(np.float32).astype(float)
        if role=='ensemble_fit' and abs(float(np.sqrt(np.mean((prediction['all5_frozen']['share']-y)**2)))-policy['candidate_fit_rmse'][policy['selected_name']])>1e-12:
            raise ValueError('Frozen fit policy witness mismatch')
        if role=='ensemble_gate':
            for key,name in [('all5_frozen','all5_frozen'),('lgbm_physics','tree_only')]:
                if abs(float(np.sqrt(np.mean((prediction[key]['share']-y)**2)))-gate_comparison['metrics'][name]['rmse'])>1e-12:raise ValueError('Gate saved metrics witness differs')
        if cells:
            _,w=relative_grid_v2(data['db']);decomp={'sample_id':ids}
            for kind,qhat in cells.items():
                contribution=qhat*w
                decomp[kind+'_minute_energy_contribution']=contribution.sum(2)
                decomp[kind+'_band_energy_contribution']=contribution.sum(1)
                decomp[kind+'_exact_temporal_contrast']=exact_components[kind]
                if np.max(np.abs(contribution.sum((1,2))-prediction[kind]['share']))>3e-6:raise ValueError('Stored q physical pool differs')
            np.savez_compressed(out/'selection_saved_NN_q_decomposition.npz',**decomp)
        for kind,pred in prediction.items():
            p,s=pred['share'],pred['score'];raw=2*s-p
            if not np.isfinite(p).all() or not np.isfinite(s).all() or ((p<0)|(p>1)|(s<0)|(s>1)).any():raise ValueError('Invalid saved prediction range')
            violation=float(max(0.,-raw.min(),raw.max()-1))
            if violation>3e-6:raise ValueError('Algebraic component exceeds numerical tolerance')
            c=np.clip(raw,0,1)
            check={'role':role,'model':kind,'formula':'2*score-share','range_violation':violation,'roundoff_clip_count':int((raw!=c).sum()),
                   'interpretation':'tree presence classifier' if kind=='lgbm_physics' else 'NN contrast' if kind!='all5_frozen' else 'frozen weighted NN contrasts plus tree classifier'}
            if kind in exact_components:
                check['direct_saved_component_max_abs_delta']=float(np.max(np.abs(raw-exact_components[kind])))
                if check['direct_saved_component_max_abs_delta']>3e-6:raise ValueError('Source formula/saved component recovery differs')
            component_checks.append(check)
            total=float(np.sum((p-y)**2));row={'role':role,'model':kind,**metrics(frame,p,s,c,total_sse=total)}
            row['AUC_component_unclipped_raw']=auc(y>0,raw);overall.append(row)
            frame[kind+'_pred_share']=p;frame[kind+'_score']=s;frame[kind+'_component_raw']=raw
            frame[kind+'_component_bounded_roundoff_only']=c;frame[kind+'_squared_error']=(p-y)**2
            dimensions=['site','recorder','sensor_group','route_actual','requested_family','weak_class','weak_background_archive',
                        'Nobs','native_minutes','selected_minutes','geometric_overlap_minutes','known_added_active_minutes','target_partition']
            for dim in dimensions:
                for value in sorted(frame[dim].unique(),key=str):
                    use=frame[dim].eq(value).to_numpy();slices.append({'role':role,'model':kind,'dimension':dim,'slice':str(value),**metrics(frame,p,s,c,use,total)})
            for label,use in [('5',Nobs==5),('>=13',Nobs>=13),('15',Nobs==15)]:
                slices.append({'role':role,'model':kind,'dimension':'requested_Nobs_overlapping','slice':label,**metrics(frame,p,s,c,use,total)})
            for label,use in [('(0,.03]',(y>0)&(y<=.03)),('(.03,.08]',(y>.03)&(y<=.08)),('(.08,.2]',(y>.08)&(y<=.2)),('>.2',y>.2),('all_positive',y>0)]:
                sel=data['explicit_no_source']|use
                for obs_label,obs in [('all',np.ones(n,bool)),('5',Nobs==5),('15',Nobs==15)]:
                    ix=sel&obs
                    contrasts.append({'role':role,'model':kind,'positive_bin':label,'Nobs_slice':obs_label,
                                      'n_positive':int((use&obs).sum()),'n_explicit_negative':int((data['explicit_no_source']&obs).sum()),
                                      'AUC_score':auc((y>0)[ix],s[ix]),'AUC_share':auc((y>0)[ix],p[ix]),'AUC_component':auc((y>0)[ix],c[ix])})
            for klass in sorted(frame.weak_class.unique()):
                if klass=='__no_reference__':continue
                use=(frame.weak_class==klass).to_numpy()&(y>0);sel=data['explicit_no_source']|use
                weak_auc.append({'role':role,'model':kind,'weak_class':klass,'n_observed_positive':int(use.sum()),
                                 'n_explicit_negative':int(data['explicit_no_source'].sum()),'AUC_score':auc((y>0)[sel],s[sel]),
                                 'AUC_share':auc((y>0)[sel],p[sel]),'AUC_component':auc((y>0)[sel],c[sel])})
            for b in bootstrap(frame,p,s,c,a.bootstrap,20261026+role_index*10007):boots.append({'role':role,'model':kind,**b})
        for label,select in [('all',np.ones(n,bool)),('5',Nobs==5),('15',Nobs==15),('>=13',Nobs>=13)]:
            for feature,values in [('Nobs',Nobs.astype(float)),('-Nobs',-Nobs.astype(float)),('HF_bump',frame.HF_bump_db.to_numpy()),('-HF_bump',-frame.HF_bump_db.to_numpy()),('safe_bump',frame.safe_bump_db.to_numpy()),('-safe_bump',-frame.safe_bump_db.to_numpy())]:
                explicit=select&(data['explicit_no_source']|(y>0))
                features.append({'role':role,'feature':feature,'Nobs_slice':label,'n':int(select.sum()),
                                 'AUC_all_observed_labels':auc((y>0)[select],values[select]),
                                 'AUC_explicit_neg_vs_positive':auc((y>0)[explicit],values[explicit])})
        if 'role' in frame:
            if not frame.role.eq(role).all():raise ValueError('Metadata role binding differs')
        else:frame.insert(0,'role',role)
        rows_saved.append(frame)
        print(json.dumps({'phase':'role_diagnostics_complete','role':role,'n':n,'groups':int(frame.background_group.nunique()),
                          'explicit_negative':int(data['explicit_no_source'].sum()),'mask_induced_zero':int(frame.mask_induced_zero.sum())}),flush=True)
    # Block-bootstrap intervals are conditional on this fixed generator and
    # shared reference pool. They do not account for reference independence.
    tables={'overall_metrics':overall,'slice_metrics':slices,'positive_vs_explicit_zero_auc':contrasts,
            'feature_only_auc':features,'weak_class_positive_vs_explicit_zero_auc':weak_auc,
            'background_group_bootstrap_CI':boots,'component_recovery_checks':component_checks}
    for name,records in tables.items():pd.DataFrame(records).to_csv(out/(name+'.csv'),index=False)
    pd.concat(rows_saved,ignore_index=True).to_csv(out/'aligned_rows_and_saved_prediction_decomposition.csv',index=False)
    if any(sha(ROOT/name)!=value for name,value in pins.items()):raise ValueError('Input bytes changed during diagnostics')
    result={'status':'passed','analysis_plan_sha256':sha(out/'analysis_plan.json'),'source_script_sha256':sha(__file__),
            'input_sha256':pins,'alignment_checks':alignments,'overall':overall,'feature_only_auc':features,
            'component_recovery_checks':component_checks,'frozen_weights':weights.tolist(),
            'bootstrap_unit':'background_group with replacement; all rows of drawn groups, 200 draws; fixed shared references not resampled',
            'bootstrap_iterations':a.bootstrap,'source_group_counts':{'selection':25,'ensemble_fit':10,'ensemble_gate':8},
            'precision_warning':'selection saved GPU BF16 NNs vs fit/gate CPU FP32; not a controlled precision comparison',
            'hard_negative_status':'No exact hard-negative flag in these three roles; TRAIN-only flags not joined; natural hard-negative500 panel not generated here',
            'cell_q_decomposition':'selection four neural members only; fit/gate q and tree q were not saved and no new inference was run',
            'missing_class_auc':None,'Nobs_slices_13plus_and15_overlap':True,
            'SSE_fraction_denominator':'same role/member total SSE; do not sum across overlapping slices',
            'causal_conclusion':'None. AUC change or feature association alone does not establish overfitting or competition failure cause.',
            'no_new_model_predictions_training_policy_or_data_change_Test_report_or_server_access':True,
            'reference_CV':'all509 shared; whole recording energy label, not certified isolated vessel or natural-absence truth',
            'elapsed_seconds':time.monotonic()-start}
    result['output_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    write(out/'summary.json',result)
    print(json.dumps({'status':'passed','report':str(out/'summary.json'),'sha256':sha(out/'summary.json'),'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
