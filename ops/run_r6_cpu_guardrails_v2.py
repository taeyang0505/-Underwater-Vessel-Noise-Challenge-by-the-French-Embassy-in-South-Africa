"""Predeclared Mac CPU pilot; no old OUTER, Test, SSH or GPU I/O.

New locked diagnostic arrays may be materialized for arithmetic QA, but they
are not used for fitting, early stopping, prior fitting or model selection.
Every requested arm is frozen before any locked model predictions are made.
"""
from pathlib import Path
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from pipeline.cpu_ablation_r6_v2 import sha,save_json,fit_cnn,fit_tree,evaluate_frozen,metrics

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/r6_cpu_guardrails_v2.json'
SOURCE=ROOT/'data/versions/source_split_r6_guardrails_v2'
HARD=ROOT/'data/versions/hard_negatives_r6_v2'
DATA=ROOT/'data/synthetic/r6_cpu_guardrails_v3_schemafix'
RUN=ROOT/'models/r6_cpu_guardrails_v2'
CODE=['pipeline/relative_grid_r6.py','pipeline/relative_grid_r6_v2.py','pipeline/relative_models_r6_v2.py','pipeline/physics_features_r6_v2.py','pipeline/hard_negatives_r6_v2.py',
      'pipeline/robust_sampler_r6.py','pipeline/source_split_r6.py','pipeline/cpu_ablation_r6_v2.py','ops/run_r6_cpu_guardrails_v2.py',
      'pipeline/passage_sampler_r4.py','pipeline/mixtures.py','pipeline/background_prior_features.py']


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def describe(d,*,allow_feature_diagnostic=False):
    from sklearn.metrics import roc_auc_score
    from pipeline.relative_grid_r6 import relative_grid,quantization_pooling_bound
    y=d['share'].astype(float);q=d['cell_share'].astype(float);x=d['unrounded_db'].astype(float);o=np.isfinite(x)
    peak=np.nanmax(x,(1,2),keepdims=True);e=np.where(o,10**((x-peak)/10),0);raw_w=e/e.sum((1,2),keepdims=True)
    _,w=relative_grid(d['db']);raw_error=np.abs((q*raw_w).sum((1,2))-y);rounded_error=np.abs((q*w).sum((1,2))-y)
    high=e[:,:,18:].sum((1,2))/e.sum((1,2));z=d['presence'];sites=d['site'].astype(str)
    report={'examples':len(y),'mean_share':float(y.mean()),'RMS_share':float(np.sqrt(np.mean(y*y))),'zero_fraction':float((y==0).mean()),
            'share_quantiles':dict(zip(['min','q10','q25','q50','q75','q90','max'],np.quantile(y,[0,.1,.25,.5,.75,.9,1]).tolist())),
            'constant0_RMSE':float(np.sqrt(np.mean(y*y))),'constant0_1_RMSE':float(np.sqrt(np.mean((y-.1)**2))),
            'high_band_fraction_presence_AUC':float(roc_auc_score(z,high)) if allow_feature_diagnostic and len(np.unique(z))==2 else None,
            'feature_diagnostic_deferred_until_all_checkpoints_frozen':not allow_feature_diagnostic,
            'original_target_identity_max_error':float(raw_error.max()),'rounded_weight_target_gap_max':float(rounded_error.max()),
            'rounded_weight_target_gap_mean':float(rounded_error.mean()),'rounding_bound_plus_float32_tolerance':quantization_pooling_bound()+3e-6,
            'site_counts':{s:int((sites==s).sum()) for s in sorted(set(sites))},
            'primary_unique_backgrounds':len(np.unique(d['base_window_index']))}
    report['no_source_count']=int(np.asarray(d['no_source']).sum())
    report['hard_negative_count']=int(np.asarray(d['hard_negative']).sum())
    report['hard_negative_unique_backgrounds']=len(np.unique(d['base_window_index'][d['hard_negative']]))
    report['source_present_but_unobserved_count']=int(d['source_present_but_unobserved'].sum())
    obs=d['db'][np.isfinite(d['db'])].astype(float)
    report['quantization_grid_max_residual_db']=float(np.abs(obs-np.round(obs,1)).max())
    if allow_feature_diagnostic:
        from pipeline.physics_features_r6_v2 import physics_features_r6_v2,FEATURE_NAMES
        f=physics_features_r6_v2(d['db'])
        report['own_source_physics_summary']={s:{FEATURE_NAMES[i]:np.quantile(f[sites==s,i],[.1,.5,.9]).tolist() for i in range(9)} for s in sorted(set(sites))}
    return report


def materialize_r6(loader,role,n,seed,families,output,*,indices=None,explicit_zero=None,hard_negative=None):
    from pipeline.robust_sampler_r6 import synthesize
    cfg=json.loads(CONFIG.read_text());g=cfg['new_generator']
    if indices is None:indices=loader.pick_indices(role,n,seed,site_balanced=role=='train',allow_diagnostic=role not in ['train','inner'])
    bg,md=loader.load_backgrounds(role,indices,allow_diagnostic=role not in ['train','inner'])
    refs=loader.load_references(role,allow_diagnostic=role not in ['train','inner'])
    rng=np.random.default_rng(seed);zero=np.zeros(n,bool);zero[rng.permutation(n)[:round(cfg['zero_fraction']*n)]]=True
    if explicit_zero is not None:zero=np.asarray(explicit_zero,dtype=bool).copy()
    hard=np.zeros(n,bool) if hard_negative is None else np.asarray(hard_negative,dtype=bool)
    if zero.shape!=(n,) or hard.shape!=(n,) or np.any(hard&~zero):raise ValueError('Invalid prescribed no-source schedule')
    # Group-balanced reference draws, never pick by target or measured outcome.
    by_group={}
    for m,s in refs:by_group.setdefault(str(m['reference_component']),[]).append((m,s))
    groups=sorted(by_group);values=[];recipes=[];ref_keys=[];family_keys=[]
    for i in range(n):
        r=np.random.default_rng(np.random.SeedSequence([seed,i,0x5236]));family=families[int(r.integers(len(families)))];snr=float(r.uniform(*g['snr_db_uniform']))
        group=groups[int(r.integers(len(groups)))];m,s=by_group[group][int(r.integers(len(by_group[group])))];ref_keys.append('zero' if zero[i] else group)
        item=synthesize(bg[i],None if zero[i] else s,r,family=family,eq_max_db=g['reference_smooth_eq_max_db'],snr_db=snr,
                        zero=bool(zero[i]),lowpass_probability=g['lowpass_probability'])
        values.append(item);family_keys.append('zero' if zero[i] else family)
        recipes.append({'row':i,'base_window_index':int(indices[i]),'role':role,'reference_clip_id':None if zero[i] else str(m['clip_id']),
                        'reference_group':ref_keys[-1],'explicit_hard_negative':bool(hard[i]),**item['recipe']})
    d={k:np.stack([v[k] for v in values]) for k in ['db','unrounded_db','share','presence','cell_share']}
    d['db']=d['db'].astype('float32');d['cell_share']=d['cell_share'].astype('float32')
    d.update(site=md.site.to_numpy(dtype=str),base_window_index=np.asarray(indices),reference_group=np.asarray(ref_keys),family=np.asarray(family_keys),
             no_source=zero.copy(),hard_negative=hard.copy(),source_present_but_unobserved=(~zero)&(d['presence']==0))
    np.savez_compressed(output,**d);save_json(output.with_suffix('.recipes.json'),recipes)
    return d,np.asarray(indices)


def materialize_r4(loader,role,n,seed,indices,output,*,explicit_zero=None,hard_negative=None):
    from pipeline.passage_sampler_r4 import PassageSamplerR4,PassageConfigR4,PassageReference,make_passage_schedule
    bg,md=loader.load_backgrounds(role,indices)
    # Only this allowed role's materialized backgrounds are exposed to the old
    # arithmetic sampler; local indices do not recover any forbidden bank row.
    b=pd.DataFrame({'base_window_index':np.arange(len(bg)),'split':['train']*len(bg),'background_group':md.background_group.astype(str).to_numpy()})
    refs=[]
    for m,s in loader.load_references(role):
        m=dict(m);m['split']='train';refs.append(PassageReference(m,s))
    sampler=PassageSamplerR4(bg,b,refs,'train',seed,PassageConfigR4());schedule=make_passage_schedule(n,seed+17,PassageConfigR4())
    if explicit_zero is not None:
        zero=np.asarray(explicit_zero,dtype=bool)
        positive=schedule['requested_share']>0
        if zero.shape!=(n,) or zero.sum()!=schedule['zero_count']:raise ValueError('R4 no-source count mismatch')
        # Keep exactly the old positive LHS share/duration pairs, but bind the
        # no-source positions to the same TRAIN hard-negative control rows.
        target=np.zeros(n);duration=np.zeros(n,dtype=np.int64)
        target[~zero]=schedule['requested_share'][positive];duration[~zero]=schedule['rendered_segments'][positive]
        schedule={**schedule,'requested_share':target,'rendered_segments':duration,
                  'scheme':schedule['scheme']+'; positive pairs remapped to prespecified shared no-source rows'}
    db,y,z,q,recipe=sampler.draw(n,np.arange(n),schedule)
    # Preserve the old sampler's true labels; model input and w are rounded.
    d={'db':np.round(db.astype('float64'),1).astype('float32'),'unrounded_db':db.astype('float64'),'share':y.astype('float64'),
       'presence':z,'cell_share':q,'site':md.site.to_numpy(dtype=str),'base_window_index':np.asarray(indices),
       'reference_group':np.asarray(['zero' if not v['references'] else v['references'][0]['reference_component'] for v in recipe['items']]),
       'family':np.asarray(['zero' if t==0 else 'r4_whole_timewarp' for t in y]),'no_source':z==0,
       'hard_negative':np.zeros(n,bool) if hard_negative is None else np.asarray(hard_negative,bool),
       'source_present_but_unobserved':np.zeros(n,bool)}
    if np.any(d['hard_negative']&(d['presence']!=0)):raise ValueError('Hard negative target changed')
    np.savez_compressed(output,**d);save_json(output.with_suffix('.recipes.json'),recipe)
    return d


def prepare():
    from pipeline.source_split_r6 import SourceSplitR6
    if DATA.exists():raise FileExistsError('Fresh dataset output required')
    DATA.mkdir(parents=True);cfg=json.loads(CONFIG.read_text());seed=cfg['seed'];loader=SourceSplitR6(SOURCE)
    freeze={'config_sha256':sha(CONFIG),'code_sha256':{n:sha(ROOT/n) for n in CODE}}
    save_json(DATA/'pre_generation_freeze.json',freeze)
    records={}
    for ix,role in enumerate(['train','inner']):
        n=cfg['samples'][role];kwargs={};draw=None
        if role=='train':
            from pipeline.hard_negatives_r6_v2 import draw_training_indices
            draw=draw_training_indices(HARD,n,seed)
            kwargs={k:draw[k] for k in ['indices','explicit_zero','hard_negative']}
            save_json(DATA/'train_source_draw_audit.json',draw['draw_audit'])
        d,ids=materialize_r6(loader,role,n,seed+ix*1009,cfg['new_generator']['training_families'],DATA/f'{role}_r6.npz',**kwargs)
        records[role+'_r6']=describe(d,allow_feature_diagnostic=True);print(json.dumps({'materialized':role+'_r6','n':n}),flush=True)
        old=materialize_r4(loader,role,n,seed+ix*1009,ids,DATA/f'{role}_r4.npz',
                           **({k:draw[k] for k in ['explicit_zero','hard_negative']} if draw is not None else {}))
        records[role+'_r4']=describe(old,allow_feature_diagnostic=True)
    cs_indices=None
    for ix,role in enumerate(['cross_generator','cross_sensor','cross_sensor_generator']):
        source_role='cross_sensor' if role=='cross_sensor_generator' else role
        families=cfg['new_generator']['cross_generator_families'] if role!='cross_sensor' else cfg['new_generator']['training_families']
        # Sensor-only and joint panels share primary source draws and row seeds.
        rs=seed+3027 if source_role=='cross_sensor' else seed+2018
        d,ids=materialize_r6(loader,source_role,cfg['samples'][role],rs,families,DATA/f'{role}.npz',indices=cs_indices if role=='cross_sensor_generator' else None)
        if role=='cross_sensor':cs_indices=ids
        records[role]=describe(d);print(json.dumps({'materialized':role,'n':len(d['share'])}),flush=True)
    for name,r in records.items():
        if r['original_target_identity_max_error']>3e-6:raise ValueError('Original label arithmetic failed: '+name)
        if r['rounded_weight_target_gap_max']>r['rounding_bound_plus_float32_tolerance']:raise ValueError('Quantization bound failed: '+name)
    # Every serialized array must load safely without pickle, including site
    # metadata. A pandas string column can otherwise serialize as object.
    for path in DATA.glob('*.npz'):
        restored=load_npz(path)
        if any(value.dtype.hasobject for value in restored.values()):raise TypeError('Object NPZ field: '+path.name)
    if sha(CONFIG)!=freeze['config_sha256'] or any(sha(ROOT/n)!=h for n,h in freeze['code_sha256'].items()):raise ValueError('Code/config changed during generation')
    manifest={'status':'ready','scope':'new CPU pilot only, no Test/cohort/LB target fitting','freeze':freeze,'source_version':str(SOURCE.relative_to(ROOT)),
              'source_manifest_sha256':sha(SOURCE/'manifest.json'),'arrays':{p.name:sha(p) for p in DATA.glob('*.npz')},'diagnostics':records,'matches_official_hidden_generator':False}
    manifest['hard_negative_manifest_sha256']=sha(HARD/'manifest.json')
    manifest['metadata_sha256']={p.name:sha(p) for p in DATA.glob('*.json') if p.name!='manifest.json'}
    save_json(DATA/'manifest.json',manifest);save_json(ROOT/'reports/R6V2_DATASET_AUDIT.json',manifest)
    print(json.dumps({'status':'prepared','source':str(SOURCE),'arrays':list(manifest['arrays'])}),flush=True)


def check_freeze(manifest):
    f=manifest['freeze']
    if sha(CONFIG)!=f['config_sha256'] or any(sha(ROOT/n)!=h for n,h in f['code_sha256'].items()):raise ValueError('Frozen source/config changed')
    if sha(SOURCE/'manifest.json')!=manifest['source_manifest_sha256']:raise ValueError('Source split changed')
    if sha(HARD/'manifest.json')!=manifest['hard_negative_manifest_sha256']:raise ValueError('Hard-negative version changed')
    for name,h in manifest['arrays'].items():
        if sha(DATA/name)!=h:raise ValueError('Frozen dataset changed: '+name)
    for name,h in manifest['metadata_sha256'].items():
        if sha(DATA/name)!=h:raise ValueError('Frozen metadata changed: '+name)


def fit_all():
    cfg=json.loads(CONFIG.read_text());manifest=json.loads((DATA/'manifest.json').read_text());check_freeze(manifest)
    if RUN.exists():raise FileExistsError('Fresh run required; never resume a failed partial pilot')
    s=cfg['neural']
    if (s['epochs'],s['early_stopping_patience'],s['optimizer'])!=(100,7,'adam'):raise ValueError('User training contract changed')
    RUN.mkdir(parents=True);save_json(RUN/'freeze.json',manifest['freeze'])
    done=[]
    try:
        arms=cfg['CNN_arms']+cfg['LightGBM_arms']
        for index,arm in enumerate(arms):
            save_json(RUN/'progress.json',{'phase':'fitting','arm':arm['name'],'completed_arms':len(done),'heldout_model_predictions_executed':False})
            # Fresh exec, never fork an initialized OpenMP/Torch runtime.
            with (RUN/(arm['name']+'.driver.log')).open('x') as log:
                subprocess.run([sys.executable,'-u','-m','ops.run_r6_cpu_guardrails_v2','fit-arm','--index',str(index)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            summary=json.loads((RUN/arm['name']/'summary.json').read_text());done.append(summary);check_freeze(manifest)
            print(json.dumps({'completed':arm['name'],'inner_rmse':summary['best_inner_metrics']['rmse'],'macro':summary['best_inner_metrics']['site_macro_rmse']}),flush=True)
        save_json(RUN/'all_checkpoints_frozen_before_heldout.json',{'status':'frozen','model_summaries':{s['arm']['name']:sha(RUN/s['arm']['name']/'summary.json') for s in done},
                   'all_requested_arms_completed':len(done)==len(arms),'no_heldout_predictions_yet':True})
        results={};heldout_data_diagnostics={}
        for role in ['cross_generator','cross_sensor','cross_sensor_generator']:
            data=load_npz(DATA/f'{role}.npz');results[role]={}
            for index,arm in enumerate(arms):
                with (RUN/(arm['name']+'.'+role+'.eval.log')).open('x') as log:
                    subprocess.run([sys.executable,'-u','-m','ops.run_r6_cpu_guardrails_v2','eval-arm','--index',str(index),'--role',role],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                m=json.loads((RUN/arm['name']/f'{role}_metrics.json').read_text());results[role][arm['name']]=m
                print(json.dumps({'evaluated':role,'arm':arm['name'],'rmse':m['rmse'],'site_macro_rmse':m['site_macro_rmse'],'auc':m['presence_auc']}),flush=True)
            # Honest no-information controls, fit constants from permitted TRAIN only.
            tr=load_npz(DATA/'train_r6.npz');mean=float(tr['share'].mean());del tr
            old=load_npz(DATA/'train_r4.npz');old_mean=float(old['share'].mean());del old
            for label,value in [('constant0',0.),('constant0_1',.1),('TRAIN_r6_mean',mean),('TRAIN_r4_mean',old_mean)]:
                results[role][label]=metrics(data,{'share':np.full(len(data['share']),value),'score':np.full(len(data['share']),.5)})
            heldout_data_diagnostics[role]=describe(data,allow_feature_diagnostic=True)
            del data;gc.collect()
        check_freeze(manifest)
        final={'status':'completed','scope':'predeclared CPU pilot, not full 5-fold or official evaluation','config_sha256':sha(CONFIG),'data_manifest_sha256':sha(DATA/'manifest.json'),
               'selection':'own source INNER site-macro only; frozen before new locked diagnostics','summaries':done,'heldout_results':results,'heldout_data_diagnostics':heldout_data_diagnostics,
               'original_frozen_R5_unchanged':True,'official_Test_OUTER_GPU_SSH_or_submission_used':False,'no_Benchmark_moment_matching':True,'no_heldout_based_retries':True}
        save_json(RUN/'results.json',final);save_json(ROOT/'reports/R6V2_CPU_ABLATION_RESULTS.json',final);save_json(RUN/'progress.json',{'phase':'completed','report':'reports/R6V2_CPU_ABLATION_RESULTS.json'})
    except BaseException as e:
        save_json(RUN/'FAILED.json',{'status':'failed','error':str(e),'type':type(e).__name__,'partial_reusable':False});raise


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fit','fit-arm','eval-arm']);p.add_argument('--index',type=int);p.add_argument('--role',choices=['cross_generator','cross_sensor','cross_sensor_generator']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='fit':fit_all()
    else:
        cfg=json.loads(CONFIG.read_text());manifest=json.loads((DATA/'manifest.json').read_text());check_freeze(manifest)
        arms=cfg['CNN_arms']+cfg['LightGBM_arms']
        if a.index is None or not 0<=a.index<len(arms):raise ValueError('Explicit valid arm index required')
        arm=arms[a.index];kind='cnn' if a.index<len(cfg['CNN_arms']) else 'tree'
        if a.action=='fit-arm':
            if (RUN/'all_checkpoints_frozen_before_heldout.json').exists():raise ValueError('Cannot fit after heldout boundary')
            train=load_npz(DATA/f"train_{arm['generator']}.npz");inner=load_npz(DATA/f"inner_{arm['generator']}.npz")
            def progress(row):save_json(RUN/'progress.json',{'phase':'fitting','latest':row,'heldout_model_predictions_executed':False})
            (fit_cnn if kind=='cnn' else fit_tree)(train,inner,arm,cfg,RUN/arm['name'],progress=progress)
        else:
            gate=RUN/'all_checkpoints_frozen_before_heldout.json'
            if a.role is None or not gate.exists():raise ValueError('All checkpoints must be frozen before heldout predictions')
            frozen=json.loads(gate.read_text())
            if set(frozen['model_summaries'])!={v['name'] for v in arms}:raise ValueError('Incomplete arm freeze gate')
            if not frozen['all_requested_arms_completed'] or any(sha(RUN/n/'summary.json')!=h for n,h in frozen['model_summaries'].items()):raise ValueError('Frozen summaries changed')
            data=load_npz(DATA/f'{a.role}.npz');m,pred=evaluate_frozen(RUN/arm['name'],data,kind)
            out=RUN/arm['name']/f'{a.role}_predictions.npz'
            if out.exists():raise FileExistsError('No repeated heldout predictions')
            np.savez_compressed(out,**pred);save_json(RUN/arm['name']/f'{a.role}_metrics.json',m)


if __name__=='__main__':main()
