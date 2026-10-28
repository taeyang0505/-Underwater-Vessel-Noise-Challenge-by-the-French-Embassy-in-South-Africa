"""Prepare native R13 full5fold data/bases; never train NN/tree or open Test.

--audit-only reports real TRAIN/source/HF capacity without materialising fixed
roles. Preparation is fresh and records provisional recipe status separately
from actual source/physics readiness. GPU readiness is never inferred here.
"""
from pathlib import Path
import argparse,json,sys,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from pipeline.full_data_r13 import (DATA_CODE,VERSION,ROLES,R13Sources,read_config,describe_backgrounds,
    train_HF_thresholds,apply_HF_thresholds,make_full_schedule,choose_refs,_basis_metadata,
    generate_item,write,sha,digest,array_sha,fixed_background_indices,
    fit_spectral_basis,save_spectral_basis)


def source_audit(sources,fold,train):
    roles={};seen={};minimum=1e99
    for role in ('train',)+ROLES:
        f=sources.original.role_metadata(fold,role);seen[role]=f
        minimum=min(minimum,float(f.minimum_other_role_support_gap_hours.min()))
        roles[role]={'windows':len(f),'background_groups':int(f.background_group.nunique()),
            'sites':{str(k):int(v) for k,v in f.site.value_counts().items()}}
    for i,a in enumerate(seen):
        for b in list(seen)[i+1:]:
            if set(seen[a].base_window_index)&set(seen[b].base_window_index) or set(seen[a].background_group)&set(seen[b].background_group):raise ValueError('Original R9 roles overlap')
    if minimum<24:raise ValueError('Existing900s+24h source purge failed')
    strata=[]
    for key,g in train.groupby('stratum_key',sort=True):
        strata.append({'key':key,'source_rows':len(g),'HF_candidates':int(g.HF_candidate.sum()),
            'TRAIN_q90_db':float(g.TRAIN_HF_q90_db.iloc[0])})
    schedules=[]
    for epoch in range(1,6):
        s=make_full_schedule(train,epoch,fold,sources.config['seed']);a=s['audit']
        schedules.append({k:v for k,v in a.items() if k!='quota_rows'})
    if sum(a['zero_count'] for a in schedules)!=len(train):raise AssertionError('Exact20pct five-epoch arithmetic failed')
    return {'fold':fold,'roles':roles,'minimum_original_cross_role_support_gap_hours':minimum,
        'source_role_ID_and_group_overlaps':0,'TRAIN_strata':strata,
        'HF_candidate_total':int(train.HF_candidate.sum()),'first5epoch_actual_schedules':schedules,
        'exact_zero20percent_over5and100epochs':True,'all_primary_once_each_epoch':True,
        'per_epoch_exact20percent_impossible_when_N_not_divisible5':len(train)%5!=0,
        'HF_assignment_not_equivalent_to_12k_outside_primary_replacement':True,
        'reference_independent_validation':False}


def prepare(config,output=None,audit_only=False):
    cfg=Path(config).resolve();c=read_config(cfg)
    source=R13Sources(cfg,bundle=output or ROOT/c['data_output']);out=Path(output or ROOT/c['data_output']).resolve()
    if out.exists():raise FileExistsError('Fresh R13 fullscale preparation required')
    out.mkdir(parents=True)
    code={n:sha(ROOT/n) for n in DATA_CODE if (ROOT/n).exists()}
    if not audit_only and set(code)!=set(DATA_CODE):raise ValueError('R13 source/helper implementation incomplete')
    plan={'version':VERSION,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_file':cfg.relative_to(ROOT).as_posix(),'config_sha256':sha(cfg),'config':c,
        'source_identity':source.original.identity,'code_sha256':code,'folds':list(range(5)),
        'original_R9_numerical_fixed_arrays_reused_for_selection':False,
        'R9_preserved_only_as_role_sources_and_later_development_comparison':True,
        'actual_models_or_CUDA_or_Test':False,'recipe_finality':c['recipe_status']}
    write(out/'preparation_plan.json',plan)
    summaries={};folds={};files={}
    try:
        for fold in range(5):
            fd=out/f'fold_{fold:02d}';fd.mkdir();tr=describe_backgrounds(source,source.original.role_metadata(fold,'train'))
            thresholds=train_HF_thresholds(tr);tr=apply_HF_thresholds(tr,thresholds)
            tr.to_parquet(fd/'TRAIN_pool.parquet',index=False);write(fd/'TRAIN_HF_thresholds.json',thresholds)
            audit=source_audit(source,fold,tr);write(fd/'source_audit.json',audit);summaries[str(fold)]=audit
            print(json.dumps({'fold':fold,'stage':'actual_source_audit','TRAIN':len(tr),'HF_candidates':audit['HF_candidate_total']}),flush=True)
            if audit_only:continue
            bmeta,quotas=_basis_metadata(tr,c['feature_policy']['basis_backgrounds_per_fold'],c['seed']+fold*10007)
            db=np.stack([source.background(row) for _,row in bmeta.iterrows()])
            bmeta.to_parquet(fd/'basis_source_metadata.parquet',index=False)
            np.savez_compressed(fd/'basis_source_backgrounds.npz',db=db,source_id=np.asarray(['SA:'+str(i) for i in bmeta.base_window_index]),base_window_index=bmeta.base_window_index.to_numpy(np.int64))
            binding={'version':VERSION,'role':'train','fold':fold,'source_identity':source.original.identity,
                'source_ids_sha256':digest(bmeta.base_window_index.astype(int).tolist()),
                'background_npz_sha256':sha(fd/'basis_source_backgrounds.npz'),
                'TRAIN_pool_sha256':sha(fd/'TRAIN_pool.parquet'),'config_sha256':sha(cfg),
                'no_added_component':True,'site_counts':quotas}
            basis=fit_spectral_basis(db,np.asarray(['SA:'+str(i) for i in bmeta.base_window_index]),training_binding=binding,seed=c['seed']+fold*10007)
            save_spectral_basis(basis,fd/'basis.npz')
            variant_entries={}
            for variant in ('baseline','highshare'):
                role_entries={}
                for role_index,role in enumerate(ROLES):
                    pool=describe_backgrounds(source,source.original.role_metadata(fold,role));pool=apply_HF_thresholds(pool,thresholds)
                    picks=fixed_background_indices(pool,c['fixed_examples'][role],np.random.SeedSequence([c['seed'],fold,130+role_index]))
                    chosen=pool.iloc[picks].reset_index(drop=True)
                    if chosen.base_window_index.duplicated().any():raise ValueError('R13 fixed roles require unique backgrounds')
                    sched=choose_refs(make_full_schedule(chosen,1,fold,c['seed'],tag=130+role_index,highshare_fraction=.25 if variant=='highshare' else 0.),source.reference_minutes,c['seed'],fold,1,130+role_index)
                    items=[];recipes=[];meta=[]
                    for i in range(len(chosen)):
                        row=chosen.iloc[int(sched['primary_order'][i])]
                        item,receipt=generate_item(source,row,sched,i,fold,1,role);items.append(item);recipes.append(receipt)
                        meta.append({k:item[k].item() if isinstance(item[k],np.generic) else item[k] for k in ('site','sample_id','source_id','background_group','base_window_index','reference_index','family','explicit_no_source','highshare')}
                            |{'sensor_group':int(row.sensor_group),'recorder':str(row.recorder),'effective_mask_bits':int(row.effective_mask_bits),
                              'start_export_clock':str(row.start_export_clock),'boundary_mask_bits':int(row.boundary_mask_bits),'qc_observed_minutes':int(row.qc_observed_minutes)})
                    arrays={k:np.asarray([v[k] for v in items]) for k in items[0]}
                    if any(v.dtype.kind=='O' for v in arrays.values()):raise ValueError('No-pickle arrays required')
                    no=arrays['explicit_no_source'];y=arrays['share'];q=arrays['cell_share'];o=np.isfinite(arrays['db'])
                    if no.sum()*5!=len(no) or np.any(y[~no]<=0) or np.any(y[no]!=0) or np.any(q[~o]!=0):raise ValueError('Fixed no-source/support/mask law failed')
                    name=variant+'_'+role+'.npz';np.savez_compressed(fd/name,**arrays)
                    with np.load(fd/name,allow_pickle=False) as z:
                        for k,v in arrays.items():
                            if not np.array_equal(z[k],v,equal_nan=v.dtype.kind=='f'):raise ValueError('No-pickle roundtrip mismatch')
                    pd.DataFrame(meta).to_parquet(fd/(variant+'_'+role+'_metadata.parquet'),index=False);write(fd/(variant+'_'+role+'_recipes.json'),recipes)
                    role_entries[role]={'path':f'fold_{fold:02d}/'+name,'sha256':sha(fd/name),
                        'source_pool_windows':len(pool),'source_groups':int(chosen.background_group.nunique()),
                        'rows':len(items),'zero_count':int(no.sum()),'positive_count':int((~no).sum()),
                        'highshare_positive_count':int(arrays['highshare'].sum()),'long_positive_count':int(np.char.startswith(arrays['family'],'long_').sum()),
                        'HF_threshold_unavailable_source_rows':int((~pool.HF_threshold_available).sum()),
                        'HF_priority_shortfall':sched['audit']['HF_priority_shortfall']}
                variant_entries[variant]=role_entries
            folds[str(fold)]={'basis_path':f'fold_{fold:02d}/basis.npz','basis_sha256':sha(fd/'basis.npz'),
                'basis_training_binding':binding,'TRAIN_rows':len(tr),'variant_roles':variant_entries,
                'basis_explained_variance_sum':sum(basis['explained_variance_ratio'])}
            write(out/f'fold_{fold:02d}_PREPARED.json',{'fold':fold,'prepared':True,'model_fit':False})
        source.original.assert_unchanged()
        for n,h in code.items():
            if sha(ROOT/n)!=h:raise ValueError('R13 preparation source changed:'+n)
        if sha(cfg)!=plan['config_sha256']:raise ValueError('Config changed during preparation')
        if audit_only:
            write(out/'SOURCE_AUDIT.json',{'version':VERSION,'status':'completed_metadata_only','folds':summaries,
                'data_ready':False,'PCA_or_fixed_arrays_or_models_fitted':False,'GPU_ready':False})
            return {'status':'source_audit_only','data_ready':False,'folds':summaries}
        for path in sorted(out.rglob('*')):
            if path.is_file():files[path.relative_to(out).as_posix()]=sha(path)
        manifest={'version':VERSION,'status':'completed','config_file':cfg.relative_to(ROOT).as_posix(),
            'config_sha256':sha(cfg),'code_sha256':code,'file_sha256':files,'folds':folds,
            'source_identity':source.original.identity,'source_hash':source.original.source_hash,'data_variants':['baseline','highshare'],
            'source_audits':summaries,'reference_independent_CV':False,'all509_references_shared':True,
            'recipe_status':c['recipe_status'],'production_launch_authorized':c['production_launch_authorized'],
            'fixed_roles_same_R13_law_as_TRAIN':True,'model_fitting_executed':False,
            'Test_or_old_unused_report_array_values_accessed':False}
        manifest['data_hash']=digest(manifest);write(out/'manifest.json',manifest)
        write(out/'readiness.json',{'version':VERSION,'ready_to_train':True,'source_checks_passed':True,
            'manifest_sha256':sha(out/'manifest.json'),'data_hash':manifest['data_hash'],
            'actual_CUDA_verified':False,'production_launch_authorized':c['production_launch_authorized'],
            'recipe_status':c['recipe_status'],'GPU_ready_claimed':False})
        return manifest
    except BaseException as exc:
        write(out/'FAILED.json',{'status':'failed_preserved','error':repr(exc),'partial_reuse':False,'ready_to_train':False})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/r13_full_training_v1.json');p.add_argument('--output');p.add_argument('--audit-only',action='store_true')
    a=p.parse_args();result=prepare(a.config,a.output,a.audit_only);print(json.dumps({'status':result.get('status'),'data_hash':result.get('data_hash'),'GPU_ready':False}),flush=True)
