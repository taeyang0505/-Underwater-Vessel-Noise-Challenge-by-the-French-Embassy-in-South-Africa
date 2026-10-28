"""R16 production raw data: immutable per-fold bundles, no challenge query IO.

Five-fold development CV is nested inside the original R9 Fold0 TRAIN pool.
The R16 fixed backgrounds and reference partitions B/C never enter TRAIN.
Selection reuses the original R16 local_fit six-operator panel (development,
not OOF). Each fold also keeps its own A-reference/background heldout roles.
Only stdlib/NumPy are imported by consumers, including the MPS runtime.
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
VERSION='r16_production_data_v1'
SEED=202610271800
DEFAULT_OUTPUT='data/synthetic/r16_production_v1'
OPERATORS=('control_eq6','mfhf_shelf6','mfhf_shelf12','mfhf_shelf12_lfminus6',
           'alpha_preEQ_inserted','independent_subband_snr')
EVAL_N={'cv_validation':1000,'ensemble_fit':500,'ensemble_gate':500,'report':1000}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _inside(name):
    p=Path(name);p=p.resolve()if p.is_absolute()else(ROOT/p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Project-contained data required')
    return p
def _no_query(event,args):
    if event=='open'and args and isinstance(args[0],(str,bytes)):
        p=str(args[0]).lower()
        if Path(p).name in ('test.csv','sample_submission.csv')or'/eda_deep/'in p or'/r15_band_comparison_'in p:
            raise PermissionError('R16 production excludes challenge query/statistics')

def resolve_fold_manifest(manifest_path,fold):
    path=_inside(manifest_path);m=read(path)
    if m.get('version')!=VERSION:raise ValueError('R16 production version required')
    if 'folds'in m:
        e=m['folds'][str(int(fold))];p=_inside(e['manifest_path'])
        if sha(p)!=e['manifest_sha256']:raise ValueError('Fold manifest changed')
        return p,read(p)
    if m.get('fold')!=int(fold):raise ValueError('Fold binding differs')
    return path,m

def validate_production_manifest(manifest_path,fold,verify_sources=True):
    p,m=resolve_fold_manifest(manifest_path,fold)
    if m.get('status')!='completed' or m.get('ready_for_training')is not True:raise ValueError('Completed per-fold bundle required')
    for role,e in m['roles'].items():
        for k,hk in [('path','sha256'),('metadata_path','metadata_sha256')]:
            if k in e and sha(_inside(e[k]))!=e[hk]:raise ValueError('Role data changed: '+role)
    if verify_sources:
        for n,h in m['source_code_sha256'].items():
            if sha(_inside(n))!=h:raise ValueError('Source code changed: '+n)
        for n,h in m['source_input_sha256'].items():
            if sha(_inside(n))!=h:raise ValueError('Source input changed: '+n)
    return {'status':'passed','fold':int(fold),'manifest_path':str(p.relative_to(ROOT)),
            'manifest_sha256':sha(p),'data_hash':m['data_hash'],'roles':{k:e['n']for k,e in m['roles'].items()},
            'Test_accessed':False}

def load_production_role(manifest_path,fold,role):
    p,m=resolve_fold_manifest(manifest_path,fold)
    if m.get('status')!='completed' or m.get('ready_for_training')is not True:raise ValueError('Data fold not ready')
    if role not in m['roles']:raise ValueError('Unknown R16 role: '+role)
    e=m['roles'][role];path=_inside(e['path'])
    if sha(path)!=e['sha256']:raise ValueError('R16 role bytes changed')
    with np.load(path,allow_pickle=False)as z:d={k:z[k]for k in z.files}
    n=e['n']
    if d['db'].shape!=(n,15,25)or d['cell_share'].shape!=(n,15,25):raise ValueError('R16 raw tensor shape differs')
    if any(a.dtype.kind=='O'for a in d.values()):raise TypeError('Object arrays forbidden')
    if not np.array_equal(d['share']>0,d['presence'].astype(bool)):raise ValueError('Observed labels inconsistent')
    if not np.isfinite(d['share']).all()or not np.isfinite(d['cell_share']).all():raise ValueError('Invalid physical labels')
    if np.isinf(d['db']).any():raise ValueError('Infinite input')
    if np.max(np.abs(d['db'][np.isfinite(d['db'])]*10-np.round(d['db'][np.isfinite(d['db'])]*10)))>.0002:raise ValueError('Not 0.1dB grid')
    return d

def _bucket_groups(frame,group_name,count=5,weight_col=None):
    # Metadata-only greedy balance; no signal/target or performance is read.
    sizes=frame.groupby(group_name).size().to_dict()
    order=sorted(sizes,key=lambda g:(-sizes[g],canonical([SEED,group_name,str(g)])))
    totals=np.zeros(count,int);assignment={}
    for g in order:
        choices=np.flatnonzero(totals==totals.min());j=int(choices[int(canonical([SEED,str(g),'tie'])[:12],16)%len(choices)])
        assignment[g]=j;totals[j]+=sizes[g]
    return assignment,totals.tolist()

def _source_code(source):
    paths=['pipeline/r16_production_data.py','ops/r16_forward_data.py','ops/r15_forward_prototype.py',
       'pipeline/background_cv_r9.py','pipeline/vtuad_reference_r4.py']
    return {**source.code_sha256,**{n:sha(ROOT/n)for n in paths}}

def _crossrole(primary,bench):
    import pandas as pd
    proofs=[]
    for name,held in bench.items():
        overlap=len(set(primary.base_window_index)&set(held.base_window_index));goverlap=len(set(primary.background_group)&set(held.background_group))
        gaps=[]
        for sensor,a in primary.groupby('sensor_group'):
            b=held[held.sensor_group==sensor]
            if not len(b):continue
            x=pd.to_datetime(a.start_export_clock).astype('int64').to_numpy()/1e9
            y=np.sort(pd.to_datetime(b.start_export_clock).astype('int64').to_numpy()/1e9);pos=np.searchsorted(y,x)
            distance=np.minimum(abs(x-y[np.clip(pos-1,0,len(y)-1)]),abs(x-y[np.clip(pos,0,len(y)-1)]))
            gaps.append(float(np.maximum(distance-900,0).min()/3600))
        minimum=min(gaps)if gaps else None
        if overlap or goverlap or minimum is not None and minimum<24-1e-8:raise ValueError('Fixed benchmark source leaked: '+name)
        proofs.append({'benchmark':name,'base_window_overlap':overlap,'background_group_overlap':goverlap,
                       'minimum_support_gap_hours':minimum,'no_shared_sensor':not bool(gaps)})
    return proofs

def prepare_sources(source,out):
    import pandas as pd
    from ops.r16_forward_data import reference_partition,domain_metadata
    from pipeline.background_cv_r9 import heldout_group_roles,proposed_purge_gap,final_cross_role_gaps
    primary=source.role_metadata(0,'train').copy()
    if len(primary)!=78869 or primary.background_group.nunique()!=206:raise ValueError('Frozen R9 Fold0 primary pool identity changed')
    ref,refproof=reference_partition(source);ra=ref[ref.r16_reference_partition.eq('A')].copy()
    ref.to_parquet(out/'reference_partitions.parquet',index=False)
    bgmap,bc=_bucket_groups(primary,'background_group');rmap,rc=_bucket_groups(ra,'reference_component')
    primary['r16_outer_fold']=primary.background_group.map(bgmap).astype(int)
    ra['r16_outer_fold']=ra.reference_component.map(rmap).astype(int)
    primary.to_parquet(out/'primary_background_pool.parquet',index=False);ra.to_parquet(out/'primary_reference_pool.parquet',index=False)
    small=ROOT/'reports/r16_cross_operator_r1/data'
    bench={name:pd.read_parquet(small/(name+'_backgrounds.parquet'))for name in ('local_fit','local_gate','local_report','domain_holdout')}
    proof=_crossrole(primary,bench)
    folds=[]
    for fold in range(5):
        f=primary.copy();held=f.r16_outer_fold.eq(fold);mapping=heldout_group_roles(f.loc[held,'background_group'],fold,SEED)
        f['proposed_role']='train';f.loc[held,'proposed_role']=f.loc[held,'background_group'].map(mapping)
        f['split']=f.proposed_role;gap=proposed_purge_gap(f);f['r16_cross_role_gap_hours']=gap/3600
        f.loc[gap<86400,'split']='purged_r16_temporal_boundary'
        active=f[f.split.ne('purged_r16_temporal_boundary')].copy().reset_index(drop=True)
        final=final_cross_role_gaps(active)
        if (final<86400).any()or active.groupby('background_group').split.nunique().max()!=1:raise ValueError('R16 ownfold purge/group isolation failed')
        active['r16_minimum_other_role_support_gap_hours']=final/3600
        active['role']=active.split.replace({'selection':'cv_validation'})
        rt=ra[ra.r16_outer_fold.ne(fold)].copy();rv=ra[ra.r16_outer_fold.eq(fold)].copy()
        if set(rt.reference_component)&set(rv.reference_component):raise ValueError('Reference fold overlap')
        directory=out/f'fold_{fold:02d}';directory.mkdir()
        active.to_parquet(directory/'all_admitted_sources.parquet',index=False);f[f.split.eq('purged_r16_temporal_boundary')].to_parquet(directory/'purged_sources.parquet',index=False)
        rt.to_parquet(directory/'train_references.parquet',index=False);rv.to_parquet(directory/'heldout_A_references.parquet',index=False)
        roleframes={};stats={}
        for role,df in active.groupby('role'):
            df=df.sort_values('base_window_index').reset_index(drop=True);stats[role]={'source_windows':len(df),'source_groups':df.background_group.nunique()}
            fullpath=directory/(role+'_source_pool.parquet');df.to_parquet(fullpath,index=False)
            if role=='train':chosen=df
            else:
                n=min(EVAL_N[role],len(df));rng=np.random.default_rng(np.random.SeedSequence([SEED,fold,0x56414c,int(canonical(role)[:8],16)]))
                # At least one selected window from every source group when budget permits.
                included=[int(rng.choice(ix))for ix in df.groupby('background_group').indices.values()]
                if len(included)>n:raise ValueError('Fixed evaluation quota smaller than source groups')
                rest=np.setdiff1d(np.arange(len(df)),included);included+=rng.choice(rest,n-len(included),replace=False).tolist()
                chosen=df.iloc[np.asarray(included)[rng.permutation(n)]].reset_index(drop=True)
            roleframes[role]=chosen;chosen.to_parquet(directory/(role+'_backgrounds.parquet'),index=False)
            stats[role].update(materialized_n=len(chosen),selected_source_groups=chosen.background_group.nunique())
        folds.append({'fold':fold,'frames':roleframes,'train_refs':rt,'held_refs':rv,'stats':stats,
                      'purged_windows':int(f.split.eq('purged_r16_temporal_boundary').sum()),
                      'minimum_cross_role_support_gap_hours':float(final[np.isfinite(final)].min()/3600),
                      'train_reference_clips':len(rt),'train_reference_components':rt.reference_component.nunique(),
                      'heldout_A_reference_clips':len(rv),'heldout_A_reference_components':rv.reference_component.nunique()})
    proof={'primary_background_windows':len(primary),'primary_background_groups':primary.background_group.nunique(),
           'background_fold_windows_before_purge':bc,'reference_A_fold_clip_counts':rc,
           'fixed_benchmark_preservation':proof,'reference_partition':refproof,
           'fivefold_scope':'R9 Fold0 TRAIN78869 subset only, not original100965/full509 CV',
           'global_B_C_TRAIN_reference_clips':0,'reference_A_pool_clips':66,'reference_A_pool_components':12,
           'source_group_rotation_not_row_random_split':True,'source_assignment_uses_acoustic_values_or_model_results':False}
    return folds,proof

def _lean_arrays(n):
    return {'db':np.empty((n,15,25),np.float32),'cell_share':np.empty((n,15,25),np.float32),
        'share':np.empty(n,np.float64),'presence':np.empty(n,np.int64),'operator':np.empty(n,dtype='<U32'),
        'reference_index':np.empty(n,np.int64),'reference_component':np.empty(n,dtype='<U64'),
        'family':np.empty(n,dtype='<U40'),'unrounded_share_pool_error':np.empty(n,np.float64),
        'rounded_share_pool_gap':np.empty(n,np.float64),'scale_alpha_low':np.empty(n,np.float64),
        'scale_alpha_high':np.empty(n,np.float64)}

def materialize(source,frame,refs,fold,role,path):
    from ops.r16_forward_data import choose_geometry,channel_zero_mean,generate_operator
    from pipeline.sampler_r9 import smooth_eq
    n=len(frame);code={'train':0,'cv_validation':1,'ensemble_fit':2,'ensemble_gate':3,'report':4}[role]
    rng=np.random.default_rng(np.random.SeedSequence([SEED,fold,code,0x534348]));zero=np.zeros(n,bool)
    zero[rng.choice(n,n//5,replace=False)]=True;positive=np.flatnonzero(~zero)
    available=refs.reference_row_index.to_numpy(int);order=[]
    while len(order)<len(positive):order.extend(rng.permutation(available).tolist())
    rindex=np.full(n,-1,int);rindex[positive]=np.asarray(order[:len(positive)])
    operations=np.full(n,'control_eq6',dtype='<U32');schedule=np.resize(np.asarray(OPERATORS),len(positive));rng.shuffle(schedule);operations[positive]=schedule
    lookup=refs.set_index('reference_row_index');a=_lean_arrays(n);start=time.perf_counter()
    for i,row in frame.iterrows():
        local=np.random.default_rng(np.random.SeedSequence([SEED,fold,code,i,0x4752]));bg=np.asarray(source.background(row),float)
        native=None if zero[i]else source.reference_minutes[int(rindex[i])]
        geom=None if zero[i]else choose_geometry(bg,native,local,report_crop=role=='report')
        eq=np.zeros(25)if zero[i]else smooth_eq(local,6.)
        post=np.zeros(25)if zero[i]else channel_zero_mean(local,6. if role=='train'else 12.)
        snr=float(local.uniform(-10,10));sub=local.uniform(-10,10,2)
        d,receipt=generate_operator(bg,native,geom,eq,post,snr,sub,str(operations[i]),zero=bool(zero[i]))
        a['db'][i]=d['db'];a['cell_share'][i]=d['cell_share'];a['share'][i]=d['share'];a['presence'][i]=d['presence']
        a['operator'][i]=operations[i];a['reference_index'][i]=rindex[i]
        a['reference_component'][i]=''if zero[i]else str(lookup.loc[int(rindex[i])].reference_component)
        a['family'][i]='no_source'if zero[i]else geom['family'];a['unrounded_share_pool_error'][i]=receipt['raw_pool_identity_error'];a['rounded_share_pool_gap'][i]=receipt['rounded_pool_gap']
        alpha=receipt['alpha_subbands']or[receipt['alpha']or 0.]*2;a['scale_alpha_low'][i]=alpha[0];a['scale_alpha_high'][i]=alpha[1]
        # Every row is verified before component arrays are released.
        if not np.array_equal(d['mixture_observed_power'],d['background_observed_power']+d['added_observed_power']):raise AssertionError('Exact E=B+A failed')
        expected=d['added_observed_power'].sum()/d['mixture_observed_power'].sum()
        if abs(expected-a['share'][i])>1e-12:raise AssertionError('Exact y integrity failed')
        if i and i%10000==0:print(json.dumps({'event':'production_data_progress','fold':fold,'role':role,'rows':i,'total':n,'seconds':time.perf_counter()-start}),flush=True)
    a.update(sample_id=np.asarray([f'R16PROD_f{fold}_{role}_{i:06d}'for i in range(n)],dtype=str),
       site=frame.site.to_numpy(dtype=str),background_group=frame.background_group.to_numpy(dtype=str),
       base_window_index=frame.base_window_index.to_numpy(np.int64),explicit_no_source=zero,
       boundary_mask_bits=frame.boundary_mask_bits.to_numpy(np.int64))
    if (not np.array_equal(a['share']>0,~zero)or(a['cell_share']<0).any()or(a['cell_share']>1).any()
        or any(v.dtype.kind=='O'for v in a.values())):raise ValueError('Production labels/schema failed')
    if np.any(a['cell_share'][~np.isfinite(a['db'])]!=0):raise ValueError('Missing q differs')
    np.savez_compressed(path,**a)
    with np.load(path,allow_pickle=False)as z:
        if len(z['share'])!=n or any(z[k].dtype.kind=='O'for k in z.files):raise ValueError('Reload verification failed')
    return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'n':n,
       'metadata_path':str(path.with_name(role+'_backgrounds.parquet').relative_to(ROOT)),
       'metadata_sha256':sha(path.with_name(role+'_backgrounds.parquet')),
       'explicit_zero_count':int(zero.sum()),'observed_positive_count':int((~zero).sum()),
       'operator_positive_counts':{str(k):int(v)for k,v in zip(*np.unique(operations[~zero],return_counts=True))},
       'max_raw_pool_error':float(a['unrounded_share_pool_error'].max()),'max_rounded_pool_gap':float(a['rounded_share_pool_gap'].max()),
       'share_mean':float(a['share'].mean()),'share_rms':float(np.sqrt(np.mean(a['share']**2))),
       'seconds':time.perf_counter()-start,'size_bytes':path.stat().st_size,
       'full_components_storage':'not retained; everyrow verified during generation, deterministic receipt seeds/source metadata permit replay'}

def shared_selection(out):
    from ops.r16_forward_data import load_panel
    m=ROOT/'reports/r16_cross_operator_r1/data/manifest.json';parts=[load_panel(m,'local_fit',op)for op in OPERATORS]
    keys=('db','cell_share','share','presence','site','background_group','base_window_index','reference_index','reference_component','family','operator','sample_id','pair_id','explicit_no_source')
    a={k:np.concatenate([d[k]for d in parts])for k in keys};p=out/'shared_selection.npz';np.savez_compressed(p,**a)
    assert a['db'].shape==(2400,15,25)
    return {'path':str(p.relative_to(ROOT)),'sha256':sha(p),'n':2400,'source_manifest':str(m.relative_to(ROOT)),
       'source_manifest_sha256':sha(m),'same_as_original_R16_local_fit':True,
       'checkpoint_and_ensemble_development_selection_only_not_OOF':True,
       'reference_partition':'B','backgrounds_unique':400,'operators':6}

def build(output=DEFAULT_OUTPUT):
    sys.addaudithook(_no_query);out=_inside(output)
    if out.exists():raise FileExistsError('Fresh production data bundle required; no failed partial reuse')
    out.mkdir(parents=True);start=time.perf_counter();own=sha(__file__)
    from pipeline.dataset_r9 import R9Sources
    source=R9Sources();code=_source_code(source)
    write(out/'PLAN.json',{'version':VERSION,'seed':SEED,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'mixed_operators':list(OPERATORS),'own_code_sha256':own,'reference_scope':'A66/12 only, allB/C globallyheld',
       'background_scope':'R9Fold0TRAIN78869 only; 5foldsourcegroups with900s+24hpurge',
       'selection':'originalR16local_fit exact2400 allsixoperators Bref; checkpointdevelopmentnotOOF',
       'internal_A_reference_heldout_roles':EVAL_N,'train_n':'all admitted ownfold backgrounds exactlyonce',
       'timewarp_or_repetition_or_artificial_bell_or_full_smooth15_or_highshare':False,
       'Test_or_Query_statistics_used':False,'input_features':'rawdb only; R16relative8 suppliedbyconsumer',
       'fixed_NPZ_no_epoch_regeneration':True,'row_recipe_seed':'SeedSequence(SEED,fold,rolecode,row,0x4752)',
       'uniform_SNR_db':[-10,10],'post_EQ_max_db_train':6.,'post_EQ_max_db_eval':12.})
    folds,source_audit=prepare_sources(source,out);write(out/'SOURCE_AUDIT.json',source_audit)
    inputs={str(p.relative_to(ROOT)):sha(p)for p in [out/'PLAN.json',out/'SOURCE_AUDIT.json',out/'reference_partitions.parquet',out/'primary_background_pool.parquet',out/'primary_reference_pool.parquet',
       ROOT/'reports/r16_cross_operator_r1/data/manifest.json',ROOT/'data/versions/vtuad_reference_r4_v1/reference_manifest.parquet',
       source.config_path,source.bg_dir/'manifest.json',source.ref_dir/'manifest.json']}
    selection=shared_selection(out);index={}
    for f in folds:
        fold=f['fold'];directory=out/f'fold_{fold:02d}';roles={'selection':selection}
        for role,frame in f['frames'].items():
            refs=f['train_refs']if role=='train'else f['held_refs']
            roles[role]=materialize(source,frame,refs,fold,role,directory/(role+'.npz'))
        stats={k:v for k,v in f.items()if k not in ('frames','train_refs','held_refs')}
        datahash=canonical({'roles':{k:e['sha256']for k,e in roles.items()},'code':code,'inputs':inputs,'fold':fold,'seed':SEED})
        m={'version':VERSION,'status':'completed','ready_for_training':True,'fold':fold,'seed':SEED,'data_hash':datahash,
           'roles':roles,'source_code_sha256':code,'source_input_sha256':inputs,'source_identity':source.identity,
           'fold_statistics':stats,'primary_benchmark_source_preserved':True,'global_B_C_reference_TRAIN_overlap':0,
           'cv_scope':'source-groupCV inside78869developmentpool; AonlyreferenceCV; sharedselectionBisnotOOF',
           'code_and_data_ready_not_training_performed':True,'Test_used':False}
        write(directory/'manifest.json',m)
        qa=validate_production_manifest(directory/'manifest.json',fold)
        qa.update(all_rows_physical_verified=True,ownfold_source_and_purge_passed=True,benchmark_source_preservation_passed=True,
           ownfold_train_reference_group_overlap=0,raw_q_y_from_components=True,db_quantization_decimals=1)
        write(directory/'DATA_QA.json',qa)
        index[str(fold)]={'manifest_path':str((directory/'manifest.json').relative_to(ROOT)),
                         'manifest_sha256':sha(directory/'manifest.json'),'qa_path':str((directory/'DATA_QA.json').relative_to(ROOT)),
                         'qa_sha256':sha(directory/'DATA_QA.json'),'train_n':roles['train']['n']}
        write(out/'progress.json',{'status':'building','completed_folds':list(index),'folds':index,'elapsed_seconds':time.perf_counter()-start})
        print(json.dumps({'event':'production_fold_ready','fold':fold,'train_n':roles['train']['n'],'manifest':index[str(fold)],'seconds':time.perf_counter()-start}),flush=True)
    source.assert_unchanged()
    if sha(__file__)!=own:raise ValueError('Production builder changed during run')
    global_manifest={'version':VERSION,'status':'completed','ready_for_training':True,'folds':index,
       'source_code_sha256':code,'source_input_sha256':inputs,'source_audit':'SOURCE_AUDIT.json',
       'elapsed_seconds':time.perf_counter()-start,'rows_train_across_folds':sum(e['train_n']for e in index.values()),
       'primary_unique_backgrounds':78869,'primary_unique_reference_clips':66,'primary_unique_reference_components':12,
       'Test_used':False,'new_models_fitted':False}
    write(out/'manifest.json',global_manifest);write(out/'progress.json',{'status':'completed','elapsed_seconds':time.perf_counter()-start,'folds':index})
    return global_manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['build','validate']);p.add_argument('--output',default=DEFAULT_OUTPUT);p.add_argument('--manifest');p.add_argument('--fold',type=int,default=0);a=p.parse_args()
    if a.command=='build':print(json.dumps(build(a.output),ensure_ascii=False))
    else:print(json.dumps(validate_production_manifest(a.manifest,a.fold),ensure_ascii=False))
