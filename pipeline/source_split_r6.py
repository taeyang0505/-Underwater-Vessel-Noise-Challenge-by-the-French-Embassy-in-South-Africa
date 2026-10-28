"""R6 source/QC version; old OUTER/calibration/domain/sealed/Test values excluded.

Actual selected original row_ids locate candidate minutes. Whole raw banks are
hashed for integrity only; values are read at admitted original indices/slices.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
VERSION='source_split_r6_cpu_v1'
ROLES=('train','inner','cross_generator','cross_sensor')
DIAGNOSTIC={'cross_generator','cross_sensor'}
SOURCE_DEFAULT='data/synthetic/group_cv_passage_r4_v1/fold_00/sources'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def resolve(root,path):
    p=Path(path);return p if p.is_absolute() else Path(root)/p
def relative(root,path):return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def rank(seed,kind,value):return hashlib.sha256(f'R6|{kind}|seed={seed}|{value}'.encode()).hexdigest()

def apply_minute_qc(db,mask_bits):
    x=np.asarray(db);bits=np.asarray(mask_bits)
    if x.ndim!=3 or x.shape[1:]!=(15,25) or bits.shape!=(len(x),) or not np.issubdtype(bits.dtype,np.integer):raise ValueError('Expected windows and integer minute masks')
    if np.isinf(x).any() or (bits<0).any() or (bits>=(1<<15)).any():raise ValueError('Invalid raw values or mask')
    out=x.copy();out[((bits[:,None]>>np.arange(15))&1).astype(bool)]=np.nan
    if not np.isnan(out)[np.isnan(x)].all():raise AssertionError('Original NaN changed')
    return out

def nearest_gap(starts,other):
    if not len(other):return np.full(len(starts),np.inf)
    other=np.sort(other);pos=np.searchsorted(other,starts)
    return np.minimum(np.abs(starts-other[np.clip(pos-1,0,len(other)-1)]),np.abs(starts-other[np.clip(pos,0,len(other)-1)])).astype(float)

def role_gaps(frame):
    sec=pd.to_datetime(frame.start_export_clock).to_numpy(dtype='datetime64[ns]').astype('int64')//10**9;gap=np.full(len(frame),np.inf)
    for ids in frame.groupby('sensor_group',sort=False).indices.values():
        ids=np.asarray(ids);labels=frame.iloc[ids].split.to_numpy()
        for label in sorted(set(labels)):
            chosen=ids[labels==label];other=ids[labels!=label]
            gap[chosen]=np.maximum(nearest_gap(sec[chosen],sec[other])-900,0)/3600
    return gap

def assign_background_roles(frame,seed):
    b=frame.copy();counts=b.groupby('sensor_group').agg(windows=('window_id','size'),weeks=('recording_week_start','nunique'))
    eligible=[int(g) for g,r in counts.iterrows() if r.windows>=200 and r.weeks>=2]
    if len(eligible)<2:raise ValueError('At least two eligible sensor groups required')
    held=min(eligible,key=lambda g:rank(seed,'cross_sensor',f'sensor_group={g}'))
    b['split']=np.where(b.r4_split.eq('validation'),'inner','train');b.loc[b.sensor_group.eq(held),'split']='cross_sensor'
    for sensor,g in b[b.split.eq('inner')].groupby('sensor_group'):
        weeks=sorted(g.recording_week_start.unique(),key=lambda w:rank(seed,'cross_generator_block',f'{sensor}|{w}'))
        if len(weeks)>=2:
            n=min(len(weeks)-1,max(1,math.ceil(.2*len(weeks))))
            b.loc[b.sensor_group.eq(sensor)&b.recording_week_start.isin(weeks[:n])&b.r4_split.eq('validation'),'split']='cross_generator'
    return b,held,{'eligible_sensor_metadata':counts.reset_index().to_dict('records'),'minimum_windows':200,'minimum_weeks':2,
        'selection_rule':'minimum SHA256(R6|cross_sensor|seed=SEED|sensor_group=G)','selected_sensor_group':held,'seed':int(seed),'model_results_used':False}

def reference_coverage(frame):
    k=frame.segment_count.to_numpy(int);missing=[m for m in range(60,451) if not ((m/k>=1/3)&(m/k<=3)).any()]
    return {'grid':[60,450],'grid_points':391,'missing_rendered_segments':missing,'passed':not missing,
        'minimum_source_seconds':float(frame.duration_seconds.min()),'maximum_source_seconds':float(frame.duration_seconds.max())}

def assign_reference_roles(frame,seed):
    r=frame[frame.split.isin(['train','validation'])].copy()
    if r.empty or r.sealed.any():raise ValueError('Admitted nonsealed refs required')
    r['r4_source_row_index']=r.index.to_numpy();r['r4_source_split']=r.split;r['provider_original_split']=r.original_split
    groups=sorted(r.loc[r.split.eq('train'),'reference_component'].unique(),key=lambda g:rank(seed,'inner_reference',g))
    if len(groups)<4:raise ValueError('Insufficient TRAIN reference groups')
    n=min(len(groups)-2,max(2,math.ceil(.2*len(groups))));inner=set(groups[:n]);t=r.r4_source_split.eq('train')
    r.loc[t,'split']=np.where(r.loc[t,'reference_component'].isin(inner),'inner','train')
    groups=sorted(r.loc[r.r4_source_split.eq('validation'),'reference_component'].unique(),key=lambda g:rank(seed,'diagnostic_reference',g))
    if len(groups)<2:raise ValueError('Separate diagnostic reference groups required')
    t=r.r4_source_split.eq('validation');cs=set(groups[:max(1,len(groups)//2)])
    r.loc[t,'split']=np.where(r.loc[t,'reference_component'].isin(cs),'cross_sensor','cross_generator');r['r6_role']=r.split
    for col in ['reference_component','hull_id','recording_day','sha256']:
        if r.groupby(col).split.nunique().max()!=1:raise ValueError('Reference identity crosses roles: '+col)
    begin=pd.to_datetime(r.utc_begin,utc=True).to_numpy(dtype='datetime64[ns]').astype('int64');end=pd.to_datetime(r.utc_end,utc=True).to_numpy(dtype='datetime64[ns]').astype('int64');labels=r.split.to_numpy()
    for i in range(len(r)):
        if ((begin[i]<end)&(end[i]>begin)&(labels[i]!=labels)).any():raise ValueError('Reference UTC support crosses roles')
    coverage={s:reference_coverage(r[r.split.eq(s)]) for s in ['train','inner']}
    if not all(v['passed'] for v in coverage.values()):raise ValueError('Fixed hash split lacks R4 rendered-grid coverage: '+str(coverage))
    return r,coverage

def minute_masks(allowed,row_manifest,boundary_flags):
    rows=row_manifest[row_manifest.window_included & row_manifest.eligible_for_minute_window].copy()
    rows['minute_clock']=pd.to_datetime(rows.t).dt.floor('min');rows['window_clock']=rows.minute_clock.dt.floor('15min')
    key=['site','recorder','window_clock'];w=allowed[['base_window_index','site','recorder','start_export_clock']].copy();w['window_clock']=pd.to_datetime(w.start_export_clock)
    if w.duplicated(key).any():raise ValueError('Ambiguous window identity')
    selected=rows.merge(w[key+['base_window_index']],on=key,how='inner',validate='many_to_one')
    selected['minute_index']=((selected.minute_clock-selected.window_clock).dt.total_seconds()/60).astype('int8')
    if selected.duplicated(['base_window_index','minute_index']).any() or not selected.minute_index.between(0,14).all():raise ValueError('Ambiguous original observed minute')
    flags=boundary_flags
    if flags.row_id.duplicated().any():raise ValueError('Duplicate boundary row_id')
    check=flags.merge(row_manifest[['row_id','site','recorder','t']],on='row_id',suffixes=('_flag','_row'),validate='one_to_one')
    if len(check)!=len(flags) or not (check.site_flag.eq(check.site_row)&check.recorder_flag.eq(check.recorder_row)&check.t_flag.eq(check.t_row)).all():raise ValueError('Boundary row provenance differs')
    flagged=selected.merge(flags[['row_id','candidate_reason']],on='row_id',how='inner',validate='one_to_one')
    bits={int(i):0 for i in allowed.base_window_index}
    for i,m in zip(flagged.base_window_index,flagged.minute_index):bits[int(i)]|=1<<int(m)
    counts={int(i):int(n) for i,n in selected.groupby('base_window_index').size().items()}
    return np.array([bits[int(i)] for i in allowed.base_window_index],dtype=np.uint16),flagged,counts

def build_source_split_r6(*,output_dir,seed=20261026,root=ROOT,source_dir=SOURCE_DEFAULT,
        raw_bank='data/processed/sa_windows_db.npy',row_manifest='data/processed/sa_row_manifest.parquet',
        window_manifest='data/processed/sa_window_manifest.parquet',boundary_flags='reports/data_quality_boundary_flags.parquet',
        config='configs/r6_cpu_ablation_v1.json',report_prefix='reports/R6_SOURCE_SPLIT_AUDIT'):
    root=Path(root).resolve();out=resolve(root,output_dir).resolve()
    if out.exists():raise FileExistsError('Fresh R6 version required; no overwrite')
    source=resolve(root,source_dir);gate_path=source.parent/'readiness_manifest.json'
    info=json.loads((source/'manifest.json').read_text());gate=json.loads(gate_path.read_text());cfg=json.loads(resolve(root,config).read_text())
    if seed!=cfg['seed'] or cfg['version']!='r6_cpu_ablation_v1':raise ValueError('Use predeclared R6 seed')
    for name in ['ready_to_train','source_checks_passed','reference_group_and_duplicate_isolation_passed','background_week_and_purge_isolation_passed','sealed_holdout_locked']:
        if gate.get(name) is not True:raise ValueError('Parent readiness guard failed: '+name)
    for name in ['challenge_test_accessed','sealed_arrays_accessed','existing_sealed_NOAA_accessed','sealed_holdout_materialized','frozen_v1_data_modified','model_evaluated_on_sealed','calibration_accessed']:
        if gate.get(name) is not False:raise ValueError('Forbidden parent access guard: '+name)
    if gate['source_hash']!=info['source_hash'] or sha(source/'manifest.json')!=gate['source_version_manifest_sha256']:raise ValueError('Frozen source identity changed')
    for n,h in info['metadata_sha256'].items():
        if sha(source/n)!=h:raise ValueError('Frozen source metadata changed')
    b=pd.read_parquet(source/'background_manifest.parquet');refs=pd.read_parquet(source/'reference_manifest.parquet')
    restore=b.split.eq('excluded_boundary_candidate')&b.parent_split.eq('train')
    allowed=b[b.split.isin(['train','validation'])|restore].copy().reset_index(drop=True)
    allowed['r4_split']=allowed.split;allowed['r4_parent_split']=allowed.parent_split
    allowed['restored_boundary_train_window']=allowed.r4_split.eq('excluded_boundary_candidate')
    if allowed.base_window_index.duplicated().any():raise ValueError('Duplicate admitted raw index')
    inputs=[source/'manifest.json',source/'background_manifest.parquet',source/'reference_manifest.parquet',gate_path,
        resolve(root,row_manifest),resolve(root,window_manifest),resolve(root,boundary_flags),resolve(root,config)]
    parent=resolve(root,'data/synthetic/group_cv_physics_v2_r1/fold_00/sources/manifest.json');old=json.loads(parent.read_text());inputs.append(parent)
    banks={raw_bank:sha(resolve(root,raw_bank))}
    for n in refs.loc[refs.split.isin(['train','validation']),'segment_file'].unique():banks[n]=sha(resolve(root,n))
    for n,h in banks.items():
        if old['frozen_v1_hashes'].get(Path(n).name)!=h:raise ValueError('Frozen numeric bank changed: '+n)
    hashes={relative(root,p):sha(p) for p in inputs}
    raw=np.load(resolve(root,raw_bank),mmap_mode='r',allow_pickle=False)
    if raw.ndim!=3 or raw.shape[1:]!=(15,25):raise ValueError('Unexpected raw window bank')
    wm=pd.read_parquet(resolve(root,window_manifest));ids=allowed.base_window_index.to_numpy(int)
    if ids.min()<0 or ids.max()>=len(raw):raise ValueError('Admitted index bounds invalid')
    original=wm.iloc[ids].reset_index(drop=True)
    for col in ['window_id','site','recorder','start_export_clock','sensor_group']:
        if not allowed[col].astype(str).equals(original[col].astype(str)):raise ValueError('Window bank identity differs: '+col)
    rows=pd.read_parquet(resolve(root,row_manifest));flags=pd.read_parquet(resolve(root,boundary_flags))
    bits,masked,counts=minute_masks(allowed,rows,flags);allowed['boundary_mask_bits']=bits
    allowed['masked_minute_indices']=[[m for m in range(15) if (int(v)>>m)&1] for v in bits]
    allowed['masked_candidate_minutes']=[int(v).bit_count() for v in bits]
    observed=[];original_observed=[]
    for start in range(0,len(allowed),512):
        take=ids[start:start+512];x=np.asarray(raw[take]);o=np.isfinite(x).any(2)
        if np.isinf(x).any() or (np.isfinite(x).any(2)!=np.isfinite(x).all(2)).any():raise ValueError('Unexpected partial/invalid raw minute')
        if any(counts.get(int(i),0)!=int(n) for i,n in zip(take,o.sum(1))):raise ValueError('Original selected rows do not reconstruct coverage')
        qc=apply_minute_qc(x,bits[start:start+len(take)])
        original_observed.extend(o.sum(1).tolist());observed.extend(np.isfinite(qc).any(2).sum(1).tolist())
    allowed['raw_observed_minutes']=original_observed;allowed['qc_observed_minutes']=observed
    if not np.array_equal(allowed.raw_observed_minutes,allowed.observed_minutes):raise ValueError('Original coverage differs')
    if allowed.loc[allowed.restored_boundary_train_window,'masked_candidate_minutes'].eq(0).any():raise ValueError('Restored boundary window has no selected flagged row')
    assigned,held,selection=assign_background_roles(allowed,seed);assigned['r6_proposed_role']=assigned.split
    assigned.loc[assigned.qc_observed_minutes.lt(3),'split']='excluded_insufficient_observed_support'
    good=assigned.split.isin(ROLES);assigned.loc[good,'nearest_proposed_other_role_support_gap_hours']=role_gaps(assigned[good].reset_index(drop=True))
    assigned.loc[good&assigned.nearest_proposed_other_role_support_gap_hours.lt(24),'split']='purged_r6_temporal_boundary'
    active=assigned[assigned.split.isin(ROLES)].copy().reset_index(drop=True);active['r6_role']=active.split
    gaps=role_gaps(active);active['nearest_other_role_support_gap_hours']=gaps
    if not len(active) or np.min(gaps)<24 or set(active.split)!=set(ROLES):raise ValueError('R6 panels or24h purge failed')
    if active.groupby(['sensor_group','recording_week_start']).split.nunique().max()!=1:raise ValueError('Source week crosses roles')
    if set(active[active.split.eq('cross_sensor')].sensor_group)!={held} or held in set(active[active.split.ne('cross_sensor')].sensor_group):raise ValueError('Sensor holdout crosses TRAIN')
    if not set(active[active.split.eq('inner')].sensor_group)<=set(active[active.split.eq('train')].sensor_group):raise ValueError('INNER not known-sensor')
    reference,coverage=assign_reference_roles(refs,seed)
    for col in ['segment_file','segment_start','segment_count']:reference['original_'+col]=reference[col]
    expected={}
    for role in ['train','validation']:expected.update(gate['partitions'][role]['reference_segment_slice_sha256'])
    reference['expected_segment_slice_sha256']=[expected.get(c) for c in reference.clip_id]
    if reference.expected_segment_slice_sha256.isna().any():raise ValueError('Unbound admitted reference slice')
    masked=masked.merge(assigned[['base_window_index','split']],on='base_window_index',validate='many_to_one')
    excluded=assigned[~assigned.split.isin(ROLES)].copy();sites=[]
    for site,g in assigned.groupby('site'):
        a=active[active.site.eq(site)]
        sites.append({'site':site,'old_clean_train_windows':int(g.r4_split.eq('train').sum()),'old_inner_windows':int(g.r4_split.eq('validation').sum()),
            'old_boundary_excluded_train_windows':int(g.restored_boundary_train_window.sum()),'new_retained_windows':len(a),
            'retained_recovered_boundary_windows':int(a.restored_boundary_train_window.sum()),**{r:int(a.split.eq(r).sum()) for r in ROLES},'new_QC_or_purge_exclusions':len(g)-len(a)})
    report={'status':'PASS','version':VERSION,'seed':int(seed),'sensor_selection':selection,
        'source_audit':{'admitted_windows':len(allowed),'forbidden_old_source_rows_not_numerically_used':len(b)-len(allowed),
            'global_boundary_candidate_rows':len(flags),'global_candidate_rows_selected_into_original_windows':int(flags.window_included.sum()),
            'admitted_flagged_selected_minutes':len(masked),'admitted_windows_with_flagged_minutes':int((bits>0).sum()),
            'minute_positions_masked':{str(k):int(v) for k,v in masked.minute_index.value_counts().sort_index().items()},
            'actual_original_row_id_join_not_first_last_heuristic':True,'boundary_physical_cause_verified':False,
            'requested_candidate_minute_masking_authorized':True,'numeric_background_windows_read':len(allowed),
            'reference_values_read_during_build':False,'global_bank_hash_read_is_integrity_only':True},
        'background_role_counts':{r:int(active.split.eq(r).sum()) for r in ROLES},
        'reference_role_counts':{r:{'clips':int(reference.split.eq(r).sum()),'groups':int(reference[reference.split.eq(r)].reference_component.nunique())} for r in ROLES},
        'known_sensor_train_without_inner':sorted(int(v) for v in set(active[active.split.eq('train')].sensor_group)-set(active[active.split.eq('inner')].sensor_group)),
        'minimum_same_sensor_cross_role_support_gap_hours':None if not np.isfinite(gaps).any() else float(np.min(gaps)),
        'R4_comparison_sampler_coverage':coverage,'site_retention':sites,'reference_group_hull_day_SHA_UTC_isolation_passed':True,
        'diagnostic_roles_locked':sorted(DIAGNOSTIC),'cross_sensor_generator_reuses_cross_sensor_sources':True,
        'old_OUTER_calibration_domain_sealed_Test_values_opened':False,'model_training_executed':False,
        'R4_R5_or_raw_banks_modified':False,'input_sha256':hashes,'bank_sha256':banks}
    out.mkdir(parents=True)
    for name,frame in [('background_manifest.parquet',active),('reference_manifest.parquet',reference),('masked_minute_rows.parquet',masked),('excluded_background_manifest.parquet',excluded)]:frame.to_parquet(out/name,index=False,compression='zstd')
    manifest={'version':VERSION,'ready_to_materialize':True,'ready_to_train':False,'seed':int(seed),
        'parent_source_dir':relative(root,source),'parent_source_hash':info['source_hash'],'background_bank':raw_bank,'banks':banks,
        'roles':list(ROLES),'diagnostic_roles':sorted(DIAGNOSTIC),
        'schema':{'background_index':'base_window_index is original global raw index','minute_mask':'boundary_mask_bits bit m masks actual minute m; preserve original NaN',
            'reference_slice':'segment_file/start/count are original bank coordinates; expected_segment_slice_sha256 binds slice bytes'},
        'file_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'input_sha256':hashes,
        'code_sha256':{n:sha(root/n) for n in ['pipeline/source_split_r6.py','ops/build_source_split_r6.py','pipeline/prepare_sa.py']},
        'audit':report,'forbidden_old_source_values_used':False,'model_training_executed':False}
    manifest['source_hash']=canonical_sha(manifest)
    for n,h in hashes.items():
        if sha(root/n)!=h:raise ValueError('Frozen input changed during build: '+n)
    for n,h in banks.items():
        if sha(root/n)!=h:raise ValueError('Frozen numeric bank changed during build: '+n)
    write_new(out/'manifest.json',manifest);prefix=resolve(root,report_prefix);write_new(prefix.with_suffix('.json'),report)
    md=['# R6 source split / minute QC audit','',f'Seed `{seed}`; metadata-hash CROSS_SENSOR sensor_group `{held}`.',
        f'허용 raw windows {len(allowed):,}; active {len(active):,}; QC/추가purge 제외 {len(excluded):,}.',
        f'실제 선택 원 row의 boundary minute {len(masked):,}개를 가렸습니다. 전체 첫/끝 분 일괄 가림이 아닙니다.',
        f'같은 센서의 타 역할 support gap 최소: {report["minimum_same_sensor_cross_role_support_gap_hours"]} h.',
        'CROSS_SENSOR/CROSS_GENERATOR는 잠긴 진단 역할입니다. 후보의 물리적 고장 원인을 확정하지 않았습니다.','',
        '## Site retention','', '```json',json.dumps(sites,ensure_ascii=False,indent=2),'```','',
        '## Reference roles','', '```json',json.dumps(report['reference_role_counts'],ensure_ascii=False,indent=2),'```','',
        '기존 OUTER/calibration/domain/sealed/Test 값은 열지 않았고 모델 학습·R4/R5 수정은 하지 않았습니다.']
    with prefix.with_suffix('.md').open('x') as f:f.write('\n'.join(md)+'\n')
    return manifest

class SourceSplitR6:
    def __init__(self,version_dir,*,root=ROOT,verify_banks=True):
        self.root=Path(root).resolve();self.folder=resolve(self.root,version_dir).resolve();m=json.loads((self.folder/'manifest.json').read_text());self.manifest=m
        if m.get('version')!=VERSION or m.get('ready_to_materialize') is not True or m.get('forbidden_old_source_values_used') is not False:raise ValueError('Verified R6 source version required')
        if canonical_sha({k:v for k,v in m.items() if k!='source_hash'})!=m['source_hash']:raise ValueError('R6 manifest payload changed')
        for n,h in m['file_sha256'].items():
            if sha(self.folder/n)!=h:raise ValueError('R6 metadata changed')
        for n,h in m['input_sha256'].items():
            if sha(self.root/n)!=h:raise ValueError('Parent metadata changed')
        for n,h in m['code_sha256'].items():
            if sha(self.root/n)!=h:raise ValueError('R6 source implementation changed: '+n)
        if verify_banks:
            for n,h in m['banks'].items():
                if sha(self.root/n)!=h:raise ValueError('Raw bank changed')
        self.background_manifest=pd.read_parquet(self.folder/'background_manifest.parquet');self.reference_manifest=pd.read_parquet(self.folder/'reference_manifest.parquet')
    def _role(self,role,allow_diagnostic):
        if role not in ROLES:raise ValueError('Only declared R6 roles allowed')
        if role in DIAGNOSTIC and not allow_diagnostic:raise ValueError('Diagnostic source panel locked; explicit allow_diagnostic required')
    def pick_indices(self,role,n,seed=None,*,site_balanced=False,allow_diagnostic=False):
        self._role(role,allow_diagnostic)
        if type(n) is not int or n<1:raise ValueError('Positive draw count required')
        frame=self.background_manifest[self.background_manifest.split.eq(role)];rng=np.random.default_rng(self.manifest['seed'] if seed is None else seed)
        if not site_balanced:return rng.choice(frame.base_window_index.to_numpy(),n,replace=n>len(frame))
        sites=sorted(frame.site.unique());pieces=[]
        for i,site in enumerate(sites):
            count=n//len(sites)+int(i<n%len(sites));pool=frame.loc[frame.site.eq(site),'base_window_index'].to_numpy()
            if count:pieces.extend(np.concatenate([rng.permutation(pool) for _ in range(math.ceil(count/len(pool)))])[:count])
        return np.asarray(pieces,dtype=np.int64)[rng.permutation(n)]
    def load_backgrounds(self,role,indices=None,*,allow_diagnostic=False):
        self._role(role,allow_diagnostic);frame=self.background_manifest[self.background_manifest.split.eq(role)]
        ids=frame.base_window_index.to_numpy(int) if indices is None else np.asarray(indices)
        if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or not len(ids) or not set(ids)<=set(frame.base_window_index):raise ValueError('Only this role admitted indices may be read')
        meta=frame.set_index('base_window_index',drop=False).loc[ids].reset_index(drop=True)
        raw=np.load(self.root/self.manifest['background_bank'],mmap_mode='r',allow_pickle=False)
        x=apply_minute_qc(np.asarray(raw[ids]),meta.boundary_mask_bits.to_numpy())
        if (np.isfinite(x).any(2).sum(1)<3).any():raise ValueError('R6 QC support changed')
        return x,meta
    def load_references(self,role,*,allow_diagnostic=False):
        self._role(role,allow_diagnostic);frame=self.reference_manifest[self.reference_manifest.split.eq(role)];result=[];banks={}
        for item in frame.to_dict('records'):
            path=item['segment_file']
            if path not in self.manifest['banks']:raise ValueError('Unbound reference bank')
            if path not in banks:banks[path]=np.load(self.root/path,mmap_mode='r',allow_pickle=False)
            start=int(item['segment_start']);count=int(item['segment_count']);s=banks[path][start:start+count]
            if len(s)!=count or hashlib.sha256(np.ascontiguousarray(s).tobytes()).hexdigest()!=item['expected_segment_slice_sha256']:raise ValueError('Admitted reference slice changed')
            s.setflags(write=False);result.append((item,s))
        return result

def source_split_fixture():
    """Synthetic metadata-only join/purge checks; no real source values read."""
    start=pd.Timestamp('2026-01-01T00:00:00')
    a=pd.DataFrame({'base_window_index':[9],'site':['fixture'],'recorder':['r'],
        'start_export_clock':[start]})
    # Only selected row 2 at minute 7 must be masked. Flagged row 3 is an
    # unselected duplicate at minute 1 and must not erase selected row 1.
    rows=pd.DataFrame({'row_id':[1,2,3],'site':['fixture']*3,'recorder':['r']*3,
        't':[start+pd.Timedelta(minutes=1),start+pd.Timedelta(minutes=7),start+pd.Timedelta(minutes=1,seconds=2)],
        'window_included':[True,True,False],'eligible_for_minute_window':[True]*3})
    flags=rows.iloc[[1,2]][['row_id','site','recorder','t']].copy();flags['candidate_reason']='fixture'
    bits,masked,counts=minute_masks(a,rows,flags)
    x=np.full((1,15,25),np.nan,dtype=np.float32);x[:,[1,7]]=3
    original=x.copy();y=apply_minute_qc(x,bits)
    assert bits.tolist()==[1<<7] and masked.row_id.tolist()==[2] and counts=={9:2}
    assert np.isfinite(y[0,1]).all() and np.isnan(y[0,7]).all()
    assert np.array_equal(x,original,equal_nan=True) and np.isnan(y)[np.isnan(x)].all()
    gapframe=pd.DataFrame({'sensor_group':[0,0,1],'split':['train','inner','cross_sensor'],
        'start_export_clock':[start,start+pd.Timedelta(hours=24,minutes=15),start]})
    g=role_gaps(gapframe);assert g[0]==24 and g[1]==24 and np.isinf(g[2])
    try:apply_minute_qc(x,np.array([1<<15],dtype=np.uint16))
    except ValueError:pass
    else:raise AssertionError('Out-of-window mask accepted')
    bare=object.__new__(SourceSplitR6)
    try:bare._role('cross_sensor',False)
    except ValueError:pass
    else:raise AssertionError('Diagnostic panel silently unlocked')
    return {'status':'PASS','synthetic_only':True,'actual_interior_minute_mask':True,
        'unselected_same_minute_flag_ignored':True,'NaN_and_input_preserved':True,
        'support_gap_24h_not_start_gap':True,'diagnostic_explicit_guard':True}
