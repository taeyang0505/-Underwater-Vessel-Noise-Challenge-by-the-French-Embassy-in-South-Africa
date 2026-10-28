"""Metadata-only, bridge-excluded VTUAD subsets with auditable direct isolation.

The selected graph is recomputed after actual clip exclusion. It is not a new
physical source corpus, and the full 525-recording graph remains connected.
No acoustic values, Test data, model outputs or training labels are inspected.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import Bounds,LinearConstraint,linear_sum_assignment,milp

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='source_group_nested5_vtuad_passage_r4'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def candidate_pool(root=ROOT):
    source=root/'data/synthetic/group_cv_physics_v2_r1/fold_00/sources/reference_manifest.parquet'
    original=root/'data/processed/external_clip_manifest.parquet'
    r=pd.read_parquet(source)
    b=pd.read_parquet(original)[['source','clip_id','split','utc_begin','utc_end']].rename(columns={'split':'original_split','utc_begin':'provider_utc_begin','utc_end':'provider_utc_end'})
    r=r.merge(b,on=['source','clip_id'],how='left',validate='one_to_one')
    keep=(r.source.eq('VTUAD_v1.3.2')&r.original_split.isin(['train','validation'])&r.reference_eligible&r.measured&r.verified&r.quality_full_band_good&
          r.role.eq('vessel_passage')&~r.weak_class.astype(str).str.lower().isin(['background','unknown','nan'])&
          r.duration_seconds.between(120,1380)&r.segment_count.between(60,690)&r.sample_rate_hz.ge(2*10**3.95))
    r=r[keep].copy().sort_values('clip_id').reset_index(drop=True)
    r['utc_begin']=r.provider_utc_begin;r['utc_end']=r.provider_utc_end
    begin=pd.to_datetime(r.utc_begin,utc=True);end=pd.to_datetime(r.utc_end,utc=True)
    if not (end>begin).all() or not (begin.dt.strftime('%Y-%m-%d')==(end-pd.Timedelta('1ns')).dt.strftime('%Y-%m-%d')).all():
        raise ValueError('This declared day-level selector requires same-day positive intervals')
    if r.clip_id.duplicated().any() or r.sha256.duplicated().any():raise ValueError('Duplicate source recording requires separate exclusion audit')
    return r,{'parent_reference_manifest':str(source.relative_to(root)),'parent_reference_sha256':digest(source),
              'original_external_manifest':str(original.relative_to(root)),'original_external_sha256':digest(original)}

def select_subset(pool,seed=42,minimum_components=20):
    """Maximum day/hull cover, then exact clip/duration retention objectives.

    Each selected hull has exactly one retained day. Other occurrences are
    excluded, not assigned a different label. All native >=900s clips are kept.
    """
    pairs=pool.groupby(['original_group_key','recording_day'],sort=True).agg(
        clips=('clip_id','size'),segments=('segment_count','sum')).reset_index()
    days=sorted(pool.recording_day.unique());hulls=sorted(pool.original_group_key.unique())
    edges=pd.crosstab(pool.recording_day,pool.original_group_key).reindex(index=days,columns=hulls,fill_value=0).to_numpy()
    rows,cols=linear_sum_assignment(np.where(edges>0,0,1_000_000))
    maximum=int(np.sum(edges[rows,cols]>0))
    if maximum<minimum_components:raise ValueError(f'Only {maximum} directly isolated components possible; need {minimum_components}')
    n=len(pairs);size=n+len(days);matrix=[];lower=[];upper=[]
    def add(indices,values,lo,hi):
        row=np.zeros(size);row[np.asarray(indices,dtype=int)]=values
        matrix.append(row);lower.append(lo);upper.append(hi)
    for hull in hulls:add(np.flatnonzero(pairs.original_group_key.eq(hull)),1,0,1)
    for j,day in enumerate(days):
        indices=np.flatnonzero(pairs.recording_day.eq(day))
        add([*indices,n+j],[*np.ones(len(indices)),-1],0,np.inf)
        add([*indices,n+j],[*np.ones(len(indices)),-len(hulls)],-np.inf,0)
    add(range(n,size),1,maximum,maximum)
    native=pool[pool.segment_count.ge(450)]
    for row in native[['original_group_key','recording_day']].drop_duplicates().itertuples(index=False):
        index=np.flatnonzero(pairs.original_group_key.eq(row.original_group_key)&pairs.recording_day.eq(row.recording_day))
        add(index,1,1,1)
    def solve(coef):
        result=milp(coef,integrality=np.ones(size),bounds=Bounds(np.zeros(size),np.ones(size)),
            constraints=LinearConstraint(np.asarray(matrix),np.asarray(lower),np.asarray(upper)),
            options={'time_limit':60.,'mip_rel_gap':0.})
        if not result.success:raise RuntimeError('Metadata retention assignment did not prove optimal: '+result.message)
        return np.rint(result.x).astype(int)
    cost=np.r_[-pairs.clips.to_numpy(dtype=float),np.zeros(len(days))];x=solve(cost)
    retained=int(pairs.clips.to_numpy()@x[:n]);add(range(n),pairs.clips.to_numpy(),retained,retained)
    x=solve(np.r_[-pairs.segments.to_numpy(dtype=float),np.zeros(len(days))])
    segments=int(pairs.segments.to_numpy()@x[:n]);add(range(n),pairs.segments.to_numpy(),segments,segments)
    ties=np.array([int(canonical([seed,str(h),str(d)])[:8],16) for h,d in pairs[['original_group_key','recording_day']].itertuples(index=False,name=None)],dtype=float)
    x=solve(np.r_[ties,np.zeros(len(days))])
    chosen=pairs[x[:n].astype(bool)][['original_group_key','recording_day']]
    selected=pool.merge(chosen,on=['original_group_key','recording_day'],how='inner',validate='many_to_one')
    excluded=pool[~pool.clip_id.isin(selected.clip_id)].copy()
    assignment=dict(zip(chosen.original_group_key,chosen.recording_day))
    excluded['retained_day_for_hull']=excluded.original_group_key.map(assignment)
    excluded['r4_exclusion_reason']='Actual clip excluded to remove cross-day hull bridges and maximize selected direct components'
    if len(selected)!=retained or selected.recording_day.nunique()!=maximum:raise RuntimeError('Retention result mismatch')
    return selected,excluded,{'maximum_matching_upper_bound':maximum,'native_clips_required':len(native),
        'selected_clips':retained,'selected_2s_segments':segments,'available_days':len(days),'available_hulls':len(hulls),
        'selection_objectives':['maximize possible selected components using day/hull matching bound',
            'retain every original >=900s native clip','maximize retained clips','maximize retained measured 2s segments',
            'fixed SHA-based metadata tie break; no acoustic values or model outcomes'],
        'maximum_bound_reason':'Each disjoint selected component contains an edge with a unique day and hull, so component count cannot exceed maximum day-hull matching.'}

def connected_components(selected):
    s=selected.copy().reset_index(drop=True);parent={}
    def find(a):
        parent.setdefault(a,a)
        if parent[a]!=a:parent[a]=find(parent[a])
        return parent[a]
    def join(a,b):parent[find(a)]=find(b)
    for i,row in s.iterrows():
        for key in [('hull',row.original_group_key),('day',row.recording_day),('bytes',row.sha256)]:join(('row',i),key)
    previous=[]
    s['_begin']=pd.to_datetime(s.utc_begin,utc=True);s['_end']=pd.to_datetime(s.utc_end,utc=True)
    for i,row in s.sort_values('_begin').iterrows():
        previous=[(j,end) for j,end in previous if end>row['_begin']]
        for j,end in previous:join(('row',i),('row',j))
        previous.append((i,row['_end']))
    members={}
    for i,row in s.iterrows():members.setdefault(find(('row',i)),[]).append(row.sha256)
    s['parent_reference_component']=s.reference_component;s['parent_outer_fold']=s.outer_fold
    s['reference_component']=['VT_R4_REF_'+canonical(sorted(members[find(('row',i))]))[:20] for i in s.index]
    s['group_key']=s.reference_component;s['hull_id']=s.original_group_key.str.removeprefix('VTUAD_HULL_')
    s['header_sample_rate_hz']=s.sample_rate_hz;s['measured_duration_seconds']=s.duration_seconds
    s['coarse_duration_bin']=np.where(s.duration_seconds<300,'120_to_under300','300_to1380')
    s['fine_duration_bin']=pd.cut(s.duration_seconds,[119.999,299.999,599.999,899.999,1380],labels=['120_to_under300','300_to_under600','600_to_under900','900_to1380']).astype(str)
    return s.drop(columns=['_begin','_end'])

def allocate_folds(selected,seed=42):
    r=selected.copy()
    groups=r.groupby('reference_component').agg(clips=('clip_id','size'),segments=('segment_count','sum'),
        seconds=('duration_seconds','sum'),days=('recording_day','nunique'),hulls=('original_group_key','nunique'),
        short=('duration_seconds',lambda x:int((x<300).sum())),long=('duration_seconds',lambda x:int((x>=300).sum())),
        native=('native_long','sum')).reset_index()
    groups['tie']=[canonical([seed,k]) for k in groups.reference_component]
    both=groups[(groups.short>0)&(groups.long>0)].sort_values(['clips','segments','tie'],ascending=[False,False,True])
    if len(both)<5:raise ValueError('Need at least five mixed-duration components for this declared coarse-bin stratification')
    capacities=np.full(5,len(groups)//5);capacities[:len(groups)%5]+=1
    assigned={};loads=np.zeros(5,int);seconds=np.zeros(5,float);counts=np.zeros(5,int)
    def assign(row,fold):
        assigned[row.reference_component]=fold;loads[fold]+=row.clips;seconds[fold]+=row.seconds;counts[fold]+=1
    for fold,row in enumerate(both.head(5).itertuples()):assign(row,fold)
    remaining=groups[~groups.reference_component.isin(assigned)].sort_values(['clips','segments','tie'],ascending=[False,False,True])
    for row in remaining.itertuples():
        allowed=np.flatnonzero(counts<capacities)
        fold=min(allowed,key=lambda f:(loads[f],seconds[f],counts[f],canonical([seed,row.reference_component,int(f)])))
        assign(row,int(fold))
    r['outer_fold']=r.reference_component.map(assigned).astype('int8');r['split']='pool'
    groups['outer_fold']=groups.reference_component.map(assigned).astype(int)
    for f in range(5):
        if set(r[r.outer_fold.eq(f)].coarse_duration_bin)!={'120_to_under300','300_to1380'}:raise RuntimeError('Coarse-bin coverage failed')
    return r.sort_values(['outer_fold','reference_component','clip_id']).reset_index(drop=True),groups.drop(columns='tie')

def build(output,seed=42,minimum_components=20):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a fresh immutable metadata version')
    pool,inputs=candidate_pool();selected,excluded,proof=select_subset(pool,seed,minimum_components)
    selected=connected_components(selected)
    if selected.reference_component.nunique()!=proof['maximum_matching_upper_bound']:raise RuntimeError('Actual selected graph did not reach matching bound')
    selected,groups=allocate_folds(selected,seed)
    output.mkdir(parents=True)
    selected.to_parquet(output/'reference_manifest.parquet',index=False)
    excluded.to_parquet(output/'excluded_manifest.parquet',index=False)
    groups.to_parquet(output/'component_manifest.parquet',index=False)
    for f in range(5):
        frame=selected.copy();frame['split']=np.where(frame.outer_fold.eq(f),'outer_validation',np.where(frame.outer_fold.eq((f+1)%5),'validation','train'))
        frame.to_parquet(output/f'fold_{f:02d}_reference_manifest.parquet',index=False)
    files={p.name:digest(p) for p in output.glob('*.parquet')}
    source=ROOT/'data/synthetic/group_cv_physics_v2_r1/fold_00/sources/manifest.json'
    frozen=json.loads(source.read_text())['frozen_v1_hashes']
    config={'seed':seed,'minimum_components':minimum_components,'duration_seconds':[120,1380],'segment_count':[60,690],
        'native_retention_required':True,'allowed_original_splits':['train','validation'],'coarse_duration_boundaries_seconds':[120,300,1380],
        'fold_count':5,'inner_fold':'(outer_fold+1)%5','source_only':'VTUAD_v1.3.2'}
    report={'version':'vtuad_reference_r4_v1','protocol':PROTOCOL,'source_hash':canonical({'inputs':inputs,'config':config,'files':files}),
        'ready_to_materialize':True,'metadata_only':True,'dataset_or_model_training_generated':False,
        'inputs':inputs,'config':config,'selection_proof':proof,'file_sha256':files,
        'selected_clips':len(selected),'excluded_allowed_pool_clips':len(excluded),'selected_components':len(groups),
        'measured_native_clips':int(selected.native_long.sum()),'component_outer_counts':groups.outer_fold.value_counts().sort_index().to_dict(),
        'coarse_counts_by_outer':selected.groupby(['outer_fold','coarse_duration_bin']).size().unstack(fill_value=0).to_dict('index'),
        'fine_counts_by_outer':selected.groupby(['outer_fold','fine_duration_bin']).size().unstack(fill_value=0).to_dict('index'),
        'bank_sha256_from_frozen_R2_source_manifest':frozen['external_segments_power.npy'],
        'source_semantics':'AIS scenario-positive measured passages may contain ambient sound; metadata/checksum/QC verification is not human vessel-only ground truth.',
        'independence_scope':'Hull/day/UTC/bytes direct isolation conditional on actually selected clips. Full 525-corpus graph is connected; no new physical source corpus is created.',
        'NOAA_or_HMS_quality_judgment':False,'reserved_sources_admitted':False,'SA_background_assignment_or_purge_modified':False,
        'R2_files_modified':False,'Test_accessed':False,'code_sha256':digest(Path(__file__))}
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'data/versions/vtuad_reference_r4_v1')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--minimum-components',type=int,default=20)
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
