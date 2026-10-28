"""Source-first, nested five-fold data packages; no model training.

Only the existing train+validation source pool is admitted. Calibration,
domain holdout, challenge Test and the new sealed source are not fold inputs.
Each source unit has exactly one outer fold. An adjacent source fold is the
inner early-stopping set; the other three are training sources.
"""
from pathlib import Path
import argparse,json,hashlib,os
import numpy as np
import pandas as pd
from pipeline.build_quality_v2 import sha256,nearest_other_gap_seconds
from pipeline.materialize_quality_v2 import materialize,make_sampler
from pipeline.native_sampler_v2 import AugmentationConfig

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='source_group_nested5_week24h_v1'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str,separators=(',',':')).encode()).hexdigest()

def connected_references(ref,old):
    r=ref[ref.reference_eligible & ref.split.isin(['train','validation'])].copy().reset_index(drop=True)
    r=r.merge(old[['source','clip_id','utc_begin','utc_end']],on=['source','clip_id'],how='left')
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def join(a,b):parent[find(a)]=find(b)
    for i,q in r.iterrows():
        join(('row',i),('group',q.source,q.group_key))
        join(('row',i),('bytes',q.sha256))
        # VT recordings share one receiver: conservatively connect all clips
        # from a recording day as well as every recurrence of a vessel hull.
        if q.source=='VTUAD_v1.3.2':join(('row',i),('day',q.source,q.recording_day))
    # Actual interval overlap is checked explicitly, including midnight.
    vt=r[r.source=='VTUAD_v1.3.2'].copy()
    vt['begin']=pd.to_datetime(vt.utc_begin,utc=True);vt['end']=pd.to_datetime(vt.utc_end,utc=True)
    previous=[]
    for i,q in vt.sort_values('begin').iterrows():
        previous=[(j,end) for j,end in previous if end>q.begin]
        for j,end in previous:join(('row',i),('row',j))
        previous.append((i,q.end))
    roots={i:find(('row',i)) for i in r.index}
    members={root:sorted(r.loc[[i for i,v in roots.items() if v==root],'sha256'].tolist()) for root in set(roots.values())}
    r['reference_component']=['REF_'+digest(members[roots[i]])[:20] for i in r.index]
    r['original_group_key']=r.group_key;r['group_key']=r.reference_component
    return r

def allocate_reference_folds(r,k,seed):
    units=r.groupby('reference_component').agg(clips=('clip_id','size'),
        native=('native_long','sum'),source=('source','first')).reset_index()
    natives=units[units.native>0].copy()
    if len(natives)<k:raise ValueError(f'{k} folds need at least {k} independent native reference components; found {len(natives)}')
    assigned={};loads=np.zeros(k,int)
    for n,q in enumerate(natives.sort_values('reference_component').itertuples()):
        f=n%k;assigned[q.reference_component]=f;loads[f]+=q.clips
    # Corpus-specific balance does not split any connected component.
    for source,g in units[units.native==0].groupby('source',sort=True):
        corpus=np.zeros(k,int)
        for q in g.sort_values(['clips','reference_component'],ascending=[False,True]).itertuples():
            smallest=np.flatnonzero(corpus==corpus.min())
            f=int(smallest[int(digest([seed,q.reference_component])[:8],16)%len(smallest)])
            assigned[q.reference_component]=f;corpus[f]+=q.clips;loads[f]+=q.clips
    r['outer_fold']=r.reference_component.map(assigned).astype('int8')
    assert r.groupby('reference_component').outer_fold.nunique().max()==1
    assert r.groupby('sha256').outer_fold.nunique().max()==1
    return r,units.assign(outer_fold=units.reference_component.map(assigned))

def allocate_background_folds(frame,k,seed):
    b=frame[frame.split.isin(['train','validation'])].copy().reset_index(drop=True)
    b['original_split']=b.split
    b['background_group']=['BG_'+digest([int(g),week])[:20] for g,week in zip(b.sensor_group,b.recording_week_start)]
    mapping={}
    for sensor,g in b.groupby('sensor_group',sort=True):
        units=g.groupby('background_group').size().sort_values(ascending=False)
        load=np.zeros(k,int)
        for group,n in units.items():
            choices=np.flatnonzero(load==load.min())
            f=int(choices[int(digest([seed,group])[:8],16)%len(choices)])
            mapping[group]=f;load[f]+=n
    b['outer_fold']=b.background_group.map(mapping).astype('int8')
    assert b.groupby(['sensor_group','recording_week_start']).outer_fold.nunique().max()==1
    return b

def fold_background(b,fold,k,purge_hours):
    result=b.copy();inner=(fold+1)%k
    result['split']=np.where(result.outer_fold==fold,'outer_validation',np.where(result.outer_fold==inner,'validation','train'))
    result['pre_purge_split']=result.split
    seconds=pd.to_datetime(result.start_export_clock).to_numpy(dtype='datetime64[ns]').astype('int64')//10**9
    gap=np.full(len(result),np.inf)
    for _,ix in result.groupby('sensor_group').groups.items():
        ids=np.asarray(ix);labels=result.pre_purge_split.iloc[ids].to_numpy()
        for chosen_label,other_labels in [('train',['validation','outer_validation']),('validation',['outer_validation'])]:
            chosen=ids[labels==chosen_label];other=ids[np.isin(labels,other_labels)]
            gap[chosen]=nearest_other_gap_seconds(seconds[chosen],seconds[other])-900
    result['purge_gap_hours']=np.maximum(gap,0)/3600
    result.loc[result.purge_gap_hours<purge_hours,'split']='purged_temporal_boundary'
    assert (result[result.outer_fold==fold].split=='outer_validation').all()
    minimum=np.inf;active=result[result.split!='purged_temporal_boundary']
    for _,g in active.groupby('sensor_group'):
        sec=pd.to_datetime(g.start_export_clock).to_numpy(dtype='datetime64[ns]').astype('int64')//10**9
        labels=g.split.to_numpy()
        for label in set(labels):
            dist=nearest_other_gap_seconds(sec[labels==label],sec[labels!=label])-900
            if len(dist):minimum=min(minimum,float(dist.min()))
    assert minimum>=purge_hours*3600
    return result,minimum/3600

def sample_metadata(package,version,split,fold):
    b=pd.read_parquet(version/'background_manifest.parquet');r=pd.read_parquet(version/'reference_manifest.parquet')
    groups=dict(zip(zip(r.source,r.clip_id),r.reference_component))
    rec=json.loads((package/f'{split}_recipes.json').read_text())
    items=[item for batch in rec['batches'] for item in batch['items']]
    values=[]
    for j,i in enumerate(items):
        group=b.iloc[i['background_index']].background_group
        refs=sorted({groups[(q['source'],q['clip_id'])] for q in i['references']})
        values.append({'sample_id':f'{PROTOCOL}_f{fold:02d}_{split}_{j:07d}','fold':fold,'validation_id':j,
            'background_group':group,'reference_group':json.dumps(refs),'observed_minutes':i['observed_minutes'],
            'base_background_index':i['base_background_index'],'desired_share':i['desired_share'],
            'is_outer_oof':split=='outer_validation'})
    table=pd.DataFrame(values);table.to_parquet(package/('validation_metadata.parquet' if split=='outer_validation' else 'inner_validation_metadata.parquet'),index=False)
    return table

def main(a):
    cfg=json.loads(a.config.read_text());out=ROOT/cfg['output_dir'];version=ROOT/cfg['source_version_dir']
    if out.exists():raise FileExistsError('Use a fresh CV version path; existing data are immutable')
    source=json.loads((version/'manifest.json').read_text());assert source['ready_to_materialize']
    for name,h in source['frozen_v1_hashes'].items():assert sha256(ROOT/'data/processed'/name)==h
    k=int(cfg['n_splits']);assert k==5
    ref=pd.read_parquet(version/'reference_manifest.parquet')
    # Exclude reserved partitions before reading any further source metadata.
    ref=ref[ref.split.isin(['train','validation'])].copy()
    old=pd.read_parquet(ROOT/'data/processed/external_clip_manifest.parquet')
    old=old[old.split.isin(['train','validation'])].copy()
    refs,units=allocate_reference_folds(connected_references(ref,old),k,cfg['split_seed'])
    b=allocate_background_folds(pd.read_parquet(version/'background_manifest.parquet'),k,cfg['split_seed'])
    out.mkdir(parents=True);b.to_parquet(out/'background_outer_assignment.parquet',index=False)
    refs.to_parquet(out/'reference_outer_assignment.parquet',index=False)
    source_hash=digest({'source_version_manifest_sha256':sha256(version/'manifest.json'),
        'background_assignment_sha256':sha256(out/'background_outer_assignment.parquet'),
        'reference_assignment_sha256':sha256(out/'reference_outer_assignment.parquet'),'config':cfg})
    cv={'protocol':PROTOCOL,'ready_to_train':False,'n_splits':k,'source_hash':source_hash,'folds':[],
        'nested_early_stopping':True,'outer_validation_used_for_early_stopping':False,
        'source_pool':'prior train + validation only; calibration/domain/sealed excluded',
        'challenge_test_accessed':False,'sealed_arrays_accessed':False,'model_training_executed':False}
    (out/'cv_manifest.json').write_text(json.dumps(cv,indent=2))
    audit={'protocol':PROTOCOL,'native_components':units[units.native>0].to_dict('records'),
        'reference_components':len(units),'reference_clips':len(refs),'background_windows':len(b),
        'background_groups':b.background_group.nunique(),'outer_source_assignment_once':True,
        'reference_hull_day_overlap_bytes_connected':True,'folds':[],
        'limitations':['Sensor-week CV measures temporal generalization within known sensors, not fully unseen-sensor transfer.',
            'Exactly five independent native components: four NOAA stations and one connected VT recording component.',
            'Native corpus differs across outer folds; report fold dispersion and corpus-specific errors.',
            'HearMyShip recurring hull identity is unavailable; day separation cannot prove hull independence.',
            'OOF covers synthetic draws from disjoint source units; it does not supply true labels for natural SA observations.',
            'No blanket run-edge removal or nominal sample-rate correction has been applied without verified provenance.']}
    validation_ids=set();outer_background_groups=set();outer_reference_groups=set()
    for fold in range(k):
        path=out/f'fold_{fold:02d}';path.mkdir();v=path/'sources';v.mkdir()
        fb,min_gap=fold_background(b,fold,k,cfg['purge_hours']);fr=refs.copy()
        fr['split']=np.where(fr.outer_fold==fold,'outer_validation',np.where(fr.outer_fold==(fold+1)%k,'validation','train'))
        assert fr.groupby('reference_component').split.nunique().max()==1
        assert fr.groupby('sha256').split.nunique().max()==1
        native=fr[fr.native_long].groupby('split').reference_component.nunique().to_dict()
        assert all(native.get(s,0)>=1 for s in ['train','validation','outer_validation'])
        fb.to_parquet(v/'background_manifest.parquet',index=False);fr.to_parquet(v/'reference_manifest.parquet',index=False)
        fh=digest({'parent':source_hash,'fold':fold,'background_sha256':sha256(v/'background_manifest.parquet'),
            'reference_sha256':sha256(v/'reference_manifest.parquet')})
        vi={'version':PROTOCOL,'source_hash':fh,'background_manifest':'background_manifest.parquet',
            'reference_manifest':'reference_manifest.parquet','base_background_array':source['base_background_array'],
            'frozen_v1_hashes':source['frozen_v1_hashes'],'ready_to_materialize':True,
            'input_quality_gate_passed':True,'sealed_holdout_locked':True,
            'metadata_sha256':{n:sha256(v/n) for n in ['background_manifest.parquet','reference_manifest.parquet']},
            'parent_source_version_manifest_sha256':sha256(version/'manifest.json')}
        (v/'manifest.json').write_text(json.dumps(vi,indent=2))
        ready={'version':'quality_v2_nested_group_cv_prebuilt','protocol':PROTOCOL,'fold':fold,'ready_to_train':False,'source_hash':fh,'parent_cv_source_hash':source_hash,
            'cv_protocol':PROTOCOL,'validation_protocol':'nested source GroupCV with independent outer OOF',
            'source_version_dir':str(v.relative_to(ROOT)),'source_version_manifest_sha256':sha256(v/'manifest.json'),
            'nested_early_stopping':True,'validation_role':'inner_early_stopping','outer_validation_role':'outer_oof_reporting_only',
            'augmentation_config':vars(AugmentationConfig(**cfg.get('augmentation',{}))),
            'challenge_test_accessed':False,'sealed_arrays_accessed':False,'model_training_executed':False,
            'source_checks_passed':True,'sealed_holdout_locked':True,'sealed_holdout_materialized':False,
            'model_evaluated_on_sealed':False,'existing_sealed_NOAA_accessed':False,'frozen_v1_data_modified':False,
            'reference_group_and_duplicate_isolation_passed':True,'background_week_and_purge_isolation_passed':True,
            'minimum_cross_partition_support_gap_hours':min_gap,'partitions':{}}
        (path/'readiness_manifest.json').write_text(json.dumps(ready,indent=2))
        for split,n,seed in [('train',cfg['n_train'],cfg['generation_seed']+fold*10007),
                             ('validation',cfg['n_inner_validation'],cfg['evaluation_seed']+fold*10007+1009),
                             ('outer_validation',cfg['n_outer_validation'],cfg['evaluation_seed']+fold*10007+2018)]:
            ready['partitions'][split]=materialize(v,path,split,n,seed,AugmentationConfig(**cfg.get('augmentation',{})),cfg['batch_size'])
        os.link(path/'validation.npz',path/'inner_validation.npz')
        inner=sample_metadata(path,v,'validation',fold);outer=sample_metadata(path,v,'outer_validation',fold)
        assert validation_ids.isdisjoint(set(outer.sample_id));validation_ids.update(outer.sample_id)
        bgset=set(fb[fb.split=='outer_validation'].background_group);refset=set(fr[fr.split=='outer_validation'].reference_component)
        assert outer_background_groups.isdisjoint(bgset) and outer_reference_groups.isdisjoint(refset)
        outer_background_groups.update(bgset);outer_reference_groups.update(refset)
        ready['files_sha256']={p.name:sha256(p) for p in path.iterdir() if p.is_file() and p.name not in ['manifest.json','readiness_manifest.json']}
        ready['file_sha256']=ready['files_sha256']
        ready['data_package_hash']=hashlib.sha256(json.dumps(ready['file_sha256'],sort_keys=True).encode()).hexdigest()
        ready['generator_code_sha256']={n:sha256(ROOT/n) for n in ['pipeline/group_cv.py','pipeline/native_sampler_v2.py',
            'pipeline/augmentation_v2.py','pipeline/materialize_quality_v2.py']}
        ready['requires_fresh_task_fit']=True;ready['ready_to_train']=True
        for name in ['manifest.json','readiness_manifest.json']:(path/name).write_text(json.dumps(ready,indent=2))
        cv['folds'].append({'fold':fold,'data_dir':str(path.relative_to(ROOT)),
            'validation_examples':cfg['n_outer_validation'],'inner_validation_examples':cfg['n_inner_validation'],
            'outer_validation_file':'outer_validation.npz','validation_metadata_file':'validation_metadata.parquet',
            'outer_validation_metadata_file':'validation_metadata.parquet',
            'data_package_hash':ready['data_package_hash'],'source_hash':fh})
        audit['folds'].append({'fold':fold,'background_counts':fb.split.value_counts().to_dict(),
            'reference_counts':fr.groupby(['split','source']).agg(clips=('clip_id','size'),
                components=('reference_component','nunique'),original_groups=('original_group_key','nunique'),
                days=('recording_day','nunique'),native_clips=('native_long','sum')).reset_index().to_dict('records'),
            'background_source_fraction_including_purged':{str(label):float(count/len(fb)) for label,count in fb.split.value_counts().items()},
            'background_groups_by_partition':fb[fb.split!='purged_temporal_boundary'].groupby('split').background_group.nunique().to_dict(),
            'reference_components_by_partition':fr.groupby('split').reference_component.nunique().to_dict(),
            'native_connected_components':native,'minimum_gap_hours':min_gap,'ready_to_train':True})
        (out/'cv_manifest.json').write_text(json.dumps(cv,indent=2))
    assert len(outer_background_groups)==b.background_group.nunique()
    assert len(outer_reference_groups)==refs.reference_component.nunique()
    for name,h in source['frozen_v1_hashes'].items():assert sha256(ROOT/'data/processed'/name)==h
    cv['ready_to_train']=True;cv['outer_oof_examples']=len(validation_ids)
    audit['ready_to_train']=True;audit['source_hash']=source_hash;audit['outer_oof_examples']=len(validation_ids)
    (out/'cv_manifest.json').write_text(json.dumps(cv,indent=2))
    (ROOT/'reports/group_cv_data_audit.json').write_text(json.dumps(audit,indent=2))
    print('READY',out,source_hash,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/group_cv_data.json');main(p.parse_args())
