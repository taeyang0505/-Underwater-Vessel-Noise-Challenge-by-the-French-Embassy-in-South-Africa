"""Fresh epoch views from frozen R4 TRAIN sources, never from held-out arrays.

Every epoch preserves R4's physical recipe: whole-source bounded monotone
power timewarp, declared fade/EQ, exact zero count, and independent full-split
LHS target/rendered-duration schedules. Only seeds and the clean background
permutation change. Known q/y/presence remain supervision; none is p_ship or
a query input. Any offline teacher p_ship must come from model inference with
teacher provenance, never an alias for these known controlled-addition labels.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from pipeline.passage_sampler_r4 import (PassageReference,PassageConfigR4,PassageSamplerR4,
    make_passage_schedule,schedule_slice)
from pipeline.grid_inputs_r5 import (binding_from_fold,validate_shared_prior,shared_grid_inputs,
    file_sha256,canonical_sha256)

ROOT=Path(__file__).resolve().parents[1]

def _resolve(root,path):
    p=Path(path);return p.resolve() if p.is_absolute() else (root/p).resolve()

class DynamicPassageEpochSource:
    def __init__(self,fold_dir,seed=42,*,prior=None,expected_binding=None,
                 expected_prior_sha256=None,project_root=ROOT,reference_version_dir=None,parent_source_dir=None):
        self.root=Path(project_root).resolve();self.folder=_resolve(self.root,fold_dir)
        if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:raise ValueError('Nonnegative integer seed required')
        self.seed=int(seed);self.gate=json.loads((self.folder/'readiness_manifest.json').read_text())
        g=self.gate
        if g.get('version')!='quality_v4_vtuad_passage_group_cv_prebuilt' or g.get('ready_to_train') is not True:raise ValueError('Frozen completed R4 fold required')
        for flag in ['source_checks_passed','reference_group_and_duplicate_isolation_passed','background_week_and_purge_isolation_passed','sealed_holdout_locked']:
            if g.get(flag) is not True:raise ValueError('Source gate failed: '+flag)
        for flag in ['challenge_test_accessed','sealed_arrays_accessed','calibration_accessed','model_evaluated_on_sealed','frozen_v1_data_modified']:
            if g.get(flag) is not False:raise ValueError('Unsafe or missing source flag: '+flag)
        self.binding=binding_from_fold(self.folder,self.seed,project_root=self.root)
        if expected_binding is not None:
            for k,v in expected_binding.items():
                if self.binding.get(k)!=v:raise ValueError('Dynamic source binding mismatch: '+k)
        self.fold=self.binding['fold'];self.source=_resolve(self.root,g['source_version_dir'])
        if file_sha256(self.source/'manifest.json')!=g['source_version_manifest_sha256']:raise ValueError('Source manifest changed')
        self.source_info=json.loads((self.source/'manifest.json').read_text())
        if self.source_info['source_hash']!=g['source_hash']:raise ValueError('Source logical identity differs from readiness binding')
        for name,h in self.source_info['metadata_sha256'].items():
            if file_sha256(self.source/name)!=h:raise ValueError('Source metadata changed: '+name)
        self.background_manifest=pd.read_parquet(self.source/'background_manifest.parquet').reset_index(drop=True)
        table=pd.read_parquet(self.source/'reference_manifest.parquet')
        if not table.source.eq('VTUAD_v1.3.2').all() or table.groupby('reference_component').split.nunique().max()!=1:raise ValueError('Unexpected reference source/isolation')
        if 'original_split' in table and not table.original_split.isin(['train','validation']).all():raise ValueError('Reserved original reference admitted')
        self.reference_manifest=table[table.split.eq('train')].copy()
        if not len(self.reference_manifest):raise ValueError('No TRAIN references')
        order_path=self.folder/'training_background_order.npy'
        if file_sha256(order_path)!=g['file_sha256']['training_background_order.npy']:raise ValueError('Clean primary order changed')
        self.clean_order=np.load(order_path,allow_pickle=False)
        if self.clean_order.ndim!=1 or not np.issubdtype(self.clean_order.dtype,np.integer):raise ValueError('Integer clean primary indices required')
        wanted=np.flatnonzero(self.background_manifest.split.eq('train').to_numpy())
        if len(self.clean_order)!=len(wanted) or set(self.clean_order)!=set(wanted) or len(np.unique(self.clean_order))!=len(wanted):raise ValueError('Clean TRAIN coverage is not exactly once')
        if self.background_manifest.iloc[self.clean_order].boundary_candidate_flag.any():raise ValueError('Boundary-candidate TRAIN input')
        if len(wanted)!=self.binding['train_examples']:raise ValueError('TRAIN source count differs from static package')
        self.config=PassageConfigR4(**g['passage_config']);self.bank_sha256={};self._bank_stats={};self._banks={}
        expected=self.source_info.get('numeric_bank_sha256')
        if expected is None:
            # R4 metadata binds its parent background metadata and its frozen
            # VT source manifest. These supply expected numeric-bank hashes.
            parent=_resolve(self.root,parent_source_dir or f'data/synthetic/group_cv_physics_v2_r1/fold_{self.fold:02d}/sources')
            old=json.loads((parent/'manifest.json').read_text())
            if file_sha256(parent/'background_manifest.parquet')!=self.source_info['parent_background_manifest_sha256']:raise ValueError('Parent background identity changed')
            rv=_resolve(self.root,reference_version_dir or 'data/versions/vtuad_reference_r4_v1')
            if file_sha256(rv/'manifest.json')!=self.source_info['reference_version_manifest_sha256']:raise ValueError('VT reference version changed')
            ref_info=json.loads((rv/'manifest.json').read_text())
            bg_name=self.source_info['base_background_array'];expected={bg_name:old['frozen_v1_hashes'][Path(bg_name).name]}
            for name in self.reference_manifest.segment_file.unique():
                h=old['frozen_v1_hashes'].get(Path(name).name)
                if not h or h!=ref_info['bank_sha256_from_frozen_R2_source_manifest']:raise ValueError('Unknown numerical reference-bank hash')
                expected[name]=h
        paths={self.source_info['base_background_array'],*self.reference_manifest.segment_file.unique()}
        for name in sorted(paths):
            if name not in expected:raise ValueError('Missing numeric bank SHA: '+name)
            path=_resolve(self.root,name);actual=file_sha256(path)
            if actual!=expected[name]:raise ValueError('Numerical source bank changed: '+name)
            self.bank_sha256[name]=actual;stat=path.stat();self._bank_stats[name]=(stat.st_size,stat.st_mtime_ns,stat.st_ino)
            self._banks[name]=np.load(path,mmap_mode='r',allow_pickle=False)
        self.background_db=self._banks[self.source_info['base_background_array']]
        if self.background_db.ndim!=3 or self.background_db.shape[1:]!=(15,25):raise ValueError('Unexpected background bank shape')
        self.references=[];self.reference_slice_sha256={}
        expected_slices=g['partitions']['train']['reference_segment_slice_sha256']
        for row in self.reference_manifest.itertuples(index=False):
            m=row._asdict();start=int(m['segment_start']);count=int(m['segment_count']);bank=self._banks[m['segment_file']]
            if start<0 or count<1 or start+count>len(bank):raise ValueError('Reference bank slice bounds changed')
            s=bank[start:start+count];h=hashlib.sha256(np.ascontiguousarray(s).tobytes()).hexdigest()
            if expected_slices.get(m['clip_id'])!=h:raise ValueError('TRAIN reference power slice changed')
            r=PassageReference(m,s);r.validate();self.references.append(r);self.reference_slice_sha256[m['clip_id']]=h
        if set(expected_slices)!=set(self.reference_slice_sha256):raise ValueError('TRAIN reference set changed')
        if not g['generator_code_sha256'].get('pipeline/passage_sampler_r4.py'):raise ValueError('Missing frozen R4 sampler identity')
        for name,h in g['generator_code_sha256'].items():
            if file_sha256(self.root/name)!=h:raise ValueError('Frozen R4 generator code changed: '+name)
        self.prior=prior;self.expected_prior_sha256=expected_prior_sha256
        if prior is not None:validate_shared_prior(prior,expected_binding=self.binding,expected_prior_sha256=expected_prior_sha256)
        self.source_summary={'selected_reference_clips':len(table),'selected_retained_components':int(table.reference_component.nunique()),
            'TRAIN_reference_clips':len(self.references),'TRAIN_reference_components':int(self.reference_manifest.reference_component.nunique()),
            'external_target_1500_reached':bool(len(table)>=1500),'model_or_GPU_training_executed':False,
            'inner_outer_arrays_or_recipes_opened':False,'reserved_source_values_sampled':False,
            'numeric_bank_hash_scope':'Integrity-only byte hashes of global frozen banks; only TRAIN slices become examples.'}
        self.identity={'binding':self.binding,'readiness_sha256':file_sha256(self.folder/'readiness_manifest.json'),
            'source_version_manifest_sha256':g['source_version_manifest_sha256'],'numeric_bank_sha256':self.bank_sha256,
            'training_order_sha256':g['file_sha256']['training_background_order.npy'],'reference_slice_sha256':self.reference_slice_sha256}
        self.completed_epochs={}

    @property
    def n_examples(self):return len(self.clean_order)

    def verify_source_integrity(self,*,full=False):
        if file_sha256(self.folder/'readiness_manifest.json')!=self.identity['readiness_sha256']:raise ValueError('Readiness identity changed')
        if file_sha256(self.source/'manifest.json')!=self.identity['source_version_manifest_sha256']:raise ValueError('Source identity changed')
        for name,expected in self.bank_sha256.items():
            path=_resolve(self.root,name);s=path.stat()
            if (s.st_size,s.st_mtime_ns,s.st_ino)!=self._bank_stats[name]:raise ValueError('Read-only source bank changed during streaming')
            if full and file_sha256(path)!=expected:raise ValueError('Source bank SHA mismatch')

    def epoch_plan(self,epoch):
        if isinstance(epoch,(bool,np.bool_)) or int(epoch)!=epoch or epoch<0:raise ValueError('Nonnegative integer epoch required')
        epoch=int(epoch);seeds=np.random.SeedSequence([self.seed,self.fold,epoch,0x5235]).generate_state(3)
        sampler_seed,schedule_seed,permutation_seed=map(int,seeds)
        order=np.random.default_rng(permutation_seed).permutation(self.clean_order)
        schedule=make_passage_schedule(len(order),schedule_seed,self.config)
        h=hashlib.sha256()
        for v in [order,schedule['requested_share'],schedule['rendered_segments']]:h.update(np.ascontiguousarray(v).tobytes())
        for v in [order,schedule['requested_share'],schedule['rendered_segments']]:v.setflags(write=False)
        return {'epoch':epoch,'sampler_seed':sampler_seed,'schedule_seed':schedule_seed,'permutation_seed':permutation_seed,
            'background_order':order,'schedule':schedule,'plan_sha256':h.hexdigest(),
            'primary_examples':len(order),'primary_unique':len(np.unique(order))}

    def iter_batches(self,epoch,batchsize,*,with_grid=False):
        if isinstance(batchsize,(bool,np.bool_)) or int(batchsize)!=batchsize or batchsize<1:raise ValueError('Positive integer batch size required')
        if with_grid and self.prior is None:raise ValueError('A bound TRAIN-fitted shared prior is required for grid8')
        self.verify_source_integrity();plan=self.epoch_plan(epoch);order=plan['background_order']
        sampler=PassageSamplerR4(self.background_db,self.background_manifest,self.references,'train',plan['sampler_seed'],self.config)
        seen=0
        for begin in range(0,len(order),int(batchsize)):
            self.verify_source_integrity();indices=order[begin:begin+int(batchsize)]
            values=sampler.draw(len(indices),indices,schedule_slice(plan['schedule'],begin,begin+len(indices)))
            batch=dict(zip(['db','share','presence','cell_share'],values[:4]));batch['background_index']=indices.copy()
            if with_grid:
                grid,w=shared_grid_inputs(batch['db'],self.prior,expected_binding=self.binding,expected_prior_sha256=self.expected_prior_sha256)
                batch['grid8']=grid;batch['energy_weights']=w
            seen+=len(indices);yield batch
        self.verify_source_integrity()
        self.completed_epochs[int(epoch)]={'epoch':int(epoch),'examples':seen,'clean_primary_once':seen==len(self.clean_order),
            'plan_sha256':plan['plan_sha256'],'schedule_seed':plan['schedule_seed'],
            'source_identity_sha256':canonical_sha256(self.identity),'heldout_arrays_opened':False}


def _make_numeric_fixture(tmp):
    """Self-contained fake-data source; no dependency on prior /tmp artifacts."""
    import shutil
    folder=tmp/'data/fold_00';source=folder/'sources';source.mkdir(parents=True)
    code_names=['pipeline/passage_sampler_r4.py','pipeline/build_passage_cv_r4.py','ops/verify_passage_cv_r4.py']
    for name in code_names:
        (tmp/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,tmp/name)
    db=np.stack([np.broadcast_to(np.linspace(30,70,25),(15,25))+.03*i*np.linspace(-1,1,25)[None,:]+np.sin(np.arange(15)[:,None]*(1+i%3)) for i in range(120)]).astype('float32')
    db[[1,2],1]=np.nan;db[[1,2],7]=np.nan;np.save(tmp/'data/bg.npy',db)
    b=pd.DataFrame({'base_window_index':np.arange(120),'background_group':['BG'+str(i) for i in range(120)],
        'sensor_group':0,'site':'numeric_fixture','boundary_candidate_flag':False,
        'split':np.where(np.arange(120)%5==0,'outer_validation',np.where(np.arange(120)%5==1,'validation','train'))})
    powers=[];rows=[];offset=0
    for group in range(5):
        for j,k in enumerate([60,180]):
            s=(np.linspace(.8,1.2,k)[:,None]*np.linspace(.5,1.5,25)[None,:]).astype('float32');powers.append(s)
            start=pd.Timestamp('2016-01-01',tz='UTC')+pd.Timedelta(days=group*7,seconds=j*10)
            rows.append({'source':'VTUAD_v1.3.2','clip_id':f'numeric_C{group}_{j}',
                'reference_component':f'G{group}','group_key':f'G{group}',
                'split':'outer_validation' if group==0 else 'validation' if group==1 else 'train',
                'original_split':'train','role':'vessel_passage','original_group_key':f'H{group}','hull_id':f'H{group}',
                'recording_day':str(start.date()),'utc_begin':start.isoformat(),'utc_end':(start+pd.Timedelta(seconds=2*k)).isoformat(),
                'sha256':hashlib.sha256(f'numeric_fixture_{group}_{j}'.encode()).hexdigest(),
                'segment_file':'data/ref.npy','segment_start':offset,'segment_count':k,'duration_seconds':2*k,
                'reference_eligible':True,'verified':True,'measured':True,'quality_full_band_good':True,'sealed':False})
            offset+=k
    np.save(tmp/'data/ref.npy',np.concatenate(powers));r=pd.DataFrame(rows)
    b.to_parquet(source/'background_manifest.parquet',index=False);r.to_parquet(source/'reference_manifest.parquet',index=False)
    bank=np.load(tmp/'data/ref.npy',mmap_mode='r');cfg=PassageConfigR4();parts={}
    for role,split in enumerate(['train','validation','outer_validation']):
        refs=[];slice_sha={}
        for row in r[r.split.eq(split)].itertuples(index=False):
            m=row._asdict();s=bank[m['segment_start']:m['segment_start']+m['segment_count']]
            refs.append(PassageReference(m,s));slice_sha[m['clip_id']]=hashlib.sha256(np.ascontiguousarray(s).tobytes()).hexdigest()
        ids=np.flatnonzero(b.split.eq(split));schedule=make_passage_schedule(len(ids),1000+role,cfg)
        values=PassageSamplerR4(db,b,refs,split,100+role,cfg).draw(len(ids),ids,schedule)
        np.savez_compressed(folder/(split+'.npz'),**dict(zip(['db','share','presence','cell_share'],values[:4])))
        parts[split]={'examples':len(ids),'reference_segment_slice_sha256':slice_sha}
    np.save(folder/'training_background_order.npy',np.flatnonzero(b.split.eq('train')))
    numeric={f'data/{name}':file_sha256(tmp/'data'/name) for name in ['bg.npy','ref.npy']}
    source_info={'version':'quality_v4_vtuad_passage_group_cv_prebuilt','source_hash':canonical_sha256({'numeric_fixture_banks':numeric}),
        'base_background_array':'data/bg.npy','numeric_bank_sha256':numeric,
        'metadata_sha256':{name:file_sha256(source/name) for name in ['background_manifest.parquet','reference_manifest.parquet']}}
    (source/'manifest.json').write_text(json.dumps(source_info,indent=2))
    files={p.name:file_sha256(p) for p in folder.iterdir() if p.is_file()}
    g={'version':'quality_v4_vtuad_passage_group_cv_prebuilt','protocol':'source_group_nested5_vtuad_passage_r4','fold':0,'ready_to_train':True,
        'source_hash':source_info['source_hash'],'source_version_dir':str(source),'source_version_manifest_sha256':file_sha256(source/'manifest.json'),
        'file_sha256':files,'data_package_hash':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
        'partitions':parts,'passage_config':cfg.to_dict(),'generator_code_sha256':{n:file_sha256(tmp/n) for n in code_names},
        'source_checks_passed':True,'reference_group_and_duplicate_isolation_passed':True,'background_week_and_purge_isolation_passed':True,
        'sealed_holdout_locked':True,'challenge_test_accessed':False,'sealed_arrays_accessed':False,'calibration_accessed':False,
        'model_evaluated_on_sealed':False,'frozen_v1_data_modified':False,'synthetic_numeric_fixture_only':True}
    for name in ['readiness_manifest.json','manifest.json']:(folder/name).write_text(json.dumps(g,indent=2))
    return folder,source,source_info,g


def run_cpu_fixtures(actual_fold=None):
    """Pure numeric fixture plus optional small actual-TRAIN-only smoke.

    No neural/teacher fitting, GPU, held-out NPZ/recipe loading, or production
    full prior is performed. The optional small TRAIN prior has a distinct
    subset NPZ/package identity and is never a production prior.
    """
    import tempfile,copy,itertools
    from pipeline.grid_inputs_r5 import fit_shared_prior,save_shared_prior,load_shared_prior
    report_path=ROOT/'reports/R5_SHARED_INPUT_DYNAMIC_FIXTURE.json'
    tmp=Path(tempfile.mkdtemp(prefix='r5_shared_dynamic_'))
    folder,source,si,g=_make_numeric_fixture(tmp)
    with np.load(folder/'train.npz',allow_pickle=False) as z:train={k:z[k] for k in ['db','share','presence','cell_share']}
    binding=binding_from_fold(folder,42,project_root=tmp);prior=fit_shared_prior(train,binding)
    prior_file=tmp/'shared_background_prior.json';saved=save_shared_prior(prior_file,prior,expected_binding=binding)
    loaded=load_shared_prior(prior_file,expected_binding=binding,expected_file_sha256=saved['file_sha256'],expected_prior_sha256=saved['payload_sha256'])
    assert prior==loaded
    grid,w=shared_grid_inputs(train['db'],loaded,expected_binding=binding)
    for _ in range(4):
        other,ww=shared_grid_inputs(train['db'],loaded,expected_binding=binding);assert np.array_equal(grid,other) and np.array_equal(w,ww)
    guard_results={}
    def rejected(name,call):
        try:call()
        except (ValueError,TypeError,FileExistsError):guard_results[name]=True
        else:raise AssertionError('Guard failed: '+name)
    rejected('heldout_prior_fit',lambda:fit_shared_prior(train,binding,split='validation'))
    rejected('query_label_argument',lambda:shared_grid_inputs(train['db'],prior,expected_binding=binding,cell_share=train['cell_share']))
    rejected('query_labeled_dictionary',lambda:shared_grid_inputs(train,prior,expected_binding=binding))
    rejected('wrong_fold_binding',lambda:shared_grid_inputs(train['db'],prior,expected_binding={**binding,'fold':1}))
    altered=copy.deepcopy(prior);altered['centres_power_shape'][0][0]*=1.01
    rejected('payload_tampering',lambda:shared_grid_inputs(train['db'],altered,expected_binding=binding))
    rejected('wrong_file_sha',lambda:load_shared_prior(prior_file,expected_binding=binding,expected_file_sha256='0'*64))
    rejected('prior_overwrite',lambda:save_shared_prior(prior_file,prior,expected_binding=binding))
    # Track np.load calls and fail before any held-out partition is opened.
    original_load=np.load;opened=[]
    def train_only_load(file,*args,**kwargs):
        if isinstance(file,(str,Path)):
            name=Path(file).name;opened.append(str(file))
            if name in ['validation.npz','inner_validation.npz','outer_validation.npz','calibration.npz','sealed_holdout.npz','Test.csv']:raise AssertionError('Forbidden partition read')
        return original_load(file,*args,**kwargs)
    np.load=train_only_load
    try:
        dynamic=DynamicPassageEpochSource(folder,42,prior=loaded,expected_binding=binding,expected_prior_sha256=saved['payload_sha256'],project_root=tmp)
        def collect(epoch,batchsize):
            batches=list(dynamic.iter_batches(epoch,batchsize,with_grid=True))
            return {k:np.concatenate([q[k] for q in batches]) for k in batches[0]}
        first=collect(0,16);replay=collect(0,7);different=collect(1,19)
        for k in first:assert np.array_equal(first[k],replay[k],equal_nan=True),k
        assert not np.array_equal(first['db'],different['db'],equal_nan=True)
        assert not np.array_equal(first['background_index'],different['background_index'])
        for output in [first,replay,different]:
            assert len(np.unique(output['background_index']))==dynamic.n_examples
            assert set(output['background_index'])==set(dynamic.clean_order)
            e=np.where(np.isfinite(output['db']),10**((output['db'].astype(float)-np.nanmax(output['db'],(1,2),keepdims=True))/10),0)
            np.testing.assert_allclose((e*output['cell_share']).sum((1,2))/e.sum((1,2)),output['share'],atol=3e-6,rtol=0)
        dynamic.verify_source_integrity(full=True)
        source_text=(source/'manifest.json').read_text();gate_text=(folder/'readiness_manifest.json').read_text()
        wrong=copy.deepcopy(si);wrong['numeric_bank_sha256']['data/ref.npy']='0'*64;(source/'manifest.json').write_text(json.dumps(wrong))
        wrong_gate={**g,'source_version_manifest_sha256':file_sha256(source/'manifest.json')};(folder/'readiness_manifest.json').write_text(json.dumps(wrong_gate))
        rejected('numeric_bank_sha',lambda:DynamicPassageEpochSource(folder,42,project_root=tmp))
        (source/'manifest.json').write_text(source_text);(folder/'readiness_manifest.json').write_text(gate_text)
        actual_report=None
        if actual_fold is not None:
            actual_folder=_resolve(ROOT,actual_fold)
            # Hash TRAIN and source identity before/after the small smoke.
            identity_paths=[actual_folder/'train.npz',actual_folder/'readiness_manifest.json']
            before={str(p):file_sha256(p) for p in identity_paths}
            real=DynamicPassageEpochSource(actual_folder,42)
            with np.load(actual_folder/'train.npz',allow_pickle=False) as z:
                subset={k:z[k][:128].copy() for k in ['db','share','presence','cell_share']}
            subset_file=tmp/'actual_train_subset_smoke.npz';np.savez_compressed(subset_file,**subset)
            subset_binding={**real.binding,'train_examples':len(subset['share']),'train_npz_sha256':file_sha256(subset_file),
                'data_package_hash':canonical_sha256({'parent_package':real.binding['data_package_hash'],'subset_npz':file_sha256(subset_file),'TRAIN_rows':'0:128'})}
            subset_prior=fit_shared_prior(subset,subset_binding)
            def prefix(epoch,bs):
                count=48//bs;chunks=list(itertools.islice(real.iter_batches(epoch,bs),count))
                return {k:np.concatenate([q[k] for q in chunks]) for k in chunks[0]}
            x=prefix(0,16);xx=prefix(0,8);changed=prefix(1,16)
            for k in x:assert np.array_equal(x[k],xx[k],equal_nan=True),k
            assert not np.array_equal(x['db'],changed['db'],equal_nan=True)
            gx,wx=shared_grid_inputs(x['db'],subset_prior,expected_binding=subset_binding)
            gc=[];wc=[]
            for start in range(0,48,7):
                aa,bb=shared_grid_inputs(x['db'][start:start+7],subset_prior,expected_binding=subset_binding);gc.append(aa);wc.append(bb)
            assert np.array_equal(gx,np.concatenate(gc)) and np.array_equal(wx,np.concatenate(wc))
            plan=real.epoch_plan(0);assert plan['primary_unique']==plan['primary_examples']==real.n_examples
            real.verify_source_integrity(full=True)
            assert before=={str(p):file_sha256(p) for p in identity_paths}
            actual_report={'fold':real.fold,'actual_dynamic_rows_per_prefix':48,'TRAIN_rows_for_smoke_prior':128,
                'production_prior_fitted':False,'production_full_epoch_generated':False,
                'smoke_prior_binding_is_separate_subset_npz_package':True,'smoke_prior_requested_clusters':32,
                'smoke_prior_actual_clusters':subset_prior['actual_clusters'],'dynamic_plan_covers_clean_primary_once':real.n_examples,
                'same_epoch_and_chunk_raw_arrays_bitwise':True,'db_only_grid_chunk_bitwise':True,
                'different_epoch_changes_views':True,'source_summary':real.source_summary,
                'source_identity':real.identity,'R4_TRAIN_and_gate_unchanged':True}
    finally:np.load=original_load
    report={'status':'passed R5 shared input and dynamic source CPU fixtures','fixture_root':str(tmp),
        'shared_grid8_identical_for_four_transform_consumers':True,'model_forward_or_trainer_wiring_checked_here':False,
        'grid_shape':list(grid.shape),'prior_requested_clusters':32,
        'prior_actual_clusters':prior['actual_clusters'],'prior_file_and_payload_SHA_reload_passed':True,
        'same_epoch_replay_and_batch_chunk_bitwise':True,'different_epoch_views':True,
        'all_clean_primary_backgrounds_once_per_completed_fixture_epoch':True,
        'completed_fixture_epochs':dynamic.completed_epochs,'guards_rejected':guard_results,
        'opened_np_load_paths_during_dynamic_checks':opened,'inner_outer_arrays_opened':False,'Test_calibration_sealed_accessed':False,
        'actual_TRAIN_small_slice_smoke':actual_report,'neural_teacher_or_GPU_training_executed':False,
        'aggregate_prior_CPU_fits_performed':True,
        'R4_code_or_data_modified':False,'label_policy':loaded['label_policy'],
        'synthetic_fixture_is_self_contained_no_previous_temp_dependency':True,
        'code_sha256':{n:file_sha256(ROOT/n) for n in ['pipeline/grid_inputs_r5.py','pipeline/dynamic_passage_r5.py']}}
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False));return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--verify-fixture',action='store_true');p.add_argument('--actual-fold',type=Path);a=p.parse_args()
    if not a.verify_fixture:p.error('Use --verify-fixture; this module never starts training')
    r=run_cpu_fixtures(a.actual_fold);print(json.dumps({'status':r['status'],'report':'reports/R5_SHARED_INPUT_DYNAMIC_FIXTURE.json','actual_TRAIN_smoke':bool(r['actual_TRAIN_small_slice_smoke'])},indent=2))
