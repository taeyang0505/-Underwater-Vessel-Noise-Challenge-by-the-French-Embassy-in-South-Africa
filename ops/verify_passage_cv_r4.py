"""Independent numeric fixtures and prebuilt R4 data checks; no fitting/Test IO."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def verify_arrays(data):
    x,q,y,z=(data[k] for k in ['db','cell_share','share','presence']);n=len(y)
    assert x.shape==q.shape==(n,15,25) and y.shape==z.shape==(n,) and n>0
    assert all(v.dtype==np.float32 for v in data.values())
    assert not np.isinf(x).any() and all(np.isfinite(v).all() for v in [q,y,z])
    mask=np.isfinite(x);assert (mask.any(2).sum(1)>=3).all()
    assert ((q>=0)&(q<=1)).all() and ((y>=0)&(y<=1)).all() and (q[~mask]==0).all()
    assert np.array_equal(z,(y>0).astype('float32'))
    worst=0.
    for begin in range(0,n,512):
        xx=x[begin:begin+512].astype('float64');o=np.isfinite(xx)
        e=np.where(o,10**((xx-np.nanmax(xx,(1,2),keepdims=True))/10),0)
        recovered=(e*q[begin:begin+512]).sum((1,2))/e.sum((1,2))
        worst=max(worst,float(np.max(np.abs(recovered-y[begin:begin+512]))))
    assert worst<3e-6,('saved energy identity',worst)
    return {'examples':n,'max_saved_energy_target_error':worst,'zero_fraction':float((y==0).mean()),
        'share_mean':float(y.mean()),'natural_observed_minutes_counts':{str(k):int(v) for k,v in pd.Series(mask.any(2).sum(1)).value_counts().sort_index().items()}}

def check_reference_isolation(r):
    active=r[r.split.isin(['train','validation','outer_validation'])]
    assert len(active)==len(r) and active.source.eq('VTUAD_v1.3.2').all()
    assert active.duration_seconds.between(120,1380).all()
    for key in ['reference_component','sha256','recording_day']:
        assert active.groupby(key).split.nunique().max()==1,('reference split leakage',key)
    hull=next((c for c in ['hull_id','hull','vessel_id'] if c in active),None)
    if hull:
        assert active.groupby(hull).split.nunique().max()==1,('hull leakage',hull)
    elif 'original_group_key' in active:
        assert active.groupby('original_group_key').split.nunique().max()==1
    else:raise AssertionError('No hull identity metadata')
    a=active.copy();a['begin']=pd.to_datetime(a.utc_begin,utc=True);a['end']=pd.to_datetime(a.utc_end,utc=True)
    assert a.begin.notna().all() and a.end.notna().all() and (a.end>a.begin).all()
    previous=[]
    for row in a.sort_values('begin').itertuples():
        previous=[p for p in previous if p.end>row.begin]
        assert all(p.split==row.split for p in previous),'Reference interval overlap crosses source split'
        previous.append(row)
    return {'clips':len(r),'components':int(r.reference_component.nunique()),'direct_hull_day_interval_byte_isolation_passed':True}

def matched_distributions(tables,tolerances):
    from scipy.stats import ks_2samp
    base=tables['train'];results={};checks=[]
    def bins(d):
        q=d[d.effective_share>0];counts=q.duration_bin.value_counts().reindex([0,1],fill_value=0).to_numpy(float)
        return counts/counts.sum()
    for split,d in tables.items():
        positive=d[d.effective_share>0]
        result={'share_mean':float(d.effective_share.mean()),'zero_fraction':float(d.effective_share.eq(0).mean()),
            'duration_bin_positive_fractions':bins(d).tolist(),'condition_mean_positive_by_duration_bin':{},
            'observed_minutes_counts':{str(k):int(v) for k,v in d.observed_minutes.value_counts().sort_index().items()},
            'fine_duration_counts':{str(k):int(v) for k,v in d.fine_duration_bin.value_counts().sort_index().items()},
            'support_coverage_quantiles':positive.support_coverage.quantile([0,.1,.5,.9,1]).to_dict(),
            'source_measured_duration_mean_seconds':float(positive.used_source_seconds.mean()),
            'rendered_duration_mean_seconds':float(positive.rendered_duration_seconds.mean()),
            'fine_rendered_duration_positive_counts':{str(k):int(v) for k,v in positive.fine_duration_bin.value_counts().sort_index().items()}}
        for k in [0,1]:
            result['condition_mean_positive_by_duration_bin'][str(k)]=float(d.loc[d.duration_bin.eq(k),'effective_share'].mean())
        if split!='train':
            bp=base[base.effective_share>0]
            f1=positive.fine_duration_bin.value_counts().reindex([0,1,2,3],fill_value=0).to_numpy(float);f1/=f1.sum()
            f0=bp.fine_duration_bin.value_counts().reindex([0,1,2,3],fill_value=0).to_numpy(float);f0/=f0.sum()
            comparisons={'mean_share_difference':abs(result['share_mean']-float(base.effective_share.mean())),
                'zero_fraction_difference':abs(result['zero_fraction']-float(base.effective_share.eq(0).mean())),
                'coarse_duration_total_variation':float(np.abs(bins(d)-bins(base)).sum()/2),
                'fine_duration_total_variation':float(np.abs(f1-f0).sum()/2),
                'rendered_duration_KS_statistic':float(ks_2samp(positive.rendered_duration_seconds,bp.rendered_duration_seconds,method='asymp').statistic),
                'rendered_duration_mean_difference_seconds':abs(result['rendered_duration_mean_seconds']-float(bp.rendered_duration_seconds.mean())),
                'maximum_conditional_mean_difference':max(abs(float(d.loc[d.duration_bin.eq(k),'effective_share'].mean())-float(base.loc[base.duration_bin.eq(k),'effective_share'].mean())) for k in [0,1])}
            for key,value in comparisons.items():assert value<=tolerances[key],(split,key,value,tolerances[key])
            result['comparison_to_train']=comparisons;checks.append(True)
        results[split]=result
    return {'passed':all(checks),'predeclared_tolerances':tolerances,'strata':results,
        'note':'Natural SA coverage and fine duration/source variation are reported separately, not tuned or forced equal.'}

def numeric_fixtures():
    from pipeline.passage_sampler_r4 import PassageConfigR4,PassageReference,PassageSamplerR4,raised_cosine_power_weights,warp_band_power,make_passage_schedule
    weights=raised_cosine_power_weights(60,30)
    # Constant power × a(t)^2 has integral D-5F/4 for two raised-cosine ramps.
    assert abs(weights.sum()*2-(120-1.25*30))<1e-10
    for k,m in [(60,60),(60,180),(180,60),(690,230),(690,450)]:
        s=np.arange(k*25,dtype=float).reshape(k,25)+1;w,a=warp_band_power(s,m)
        np.testing.assert_allclose(w.mean(0),s.mean(0),rtol=1e-12,atol=1e-12)
        assert w.shape==(m,25) and (w>=0).all() and a['source_coordinate_begin']==0 and a['source_coordinate_end']==k
    schedule=make_passage_schedule(3000,20261025);pos=schedule['requested_share']>0
    assert (~pos).sum()==600 and np.array_equal(schedule['rendered_segments'][~pos],np.zeros(600))
    ranks=np.sort((schedule['requested_share'][pos]-.005)/(.95-.005))*pos.sum()-np.arange(pos.sum())
    assert ((ranks>=-1e-10)&(ranks<1+1e-10)).all()
    b=np.tile(np.linspace(30,70,25),(2,15,1)).astype('float32');b[1,[1,7]]=np.nan
    frame=pd.DataFrame({'split':['train']*2,'base_window_index':[0,1],'background_group':['BG0','BG1']})
    references=[]
    for i,k in enumerate([60,180,690]):
        meta={'source':'VTUAD_v1.3.2','split':'train','clip_id':f'fixture{i}','reference_component':f'G{i}',
            'sha256':str(i)*64,'duration_seconds':2*k,'reference_eligible':True,'verified':True,'measured':True,'quality_full_band_good':True}
        power=np.linspace(.8,1.2,k)[:,None]*np.linspace(.5,1.5,25)[None,:];references.append(PassageReference(meta,power))
    cfg=PassageConfigR4();indices=np.tile([0,1],64);outputs=[]
    for split in ['train','validation','outer_validation']:
        ff=frame.assign(split=split);rr=[PassageReference({**r.metadata,'split':split},r.segments) for r in references]
        sampler=PassageSamplerR4(b,ff,rr,split,42,cfg);values=sampler.draw(len(indices),indices)
        data=dict(zip(['db','share','presence','cell_share'],values[:4]));verify_arrays(data)
        assert np.array_equal(np.isfinite(data['db']),np.isfinite(b[indices]))
        assert any(item['desired_share']>.4 and item['references'] and item['references'][0]['measured_used_seconds']<900 for item in values[4]['items'])
        assert any(item['desired_share']==0 for item in values[4]['items'])
        for item in values[4]['items']:
            assert not item['hard_snr_or_support_cap_applied']
            for ref in item['references']:
                assert np.abs(ref['reference_eq_db']).max()<=4+1e-12
                assert not ref['waveform_repetition'] and not ref['dB_interpolation'] and not ref['original_UTC_modified']
                assert 0<=ref['placement_segment']<=450-ref['rendered_segments']
                assert ref['source_segment_begin']==0 and 1/3<=ref['timewarp_ratio']<=3
        outputs.append(values)
    for values in outputs[1:]:
        for x,y in zip(outputs[0][:4],values[:4]):assert np.array_equal(x,y,equal_nan=True)
    replay=PassageSamplerR4(b,frame,references,'train',42,cfg).draw(len(indices),indices)
    assert outputs[0][4]==replay[4]
    for invalid in [60.9,float('nan'),float('inf')]:
        bad={'requested_share':np.array([.5]),'rendered_segments':np.array([invalid]),'schedule_seed':1,'scheme':'invalid fixture'}
        try:PassageSamplerR4(b,frame,references,'train',42,cfg).draw(1,np.array([0]),bad)
        except ValueError:pass
        else:raise AssertionError('Noninteger/nonfinite schedule was silently accepted')
    # Absolute digital reference gain is removed before alpha is solved.
    scaled=[PassageReference(r.metadata,r.segments*1e8) for r in references]
    changed=PassageSamplerR4(b,frame,scaled,'train',42,cfg).draw(len(indices),indices)
    for x,y in zip(outputs[0][:4],changed[:4]):np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-6,equal_nan=True)
    return {'status':'passed R4 numeric fixtures','same_seed_replay':True,'all_three_split_policy_bitwise_equal':True,
        'analytic_raised_cosine_power_integral_passed':True,'natural_mask_and_zero_preserved':True,
        'high_share_short_passage_allowed':True,'digital_reference_gain_invariant':True,
        'bounded_whole_source_mean_power_preserving_timewarp':True,'full_split_LHS_and_exact_zero_count':True,'actual_competition_data_used':False}

def verify_package(folder,require_ready=True):
    from pipeline.passage_sampler_r4 import PassageConfigR4,make_passage_schedule
    gate=json.loads((folder/'readiness_manifest.json').read_text());files=gate['file_sha256']
    for name,h in files.items():assert sha(folder/name)==h,(name,'changed')
    assert hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()==gate['data_package_hash']
    source=ROOT/gate['source_version_dir'];assert sha(source/'manifest.json')==gate['source_version_manifest_sha256']
    r=pd.read_parquet(source/'reference_manifest.parquet');refs=check_reference_isolation(r)
    parts={};tables={};sampleids=set()
    for role,split in enumerate(['train','validation','outer_validation']):
        with np.load(folder/(split+'.npz'),allow_pickle=False) as z:data={k:z[k] for k in ['db','share','presence','cell_share']}
        parts[split]=verify_arrays(data);md=pd.read_parquet(folder/(split+'_sample_metadata.parquet'))
        assert len(md)==len(data['share']) and np.allclose(md.effective_share,data['share'],atol=1e-7)
        assert md.sample_id.is_unique and sampleids.isdisjoint(md.sample_id);sampleids.update(md.sample_id)
        assert md.loc[md.effective_share>0,'reference_count'].eq(1).all() and md.loc[md.effective_share.eq(0),'reference_count'].eq(0).all()
        assert set(md.loc[md.effective_share>0,'reference_component'])<=set(r.loc[r.split.eq(split),'reference_component'])
        seed=gate['generation_seed_base']+gate['fold']*10007+role*1000003+170003
        expected=make_passage_schedule(len(md),seed,PassageConfigR4(**gate['passage_config']))
        with np.load(folder/(split+'_schedule.npz'),allow_pickle=False) as z:
            assert np.array_equal(z['requested_share'],expected['requested_share'])
            assert np.array_equal(z['rendered_segments'],expected['rendered_segments'])
        np.testing.assert_allclose(data['share'],expected['requested_share'],atol=1e-7,rtol=0)
        assert np.array_equal(md.rendered_duration_seconds.to_numpy(),2*expected['rendered_segments'])
        assert int(data['share'].size-np.count_nonzero(data['share']))==int(round(.2*len(md)))
        tables[split]=md
    distributions=matched_distributions(tables,gate['distribution_tolerances'])
    if require_ready:assert gate['ready_to_train']
    assert gate['challenge_test_accessed'] is False and gate['sealed_arrays_accessed'] is False
    assert gate['model_evaluated_on_sealed'] is False
    return {'status':'passed independent R4 package verification','source':refs,'partitions':parts,'matched_distributions':distributions}

def data_only_pilot(config_path,examples=1024):
    """Fixed-seed data QA. Never fits a model or changes a generation rule."""
    from scipy.stats import ks_2samp
    from pipeline.passage_sampler_r4 import PassageSamplerR4,PassageConfigR4,make_passage_schedule
    from pipeline.build_passage_cv_r4 import load_records
    config=json.loads(config_path.read_text());parent=ROOT/config['parent_cv_dir'];refsdir=ROOT/config['reference_version_dir']
    rv=json.loads((refsdir/'manifest.json').read_text());assert rv['ready_to_materialize']
    config_hash=sha(config_path);source_hash=sha(refsdir/'manifest.json');cfg=PassageConfigR4(**config['passage_config'])
    parent_cv=json.loads((parent/'cv_manifest.json').read_text());folds=[]
    for f in parent_cv['folds']:
        fold=f['fold'];old=ROOT/f['data_dir'];gate=json.loads((old/'readiness_manifest.json').read_text())
        source=ROOT/gate['source_version_dir'];si=json.loads((source/'manifest.json').read_text())
        b=pd.read_parquet(source/'background_manifest.parquet');db=np.load(ROOT/si['base_background_array'],mmap_mode='r')
        r=pd.read_parquet(refsdir/f'fold_{fold:02d}_reference_manifest.parquet');check_reference_isolation(r)
        arrays={};summaries={}
        for j,split in enumerate(['train','validation','outer_validation']):
            seed=config['generation_seed']+fold*10007+j*1000003
            records,_=load_records(r,split);sampler=PassageSamplerR4(db,b,records,split,seed,cfg)
            indices=np.random.default_rng(seed+19).choice(np.flatnonzero(b.split.eq(split)),size=examples,replace=True)
            schedule=make_passage_schedule(examples,seed+170003,cfg)
            values=sampler.draw(examples,indices,schedule);data=dict(zip(['db','share','presence','cell_share'],values[:4]));qa=verify_arrays(data)
            rows=values[4]['items'];positive=[item for item in rows if item['references']]
            duration=np.asarray([item['references'][0]['measured_used_seconds'] for item in positive])
            rendered=np.asarray([item['references'][0]['derived_duration_seconds'] for item in positive])
            coarse=np.asarray([item['references'][0]['duration_bin'] for item in positive]);fine=np.minimum(np.searchsorted([120,300,600,900,1380],rendered,side='right')-1,3)
            source_fine=np.minimum(np.searchsorted([120,300,600,900,1380],duration,side='right')-1,3)
            y=np.asarray([item['desired_share'] for item in positive]);coarse_p=np.bincount(coarse,minlength=2)/len(coarse);fine_p=np.bincount(fine,minlength=4)/len(fine)
            source_fine_p=np.bincount(source_fine,minlength=4)/len(source_fine)
            arrays[split]={'source_duration':duration,'rendered_duration':rendered,'coarse':coarse_p,'fine':fine_p,'source_fine':source_fine_p,'y':y,'coarse_ids':coarse}
            summaries[split]={**qa,'positive_examples':len(positive),'source_duration_mean_seconds':float(duration.mean()),
                'source_duration_min_max_seconds':[int(duration.min()),int(duration.max())],
                'rendered_duration_mean_seconds':float(rendered.mean()),'coarse_duration_fractions':coarse_p.tolist(),
                'fine_rendered_duration_fractions':fine_p.tolist(),'fine_measured_source_duration_fractions':source_fine_p.tolist(),'positive_share_mean':float(y.mean()),
                'exact_scheduled_zero_count':int((schedule['requested_share']==0).sum()),
                'conditional_positive_share_means':[float(y[coarse==k].mean()) for k in [0,1]],
                'observed_active_minutes_mean_positive':float(np.mean([item['active_ref_minutes'] for item in positive])),
                'observed_support_coverage_mean_positive':float(np.mean([item['support_coverage'] for item in positive]))}
        comparisons={}
        for split in ['validation','outer_validation']:
            a=arrays['train'];q=arrays[split];t=summaries['train'];s=summaries[split]
            comparisons[split]={'mean_share_difference':abs(t['share_mean']-s['share_mean']),
                'zero_fraction_difference':abs(t['zero_fraction']-s['zero_fraction']),
                'coarse_duration_total_variation':float(np.abs(a['coarse']-q['coarse']).sum()/2),
                'fine_duration_total_variation':float(np.abs(a['fine']-q['fine']).sum()/2),
                'source_measured_fine_duration_TV_nonblocking':float(np.abs(a['source_fine']-q['source_fine']).sum()/2),
                'source_duration_KS_statistic':float(ks_2samp(a['source_duration'],q['source_duration']).statistic),
                'rendered_duration_KS_statistic':float(ks_2samp(a['rendered_duration'],q['rendered_duration']).statistic),
                'rendered_duration_mean_difference_seconds':abs(t['rendered_duration_mean_seconds']-s['rendered_duration_mean_seconds']),
                'maximum_conditional_mean_difference':max(abs(t['conditional_positive_share_means'][k]-s['conditional_positive_share_means'][k]) for k in [0,1])}
            comparisons[split]['predeclared_coarse_target_gates_passed']=all(comparisons[split][k]<=limit for k,limit in config['distribution_tolerances'].items())
        folds.append({'fold':fold,'splits':summaries,'comparison_to_train':comparisons})
    assert sha(config_path)==config_hash and sha(refsdir/'manifest.json')==source_hash
    report={'scope':'Data-only fixed-seed bounded whole-source power timewarp + fade/EQ + full-split LHS pilot; no model performance or parameter search',
        'examples_per_split_per_fold':examples,'reference_source_hash':rv['source_hash'],'reference_manifest_sha256':source_hash,
        'config_sha256':config_hash,'folds':folds,'full_R4_dataset_generated':False,'model_training_executed':False,
        'Test_calibration_sealed_accessed':False,'timewarp_used':True,
        'pilot_background_sampling':'Uniform in each eligible split with replacement; production TRAIN remains the frozen full one-pass order.',
        'interpretation':'Coarse stratification does not equalize fine durations. KS/TV are descriptive synthetic-draw diagnostics, not independent-field p-values.'}
    path=ROOT/'reports/R4_DATA_PILOT_WARP.json';path.write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# R4 bounded-timewarp data pilot','',f'{examples} fixed-seed examples × three roles × five folds; no models.','',
        '| Fold | Role | Mean source seconds | Mean rendered seconds | Mean y | Zero fraction | Coarse TV vs TRAIN | Fine TV | Duration KS |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for f in folds:
        for split,s in f['splits'].items():
            c=f['comparison_to_train'].get(split,{})
            lines.append(f'| {f["fold"]} | {split} | {s["source_duration_mean_seconds"]:.1f} | {s["rendered_duration_mean_seconds"]:.1f} | {s["share_mean"]:.4f} | {s["zero_fraction"]:.4f} | {c.get("coarse_duration_total_variation",0):.3f} | {c.get("fine_duration_total_variation",0):.3f} | {c.get("source_duration_KS_statistic",0):.3f} |')
    lines+=['','Coarse rendered bins: [120,300), [300,900]. Fine rendered bins: [120,300), [300,600), [600,900), {900}.',
        'Rendered support is independently LHS-scheduled discrete Uniform{60,...,450} two-second bins. Source power passes once monotonically through cumulative-integral rebinning, ratio in [1/3,3]; original measurement UTC/duration is unchanged.',
        'Original-source duration differences are nonblocking diagnostics. Rendered duration/target matching is checked against fixed gates; natural SA mask coverage is separate. R2 and the original fade-only pilot are unchanged.','', '[Full numeric pilot](R4_DATA_PILOT_WARP.json)']
    (ROOT/'reports/R4_DATA_PILOT_WARP.md').write_text('\n'.join(lines)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixture',action='store_true');p.add_argument('--data-dir',type=Path)
    p.add_argument('--pilot-config',type=Path);p.add_argument('--pilot-examples',type=int,default=1024);a=p.parse_args()
    if a.fixture:print(json.dumps(numeric_fixtures(),indent=2))
    if a.data_dir:print(json.dumps(verify_package(a.data_dir),indent=2))
    if a.pilot_config:
        r=data_only_pilot(a.pilot_config,a.pilot_examples);print(json.dumps({'report':'reports/R4_DATA_PILOT_WARP.json','fold_comparisons':[{**f['comparison_to_train'],'fold':f['fold']} for f in r['folds']]},indent=2))
    if not a.fixture and a.data_dir is None and a.pilot_config is None:p.error('Choose --fixture, --data-dir or --pilot-config')
