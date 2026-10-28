"""Read-only source/physics QA for new pilot; locked panel arrays stay unopened."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from ops.r16_support_temporal_data import load_training,load_panel,background_gaps,guard
from pipeline.sampler_r9 import energy_weights

OUT=Path(__file__).parent;DATA=OUT/'data';MANIFEST=DATA/'manifest.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def physical(d):
    n=len(d['share']);o=np.isfinite(d['db']);b=d['background_observed_power'];a=d['added_observed_power'];e=d['mixture_observed_power']
    q=d['cell_share'];y=d['share'];z=d['presence'].astype(bool)
    assert d['db'].shape==(n,15,25) and o.any((1,2)).all()
    assert not np.isinf(d['db']).any() and np.array_equal(o,d['observed_mask'])
    for v in (b,a,e,q,y):assert np.isfinite(v).all() and (v>=0).all()
    assert (q<=1).all() and (y<=1).all() and np.array_equal(z,y>0)
    assert np.array_equal(e,b+a) and np.all(e[~o]==0) and np.all(q[~o]==0)
    expected_y=a.sum((1,2))/e.sum((1,2));expected_q=np.divide(a,e,out=np.zeros_like(e),where=o)
    yerr=float(abs(y-expected_y).max());qerr=float(abs(q-expected_q).max())
    assert yerr<1e-12 and qerr<1e-12
    w=np.stack([energy_weights(x)for x in d['db']]);raw_w=np.stack([energy_weights(x)for x in d['unrounded_db']])
    rawerr=float(abs((q*raw_w).sum((1,2))-y).max());rounderr=float(abs((q*w).sum((1,2))-y).max())
    assert rawerr<1e-10 and rounderr<.0058
    griderr=float(abs(d['db'][o].astype(float)*10-np.round(d['db'][o].astype(float)*10)).max())
    assert griderr<.0002
    active=(a>0).any(2).sum(1)
    return {'n':n,'zero_n':int((~z).sum()),'positive_n':int(z.sum()),'zero_fraction':float((~z).mean()),
            'share_mean':float(y.mean()),'share_rms':float(np.sqrt((y*y).mean())),
            'positive_observed_active_minutes_histogram':{str(int(k)):int(v)for k,v in zip(*np.unique(active[z],return_counts=True))},
            'positive_observed_support_ge12':int((active[z]>=12).sum()),
            'physical_share_max_abs':yerr,'physical_cell_share_max_abs':qerr,
            'unrounded_physical_pool_max_abs':rawerr,'rounded_weight_pool_max_abs':rounderr,
            'float32_point1dB_grid_max_abs_times10':griderr}

def main():
    started=time.perf_counter();sys.addaudithook(guard)
    ready=read(DATA/'readiness.json');m=read(MANIFEST);manifest_sha=sha(MANIFEST)
    assert ready['ready'] is True and ready['manifest_sha256']==manifest_sha and m['Test_accessed'] is False
    pins=m['source_pins'];source_verified=0
    for p,h in pins.items():
        assert sha(ROOT/p)==h,p;source_verified+=1
    artifacts=m['artifact_sha256'];artifact_verified=0
    for p,h in artifacts.items():
        assert sha(DATA/p)==h,p;artifact_verified+=1
    assert sha(ROOT/'ops/r16_support_temporal_data.py')==m['code_sha256']
    refs=pd.read_parquet(DATA/'reference_partitions.parquet')
    part='r16_reference_partition'
    for key in ('reference_component','sha256','hull_id','recording_day'):
        assert refs.groupby(key)[part].nunique().max()==1,key
    counts={k:{'clips':len(g),'components':g.reference_component.nunique()}for k,g in refs.groupby(part)}
    assert counts=={'A':{'clips':75,'components':13},'B':{'clips':19,'components':5},'C':{'clips':25,'components':4}}
    long=refs[refs.minute_count>=11][['clip_id','minute_count','reference_component',part]].to_dict('records')
    assert sorted(refs.loc[refs.reference_component.eq('VT_R4_REF_474fe510b4e0aa460cad'),'minute_count'].nlargest(3))==[19,20,23]
    assert refs.reference_component.eq('VT_R4_REF_474fe510b4e0aa460cad').sum()==9
    assert refs.loc[refs.reference_component.eq('VT_R4_REF_474fe510b4e0aa460cad'),part].eq('A').all()
    assert refs.loc[refs.reference_component.eq('VT_R4_REF_de02fbd6bbb753c153bd'),part].eq('C').all()
    assert refs.loc[refs.reference_component.eq('VT_R4_REF_de02fbd6bbb753c153bd'),'minute_count'].max()==12
    frames={role:pd.read_parquet(DATA/(role+'_backgrounds.parquet'))for role in ('train','local_fit','local_gate','local_report','domain_holdout')}
    newtrain=pd.read_parquet(DATA/'train_D1_backgrounds.parquet')
    gaps=background_gaps({**frames,'train':pd.concat([frames['train'],newtrain]).drop_duplicates('base_window_index')})
    d0=load_training(MANIFEST,'D0');d1=load_training(MANIFEST,'D1')
    stats={'D0':physical(d0),'D1':physical(d1)}
    for name,d in [('D0',d0),('D1',d1)]:
        assert len(d['share'])==4000 and (d['share']==0).sum()==800
        assert not set(d['reference_component'][d['presence']>0])-set(refs[refs[part].eq('A')].reference_component)
    for key in ('pair_id','reference_index','reference_component','explicit_no_source','operator','effective_mask_bits'):
        assert np.array_equal(d0[key],d1[key]),key
    pos=d0['presence']>0;zero=~pos
    assert np.array_equal(d0['base_window_index'][pos],d1['base_window_index'][pos])
    assert stats['D1']['positive_observed_support_ge12']>stats['D0']['positive_observed_support_ge12']
    receipts=read(DATA/'HF_SWAP_RECEIPTS.json');swaps=[r for r in receipts if r['replaced']]
    assert len(receipts)==800
    strata=['site','sensor_group','recorder','effective_mask_bits']
    for key in strata:assert np.array_equal(d0[key],d1[key]),key
    for r in swaps:
        i=r['row'];assert zero[i] and d0['base_window_index'][i]==r['original_base_window_index'] and d1['base_window_index'][i]==r['base_window_index']
        assert r['new_hf_bump_db']>r['stratum_q90_db']
    assert not set(d0['base_window_index'])&set(r['base_window_index']for r in swaps)
    for arm,d in [('D0',d0),('D1',d1)]:
        recipes=read(DATA/'train'/(arm+'_recipes.json'))
        for r in recipes:
            geometry=r['geometry']
            if geometry is None:continue
            assert geometry['time_warp'] is False and geometry['source_repetition'] is False and geometry['added_bell_or_fade'] is False
            assert geometry['crop_begin_minute']>=0 and geometry['crop_begin_minute']+geometry['selected_minutes']<=geometry['full_native_minutes']
            assert geometry['window_start_minute']>=0 and geometry['window_start_minute']+geometry['selected_minutes']<=15
        if arm=='D0':r0=recipes
        else:r1=recipes
    assert all(a['channel_seed']==b['channel_seed']for a,b in zip(r0,r1))
    fitstats={}
    for op in m['panels']['local_fit']:
        d=load_panel(MANIFEST,'local_fit',op);fitstats[op]=physical(d)
        for train in (d0,d1):
            assert not set(d['base_window_index'])&set(train['base_window_index'])
            assert not set(d['reference_component'][d['presence']>0])&set(train['reference_component'][train['presence']>0])
    assert sha(MANIFEST)==manifest_sha
    for p,h in pins.items():assert sha(ROOT/p)==h
    result={'status':'passed','manifest_sha256':manifest_sha,'source_pins_verified':source_verified,'bundle_artifact_SHA_verified':artifact_verified,
            'reference_partition_counts':counts,'long_source_records':long,'background_crossrole_purge':gaps,
            'training_physics':stats,'local_fit_physics':fitstats,
            'paired_D0_D1_positive_sources_and_channel_draws':True,'paired_D0_D1_site_sensor_recorder_effective_masks':True,
            'HF_swap_requested':800,'HF_swap_completed':len(swaps),'HF_swap_shortfall':800-len(swaps),
            'legacy_C_source_unseen_reuse_prohibited':True,'independent_long_reference_minutes':12,
            'independent_ge15_reference_components':0,'locked_report_domain_long_arrays_opened':False,
            'model_training_performed':False,'official_query_accessed':False,'elapsed_seconds':time.perf_counter()-started}
    path=OUT/'INDEPENDENT_DATA_QA.json';path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in('local_fit_physics','long_source_records','background_crossrole_purge')},ensure_ascii=False))

if __name__=='__main__':main()
