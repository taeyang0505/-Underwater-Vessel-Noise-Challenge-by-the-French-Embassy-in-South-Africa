"""Independent R16 component/source QA, not a performance evaluation.

Never reads Test or Test-derived statistics and never loads a model. Locked
panels are inspected solely for physical/schema/source invariants, with no
target distributions or model metrics published.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'reports/r16_cross_operator_r1/data'
OUT=ROOT/'reports/r16_independent_qa_r1/DATA_INDEPENDENT_QA.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def reject_query(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        name=str(args[0]).lower()
        if (Path(name).name in ('test.csv','sample_submission.csv') or '/eda_deep/' in name
                or any(x in name for x in ('r15_score_duration','r15_descriptive_audit','r15_band_comparison','r15_static_channel','r15_member_audit'))):
            raise PermissionError('R16 QA may not read Test or prior query statistics')


def load(path):
    with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    assert all(v.dtype.kind!='O' for v in d.values())
    return d


def graph(frame):
    parent=list(range(len(frame)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(i)]=find(j)
    for col in ['original_group_key','recording_day','sha256']:
        previous={}
        for i,value in enumerate(frame[col]):
            if value in previous:union(i,previous[value])
            else:previous[value]=i
    starts=pd.to_datetime(frame.utc_begin,utc=True).astype('int64').to_numpy()
    ends=pd.to_datetime(frame.utc_end,utc=True).astype('int64').to_numpy()
    for i in range(len(frame)):
        for j in range(i):
            if starts[i]<ends[j] and starts[j]<ends[i]:union(i,j)
    return [find(i) for i in range(len(frame))]


def main():
    if OUT.exists():raise FileExistsError('Preserve previous QA')
    sys.addaudithook(reject_query)
    mpath=FOLDER/'manifest.json';m=json.loads(mpath.read_text());ready=json.loads((FOLDER/'readiness.json').read_text());plan=json.loads((FOLDER/'plan.json').read_text())
    assert ready['ready'] and ready['manifest_sha256']==sha(mpath) and m['status']=='completed'
    assert not m['Test_accessed'] and not m['reference_unseen_for_old_models']
    assert sha(FOLDER/'plan.json')==m['plan_sha256'] and sha(ROOT/'ops/r16_forward_data.py')==m['code_sha256']==plan['code_sha256']
    assert not plan['Test_read_or_Test_based_generator_tuning'] and plan['training_per_arm_rows']==4000
    before={str((FOLDER/p).relative_to(ROOT)):sha(FOLDER/p) for p in m['artifact_sha256']}
    before[str(mpath.relative_to(ROOT))]=sha(mpath)
    for p,h in m['artifact_sha256'].items():assert sha(FOLDER/p)==h
    for p,h in m['source_pins'].items():assert sha(ROOT/p)==h
    config=json.loads((ROOT/'configs/r9_full_training_v1.json').read_text())
    bgdir=ROOT/config['background_version'];refdir=ROOT/config['reference_version']
    bm=json.loads((bgdir/'manifest.json').read_text());rm=json.loads((refdir/'manifest.json').read_text())
    bgpath=ROOT/bm['background_bank']['path'];refpath=ROOT/rm['bank']['path']
    bgsha=sha(bgpath);refsha=sha(refpath)
    assert bgsha==m['source_identity']['background_bank_sha256']==bm['background_bank']['sha256']
    assert refsha==m['source_identity']['reference_bank_sha256']==rm['bank']['sha256']
    refs=pd.read_parquet(FOLDER/'reference_partitions.parquet').reset_index(drop=True)
    assert len(refs)==refs.reference_row_index.nunique()==119
    labels=graph(refs);assert len(set(labels))==22
    for label in set(labels):assert refs.iloc[[i for i,x in enumerate(labels) if x==label]].r16_reference_partition.nunique()==1
    partition={k:{'clips':len(v),'components':v.reference_component.nunique()} for k,v in refs.groupby('r16_reference_partition')}
    assert partition=={'A':{'clips':66,'components':12},'B':{'clips':19,'components':5},'C':{'clips':34,'components':5}}
    fullrefs=pd.read_parquet(refdir/'reference_manifest.parquet').sort_values('reference_row_index').reset_index(drop=True)
    assert len(fullrefs)==509 and set(fullrefs.archive_role)=={'vessel_passage'}
    for col in ['hull_id','recording_day','sha256']:
        assert refs.groupby(col).r16_reference_partition.nunique().max()==1
    native={};reference_bank=np.load(refpath,mmap_mode='r',allow_pickle=False)
    for r in refs.itertuples(index=False):
        original=fullrefs.iloc[int(r.reference_row_index)];assert original.sha256==r.sha256
        a=np.asarray(reference_bank[int(original.segment_start):int(original.segment_start)+int(original.segment_count)])
        assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==original.expected_segment_slice_sha256
        n=len(a)//30;native[int(r.reference_row_index)]=a[:n*30].astype(float).reshape(n,30,25).mean(1)
    role_partition={'train':'A','local_fit':'B','local_gate':'B','local_report':'C','domain_holdout':'C'}
    backgrounds={role:pd.read_parquet(FOLDER/(role+'_backgrounds.parquet')) for role in role_partition}
    source_gaps=[]
    for i,(ra,a) in enumerate(backgrounds.items()):
        assert len(a)==a.base_window_index.nunique()==(4000 if ra=='train' else 400)
        for rb,b in list(backgrounds.items())[i+1:]:
            assert not(set(a.base_window_index)&set(b.base_window_index)) and not(set(a.background_group)&set(b.background_group))
            gaps=[]
            for sensor,group in a.groupby('sensor_group'):
                other=b[b.sensor_group==sensor]
                if not len(other):continue
                x=pd.to_datetime(group.start_export_clock).astype('int64').to_numpy()/1e9
                y=np.sort(pd.to_datetime(other.start_export_clock).astype('int64').to_numpy()/1e9);p=np.searchsorted(y,x)
                distance=np.minimum(abs(x-y[np.clip(p-1,0,len(y)-1)]),abs(x-y[np.clip(p,0,len(y)-1)]))
                gaps.extend(np.maximum(distance-900,0)/3600)
            assert not gaps or min(gaps)>=24-1e-8
            source_gaps.append({'a':ra,'b':rb,'ID_group_overlap':0,'minimum900s_support_gap_hours':float(min(gaps)) if gaps else None})
    bank=np.load(bgpath,mmap_mode='r',allow_pickle=False);checks={};allrows=0;max_replay=maxq=maxy=maxpool=maxrounded=0.
    freq=np.round(10**(1.5+.1*np.arange(25)),1);z=np.clip((np.log(freq)-np.log(200))/(np.log(500)-np.log(200)),0,1);shelf=.5-.5*np.cos(np.pi*z)
    legacy={'control_eq6':(0,0),'mfhf_shelf6':(6,0),'mfhf_shelf12':(12,0),'mfhf_shelf12_lfminus6':(12,-6)}
    for role,frame in backgrounds.items():
        raw=np.asarray(bank[frame.base_window_index.to_numpy(int)]).astype(float);bits=frame.boundary_mask_bits.to_numpy(int)
        raw[((bits[:,None]>>np.arange(15))&1).astype(bool)]=np.nan
        o=np.isfinite(raw);assert (o.any(2)==o.all(2)).all() and (o.all(2).sum(1)>=3).all()
        offset=np.nanmax(raw,(1,2),keepdims=True);b=np.where(o,10**((np.where(o,raw,offset)-offset)/10),0)
        recipes=json.loads((FOLDER/(role+'_recipes.json')).read_text());n=len(frame);assert len(recipes)==n
        entries=m['training'] if role=='train' else m['panels'][role];first=None
        for name,e in entries.items():
            p=FOLDER/e['path'];assert sha(p)==e['sha256'];d=load(p);allrows+=n
            assert d['db'].shape==(n,15,25) and d['share'].shape==(n,) and d['cell_share'].dtype==np.float64
            assert not any(k.lower().startswith(('pca','anchor','feature')) for k in d)
            assert np.array_equal(d['base_window_index'],frame.base_window_index.to_numpy())
            assert np.array_equal(d['observed_mask'],o) and np.array_equal(np.isfinite(d['db']),o) and np.array_equal(np.isfinite(d['unrounded_db']),o)
            assert np.array_equal(d['background_observed_power'],b)
            assert np.array_equal(d['added_observed_power'],np.where(o,d['added_full_power'],0))
            assert np.array_equal(d['mixture_observed_power'],b+d['added_observed_power'])
            zero=d['explicit_no_source'].astype(bool);assert zero.sum()==n//5 and np.array_equal(d['presence'].astype(bool),~zero)
            assert (d['share'][zero]==0).all() and (d['share'][~zero]>0).all() and not d['added_full_power'][zero].any()
            q=np.divide(d['added_observed_power'],d['mixture_observed_power'],out=np.zeros_like(b),where=o)
            y=d['added_observed_power'].sum((1,2))/d['mixture_observed_power'].sum((1,2))
            qe=float(abs(q-d['cell_share']).max());ye=float(abs(y-d['share']).max());maxq=max(maxq,qe);maxy=max(maxy,ye);assert max(qe,ye)<1e-12
            assert ((q>=0)&(q<=1)).all() and np.array_equal(np.round(d['unrounded_db'],1).astype(np.float32),d['db'],equal_nan=True)
            for j,recipe in enumerate(recipes):
                assert recipe['background_index']==int(d['base_window_index'][j]) and recipe['pair_id']==str(d['pair_id'][j])
                assert recipe['reference_index']==int(d['reference_index'][j]) and recipe['explicit_no_source']==bool(zero[j])
                if zero[j]:continue
                rid=int(d['reference_index'][j]);ref=refs[refs.reference_row_index==rid].iloc[0]
                assert ref.r16_reference_partition==role_partition[role] and str(ref.reference_component)==str(d['reference_component'][j])
                r=native[rid];g=recipe['geometry'];start=g['window_start_minute'];begin=g['crop_begin_minute'];length=g['selected_minutes']
                assert 0<=start<=15-length and 0<=begin<=len(r)-length and 1<=length<=15
                assert not g['time_warp'] and not g['source_repetition'] and not g['added_bell_or_fade']
                if g['family']=='natural_full':assert length==len(r)<=15 and begin==0
                elif g['family']=='natural_peaked_1to6':assert length<=6 and begin<=int(r.sum(1).argmax())<begin+length
                else:assert role in ('local_report','domain_holdout') and g['family']=='natural_uniform_crop_1to6' and length<=6
                op=str(d['operator'][j]);rec=recipe['operator_receipts'][op];part=r[begin:begin+length];ob=o[j,start:start+length]
                expected=np.zeros((15,25));bmed=np.median(b[j].sum(1)[o[j].all(1)])
                if op in legacy:
                    boost,low=legacy[op];curve=np.asarray(rec['common_random_eq_db'])+low*(1-shelf)+boost*shelf
                    assert np.max(np.abs(curve-np.asarray(rec['channel_db'])))<1e-12
                    tr=r*10**(curve[None,:]/10);peak=tr.sum(1).max();alpha=10**(rec['requested_SNR_db']/10)*bmed/peak
                    assert abs(alpha-rec['alpha'])/alpha<1e-12
                    expected[start:start+length]=alpha*tr[begin:begin+length]
                else:
                    curve=np.asarray(rec['channel_db']);bound=6 if role=='train' else 12
                    assert abs(curve.mean())<1e-12 and abs(curve).max()<=bound+1e-12
                    if op=='alpha_preEQ_inserted':
                        peak=np.where(ob,part,0).sum(1).max();alpha=10**(rec['requested_SNR_db']/10)*bmed/peak
                        assert abs(alpha-rec['alpha'])/alpha<1e-12
                        expected[start:start+length]=alpha*part*10**(curve[None,:]/10)
                    else:
                        assert op=='independent_subband_snr'
                        for k,ix in enumerate((np.arange(9),np.arange(9,25))):
                            peak=np.where(ob[:,ix],part[:,ix],0).sum(1).max();den=np.median(b[j,:,ix].sum(0)[o[j].all(1)])
                            alpha=10**(rec['subband_SNR_db'][k]/10)*den/peak
                            assert abs(alpha-rec['alpha_subbands'][k])/alpha<1e-12
                            expected[start:start+length,ix]=alpha*part[:,ix]*10**(curve[ix][None,:]/10)
                er=float(abs(expected-d['added_full_power'][j]).max());max_replay=max(max_replay,er);assert er<1e-10
            for begin in range(0,n,128):
                sl=slice(begin,begin+128);mask=o[sl];rr=d['unrounded_db'][sl];rd=d['db'][sl].astype(float);y0=y[sl]
                er=np.where(mask,10**((np.where(mask,rr,offset[sl])-offset[sl])/10),0);wr=er/er.sum((1,2),keepdims=True)
                eq=np.where(mask,10**((np.where(mask,rd,offset[sl])-offset[sl])/10),0);wq=eq/eq.sum((1,2),keepdims=True)
                rawgap=abs((q[sl]*wr).sum((1,2))-y0);roundgap=abs((q[sl]*wq).sum((1,2))-y0)
                delta=np.where(mask,abs(np.where(mask,rr,0)-np.where(mask,rd,0)),0).max((1,2));bound=np.tanh(np.log(10)*delta/20)
                assert (roundgap-bound).max()<1e-10
                maxpool=max(maxpool,float(rawgap.max()));maxrounded=max(maxrounded,float(roundgap.max()))
            paired=['pair_id','base_window_index','reference_index','reference_component','observed_mask','background_observed_power','family','explicit_no_source','site','background_group']
            if first is None:first={k:d[k].copy() for k in paired};zero_input=d['db'][zero].copy()
            else:
                assert all(np.array_equal(first[k],d[k]) for k in paired)
                assert np.array_equal(zero_input,d['db'][zero],equal_nan=True)
            checks[role+'/'+name]={'rows':n,'zeros':n//5,'positive_observed':4*n//5,'physical_q_y_and_support_passed':True,'paired_source_identity_passed':True}
    assert allrows==17600 and maxpool<1e-10
    from ops.r16_forward_data import load_panel
    for role in ['local_report','domain_holdout']:
        try:load_panel(mpath,role,'control_eq6')
        except PermissionError:pass
        else:raise AssertionError('Locked role admitted without freeze receipt')
    for p,h in before.items():assert sha(ROOT/p)==h
    for p,h in m['source_pins'].items():assert sha(ROOT/p)==h
    result={'status':'PASS','verified_utc':datetime.now(timezone.utc).isoformat(),'scope':'Components/source/loader QA only; no role target moments, model metrics, Test inputs/statistics, model training or inference',
        'manifest_sha256':sha(mpath),'data_rows_checked':allrows,'role_operator_checks':checks,'reference_partition':partition,
        'selected_reference_graph_components_independently_rebuilt':22,'selected119_original_raw_slices_SHA_verified':True,
        'source_bank_global_SHA256':{'background':bgsha,'reference':refsha},'background_all_role_pair_gaps':source_gaps,
        'raw_q_max_error':maxq,'raw_y_max_error':maxy,'native_reference_to_A_replay_max_error':max_replay,'raw_pool_max_error':maxpool,
        'max_rounded_pool_gap':maxrounded,'individual_actual_quantization_bound_verified':True,
        'no_PCA_anchor_or_model_features':True,'paired_zero_and_source_exact':True,'report_domain_loader_requires_global_freeze':True,
        'old_models_already_saw_full509_references':True,'new_mini_only_A_to_B_C_reference_holdout':True,
        'new_inserted_SNR_mask_conditioning_declared':True,'source_data_unchanged':True,'input_artifact_sha256':before,
        'limits':['This is physical/data QA, not a blind model performance evaluation.',
            'Reference A/B/C independence holds inside selected119/22pool; full corpus has connected bridges not used here.',
            'Prior domain site panels were development-exposed; not a new blind benchmark.',
            'Fixed6 operators do not identify hidden official generation; no Test matching or operator selection performed.']}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'PASS','output':str(OUT.relative_to(ROOT)),'sha256':sha(OUT),'rows':allrows,'A_replay_max':max_replay}))


if __name__=='__main__':main()
