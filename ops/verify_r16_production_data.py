"""Independent read-only R16 full bundle source/schema/physics audit."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def noquery(event,args):
    if event=='open'and args and isinstance(args[0],(str,bytes)):
        p=str(args[0]).lower()
        if Path(p).name in ('test.csv','sample_submission.csv')or'/eda_deep/'in p:raise PermissionError('No query audit access')
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}

def verify(manifest_path,fold_only=None):
    import pandas as pd
    from pipeline.dataset_r9 import R9Sources
    from ops.r16_forward_data import choose_geometry,channel_zero_mean
    from pipeline.sampler_r9 import smooth_eq
    from ops.r15_forward_prototype import shelf_curve
    manifest=ROOT/manifest_path;m=read(manifest);started=time.perf_counter();source=R9Sources();results=[]
    folds=[int(fold_only)]if fold_only is not None else list(range(5))
    if 'folds'in m:entries={int(k):ROOT/v['manifest_path']for k,v in m['folds'].items()}
    else:entries={int(m['fold']):manifest}
    r16dir=ROOT/'reports/r16_cross_operator_r1/data';fixed={r:pd.read_parquet(r16dir/(r+'_backgrounds.parquet'))for r in ('local_fit','local_gate','local_report','domain_holdout')}
    refs=pd.read_parquet(manifest.parent/'reference_partitions.parquet')if'folds'in m else pd.read_parquet(manifest.parent.parent/'reference_partitions.parquet')
    reserved=set(refs.loc[refs.r16_reference_partition.ne('A'),'reference_row_index'])
    seen_training_backgrounds=set();seen_training_references=set();model_data_rows=0
    for fold in folds:
        p=entries[fold];f=read(p);directory=p.parent
        for n,h in {**f['source_code_sha256'],**f['source_input_sha256']}.items():
            if sha(ROOT/n)!=h:raise ValueError('Frozen source changed: '+n)
        trainmeta=pd.read_parquet(directory/'train_backgrounds.parquet');held=pd.read_parquet(directory/'all_admitted_sources.parquet')
        ownrefs=pd.read_parquet(directory/'train_references.parquet');otherrefs=pd.read_parquet(directory/'heldout_A_references.parquet')
        if set(ownrefs.reference_component)&set(otherrefs.reference_component)or set(ownrefs.reference_row_index)&reserved:raise AssertionError('Reference source split leak')
        # Compute source isolation directly from exported timestamps/support.
        gap_min=np.inf
        for r,g in held.groupby('role'):
            foreign=held[held.role.ne(r)]
            if set(g.background_group)&set(foreign.background_group):raise AssertionError('Ownfold background groups cross')
            for sensor,a in g.groupby('sensor_group'):
                b=foreign[foreign.sensor_group.eq(sensor)]
                if not len(b):continue
                x=pd.to_datetime(a.start_export_clock).astype('int64').to_numpy()/1e9
                y=np.sort(pd.to_datetime(b.start_export_clock).astype('int64').to_numpy()/1e9);i=np.searchsorted(y,x)
                distance=np.minimum(abs(x-y[np.clip(i-1,0,len(y)-1)]),abs(x-y[np.clip(i,0,len(y)-1)]));gap=float(np.maximum(distance-900,0).min()/3600)
                gap_min=min(gap_min,gap)
        if gap_min<24-1e-8:raise AssertionError('Ownfold 24h purge failed')
        for role,b in fixed.items():
            if set(trainmeta.base_window_index)&set(b.base_window_index)or set(trainmeta.background_group)&set(b.background_group):raise AssertionError('Fixed benchmark TRAIN background leak')
        role_results={}
        for role,e in f['roles'].items():
            if sha(ROOT/e['path'])!=e['sha256']:raise AssertionError('Role hash mismatch')
            d=load(ROOT/e['path']);n=e['n'];model_data_rows+=n
            if d['db'].shape!=(n,15,25)or d['cell_share'].shape!=(n,15,25)or d['share'].shape!=(n,):raise AssertionError('Shape differs')
            if any(v.dtype.kind=='O'for v in d.values()):raise AssertionError('Object payload')
            o=np.isfinite(d['db']);q=d['cell_share'];y=d['share'];positive=d['presence'].astype(bool)
            if np.isinf(d['db']).any()or not np.isfinite(q).all()or not np.isfinite(y).all()or(q<0).any()or(q>1).any()or(y<0).any()or(y>1).any():raise AssertionError('Invalid range')
            if (o.any(2)!=o.all(2)).any()or(q[~o]!=0).any()or not np.array_equal(positive,y>0):raise AssertionError('Mask/label inconsistency')
            if (np.isfinite(d['db']).all(2).sum(1)<3).any():raise AssertionError('Insufficient source support')
            finite=d['db'][o];grid=float(abs(finite*10-np.round(finite*10)).max())
            if grid>.0002:raise AssertionError('Quantization differs')
            shift=np.max(np.where(o,d['db'],-np.inf),axis=(1,2));power=np.where(o,10**((np.where(o,d['db'],shift[:,None,None])-shift[:,None,None])/10),0.)
            weights=power/power.sum((1,2))[:,None,None];pool=float(abs((weights*q).sum((1,2))-y).max())
            if pool>.0058:raise AssertionError('Rounded observed weights/raw q target gap exceeds bound')
            if role!='selection':
                meta=pd.read_parquet(ROOT/e['metadata_path'])
                if len(meta)!=n or not np.array_equal(meta.base_window_index,d['base_window_index']):raise AssertionError('Metadata order differs')
                if np.sum(y==0)!=n//5:raise AssertionError('Explicit zero quota differs')
                if role=='train':
                    seen_training_backgrounds.update(d['base_window_index'].tolist());seen_training_references.update(d['reference_index'][positive].tolist())
                    if set(d['reference_index'][positive])&reserved:raise AssertionError('Reserved B/C entered TRAIN')
                # Recreate known source component/operator calculations for fixed hash-selected rows.
                chosen=np.random.default_rng(20261027+fold*7+len(role)).choice(n,min(60,n),replace=False)
                maxdb=0.;maxq=0.;maxy=0.
                for i in chosen:
                    row=meta.iloc[int(i)];bg=np.asarray(source.background(row),float);obs=np.isfinite(bg);offset=np.nanmax(bg)
                    b=np.where(obs,10**((np.where(obs,bg,offset)-offset)/10),0.);added=np.zeros((15,25));zero=bool(d['explicit_no_source'][i])
                    code={'train':0,'cv_validation':1,'ensemble_fit':2,'ensemble_gate':3,'report':4}[role]
                    rng=np.random.default_rng(np.random.SeedSequence([f['seed'],fold,code,int(i),0x4752]))
                    if not zero:
                        native=source.reference_minutes[int(d['reference_index'][i])];g=choose_geometry(bg,native,rng,report_crop=role=='report')
                        eq=smooth_eq(rng,6.);post=channel_zero_mean(rng,6. if role=='train'else 12.)
                    snr=float(rng.uniform(-10,10));sub=rng.uniform(-10,10,2)
                    if not zero:
                        start=g['window_start_minute'];begin=g['crop_begin_minute'];length=g['selected_minutes'];part=native[begin:begin+length];om=obs[start:start+length];op=str(d['operator'][i])
                        if op in ('control_eq6','mfhf_shelf6','mfhf_shelf12','mfhf_shelf12_lfminus6'):
                            shelf={'control_eq6':(0.,0.),'mfhf_shelf6':(6.,0.),'mfhf_shelf12':(12.,0.),'mfhf_shelf12_lfminus6':(12.,-6.)}[op]
                            r=native*10**((eq+shelf_curve(*shelf))[None,:]/10);alpha=10**(snr/10)*np.median(b.sum(1)[obs.all(1)])/r.sum(1).max();added[start:start+length]=alpha*r[begin:begin+length]
                        elif op=='alpha_preEQ_inserted':
                            alpha=10**(snr/10)*np.median(b.sum(1)[obs.all(1)])/np.where(om,part,0.).sum(1).max();added[start:start+length]=alpha*part*10**(post[None,:]/10)
                        elif op=='independent_subband_snr':
                            for j,ix in enumerate((np.arange(9),np.arange(9,25))):
                                alpha=10**(sub[j]/10)*np.median(b[:,ix].sum(1)[obs.all(1)])/np.where(om[:,ix],part[:,ix],0.).sum(1).max();added[start:start+length,ix]=alpha*part[:,ix]*10**(post[ix][None,:]/10)
                        else:raise AssertionError('Unknown operator')
                    oa=np.where(obs,added,0.);energy=b+oa;truth=oa.sum()/energy.sum();mask=np.divide(oa,energy,out=np.zeros_like(energy),where=obs)
                    expected=np.full((15,25),np.nan);expected[obs]=10*np.log10(energy[obs])+offset;expected=np.round(expected,1).astype(np.float32)
                    maxdb=max(maxdb,float(np.nanmax(abs(expected-d['db'][i]))));maxq=max(maxq,float(abs(mask.astype(np.float32)-q[i]).max()));maxy=max(maxy,float(abs(truth-y[i])))
                if maxdb!=0 or maxq!=0 or maxy>1e-12:raise AssertionError('Source/operator replay failed')
            else:
                maxdb=maxq=maxy=0.;chosen=[]
            role_results[role]={'n':n,'zero_n':int((y==0).sum()),'quantization_max_gap':grid,'rounded_pool_max_gap':pool,
                'replay_n':len(chosen),'source_replay_db_max_gap':maxdb,'source_replay_q_max_gap':maxq,'source_replay_y_max_gap':maxy}
        results.append({'fold':fold,'manifest_sha256':sha(p),'ownfold_minimum_support_gap_hours':gap_min,
           'reference_B_C_TRAIN_overlap':0,'ownfold_reference_TRAIN_heldout_overlap':0,'fixed_benchmark_TRAIN_background_overlap':0,'roles':role_results})
    return {'status':'passed','folds':results,'loaded_rows_including_shared_selection_repeats':model_data_rows,
       'unique_TRAIN_backgrounds_across_evaluated_folds':len(seen_training_backgrounds),'unique_TRAIN_references_across_evaluated_folds':len(seen_training_references),
       'TRAIN_background_ID_sha256':hashlib.sha256(np.asarray(sorted(seen_training_backgrounds),np.int64).tobytes()).hexdigest(),
       'raw_label_and_quantized_observation_checks':'every loaded row; source/operator replay up to60 per ownfoldrole',
       'Test_read_or_model_fitting':False,'production_data_code_sha256':sha(ROOT/'pipeline/r16_production_data.py'),
       'verifier_code_sha256':sha(__file__),'elapsed_seconds':time.perf_counter()-started}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='data/synthetic/r16_production_v1/manifest.json');ap.add_argument('--fold',type=int);ap.add_argument('--output',default='reports/R16_PRODUCTION_DATA_INDEPENDENT_QA.json');a=ap.parse_args()
    sys.addaudithook(noquery);r=verify(a.manifest,a.fold);p=ROOT/a.output;p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':r['status'],'output':str(p),'seconds':r['elapsed_seconds']}))
