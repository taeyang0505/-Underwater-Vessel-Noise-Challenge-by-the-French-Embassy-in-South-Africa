"""Frozen R9 long-support positives crossed with known-A0 HF/edge panels.

Challenge-set diagnostics, never a reconstruction of the hidden Test law.
All choices are declared before predictions. Original R9 artifacts stay fixed.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops.infer_r9_fold0_all5_pilot import (
    inputs,predict_members,combine,environment,sha,verify_pins,
    DEFAULT_TREE,DEFAULT_BUNDLE,DEFAULT_CONFIG,MEMBERS)
from ops.r10_factorial_transfer_audit import eq_curve,render
SEED=2026102611
POS_MODES=('r9_original','full_smooth12','full_smooth13','full_smooth15',
           'stationary15_control','native_long15','full_smooth15_share030')

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def choose(frame,n,tag):
    import numpy as np
    keys=[hashlib.sha256(f'R11|{SEED}|{tag}|{int(i)}'.encode()).hexdigest() for i in frame.base_window_index]
    return frame.iloc[np.argsort(keys,kind='stable')[:n]].copy().reset_index(drop=True)

def hf_bump(db):
    import numpy as np
    x=np.asarray(db,dtype=float);o=np.isfinite(x).all(2);v=x[:,:,18:25]
    peak=np.nanmax(v,axis=2);safe=np.where(o[:,:,None],v-peak[:,:,None],0.)
    series=peak+10.*np.log10(np.power(10.,safe/10.).sum(2));series[~o]=np.nan
    return np.nanmax(series,1)-np.nanmedian(series,1)

def scalar_metrics(y,pred):
    import numpy as np
    y=np.asarray(y,float);p=np.asarray(pred['share'],float);s=np.asarray(pred['score'],float);e=p-y
    return {'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.abs(e).mean()),'bias':float(e.mean()),
            'prediction_mean':float(p.mean()),'prediction_rms':float(np.sqrt(np.mean(p*p))),
            'score_mean':float(s.mean()),'score_ge050_fraction':float(np.mean(s>=.5)),
            'target_mean':float(y.mean()),'target_rms':float(np.sqrt(np.mean(y*y)))}

def crossing(y,positive,negative):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y,float);p=np.asarray(positive['share'],float);n=np.asarray(negative['share'],float)
    z=np.r_[np.ones(len(y),int),np.zeros(len(n),int)]
    result={'positive_n':len(y),'negative_n':len(n),
            'rmse':float(np.sqrt((np.sum((p-y)**2)+np.sum(n*n))/(len(y)+len(n)))),
            'class_balanced_rmse':float(np.sqrt(.5*np.mean((p-y)**2)+.5*np.mean(n*n))),
            'positive_share_prediction_mean':float(p.mean()),'negative_share_prediction_mean':float(n.mean()),
            'positive_true_share_mean':float(y.mean())}
    for key in ['share','score']:
        a=np.asarray(positive[key],float);b=np.asarray(negative[key],float)
        delta=b[:,None]-a[None,:]
        av=float(roc_auc_score(z,np.r_[a,b]));inverse=float(np.mean(delta>0));tie=float(np.mean(delta==0))
        if abs(av+inverse+.5*tie-1)>1e-12:raise AssertionError('AUC inversion oracle')
        result['auc_'+key]=av;result['negative_gt_positive_'+key+'_fraction']=inverse
        result['tie_'+key+'_fraction']=tie
    return result

def mixture(bg,reference,target,receipt):
    import numpy as np
    from pipeline.sampler_r9 import energy_weights
    from pipeline.relative_grid_r6 import quantization_pooling_bound
    x=np.asarray(bg,dtype=float);o=np.isfinite(x);top=float(np.nanmax(x))
    b=np.where(o,10.**((np.where(o,x,top)-top)/10.),0.)
    r=np.asarray(reference,dtype=float);s=float(np.sum(o*r))
    if not 0<target<1 or s<=0:raise ValueError('Positive physical target/support required')
    alpha=float(target)*b.sum()/((1.-float(target))*s);a=alpha*r*o;e=b+a
    q=np.divide(a,e,out=np.zeros_like(a),where=o&(e>0));y=float(a.sum()/e.sum())
    raw=np.full((15,25),np.nan);raw[o]=10.*np.log10(e[o])+top
    db=np.round(raw,1).astype(np.float32);w=energy_weights(db)
    gap=float(abs(np.sum(w*q)-y));rawgap=float(abs(np.sum(e/e.sum()*q)-y))
    bound=quantization_pooling_bound(.05+float(np.max(np.abs(np.spacing(db[o]))))/2+1e-12)
    if rawgap>1e-10 or abs(y-target)>1e-10 or gap>bound:raise AssertionError('Physical/quantized label identity')
    return db,q.astype(np.float32),np.float32(y),{**receipt,'alpha':alpha,
        'active_added_minutes':int(np.sum(a.sum(1)>0)),'pure_background_minutes':int(np.sum(o.any(1)&(a.sum(1)==0))),
        'raw_label_identity_error':rawgap,'rounded_pool_gap':gap,'rounding_bound':bound}

def long_reference(native,spec,mode):
    import numpy as np
    rng=np.random.default_rng(spec['render_seed']);eq=eq_curve(rng,6.)
    r=np.asarray(native,dtype=float)*10.**(eq[None,:]/10.)
    out=np.zeros((15,25))
    if mode=='stationary15_control':out[:]=r.mean(0)
    elif mode=='native_long15':
        if len(r)<15:raise ValueError('Native long reference too short')
        begin=int(np.random.default_rng(spec['placement_seed']).integers(len(r)-14));out[:]=r[begin:begin+15]
    else:
        length=12 if mode.endswith('12') else 13 if mode.endswith('13') else 15
        pos=np.linspace(0,len(r)-1,length)
        segment=np.stack([np.interp(pos,np.arange(len(r)),r[:,j]) for j in range(25)],1)
        envelope=.6+.4*np.sin(np.pi*(np.arange(length)+.5)/length)**2
        start=int(np.random.default_rng(spec['placement_seed']).integers(16-length))
        out[start:start+length]=segment*envelope[:,None]
    return out,{'mode':mode,'native_minutes':len(r),'eq_min_db':float(eq.min()),'eq_max_db':float(eq.max()),
                'statistical_time_resampling_not_new_WAV':mode not in ['native_long15','stationary15_control'],
                'ideal_stationary_counterexample':mode=='stationary15_control'}

def fixture():
    import numpy as np
    bg=np.full((15,25),75.,np.float32);spec={'render_seed':1,'placement_seed':2}
    count=0
    for minutes in [1,2,4,15,23]:
        for mode in POS_MODES[1:]:
            if mode=='native_long15' and minutes<15:continue
            r,receipt=long_reference(np.ones((minutes,25)),spec,mode)
            db,q,y,z=mixture(bg,r,.3,receipt)
            if abs(float(y)-.3)>3e-8 or z['active_added_minutes']<12:raise AssertionError('Full-support fixture')
            count+=1
    return {'passed':True,'cases':count,'known_share_exact':True,'rounding_and_support_checked':True}

def run(a):
    import numpy as np,pandas as pd
    from pipeline.dataset_r9 import R9Sources
    from pipeline.source_split_r6 import apply_minute_qc
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    plan={'version':'r11_full_support_hf_v1','seed':SEED,'positive_per_mode':a.n,'max_negative_per_panel':a.n,
        'positive_modes':list(POS_MODES),'source_role':'Fold0 selection only, existing TRAIN group split+24h purge',
        'negative_panels':['HF2..8k above selection Nobs>=13 q90','actual StFrancis Nobs>=13 all up to N',
                           'original edge-candidate minute restored beforeQC','paired same-background edge QC control','HF quiet below q50'],
        'positive_background':'complete15min, unique, excludes union of negative background IDs',
        'share_policy':'same per-row share from positive R9 original baseline in each paired long mode; extra fixed .30 probe is explicitly separate',
        'long_curve':'whole source minute matrix linearly resampled, envelope .6+.4sin², no repeat waveform; native_long uses only actual>=15min clips',
        'support_policy':'13/15min main arms have <=2 pure-B minutes;12min arm has3 and is separately reported',
        'stationary_control':'native mean spectral vector constant15min; idealized observability counterexample, not claimed real passing ship',
        'negative_truth':'A=0 controlled-addition target, not proof of natural vessel absence or artifact status',
        'crossing':'each positive mode vs each negative panel; AUCshare/score,strict neg>pos pairwise rate,ties,raw/class-balancedRMSE',
        'do_not_select_by_public_or_model_errors':True,'Test_report_training_GPU_network_or_policy_changes':False,
        'script_sha256':sha(__file__)}
    write(out/'predeclared_plan.json',plan);write(out/'numeric_fixture.json',fixture())
    sources=R9Sources();sel=sources.role_metadata(0,'selection');train=sources.role_metadata(0,'train')
    if set(sel.base_window_index)&set(train.base_window_index) or set(sel.background_group)&set(train.background_group):raise AssertionError('TRAIN source overlap')
    if float(sel.minimum_other_role_support_gap_hours.min())<24:raise AssertionError('Support purge')
    bank=sources.background_bank();all_db=apply_minute_qc(np.asarray(bank[sel.base_window_index.to_numpy(int)]),sel.boundary_mask_bits.to_numpy(int))
    dense=sel[sel.qc_observed_minutes.ge(13)].copy();dense['hf_bump_2_8_db']=hf_bump(all_db[dense.index])
    q90=float(dense.hf_bump_2_8_db.quantile(.9));q50=float(dense.hf_bump_2_8_db.quantile(.5))
    high=choose(dense[dense.hf_bump_2_8_db.gt(q90)],a.n,'HFhigh')
    sf=choose(dense[dense.site.str.contains('francis',case=False)],a.n,'SFdense')
    edge_bits=sel.boundary_mask_bits.to_numpy(int)&((1<<0)|(1<<14))
    edge=choose(sel[(edge_bits>0)&sel.raw_observed_minutes.ge(13)],a.n,'edge')
    quiet=choose(dense[dense.hf_bump_2_8_db.lt(q50)],a.n,'HFquiet')
    panels={'hf_high':high,'st_francis_dense':sf,'edge_restored':edge,'edge_QC_paired':edge,'hf_quiet':quiet}
    excluded=set(int(v) for p in panels.values() for v in p.base_window_index)
    posframe=choose(sel[sel.qc_observed_minutes.eq(15)&~sel.base_window_index.isin(excluded)],a.n,'positiveBG')
    if len(posframe)!=a.n:raise ValueError('Insufficient disjoint complete positive backgrounds')
    rng=np.random.default_rng(SEED);refs=np.resize(rng.permutation(509),a.n)
    families=np.resize(np.array(['inside','inside','short','edge']),a.n);rng.shuffle(families)
    specs=[{'render_seed':int(rng.integers(2**31)),'placement_seed':int(rng.integers(2**31)),
            'warp_seed':0,'family':str(families[i]),'snr_db':float(rng.uniform(-10,10)),
            'no_source':False,'reference_index':int(refs[i])} for i in range(a.n)]
    longs=[i for i,m in enumerate(sources.reference_minutes) if len(m)>=15]
    if not longs:raise ValueError('No actual long native clips')
    source_evidence={'negative_quantile_population':len(dense),'HF_band_indices':list(range(18,25)),
        'HF_band_centers_hz':[float(10**(1.5+.1*j)) for j in range(18,25)],'hf_q90_db':q90,'hf_q50_db':q50,
        'panel_counts':{k:len(p) for k,p in panels.items()},
        'panel_site_counts':{k:p.site.value_counts().to_dict() for k,p in panels.items()},
        'panel_observed_minutes':{k:p.qc_observed_minutes.value_counts().sort_index().to_dict() for k,p in panels.items()},
        'positive_n':len(posframe),'positive_vs_negative_bg_overlap':0,'native_ge15_reference_count':len(longs),
        'native_ge15_reference_indices':longs,'reference_minutes_mean':float(np.mean([len(m) for m in sources.reference_minutes])),
        'all_reference_unique':509,'negative_panels_can_overlap_each_other':True,
        'TRAIN_group_and_window_overlap':0,'minimum_other_role_support_gap_hours':float(sel.minimum_other_role_support_gap_hours.min())}
    # Convert numpy dict keys before JSON serialization.
    source_evidence['panel_observed_minutes']={k:{str(i):int(v) for i,v in d.items()} for k,d in source_evidence['panel_observed_minutes'].items()}
    write(out/'source_panels.json',source_evidence);write(out/'positive_schedule.json',specs)
    posframe.to_parquet(out/'positive_background_manifest.parquet',index=False)
    arrays={};truth={};recipes={};q_arrays={}
    for name,frame in panels.items():
        if not len(frame):continue
        frame.to_parquet(out/(name+'_background_manifest.parquet'),index=False)
        raw=np.asarray(bank[frame.base_window_index.to_numpy(int)]).copy()
        qc=apply_minute_qc(raw,frame.boundary_mask_bits.to_numpy(int))
        if name=='edge_restored':
            for i,bits in enumerate(frame.boundary_mask_bits):
                for t in [0,14]:
                    if int(bits)&(1<<t):qc[i,t]=raw[i,t]
        arrays[name]=qc;truth[name]=np.zeros(len(qc),np.float32)
    backgrounds=[sources.background(r) for _,r in posframe.iterrows()]
    original=[render(bg,sources.reference_minutes[s['reference_index']],s,'r9_complete') for bg,s in zip(backgrounds,specs)]
    targets=np.array([r[2] for r in original],np.float32)
    if np.any(targets<=0):raise AssertionError('All full-support positives must be observed positive')
    arrays['r9_original']=np.stack([r[0] for r in original]);truth['r9_original']=targets
    q_arrays['r9_original']=np.stack([r[1] for r in original]);recipes['r9_original']=[r[3] for r in original]
    for mode in POS_MODES[1:]:
        records=[]
        for i,(bg,s) in enumerate(zip(backgrounds,specs)):
            ri=longs[i%len(longs)] if mode=='native_long15' else s['reference_index']
            reference,receipt=long_reference(sources.reference_minutes[ri],s,mode)
            target=.3 if mode=='full_smooth15_share030' else float(targets[i])
            records.append(mixture(bg,reference,target,{**receipt,'reference_index':int(ri)}))
        arrays[mode]=np.stack([r[0] for r in records]);truth[mode]=np.array([r[2] for r in records],np.float32)
        q_arrays[mode]=np.stack([r[1] for r in records]);recipes[mode]=[r[3] for r in records]
    for mode in arrays:
        np.savez_compressed(out/(mode+'_samples.npz'),db=arrays[mode],share=truth[mode],
                **({'cell_share':q_arrays[mode]} if mode in q_arrays else {}))
        if mode in recipes:write(out/(mode+'_recipes.json'),recipes[mode])
    write(out/'FROZEN_BEFORE_PREDICTIONS.json',{'plan_sha256':sha(out/'predeclared_plan.json'),
            'panel_file_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'fitting':False})
    nn=json.loads((ROOT/'ops/r9_fold0_pilot_nn_map.json').read_text());pins,modelpins,binding,tree=inputs(nn,ROOT/DEFAULT_TREE,ROOT/DEFAULT_BUNDLE,ROOT/DEFAULT_CONFIG)
    policy=ROOT/'submissions/r9_fold0_all5_cpu_r1/frozen_policy.json';pins[str(policy.relative_to(ROOT))]=sha(policy);weights=json.loads(policy.read_text())['weights']
    write(out/'source_model_pins.json',{'pins':pins,'modelpins':modelpins})
    # One joined query to avoid reloading each model for each diagnostic arm.
    order=list(arrays);offsets={};begin=0
    for mode in order:offsets[mode]=(begin,begin+len(arrays[mode]));begin+=len(arrays[mode])
    joined=np.concatenate([arrays[mode] for mode in order]);pred,timing=predict_members(nn,tree,joined,binding,batch_size=128,threads=a.threads)
    pred['all5_frozen']=combine(pred,weights);splitpred={};model_metrics={}
    for mode,(lo,hi) in offsets.items():
        splitpred[mode]={m:{k:np.asarray(p[k])[lo:hi] for k in ['share','score']} for m,p in pred.items()}
        np.savez_compressed(out/(mode+'_predictions.npz'),**{m+'_'+k:v for m,p in splitpred[mode].items() for k,v in p.items()})
        model_metrics[mode]={m:scalar_metrics(truth[mode],p) for m,p in splitpred[mode].items()}
    crossings=[]
    for positive in POS_MODES:
        for negative in panels:
            if negative not in splitpred:continue
            for model in pred:
                crossings.append({'positive_mode':positive,'negative_panel':negative,'model':model,
                    **crossing(truth[positive],splitpred[positive][model],splitpred[negative][model])})
    frame=pd.DataFrame(crossings);frame.to_csv(out/'crossing_metrics.csv',index=False)
    summaries={}
    for mode in arrays:
        z=arrays[mode];summary={'n':len(z),'HFbump_median':float(np.median(hf_bump(z))),
                'target_mean':float(truth[mode].mean()),'target_rms':float(np.sqrt(np.mean(truth[mode].astype(float)**2)))}
        if mode in q_arrays:
            active=(q_arrays[mode].sum(2)>0).sum(1)
            summary['active_added_minutes_histogram']={str(k):int(v) for k,v in zip(*np.unique(active,return_counts=True))}
            summary['true_q_temporal_max_minus_median_mean']=float(np.mean(np.max(q_arrays[mode],axis=1)-np.median(q_arrays[mode],axis=1)))
        summaries[mode]=summary
    sources.assert_unchanged();verify_pins(pins)
    result={'status':'completed','seconds':time.perf_counter()-start,'plan_sha256':sha(out/'predeclared_plan.json'),
            'source_panels':source_evidence,'panel_summary':summaries,'per_model_metrics':model_metrics,
            'crossing_metrics':'crossing_metrics.csv','timings':timing,'offsets':offsets,
            'production_sources_models_policy_unchanged':True,'Test_report_new_fit_GPU_used':False,
            'limits':['NewchallengepanelsarenotestimatedTestprevalenceorhiddenTestgenerator.',
                    'Resampling3minstatisticsto15mincancreatesynthetictraces;native_longusesfewindependentrecordings.',
                    'Counterexamplecollapsewouldsupportalocalfailuremode,nottheconfirmedPubliccause.',
                    'Negativepanelsdifferinsite,observationdensity,boundaryQC;matchedcontrolsareidentifiedseparately.']}
    write(out/'RESULTS.json',result)
    table=frame[frame.model.eq('all5_frozen')]
    print(table[['positive_mode','negative_panel','rmse','auc_score','auc_share','negative_gt_positive_score_fraction']].to_string(index=False),flush=True)
    print(json.dumps({'completed':str(out),'seconds':result['seconds'],'rows':len(joined)}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='reports/r11_diagnostics/full_support_hf_r1');p.add_argument('--n',type=int,default=500)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--self-test',action='store_true');a=p.parse_args();environment(a.threads)
    if a.self_test:print(json.dumps(fixture()));return
    run(a)
if __name__=='__main__':main()
