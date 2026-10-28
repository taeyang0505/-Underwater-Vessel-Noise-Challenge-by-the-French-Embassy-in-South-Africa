"""Frozen Fold0 SF-density diagnostic, not a new training simulator.

Actual selection source windows only. Each unique B gets one A=0 and four
supported positive views. A fixed32-treatment bank has exactly the same
empirical reference/family/SNR/EQ distribution in sparse and dense panels.
Only placement is support-conditioned; paired masks retain dense A/B/alpha.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json,csv,gc,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r11_diagnostics/st_francis_density_r1'
SEED=2026102614
BOOT=200


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,v):
    with Path(p).open('x') as f:f.write(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def array_sha(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(json.dumps([str(a.dtype),list(a.shape)]).encode()+a.tobytes()).hexdigest()
def sample(frame,n,tag):
    keys=[hashlib.sha256(f'R11SF|{SEED}|{tag}|{g}|{v}'.encode()).hexdigest() for g,v in zip(frame.background_group,frame.base_window_index)]
    return frame.iloc[np.argsort(np.asarray(keys),kind='stable')[:n]].copy().reset_index(drop=True)
def safe_min(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)];return float(a.min()) if len(a) else None


def source_summary(sources):
    summaries={};sel=sources.role_metadata(0,'selection');train=sources.role_metadata(0,'train')
    original=pd.read_parquet(ROOT/'data/processed/sa_window_manifest.parquet')
    pool=pd.read_parquet(sources.bg_dir/'background_pool_manifest.parquet')
    for label,frame in [('original',original),('R9_allowed_pool',pool),('selection',sel)]:
        sf=frame[frame.site.astype(str).str.contains('Francis',case=False)]
        ncol='qc_observed_minutes' if 'qc_observed_minutes' in sf else 'observed_minutes'
        summaries[label]={'windows':len(sf),'Nobs_counts':{str(k):int(v) for k,v in sf[ncol].value_counts().sort_index().items()},
            'recorders':sorted(sf.recorder.astype(str).unique()),'sensor_groups':sorted(map(int,sf.sensor_group.unique()))}
    sf=sel[sel.site.astype(str).str.contains('Francis',case=False)].copy()
    assert not set(sf.base_window_index)&set(train.base_window_index)
    assert not set(sf.background_group)&set(train.background_group)
    roles={}
    for label,frame in [('sparse',sf[sf.qc_observed_minutes==5]),('dense',sf[sf.qc_observed_minutes>=13])]:
        times=[];patterns={}
        for row in frame.to_dict('records'):
            observed=np.isfinite(sources.background(row)).all(1);key=','.join(map(str,np.flatnonzero(observed)))
            patterns[key]=patterns.get(key,0)+1
            times.extend(pd.Timestamp(row['start_export_clock'])+pd.Timedelta(minutes=int(i)) for i in np.flatnonzero(observed))
        timeidx=pd.DatetimeIndex(sorted(set(times)));gaps=np.diff(timeidx.asi8)/60e9
        val,cnt=np.unique(gaps,return_counts=True);order=np.argsort(cnt)[::-1][:8]
        roles[label]={'windows':len(frame),'groups':int(frame.background_group.nunique()),
            'clock_range':[str(frame.start_export_clock.min()),str(frame.start_export_clock.max())],
            'recording_days':int(pd.to_datetime(frame.start_export_clock).dt.date.nunique()),
            'observed_minute_patterns':patterns,'top_observed_minute_start_gaps_minutes':{str(float(val[i])):int(cnt[i]) for i in order}}
    # Independently recompute nearest TRAIN support gap for the same sensor.
    gaps=[];duration=900*10**9
    for sensor,ev in sf.groupby('sensor_group'):
        tr=train[train.sensor_group==sensor]
        starts=np.sort(pd.to_datetime(tr.start_export_clock).astype('int64').to_numpy())
        for stamp in pd.to_datetime(ev.start_export_clock).astype('int64'):
            i=np.searchsorted(starts,stamp);c=[]
            if i:c.append(max(0,int(stamp)-int(starts[i-1])-duration)/3.6e12)
            if i<len(starts):c.append(max(0,int(starts[i])-int(stamp)-duration)/3.6e12)
            if c:gaps.append(min(c))
    if gaps and min(gaps)<24-1e-9:raise ValueError('Selection/TRAIN24h support purge violated')
    return sel,train,sf,{'source_scopes':summaries,'selection_regimes':roles,
        'TRAIN_group_or_window_overlap':0,'independent_minimum_TRAIN_support_gap_hours':min(gaps) if gaps else None,
        'clock_basis':'source export clock, not independently verified UTC',
        'cadence_note':'Sparse mask is a contiguous5-minute block rotating within15-minute bins; gaps1/16minutes match5-minute blocks every20minutes. Dense gaps1/46match15-minute blocks every60minutes. Deployment configuration itself was not supplied.'}


def compose(bg,placed,full_peak,snr,*,zero=False):
    from pipeline.sampler_r9 import energy_weights
    from pipeline.relative_grid_r6 import quantization_pooling_bound
    observed=np.isfinite(bg);peak=float(np.nanmax(bg));b=np.where(observed,np.power(10.,(np.asarray(bg,float)-peak)/10),0.)
    bmed=float(np.median(b.sum(1)[observed.any(1)]))
    alpha=0. if zero else float(10.**(snr/10)*bmed/full_peak)
    a=np.zeros_like(b) if zero else np.where(observed,alpha*placed,0.)
    return from_components(b,a,observed,peak),{'alpha':alpha,'background_observed_median_minute_power':bmed,
        'full_EQ_reference_minute_peak':None if zero else full_peak,
        'SNR_definition_error':None if zero else abs(10*np.log10(alpha*full_peak/bmed)-snr)}


def from_components(b,a,observed,peak):
    from pipeline.sampler_r9 import energy_weights
    from pipeline.relative_grid_r6 import quantization_pooling_bound
    b=np.where(observed,b,0.);a=np.where(observed,a,0.);e=b+a
    q=np.divide(a,e,out=np.zeros_like(e),where=observed);y=float(a.sum()/e.sum())
    unrounded=np.full((15,25),np.nan);unrounded[observed]=10*np.log10(e[observed])+peak
    db=np.round(unrounded,1).astype(np.float32);w=energy_weights(db)
    identity=abs(float((q*e).sum()/e.sum())-y)
    gap=abs(float((q*w).sum())-y)
    half_ulp=float(np.max(abs(np.spacing(db[observed])))/2)
    bound=quantization_pooling_bound(.05+half_ulp+1e-12)
    if identity>1e-12 or gap>bound+2e-7:raise ValueError('Physical target/rounding oracle failed')
    if y>0 and np.float32(y)==0:raise ValueError('Positive share underflow; do not relabel')
    return {'db':db,'share':np.float32(y),'presence':np.float32(y>0),'cell_share':q.astype(np.float32),
        'B':b,'A':a,'E':e,'peak':peak,'raw_identity_error':identity,'rounded_pool_gap':gap,'rounding_bound':bound}


def treatments(sources):
    from pipeline.sampler_r9 import smooth_eq
    rng=np.random.default_rng(np.random.SeedSequence([SEED,32]));refs=rng.choice(509,32,replace=False)
    families=np.asarray(['native_inside']*16+['native_short']*8+['native_edge']*8)[rng.permutation(32)]
    levels=-10+(np.arange(32)+rng.random(32))*20/32;levels=levels[rng.permutation(32)]
    table=[];power=[]
    for i,(ref,family,snr) in enumerate(zip(refs,families,levels)):
        eq=smooth_eq(np.random.default_rng(np.random.SeedSequence([SEED,i,0x4551])),6.)
        p=sources.reference_minutes[int(ref)]*10.**(eq[None,:]/10)
        row=sources.references.iloc[int(ref)]
        table.append({'treatment_id':i,'reference_index':int(ref),'reference_clip_id':str(row.clip_id),
            'reference_sha256':str(row.sha256),'weak_class':str(row.weak_class),'family':str(family),
            'snr_db':float(snr),'EQ_db':eq.tolist(),'full_EQ_minute_peak':float(p.sum(1).max()),
            'native_minutes':len(p),'reference_interpretation':'whole measured recording; not pure vessel ground truth'})
        power.append(p)
    return table,power


def supported_placement(power,treatment,observed,cache):
    from pipeline.sampler_r9 import place_minutes
    mask=tuple(np.flatnonzero(observed.any(1)));key=(treatment['treatment_id'],mask)
    if key in cache:return cache[key]
    for attempt in range(4096):
        rng=np.random.default_rng(np.random.SeedSequence([SEED,treatment['treatment_id'],attempt,0x504c]))
        placed,trajectory=place_minutes(power,rng,treatment['family'])
        if float(placed[observed].sum())>0:
            value=(placed,{'placement_candidates_attempted':attempt+1,'placement_trajectory':trajectory,
                'placement_seed':[SEED,treatment['treatment_id'],attempt,0x504c]})
            cache[key]=value;return value
    raise RuntimeError('No supported positive after4096prespecified placements; no treatment/seed replacement')


def masked_pair(value,base,view,patterns):
    first=int(hashlib.sha256(f'R11MASK|{SEED}|{base}|{view}'.encode()).hexdigest()[:8],16)%3
    for attempt in range(3):
        idx=(first+attempt)%3;observed=np.zeros((15,25),bool);observed[patterns[idx]]=True
        if value['presence']==0 or value['A'][observed].sum()>0:
            result=from_components(value['B'],value['A'],observed,value['peak'])
            assert np.array_equal(result['db'][observed],value['db'][observed])
            return result,{'mask_pattern_id':idx,'mask_minutes':list(map(int,patterns[idx])),
                'mask_candidates_attempted':attempt+1,'same_dense_A_B_alpha_retained':True}
    raise RuntimeError('No supported paired mask among all3real patterns')


def generate(sources,frame,table,power,*,paired=False):
    n=len(frame)*5;keys=['db','share','presence','cell_share'];data={k:[] for k in keys};recipes=[]
    pair={k:[] for k in keys};pair_recipes=[];cache={};patterns=[np.arange(0,5),np.arange(5,10),np.arange(10,15)]
    positive_index=0
    for row in frame.to_dict('records'):
        bg=sources.background(row);observed=np.isfinite(bg)
        if paired and observed.all()!=True:raise ValueError('Paired mask requires actual full15-minute dense source')
        for view in range(5):
            zero=view==0;t=None if zero else table[positive_index%32]
            if zero:
                value,physics=compose(bg,np.zeros((15,25)),1.,0.,zero=True);placement={'placement_candidates_attempted':0}
            else:
                placed,placement=supported_placement(power[t['treatment_id']],t,observed,cache)
                value,physics=compose(bg,placed,t['full_EQ_minute_peak'],t['snr_db'])
                if value['presence']!=1:raise ValueError('Supported positive became zero')
                positive_index+=1
            recipe={'base_window_index':int(row['base_window_index']),'view':view,'explicit_no_source':zero,
                'treatment_id':-1 if zero else t['treatment_id'],'known_share':float(value['share']),
                'observed_minutes':int(observed.any(1).sum()),**placement,**physics,
                'raw_identity_error':value['raw_identity_error'],'rounded_pool_gap':value['rounded_pool_gap'],
                'rounding_bound':value['rounding_bound']}
            for k in keys:data[k].append(value[k])
            recipes.append(recipe)
            if paired:
                p,maskinfo=masked_pair(value,int(row['base_window_index']),view,patterns)
                for k in keys:pair[k].append(p[k])
                pair_recipes.append({**recipe,**maskinfo,'known_share':float(p['share']),
                    'observed_minutes':5,'raw_identity_error':p['raw_identity_error'],
                    'rounded_pool_gap':p['rounded_pool_gap'],'rounding_bound':p['rounding_bound']})
    data={k:np.asarray(v) for k,v in data.items()}
    pair={k:np.asarray(v) for k,v in pair.items()} if paired else None
    assert len(data['share'])==n and int((data['presence']==0).sum())==len(frame)
    assert float((data['presence']==0).mean())==.2
    if paired:assert np.array_equal(data['presence'],pair['presence'])
    return data,recipes,pair,pair_recipes


def auc(z,p):
    z=np.asarray(z,bool);p=np.asarray(p,float);ix=np.argsort(p,kind='stable');s=p[ix];rank=np.empty(len(p),float);i=0
    while i<len(p):
        j=i+1
        while j<len(p) and s[j]==s[i]:j+=1
        rank[ix[i:j]]=(i+1+j)/2;i=j
    return float((rank[z].sum()-z.sum()*(z.sum()+1)/2)/(z.sum()*(~z).sum())) if z.any() and (~z).any() else None


def metric(data,pred,bootstrap):
    y=data['share'].astype(float);p=pred['share'].astype(float);score=pred['score'].astype(float);err=p-y;zero=y==0
    row_sse=(err*err).reshape(-1,5).mean(1);row_bias=err.reshape(-1,5).mean(1)
    brmse=np.sqrt(row_sse[bootstrap].mean(1));bbias=row_bias[bootstrap].mean(1)
    return {'n':len(y),'unique_B':len(y)//5,'RMSE':float(np.sqrt(np.mean(err*err))),
        'MAE':float(abs(err).mean()),'bias':float(err.mean()),'AUC_share':auc(~zero,p),'AUC_score':auc(~zero,score),
        'zero_n':int(zero.sum()),'zero_pred_mean':float(p[zero].mean()),
        'zero_RMSE':float(np.sqrt(np.mean(p[zero]**2))),'zero_score_ge_.5_rate':float((score[zero]>=.5).mean()),
        'positive_n':int((~zero).sum()),'positive_RMSE':float(np.sqrt(np.mean(err[~zero]**2))),
        'sourceID_cluster_bootstrap_RMSE_SE':float(brmse.std(ddof=1)),
        'sourceID_cluster_bootstrap_RMSE_CI95':np.quantile(brmse,[.025,.975]).tolist(),
        'sourceID_cluster_bootstrap_bias_SE':float(bbias.std(ddof=1))},brmse


def main():
    from ops.infer_r9_fold0_all5_pilot import environment,verify_pins,inputs,predict_members,combine,MEMBERS
    start=time.perf_counter()
    if OUT.exists():raise FileExistsError('Fresh audit directory required')
    environment(2);pilot=ROOT/'submissions/r9_fold0_all5_cpu_r1';old=json.loads((pilot/'prequery_plan.json').read_text())
    policy=json.loads((pilot/'frozen_policy.json').read_text());verify_pins(old['source_pins'])
    if canonical({k:v for k,v in policy.items() if k!='policy_sha256'})!=policy['policy_sha256']:raise ValueError('Frozen policy changed')
    from pipeline.dataset_r9 import R9Sources
    sources=R9Sources(ROOT/old['config'],project_root=ROOT,verify_banks=True)
    sel,train,sf,feasible=source_summary(sources)
    sparse=sample(sf[sf.qc_observed_minutes==5],1000,'sparse')
    dense=sample(sf[sf.qc_observed_minutes>=13],1000,'dense')
    if not len(sparse) or not len(dense):raise ValueError('No actual source support; do not fabricate dense windows')
    if len(sparse)*4%32 or len(dense)*4%32:raise ValueError('Prespecified32-treatment exact distribution cannot be balanced; stop')
    OUT.mkdir(parents=True);save(OUT/'source_feasibility.json',feasible)
    plan={'format':'r11_SF_density_predeclared_v1','created_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,
        'source_role':'Fold0selection only','source_feasibility_sha256':sha(OUT/'source_feasibility.json'),
        'unique_B_sparse':len(sparse),'unique_B_dense':len(dense),'views_per_B':5,'per_B':'1explicitA0+4supportedpositives',
        'generated_counts':{'sparse':len(sparse)*5,'dense':len(dense)*5,'dense_mask5':len(dense)*5},
        'exact_explicit_zero_fraction':.2,'no_fake_dense_or_stitching':True,
        'treatment_bank':{'count':32,'unique_references':32,'from_current_admitted509':True,
            'families':{'native_inside':16,'native_short':8,'native_edge':8},'SNR':'fixed32LHS strataUniform[-10,10]',
            'smooth_EQ_max_db':6.,'empirical_reference_family_SNR_EQ_distribution_exact_same_across_main_groups':True,
            'small_bank_limit':'conditional experiment on32fixed treatments, not generalization over all509'},
        'positive_overlap':'placement only retried, same reference/family/SNR/EQ/fullnativepeak; max4096; recordeveryattemptcount',
        'paired_mask':'same denseB/A/alpha, delete to one real5-minute consecutiveblock; firsthashpattern then cyclicretry over3patterns ifpositiveAunobserved; max3',
        'clock_confound':'sparse2016-2018 vsdense2015; paired dense deletion separately isolates intervention within dense sources',
        'sourceID_cluster_bootstrap':{'unit':'base_window_index, all5views together','replicates':BOOT,
            'conditional_on':'fixed32treatments; does not account for all week/station autocorrelation'},
        'metrics':['RMSE','MAE','bias','AUCshare','AUCscore','zeros','targetmoments','paired RMSE difference'],
        'precision':'CPUFP32, NN2threads; treefrozenAPI4threads','frozen_weights':policy['weights'],
        'original_source_pins':old['source_pins'],'policy_file_sha256':sha(pilot/'frozen_policy.json'),
        'code_sha256':sha(__file__),'Test_report_training_weight_change':False}
    save(OUT/'plan.json',plan)
    def guard(event,args):
        if event=='socket.connect':raise PermissionError('Network prohibited')
        if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
            if Path(args[0]).name.lower() in ['test.csv','report.npz','report_metadata.parquet','report_recipes.json']:
                raise PermissionError('Test/report forbidden')
    sys.addaudithook(guard)
    # Scalar energy and paired-mask oracles, before actual generation.
    b=np.ones((15,25));a=np.zeros_like(b);a[:3]=1;obs=np.ones_like(b,bool)
    v=from_components(b,a,obs,0.);assert abs(float(v['share'])-1/6)<1e-8
    p,pr=masked_pair(v,17,1,[np.arange(0,5),np.arange(5,10),np.arange(10,15)])
    assert abs(float(p['share'])-.375)<1e-8
    z=from_components(b,np.zeros_like(b),obs,0.);assert z['share']==0 and z['presence']==0
    table,power=treatments(sources);save(OUT/'treatment_bank.json',table)
    sparse.to_parquet(OUT/'sparse_sources.parquet',index=False);dense.to_parquet(OUT/'dense_sources.parquet',index=False)
    sparse_data,sparse_recipes,_,_=generate(sources,sparse,table,power)
    dense_data,dense_recipes,pair_data,pair_recipes=generate(sources,dense,table,power,paired=True)
    panels={'sparse':sparse_data,'dense':dense_data,'dense_mask5':pair_data}
    recipes={'sparse':sparse_recipes,'dense':dense_recipes,'dense_mask5':pair_recipes}
    counts={}
    for name,data in panels.items():
        r=recipes[name];t=np.asarray([z['treatment_id'] for z in r]);counts[name]=np.bincount(t[t>=0],minlength=32)
        assert counts[name].min()==counts[name].max()
        np.savez_compressed(OUT/(name+'_known_components.npz'),**data)
        with (OUT/(name+'_recipes.jsonl')).open('x') as f:
            for row in r:f.write(json.dumps(row,allow_nan=False)+'\n')
    save(OUT/'generation_gate.json',{'status':'passed','exact_zero_fraction_each':.2,
        'treatment_counts_per_id':{k:v.tolist() for k,v in counts.items()},'physical_oracles_passed':True,
        'maximum_raw_identity_error':max(x['raw_identity_error'] for rr in recipes.values() for x in rr),
        'maximum_rounded_weight_target_gap':max(x['rounded_pool_gap'] for rr in recipes.values() for x in rr),
        'label_arrays_fixed_before_any_predictions':True})
    pins,_,binding,tree=inputs(old['nn_dirs'],ROOT/old['tree_job'],ROOT/old['bundle'],ROOT/old['config'])
    if pins!=old['source_pins'] or binding!=old['binding']:raise ValueError('Frozen model binding changed')
    predictions={};timings={}
    for name,data in panels.items():
        pred,timing=predict_members(old['nn_dirs'],tree,data['db'],binding,batch_size=128,threads=2)
        pred['all5_frozen']=combine(pred,policy['weights']);predictions[name]=pred;timings[name]=timing
        np.savez_compressed(OUT/(name+'_predictions.npz'),share=np.stack([pred[k]['share'] for k in list(MEMBERS)+['all5_frozen']],1),
            score=np.stack([pred[k]['score'] for k in list(MEMBERS)+['all5_frozen']],1))
        print(json.dumps({'panel':name,'generated_rows':len(data['share']),'unique_B':len(data['share'])//5,'status':'predicted'}),flush=True)
    bs={n:np.random.default_rng(np.random.SeedSequence([SEED,0x4253,i])).integers(0,len(p['share'])//5,(BOOT,len(p['share'])//5))
        for i,(n,p) in enumerate(panels.items())};bs['dense_mask5']=bs['dense']
    metrics={};boot={};moments={};support={}
    for name,data in panels.items():
        metrics[name]={};boot[name]={};y=data['share'].astype(float)
        moments[name]={'n':len(y),'unique_B':len(y)//5,'explicit_zero_n':int((y==0).sum()),'mean':float(y.mean()),
            'std_ddof0':float(y.std()),'q10_q50_q90':np.quantile(y,[.1,.5,.9]).tolist()}
        rr=[x for x in recipes[name] if not x['explicit_no_source']]
        attempts=[x['placement_candidates_attempted'] for x in rr]
        support[name]={'placement_attempts_mean':float(np.mean(attempts)),'placement_attempts_max':int(max(attempts)),
            'placement_retry_rows':int(sum(v>1 for v in attempts)),
            'mask_retry_rows':int(sum(x.get('mask_candidates_attempted',1)>1 for x in rr)),
            'conditional_selection_bias':'positive source placement / paired mask is conditioned on observed A support; these are not unconditional natural duty-cycle effects'}
        for kind,pred in predictions[name].items():metrics[name][kind],boot[name][kind]=metric(data,pred,bs[name])
    differences={}
    for kind in list(MEMBERS)+['all5_frozen']:
        real=metrics['sparse'][kind]['RMSE']-metrics['dense'][kind]['RMSE'];paired=metrics['dense_mask5'][kind]['RMSE']-metrics['dense'][kind]['RMSE']
        delta=boot['dense_mask5'][kind]-boot['dense'][kind]
        differences[kind]={'real_sparse_minus_dense_RMSE':real,'paired_mask5_minus_dense_RMSE':paired,
            'paired_delta_sourceID_cluster_SE':float(delta.std(ddof=1)),
            'paired_delta_sourceID_cluster_CI95':np.quantile(delta,[.025,.975]).tolist(),
            'real_density_ordering':('sparse higher error' if real>0 else 'dense higher error'),
            'real_comparison_confounded_by_period_and_acoustics':True,'paired_target_recomputed_not_held_constant':True}
    sources.assert_unchanged(verify_banks=True);verify_pins(pins)
    if sha(pilot/'frozen_policy.json')!=plan['policy_file_sha256']:raise ValueError('Frozen weights changed')
    result={'format':'r11_SF_density_audit_v1','status':'completed','plan_sha256':sha(OUT/'plan.json'),
        'source_feasibility':feasible,'metrics':metrics,'target_moments':moments,'differences':differences,
        'support_conditioning':support,'model_timings':timings,'member_order':list(MEMBERS)+['all5_frozen'],
        'component_npz_sha256':{n:sha(OUT/(n+'_known_components.npz')) for n in panels},
        'prediction_npz_sha256':{n:sha(OUT/(n+'_predictions.npz')) for n in panels},
        'all_original_source_model_policy_pins_unchanged':True,'Test_report_newfit_network_upload':False,
        'limitations':['unique sources1000sparse/312dense; five views per source are clustered replicates, not new independent recordings',
            'only32fixed treatments shared; conditional bootstrap does not generalize across all509references',
            'real sparse and dense have different years and6/2source groups, so density alone is not identified',
            'paired deletion retains dense A/B/alpha and recomputes observed target; it changes target energy support',
            'support-conditioned placement/mask retries induce an explicit selection bias',
            'selection-source diagnostic, not report/OOF/officialTest; natural vessel presence is not GT'],
        'elapsed_seconds':time.perf_counter()-start,'code_sha256':sha(__file__)}
    save(OUT/'results.json',result)
    flat=[]
    for panel,models in metrics.items():
        for kind,v in models.items():flat.append({'panel':panel,'model':kind,**v})
    with (OUT/'metrics.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps({'status':'completed','results_sha256':sha(OUT/'results.json'),'differences':differences,
                      'seconds':result['elapsed_seconds']},indent=2))


if __name__=='__main__':main()
