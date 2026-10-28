"""Read-only independent physics/scoring QA for the twelve R10 r2 panels.

Only this new QA report is written. No predictor is called and no original
report/Test array, model, data, policy, or producer output is modified.
"""
from __future__ import annotations
import hashlib,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.sampler_r9 import synthesize
from ops.r10_saved_prediction_audit import auc

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r10_diagnostics/factorial_transfer_r2'
ARMS=('r9_complete','mean_scale_only','smooth_trajectory_only','eq12_only','combined_stress')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def independent_eq(seed,maximum):
    rng=np.random.default_rng(seed);u=float(rng.uniform(-maximum,maximum));v=float(rng.uniform(-maximum/2,maximum/2))
    axis=np.linspace(-1.,1.,25);curve=u*axis+v*(axis**2-np.mean(axis**2))
    top=float(np.abs(curve).max())
    if top>maximum:curve=curve*(maximum/top)
    return curve


def linear_components(background,native,spec,arm,recipe,target=None):
    """Integer placement from receipt plus independent interpolation formula."""
    transformed=np.asarray(native,np.float64)*10.**(independent_eq(spec['render_seed'],12 if arm in ('eq12_only','combined_stress') else 6)[None]/10.)
    tr=recipe['trajectory'];m=len(native);begin=int(tr['crop_begin_minute']);length=int(tr['selected_minutes']);start=int(tr['window_start_minute'])
    assert int(tr['native_minutes'])==m and 0<=begin and begin+length<=m
    if spec['family']=='inside':assert length==min(15,m) and 0<=start<=15-length
    elif spec['family']=='short':assert 1<=length<=min(3,m) and 0<=start<=15-length
    else:
        assert length==m and begin==0
        if m>1:assert start<0 or start+m>15
    placed=np.zeros((15,25),float)
    for t in range(15):
        j=t-start
        if 0<=j<length:placed[t]=transformed[begin+j]
    if arm in ('smooth_trajectory_only','combined_stress'):
        active=np.flatnonzero(placed.sum(1)>0);center=(active[0]+active[-1])/2.
        speed=float(np.random.default_rng(spec['warp_seed']).uniform(.8,1.25))
        positions=np.maximum(0.,np.minimum(14.,center+(np.arange(15)-center)*speed))
        left=np.floor(positions).astype(int);right=np.minimum(left+1,14);fraction=(positions-left)[:,None]
        placed=placed[left]+fraction*(placed[right]-placed[left])
        ec=float(np.random.default_rng(spec['warp_seed']+1).uniform(-2,17));width=float(np.random.default_rng(spec['warp_seed']+2).uniform(3,10))
        placed*= (.15+.85*np.exp(-.5*((np.arange(15)-ec)/width)**2))[:,None]
    assert (placed>=0).all() and placed.sum()>0
    db=np.asarray(background,float);o=np.isfinite(db);assert o.all()
    peak=float(db.max());B=10.**((db-peak)/10.)
    snr=10.**(spec['snr_db']/10.)
    if target is not None:alpha=target*B.sum()/((1-target)*placed.sum())
    elif spec['no_source']:alpha=0.
    elif arm in ('mean_scale_only','combined_stress'):alpha=snr*B.sum()/placed.sum()
    else:alpha=snr*np.median(B.sum(1))/transformed.sum(1).max()
    A=alpha*placed;E=B+A;q=A/E;y=float(A.sum()/E.sum());raw=10*np.log10(E)+peak
    rounded=np.round(raw,1).astype(np.float32)
    weight=10.**((rounded.astype(float)-float(rounded.max()))/10.);weight/=weight.sum()
    half_ulp=float(np.abs(np.spacing(rounded)).max()/2.)
    bound=math.tanh(math.log(10)*(.05+half_ulp+1e-12)/20.)+2e-7
    assert abs(float((q*(E/E.sum())).sum())-y)<1e-10
    assert abs(float((q*weight).sum())-y)<=bound
    assert abs(alpha-recipe['alpha'])<=max(1e-12,abs(alpha)*2e-12)
    return rounded,q,y,bound


class CoupledProductionRng:
    """Same exogenous EQ/placement/SNR draws, not the original seed stream."""
    def __init__(self,spec):self.eq=np.random.default_rng(spec['render_seed']);self.place=np.random.default_rng(spec['placement_seed']);self.spec=spec;self.calls=0
    def uniform(self,lo,hi):
        self.calls+=1
        if self.calls<=2:return self.eq.uniform(lo,hi)
        assert (lo,hi)==(-10.,10.);return self.spec['snr_db']
    def integers(self,*args,**kwargs):return self.place.integers(*args,**kwargs)
    def choice(self,*args,**kwargs):return self.place.choice(*args,**kwargs)


def main():
    start=time.monotonic();report=ROOT/'reports/r10_diagnostics/factorial_independent_qa_r1.json'
    if report.exists():raise FileExistsError('Fresh independent QA report required')
    def load(path):return json.loads(Path(path).read_text())
    plan=load(OUT/'predeclared_plan.json');results=load(OUT/'RESULTS.json');schedule=load(OUT/'schedule.json');pins=load(OUT/'input_pins.json')
    generation_script=ROOT/'ops/r10_factorial_transfer_audit.py'
    assert sha(generation_script)==plan['script_sha256'] and sha(OUT/'predeclared_plan.json')==results['plan_sha256']
    expected={d+'__'+arm for d in ('seen','unseen') for arm in ARMS}|{'paired_ship','paired_ambient'}
    assert set(results['results'])==expected and len(schedule)==1000
    frozen_sources={str(generation_script.relative_to(ROOT)):sha(generation_script)}
    for n,h in pins['pins'].items():
        assert Path(n).name.lower() not in {'test.csv','report.npz','query_member_predictions.npz','submission.csv'}
        assert sha(ROOT/n)==h,n;frozen_sources[n]=h
    # Metadata-only schedule oracle, independent of saved metric values.
    rng=np.random.default_rng(2026102604+10);families=np.array(['inside']*500+['short']*250+['edge']*250);rng.shuffle(families)
    zeros=np.zeros(1000,bool);zeros[rng.choice(1000,200,replace=False)]=True
    refs=np.resize(np.arange(509),1000);rng.shuffle(refs)
    rebuilt=[{'render_seed':int(rng.integers(2**31)),'placement_seed':int(rng.integers(2**31)),
              'warp_seed':int(rng.integers(2**31)),'family':str(families[i]),'snr_db':float(rng.uniform(-10,10)),
              'no_source':bool(zeros[i]),'reference_index':int(refs[i])} for i in range(1000)]
    assert rebuilt==schedule
    selected_indices=sorted({0,999,int(np.flatnonzero(zeros)[0])}|
                             {next(i for i,s in enumerate(schedule) if s['family']==family and not s['no_source']) for family in ('inside','short','edge')})
    background=np.load(ROOT/'data/processed/sa_windows_db.npy',mmap_mode='r',allow_pickle=False)
    reference=np.load(ROOT/'data/processed/external_segments_power.npy',mmap_mode='r',allow_pickle=False)
    refmeta=pd.read_parquet(ROOT/'data/versions/vtuad_full_r9/reference_manifest.parquet').set_index('reference_row_index')
    ambient=pd.read_parquet(ROOT/'data/versions/vtuad_full_r9/ambient_control_manifest.parquet').set_index('ambient_row_index')
    def native(row,length=None):
        a=np.asarray(reference[int(row.segment_start):int(row.segment_start)+int(row.segment_count)],float)
        n=len(a)//30;value=a[:n*30].reshape(n,30,25).sum(1)/30.
        return value if length is None else value[:length]
    frames={d:pd.read_parquet(OUT/(d+'_source_manifest.parquet')) for d in ('seen','unseen')}
    train=pd.read_parquet(ROOT/'data/versions/background_cv_r9/fold_00/train_manifest.parquet')
    selection=pd.read_parquet(ROOT/'data/synthetic/group_cv_minute_full_r9_v1/fold_00/selection_metadata.parquet')
    source_review={}
    for domain,f in frames.items():
        assert len(f)==1000 and f.base_window_index.nunique()==1000 and f.qc_observed_minutes.eq(15).all()
        assert not set(f.base_window_index)&set(train.base_window_index)
        source_review[domain]={'site_counts':{str(k):int(v) for k,v in f.site.value_counts().items()},
            'TRAIN_base_window_overlap':0,'TRAIN_recorder_string_overlap':sorted(set(f.recorder)&set(train.recorder)),
            'TRAIN_sensor_component_overlap':sorted(map(int,set(f.sensor_group)&set(train.sensor_group))),
            'original_fixed_selection_background_window_overlap':len(set(f.base_window_index)&set(selection.base_window_index))}
    assert source_review['unseen']['TRAIN_recorder_string_overlap']==[] and source_review['unseen']['TRAIN_sensor_component_overlap']==[]
    metric_cases=[];oracle_cases=[];max_y_error=0.;max_db_error=0.;max_gap=0.;prod_equal=[]
    for name,metrics in results['results'].items():
        with np.load(OUT/name/'samples.npz',allow_pickle=False) as z:db=z['db'];y=z['share'].astype(float)
        with np.load(OUT/name/'predictions.npz',allow_pickle=False) as z:pred={k:z[k] for k in z.files}
        assert np.isfinite(db).all() and db.shape[1:]==(15,25) and np.abs(db*10-np.round(db*10)).max()<1e-3
        for kind,m in metrics.items():
            p=pred[kind+'_share'].astype(float);score=pred[kind+'_score'].astype(float)
            rmse=float(np.sqrt(np.sum((p-y)**2)/len(y)));a=auc(y>0,score)
            assert abs(rmse-m['rmse'])<1e-12
            assert a is None and m['auc_score'] is None or abs(a-m['auc_score'])<1e-12
            metric_cases.append({'panel':name,'model':kind,'rmse':rmse,'auc_score':a})
        if name.startswith('paired_'):continue
        domain,arm=name.split('__');meta=load(OUT/name/'metadata.json');f=frames[domain]
        assert meta['base_window_indices']==f.base_window_index.astype(int).tolist()
        assert meta['reference_indices']==refs.tolist() and meta['no_source']==zeros.tolist()
        assert np.array_equal(y==0,zeros)
        if arm in ('mean_scale_only','combined_stress'):
            expected_y=1/(1+10.**(-np.asarray([s['snr_db'] for s in schedule])/10.));expected_y[zeros]=0.
            assert np.max(np.abs(y-expected_y))<3e-8
        for i in selected_indices:
            row=f.iloc[i];bg=np.array(background[int(row.base_window_index)],copy=True)
            bits=int(row.boundary_mask_bits);bg[((bits>>np.arange(15))&1).astype(bool)]=np.nan
            assert np.isfinite(bg).all()
            s=schedule[i];r=native(refmeta.loc[s['reference_index']]);recipe=meta['recipes'][i]
            qdb,q,truth,bound=linear_components(bg,r,s,arm,recipe)
            de=float(np.abs(qdb.astype(float)-db[i].astype(float)).max());ye=abs(truth-y[i])
            assert de==0 and ye<3e-8
            max_y_error=max(max_y_error,ye);max_db_error=max(max_db_error,de)
            assert recipe['rounded_pool_label_gap']<=bound
            max_gap=max(max_gap,float(recipe['rounded_pool_label_gap']))
            oracle_cases.append({'panel':name,'index':i,'exact_db_match':True,'float32_target_error':ye,
                                 'independent_raw_q_y_identity_passed':True,'rounded_pool_theoretical_bound':bound})
            if arm=='r9_complete':
                production=synthesize(bg,None,CoupledProductionRng(s),family='native_'+s['family'],
                                      snr_range=(-10.,10.),eq_max_db=6.,no_source=s['no_source'],reference_minute_power=r)
                assert np.array_equal(production['db'],db[i]) and float(production['share'])==y[i]
                assert np.max(np.abs(production['cell_share'].astype(float)-q))<3e-8
                prod_equal.append({'domain':domain,'index':i,'coupled_exogenous_draws_db_share_exact':True,
                                   'same_seed_stream_as_production':False})
    pairmeta=load(OUT/'paired_ship/metadata.json')['pairs'];other=load(OUT/'paired_ambient/metadata.json')['pairs'];assert pairmeta==other
    lengths=pd.Series([m['matched_native_minutes'] for m in pairmeta]).value_counts().sort_index().to_dict()
    with np.load(OUT/'paired_ship/samples.npz') as z:ship_db=z['db'];ship_y=z['share']
    with np.load(OUT/'paired_ambient/samples.npz') as z:amb_db=z['db'];amb_y=z['share']
    assert np.array_equal(ship_y,amb_y) and (ship_y>0).all() and len(ship_y)==500
    pair_indices=sorted({0,499}|{next(i for i,m in enumerate(pairmeta) if m['matched_native_minutes']==length) for length in lengths})
    for i in pair_indices:
        m=pairmeta[i];bg=np.asarray(background[m['background']]);s={**schedule[i],'family':'inside','no_source':False}
        for role,ref,stored in [('ship',native(refmeta.loc[m['ship_reference']],m['matched_native_minutes']),ship_db),
                                ('ambient',native(ambient.loc[m['ambient_reference']],m['matched_native_minutes']),amb_db)]:
            qdb,q,y,bound=linear_components(bg,ref,s,'r9_complete',m[role+'_recipe'],m['target'])
            assert np.array_equal(qdb,stored[i]) and abs(y-float(ship_y[i]))<3e-8
            oracle_cases.append({'panel':'paired_'+role,'index':i,'exact_db_match':True,'float32_target_error':abs(y-float(ship_y[i])),
                                 'independent_target_alpha_solution_verified':True})
    initial=load(OUT/'OUTPUT_MANIFEST.json');corrections=load(OUT/'REVIEW_CORRECTIONS.json');authorized=[]
    for n,h in initial['files_sha256'].items():
        current=sha(OUT/n)
        if current!=h:
            assert n=='RESULTS.md' and h==corrections['before_md_sha256'] and current==corrections['after_md_sha256']
            authorized.append(n)
    assert all(sha(ROOT/n)==h for n,h in frozen_sources.items())
    result={'status':'passed','source_script_sha256':sha(__file__),'generation_script_matches_predeclared_SHA':True,
            'source_order_declares_plan_before_fixtures_sources_predictions':True,
            'plan_SHA_matches_RESULT':True,'all_frozen_input_pins_unchanged':True,
            'all_12_panels_scoring_recomputed':metric_cases,'source_review':source_review,
            'deterministic_shared_schedule_exact_replay':True,'explicit_zero_count_each_factorial_panel':200,
            'factorial_total_reference_pool':509,'factorial_positive_request_unique_references':int(len(set(refs[~zeros]))),
            'independent_B_A_oracle_cases':oracle_cases,'maximum_oracle_target_float32_error':max_y_error,
            'maximum_saved_rounded_db_difference':max_db_error,'production_formula_coupled_draw_checks':prod_equal,
            'baseline_parity_scope':'Same EQ/placement/SNR exogenous values reproduce production synthesize; distinct RNG streams, family count/reference schedule; not bitwise reproduction of original validation population',
            'mean_scale_oracle':'For positive all-observed examples, alpha=s*sumB/sumR and y=s/(1+s); exact on all4000 mean/combined panel rows including zero policy',
            'paired_actual_support_counts':{str(k):int(v) for k,v in lengths.items()},
            'paired_matched_y_and_same_background_EQ_support_verified':True,
            'paired_is_all_positive_AUC_undefined':True,
            'authorized_document_changes_since_initial_manifest':authorized,
            'review_corrections_sha256':sha(OUT/'REVIEW_CORRECTIONS.json'),
            'current_results_md_sha256':sha(OUT/'RESULTS.md'),
            'remaining_interpretation_limits':['Unseen935Saldhana/35FishHoek/30SeaPoint cannot establish all-domain generalization.',
                'Seen raw-background overlaps124 original selection windows; new mixtures, not entirely new background identities.',
                'Mean scale and unnormalized temporal warp both alter target magnitude/distribution; raw RMSE contrasts do not isolate texture shift.',
                'Temporal stress bundles bounded resampling with Gaussian envelope; EQ12 is a separate diagnostic stress, not production or known Test law.',
                'Mean-SNR/peak-SNR use different definitions and need not have equal target fractions or active-minute SNR.',
                'The unmodified finalizer gap threshold.012 is looser than theoretical~.005757; selected independent cases satisfy the stricter bound.',
                'MSE/RMSE/AUC contrasts and interactions are descriptive development effects, not a causal overfitting or memorization diagnosis.',
                'Paired ship/ambient are whole recordings with1..3minute prefixes and matched y; not verified isolated ship energy or hardware identification.',
                'The separately proposed matched-target followup is not numerically audited by this report.'],
            'original_production_data_code_models_policy_changed':False,
            'Test_original_report_or_new_model_inference_or_fit_or_server_accessed':False,
            'elapsed_seconds':time.monotonic()-start}
    with report.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'passed','report':str(report),'sha256':sha(report),'oracle_cases':len(oracle_cases),
                      'production_formula_checks':len(prod_equal),'metric_rows':len(metric_cases),'elapsed_seconds':result['elapsed_seconds']}))


if __name__=='__main__':main()
