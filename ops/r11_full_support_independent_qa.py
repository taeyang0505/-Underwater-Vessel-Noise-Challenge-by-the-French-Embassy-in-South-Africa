"""Read-only independent QA of already generated R11 long-support panels.

No predictor imports, model inference, fitting, Test, report-role access, or
changes to any input. Source bank numerical reads are confined to the saved
selection-source rows/reference slices. Only new QA reports are written.
"""
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'reports/r11_diagnostics/full_support_hf_r1'
OUTPUT=ROOT/'reports/r11_diagnostics/full_support_independent_qa_r1.json'
MODES=('r9_original','full_smooth12','full_smooth13','full_smooth15','stationary15_control','native_long15','full_smooth15_share030')
NEG=('hf_high','st_francis_dense','edge_restored','edge_QC_paired','hf_quiet')
MODEL=('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer','lgbm_physics','all5_frozen')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,Path)):
        s=str(args[0]).replace('\\','/');n=Path(s).name
        if n in ('Test.csv','report.npz','report_manifest.parquet','report_metadata.parquet','report_recipes.json'):
            raise PermissionError('R11 QA forbids Test/unused report-role input')
    if event in ('socket.connect','subprocess.Popen','os.system'):raise PermissionError('No external/process operations')


def qc(raw,bits):
    a=np.asarray(raw).copy()
    mask=((np.asarray(bits,np.int64)[:,None]>>np.arange(15))&1).astype(bool)
    a[mask]=np.nan
    return a


def histogram(a):return {str(k):int(v) for k,v in zip(*np.unique(a,return_counts=True))}


def eq(spec):
    rng=np.random.default_rng(spec['render_seed']);x=np.linspace(-1,1,25)
    curve=rng.uniform(-6,6)*x+rng.uniform(-3,3)*(x*x-np.mean(x*x))
    if max(abs(curve))>6:curve*=6/max(abs(curve))
    return 10**(curve/10)


def placement(r,spec):
    rng=np.random.default_rng(spec['placement_seed']);m=len(r);kind=spec['family']
    if kind=='short':
        length=int(rng.integers(1,min(3,m)+1));begin=int(rng.integers(m-length+1));part=r[begin:begin+length];start=int(rng.integers(16-length))
    elif kind=='inside':
        length=min(15,m);begin=int(rng.integers(m-length+1));part=r[begin:begin+length];start=int(rng.integers(16-length))
    elif kind=='edge':
        part=r;length=m;starts=np.arange(-m+1,15);starts=starts[(starts<0)|(starts+m>15)]
        start=int(rng.choice(starts)) if len(starts) else int(rng.integers(15))
    else:raise AssertionError('Unknown original family')
    a=np.zeros((15,25));lo,hi=max(0,start),min(15,start+length);a[lo:hi]=part[lo-start:hi-start]
    return a


def oracle(bg,native,spec,mode,target):
    """Independent linear B/A reconstruction, never calling the generator."""
    x=np.asarray(bg,float);o=np.isfinite(x);peak=np.nanmax(x);b=np.zeros_like(x);b[o]=10**((x[o]-peak)/10)
    r=native*eq(spec)[None,:]
    if mode=='r9_original':
        placed=placement(r,spec);alpha=10**(spec['snr_db']/10)*np.median(b.sum(1)[o.any(1)])/r.sum(1).max()
    else:
        placed=np.zeros((15,25))
        if mode=='stationary15_control':placed[:]=r.mean(0)
        elif mode=='native_long15':
            begin=int(np.random.default_rng(spec['placement_seed']).integers(len(r)-14));placed[:]=r[begin:begin+15]
        else:
            length=12 if mode.endswith('12') else 13 if mode.endswith('13') else 15
            source_axis=np.arange(len(r));query=np.linspace(0,len(r)-1,length)
            part=np.column_stack([np.interp(query,source_axis,r[:,f]) for f in range(25)])
            part*= (.6+.4*np.sin(np.pi*(np.arange(length)+.5)/length)**2)[:,None]
            start=int(np.random.default_rng(spec['placement_seed']).integers(16-length));placed[start:start+length]=part
        alpha=float(target)*b.sum()/((1-float(target))*np.sum(placed*o))
    a=alpha*placed*o;e=b+a;q=np.divide(a,e,out=np.zeros_like(a),where=o)
    y=float(a.sum()/e.sum());raw=np.full_like(x,np.nan);raw[o]=10*np.log10(e[o])+peak
    return np.round(raw,1).astype(np.float32),q.astype(np.float32),np.float32(y),alpha,float(abs(np.sum(q*e/e.sum())-y))


def main():
    started=time.monotonic()
    if OUTPUT.exists():raise FileExistsError('Fresh independent QA required')
    sys.addaudithook(guard)
    sourcepins=read(INPUT/'source_model_pins.json');pins=dict(sourcepins['pins'])
    for p in INPUT.iterdir():
        if p.is_file():pins[str(p.relative_to(ROOT))]=sha(p)
    for p in ['ops/r11_full_support_hf_audit.py','ops/r10_factorial_transfer_audit.py',
              'data/versions/background_cv_r9/manifest.json','data/versions/vtuad_full_r9/manifest.json',
              'data/versions/vtuad_full_r9/reference_manifest.parquet',
              'data/versions/background_cv_r9/fold_00/train_manifest.parquet',
              'data/versions/background_cv_r9/fold_00/selection_manifest.parquet']:
        pins[p]=sha(ROOT/p)
    plan=read(INPUT/'predeclared_plan.json');frozen=read(INPUT/'FROZEN_BEFORE_PREDICTIONS.json');result=read(INPUT/'RESULTS.json')
    assert plan['script_sha256']==sha(ROOT/'ops/r11_full_support_hf_audit.py')
    assert frozen['plan_sha256']==sha(INPUT/'predeclared_plan.json')==result['plan_sha256']
    assert all(sha(INPUT/p)==v for p,v in frozen['panel_file_sha256'].items())
    assert all(sha(ROOT/p)==v for p,v in sourcepins['pins'].items())
    dump(OUTPUT.with_name('full_support_independent_qa_r1_PLAN.json'),{
        'format':'r11_full_support_independent_qa_plan_v1','input_sha256':pins,'script_sha256':sha(__file__),
        'source_numeric_rows':'saved positive and negative selection-source indices only',
        'numeric_oracle_indices':[0,1,2,3,4,5,499],
        'new_inference_fit_or_Test_report_access':False})
    ref_manifest=read(ROOT/'data/versions/vtuad_full_r9/manifest.json');bg_manifest=read(ROOT/'data/versions/background_cv_r9/manifest.json')
    for m,key in [(ref_manifest,'bank'),(bg_manifest,'background_bank')]:
        assert sha(ROOT/m[key]['path'])==m[key]['sha256'];pins[m[key]['path']]=m[key]['sha256']
    refmeta=pd.read_parquet(ROOT/'data/versions/vtuad_full_r9/reference_manifest.parquet').sort_values('reference_row_index').reset_index(drop=True)
    assert len(refmeta)==509 and refmeta.reference_row_index.tolist()==list(range(509))
    longs=np.flatnonzero(refmeta.native_complete_minute_count.to_numpy()>=15).tolist()
    assert longs==result['source_panels']['native_ge15_reference_indices'] and len(longs)==7
    refbank=np.load(ROOT/ref_manifest['bank']['path'],mmap_mode='r',allow_pickle=False)
    bg_bank=np.load(ROOT/bg_manifest['background_bank']['path'],mmap_mode='r',allow_pickle=False)
    native={}
    def get_native(i):
        if i not in native:
            row=refmeta.iloc[i];a=np.asarray(refbank[int(row.segment_start):int(row.segment_start)+int(row.segment_count)])
            assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==row.expected_segment_slice_sha256
            n=len(a)//30;m=a[:n*30].astype(float).reshape(n,30,25).mean(1)
            assert n==int(row.native_complete_minute_count) and np.isfinite(m).all() and (m>=0).all()
            native[i]=m
        return native[i]
    positive_meta=pd.read_parquet(INPUT/'positive_background_manifest.parquet')
    train=pd.read_parquet(ROOT/'data/versions/background_cv_r9/fold_00/train_manifest.parquet')
    selection=pd.read_parquet(ROOT/'data/versions/background_cv_r9/fold_00/selection_manifest.parquet')
    assert not set(selection.base_window_index)&set(train.base_window_index)
    assert not set(selection.background_group)&set(train.background_group)
    assert float(selection.minimum_other_role_support_gap_hours.min())>=24
    assert len(positive_meta)==500 and positive_meta.base_window_index.is_unique and positive_meta.qc_observed_minutes.eq(15).all()
    source_hists={};negative_source={};samples={};pred={}
    for mode in list(NEG)+list(MODES):
        with np.load(INPUT/(mode+'_samples.npz'),allow_pickle=False) as z:samples[mode]={k:z[k] for k in z.files}
        with np.load(INPUT/(mode+'_predictions.npz'),allow_pickle=False) as z:pred[mode]={k:z[k] for k in z.files}
        s=samples[mode];o=np.isfinite(s['db']);assert np.array_equal(o.any(2),o.all(2)) and not np.isinf(s['db']).any()
        source_hists[mode]=histogram(o.all(2).sum(1))
        assert set(pred[mode])=={m+'_'+k for m in MODEL for k in ('share','score')}
        assert all(v.shape==(len(s['db']),) and np.isfinite(v).all() and ((v>=0)&(v<=1)).all() for v in pred[mode].values())
        if mode in NEG:
            f=pd.read_parquet(INPUT/(mode+'_background_manifest.parquet'));negative_source[mode]=f
            assert not set(positive_meta.base_window_index)&set(f.base_window_index)
            assert s['share'].shape==(len(f),) and not s['share'].any()
            raw=np.asarray(bg_bank[f.base_window_index.to_numpy(int)]);expected=qc(raw,f.boundary_mask_bits)
            if mode=='edge_restored':
                for j,bits in enumerate(f.boundary_mask_bits):
                    for t in (0,14):
                        if int(bits)&(1<<t):expected[j,t]=raw[j,t]
            assert np.array_equal(s['db'],expected,equal_nan=True)
    assert len(samples['st_francis_dense']['db'])==312 and source_hists['st_francis_dense']=={'15':312}
    assert negative_source['st_francis_dense'].site.str.contains('francis',case=False).all()
    assert source_hists['edge_restored']!={str(k):int(v) for k,v in negative_source['edge_restored'].qc_observed_minutes.value_counts().items()}
    assert np.array_equal(negative_source['edge_restored'].base_window_index,negative_source['edge_QC_paired'].base_window_index)
    specs=read(INPUT/'positive_schedule.json');targets=samples['r9_original']['share'];oracle_rows=[];raw_identity=[];pool_gaps=[]
    for mode in MODES:
        s=samples[mode];assert len(s['db'])==500 and (s['share']>0).all() and (s['share']<1).all()
        assert np.array_equal(s['share'],np.full(500,.3,np.float32) if mode=='full_smooth15_share030' else targets)
        q=s['cell_share'];assert q.shape==s['db'].shape and np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
        assert not q[~np.isfinite(s['db'])].any()
        e=10**((s['db'].astype(float)-np.nanmax(s['db'],axis=(1,2),keepdims=True))/10)
        e=np.nan_to_num(e);w=e/e.sum((1,2),keepdims=True)
        gap=np.abs(np.sum(w*q,(1,2))-s['share']);pool_gaps.extend(gap.tolist())
        assert gap.max()<=np.tanh(np.log(10)*(.05+1e-5)/20)+2e-7
        recipes=read(INPUT/(mode+'_recipes.json'))
        for i in [0,1,2,3,4,5,499]:
            row=positive_meta.iloc[i];bg=qc(np.asarray(bg_bank[int(row.base_window_index):int(row.base_window_index)+1]),[int(row.boundary_mask_bits)])[0]
            ri=longs[i%len(longs)] if mode=='native_long15' else specs[i]['reference_index']
            oracle_db,oracle_q,oracle_y,alpha,identity=oracle(bg,get_native(ri),specs[i],mode,.3 if mode=='full_smooth15_share030' else float(targets[i]))
            assert np.array_equal(oracle_db,s['db'][i],equal_nan=True) and np.array_equal(oracle_q,q[i]) and oracle_y==s['share'][i]
            assert abs(alpha-recipes[i]['alpha'])<=max(1,abs(alpha))*1e-12
            raw_identity.append(identity);oracle_rows.append({'mode':mode,'index':i,'reference_index':int(ri),'db_q_share_exact':True})
    # Independent pairwise AUC, inversion/tie and two RMSE definitions for ALL crossings.
    crossing=pd.read_csv(INPUT/'crossing_metrics.csv');assert len(crossing)==7*5*6 and not crossing.duplicated(['positive_mode','negative_panel','model']).any()
    errors=[]
    for row in crossing.itertuples(index=False):
        mode,neg,model=row.positive_mode,row.negative_panel,row.model;y=samples[mode]['share'].astype(float)
        p=pred[mode][model+'_share'].astype(float);n=pred[neg][model+'_share'].astype(float)
        expected={'rmse':np.sqrt((np.sum((p-y)**2)+np.sum(n*n))/(len(p)+len(n))),
                  'class_balanced_rmse':np.sqrt(.5*np.mean((p-y)**2)+.5*np.mean(n*n))}
        for k in ('share','score'):
            a=pred[mode][model+'_'+k].astype(float);b=pred[neg][model+'_'+k].astype(float)
            win=float((a[:,None]>b[None,:]).mean());tie=float((a[:,None]==b[None,:]).mean());inverse=float((b[:,None]>a[None,:]).mean())
            expected['auc_'+k]=win+.5*tie;expected['tie_'+k+'_fraction']=tie;expected['negative_gt_positive_'+k+'_fraction']=inverse
            assert abs(win+tie+inverse-1)<1e-12
        for k,v in expected.items():errors.append(abs(float(getattr(row,k))-float(v)))
    assert max(errors)<1e-12
    weights=np.asarray(read(ROOT/'submissions/r9_fold0_all5_cpu_r1/frozen_policy.json')['weights'],float)
    assert np.array_equal(weights,np.asarray([.1,.1,.1,.1,.6]))
    for mode in samples:
        for k in ('share','score'):
            expected=np.clip(np.column_stack([pred[mode][m+'_'+k] for m in MODEL[:5]]).astype(float)@weights,0,1).astype(np.float32)
            assert np.array_equal(expected,pred[mode]['all5_frozen_'+k])
    assert all(sha(ROOT/p)==v for p,v in pins.items())
    checks={'all_210_metric_rows_and_420_AUCs_pairwise_checked':True,'metric_max_abs_error':max(errors),
        'frozen_before_predictions_panel_SHA_exact':len(frozen['panel_file_sha256']),
        'source_model_policy_input_SHA_exact':len(sourcepins['pins']),
        'all_inputs_unchanged_count':len(pins),'all5_existing_weighted_saved_members_exact':True,
        'positive_backgrounds_unique_500_and_disjoint_all_negatives':True,
        'TRAIN_selection_background_groups_windows_disjoint_and_support_gap_hours':24,
        'real_native_ge15_minutes_reference_count':len(longs),
        'real_native_ge15_reference_indices':longs,
        'linear_B_A_oracle_rows':oracle_rows,'linear_B_A_oracle_max_raw_identity_error':max(raw_identity),
        'positive_saved_q_rounded_weight_label_gap_max':max(pool_gaps),
        'same_per_row_target_all_modes_except_declared_share030':True,
        'source_positive_sites':positive_meta.site.value_counts().to_dict(),
        'actual_panel_observed_minutes_histograms':source_hists,
        'edge_metadata_qc_histogram_NOT_restored_actual':histogram(negative_source['edge_restored'].qc_observed_minutes),
        'source_training_or_unused_report_or_Test_or_external_access':False,
        'new_model_inference_fit_GPU':False,'torch_imported':'torch' in sys.modules}
    dump(OUTPUT,{'status':'passed','checks':checks,'input_sha256':pins,'script_sha256':sha(__file__),
        'limitations':['Selection-source challenge sets, not hidden-query-law estimates.',
            'Native-long changes reference identity as well as duration; seven shared recordings are not 500 independent sources.',
            'Interpolated long-power and stationary controls are synthetic/idealized, not new measured long waveforms.',
            'Positive background sites exclude StFrancis whereas SF-negative is actual SF metadata: contrast is not source-matched.',
            'Edge restore includes candidate outlying minutes; A=0 means no controlled addition, not natural ship absence.',
            'Fixed .30 target arm is a distinct target-level intervention.',
            'Crossings share backgrounds/references and negative panels can overlap; 210 rows are not independent trials.'],
        'seconds':time.monotonic()-started})
    print(json.dumps({'status':'passed','output':str(OUTPUT),'SHA256':sha(OUTPUT),'seconds':time.monotonic()-started}))

if __name__=='__main__':main()
