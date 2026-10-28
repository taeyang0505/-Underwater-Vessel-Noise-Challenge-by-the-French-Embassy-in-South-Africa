"""R15 descriptive 25-band audit. No fit, synthesis, model or policy imports."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,sys,time,warnings
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TAGS=('31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5','251.2','316.2','398.1','501.2','631','794.3','1000','1258.9','1584.9','1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3')
FREQ=np.asarray(TAGS,float)
GROUPS={'LF_le200':list(range(9)),'MF_300_3000':list(range(10,20)),
        'HF_3000_8000':list(range(20,25)),'MFHF_300_8000':list(range(10,25)),
        'LF_EDA_le520':list(range(13)),'SAFE_1000_5012':list(range(15,23)),
        'ALL25':list(range(25))}
NAMES=('max_minus_median','max_minus_q20','median_minus_window_median',
       'median_minus_band_median_center','max_minus_min')
SEED=202610271502

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def handle_sha(f):
    f.seek(0);h=hashlib.sha256()
    for block in iter(lambda:f.read(2**20),b''):h.update(block)
    f.seek(0);return h.hexdigest()
def dump(p,value):Path(p).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summary(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    return {'n_valid':len(a),'p10':float(np.quantile(a,.1)) if len(a) else None,
            'p50':float(np.quantile(a,.5)) if len(a) else None,
            'p90':float(np.quantile(a,.9)) if len(a) else None}
def temporal_quantile(sorted_values,counts,q):
    pos=q*np.maximum(counts-1,0);lo=np.floor(pos).astype(np.int64);hi=np.minimum(lo+1,np.maximum(counts-1,0))
    a=np.take_along_axis(sorted_values,lo[:,None,:],1)[:,0,:]
    b=np.take_along_axis(sorted_values,hi[:,None,:],1)[:,0,:]
    return np.where(counts>0,a*(1-(pos-lo))+b*(pos-lo),np.nan)

def features(db,chunk=2048):
    n=len(db);band={name:np.empty((n,25),np.float32) for name in NAMES}
    grouped={g:{name:np.empty(n,np.float32) for name in ('power_max_minus_median','power_max_minus_q20','power_range','valid_minutes')}
             for g in GROUPS}
    nobs=np.empty(n,np.int16);counts=np.empty((n,25),np.int16)
    for start in range(0,n,chunk):
        end=min(start+chunk,n);d=np.asarray(db[start:end],np.float64);o=np.isfinite(d)
        if np.isinf(d).any():raise ValueError('Infinite stored dB forbidden')
        srt=np.sort(d,axis=1);cnt=o.sum(1);counts[start:end]=cnt;nobs[start:end]=o.all(2).sum(1)
        med=temporal_quantile(srt,cnt,.5);q20=temporal_quantile(srt,cnt,.2)
        maximum=temporal_quantile(srt,cnt,1.);minimum=temporal_quantile(srt,cnt,0.)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            window=np.nanmedian(d.reshape(len(d),-1),axis=1)
            centre=np.nanmedian(med,axis=1)
        vals=(maximum-med,maximum-q20,med-window[:,None],med-centre[:,None],maximum-minimum)
        for name,v in zip(NAMES,vals):band[name][start:end]=v
        energy=np.where(o,np.power(10.,np.where(o,d,0.)/10.),0.)
        if not np.isfinite(energy).all():raise ValueError('Energy overflow')
        for group,indices in GROUPS.items():
            # A group level requires all its frequencies, preserving observed
            # masks. EDA legacy variants additionally require all25 bands.
            valid=o[:,:,indices].all(2)
            if group in ('LF_EDA_le520','HF_3000_8000'):valid &=o.all(2)
            level=np.where(valid,10*np.log10(np.maximum(energy[:,:,indices].sum(2),np.finfo(float).tiny)),np.nan)
            ss=np.sort(level[:,:,None],axis=1);cc=valid.sum(1)[:,None]
            mx=temporal_quantile(ss,cc,1.)[:,0];md=temporal_quantile(ss,cc,.5)[:,0]
            qq=temporal_quantile(ss,cc,.2)[:,0];mn=temporal_quantile(ss,cc,0.)[:,0]
            for name,v in zip(('power_max_minus_median','power_max_minus_q20','power_range','valid_minutes'),(mx-md,mx-qq,mx-mn,cc[:,0])):
                grouped[group][name][start:end]=v
    return band,grouped,counts,nobs

def load_train(path,metadata):
    p=ROOT/path;mp=ROOT/metadata;raw=mp.read_bytes();m=json.loads(raw)
    expected=m.get('npz_sha256',m.get('NPZ_sha256'))
    if expected is None:raise ValueError('Source manifest does not pin NPZ')
    with p.open('rb') as f:
        before=handle_sha(f)
        if before!=expected:raise ValueError('Stored TRAIN differs from source receipt')
        with np.load(f,allow_pickle=False) as z:
            db=z['db'];y=z['share'];presence=z['presence'];ids=z['sample_id'].astype(str)
        after=handle_sha(f)
    if before!=after or mp.read_bytes()!=raw:raise ValueError('Source changed during read')
    if db.shape!=(len(y),15,25) or len(set(ids))!=len(ids) or not np.array_equal(y>0,presence.astype(bool)):
        raise ValueError('Stored source shape/labels/order invalid')
    return db,y,ids,{'path':path,'sha256':before,'metadata_path':metadata,'metadata_sha256':hashlib.sha256(raw).hexdigest(),
        'metadata_snapshot':m,'n':len(y),'stored_db_dtype':str(db.dtype),'known_added_positive':int((y>0).sum()),
        'known_added_zero':int((y==0).sum()),'read_handle_before_after_identical':True,
        'rows_are_synthetic_mixture_windows':True,'physical_share_label_mean':float(y.mean())}

def load_test():
    p=ROOT/'data/raw/Test.csv';before=sha(p)
    names=[f'm{minute:02d}_TOL_{tag}' for minute in range(1,16) for tag in TAGS]
    with p.open(newline='') as f:
        reader=csv.reader(f);header=next(reader)
        if header!=['ID',*names]:raise ValueError('Official375-column schema/order differs')
        rows=list(reader)
    ids=np.asarray([r[0] for r in rows]);db=np.asarray([[float(v) if v.strip() else np.nan for v in r[1:]] for r in rows],float).reshape(-1,15,25)
    if len(ids)!=1500 or len(set(ids))!=1500 or sha(p)!=before:raise ValueError('Query identity/count/immutability failed')
    return db,None,ids,{'path':'data/raw/Test.csv','sha256':before,'n':len(ids),'stored_db_dtype':'CSV numeric parsed float64',
        'physical_share_labels_available':False,'not_a_ship_presence_ground_truth':True,'read_before_after_identical':True}

def fixture():
    rng=np.random.default_rng(42);x=rng.normal(90,8,(11,15,25));x[0,:4]=np.nan;x[1,:,24]=np.nan
    b,g,c,n=features(x,4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        oracle={'max_minus_median':np.nanmax(x,1)-np.nanmedian(x,1),
                'max_minus_q20':np.nanmax(x,1)-np.nanquantile(x,.2,axis=1),
                'median_minus_window_median':np.nanmedian(x,1)-np.nanmedian(x.reshape(11,-1),1)[:,None],
                'median_minus_band_median_center':np.nanmedian(x,1)-np.nanmedian(np.nanmedian(x,1),1)[:,None],
                'max_minus_min':np.nanmax(x,1)-np.nanmin(x,1)}
    errors={k:float(np.nanmax(abs(b[k]-v))) for k,v in oracle.items()}
    if max(errors.values())>4e-6:raise AssertionError(errors)
    a,_,_,_=features(x+20.,4)
    assert all(np.allclose(a[k],b[k],atol=2e-6,rtol=0,equal_nan=True) for k in NAMES)
    offset=rng.normal(0,10,25);a,_,_,_=features(x+offset,4)
    assert np.allclose(a['max_minus_median'],b['max_minus_median'],atol=2e-6,rtol=0,equal_nan=True)
    assert np.allclose(a['max_minus_q20'],b['max_minus_q20'],atol=2e-6,rtol=0,equal_nan=True)
    return {'status':'passed','quantiles_vs_numpy_max_error':errors,'common_gain_invariance':True,
            'per_band_static_gain_temporal_bump_invariance':True,'profile_per_band_gain_invariance_claimed':False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--r14-path',default='models/r14_production_nn_fold0_r1/raw_cache/epoch_038.npz');a=ap.parse_args()
    out=ROOT/'reports/r15_band_comparison_r1'
    if out.exists():raise FileExistsError('Fresh descriptive audit required; no overwrite')
    out.mkdir(parents=True);start=time.perf_counter();own=sha(__file__)
    inputs={'Test':'data/raw/Test.csv','R9_TRAIN':'models/r9_mac_trees_r2/fold_00/tree_epoch1_train.npz','R14_TRAIN':a.r14_path}
    plan={'version':'r15_band_descriptive_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_sha256':own,
       'input_paths':inputs,'seed':SEED,'sample_n_per_dataset':1500,'complete_training_populations':True,'chunk':2048,
       'no_fitting_synthesis_or_training':True,'no_distribution_matching_or_new_calibration':True,
       'bands_hz':FREQ.tolist(),'groups_zero_based':GROUPS,'group_boundary_note':'251.2Hz is outside LF<=200 and MF>=300; ALL25/EDA13 kept separately',
       'feature_definitions':{'max_minus_median':'max_t dB[t,f] - median_t dB[t,f]',
        'max_minus_q20':'max_t dB[t,f] - q20_t dB[t,f]; linear-interpolated quantile in dB',
        'median_minus_window_median':'median_t dB[t,f] - median(all finite375 window dB cells)',
        'median_minus_band_median_center':'alternate window centre: median_t dB[t,f] - median_f(median_t dB[t,f])',
        'max_minus_min':'max_t dB[t,f] - min_t dB[t,f]',
        'group_power_bump':'10log10(sum_f10^(dB[t,f]/10)) then temporal max-minus-median/q20, never sum dB',
        'group_band_metric':'unweighted mean of constituent per-band metrics, followed by between-window quantiles'},
       'mask_policy':'keep every observed finite cell; no NaN imputation; undefined band summaries remainNaN; group level requires complete constituent bands, EDA low/high require complete25 minute',
       'quantiles':'p10,p50,p90 across windows; interpolation linear; calculationsfloat64, saved derived arraysfloat32',
       'input_quantization':'no additional rounding; compare exactly stored values',
       'TRAIN_label_subsets':'all, known controlled-added share>0, known share==0; no-source label is not confirmed natural vessel absence',
       'Test_labels':'unknown; do not infer target mean, AUC, or per-window physical added share',
       'causal_scope':'input distribution comparison does not establish why public RMSE/AUC changed or which source generated Test'}
    dump(out/'plan.json',plan);numeric=fixture();dump(out/'NUMERIC_FIXTURE.json',numeric)
    bands=[];groups=[];conditionals=[];metadata={};saved={};primary={};population={}
    loaders=[('R14_TRAIN',lambda:load_train(a.r14_path,str(Path(a.r14_path).with_suffix('.json')))),
             ('R9_TRAIN',lambda:load_train(inputs['R9_TRAIN'],'models/r9_mac_trees_r2/fold_00/tree_epoch1_manifest.json')),
             ('Test',load_test)]
    for name,loader in loaders:
        db,y,ids,meta=loader();meta_path=out/(name+'_source.json');dump(meta_path,meta);metadata[name]=meta
        b,g,counts,nobs=features(db);sample=np.sort(np.random.default_rng(SEED).choice(len(db),min(1500,len(db)),replace=False))
        dump(out/(name+'_sample1500_indices.json'),{'seed':SEED,'indices':sample.tolist(),'source_n':len(db),
            'row_order_note':'same deterministic index rule; no claim of matched backgrounds across different generators'})
        population[name]={'n':len(db),'Nobs':{str(int(k)):int(v) for k,v in zip(*np.unique(nobs,return_counts=True))},
          'unobserved_band_windows':int((counts==0).sum()),'partial_frequency_minutes':int((np.isfinite(db).any(2)&~np.isfinite(db).all(2)).sum())}
        payload={'sample_id':ids,'sample1500_indices':sample,'observed_minutes':nobs,'valid_counts_per_band':counts,
                 **{'band_'+k:v for k,v in b.items()},**{'group_'+gn+'_'+k:v for gn,vals in g.items() for k,v in vals.items()}}
        if y is not None:payload['known_added_share']=y
        np.savez_compressed(out/(name+'_derived.npz'),**payload)
        for scope,index in [('full',np.arange(len(db))),('sample1500',sample)]:
            subsets={'all':index}
            if y is not None:subsets.update(positive=index[y[index]>0],zero=index[y[index]==0])
            for subset,ii in subsets.items():
                for feature,v in b.items():
                    for f in range(25):
                        bands.append({'dataset':name,'scope':scope,'label_subset':subset,'n_rows':len(ii),'band_index':f,'frequency_hz':FREQ[f],'feature':feature,**summary(v[ii,f])})
                for group,gg in g.items():
                    for feature,v in gg.items():groups.append({'dataset':name,'scope':scope,'label_subset':subset,'n_rows':len(ii),'group':group,'aggregation':'linear_sum_then_db','feature':feature,**summary(v[ii])})
                    for feature,v in b.items():
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore',RuntimeWarning);value=np.nanmean(v[ii][:,GROUPS[group]],1)
                        groups.append({'dataset':name,'scope':scope,'label_subset':subset,'n_rows':len(ii),'group':group,'aggregation':'mean_of_band_metric','feature':feature,**summary(value)})
        primary[name]={k:[summary(v[:,f]) for f in range(25)] for k,v in b.items()}
        power=g['HF_3000_8000']['power_max_minus_median'];saved[name]={'HF_power_bump':summary(power),'LF_EDA_power_bump':summary(g['LF_EDA_le520']['power_max_minus_median']),
             'HF_bump_gt5_fraction':float(np.mean(power>5)), 'HF_mean_of_band_bump':summary(np.nanmean(b['max_minus_median'][:,20:25],1))}
        if y is not None:
            edges=[-np.inf,5,10,15,20,np.inf]
            for low,high in zip(edges[:-1],edges[1:]):
                take=(power>=low)&(power<high)
                conditionals.append({'dataset':name,'HF_bump_lower_inclusive':str(low),'HF_bump_upper_exclusive':str(high),'n_rows':int(take.sum()),
                   'known_share_mean':float(y[take].mean()) if take.any() else None,'known_positive_fraction':float((y[take]>0).mean()) if take.any() else None,
                   **summary(y[take])})
        if name=='Test':
            cached=ROOT/'eda_deep/tables/test_window_features.csv'
            with cached.open(newline='') as f:records=list(csv.DictReader(f))
            if [r['ID'] for r in records]!=ids.tolist():raise ValueError('Existing EDA cache ID order differs')
            old=np.asarray([float(r['bump_high_db']) for r in records])
            saved[name]['EDA_cache_comparison']={'path':str(cached.relative_to(ROOT)),'sha256':sha(cached),'ID_order_exact':True,
                 'old_bump_high':summary(old),'recomputed64_vs_EDA32_max_abs_error':float(np.max(abs(power-old))),
                 'old_bump_gt5_fraction':float((old>5).mean()),'definition':'high5 linear-energy sum then dB curve, complete25-minute mask; NOT mean/median of25 independent band bumps'}
        print(json.dumps({'dataset':name,'n':len(db),'HF_bump':saved[name]['HF_power_bump'],'seconds_total':time.perf_counter()-start}),flush=True)
        del db,y,b,g,payload
    table(out/'band25_quantiles.csv',bands);table(out/'group_quantiles.csv',groups);table(out/'TRAIN_hf_bump_known_share_bins.csv',conditionals)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    colors={'Test':'#d55e00','R9_TRAIN':'#0072b2','R14_TRAIN':'#009e73'}
    for ax,key,title in zip(axes,NAMES[:3],['Temporal max - median','Temporal max - q20','Band median - window median']):
        for name in ('Test','R9_TRAIN','R14_TRAIN'):
            q=primary[name][key];lo=np.asarray([v['p10'] for v in q],float);md=np.asarray([v['p50'] for v in q],float);hi=np.asarray([v['p90'] for v in q],float)
            ax.plot(FREQ,md,label=name,color=colors[name],lw=1.8);ax.fill_between(FREQ,lo,hi,color=colors[name],alpha=.12)
        ax.set_xscale('log');ax.set_xlabel('Band centre (Hz)');ax.set_ylabel('dB');ax.set_title(title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('R15 descriptive input audit: median line; p10-p90 shaded; TRAIN labels known, Test labels unknown',fontsize=10)
    fig.savefig(out/'band25_comparison.png',dpi=160);plt.close(fig)
    lines=['# R15 대역별 입력 분포 비교','',
      '이 문서는 원본 Test 1,500창과 실제 저장된 R9·R14 TRAIN 혼합창의 기술 통계다. 새 합성·모델 학습·보정·분포 맞춤·출력 변경은 수행하지 않았다. Test 정답과 실제 추가 선박 에너지 비율은 알 수 없다. 차이가 있어도 Public 오차의 인과 원인을 확정하지 않는다.','',
      '![대역별 비교](band25_comparison.png)','', '## 원천과 분모','', '| 자료 | 전체 창 | 샘플 | TRAIN 양성/추가0 | 원천 |','|---|---:|---:|---|---|']
    for name in ('Test','R9_TRAIN','R14_TRAIN'):
        m=metadata[name];label=f"{m['known_added_positive']}/{m['known_added_zero']}" if 'known_added_positive' in m else '정답 없음'
        lines.append(f"| {name} | {m['n']} | {min(1500,m['n'])} | {label} | `{m['path']}` |")
    lines +=['','R9는 실제 Mac Tree 학습에 사용한 Fold0 epoch1이다. R14는 실행 중 NN이 실제로 사용한 Fold0 epoch38 rolling raw-cache이며 별도의 새 합성이 아니다. 한 epoch 스냅샷이므로 다른 epoch·모든 fold를 대표한다고 주장하지 않는다. 동일 개수 샘플은 seed202610271502의 무복원 추출이며 양 데이터 간 동일 배경·참조로 짝지은 실험이 아니다. 모든 원천의 NPZ/동반 receipt SHA는 source JSON에 기록했다. rolling 원본은 원학습의 캐시 정책으로 추후 삭제될 수 있어 이번 파생 수치를 보존했다.','',
       '## 계산식과 마스크','',
       '시간 통계는 각 대역에서 관측된 dB만 사용한다. `max−q20`의 q20은 dB 영역 선형 보간 분위수다. `median−window_median`의 window median은 관측된 전체 375셀의 중앙값으로 명시했으며, 대역별 중앙값 25개의 중앙값을 사용하는 대안도 별도 열에 보존했다. 추가 반올림·NaN 대체는 없다. 계산 float64, 파생 NPZ float32. 분포 p10/p50/p90의 분모는 유효 창 수이고 개별 열에 함께 기록한다.','',
       'LF≤200Hz는 index0..8, MF300..3000Hz는10..19, HF3000..8000Hz는20..24다. 251.2Hz는 이 세 군의 경계 밖이며 ALL25에 포함된다. 원 EDA 저대역31.6..501.2Hz(index0..12)는 다른 군으로 유지한다. 그룹 파워곡선은 선형 파워를 합한 후 dB로 변환하며, 대역별 bump의 평균과 구분한다.','',
       '## EDA 16.25dB와의 정의 대조','',
       '| 자료 | HF 5대역 선형합 곡선 bump p50 | HF 개별대역 bump 평균의 p50 | HF 선형합 bump>5dB |','|---|---:|---:|---:|']
    for name in ('Test','R9_TRAIN','R14_TRAIN'):
        v=saved[name];lines.append(f"| {name} | {v['HF_power_bump']['p50']:.6f} | {v['HF_mean_of_band_bump']['p50']:.6f} | {100*v['HF_bump_gt5_fraction']:.3f}% |")
    v=saved['Test']['EDA_cache_comparison'];lines +=['',f"기존 캐시와 ID 순서가 일치한다. 기존 `bump_high_db` p50={v['old_bump_high']['p50']:.9f}; 이번 float64 재계산과 기존 float32 EDA의 최대 절대 차이는 {v['recomputed64_vs_EDA32_max_abs_error']:.9g}dB다. 16.25dB는 **HF5대역 파워합 시계열**의 `max−median`이며, 25개 대역 bump의 중앙값이나 실제 선박 share가 아니다.",'',
       '기존 EDA의 TRAIN 대조군은 원 SA 배경 샘플이다. 이번 R9·R14 TRAIN은 알려진 추가 에너지를 섞은 학습 혼합창이다. 두 TRAIN 분모와 레이블 의미를 혼동하지 않는다. 원 EDA의 사이트 프로브는 추정값이며 본 비교에서 실제 사이트 정답으로 사용하지 않았다.','',
       '## 25개 대역 전체표','', '각 셀은 p10 / p50 / p90 dB이다. 아래는 전체 창이며, 동일1500 샘플 및 TRAIN 추가0/양성 분리는 CSV에 모두 있다.']
    for feature in NAMES[:3]:
        lines +=['',f'### {feature}','','| Hz | Test | R9 TRAIN | R14 TRAIN |','|---:|---:|---:|---:|']
        for f in range(25):
            cells=[]
            for name in ('Test','R9_TRAIN','R14_TRAIN'):
                s=primary[name][feature][f];cells.append(' / '.join(f"{s[k]:.3f}" for k in ('p10','p50','p90')))
            lines.append('| '+str(FREQ[f])+' | '+' | '.join(cells)+' |')
    lines +=['','## 해석 한계','',
       'TRAIN의 share==0은 의도적으로 추가한 성분이 관측 에너지에 기여하지 않는다는 뜻이며, 원래 바다에 자연 선박 소리가 없다는 뜻은 아니다. 고주파 bump가 크다는 사실만으로 선박 유무·추가 share·숨은 정답을 정할 수 없다. TRAIN의 bump 구간별 알려진 share 표는 실제 생성 법칙의 조건부 기술 통계로만 제공하며 Test에 이식·보정하지 않는다. 데이터 구성·결측·epoch·기기·기본 배경과 참조 분포가 함께 다르므로 이번 대조는 원인을 분리한 ablation이 아니다.','',
       '파일: [25대역 전체/샘플/양성·추가0](band25_quantiles.csv), [대역군 정의별 통계](group_quantiles.csv), [TRAIN bump별 알려진 share](TRAIN_hf_bump_known_share_bins.csv).']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    if sha(__file__)!=own:raise ValueError('Diagnostic source changed while running')
    # R14 rolling cache is already hash-checked twice through the same open
    # handle; its later removal by training is not a diagnostic mutation.
    for name in ('Test','R9_TRAIN'):
        if sha(ROOT/inputs[name])!=metadata[name]['sha256']:raise ValueError('Fixed source changed during audit')
    report={'status':'completed','code_sha256':own,'scope':plan,'populations':population,'source_metadata_files':{n:n+'_source.json' for n in metadata},
       'key_descriptive_statistics':saved,'numeric_fixture':numeric,'elapsed_seconds':time.perf_counter()-start,
       'model_fit_or_synthesis':False,'model_or_policy_state_changed':False,'Test_share_labels_known':False,
       'Test_input_distribution_used_to_change_training':False,'original_fixed_sources_unchanged':True,
       'output_sha256':{str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()}}
    dump(out/'RESULTS.json',report);print(json.dumps({'status':'completed','report':str(out.relative_to(ROOT)),'elapsed_seconds':report['elapsed_seconds'],'key_statistics':saved},ensure_ascii=False))

if __name__=='__main__':main()
