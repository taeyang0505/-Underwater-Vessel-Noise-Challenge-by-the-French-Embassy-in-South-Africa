"""Present frozen R15 descriptive audits; no fitting, prediction or calibration."""
from pathlib import Path
import hashlib, html, json, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/r15_review_r1'
OUT.mkdir(exist_ok=True)
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
D = ROOT / 'reports/r15_descriptive_audit_r1'
M = ROOT / 'reports/r15_member_audit_r1'
B = ROOT / 'reports/r15_band_comparison_r1'
S = ROOT / 'reports/r15_static_channel_audit_r1'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def mdtable(frame, digits=5):
    def val(x):
        if pd.isna(x): return '미저장'
        if isinstance(x, (float, np.floating)): return f'{x:.{digits}f}'
        return str(x).replace('|', '/')
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(val(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None)) + '\n'

df = pd.read_csv(D / 'joined1500.csv').set_index('ID')
dist = pd.read_csv(M / 'distribution_summary.csv').set_index('series')
bands = pd.read_csv(B / 'band25_quantiles.csv')
bands = bands[(bands.scope == 'full') & (bands.label_subset == 'all') & (bands.feature == 'max_minus_median')]
groups = pd.read_csv(B / 'group_quantiles.csv')
static = pd.read_csv(S / 'joined1500.csv').set_index('ID')
assert df.index.equals(static.index) and df.index.is_unique and len(df) == 1500
for c in static.columns:
    if c not in df: df[c] = static[c]
static_corr = pd.read_csv(S / 'feature_prediction_correlations.csv')
static_features = ['PCA_positive_residual_RMS_db','MF_minus_LF_mean_db','LF_anchor_residual_RMS_db']
static_table = static_corr[static_corr.feature.isin(static_features) & static_corr.series.isin(['R9_all5.share','R14_step1.share','R14_step1.score'])][['feature','series','Pearson','Spearman']]
static_groups = pd.read_csv(S / 'lowtemporal_x_actualstatic_groups.csv')
static_groups = static_groups[static_groups.group.str.startswith('low_temporal') & static_groups.series.isin(['R9_all5.share','R14_step1.share','R14.nn_bg_cnn.share'])]
temporal_corr = pd.read_csv(D/'feature_prediction_correlations.csv')
temporal_table = temporal_corr[temporal_corr.feature.isin(['bump_high_db','bump_low_db','corr_low_high','naive_share_q20','S_phys_safe_mid_db']) & temporal_corr.series.isin(['R9_all5.share','R9_all5.score','R14_step1.share','R14_step1.score'])][['feature','series','Pearson','Spearman']]
group_table = groups[(groups.scope=='full') & (groups.label_subset=='all') & (groups.aggregation=='linear_sum_then_db') & (groups.feature=='power_max_minus_median') & groups['group'].isin(['LF_EDA_le520','MF_300_3000','HF_3000_8000'])][['dataset','group','n_rows','p10','p50','p90']]

models = ['R5_CNN', 'R9_tree5', 'R9_all5', 'R14_step1', 'R14.nn_bg_cnn', 'R14.nn_fdy', 'R14.nn_itransformer', 'R14.tree_baseline', 'R14.tree_highshare']
names = {'R5_CNN':'R5 CNN', 'R9_tree5':'R9 Tree5', 'R9_all5':'R9 All5', 'R14_step1':'R14 Step1', 'R14.nn_bg_cnn':'R14 Wide CNN', 'R14.nn_fdy':'R14 FDY', 'R14.nn_itransformer':'R14 iTransformer', 'R14.tree_baseline':'R14 Tree baseline5', 'R14.tree_highshare':'R14 Tree highshare5'}
model_rows = []
for model in models:
    for kind in ('share', 'score'):
        col = model + '.' + kind
        if col not in df: continue
        x = df[col]; ref = df['R9_all5.' + kind]
        model_rows.append({'모델': names[model], '출력': kind, '평균': x.mean(), 'RMS': np.sqrt(np.mean(x*x)), '중앙값': x.median(), '>=0.90 비율': (x >= .9).mean(), 'R9 Pearson': x.corr(ref), 'R9 Spearman': x.corr(ref, method='spearman')})
mt = pd.DataFrame(model_rows)
mt.to_csv(OUT / 'model_distribution_comparison.csv', index=False)

site_rows = []
for site, group in df.groupby('probe_site'):
    for model in ['R9_all5', 'R14_step1', 'R14.nn_bg_cnn', 'R14.tree_baseline', 'R14.tree_highshare', 'R14.nn_fdy', 'R14.nn_itransformer']:
        x = group[model + '.share']
        site_rows.append({'추정 사이트': site, 'N': len(group), '모델': names[model], 'share 평균': x.mean(), 'share RMS': np.sqrt(np.mean(x*x)), 'R9 대비 평균차': (x-group['R9_all5.share']).mean()})
st = pd.DataFrame(site_rows)
st.to_csv(OUT / 'site_share_comparison.csv', index=False)

band_table = None
for ds, label in [('Test', 'Test'), ('R9_TRAIN', 'R9 TRAIN'), ('R14_TRAIN', 'R14 TRAIN')]:
    a = bands[bands.dataset == ds][['band_index','frequency_hz','p50','p90']].sort_values('band_index')
    a = a.rename(columns={'p50':label+' p50', 'p90':label+' p90'})
    band_table = a if band_table is None else band_table.merge(a, on=['band_index','frequency_hz'], validate='1:1')
assert len(band_table) == 25
band_table.to_csv(OUT / 'band25_max_minus_median_comparison.csv', index=False)

plt.rcParams.update({'font.size':10, 'axes.spines.top':False,'axes.spines.right':False, 'figure.facecolor':'white'})
fig, axs = plt.subplots(2,2,figsize=(13,9),layout='constrained')
colors = {'R9_all5':'#23749c','R14_step1':'#c66b32'}
for model in ['R9_all5','R14_step1']:
    axs[0,0].scatter(df.S_phys_safe_mid_db, df[model+'.share'],s=8,alpha=.25,label=names[model],color=colors[model])
axs[0,0].set(xlabel='Fixed temporal descriptor S_phys (dB)',ylabel='Predicted share',title='Temporal response remains positive')
axs[0,0].legend(frameon=False)
for model in ['R9_all5','R14_step1']:
    means = df.groupby(pd.cut(df.bump_high_db,[-np.inf,5,10,15,20,30,np.inf]),observed=False)[model+'.share'].mean()
    axs[0,1].plot(np.arange(len(means)),means.to_numpy(),'o-',label=names[model],color=colors[model])
axs[0,1].set(xticks=np.arange(6),xticklabels=['<=5','5–10','10–15','15–20','20–30','>30'],xlabel='HF bump (dB)',ylabel='Mean predicted share',title='Input bins are descriptive, not vessel labels')
axs[0,1].legend(frameon=False)
sites=['01 Richards Bay','07 St Francis','13 Saldanha Bay','05 St Croix']
xx=np.arange(4)
for offset,model in [(-.18,'R9_all5'),(.18,'R14_step1')]:
    vals=[df.loc[df.probe_site==site,model+'.share'].mean() for site in sites]
    axs[1,0].bar(xx+offset,vals,width=.34,label=names[model],color=colors[model])
axs[1,0].set(xticks=xx,xticklabels=['Richards Bay','St Francis','Saldanha','St Croix'],ylabel='Mean predicted share',title='Sites are historical kNN domain estimates')
axs[1,0].legend(frameon=False)
for ds,label,color in [('Test','Test','#222222'),('R9_TRAIN','R9 TRAIN','#23749c'),('R14_TRAIN','R14 TRAIN','#c66b32')]:
    a=bands[bands.dataset==ds].sort_values('frequency_hz')
    axs[1,1].plot(a.frequency_hz,a.p50,label=label,color=color)
axs[1,1].set(xscale='log',xlabel='Band center (Hz)',ylabel='Per-band max − median (dB)',title='All rows; exact masks retained')
axs[1,1].legend(frameon=False)
fig.savefig(FIG/'01_input_output_comparison.png',dpi=150);plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
for ax,c,title in zip(axs,['bump_high_db','bump_low_db','corr_low_high'],['HF bump (dB)','LF bump (dB)','LF–HF temporal correlation']):
    ax.hist(df[c].dropna(),bins=30,color='#557f8b',alpha=.9)
    ax.set(xlabel=title,ylabel='Number of windows')
    ax.axvline(df[c].median(),color='#c66b32',linestyle='--',label=f'Median {df[c].median():.2f}')
    ax.legend(frameon=False)
fig.suptitle('Unlabelled mixture histograms: modes do not identify positive / negative classes',fontsize=11)
fig.savefig(FIG/'02_input_histograms.png',dpi=150);plt.close(fig)

# Earlier attachment requested top50 visual review. No new model inference.
raw=ROOT/'data/raw/Test.csv'; before=sha(raw)
q=pd.read_csv(raw).set_index('ID').loc[df.index]
assert q.shape == (1500,375)
db=q.to_numpy(dtype=np.float64).reshape(1500,15,25)
top=pd.read_csv(D/'top50_absdelta.csv')
figures=[]
for page in range(5):
    fig,axes=plt.subplots(5,4,figsize=(17,15),layout='constrained')
    for local,(_,record) in enumerate(top.iloc[page*10:(page+1)*10].iterrows()):
        row=local//2;col=(local%2)*2;i=df.index.get_loc(record.ID);a=db[i]
        baseline=np.nanmedian(a,axis=0);rel=a-baseline
        ax=axes[row,col];im=ax.imshow(rel.T,origin='lower',aspect='auto',vmin=-5,vmax=35,cmap='magma',extent=[.5,15.5,-.5,24.5])
        ax.set(xlabel='Minute',ylabel='Band index',xticks=[1,5,10,15],yticks=[0,8,15,24])
        ax.set_title(f'{page*10+local+1}. {record.ID[:10]} | Δ={record.delta_R14minusR9_share:+.3f}',fontsize=9)
        ax=axes[row,col+1];prof=baseline-np.nanmedian(a)
        ax.plot(np.arange(25),prof,color='#23749c');ax.axhline(0,color='#999999',lw=.5)
        ax.set(xlabel='Band index',ylabel='Median − window median (dB)',xticks=[0,8,15,24])
        ax.set_title(f'R9={record["R9_all5.share"]:.3f}  R14={record["R14_step1.share"]:.3f}',fontsize=9)
    fig.suptitle('Top absolute prediction changes | left: within-band temporal contrast; right: relative static profile',fontsize=12)
    dest=FIG/f'top50_page_{page+1:02d}.png';fig.savefig(dest,dpi=110);plt.close(fig);figures.append(dest.name)
assert sha(raw)==before

now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')
pause=json.loads((ROOT/'ops/r15_training_pause_receipt.json').read_text())
md=f'''# R15 입력–출력 직접 대조 진단

작성 시각: {now}. TFGrid는 명시적 사용자 요청에 따라 SIGSTOP으로 일시 정지했다. 마지막 완료 epoch는 {pause['last_complete_epoch']}, 해당 시점 best selection RMSE는 {pause['best_selection_rmse']:.6f}다. Fold 1~4는 시작하지 않았고 새 CSV 생성 감독도 보류했다. 기존 체크포인트·정책·데이터·제출 파일은 보존했다.

## 핵심 관측

1. EDA의 Test HF bump 중앙값 16.2463 dB와 >5 dB 비율 83.8667%를 재현했다. 그러나 같은 정의의 TRAIN HF bump 중앙값은 R9 8.1262 dB, R14 4.6632 dB다. 이 중앙값을 기준으로 R14 TRAIN과 Test의 차이가 더 크다. 이는 기술통계 차이이며 어느 Test 창이 양성인지나 실패 원인을 확정하는 수치는 아니다.
2. R14 score는 HF bump와 Spearman +0.5761, 사전 고정한 S_phys와 +0.6962, LF–HF 시간 상관과 +0.7159의 양의 상관이다. 시간 변화를 전혀 보지 않는다거나 역상관이라는 설명은 이 관측과 맞지 않는다.
3. 시간 변화 하위25%와 정적 고주파 비중 proxy 상위25%의 교집합36창에서는 R9 share 평균0.08614→R14 0.07283으로 감소했다. 실제 PCA/LF 채널의 결과도 별도 표에 보존했다. 한계: 주변 상관은 모델의 비선형 의존성이나 인과효과를 증명·반증하지 않는다.
4. CNN share 평균0.30982는 높지만 highshare Tree 평균0.32935가 더 높다. CNN만 높은 출력을 낸 것은 아니다. 이번 정책의 CNN 비중81.05%와 성능 악화는 함께 관측됐으나, 다중 변경된4개 제출의 비교가 CNN 용량·비중의 인과효과를 분리하지는 않는다.
5. kNN 추정 St Francis는487창이다. 첨부의489와 다르며 이 값은 실제 사이트 정답이 아니다. 입력 HF bump의 KS 차이는 남아공 원본 대 Test의 분포 차이로, Test 내부 양성·음성 AUC가 아니다.

## 1. 개별 모델 분포·R9 대비 상관

{mdtable(mt)}

R14 NN의 score는 설계상 share와 동일하고 별도 classifier head가 없다. R14 Tree별/Fold별 독립 classifier raw 확률은 기존 query NPZ에 저장되지 않았다. 이미 저장된 최종 결합 score를 raw classifier 확률로 오인하거나 역산해서 실측이라고 표시하지 않았다. R9 Tree raw classifier는 원본 저장값을 별도 전체57series 표에서 제공한다.

## 2. 추정 사이트별 share 평균과 RMS

{mdtable(st)}

상관계수 전체는 [특징–예측 Pearson/Spearman](../r15_descriptive_audit_r1/feature_prediction_correlations.csv), [실제 정적 채널–예측 상관](../r15_static_channel_audit_r1/feature_prediction_correlations.csv)에 있다. 각 대역의 정적 오프셋을 상쇄하는 사전 고정 S_phys는 316.2–5011.9Hz의 대역별 max−median 평균이며, 선박 확률이나 share 정답이 아니다.

### 시간 돌출·동조와 예측의 실제 상관

{mdtable(temporal_table)}

### 실제 PCA·LF-anchor 요약과 예측의 상관

{mdtable(static_table)}

### 낮은 시간변화와 실제 정적 채널 상위25% 교차

{mdtable(static_groups[['group','series','n','mean','median','rms']])}

PCA 잔차 상위 교집합은10개, MF−LF는19개, anchor RMS는8개다. 부호가 혼재하고 표본이 작으며 정답이 없으므로 오탐률로 해석하지 않았다. 모든 사이트별 share·score 분위수와 score≥0.90 비율은 [사이트×모델 전체표](../r15_descriptive_audit_r1/site_model_stats.csv)에 있다.

## 3. 25대역 max−median 비교 (dB)

R9와 R14는 각각 실제 Fold0 TRAIN 78,869창 전수를 사용했다. R9는 저장된 epoch1, R14는 실제 TFGrid epoch38용으로 생성된 snapshot이다. 이는 입력 캐시이며 마지막 완료 학습 epoch37과 구분한다. 원본 관측 마스크를 그대로 유지했고 Test는1,500창이다. 전체와 동일 seed1,500표본·양성/무추가 분리 결과를 모두 저장했다.

{mdtable(band_table,3)}

### 기존 EDA와 동일한 대역 파워합→dB 시간 돌출

{mdtable(group_table,3)}

대역별 dB 시간 돌출과 관측 추가 에너지 share는 같은 값이 아니다. HF bump15–20dB TRAIN의 알려진 share 중앙값은 R9 0.09261, R14 0.06641이다. 따라서 그 HF 돌출이 반드시 share0.3–0.6을 뜻한다는 주장은 실측과 다르다. Test share는 미공개이므로 이 값으로 추정·보정하지 않았다.

## 시각화·전수 행

![입력과 예측 직접 비교](figures/01_input_output_comparison.png)

![입력 히스토그램](figures/02_input_histograms.png)

히스토그램 모양이나 봉우리만으로 숨겨진 양성·음성을 나누지 않았다. HF bump≤5dB 창도242개 존재하므로 중앙값16.25라는 이유로 평탄한 창이 전혀 없다고 결론 내릴 수 없다.

- [1,500창 입력·출력 통합표](../r15_descriptive_audit_r1/joined1500.csv)
- [정적14채널 요약 통합표](../r15_static_channel_audit_r1/joined1500.csv)
- [상위50 절대 변화 창](../r15_descriptive_audit_r1/top50_absdelta.csv)
- [개별 모델 분포57series](../r15_member_audit_r1/distribution_summary.csv)
- [25대역 p10/p50/p90 전수·샘플·양성 분리표](../r15_band_comparison_r1/band25_quantiles.csv)
- [관측 밀도·알려진share 조건부·대역 그룹 분석](../r15_band_comparison_r1/README.md)
- [정적 채널 실제 수식 검산](../r15_static_channel_audit_r1/RESULTS.json)
- [인과 주장·물리식 점검](../r15_member_audit_r1/CLAIM_CHECK.md)

## 범위

신규 학습·모델 추론·Test 라벨 역산·가중치 조정·예측 보정·새 제출 CSV 생성은 수행하지 않았다. 정적 채널은 기존 TRAIN 적합 basis를 고정한 상태에서 입력 특징만 계산했고 NumPy와 공식 FP64 frontend를 전수 대조했다. 기존57출력 series의 ID 정렬·앙상블 산술·원본SHA를 검증했다. 이 결과는 원인 분리를 위한 기술통계이며 공식 AUC를 계산하거나 개선을 보장하지 않는다.
'''
md+='\n'.join(f'\n![상위50 변화 창 {i+1}/5](figures/{name})\n' for i,name in enumerate(figures))
(OUT/'R15_DIRECT_COMPARISON_DIAGNOSTICS.md').write_text(md)
# Markdown is the detailed record; compact HTML gives direct access to every table and image.
intro='''<p>TFGrid 일시 정지 · 추가 Fold 보류 · 새 제출 CSV 생성 보류</p><p>1,500개 Test 창과 저장 예측57열을 직접 대조했습니다. 새 학습·모델 추론·정답 역산·보정은 없습니다.</p><p><strong>중요:</strong> 시간 변화를 무시했다거나 정적 PCA/LF 특징이 단독 원인이라는 주장은 현재 기술통계로 확정되지 않습니다. TRAIN–Test의 HF 시간 돌출 분포 차이는 실제로 확인됐습니다.</p>'''
links=''.join(f'<li><a href="{p}">{label}</a></li>' for p,label in [('r15_review_r1/R15_DIRECT_COMPARISON_DIAGNOSTICS.md','상세 Markdown 보고서'),('r15_review_r1/model_distribution_comparison.csv','모델별 평균·RMS·중앙값·상관'),('r15_review_r1/site_share_comparison.csv','추정 사이트별 모델 비교'),('r15_review_r1/band25_max_minus_median_comparison.csv','25개 대역 p50·p90 비교'),('r15_descriptive_audit_r1/feature_prediction_correlations.csv','물리 특징–출력 상관 전체'),('r15_static_channel_audit_r1/feature_prediction_correlations.csv','실제 PCA·LF 채널–출력 상관'),('r15_descriptive_audit_r1/joined1500.csv','1,500개 창 통합표'),('r15_static_channel_audit_r1/FINDINGS.md','실제 정적14채널 FE 검산'),('r15_member_audit_r1/CLAIM_CHECK.md','인과 주장·수식 점검')])
body=f'<h1>R15 직접 대조 진단</h1><small>{html.escape(now)}</small>{intro}<ul>{links}</ul><h2>개별 모델 분포</h2>'+mt.round(5).to_html(index=False,na_rep='미저장')+'<p>R14 NN score=share. R14 각 Tree raw classifier는 미저장입니다.</p><h2>시간 특징–출력 상관</h2>'+temporal_table.round(5).to_html(index=False)+'<h2>실제 PCA·LF 채널–출력 상관</h2>'+static_table.round(5).to_html(index=False)+'<h2>낮은 시간변화×높은 정적 채널</h2>'+static_groups[['group','series','n','mean','median','rms']].round(5).to_html(index=False)+'<h2>25개 대역 max−median (dB)</h2>'+band_table.round(3).to_html(index=False)+'<h2>기존 EDA 정의 대역 파워합의 시간 돌출</h2>'+group_table.round(3).to_html(index=False)+'<h2>입력–출력 비교</h2><img src="r15_review_r1/figures/01_input_output_comparison.png"><img src="r15_review_r1/figures/02_input_histograms.png"><h2>추정 사이트별 비교</h2>'+st.round(5).to_html(index=False)+'<h2>상위50 변화 창</h2>'
body+=''.join(f'<details><summary>상위 변화 창 {i*10+1}–{i*10+10}</summary><img loading="lazy" src="r15_review_r1/figures/{name}"></details>' for i,name in enumerate(figures))
page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>R15 직접 대조 진단</title><style>body{font:16px/1.7 -apple-system,BlinkMacSystemFont,sans-serif;max-width:1280px;margin:32px auto;padding:0 24px;background:#f7f8fa;color:#172334}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:13px;background:white;display:block;overflow:auto}th,td{padding:8px 10px;border-bottom:1px solid #dae0e8;text-align:right;white-space:nowrap}th{background:#e9eff4}img{max-width:100%;height:auto;background:white;margin:20px 0}a{color:#146b9a}details{margin:16px 0}summary{cursor:pointer;font-weight:bold}p{max-width:1050px}</style><body>'+body+'</body></html>'
(ROOT/'reports/R15_DIAGNOSTICS_START.html').write_text(page)
source_paths=[D/'RESULTS.json',M/'RESULTS.json',B/'RESULTS.json',S/'RESULTS.json',ROOT/'ops/r15_training_pause_receipt.json']
manifest={'status':'completed_descriptive_presentation','KST':now,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in source_paths},'raw_Test_sha256':before,'new_model_training_or_inference':False,'new_submission_CSV':False,'script_sha256':sha(__file__),'self_and_independent_QA_excluded':True,'outputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in {'MANIFEST.json','INDEPENDENT_QA.json'}},'main_HTML_sha256':sha(ROOT/'reports/R15_DIAGNOSTICS_START.html')}
(OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'completed','page':'reports/R15_DIAGNOSTICS_START.html','report':str((OUT/'R15_DIRECT_COMPARISON_DIAGNOSTICS.md').relative_to(ROOT)),'figures':len(list(FIG.glob('*.png'))),'rows':len(df)},ensure_ascii=False))
