"""Saved-output oracles and full-support review. Does not infer or fit."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/r11_diagnostics/full_support_hf_r1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def auc(y,s):
    z=y>0;n=int(z.sum());m=len(y)-n
    ranks=rankdata(np.asarray(s,dtype=np.float64)).astype(np.float64)
    return float((ranks[z].sum()-n*(n+1)/2)/(n*m)) if n and m else None
def main():
    d=json.loads((OUT/'RESULTS.json').read_text());frame=pd.read_csv(OUT/'crossing_metrics.csv')
    samples={k:np.load(OUT/(k+'_samples.npz'),allow_pickle=False) for k in d['panel_summary']}
    predictions={k:np.load(OUT/(k+'_predictions.npz'),allow_pickle=False) for k in samples}
    checks=[];maximum=0.
    for row in frame.itertuples(index=False):
        y=samples[row.positive_mode]['share'].astype(float);p=predictions[row.positive_mode];n=predictions[row.negative_panel]
        ps=p[row.model+'_share'].astype(float);ns=n[row.model+'_share'].astype(float)
        rmse=np.linalg.norm(np.r_[ps-y,ns])/np.sqrt(len(ps)+len(ns))
        errors=[abs(rmse-row.rmse)]
        label=np.r_[np.ones(len(y)),np.zeros(len(ns))]
        for k in ['score','share']:
            q=np.r_[p[row.model+'_'+k],n[row.model+'_'+k]]
            errors.append(abs(auc(label,q)-getattr(row,'auc_'+k)))
        maximum=max(maximum,*errors)
    if maximum>1e-12:raise AssertionError('Independent saved metric mismatch')
    source=json.loads((OUT/'source_panels.json').read_text())
    shape=[];bins=[];components=[]
    for mode,z in samples.items():
        db=z['db'];obs=np.isfinite(db).all(2);counts=obs.sum(1)
        row={'panel':mode,'n':len(db),'actual_observed_minutes_histogram':{str(k):int(v) for k,v in zip(*np.unique(counts,return_counts=True))}}
        if 'cell_share' in z:
            q=z['cell_share'].astype(float)
            row.update({'q05_mean':float(np.quantile(q,.05,axis=1).mean()),
                'q20_mean':float(np.quantile(q,.20,axis=1).mean()),'qmedian_mean':float(np.median(q,axis=1).mean()),
                'q05_positive_cell_fraction':float(np.mean(np.quantile(q,.05,axis=1)>0)),
                'q20_positive_cell_fraction':float(np.mean(np.quantile(q,.20,axis=1)>0))})
            if mode!='full_smooth15_share030' and not np.array_equal(z['share'],samples['r9_original']['share']):raise AssertionError('Paired labels changed')
        shape.append(row)
        for model in d['per_model_metrics'][mode]:
            pr=predictions[mode];p=pr[model+'_share'].astype(float);s=pr[model+'_score'].astype(float)
            comp=2*s-p
            components.append({'panel':mode,'model':model,'share_mean':float(p.mean()),'score_mean':float(s.mean()),
                'recovered_component_mean':float(comp.mean()),'recovered_component_ge099_fraction':float(np.mean(comp>=.99))})
    for positive in [k for k in samples if 'cell_share' in samples[k]]:
        for low,high,label in [(0,.03,'(0,.03]'),(.03,.08,'(.03,.08]'),(.08,.2,'(.08,.2]'),(.2,1.,'(.2,1]')]:
            mask=(samples[positive]['share']>low)&(samples[positive]['share']<=high)
            for neg in ['hf_high','st_francis_dense','hf_quiet']:
                if not mask.any():continue
                for model in d['per_model_metrics'][positive]:
                    p=predictions[positive];n=predictions[neg];z=np.r_[np.ones(mask.sum()),np.zeros(len(samples[neg]['share']))]
                    bins.append({'positive_mode':positive,'negative_panel':neg,'model':model,'true_share_bin':label,
                        'n_positive':int(mask.sum()),'n_negative':int((z==0).sum()),
                        'auc_score':auc(z,np.r_[p[model+'_score'][mask],n[model+'_score']]),
                        'auc_share':auc(z,np.r_[p[model+'_share'][mask],n[model+'_share']])})
    pd.DataFrame(shape).to_csv(OUT/'support_and_quantile_pollution.csv',index=False)
    pd.DataFrame(components).to_csv(OUT/'prediction_components.csv',index=False)
    pd.DataFrame(bins).to_csv(OUT/'conditional_target_bin_AUC.csv',index=False)
    pins=json.loads((OUT/'source_model_pins.json').read_text())['pins']
    for name,value in pins.items():
        if sha(ROOT/name)!=value:raise AssertionError('Original immutable input changed:'+name)
    freeze=json.loads((OUT/'FROZEN_BEFORE_PREDICTIONS.json').read_text())
    for name,value in freeze['panel_file_sha256'].items():
        if sha(OUT/name)!=value:raise AssertionError('Frozen diagnostic panels changed:'+name)
    write(OUT/'SAVED_OUTPUT_VERIFICATION.json',{'passed':True,'crossing_rows_recomputed':len(frame),
        'maximum_numeric_metric_error':maximum,'original_pins_rechecked':len(pins),'frozen_panel_hashes_exact':True,
        'same_target_per_window_checked_except_named_fixed030_arm':True,'actual_observation_histograms':shape,
        'new_fit_inference_Test_access':False,'script_sha256':sha(__file__)})
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
    modes=['r9_original','full_smooth12','full_smooth13','full_smooth15']
    for negative,color in [('hf_high','#ba683a'),('st_francis_dense','#8f4665'),('hf_quiet','#267290')]:
        vals=[float(frame[(frame.model=='all5_frozen')&(frame.positive_mode==m)&(frame.negative_panel==negative)].auc_score.iloc[0]) for m in modes]
        axes[0].plot(range(4),vals,'o-',color=color,label=negative)
    axes[0].set_xticks(range(4),['Native short','12 min','13 min','15 min']);axes[0].set_ylim(0,1)
    axes[0].axhline(.5,color='#aaa',linestyle='--');axes[0].set_ylabel('Known-addition ROC-AUC (score)')
    axes[0].set_title('Same per-window target; changed positive support');axes[0].legend(frameon=False)
    pp=pd.DataFrame(components);subset=['r9_original','full_smooth15','hf_high','st_francis_dense']
    for model,color in [('bg_cnn','#3386a0'),('tfgrid_fdy_conformer','#be8835'),('fdy','#728251'),('itransformer','#916595'),('lgbm_physics','#545c65')]:
        vals=[pp[(pp.panel==k)&(pp.model==model)].recovered_component_mean.iloc[0] for k in subset]
        axes[1].plot(range(4),vals,'o-',label=model,color=color)
    axes[1].set_xticks(range(4),['Short positive','15-min positive','HF A=0','SF A=0']);axes[1].set_ylabel('Mean contrast / tree classifier component')
    axes[1].set_title('Frozen score component response');axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/'full_support_diagnostics.png',dpi=180);plt.close(fig)
    lines=['# R11-1 장시간 추가 × 고주파·St Francis·경계 분 대조', '',
      '기존 R9 Fold0 모델5종과 동결 all5를 CPU FP32로 실행했다. 7개양성모드×500건과 음성5패널, 총5812개입력을 한 번의결합query로추론했다. 학습·Test추론·정책변경은없다.', '',
      '![장시간 support와 score 반응](full_support_diagnostics.png)', '',
      '## 주요 결과: 같은 정답을 유지한 all5 비교', '',
      '|양성 support|HF상위 음성: RMSE / scoreAUC / shareAUC|St Francis 음성: RMSE / scoreAUC / shareAUC|',
      '|---|---:|---:|']
    for mode in ['r9_original','full_smooth12','full_smooth13','full_smooth15','native_long15','stationary15_control','full_smooth15_share030']:
        cells=[]
        for neg in ['hf_high','st_francis_dense']:
            z=frame[(frame.model=='all5_frozen')&(frame.positive_mode==mode)&(frame.negative_panel==neg)].iloc[0]
            cells.append(f"{z.rmse:.6f} / {z.auc_score:.6f} / {z.auc_share:.6f}")
        lines.append('|'+mode+'|'+'|'.join(cells)+'|')
    lines+=['', '장시간15분으로 바꾼 양성과 동일 StFrancis 무추가 패널의 비교에서 AUC0.897949→0.467513으로 하락했다. y평균0.141007/RMS0.213753은 창별로 정확히 같았다. 음성 score가 양성보다 더 높은 strict pair 비율은53.25%였다. 이 조건은 낮은 AUC 실패를 재현했지만 RMSE0.116927은 Public0.217536과 같지 않다. 임의로 두수치를 맞추도록 prevalence/target을조정하지않았다.', '',
      'Native장시간참조에서도 StFrancis 비교 AUC0.520109였다. 현재509풀에는 완전native15분이상 참조가7개이며500개독립장시간녹음이아니다. 모든7개는weak-class tug/fishing이고4분원자료를늘린smooth실험과별도다. 동일y라도참조identity분포가달라native대조하나로장시간외다른요인을제거하진못한다.', '',
      'stationary15는평균스펙트럼을15분유지하는관측가능성반례다. 실제선박통과가항상정적이라는주장은아니다. 고정y=.30의별도팔은양성분포가다르며기존y동일비교와혼동하지않는다.', '',
      '## 패널·물리적 통제', '',
      '- 모든 양성배경은 원 R9selection에서 완전관측500개를 선택했고음성배경과ID중복0이다. baseline가족inside/short/edge50/25/25%, EQ±6·nativepeakSNR−10..10을사전고정했다.',
      '- full-smooth는동일원minute행렬을12/13/15분으로선형리샘플링하고0.6~1.0의완만한envelope를적용했다. waveform반복이나새실측음원을만든것이아니다. 13/15팔은순수배경분이각각2/0개이고12팔은3개로별도표시한다.',
      '- α는실제관측추가energy로동일y를풀었다. cell_share는A/(B+A), rawy는sum(A)/sum(B+A)다. 입력만0.1dB반올림하고w를그입력에서계산했다.',
      '- HF음성은band18..24(1995..7943Hz) 선형합의시간max−median이selection고밀도6920개q90=10.725742dB보다큰500개다. 조용한대조는q50=0.999508dB미만500개다.',
      '- StFrancis고밀도312개는모두15분관측이며실제metadata사이트다. Test의추정StFrancis-like분류와혼동하지않는다.',
      '- 경계대조500쌍은같은원배경에서flag가있는첫/끝분만QC이전값으로복원했다. metadata의qc_observed_minutes와복원후실제입력관측량은다르므로후자는별도CSV/QA에계산했다. flag는오류후보이지모든flag분이실제오류라는확정라벨은아니다.',
      '- 음성정답은추가A=0이다. 자연선박부재의정답이나선박/생물음클래스를판정한것은아니다. 일부음성패널끼리배경ID는겹치므로실험독립표본수로합치지않는다.', '',
      '## 해석', '',
      '새로운지원범위에따른취약점은분명하지만모든score가HFbump하나에절대의존한다고일반화하지않는다. NNscore는정확히0.5share+0.5×관측q의대역별시간최대−최소대비이고, HFbump는입력dB의최대−중앙값이다. LightGBMscore의후반절반은별도분류기다. 둘은다른통계량이다.',
      '13→15분에서q05/q20등기준선이추가성분에오염되는정도가달라진다. trueq분위수·component·target구간별AUC를보존했다. 낮은y에서만주로붕괴하는지와큰y도실패하는지를구분할수있다.',
      '장시간양성부족과고주파자연변동음성부족을다음학습ablation의별도축으로고정할근거다. 하지만이번challenge는공식Test의동일생성법이나원인증명이아니다. 같은사이트분포·knowny에서새모델의개선을확인하기전본학습이나제출정책을바꾸지않았다.', '',
      '[전210조합·역전율](crossing_metrics.csv) · [target구간별AUC](conditional_target_bin_AUC.csv) · [분위수오염·실제관측량](support_and_quantile_pollution.csv) · [score구성요소](prediction_components.csv) · [독립saved산술QA](SAVED_OUTPUT_VERIFICATION.json) · [선언계획](predeclared_plan.json)']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print('PASS',len(frame),'crossing metric oracles; unchanged pins',len(pins))
if __name__=='__main__':main()
