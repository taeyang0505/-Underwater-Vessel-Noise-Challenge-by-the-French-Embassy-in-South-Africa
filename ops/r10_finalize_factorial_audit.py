"""Independent saved-output oracles and review tables for R10 probes."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r10_diagnostics/factorial_transfer_r2'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def auc(y,s):
    z=np.asarray(y)>0;n=int(z.sum());m=len(z)-n
    if not n or not m:return None
    return float((rankdata(s,method='average')[z].sum()-n*(n+1)/2)/(n*m))
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def main():
    results=json.loads((OUT/'RESULTS.json').read_text())
    schedule=json.loads((OUT/'schedule.json').read_text())
    labels=np.array([s['no_source'] for s in schedule]);snr=np.array([s['snr_db'] for s in schedule])
    expected=1/(1+10**(-snr/10));expected[labels]=0
    rows=[];slices=[];checks=[]
    for name,reported in results['results'].items():
        z=np.load(OUT/name/'samples.npz',allow_pickle=False);p=np.load(OUT/name/'predictions.npz',allow_pickle=False)
        x=z['db'];y=z['share'].astype(float)
        if not np.isfinite(x).all() or x.shape[1:]!=(15,25):raise AssertionError('Complete-observation contract')
        grid=float(np.abs(x*10-np.round(x*10)).max())
        if grid>1e-3:raise AssertionError('0.1dB grid broke')
        if name.startswith(('seen__','unseen__')):
            if not np.array_equal(y==0,labels):raise AssertionError('Shared zero policy')
            if 'mean_scale' in name or 'combined' in name:
                if np.max(np.abs(y-expected))>3e-8:raise AssertionError('Independent mean-SNR target identity')
            recipes=json.loads((OUT/name/'metadata.json').read_text())['recipes']
            maxgap=max(r['raw_pool_identity_error'] for r in recipes)
            maxround=max(r['rounded_pool_label_gap'] for r in recipes)
            if maxgap>1e-10 or maxround>.012:raise AssertionError('Physical identity/bound')
        else:maxgap=maxround=None
        checks.append({'panel':name,'n':len(y),'zero_count':int(np.sum(y==0)),
            'grid_abs':grid,'raw_energy_identity_max':maxgap,'rounded_pool_gap_max':maxround})
        source=None
        if name.startswith(('seen__','unseen__')):
            source=pd.read_parquet(OUT/(name.split('__')[0]+'_source_manifest.parquet'))
        for model,r in reported.items():
            pred=p[model+'_share'].astype(float);score=p[model+'_score'].astype(float)
            rmse=float(np.linalg.norm(pred-y)/np.sqrt(len(y)));a=auc(y,score);a_share=auc(y,pred)
            if abs(rmse-r['rmse'])>1e-12 or (a is not None and abs(a-r['auc_score'])>1e-12):raise AssertionError('Metric oracle mismatch')
            row={'panel':name,'model':model,**r};rows.append(row)
            if source is not None:
                for category,ids in [('site',source.site),('sensor_group',source.sensor_group),
                                     ('target_bin',pd.cut(y,[-1e-9,0,.03,.08,.2,.4,1.],include_lowest=True).astype(str))]:
                    for key in sorted(set(ids),key=str):
                        mask=np.asarray(ids==key);e=pred[mask]-y[mask]
                        slices.append({'panel':name,'model':model,'category':category,'slice':str(key),'n':int(mask.sum()),
                            'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'auc_score':auc(y[mask],score[mask]),
                            'target_mean':float(y[mask].mean()),'target_rms':float(np.sqrt(np.mean(y[mask]**2)))})
    pd.DataFrame(rows).to_csv(OUT/'all_models_metrics.csv',index=False)
    pd.DataFrame(slices).to_csv(OUT/'site_and_target_slices.csv',index=False)
    write(OUT/'INDEPENDENT_VERIFICATION.json',{'passed':True,'script_sha256':sha(__file__),
        'panels':checks,'metric_oracle':'norm(error)/sqrt(N), independently ranked Mann-Whitney AUC',
        'mean_scale_oracle':'y=SNR_linear/(1+SNR_linear) on positive observed-window mean-power draws',
        'production_sources_models_policy_unchanged':results['original_source_policy_model_pins_unchanged'],
        'results_sha256':sha(OUT/'RESULTS.json')})
    # Saved output plots, no Test queries, no parameter fitting.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['r9_complete','mean_scale_only','smooth_trajectory_only','eq12_only','combined_stress']
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for d,c in [('seen','#2c6785'),('unseen','#c07932')]:
        rm=[results['results'][d+'__'+k]['frozen_all5']['rmse'] for k in names]
        av=[results['results'][d+'__'+k]['frozen_all5']['auc_score'] for k in names]
        axes[0].plot(range(5),rm,'o-',color=c,label=d)
        axes[1].plot(range(5),av,'o-',color=c,label=d)
    for ax in axes:
        ax.set_xticks(range(5),['R9\ncomplete','Mean-power\nscale','Smooth\ntrajectory','EQ\n±12 dB','Combined\nstress'])
        ax.grid(alpha=.2);ax.legend(frameon=False)
    axes[0].set_ylabel('RMSE');axes[1].set_ylabel('Presence ROC-AUC')
    axes[1].set_ylim(.5,1.01);fig.suptitle('Frozen Fold0 all-five: controlled CPU development probes (N=1,000/cell)')
    fig.savefig(OUT/'factorial_transfer.png',dpi=180);plt.close(fig)
    lines=['# R10 동결 모델: 배경 × 생성 연산자 교차 진단','',
      'Mac CPU FP32, 동결 Fold0 모델 5종과 기존 가중치(신경망 각각0.1, LightGBM0.6)를 사용했다. Test·미사용 report·GPU·재학습은 사용하지 않았다. 10개 교차 셀 × 1,000건과 ambient 500쌍을 새 development 진단 자료로 생성했다.',
      '', '![배경·연산자별 RMSE와 AUC](factorial_transfer.png)', '',
      '모든 교차 셀은 경계 분 QC 이후 **15분 모두 관측된 창**으로 제한했다. 따라서 관측량은 정확히 같고 무추가 비율은20%다. 이는 기존 R9 전체 검증 분포가 아닌 완전 관측 조건부 시험이다. 기존 selection과 같은 원천 풀에서 새로운 배경 창을 뽑았으며 모델 선택에 사용하지 않았다.', '',
      '| 배경 | 생성 연산자 | RMSE | AUC score | 편향 | 정답 RMS | 평균 상수 RMSE |',
      '|---|---|---:|---:|---:|---:|---:|']
    for d in ['seen','unseen']:
        for name in names:
            r=results['results'][d+'__'+name]['frozen_all5']
            lines.append(f"|{d}|{name}|{r['rmse']:.6f}|{r['auc_score']:.6f}|{r['bias']:+.6f}|{r['target_rms']:.6f}|{r['constant_mean_rmse']:.6f}|")
    lines+=['', '기존 생성기를 새 배경에 옮겨도 이번 표본에서는 성능이 무너지지 않았다. 관측 평균 전력 SNR로 바꾸면 목표 비율 자체도 커지며(양성은 SNR/(1+SNR)), 모델이 큰 비율을 과소추정했다. 따라서 RMSE 차이를 순수한 사이트 암기·생성기 과적합의 인과 효과라고 단정하지 않는다. AUC는 높게 남아서 공식 AUC0.555 붕괴까지 재현한 것은 아니다.', '',
       'mean_scale_only는 관측된15분 전체의 참조 평균 전력(무추가 분의0포함)과 배경 평균 전력으로 스케일을 계산한다. smooth_trajectory_only는 활성 support 중심으로0.8~1.25배 resampling과 끝값이0으로 강제되지 않는 Gaussian envelope를 적용한다. EQ±12는 **진단 스트레스**이며 실제 학습 레시피±6을 변경하지 않았다. combined만으로 원인을 귀속하지 않도록 단일 연산자 변화도 각각 실행했다.', '',
       '양쪽의 TRAIN 배경 창 교집합은0이다. recorder 문자열/sensor component의 TRAIN 교집합0은 unseen에만 해당하며, seen은 기존 recorder와sensor component를 공유한다. unseen의 QC후9394개 완전 관측 창에서1000개를 비복원 추출했으며 Saldhana935·Fish Hoek35·Sea Point30개다. site별 표본이 작고 불균형하므로 다른 해역 전체의 일반화를 입증하지 않는다. 메타데이터 sensor component는 실제 장비 개체 검증을 대신하지 않는다. seen1000개 중124개 배경 창은 원래 fixed selection에도 있었으므로 새 배경 독립 검증이 아닌 같은 원천 풀의 새 혼합 진단이다.', '',
       '## 동일 정답의 선박 참조 대 ambient 참조', '',
       '같은 남아공 배경, 동일1~3분 support·smooth EQ·목표 y∈[.01,.60]를 사용했다(1분1쌍,2분160쌍,3분339쌍). 두 참조 각각의α를 물리식으로 풀어 정답을 동일하게 만들었으며 추가 ambient를0으로 재라벨링하지 않았다. 모두 양성이어서 AUC는 정의되지 않는다.', '',
       '| 참조 | RMSE | 편향 | 정답 RMS |', '|---|---:|---:|---:|']
    for name in ['ship','ambient']:
        r=results['results']['paired_'+name]['frozen_all5']
        lines.append(f"|{name}|{r['rmse']:.6f}|{r['bias']:+.6f}|{r['target_rms']:.6f}|")
    lines+=['', '같은 정답에 비슷하게 예측하는 것은 에너지 회귀 목표에 부합한다. ambient의 오차가 크다는 것은 이 대조 조건에서 참조 질감에 따른 민감도가 있다는 뜻이며, 선박 순수 소음이나 장비 암기를 단독으로 식별하는 실험은 아니다.', '',
       '최초r1 진단은 실험용 warp가 짧은 edge support를 지우는 자체 오류로 중단했다. 원본 R9에는 이warp가 없다. 활성 support 중심 resampling으로 수정하고1080개 지지구간 fixture를 통과한 뒤 새r2 디렉터리에서 전체 재실행했다. 실패r1은 FAILED기록과 함께 보존했으며 결과선택에는 쓰지 않았다.', '',
       '[전 모델 지표](all_models_metrics.csv) · [사이트·타깃 구간별 지표](site_and_target_slices.csv) · [MSE/RMSE/AUC 대조와 상호작용](factorial_effects.json) · [물리·산술 독립 검산](INDEPENDENT_VERIFICATION.json) · [원천 분리/QC](background_source_audit.json) · [사전 계획](predeclared_plan.json)']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    write(OUT/'OUTPUT_MANIFEST.json',{'files_sha256':files,'files':len(files),'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())})
    print('PASS',len(rows),'metric-oracle rows',len(checks),'panels')
if __name__=='__main__':main()
