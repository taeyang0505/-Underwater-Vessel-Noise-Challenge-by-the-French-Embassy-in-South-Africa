"""Render R16 local-only experiment evidence. No challenge query IO."""
from __future__ import annotations
import csv, html, json, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops.r16_forward_data import OPERATORS,sha,guard
from ops.r16_clean_models import metrics,write
from ops.r16_locked_comparison import get_saved,verify
BASE=ROOT/'reports/r16_locked_comparison_r1';REPORT=ROOT/'reports/r16_review_r1'
ROLES=('local_fit','local_gate','local_report','domain_holdout')
LABELS={'local_fit':'Development fit','local_gate':'Background gate','local_report':'Reference + crop','domain_holdout':'Site + reference + crop'}

def read(p):return json.loads(Path(p).read_text())
def npz(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def table(rows,cols):
    def v(x):return f'{x:.6f}'if isinstance(x,float)else str(x)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(v(r.get(k,'')) for k in cols)+' |'for r in rows)
def loadsmall(role,arm):
    ds=[npz(BASE/'small_locked'/f'{role}__{op}__{arm}.npz')for op in OPERATORS]
    return{k:np.concatenate([d[k]for d in ds])for k in ds[0]}

def paired_ci(role):
    a=loadsmall(role,'control');b=loadsmall(role,'mixed');y=a['target_share'].reshape(6,400)
    assert np.array_equal(a['sample_id'],b['sample_id'])
    ea=(a['nn_share'].reshape(6,400)-y)**2;eb=(b['nn_share'].reshape(6,400)-y)**2
    rng=np.random.default_rng(2026102717);draw=[]
    # Resample paired background units, retaining all six operator results.
    for _ in range(2000):
        ix=rng.integers(0,400,400);draw.append(float(np.sqrt(eb[:,ix].mean())-np.sqrt(ea[:,ix].mean())))
    oldfolder=ROOT/'reports/r16_frozen_evaluation_r1'/('development'if role=='local_fit'else'locked')
    groups=npz(oldfolder/'predictions'/f'{role}__control_eq6.npz')['background_group']
    unique=np.unique(groups);cluster=[];idx=[np.flatnonzero(groups==g)for g in unique]
    for _ in range(2000):
        ix=np.concatenate([idx[j]for j in rng.integers(0,len(unique),len(unique))])
        cluster.append(float(np.sqrt(eb[:,ix].mean())-np.sqrt(ea[:,ix].mean())))
    return {'role':role,'RMSE_mixed_minus_control':float(np.sqrt(eb.mean())-np.sqrt(ea.mean())),
            'paired_background_bootstrap_p025':float(np.quantile(draw,.025)),
            'paired_background_bootstrap_p975':float(np.quantile(draw,.975)),
            'background_source_groups':int(len(unique)),
            'paired_source_group_bootstrap_p025':float(np.quantile(cluster,.025)),
            'paired_source_group_bootstrap_p975':float(np.quantile(cluster,.975)),
            'not_ref_cluster_ci':'reference components only5 per evaluation partition; CI conditions on chosen references'}

def main():
    sys.addaudithook(guard);verify()
    REPORT.mkdir(exist_ok=True);fig=REPORT/'figures';fig.mkdir(exist_ok=True)
    policies=read(BASE/'POLICIES.json')['policies'];small=read(BASE/'small_locked/RESULTS.json');old=read(BASE/'old_policy_audit/RESULTS.json')
    metrics_rows=[];role_data={}
    for role in ROLES:
        ds={a:loadsmall(role,a)for a in ('control','mixed')};olddata=get_saved(ROOT/'reports/r16_frozen_evaluation_r1'/('development'if role=='local_fit'else'locked'),role)
        names=list(olddata['model_order']);cn=list(olddata['classifier_model_order']);y=ds['control']['target_share'];site=ds['control']['site']
        assert np.array_equal(y,olddata['target_share'])
        role_data[role]=ds
        for arm,d in ds.items():
            for m,pk,ck in [('CNN','nn_share','nn_classifier'),('Tree','tree_share','tree_classifier'),('60T40N','balanced_share','balanced_classifier')]:
                name=arm+' '+m;p=d[pk];c=d[ck];policyname=arm+'__'+{'CNN':'nn','Tree':'tree','60T40N':'fixed60tree40nn'}[m]
                lam=policies[policyname]['score_share_lambda'];mm=metrics(y,p,c,site)
                mm['auc_frozen_score']=metrics(y,p,lam*p+(1-lam)*c)['auc_classifier'];mm['score_lambda']=lam
                metrics_rows.append({'role':role,'model':name,**mm})
        for n in ('R9_all5_frozen_Fold0','R9_tree5','R14_bg_cnn','R14_fdy','R14_itransformer','R14_tfgrid_fdy_conformer','R14_tree5_baseline','R14_fast3_frozen_exactFold0'):
            p=olddata['share'][:,names.index(n)];c=olddata['own_score'][:,names.index(n)]
            mm=metrics(y,p,c,site);mm['auc_frozen_score']=mm['auc_classifier']
            if n in cn:mm['raw_independent_classifier_auc']=metrics(y,p,olddata['raw_classifier'][:,cn.index(n)])['auc_classifier']
            metrics_rows.append({'role':role,'model':n,**mm})
        for n,pol in policies.items():
            if 'members'not in pol:continue
            p=olddata['share'][:,[names.index(k)for k in pol['members']]]@np.asarray(pol['weights'])
            c=olddata['raw_classifier'][:,cn.index(pol['classifier_member'])];lam=pol['score_share_lambda'];mm=metrics(y,p,c,site)
            mm['auc_frozen_score']=metrics(y,p,lam*p+(1-lam)*c)['auc_classifier'];mm['score_lambda']=lam
            metrics_rows.append({'role':role,'model':n,**mm})
    cis=[paired_ci(r)for r in ROLES[1:]]
    summary={'status':'completed_local_only','created_unix':time.time(),'aggregate_metrics':metrics_rows,'paired_CI':cis,
             'panel_metrics':small['rows'],'old_panel_metrics':old['rows'],
             'Test_read_or_new_submission':False,'production_training_resumed':False,
             'global_freeze_sha256':sha(BASE/'GLOBAL_FREEZE.json')}
    write(REPORT/'RESULTS.json',summary)
    cols=['role','model','n','rmse','bias','auc_share','auc_classifier','auc_frozen_score','site_macro_rmse','prediction_mean','prediction_rms']
    with(REPORT/'aggregate_metrics.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,cols);w.writeheader();w.writerows([{k:r.get(k)for k in cols}for r in metrics_rows])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    colors=['#7d8898','#146c94','#ca8c3a','#7c5f9b','#b24d4b']
    models=['control CNN','mixed CNN','mixed 60T40N','R9_all5_frozen_Fold0','R14_fast3_frozen_exactFold0']
    lookup={(r['role'],r['model']):r for r in metrics_rows}
    f,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    x=np.arange(3);width=.15
    for i,m in enumerate(models):
        for ax,key in zip(axs,['rmse','auc_frozen_score']):
            ax.bar(x+(i-2)*width,[lookup[r,m][key]for r in ROLES[1:]],width,label=m,color=colors[i])
    axs[0].set_ylabel('RMSE (lower is better)');axs[1].set_ylabel('Frozen score ROC-AUC (higher is better)')
    for ax in axs:ax.set_xticks(x,[LABELS[r]for r in ROLES[1:]],fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axs[1].set_ylim(.45,1.);axs[0].legend(fontsize=8)
    f.suptitle('R16 locked local comparison — no challenge Test used',fontsize=14);f.savefig(fig/'01_locked_comparison.png',dpi=180);plt.close(f)
    f,axs=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
    for arm,c in [('control',colors[0]),('mixed',colors[1])]:
        rec=[json.loads(l)for l in (ROOT/'models/r16_clean_pilot_r1'/arm/'epochs.jsonl').read_text().splitlines()]
        axs[0].plot([r['epoch']for r in rec],[r['train_online_rmse']for r in rec],color=c,alpha=.65,ls='--',label=arm+' train online')
        axs[0].plot([r['epoch']for r in rec],[r['fit_rmse']for r in rec],color=c,label=arm+' development RMSE')
        axs[1].plot([r['epoch']for r in rec],[r['fit_classifier_auc']for r in rec],color=c,label=arm+' independent classifier')
    axs[0].set_ylabel('RMSE');axs[1].set_ylabel('Development ROC-AUC')
    for ax in axs:ax.set_xlabel('Epoch');ax.legend(fontsize=8);ax.grid(alpha=.15)
    f.suptitle('100 epochs / Adam / no early stopping — checkpoint selection on development only');f.savefig(fig/'02_training_curves.png',dpi=180);plt.close(f)
    panel_lookup={(r['role'],r['operator'],r['arm'],r['model']):r for r in small['rows']}
    f,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
    values=np.array([[panel_lookup[r,op,'mixed','nn']['rmse']-panel_lookup[r,op,'control','nn']['rmse']for op in OPERATORS]for r in ROLES[1:]])
    im=ax.imshow(values,cmap='RdBu_r',vmin=-.06,vmax=.06,aspect='auto');f.colorbar(im,ax=ax,label='Mixed minus control RMSE; blue = improvement')
    ax.set_xticks(np.arange(6),['control EQ6','shelf +6','shelf +12','shelf +12 / LF−6','alpha pre-EQ','subband SNR'],rotation=20,ha='right')
    ax.set_yticks(np.arange(3),[LABELS[r]for r in ROLES[1:]])
    for i in range(3):
        for j in range(6):ax.text(j,i,f'{values[i,j]:+.4f}',ha='center',va='center',color='black',fontsize=10)
    ax.set_title('Per-operator frozen small CNN comparison');f.savefig(fig/'03_operator_deltas.png',dpi=180);plt.close(f)
    # All narrative below distinguishes development selection from held-out evaluation.
    fitrows=[r for r in metrics_rows if r['role']=='local_fit'and r['model']in('control CNN','mixed CNN','control 60T40N','mixed 60T40N')]
    selected=[r for r in metrics_rows if r['role']!='local_fit'and r['model']in models]
    oldpol=[r for r in metrics_rows if r['role']!='local_fit'and '__'in r['model']]
    op_rows=[{'role':r,'operator':op,'control_RMSE':panel_lookup[r,op,'control','nn']['rmse'],'mixed_RMSE':panel_lookup[r,op,'mixed','nn']['rmse'],'mixed_score_AUC':panel_lookup[r,op,'mixed','nn']['auc_frozen_score']}for r in ROLES[1:]for op in OPERATORS]
    score_rows=[r for r in metrics_rows if r['role']!='local_fit'and r['model']in('mixed CNN','mixed Tree','mixed 60T40N')]
    md='''# R16 교차 생성기·참조·사이트 소형 학습 실험

기존 모델 평가, 새 소형 모델 2조건의 100에포크 학습, 정책 동결 후 별도 평가를 완료했다. **혼합 생성기 학습이 같은 소형 CNN의 조건 전이 오차를 줄였다. 다만 기존 모든 본체 모델을 이긴다는 결론이나 공식 LB 개선은 확인되지 않았다.** 이 실험에서는 `Test.csv` 및 기존 Test 통계 산출물을 읽지 않았다. R14 TFGrid와 Fold1–4 학습은 재개하지 않았다.

## 1. 실제 수행 범위

- 17,600개 저장 행: TRAIN control/mixed 각 4,000개, 평가 4역할×6연산자×400개. TRAIN 두 조건은 같은 4,000개 배경을 사용한다. 전체 고유 배경은 5,600개로, 연산자별 반복을 독립 음원 수로 세지 않는다.
- 참조 119클립/22개 선택 연결 성분을 A 66클립/12성분(TRAIN), B 19/5(fit·gate), C 34/5(report·domain)로 분리했다. hull/day/bytes/선택 구간 직접 중복은 0이며, 배경 역할 간 900초 support 기준 24시간 purge를 유지했다.
- 새 소형 회귀 CNN 26,305파라미터와 독립 분류 CNN 3,745파라미터(총 30,050), 회귀 466특징/분류 700특징 LightGBM을 control/mixed 각각 학습했다. 두 조건의 데이터량·seed·학습 설정은 같다.
- NN은 100epoch·Adam·dropout 0.1·cosine LR·조기 종료 없음으로 실행하고, 회귀 최저 fit RMSE/분류 최고 fit AUC 체크포인트를 따로 보관한다. Tree는 최대 600round·ES40이다. `local_fit`는 반복 선택에 사용한 개발 점수다.
- regression loss=`share MSE + 0.25×energy-weighted cell MSE`. 독립 분류 경로 BCE는 회귀 파라미터와 공유되지 않으며 교차 gradient0을 검증했다.
- 기존 R9/R14 개별 23모델과 고정 집계 6종도 동결 추론했다. R14 TFGrid는 예산을 끝까지 실행한 모델이 아닌 **중지된 epoch37 체크포인트**다.

## 2. 바꾼 생성 조건과 보존한 물리식

기준법은 기존의 EQ 후 전체 native broadband peak로 α를 계산하는 유효한 고정 SNR 생성 법칙이다. 새 조건에서는 **실제 관측되는 삽입 구간의 EQ 전 peak**로 α를 먼저 정하고, 25대역 로그 이득 평균이 0인 매끄러운 EQ를 적용한 뒤 재정규화하지 않는다. 다른 조건은 ≤200Hz와 >200Hz의 SNR을 독립 추출한다. 두 방식 모두 최종 변형 A에서 정답을 다시 계산했다.

`E=B+A`, `q=OA/E`, `y=ΣOA/ΣOE`. 저장 dB는 0.1dB 격자이며 raw q/y는 그대로 보존한다. 반올림 관측 에너지와 raw 정답 사이의 작은 풀링 차이는 정량화 오차 상한으로 검사한다. 각 파일의 명시적 무추가 비율은 정확히 20%다.

TRAIN의 새 EQ는 ±6dB, 평가에서는 ±12dB로 사전 고정했다. 자연 minute-first 전체 궤적 또는 연속 1–6분 crop을 사용하며 인공 평탄 full_smooth15·강제 highshare·반복·warp는 제거했다. report/domain은 peak 강제 포함 없이 원생 crop 시작점을 균일 추출한다. 관측 양성을 보장하는 배치와 삽입 구간 기준 α는 마스크에 조건화되므로, 이번 실험을 채널만 단독 변경한 완벽한 인과 분리로 해석하면 안 된다.

**수식 검산 결과:** scalar α 재계산은 HF/LF 상대비를 지우지 않는다. 18dB 기울기는 상대 파워비를 63.1배 바꾼다. 기존 법칙이 잘못된 정답을 만들었다고 단정하지 않고 서로 다른 생성 사전분포를 비교했다. zero-mean 로그 이득은 선형 에너지 합 보존과도 다르다.

## 3. 개발 점수와 독립 평가

''' + table(fitrows,['model','rmse','auc_classifier','auc_share'])+'''

아래 세 역할은 체크포인트·가중치·score 혼합률을 1,874개 산출물의 SHA와 함께 동결한 뒤 평가했다. gate는 별도 배경+B 참조, report는 C 참조+다른 crop, domain은 학습에서 제외한 사이트+C 참조+다른 crop이다. 각 행은 6연산자×400=2,400개를 합산한 RMSE이며 전체 OOF나 공식 LB가 아니다.

'''+table(selected,['role','model','rmse','bias','auc_frozen_score'])+'''

![Locked local comparison](figures/01_locked_comparison.png)

같은 소형 CNN의 혼합 학습 효과에 대한 paired-background bootstrap(2,000회; 음수=개선):

'''+table(cis,['role','RMSE_mixed_minus_control','paired_background_bootstrap_p025','paired_background_bootstrap_p975','background_source_groups','paired_source_group_bootstrap_p025','paired_source_group_bootstrap_p975'])+'''

창 단위 구간과 배경 source-group 단위 구간을 구분했다. 창 단위는 같은 주간·녹음기 그룹의 상관을 무시하므로 모집단/source-group 신뢰구간으로 해석하면 안 된다. source-group 재추출도 각 역할의 8/8/32개 제한된 그룹에 조건부다. 두 방식 모두 참조 5개 성분의 불확실성을 완전히 포함하지 않는다. 6연산자 반복을 독립 표본으로 재추출하지 않았다.

## 4. score 독립화와 앙상블 상한 결과

새 score는 회귀와 분리된 classifier 확률을 기본으로, share 혼합률 0 또는 0.25를 fit에서만 선택했다. 비교용으로 share 단독 및 0.75 혼합도 함께 저장했다. 독립 classifier가 모든 조건에서 항상 우월하다고 가정하지 않는다.

'''+table(score_rows,['role','model','auc_share','auc_classifier','auc_25share_75clf','auc_75share_25clf','auc_frozen_score','classifier_ge099_fraction'])+'''

분류 확률≥0.99 비율도 공개한다. 이 비율이 높은 원인이 정당한 확신인지 과도한 확신인지는 이 표만으로 단정할 수 없다. 따라서 ‘비포화가 완전히 해결됐다’고 표현하지 않는다. 확률 교정은 이번 실험에서 추가하지 않았다. report에서는 순수 분류 확률이 fit에서 선택한 0.25 혼합보다 AUC가 높았지만, 이 결과를 본 뒤 정책을 바꾸지 않았다.

기존 본체의 새 앙상블은 highshare를 제외하고 baseline Tree5+4NN을 사용했다. 고정 균형(50% Tree5+NN 각12.5%), 무제약 fit, 상한 fit을 모두 보고했다. 상한은 Tree5 가족40–80%, NN 각5–40%이며 Tree의 개별 fold는 최대16%다. **상한을 두면 반드시 RMSE가 개선된다는 가정은 하지 않는다.** 소형 60Tree40NN도 단독 CNN보다 나쁠 수 있어 모두 병기한다.

'''+table(oldpol,['role','model','rmse','auc_frozen_score'])+'''

실제 고정 가중치는 [POLICIES.json](../r16_locked_comparison_r1/POLICIES.json)에 있다. gate/report 결과를 본 뒤 가중치를 다시 맞추지 않았다. 기존 R14 모델의 가중치 상한만 바꾼 경우 domain RMSE는 약0.118로, 기존 고정 정책의 0.120보다 조금 나아졌을 뿐이다. 새로운 학습 데이터가 필요한 문제를 가중치 조정만으로 해결했다고 볼 수 없다.

## 5. 연산자별 결과와 학습 곡선

'''+table(op_rows,['role','operator','control_RMSE','mixed_RMSE','mixed_score_AUC'])+'''

![Operator deltas](figures/03_operator_deltas.png)

![Training curves](figures/02_training_curves.png)

**개선은 모든 연산자에서 균일하지 않았다.** domain의 기존 control 조건은 0.06495→0.06861로 소폭 악화했고, EQ 전 삽입 구간 기준 조건도 0.10806→0.10964로 개선되지 않았다. 큰 이득은 독립 서브밴드 SNR 조건의 0.19207→0.12668에서 나왔다. 따라서 ‘어떤 채널에서도 더 강해졌다’는 결론은 부적절하다.

학습 온라인 RMSE는 다른 dropout 상태와 epoch 내 변화하는 모델을 섞어 측정하므로 최종 체크포인트 TRAIN RMSE와 동일하지 않다. 100epoch 탐색은 global optimum을 보장하지 않는다.

## 6. 이번 결과로 결정할 수 있는 것과 남은 한계

1. 같은 소형 모델의 교차 조건 성능 개선은 실측으로 판단할 수 있다. 그러나 모델 크기·데이터 양이 다른 기존 R9/R14와의 차이를 순수하게 생성기 효과라고 분리할 수 없다.
2. 참조 분리는 119개 선택 클립 부분 그래프 내의 조건부 독립이다. 전체525개 그래프는1개 성분이며 제외 bridge는 사용하지 않았다. 기존 R9/R14는509개 참조를 이미 학습했으므로 이들의 B/C 평가를 미관측 참조라고 부르지 않는다.
3. domain의3개 사이트는 이번 새 TRAIN에는 없지만 과거 개발 진단에서 사용한 도메인이다. 새 완전 blind 평가가 아니며 비공개 Test 생성기와 동일하다는 보장도 없다.
4. 이번 소형 실험은 Fold0 기반이며 전체5-Fold OOF가 아니다. 고정 R11 장시간/SF/HF 스트레스 패널 전체 회귀 시험과 다른 seed 반복은 남아 있다. 자연 장시간 실측이 적다는 제약도 그대로다.
5. 이번 Test 추론·통계 튜닝·적대적 검증·공식 제출은0회다. 과거 개발에서 Test 기술 통계를 본 사실을 없었던 일로 만들 수 없으므로 완전 무오염 개발을 주장하지 않는다.

## 7. 본체 확장 실행 계획

지금 중지된 R14 TFGrid는14채널/구레시피 체크포인트이므로 새8채널 학습으로 이어붙이지 않는다. 기존 파일은 보존하고 새 실행 ID로 시작해야 한다.

1. 이번 고정 채널 사전분포·상대8채널·독립 분류 계약을 본체 러너에 연결하고1-step 값/gradient/CPU–GPU 일치 검사를 실행한다. 고정 R11 스트레스 패널도 코드 변경 없이 추가 평가한다.
2. 순서는 빠른 본체 CNN, FDY-CRNN, iTransformer, TFGrid-FDY-Conformer, Mac CPU PhysicsTree다. 각 신경망은100epoch·Adam·조기 종료 없음·fit 최적 체크포인트를 독립 저장하며 Fold0부터 측정한다. Tree는별도 boosting/ES 규약을 사용한다.
3. 새 입력/레시피 본체의 개선이 확인될 때만 Fold1–4로 확장한다. 생성법·seed·원천 분할·정책 SHA를 미리 고정하고 각 fold의 외부 평가는 선택에 재사용하지 않는다.
4. 실제1epoch 처리량과 메모리를 측정하기 전에는 전체 소요 시간이나 LB 목표 달성을 보장하지 않는다. **이번 보고서는 GPU/MPS 본체 학습을 재개했다는 뜻이 아니다.**

## 8. 재현 자료

- [자료/API/원천 분리](../r16_cross_operator_r1/data/README.md)
- [데이터 독립 전수 QA](../r16_independent_qa_r1/DATA_INDEPENDENT_QA.json)
- [소형 학습 독립 QA](../r16_independent_qa_r1/SMALL_MODEL_TRAINING_QA.json)
- [전역 동결 기록](../r16_locked_comparison_r1/GLOBAL_FREEZE.json)
- [전수 지표 JSON](RESULTS.json) · [합산 지표 CSV](aggregate_metrics.csv)
- [소형 패널별 지표/저장예측](../r16_locked_comparison_r1/small_locked/RESULTS.json)
- [기존 모델 동결 평가](../r16_frozen_evaluation_r1/locked/RESULTS.json)
'''
    (REPORT/'R16_CROSS_OPERATOR_LEARNING_RESULTS.md').write_text(md)
    # Lightweight local browser report, no remote scripts/fonts/telemetry.
    def ht(rows,cols):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>'for c in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r.get(c):.6f}'if isinstance(r.get(c),float)else str(r.get(c,'')))+'</td>'for c in cols)+'</tr>'for r in rows)+'</tbody></table>'
    body='''<!doctype html><meta charset="utf-8"><title>R16 local experiment</title><style>body{font:16px/1.6 -apple-system,sans-serif;background:#f3f6f9;color:#183044;max-width:1320px;margin:36px auto;padding:0 24px}section{background:white;border-radius:16px;padding:24px;margin:20px 0}h1{font-size:34px}h2{font-size:23px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid #dde5ec}th{background:#edf3f7}img{width:100%;height:auto}a{color:#126a92}.notice{border-left:5px solid #bd8540;background:#fff5e4;padding:16px}.scroll{overflow-x:auto}</style><h1>R16 교차 조건 학습 실험</h1><p>소형 모델 100epoch 완료 · 정책 고정 후 별도 평가 완료 · Test 사용 없음</p><div class="notice">혼합 생성기 학습이 같은 소형 CNN의 교차 조건 RMSE를 개선했습니다. 공식 LB 개선 또는 대회와의 간극 해소는 아직 확인하지 않았습니다. 본체 TFGrid와 Fold1–4는 중지 상태입니다.</div>'''
    body+='<section><h2>평가 결과</h2><div class="scroll">'+ht(selected,['role','model','rmse','bias','auc_frozen_score'])+'</div><img src="figures/01_locked_comparison.png"></section>'
    body+='<section><h2>연산자별 변화</h2><img src="figures/03_operator_deltas.png"></section><section><h2>학습 추이</h2><img src="figures/02_training_curves.png"></section>'
    body+='<section><h2>자료와 한계</h2><p>17,600개 저장 행 / 고유배경5,600개 / 참조A66·B19·C34클립. local_fit는개발용, 나머지패널은전역동결후개방했습니다. 기존모델은509참조를이미학습했고domain은과거개발에서평가된사이트입니다. 본 실험은전체5-Fold OOF가아닙니다.</p><p><a href="R16_CROSS_OPERATOR_LEARNING_RESULTS.md">상세 보고서 MD</a> · <a href="RESULTS.json">전수 결과 JSON</a> · <a href="aggregate_metrics.csv">합산 지표 CSV</a> · <a href="../r16_locked_comparison_r1/POLICIES.json">동결 가중치</a></p></section>'
    (REPORT/'index.html').write_text(body)
    (ROOT/'reports/R16_REVIEW_START.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=r16_review_r1/index.html"><a href="r16_review_r1/index.html">R16 실험 결과 열기</a>')
    write(REPORT/'ARTIFACTS.json',{'status':'completed','sha256':{str(p.relative_to(REPORT)):sha(p)for p in REPORT.rglob('*')if p.is_file()and p.name!='ARTIFACTS.json'},'input_freeze_sha256':sha(BASE/'GLOBAL_FREEZE.json')})
    verify();print(json.dumps({'status':'review_completed','report':str(REPORT/'R16_CROSS_OPERATOR_LEARNING_RESULTS.md')}))

if __name__=='__main__':main()
