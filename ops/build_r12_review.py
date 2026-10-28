"""Build the evidence-only R12 review AFTER frozen evaluations complete."""
from pathlib import Path
import json,hashlib,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops.build_r10_diagnostics_review import simple_html,tab

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import numpy as np,pandas as pd
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/r12_mpl_cache')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190})
    evaldir=ROOT/'reports/r12_cpu_ablation_r1';R=ROOT/'reports';fig=R/'r12_figures';fig.mkdir(exist_ok=True)
    results=json.loads((evaldir/'RESULTS.json').read_text());policy=json.loads((evaldir/'selected_policy.json').read_text())
    extra_dir=ROOT/'reports/r12_mask_matched_evaluation_r2'
    extra=json.loads((extra_dir/'RESULTS.json').read_text())
    independent_qa=json.loads((R/'R12_EVALUATION_INDEPENDENT_QA_v2.json').read_text())
    if independent_qa.get('status')!='passed':raise ValueError('Independent evaluation QA not passed')
    if extra.get('status')!='completed' or not extra.get('additional_control_not_in_original8_selector'):raise ValueError('Additional matched control not fully evaluated')
    for name,h in extra['output_sha256'].items():
        if sha(extra_dir/name)!=h:raise ValueError('Additional evaluation output changed:'+name)
    extra_table=pd.read_csv(extra_dir/'results.csv');extra_rows=[]
    for _,row in extra_table[extra_table.model.eq('Blend50_L1HM1P1')].iterrows():
        extra_rows.append([row.panel,f'{row.rmse:.6f}',f'{row.auc_share:.6f}',f'{row.auc_score:.6f}',f'{row.bias:+.6f}'])
    if results.get('status')!='completed' or not results.get('all_models_frozen_before_gate') or not results.get('policy_frozen_before_locked_eval') or (evaldir/'FAILED.json').exists():raise ValueError('Evaluation not fully completed')
    from ops.evaluate_r12_ablation import check_freeze,check_policy
    _,frozen=check_freeze('reports/R12_ALL17_GLOBAL_FREEZE.json')
    check_policy(evaldir/'selected_policy.json',frozen['freeze_sha256'])
    for name,h in results['output_sha256'].items():
        if sha(evaldir/name)!=h:raise ValueError('Evaluation output changed:'+name)
    table=pd.read_csv(evaldir/'results.csv');chosen=policy['selected_arm'];model='Blend50_'+chosen
    selected=table[table.model.eq(model)];old=table[table.model.eq('R9_frozen_all5')]
    data=json.loads((ROOT/'data/synthetic/r12_cpu_ablation_v3/manifest.json').read_text())
    basispath=ROOT/'models/r12_shared_basis_v1/basis.npz'
    from pipeline.features_r12 import load_spectral_basis
    basis=load_spectral_basis(basispath)
    cnn=ROOT/'models/r12_cpu_cnn_r1';summaries={p.parent.name:json.loads(p.read_text()) for p in cnn.glob('*/summary.json')}
    if len(summaries)!=9 or any(v['epochs_run']!=100 for v in summaries.values()):raise ValueError('Do not publish incomplete CNN ablation as completed')
    f,ax=plt.subplots(1,2,figsize=(12.8,4.5))
    for name,s in summaries.items():
        history=pd.read_json(cnn/name/'epochs.jsonl',lines=True)
        ax[0].plot(history.epoch,history.validation_fit_rmse,label=name,alpha=.9,lw=1.3)
    ax[0].set(xlabel='Epoch (all runs:100)',ylabel='Validation-fit RMSE',title='All budget-matched CPU CNN runs');ax[0].legend(fontsize=6.8,ncol=2)
    for name in ['legacy_contrast_control',chosen]:
        h=pd.read_json(cnn/name/'epochs.jsonl',lines=True)
        ax[1].plot(h.epoch,h.train_online_rmse,'--',label=name+' train-online')
        ax[1].plot(h.epoch,h.validation_fit_rmse,label=name+' validation-fit')
    ax[1].set(xlabel='Epoch',ylabel='RMSE',title='Control and gate-selected recipe');ax[1].legend(fontsize=7)
    f.tight_layout();f.savefig(fig/'01_training_curves.png');plt.close(f)
    gm=policy.get('gate_metrics',{})
    # Evaluation schema intentionally preserved verbatim alongside this prose.
    records=[]
    for _,row in selected.iterrows():
        candidates=old[old.panel.eq(row.panel)]
        previous=candidates.iloc[0] if len(candidates)==1 else None
        records.append([row.panel,f'{row.rmse:.6f}',f'{row.auc_share:.6f}' if pd.notna(row.auc_share) else '—',
          f'{row.auc_score:.6f}' if pd.notna(row.auc_score) else '—',f'{row.bias:+.6f}',
          '—' if previous is None else f'{previous.rmse:.6f} / {previous.auc_share:.6f} / {previous.auc_score:.6f}'])
    nrows=[]
    for name,s in summaries.items():
        m=s['validation_fit_metrics'];nrows.append([name,s['parameters'],s['epochs_run'],s['best_epoch'],f"{m['rmse']:.6f}",f"{m['auc_share']:.6f}",f"{m['contrast_ge099_fraction']:.1%}"])
    factorial=[]
    def number(arm,panel,field):
        v=table[table.model.eq('Blend50_'+arm)&table.panel.eq(panel)]
        return '—' if len(v)!=1 or pd.isna(v.iloc[0][field]) else f'{v.iloc[0][field]:.6f}'
    for arm in sorted(v for v in summaries if v!='legacy_contrast_control'):
        factorial.append([arm,number(arm,'R12_validation_gate','rmse'),number(arm,'R12_eval','rmse'),
            number(arm,'R11_full_smooth15_VS_st_francis_dense','rmse'),number(arm,'R11_full_smooth15_VS_st_francis_dense','auc_score'),
            number(arm,'R11_native_long15_VS_hf_high','auc_score')])
    components=[]
    for panel in ['R11_full_smooth15_VS_st_francis_dense','R11_full_smooth15_VS_hf_high','R11_native_long15_VS_st_francis_dense','R11_native_long15_VS_hf_high']:
        def value(name,col):
            r=table[table.model.eq(name)&table.panel.eq(panel)]
            return '—' if len(r)!=1 or pd.isna(r.iloc[0][col]) else f'{r.iloc[0][col]:.6f}'
        components.append([panel,value('CNN_'+chosen,'auc_share'),value('Tree_'+chosen,'auc_share'),value('Tree_'+chosen,'auc_classifier'),value('Tree_'+chosen,'auc_score'),value(model,'auc_score')])
    transfer=selected.merge(old,on='panel',suffixes=('_new','_R9'))
    if len(transfer):
        view=transfer;f,axes=plt.subplots(1,2,figsize=(15,max(5,len(view)*.4)))
        ix=np.arange(len(view));axes[0].barh(ix-.16,view.rmse_R9,.3,label='Frozen R9 all5');axes[0].barh(ix+.16,view.rmse_new,.3,label='R12 selected fixed50/50')
        axes[0].set(yticks=ix,yticklabels=view.panel,xlabel='RMSE (lower is better)',title='Same observations and labels');axes[0].tick_params(axis='y',labelsize=7);axes[0].legend(fontsize=8)
        axes[1].barh(ix-.23,view.auc_score_R9,.22,label='R9 original score');axes[1].barh(ix,view.auc_share_R9,.22,label='R9 share-only');axes[1].barh(ix+.23,view.auc_score_new,.22,label='R12 share-only score')
        axes[1].set(yticks=ix,yticklabels=[],xlim=(0,1),xlabel='ROC-AUC (higher is better)',title='Score change distinguished from regression');axes[1].axvline(.5,color='gray',lw=.6);axes[1].legend(fontsize=8)
        f.tight_layout();f.savefig(fig/'02_selected_transfer.png');plt.close(f)
    body=['# R12 CPU 개선 비교 결과', '',
      '**범위:** 실제 Mac CPU 소형 학습과 고정 패널 비교다. Test.csv 및 공식 제출은 사용하지 않았다. 비공개 대회 생성기의 원인을 확정하거나 새 LB 점수를 예측하는 보고서가 아니다.', '',
      '**실행 결과:** 원래 비교 17개(100 epoch CNN 9개 + LightGBM 회귀·분류 묶음 8개)와 엄격한 마스크 일치 추가 대조 2개를 완료했다. LightGBM 묶음마다 회귀기와 분류기를 각각 학습했다. 장시간 통과와 자연 HF 배경을 구분하지 못하던 실패는 로컬에서 상당히 개선됐지만, 미관측 사이트의 일반 패널은 악화됐다. 따라서 모든 환경에서 동시에 개선됐다는 판정이나 GPU 전체 학습 재개 판정은 내리지 않았다.', '',
      tab(['동일 패널 비교','기존 R9 RMSE → R12','기존 score AUC → R12 share-only AUC'],[
        ['15분 smooth 통과 / St Francis','0.116927 → 0.070341','0.467513 → 0.881891'],
        ['실제 native 15분 / St Francis','0.095733 → 0.063309','0.520109 → 0.872808'],
        ['15분 smooth 통과 / HF 변동','0.116521 → 0.075561','0.589224 → 0.852800'],
        ['고SNR·장시간 추가음','0.378832 → 0.155999','0.968250 → 0.996725'],
        ['미관측 사이트의 일반 조건','0.048462 → 0.052278','0.978000 → 0.837713']]), '',
      '위 AUC는 score 정의가 달라진 비교다. 기존 R9의 share-only AUC를 기준으로 보아도 15분 smooth/SF는 0.463295→0.881891, native/SF는 0.536763→0.872808로 개선됐다. 반면 미관측 일반 조건에서는 R9 share-only 0.899206→0.837713으로 떨어졌다. 아래 표에는 두 score를 모두 보존했다.', '',
      '## 1. 무엇을 바꿨는가', '',
      '모든 개선 신경망의 손실을 `Share MSE + 0.25 × energy-weighted cell MSE`로 고정했다. 기존 contrast BCE는 동일 예산의 대조 CNN 한 번에만 사용했다. 개선 후보의 진단 score는 예측 share 자체다. Contrast는 포화도를 관찰하기 위해서만 계산하며 손실과 제출 점수의 결합에 쓰지 않는다.',
      '비교 축은 장시간 양성 보강(L), TRAIN의 고주파 변동·St Francis 고밀도 무추가 배경 보강(H), TRAIN 전용 PCA 잔차 및 주파수 곡률 특징(P)이다. 2×2×2의 8개 조건과 기존 손실 대조 1개를 비교했다. PCA 추가는 기존 관측값과 TRAIN 배경으로부터 계산한 표현·사전정보이며, 새로운 센서 정보를 얻거나 성분 분리의 식별성을 보장하는 방법이 아니다.', '',
      '**H1 해석의 주의점:** TRAIN 감사에서 원형 HF/SF 보강은 관측 밀도와 정답의 연관도 바꿨다. SF 15분 관측군의 무추가 비율은 20.68%→66.97%, SF 내부 −Nobs 단독 AUC는 0.503→0.795였다. 따라서 H1의 성능 차이를 HF 대조 효과만으로 해석하면 안 된다. 기존 실험과 선택 정책은 보존하고, 별도의 사이트·센서·녹음기 라벨·관측 마스크 일치 대조군을 추가했다. [전체 TRAIN 마스크 교차표](R12_TRAIN_MASK_PROVENANCE_AUDIT.md)', '',
      '## 2. 데이터와 검증', '',
      '각 TRAIN 파일은 12,000개 합성 창(15분×25대역)이며, 무추가 2,400개·양성 9,600개다. CNN은 고정된 공통 6,000개 부분집합(무추가 1,200개·양성 4,800개), LightGBM은 12,000개 전체를 사용했다. 같은 row index를 조건별로 대응시켰다. 변형 파일의 수를 독립 녹음 수로 세지 않는다.',
      '장시간 조건은 L1 TRAIN 양성의 30%다. 공통 validation의 양성에서는 50%, 별도 고SNR·장시간 패널에서는 100%로 구성해 장시간 전이를 더 엄격히 평가했다. 실제 native 장시간 참조 7개뿐 아니라 5~14분 참조 106개의 통계 행렬에 완만한 12~15분 궤적을 적용했다. 12~15분은 마스킹 전 생성 궤적의 길이이며, 결측이 있으면 실제 관측 support는 더 짧을 수 있다. 이는 새로 수집한 장시간 WAV가 아니다. 같은 hull·인접 시각이어도 수신기 연속성을 증명할 메타데이터가 없어 임의로 연결하지 않았다.',
      '모든 양성은 `sum(OA)>0`을 만족한다. 관측 마스크와 짧은 edge 경로가 충돌하면 같은 참조의 유효 위치로 재배치하고, 불가능한 edge는 inside로 바꿔 기록했다. 실제 무추가 비율은 정확히 20%다. 정답 q/y는 알려진 비반올림 성분 에너지로 계산하며, 입력과 풀링 가중치는 0.1 dB 관측에서 계산한다.',
      '검증은 fit 1,000개, recipe 선택용 gate 500개, 선택 후 평가 500개다. 원천 그룹 분리와 900초 support 기준 24시간 purge를 유지했다. 새 검증 row는 과거 R11에서 사용한 row와 겹치지 않지만 원천 그룹은 이미 진단에 노출됐고 VTUAD 참조도 역할 간 공유한다. 따라서 완전한 맹검 또는 참조 독립 CV라고 부르지 않는다.',
      '새 validation의 고밀도 St Francis 배경은 0개다. 특별 HF 무추가 배경은 fit/gate/eval 각각 28/4/10개로, 계획 대비 72/46/40개 부족하다. 부족한 부분은 일반 무추가 배경으로 남겼다. TRAIN 보강 할당량은 충족됐다. 이 한계는 기존 R11 SF/HF 패널의 별도 보고로 구분했다.', '',
      f"PCA는 고유 TRAIN 배경 2,000개에서만 적합했다. 안전 대역 0~22에서 분별 평균 레벨을 제거한 3성분의 설명 분산은 **{sum(basis['explained_variance_ratio']):.2%}**다. 첨부 제안의 93% 전제를 그대로 적용하지 않았다. 신규 잔차 특징은 상위 두 불확실 대역을 쓰지 않는다. 0.1 dB 격자와 교환 가능한 공통 gain 이동은 불변이며, 반올림 경계를 넘는 임의 gain 이동까지 수치적으로 완전 불변이라고 주장하지 않는다.",
      '사전 고정된 정방향 스칼라 점검에서 양의 PCA 잔차 RMS는 full15/SF AUC 0.1032, native15/SF AUC 0.0989로 오히려 역전됐다. 따라서 “양의 잔차가 크면 추가 선박음”이라고 단정하면 안 된다. 부호가 있는 잔차 평균은 각각 0.8741/0.8711이었지만, 이 역시 전체 모델의 성능이나 에너지 분리의 식별성을 증명하지 않는다. 이미 구성한 signed 잔차·양의 잔차·곡률 세 채널을 모두 유지했고 이 결과로 실험 설정을 바꾸지 않았다. [모든 스칼라와 독립 검산](r12_spectral_probe/README.md)', '',
      '기존 8채널에도 주파수 형태 정보가 전혀 없었던 것은 아니다. 관측 셀에서는 시간 중앙값 차감 채널과 상대 스펙트럼 채널을 합쳐 `Q(db) − median_tf(Q(db)) = 10 × ch0 + 20 × ch4`를 복원할 수 있다. 이는 반올림 전 원시 dB가 아니라 공통 gain을 제거한 0.1 dB 격자다. PCA 잔차는 사라진 원시 정보를 되찾는 것이 아니라, TRAIN 배경으로부터 얻은 사전정보를 더 쉽게 학습하도록 명시적으로 표현한 것이다.', '',
      '[데이터 구축과 423개 검산](R12_DATASET_CONSTRUCTION.md) · [특징 검증](R12_FEATURES_FIXTURE.md) · [학습 엔진 검증](R12_TRAINING_ENGINE_VERIFICATION.md)', '',
      '## 3. 실제 100 epoch CNN 결과', '',
      '모든 CNN은 동일한 약 2.7만 파라미터 구조다. Shape-off도 추가 세 채널을 0으로 패딩해 모델 크기를 유지했다. Adam, dropout 0.1, 학습률 0.001에서 고정 100 epoch cosine schedule, early stopping 없음이다. 최고 checkpoint는 fit RMSE로만 골랐다. 이 선택은 관찰한 checkpoint 중 최적값이며 전역 최소해의 수학적 증명이 아니다.', '',
      tab(['조건','파라미터','실행 epoch','best epoch','fit RMSE','fit share AUC','진단 contrast≥.99'],nrows), '',
      '![학습 곡선](r12_figures/01_training_curves.png)', '',
      '## 4. 선택된 레시피와 고정 패널 전이', '',
      f"선택된 조건은 **{chosen}**다. 새 gate의 RMSE로만 선택했으며, CNN과 트리 share를 각각 50%로 고정했다. Test/리더보드 값으로 가중치·보정값·타깃 분포를 맞추지 않았다. 선택 정책 파일을 고정한 후 eval을 열었다. R11 패널은 이미 실패 조건을 발견하는 데 사용했으므로 개발용 전이 점검이지 새 맹검은 아니다.", '',
      tab(['고정 50/50 레시피','gate RMSE','선택 후 eval RMSE','full15/SF RMSE','full15/SF score AUC','native15/HF score AUC'],factorial), '',
      tab(['패널','R12 RMSE','R12 share AUC','R12 score AUC','R12 bias','기존 R9 RMSE / share AUC / score AUC'],records), '',
      tab(['장시간 대조','CNN share AUC','Tree share AUC','독립 Tree classifier AUC','Tree hybrid score AUC','선택 Blend share AUC'],components), '',
      'R12 소형 blend와 기존 R9 all5는 관측·정답을 똑같이 평가하지만 모델 구조·훈련량·앙상블 구성이 다르다. 순수한 loss/L/H/P 효과는 동일 예산의 R12 조건끼리 비교해야 한다. R9 대조는 기존 실제 파이프라인에서 얼마나 달라졌는지 보여주는 전이 비교다.', '',
      'R12 blend의 score는 share이므로 두 AUC는 같다. 기존 R9의 score 산식이 달랐기 때문에 R9 share-only AUC도 함께 비교해야 한다. AUC 상승만으로 회귀 추정이 좋아졌다고 판정하지 않는다. 목표 AUC 0.80과 세 패널의 RMSE 동시 개선을 달성하지 못한 조건도 그대로 포함했다.', '',
      '**남은 문제:** 미관측 사이트 일반 패널에서 선택 모델의 RMSE는 0.052278로, 동일 예산 R12 원형 0.049061 및 기존 R9 0.048462보다 나쁘다. 고SNR·장시간 패널은 RMSE가 줄었어도 평균 bias가 −0.110986이고 `y>0.2`가 SSE의 99.415%를 차지해 과소추정이 남았다. 장시간 예제 보강과 기존 짧은 통과 성능 사이의 균형이 아직 해결되지 않았다.', '',
      ('![선택 모델의 동일 패널 비교](r12_figures/02_selected_transfer.png)' if len(transfer) else ''), '',
      '## 5. 전체 수치와 해석 범위', '',
      '[전체 모델×패널 CSV](r12_cpu_ablation_r1/results.csv) · [선택 정책](r12_cpu_ablation_r1/selected_policy.json) · [대응 bootstrap 구간](r12_cpu_ablation_r1/paired_cluster_CI.csv) · [검증 실행 기록](r12_cpu_ablation_r1/RESULTS.json)',
      '독립 검산에서 주요 지표 7,497개는 저장 예측값으로 재계산한 결과와 일치했다. 최초 검산은 bootstrap의 float32/float64 차이 약 8×10⁻⁹에서 실패했고 그 기록을 보존했다. 검산 v2는 원 float32 방식의 bootstrap 끝점을 정확히 재현하고, 별도 float64 계산의 미세 정밀도 차이도 분리 확인했다. 평가 결과나 모델을 바꿔 통과시킨 것이 아니다. [독립 검산 v2](R12_EVALUATION_INDEPENDENT_QA_v2.md)',
      '원 평가의 classifier 136개·contrast 153개 집계행은 구성요소 원배열이 보존되지 않아 전체 raw 단위 독립 재검산 범위에 포함되지 않는다. share/score 원예측 기반 지표와 별도로 이 한계를 기록했다. 추가 matched 평가에는 전체 q/contrast/classifier 원배열을 저장했다.',
      '반복 비교의 신뢰구간은 평가 로더가 제공한 background_group 또는 source ID를 군집으로 사용했다. 단, R10 unseen 2개 패널의 CI는 null이다. 원자료에 base_window_index 및 sensor/week 메타데이터는 있지만 현재 로더의 군집 fallback에 연결되지 않았다. 이를 원천 식별자가 전혀 없다는 뜻으로 해석하면 안 된다. 점추정 RMSE/AUC는 계산됐다. [CI coverage 메모](R12_EVALUATION_POSTFLIGHT_NOTES.json)',
      'CI는 전체 참조·녹음일의 불확실성과 다중 비교를 완전히 보정하지 않는다. 새 eval에서 동일 예산 원형 대비 ΔRMSE는 −0.055995, 조건부 95% CI는 [−0.071100, −0.043100]이다. Δshare-AUC는 +0.0079, CI [−0.00928, +0.02207]로 0을 포함하므로 해당 새 eval의 AUC까지 확실히 개선됐다고 주장하지 않는다. 고SNR 조건은 타깃 분포도 바뀌므로 RMSE가 커졌다는 이유만으로 원인을 하나로 확정할 수 없다.',
      'GPU 본학습 준비 여부는 위 교차 패널의 실제 개선으로 판단해야 한다. 이 소형 비교가 끝났다는 이유만으로 5-Fold 대형 모델 전체가 학습됐거나, 공식 LB 성능이 입증됐다고 표시하지 않는다.', '',
      '## 6. 추가 대조: 관측 밀도와 녹음기까지 고정한 HF 보강', '',
      '원래 8개 실험의 H1에 새 마스크 교란이 있음을 TRAIN 감사로 발견한 뒤, 추가 대조군 L1HM1P1을 별도로 만들었다. 기본은 L1H0P1과 같은 장시간·스펙트럼 잔차 조건이고, 무추가 배경만 동일 사이트·센서 그룹·세부 녹음기 라벨·정확한 관측 마스크 안에서 HF 변동 상위 후보로 바꿨다. 양성 배열과 정답·모든 행의 마스크/사이트/녹음기 분포를 보존했다. 이는 메타데이터 라벨 일치이며 물리적 기기의 모든 특성이 영원히 동일하다는 보장은 아니다.',
      '목표 1,200개 중 1,199개를 교체했고, 후보가 부족한 1개는 원래 배경으로 남겼다. CNN 부분집합에서는 599개를 교체했다. 마스크 단독 AUC의 변화는 정확히 0이었다. 추가 CNN도 100 epoch/Adam/동일 예산으로 학습하고 트리도 같은 설정을 사용했다. 원래 8개 레시피의 gate 선택을 소급 변경하지 않고, 결과를 별도 비교했다.',
      '추가 대조 초기 v1은 최종 녹음기 라벨 검산 전에 시작되어 중단했다. 해당 부분 checkpoint는 보존하고 재사용하지 않았다. 엄격한 v2 자료의 독립 검산을 마친 뒤 r2에서 처음부터 다시 학습했다. 원래 17개 모델의 학습은 중단하지 않았다. [중단 기록](R12_INTERRUPTED_CONTROL_R1.json) · [최종 대조자료와 검산](R12_MASK_MATCHED_CONTROL_DATA.md)', '',
      tab(['같은 패널','추가 matched Blend RMSE','share AUC','score AUC','bias'],extra_rows), '',
      '엄격한 matched 추가 대조에서도 full15/SF AUC 0.820224, native15/SF AUC 0.806051로 기존 붕괴보다 좋아졌고 새 eval RMSE는 0.062234였다. 그러나 원 H1 선택 조합의 SF AUC(0.881891/0.872808)보다는 낮았다. 원래 H1의 향상 전부를 HF 배경 학습 효과라고 볼 수 없다는 경고가 실제 비교에서도 남는다. matched 대조도 미관측 일반 RMSE 0.052679, 고SNR bias −0.111366을 보여 두 문제를 해소하지 못했다. 추가 대조는 원래 선택된 레시피를 소급 교체하지 않았다.', '',
      '[추가 대조 전체 모델·기존 조건 비교](r12_mask_matched_evaluation_r2/RESULTS.md) · [대응 CI](r12_mask_matched_evaluation_r2/paired_cluster_CI.csv) · [raw 마스크·진단 구성요소 및 입력 증빙](r12_mask_matched_evaluation_r2/RESULTS.json)', '',
      '[사전 고정 실험 설정](../configs/r12_cpu_ablation_v1.json) · [R11 진단](R11_PINPOINT_DIAGNOSTICS.md)']
    md='\n\n'.join(body)+'\n';(R/'R12_CPU_ABLATION_RESULTS.md').write_text(md)
    css='body{margin:0;background:#f3f6f8;color:#192a3a;font:16px/1.85 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif}main{max-width:1250px;margin:30px auto;padding:35px;background:white;border-radius:15px}h1{font-size:29px}h2{margin-top:42px;border-bottom:2px solid #dce7ec;padding-bottom:8px}table{border-collapse:collapse;display:block;overflow:auto;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dde5ea;text-align:left}th{background:#ebf2f6}img{max-width:100%}a{color:#146896}code{background:#edf2f6;padding:3px}'
    (R/'R12_REVIEW_START.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R12 CPU 개선 비교</title><style>'+css+'</style><main>'+simple_html(md)+'</main></html>')
    print(json.dumps({'selected_arm':chosen,'report':'reports/R12_CPU_ABLATION_RESULTS.md','evaluated_rows':len(table)},ensure_ascii=False))
if __name__=='__main__':main()
