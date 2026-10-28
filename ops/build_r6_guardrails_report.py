"""Render all seven fixed R6-v2 CPU arms after the final freeze/evaluation.

This reporter does not fit, select, reload a model, inspect a raw array, or
read Test/old OUTER. It consumes completed JSON results and epoch histories.
No measured results are read without the explicit --generate CLI flag.
The parent must additionally authorize generation after checkpoint freeze.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/r6_cpu_guardrails_v2.json'
SOURCE = 'reports/R6V2_SOURCE_SPLIT_AUDIT.json'
DATA = 'reports/R6V2_DATASET_AUDIT.json'
RESULT = 'reports/R6V2_CPU_ABLATION_RESULTS.json'
RUN = 'models/r6_cpu_guardrails_v2'
HARDNEG = 'data/versions/hard_negatives_r6_v2/manifest.json'
INDEPENDENT = 'reports/R6_GUARDRAILS_INDEPENDENT_AUDIT.json'
EVIDENCE = 'reports/r6_guardrails_training_examples.selection.json'
OUTPUT = 'reports/R6_GUARDRAILS_AND_CPU_ABLATION_REPORT.md'
FIGURES = 'reports/r6_guardrails_v2_figures'
ROLES = ('cross_generator', 'cross_sensor', 'cross_sensor_generator')
CONSTANTS = ('constant0', 'constant0_1', 'TRAIN_r6_mean', 'TRAIN_r4_mean')
SHORT = {
    'r4law_legacy8_global': 'R4 / legacy8 / context',
    'r4law_relative8_global': 'R4 / relative8 / context',
    'r6law_legacy8_global': 'R6 / legacy8 / context',
    'r6law_relative8_global': 'R6 / relative8 / context',
    'r6law_relative8_local': 'R6 / relative8 / no added context',
    'r6law_legacy_tree': 'R6 / legacy tree',
    'r6law_relative_tree': 'R6 / relative tree',
    'constant0': 'Constant 0', 'constant0_1': 'Constant 0.1',
    'TRAIN_r6_mean': 'R6 TRAIN mean', 'TRAIN_r4_mean': 'R4 TRAIN mean',
}
CONTRASTS = (
    ('r4law_legacy8_global', 'r4law_relative8_global', 'R4 law: legacy8 → relative8'),
    ('r6law_legacy8_global', 'r6law_relative8_global', 'R6 law: legacy8 → relative8'),
    ('r4law_relative8_global', 'r6law_relative8_global', 'relative8: R4 → R6 law'),
    ('r6law_relative8_local', 'r6law_relative8_global', 'R6 relative8: explicit global context 추가'),
    ('r6law_legacy_tree', 'r6law_relative_tree', 'Tree: legacy8 → relative8'),
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def fmt(value, digits=6, signed=False):
    if value is None:
        return '—'
    value = float(value)
    if not np.isfinite(value):
        raise ValueError('A report metric is nonfinite')
    return format(value, ('+' if signed else '') + '.' + str(digits) + 'f')


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |',
                      '| ' + ' | '.join(['---']*len(header)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def validate_result_structure(config, result):
    """Validate fixed coverage/order/finite metrics; never select a winner."""
    arms = config['CNN_arms'] + config['LightGBM_arms']
    names = [a['name'] for a in arms]
    if len(config['CNN_arms']) != 5 or len(config['LightGBM_arms']) != 2 or len(set(names)) != 7:
        raise ValueError('Exactly the predeclared five CNN/two tree arms are required')
    if result.get('status') != 'completed':
        raise ValueError('Final completed results required; partial runs cannot be rendered')
    summaries = result['summaries']
    if [s['arm'] for s in summaries] != arms or any(s.get('status') != 'completed' for s in summaries):
        raise ValueError('Missing, reordered, changed or incomplete arm summary')
    if set(result['heldout_results']) != set(ROLES):
        raise ValueError('All three common locked panels are required')
    if set(result['heldout_data_diagnostics']) != set(ROLES):
        raise ValueError('Final diagnostics for every locked panel are required')
    keys = {'rmse', 'site_macro_rmse', 'mae', 'bias', 'presence_auc'}
    for summary in summaries:
        if not keys <= set(summary['best_inner_metrics']):
            raise ValueError('Incomplete INNER metrics')
    for role in ROLES:
        panel = result['heldout_results'][role]
        if set(panel) != set(names + list(CONSTANTS)):
            raise ValueError('All arms and four controls must be reported: '+role)
        sizes = set()
        for name in names + list(CONSTANTS):
            metric = panel[name]
            if not keys <= set(metric):
                raise ValueError('Incomplete panel metrics: '+role+'/'+name)
            for key in keys:
                fmt(metric[key])  # includes nullable single-class AUC
            sizes.add(int(metric['n']))
        if sizes != {int(config['samples'][role])}:
            raise ValueError('Arms/controls have different panel row counts: '+role)
    return names


def read_completed_inputs(root):
    """Freeze gate is inspected before reading final measured result payloads."""
    run = root/RUN
    if (run/'FAILED.json').exists():
        raise ValueError('A failed pilot is not a completed results source')
    gate_path = run/'all_checkpoints_frozen_before_heldout.json'
    gate = load(gate_path)
    if (gate.get('status') != 'frozen' or gate.get('all_requested_arms_completed') is not True
            or gate.get('no_heldout_predictions_yet') is not True):
        raise ValueError('All seven checkpoints must have been frozen before locked evaluation')
    config = load(root/CONFIG)
    names = [a['name'] for a in config['CNN_arms']+config['LightGBM_arms']]
    if set(gate['model_summaries']) != set(names):
        raise ValueError('Freeze receipt does not cover all fixed arms')
    receipts = {str(gate_path.relative_to(root)): sha(gate_path)}
    for name, digest in gate['model_summaries'].items():
        path = run/name/'summary.json'
        if sha(path) != digest:
            raise ValueError('Frozen model summary changed: '+name)
        receipts[str(path.relative_to(root))] = digest
    result = load(root/RESULT)
    validate_result_structure(config, result)
    if result['config_sha256'] != sha(root/CONFIG):
        raise ValueError('Final result/config binding mismatch')
    if sha(run/'results.json') != sha(root/RESULT):
        raise ValueError('Run results and published JSON differ')
    data = load(root/DATA)
    if data.get('status') != 'ready' or data['freeze']['config_sha256'] != sha(root/CONFIG):
        raise ValueError('Dataset/config binding mismatch')
    if result['data_manifest_sha256'] != sha(root/DATA):
        raise ValueError('Result/dataset audit binding mismatch')
    for name, digest in data['freeze']['code_sha256'].items():
        if sha(root/name) != digest:
            raise ValueError('Data-generation/fitting source changed: '+name)
    source = load(root/SOURCE)
    hardneg = load(root/HARDNEG)
    independent = load(root/INDEPENDENT)
    evidence = load(root/EVIDENCE)
    if source.get('status') != 'PASS' or hardneg.get('ready_for_train_source_draws') is not True:
        raise ValueError('Source/hard-negative preparation gate is incomplete')
    if source.get('reference_group_hull_day_SHA_UTC_isolation_passed') is not True:
        raise ValueError('Reference source isolation gate failed')
    if (independent.get('status') != 'PASS_WITH_DECLARED_LIMITATIONS'
            or independent['dataset_manifest_sha256'] != sha(root/DATA)):
        raise ValueError('Independent TRAIN/INNER audit binding mismatch')
    if (evidence.get('status') != 'completed'
            or evidence['dataset_manifest_sha256'] != sha(root/DATA)
            or evidence.get('selection_not_based_on_predictions_errors_or_appearance') is not True
            or evidence.get('model_predictions_or_performance_used') is not False):
        raise ValueError('TRAIN illustration selection/binding mismatch')
    for filename, digest in evidence['figures_sha256'].items():
        path = root/'reports'/filename
        if Path(filename).name != filename or sha(path) != digest:
            raise ValueError('TRAIN illustration changed or path is invalid')
        receipts[str(path.relative_to(root))] = digest
    if evidence['script_sha256'] != sha(root/'ops/plot_r6_train_evidence.py'):
        raise ValueError('TRAIN illustration script changed')
    histories = {}
    for summary in result['summaries']:
        name = summary['arm']['name']
        folder = run/name
        if load(folder/'summary.json') != summary:
            raise ValueError('Result contains another arm summary: '+name)
        for artifact, digest in summary.get('model_sha256', {}).items():
            if sha(folder/artifact) != digest:
                raise ValueError('Tree artifact changed: '+name+'/'+artifact)
        if summary.get('checkpoint_sha256') and sha(folder/'best.pt') != summary['checkpoint_sha256']:
            raise ValueError('CNN checkpoint changed: '+name)
        if summary.get('prior_sha256') and sha(folder/'legacy_train_prior.json') != summary['prior_sha256']:
            raise ValueError('Legacy TRAIN prior changed: '+name)
        if name in [a['name'] for a in config['CNN_arms']]:
            history_path = folder/'epochs.jsonl'
            rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
            if (len(rows) != summary['epochs_run'] or not rows
                    or [r['epoch'] for r in rows] != list(range(1,len(rows)+1))
                    or not 1 <= summary['best_epoch'] <= len(rows)):
                raise ValueError('Incomplete epoch history: '+name)
            histories[name] = rows
            receipts[str(history_path.relative_to(root))] = sha(history_path)
    for path in [CONFIG, SOURCE, DATA, RESULT, HARDNEG, INDEPENDENT, EVIDENCE,
                 'ops/plot_r6_train_evidence.py', 'ops/build_r6_guardrails_report.py']:
        receipts[path] = sha(root/path)
    return config, source, data, result, hardneg, independent, histories, receipts


def plots(root, config, data, result, histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'savefig.facecolor':'white'})
    output = root/FIGURES
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ax, arm in zip(axes.flat, config['CNN_arms']):
        rows = histories[arm['name']]
        epoch = [r['epoch'] for r in rows]
        for key, label, style in [('train_rmse','TRAIN online RMSE','-'),
                                  ('inner_site_macro_rmse','INNER site-macro','-'),
                                  ('inner_rmse','INNER window RMSE',':')]:
            ax.plot(epoch,[r[key] for r in rows],style,label=label,lw=1.3)
        ax.set(title=SHORT[arm['name']],xlabel='Epoch',ylabel='RMSE')
        ax.grid(alpha=.15); ax.legend(fontsize=7)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0,.9,'Five predeclared CNN arms\n\nOwn-law INNER targets differ.\nUse common locked panels for comparisons.\n\nNo official Test evaluation.',
                       va='top',transform=axes.flat[-1].transAxes,linespacing=1.6)
    fig.suptitle('R6-v2: actual CPU histories; no synthetic or smoothed curves')
    fig.savefig(output/'01_cpu_curves.png',dpi=180); plt.close(fig)

    names = [a['name'] for a in config['CNN_arms']+config['LightGBM_arms']]
    keys = names+list(CONSTANTS)
    colors = ['#939ba1','#5e83ab','#c19b69','#4e8c78','#879466','#ad8aab','#687a96']+['#d6d9db']*4
    fig, axes = plt.subplots(1,3,figsize=(16,6),layout='constrained')
    for ax, role in zip(axes,ROLES):
        panel = result['heldout_results'][role]
        values = [panel[k]['rmse'] for k in keys]
        ax.barh(np.arange(len(keys)),values,color=colors)
        ax.set_yticks(np.arange(len(keys)),[SHORT[k] for k in keys],fontsize=7)
        ax.invert_yaxis(); ax.set(xlabel='Window RMSE',title=role.replace('_','\n'))
        ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('All seven frozen arms + four controls on common locked synthetic panels')
    fig.savefig(output/'02_locked_panels.png',dpi=180); plt.close(fig)

    samples = list(data['diagnostics'].items())
    fig, ax = plt.subplots(figsize=(10,4),layout='constrained')
    locations = np.arange(len(samples))
    for offset, key, label in [(-.15,'constant0_RMSE','Constant 0'),(.15,'constant0_1_RMSE','Constant 0.1')]:
        ax.bar(locations+offset,[d[key] for _,d in samples],width=.3,label=label)
    ax.set_xticks(locations,[k for k,_ in samples],rotation=20,ha='right')
    ax.set(ylabel='RMSE from stored labels',title='Target scale changes by generation law and panel')
    ax.legend(); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    fig.savefig(output/'03_target_scale_controls.png',dpi=180); plt.close(fig)
    return [p.relative_to(root).as_posix() for p in sorted(output.glob('*.png'))]


def render(config, source, data, result, hardneg, independent, histories, receipts):
    names = [a['name'] for a in config['CNN_arms']+config['LightGBM_arms']]
    neural = config['neural']; tree = config['tree']; gen = config['new_generator']
    inner_rows = []
    for summary in result['summaries']:
        m = summary['best_inner_metrics']
        if 'best_epoch' in summary:
            best = f"{summary['best_epoch']} / {summary['epochs_run']}"
            size = f"{summary['parameters']:,} parameters"
        else:
            best = f"share {summary['best_iteration']}; presence {summary['presence_best_iteration']}"
            size = f"{summary['n_features']:,} features"
        inner_rows.append([summary['arm']['name'],size,best,fmt(m['rmse']),fmt(m['site_macro_rmse']),
                           fmt(m['presence_auc']),fmt(m.get('presence_auc_share_only')),
                           fmt(m.get('presence_auc_classifier_only')),fmt(summary['elapsed_seconds']/60,2)])
    scales, sampling_rows = [], []
    for name,d in data['diagnostics'].items():
        no_source = int(d['no_source_count'])
        size = int(d['examples'])
        scales.append([name,f"{d['examples']:,}",fmt(d['mean_share']),fmt(d['zero_fraction'],4),
                       f"{no_source:,}/{size:,} ({100*no_source/size:.2f}%)",fmt(d['constant0_RMSE']),
                       fmt(d['constant0_1_RMSE']),fmt(d.get('high_band_fraction_presence_AUC'))])
        sampling_rows.append([name,f"{size:,}",f"{d['primary_unique_backgrounds']:,}",
                              f"{d['hard_negative_count']:,}",f"{d['hard_negative_unique_backgrounds']:,}",
                              f"{d['source_present_but_unobserved_count']:,}"])
    locked_tables = []
    for role in ROLES:
        panel = result['heldout_results'][role]
        rows = []
        for name in names+list(CONSTANTS):
            m = panel[name]
            rows.append([name,fmt(m['rmse']),fmt(m['site_macro_rmse']),fmt(m['mae']),fmt(m['bias'],signed=True),
                         fmt(m['presence_auc']),fmt(m.get('presence_auc_share_only')),
                         fmt(m.get('presence_auc_classifier_only'))])
        locked_tables.append('### '+role+'\n\n'+table(
            ['모델/상수','RMSE','Site macro','MAE','Bias','고정 score AUC','share AUC','classifier AUC'],rows))
    contrasts = []
    for role in ROLES:
        panel = result['heldout_results'][role]
        for a,b,label in CONTRASTS:
            contrasts.append([role,label,fmt(panel[b]['rmse']-panel[a]['rmse'],signed=True)])
    retained = [[s['site'],f"{s['old_clean_train_windows']:,}",f"{s['old_boundary_excluded_train_windows']:,}",
                 f"{s['train']:,}",f"{s['inner']:,}",f"{s['cross_generator']:,}",f"{s['cross_sensor']:,}"]
                for s in source['site_retention']]
    refrows = [[k,v['clips'],v['groups']] for k,v in source['reference_role_counts'].items()]
    hardaudit = hardneg['audit']
    hardrows = [[k,v] for k,v in hardaudit['candidate_site_counts'].items()]
    train_audit = independent['roles']['train']
    inner_audit = independent['roles']['inner']
    lineage = independent['reference_lineage']
    train_data = data['diagnostics']['train_r6']
    if (train_data['hard_negative_count'] != train_audit['hard_negative_count']
            or train_data['primary_unique_backgrounds'] != train_audit['unique_primary_windows']):
        raise ValueError('Dataset hard-negative/train-source counts disagree with independent audit')
    qc = source['source_audit']
    recovered = sum(s['retained_recovered_boundary_windows'] for s in source['site_retention'])
    purge = sum(s['new_QC_or_purge_exclusions'] for s in source['site_retention'])
    total_examples = sum(d['examples'] for d in data['diagnostics'].values())
    backgrounds = ', '.join(k+' '+format(v,',') for k,v in source['background_role_counts'].items())
    time_minutes = sum(s['elapsed_seconds'] for s in result['summaries'])/60.
    worst_identity = max(d['original_target_identity_max_error'] for d in data['diagnostics'].values())
    worst_gap = max(d['rounded_weight_target_gap_max'] for d in data['diagnostics'].values())
    diagrows = [[role,fmt(result['heldout_data_diagnostics'][role].get('high_band_fraction_presence_AUC')),
                 fmt(result['heldout_data_diagnostics'][role]['rounded_weight_target_gap_max'],8)] for role in ROLES]
    provenance = table(['입력/출력 근거','SHA256'],[[k,'`'+v+'`'] for k,v in receipts.items()])
    return f'''# R6 guardrails 개편과 Mac CPU ablation 결과

생성 시각: {datetime.now(timezone.utc).isoformat()}. **사전 정의한 5개 CNN과 2개 LightGBM의 작은 CPU 실험이다. 공식 Test 결과·5-fold OOF·대형 모델 성능·실제 LB 예측을 뜻하지 않는다.** 모든 arm과 상수 대조, 세 locked 합성 패널을 결과와 관계없이 보고한다. 이 보고서 생성기는 모델·앙상블·후처리 정책을 선택하거나 적합하지 않는다.

## 1. 이번 버전의 범위

- relative8은 기존 상대 채널 6개에 band q05 상대 dB/15와 `sqrt(w)`를 추가한다. q05는 창의 관측 통계이며, 확인된 순수 배경이나 정답 분리량이 아니다. `w`는 같은 0.1dB 관측에서 구한 정규화 선형 에너지다.
- 새 R6 law의 reference smooth EQ는 **±{gen['reference_smooth_eq_max_db']:g}dB**, low-pass 선택 확률은 **{gen['lowpass_probability']:g}**다. 시간 늘이기/반복 없이 native 궤적과 crop을 사용한다. R4 law 대조는 기존 R4 생성 법칙을 그대로 사용하므로 이 수치를 R4 대조의 EQ 설정으로 읽으면 안 된다.
- 명시적 hidden frequency context on/off를 비교하고, 자유 presence head 대신 사전 정의된 derived score를 사용한다. tree에는 safe 1–5kHz를 포함한 물리 특징 27개를 추가한다.
- TRAIN 원천만으로 LF hard-negative 후보를 정하고, 알려진 추가 성분 A=0인 합성 대조에 사용한다. 원래 바다에 선박이 전혀 없다는 정답을 얻은 것은 아니다.
- **Test/LB 분포·지역 비중·Benchmark RMSE/target moment에 맞추지 않았다.** 아래 우리 합성 데이터의 수치가 공식 hidden generator를 대표한다거나 LB 실패의 인과를 증명한다고 해석하지 않는다.

고정 설정: [r6_cpu_guardrails_v2.json](../{CONFIG}). 이번 arm은 7개로 고정했으며 자동 탐색·재시도 선택·후속 ensemble fitting을 하지 않는다.

적용하지 않은 조건도 명시한다. Benchmark 알고리즘은 확인되지 않아 그 RMSE²를 Test의 E[y²]로 읽을 수 없다. 상수 c의 MSE는 E[y²]−2cE[y]+c²이다. 지정한 전체 RMS .246, 양성 비율 .75~.8, 양성 평균 .20~.23에서는 상수 .1 RMSE가 약 .184~.201이므로 두 상수 baseline을 모두 .24~.26으로 만드는 조건은 양립하지 않는다. [사전 실행 계획](R6_GUARDRAILS_EXECUTION_PLAN.md)에 계산과 제외 범위를 남겼다. Test 집단에서 추정한 target·생성 분포에 맞추지 않는 방침은 [공식 규정](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge)의 Test 추론 전용 조건에 따른 것이며, 고정된 창별 전처리와 구분한다.

## 2. 원천 분리와 QC

배경 원천 창 수: **{backgrounds}**. 기존 R4 Fold0에서 허용된 TRAIN/INNER 원천 안에서 새 역할을 분리했다. 옛 OUTER/calibration/domain/sealed/Test 수치는 사용하지 않았다. 같은 센서 역할 간 support gap 최소 **{source['minimum_same_sensor_cross_role_support_gap_hours']:g}시간**, reference hull/day/UTC/SHA 격리를 감사했다.

| Site | 옛 clean TRAIN | 옛 경계 제외 TRAIN | 새 TRAIN | 새 INNER | cross-generator | cross-sensor |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join('| '+' | '.join(map(str,row))+' |' for row in retained)}

실제 원본 row ID로 찾은 후보 분 **{qc['admitted_flagged_selected_minutes']:,}개**를 가렸으며 창 전체 제외를 대신해 TRAIN **{recovered:,}창**을 복구했다. 추가 QC/purge 제외는 **{purge:,}창**이다. 경계 flag의 물리적 원인이 확인된 것은 아니며, 첫/끝 분을 일괄 삭제한 처리도 아니다.

{table(['역할','참조 clips','선택된 source components'],refrows)}

cross-sensor는 metadata hash로 정한 sensor group {source['sensor_selection']['selected_sensor_group']}이며, 모델 성능이나 Test site 구성을 보고 선정하지 않았다. 여기서는 St Francis 원천이다. 이번 fit의 holdout이지만 연구자가 과거에 전혀 보지 않은 새 코퍼스라고 주장하지 않는다. TRAIN에 존재하면서 INNER가 없는 site도 있어 INNER macro가 모든 TRAIN site를 대표하지 않는다. 참조 그룹은 선택 pool 안에서의 직접 제약을 만족하며, 새 독립 물리 녹음이 늘어난 것은 아니다. [상세 원천 감사](R6V2_SOURCE_SPLIT_AUDIT.md).

기존 선택 pool **{lineage['original_clips']}clips/{lineage['original_groups']}groups**에서 현재 허용 pool **{lineage['admitted_clips']}/{lineage['admitted_groups']}**가 된 이유는 정확히 옛 OUTER **{lineage['excluded_old_outer_clips']}/{lineage['excluded_old_outer_groups']}**를 제외했기 때문이다. 새로운 clip 손실이나 추가 수집으로 생긴 숫자가 아니다. [독립 TRAIN/INNER 통합 감사](R6_GUARDRAILS_INDEPENDENT_AUDIT.md)는 모든 사용 reference ID가 허용된 역할에 속하고 TRAIN의 배경 순서·no-source·hard-negative 위치가 두 law에서 같음을 확인했다.

## 3. 물리 생성 법칙과 관측 정답

R6는 목표 share를 먼저 뽑아 alpha를 역산하지 않는다. 변형된 전체 참조 궤적의 **창 crop/mask 전** 완전 분 broadband peak를 `Rpeak`, 배경 관측 분 broadband 에너지 중앙값을 `Bmedian`으로 두고 `alpha=10^(ΔL/10)·Bmedian/Rpeak`를 사용한다. ΔL은 **Uniform[{gen['snr_db_uniform'][0]:g},{gen['snr_db_uniform'][1]:g}]dB**인 사전 감도 범위이며 공식 관측 SNR의 추정은 아니다. CPA가 창 밖이면 tail을 다시 증폭하지 않는다.

TRAIN/INNER: minute-first와 실제 1–3분 short crop. locked generator: edge-straddling과 native 2s-first-no-warp. 배치는 관측 mask가 아니라 기하학적 겹침으로 정한다. source가 미관측 셀에만 있으면 관측 q/share/presence=0이며 이를 삭제하거나 seed를 다시 고르지 않는다. caller의 **no-source 20%**와 전체 관측 y=0 비율은 다르다. 관측 추가 에너지 0일 때 observed/global SNR은 null이다. [사전 support 수정](R6_PREFIT_SUPPORT_CORRECTION.md).

정답 q/y는 unrounded known B+A에서 계산하고 유지했다. 모델의 rounded-energy w로 같은 q를 pooling하면 근사 차이가 생긴다. ±.05dB 양자화의 일반 bound는 약 .0057564이며 float32 오차는 추가다. 저장 세트의 원성분 정답 복원 최대 오차 **{worst_identity:.8g}**, rounded-w gap 최대 **{worst_gap:.8g}**다. 이 gap을 없애려고 정답을 반올림된 입력 기준으로 바꾸지 않았다.

## 4. 입력·score·물리 특징과 receptive field

relative8의 순서는 band median 상대/10, q20 상대/15, 인접 관측 ΔdB/10, 저역 상승과 중고역 상승의 연속 동조, band median−창 median profile/20, 관측 mask, q05 상대/15, `sqrt(w)`다. db만 입력하며 query true q/y·site/recorder·recipe를 입력하지 않는다. common dB gain 불변성은 양자화된 값 위에서 성립한다. 임의 gain을 반올림 전에 더하면 경계를 넘을 수 있고, profile·w·sqrt(w)는 band별 offset 불변이 아니다.

CNN의 고정 score는 `0.5·predicted_share + 0.5·temporal_mask_contrast`다. contrast는 관측된 band indices10..22(약316–5012Hz)에서 각 band의 시간 최소 예측 q를 빼고 양수 차이의 전체 최댓값을 쓴다. 단일 관측 분은 그 band contrast=0이다. 별도 자유 presence classifier는 없고 BCE는 이 derived score의 logit에 작용한다. 수치상 logit의 endpoint clamp가 있다. AUC 향상은 설계만으로 보장되지 않는다. tree는 **0.5·share+0.5·TRAIN-fitted classifier probability**를 고정 사용하며 share/classifier 개별 AUC도 보고한다.

이 score는 [0,1]의 share·contrast에 대한 단조 결합이지만 **보정된 presence 확률은 아니다.** contrast≤1이므로 `score≤0.5+0.5·predicted_share`다. 낮은 share를 정확하게 예측하는 양성에서는 score를 1로 만들 수 없어 presence BCE를 0으로 낮출 수 없다. 그때의 BCE 하한은 `−log(0.5+0.5·predicted_share)`이며, share→0이면 log2에 접근한다. 따라서 **{neural['presence_loss_weight']}·BCE를 줄이려는 gradient와 정확한 share 회귀가 상충할 수 있다.** 이는 사전 고정한 loss 실험의 수식적 한계이고 무부작용·성능 향상을 보장하지 않는다. 결과를 본 뒤 이 loss나 score를 바꾸지 않는다.

global context는 매 분 hidden의 mid(10..19), high(20..24), safe(15..22) 평균을 연결해 projection×sigmoid gate를 만들고 모든 주파수 위치에 더한다. 입력 음원의 분리 성분이나 정답 oracle이 아니다. **no-added-context arm을 엄밀한 국소 모델이라고 부를 수 없다.** 7개 3×3 convolution의 명목 경로 receptive field는 15×15이며, GroupNorm은 T/F 전체 통계에 의존한다. 상대 전처리의 창 median/q05/profile와 정규화 w도 이미 전역 정보다. 따라서 이 ablation은 명시적으로 추가한 hidden frequency context의 효과를 비교한다.

또한 context ON은 **64,449 parameters**, OFF는 **58,241 parameters**로 6,208개 차이가 있다. 입력·생성 law를 같게 두더라도 추가 context와 모델 용량이 함께 달라지므로 순수한 구조 한 요소의 인과 효과만 분리하는 비교는 아니다. 실제 arm별 크기도 아래 학습 표에 기록한다.

tree 물리 특징은 9값+9defined mask+9complete-minute count다. low0..8(≤200Hz), mid10..19(316–2512Hz), high20..24(3162–7943Hz), safe15..22(1000–5012Hz)의 선형 파워 합을 dB로 바꿔 max−median bump4개, low와 나머지3그룹의 시간 Pearson3개, 전체25band broadband dB 표준편차1개, paired high−low dB median tilt1개다. 각 그룹 전대역이 완전한 분만 사용한다. 정의되지 않은 값은 0과 mask/count로 표현한다. fitted state가 없는 db-only 특징이며 고정 0.1dB 입력을 내부에서 사용한다. [숫자 fixture](R6_PHYSICS_FEATURES_V2_FIXTURE.json).

## 5. TRAIN hard-negative의 의미와 규모

실제 TRAIN 원천 {hardaudit['train_windows_examined']:,}창 중 **{hardaudit['candidate_count']:,}개 후보**를 찾았다. LF bump≥TRAIN pooled q90(**{hardaudit['lf_train_q90_threshold_db']:.6f}dB**)이면서 LF bump>max(safe bump,high bump)+3dB다. 후보 선택은 raw SA QC dB로 하며 추가 반올림을 하지 않는다. 모델 입력 round.1 정책과 구분한다. 최소 지역 관측 분3, complete-region power만 사용했다. Test 통계나 과거 EDA 후보 숫자는 쓰지 않았다.

{table(['Site','후보 원천 창'],hardrows)}

이는 **추가한 A가 0인 대조**이지 자연 선박 부재 라벨이 아니다. 후보 풀과 실제 합성 주입 예제 수를 구분한다. 사전 계획은 전체 TRAIN의 10%, 명시적 no-source 중 절반이며, 평가 패널에서는 이 선정 정책으로 sampling하지 않는다. 실제 TRAIN **{train_data['hard_negative_count']:,}개 draw**는 **{train_data['hard_negative_unique_backgrounds']:,}개의 고유 창**을 사용했다. 전체 TRAIN **{train_audit['rows']:,}개**도 고유 배경은 **{train_audit['unique_primary_windows']:,}개**다. 작은 site의 후보를 cycling하므로 독립 원천 {train_data['hard_negative_count']:,}개가 늘어난 것이 아니다. pooled threshold 때문에 원래 후보 site 구성이 균일하지 않고, 주입 시 site 동수 할당이 후보 희소 site의 반복을 늘린다. TRAIN-only 선정도 분포 변화/shortcut을 만들 수 있으므로 일반화 성공을 보장하지 않는다.

[후보 manifest](../{HARDNEG}).

![고정 순서로 선택한 실제 TRAIN 예시](r6_guardrails_training_examples.png)

위 그림은 저장 순서상 **첫 hard-negative, 첫 R6 양성, 첫 R4 양성**을 선택했다. 예측·오차·그림의 모양을 보고 고른 사례가 아니다. rounded 관측, 알려진 인공 추가 성분 q, sqrt(w), q05 상대값, LF/MF/HF 관측 궤적을 함께 보여준다. q는 모든 자연 선박 에너지의 정답이 아니며, **A=0은 자연 선박 부재를 뜻하지 않는다. q05도 확인된 배경 B의 정답이 아니다.** 관측 결측은 회색이고 셀을 보간하지 않았다. [선택·SHA 근거](r6_guardrails_training_examples.selection.json) · [SVG](r6_guardrails_training_examples.svg) · [TRAIN-only 생성 코드](../ops/plot_r6_train_evidence.py).

## 6. 저장 데이터의 target scale

R4 대조를 포함해 저장한 역할 예제 합계는 **{total_examples:,}개**다. 이를 독립 녹음 수로 세지 않는다. TRAIN은 site 동수 quota의 기본 draw에 hard-negative 정책을 더한 샘플링이며 작은 site의 원천은 반복될 수 있다. 모든 arm의 최종 locked 평가 입력/정답은 같은 패널이다.

{table(['저장 세트','n','mean share','관측 y=0','no-source 실측','상수0 RMSE','상수.1 RMSE','단일 upper-band AUC'],scales)}

{table(['저장 세트','전체 draw','고유 배경','hard-negative draw','고유 hard-negative 배경','source있지만 관측추가0'],sampling_rows)}

`—`는 해당 필드가 최종 JSON에 없거나 단일 class여서 AUC가 정의되지 않음을 뜻한다. 단일 upper-band 비율 진단은 모델 score와 다르며, 여기서는 indices18..24의 energy fraction이다. 이를 목표치로 낮추도록 generator를 조정하지 않았다.

![Target scale과 상수 대조](r6_guardrails_v2_figures/03_target_scale_controls.png)

**각 arm의 own-law INNER는 R4/R6 사이 target population이 다르다.** 두 law의 INNER는 배경 {inner_audit['rows']:,}개 순서를 공유하지만 명시적 no-source 각각 {inner_audit['no_source_count_r4']:,}개 위치는 독립 생성되어 **{inner_audit['no_source_position_difference_count']:,}행**에서 위치가 다르다. TRAIN은 두 law에서 no-source 위치가 일치하지만 INNER는 공통 target의 paired 비교가 아니다. 더 낮은 INNER RMSE를 곧 개선으로 읽지 않는다. 선별에 사용한 INNER 자체도 early-stop bias가 있으므로 완전 독립 OOF가 아니다. 상수 대조와 아래 공통 locked 패널을 함께 본다.

## 7. 실제 학습 설정과 INNER 결과

CNN: width{neural['width']}, blocks{neural['blocks']}, dropout{neural['dropout']}, Adam LR{neural['learning_rate']}, weight decay{neural['weight_decay']}, 최대 **{neural['epochs']}epochs/ES{neural['early_stopping_patience']}**, batch{neural['batch_size']}, CPU threads{neural['threads']}, gradient clip{neural['gradient_clip']}. loss=share MSE+{neural['cell_loss_weight']}·energy-weighted cell MSE+{neural['presence_loss_weight']}·derived presence BCE. checkpoint는 own-law INNER site-macro RMSE로 선택한다. window RMSE도 함께 보고한다.

tree: 최대{tree['maximum_trees']}trees/ES{tree['early_stopping_patience']}, LR{tree['learning_rate']}, leaves{tree['num_leaves']}, 최소leaf{tree['minimum_leaf_rows']}rows. 회귀는 INNER site-macro, classifier는 INNER binary logloss로 stopping한다. CNN/tree는 엔진 충돌을 피하려 별도 프로세스에서 실행한다. 모델 fit 시간 합은 **{time_minutes:.2f}분**이며 전처리·최종 평가 시간을 제외한다.

{table(['Arm','크기','best / 실행','INNER RMSE','INNER macro','고정 score AUC','share AUC','classifier AUC','분'],inner_rows)}

![실제 학습곡선](r6_guardrails_v2_figures/01_cpu_curves.png)

TRAIN online RMSE는 epoch 중 변하는 weight의 minibatch 결과이고 INNER는 epoch 끝 eval이므로 직접 같은 평가 프로토콜의 train/val 격차가 아니다. R4 law 대조도 이 작은 CPU 구조와 새 split/QC/quantization을 사용하며 예전1.488M CNN의 완전 재현은 아니다. legacy8은 TRAIN q를 사용해 만든 배경 prior를 포함한 대조군이고 relative8은 이를 쓰지 않는다. 채널 비교는 여러 입력 정의를 함께 바꿔 각 요소의 독립 기여를 분해하지 않는다.

## 8. 전체 checkpoint 동결 후 공통 locked 패널

{chr(10).join(locked_tables)}

![모든 arm·상수 대조](r6_guardrails_v2_figures/02_locked_panels.png)

세 패널은 모두 우리 합성 정답이다. cross-generator는 generator와 참조 원천이 함께 달라져 연산자 하나의 순수 효과가 아니다. cross-sensor와 joint 패널은 배경 draw를 공유하므로 서로 독립 표본이 아니다. 선택 pool 경계 밖의 공식 평가를 대신하지 않는다.

{table(['Locked 패널','단일 upper-band energy AUC','rounded-w target gap 최대'],diagrows)}

## 9. 사전 대조의 동일 패널 RMSE 변화

Δ는 오른쪽 개편−왼쪽 대조다. 음수는 해당 합성 패널에서의 RMSE 감소이며 아래 전체 비교를 보고한다. 이를 보고 arm을 새로 선별하거나 앙상블을 맞추지 않는다.

{table(['패널','대조','Δ RMSE'],contrasts)}

단일 seed·제한된 source pool·8192 TRAIN의 작은 CPU 실험이므로, 이 결과만으로 실제 선박 분리의 SOTA·공식 hidden generator 재현·LB 개선 폭·이전 실패의 특정 인과를 확정하지 않는다. Group split·24h purge는 직접 공유를 줄이는 제약이며 같은 합성 법칙의 shortcut까지 제거하는 보장은 아니다. 무효/악화 결과도 삭제하지 않는다.

## 10. 재현 근거와 완료 범위

- 완료한 범위: 사전 정의한 source/QC·hard-negative 후보·8채널/27특징·5CNN/2tree CPU fit·공통3패널·상수4개·저장곡선과 모든 metric이다.
- 미완료 범위: production5fold, 전체 배경 온라인 합성 학습, 대형 backbone 공식 비교, 실제 Test 성능 검증이다.
- 이 보고서 생성에서는 모델 fitting/선택, 원시 배열 로드, Test/old OUTER 접근, GPU/SSH, 공개 게시/제출을 하지 않았다.
- R4/R5 frozen source를 수정하지 않는다. 새 guardrails 결과를 공식 Benchmark의 target moment나 Test 지역 구성에 맞추는 사후 조정은 하지 않는다.

첫 fit 진입은 모델 초기화 전에 중단됐다. pandas에서 온 site 문자열이 NPZ의 object dtype으로 저장되어 `allow_pickle=False`가 안전하게 거부한 직렬화 오류였다. 실제 모델 fit이나 결과를 보고 고른 재시도가 아니며, 첫 데이터와 실패 run은 보존했다. site를 명시적 Unicode로 저장하고 prepare에서 모든 NPZ 필드를 no-pickle로 재로드하는 검사를 추가한 뒤, **동일 seed/config/source**로 `r6_cpu_guardrails_v3_schemafix`에 전체 재생성했다. partial 데이터·모델을 재사용하지 않았다. 이 보고서는 합성 배열 경로를 하드코딩하거나 그 배열을 열지 않고 위 최종 dataset 감사와 결과 SHA binding을 검증한다.

운영 한계: locked 패널의 target 분포·에너지 등 기술 QA 요약은 데이터 준비 때 저장됐다. 이를 후보 선정·모델 fit·튜닝에 사용하지 않았으며, 모델의 locked 예측은 전 arm checkpoint 동결 뒤 실행했다. 완전한 정보 비노출 금고와 같은 절차였다고 과장하지 않는다. 또한 현재 재평가 방지용 출력 존재 검사는 inference 호출 **뒤**에 있으므로 재호출 시 계산 자체까지 사전에 막는 guard는 아니다. **이번 실행에서는 중복 locked 평가를 하지 않았다.** 이 한계를 보고하며 동결된 엔진을 결과 확인 뒤 변경하지 않는다.

[전체 결과](R6V2_CPU_ABLATION_RESULTS.json) · [Dataset audit](R6V2_DATASET_AUDIT.json) · [Source audit](R6V2_SOURCE_SPLIT_AUDIT.json) · [독립 통합 감사](R6_GUARDRAILS_INDEPENDENT_AUDIT.json) · [설정](../{CONFIG})

{provenance}
'''


def self_test():
    """In-memory schema/format fixture only; no real result file is opened."""
    arms = [{'name':n} for n in list(SHORT)[:7]]
    config = {'CNN_arms':arms[:5], 'LightGBM_arms':arms[5:], 'samples':{k:4 for k in ROLES}}
    metric = {'n':4,'rmse':.2,'site_macro_rmse':.21,'mae':.15,'bias':-.01,'presence_auc':None}
    result = {'status':'completed', 'summaries':[{'status':'completed','arm':a,'best_inner_metrics':metric.copy()} for a in arms],
              'heldout_results':{k:{n:metric.copy() for n in [a['name'] for a in arms]+list(CONSTANTS)} for k in ROLES},
              'heldout_data_diagnostics':{k:{} for k in ROLES}}
    assert len(validate_result_structure(config,result)) == 7
    assert fmt(None) == '—' and fmt(-.01,signed=True) == '-0.010000'
    result['heldout_results']['cross_sensor'].pop(arms[0]['name'])
    try:
        validate_result_structure(config,result)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing arm must fail')
    print('Template self-test passed; no real results/data/model files opened.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--generate',action='store_true',help='Use only after parent authorizes final generation')
    choice.add_argument('--self-test',action='store_true',help='Synthetic in-memory schema fixture only')
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    output = ROOT/OUTPUT
    if output.exists() or (ROOT/FIGURES).exists():
        raise FileExistsError('Fresh report/figures required; existing results are preserved')
    config, source, data, result, hardneg, independent, histories, receipts = read_completed_inputs(ROOT)
    figure_paths = plots(ROOT,config,data,result,histories)
    text = render(config,source,data,result,hardneg,independent,histories,receipts)
    with output.open('x') as stream:
        stream.write(text)
    receipt = {'format':'r6_guardrails_report_render_v1','scope':'completed JSON/history rendering only; no fit or raw data arrays',
               'input_sha256':receipts,'report':OUTPUT,'report_sha256':sha(output),
               'figure_sha256':{n:sha(ROOT/n) for n in figure_paths}}
    with output.with_suffix('.receipt.json').open('x') as stream:
        stream.write(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'rendered','report':OUTPUT,'figures':figure_paths},ensure_ascii=False))


if __name__ == '__main__':
    main()
