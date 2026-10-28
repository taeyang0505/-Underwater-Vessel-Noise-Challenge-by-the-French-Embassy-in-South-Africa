"""Render the fixed five-arm100-epoch TRAIN/INNER study, without fitting.

Schema is pinned to pipeline.fixed100_r6's completed result manifest.
Only --generate opens measured summaries/histories. Former ES comparison
uses each arm's INNER-only summary/epochs, never old locked-panel results.
An in-history minimum is not a global minimum or independent OOF estimate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = 'models/r6_fixed100_v1'
OLD_RUN = 'models/r6_cpu_guardrails_v2'
CONFIG = 'configs/r6_fixed100_v1.json'
RESULT = 'reports/R6_FIXED100_RESULTS.json'
DATA_AUDIT = 'reports/R6V2_DATASET_AUDIT.json'
OLD_CONFIG = 'configs/r6_cpu_guardrails_v2.json'
OUTPUT = 'reports/R6_FIXED100_TRAINING_REPORT.md'
FIGURES = 'reports/r6_fixed100_figures'
EXPECTED = (
    'r4law_legacy8_global', 'r4law_relative8_global',
    'r6law_legacy8_global', 'r6law_relative8_global', 'r6law_relative8_local',
)
SHORT = {
    'r4law_legacy8_global':'R4 / legacy8 / context',
    'r4law_relative8_global':'R4 / relative8 / context',
    'r6law_legacy8_global':'R6 / legacy8 / context',
    'r6law_relative8_global':'R6 / relative8 / context',
    'r6law_relative8_local':'R6 / relative8 / no added context',
}
PATIENT = 7
MIN_DELTA = 1e-5
ARTIFACTS = {
    'best_window.pt', 'best_macro.pt', 'last.pt', 'final_training_state.pt',
    'window_inner_predictions.npz', 'macro_inner_predictions.npz', 'last_inner_predictions.npz',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def fmt(x, digits=6, signed=False):
    if x is None:
        return '—'
    x = float(x)
    if not np.isfinite(x):
        raise ValueError('Nonfinite report value')
    return format(x, ('+' if signed else '')+'.'+str(digits)+'f')


def table(header, rows):
    return '\n'.join(['| '+' | '.join(header)+' |',
                      '| '+' | '.join(['---']*len(header))+' |']
                     +['| '+' | '.join(map(str,r))+' |' for r in rows])


def history_metrics(rows, *, epochs=100):
    """Exact first minima of stored values, not min-delta-selected snapshots."""
    if (len(rows) != epochs or [r['epoch'] for r in rows] != list(range(1,epochs+1))):
        raise ValueError('A complete ordered epoch history is required')
    keys = ('train_rmse','inner_rmse','inner_site_macro_rmse','learning_rate')
    for row in rows:
        if any(not np.isfinite(float(row[k])) for k in keys):
            raise ValueError('Nonfinite epoch history')
        if min(row[k] for k in keys) < 0:
            raise ValueError('RMSE/learning-rate history cannot be negative')
    window_index = int(np.argmin([r['inner_rmse'] for r in rows]))
    macro_index = int(np.argmin([r['inner_site_macro_rmse'] for r in rows]))
    return {'window':dict(rows[window_index]), 'macro':dict(rows[macro_index]),
            'last':dict(rows[-1]),
            'minimum_semantics':'first exact minimum among recorded epoch metrics; no global-optimum claim'}


def counterfactual_es7(rows, *, monitor='inner_site_macro_rmse'):
    """Replay ES7/min_delta1e-5 on one prespecified stored metric.

    The first stop is informative only for this trajectory. Continuing past
    it is an actual no-ES run, not a simulated alternative optimizer. The
    scheduler remains the actual MACRO-driven scheduler for BOTH monitors;
    it is not recomputed here. Prefix equality to the old run is separately
    reported rather than assumed. Window ES is analysis only, not a new run.
    """
    if monitor not in {'inner_site_macro_rmse','inner_rmse'}:
        raise ValueError('Only the predeclared macro/window ES monitors are supported')
    best = float('inf'); stale = 0; best_epoch = 0
    for row in rows:
        score = float(row[monitor])
        if not np.isfinite(score):
            raise ValueError('Nonfinite ES score')
        if score < best - MIN_DELTA:
            best = score; stale = 0; best_epoch = int(row['epoch'])
        else:
            stale += 1
        if stale >= PATIENT:
            return {'stop_reached':True,'stop_epoch':int(row['epoch']),
                    'best_epoch':best_epoch,'best_site_macro_rmse':best,
                    'patience':PATIENT,'minimum_improvement':MIN_DELTA,
                    'monitor':monitor}
    return {'stop_reached':False,'stop_epoch':None,'observed_through_epoch':int(rows[-1]['epoch']),
            'best_epoch':best_epoch,'best_site_macro_rmse':best,
            'patience':PATIENT,'minimum_improvement':MIN_DELTA,
            'monitor':monitor}


def stopping_window_analysis(rows, counterfactual):
    """First exact window minimum available by the hypothetical stop."""
    end=counterfactual['stop_epoch'] if counterfactual['stop_reached'] else len(rows)
    prefix=rows[:end]
    best_prefix=prefix[int(np.argmin([r['inner_rmse'] for r in prefix]))]
    best_full=rows[int(np.argmin([r['inner_rmse'] for r in rows]))]
    return {'monitor':counterfactual['monitor'],'stop_reached':counterfactual['stop_reached'],
            'through_epoch':end,'best_window_epoch_through_stop':best_prefix['epoch'],
            'best_window_rmse_through_stop':best_prefix['inner_rmse'],
            'full_history_best_window_epoch':best_full['epoch'],
            'full_history_best_window_rmse':best_full['inner_rmse'],
            'recorded_window_improvement_beyond_stop_prefix':best_prefix['inner_rmse']-best_full['inner_rmse'],
            'scope':'same saved trajectory and actual macro-driven scheduler; not a retrained counterfactual'}


def compare_prefix(new, old):
    if not old or len(new) < len(old):
        raise ValueError('Former run prefix is empty or longer than new run')
    if [r['epoch'] for r in old] != list(range(1,len(old)+1)):
        raise ValueError('Former epoch history is incomplete')
    values = {}
    for key in ['train_rmse','inner_rmse','inner_site_macro_rmse','learning_rate']:
        a = np.asarray([r[key] for r in new[:len(old)]],float)
        b = np.asarray([r[key] for r in old],float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite former/new prefix')
        values[key] = float(np.max(abs(a-b)))
    return {'epochs_compared':len(old),'max_abs_by_metric':values,
            'all_recorded_metrics_exactly_equal':all(v==0 for v in values.values())}


def load_inputs(root):
    """SHA/checkpoint/history verification only; never deserialize a model."""
    run = root/RUN
    if (run/'FAILED.json').exists():
        raise ValueError('A failed/partial run cannot be reported as complete')
    gate_path=run/'all_models_frozen.json'
    gate=read_json(gate_path)
    if (gate.get('status') != 'frozen' or gate.get('heldout_evaluated') is not False
            or set(gate['model_summary_sha256']) != set(EXPECTED)):
        raise ValueError('All five completed100-epoch model summaries must be frozen')
    final = read_json(root/RESULT)
    config = read_json(root/CONFIG)
    if final.get('status') != 'completed' or final.get('heldout_evaluated') is not False:
        raise ValueError('Completed100-epoch INNER-only final manifest required')
    config_sha=sha(root/CONFIG)
    if final['config_sha256'] != config_sha or gate['config_sha256'] != config_sha:
        raise ValueError('Fixed100 config SHA binding mismatch')
    if final.get('former_ES_run') != OLD_RUN or sha(run/'results.json') != sha(root/RESULT):
        raise ValueError('Final/run results or former-run binding mismatch')
    if ([a['name'] for a in config['CNN_arms']] != list(EXPECTED)
            or [s['arm'] for s in final['summaries']] != config['CNN_arms']):
        raise ValueError('All five unchanged arms are required in fixed order')
    if (config['neural']['epochs'] != 100 or config['neural']['optimizer'] != 'adam'
            or config['neural']['early_stopping_enabled'] is not False):
        raise ValueError('100epoch/Adam/no-ES/shadow-ES7 contract changed')
    expected_scheduler={'name':'ReduceLROnPlateau','metric':'INNER_site_macro_rmse',
                        'factor':.5,'patience':2,'min_lr':1e-6,'changed_from_original':False}
    if config['scheduler'] != expected_scheduler:
        raise ValueError('Frozen macro scheduler policy changed')
    if (config.get('data_regeneration_or_split_change') is not False
            or config.get('global_minimum_guaranteed') is not False):
        raise ValueError('Data scope/global-optimum declaration changed')
    cp_policy=config['checkpoint_policy']
    if (cp_policy['keep_every_epoch'] is not True or cp_policy['last_epoch'] != 100
            or cp_policy['minimum_improvement_for_checkpoints'] != 0.):
        raise ValueError('Every epoch/strict first-minimum checkpoint policy changed')
    # The data audit embeds old locked descriptive fields. Hash its bytes
    # without parsing that JSON; only TRAIN/INNER file bytes are hashed below.
    dataset_sha=sha(root/DATA_AUDIT)
    if config['data_manifest_sha256'] != dataset_sha or final['data_manifest_sha256'] != dataset_sha:
        raise ValueError('Immutable TRAIN/INNER dataset binding mismatch')
    old_config=read_json(root/OLD_CONFIG)
    if config['CNN_arms'] != old_config['CNN_arms'] or config['seed'] != old_config['seed']:
        raise ValueError('Former/new arm or seed identity changed')
    old_freeze_path=root/OLD_RUN/'freeze.json'
    old_freeze=read_json(old_freeze_path)
    if (old_freeze['config_sha256'] != sha(root/OLD_CONFIG)
            or old_config['neural']['threads'] != 4 or config['neural']['threads'] != 2):
        raise ValueError('Original recipe or declared threads4→2 comparison changed')
    old_neural=old_config['neural']
    if old_neural['minimum_improvement'] != MIN_DELTA or old_neural['early_stopping_patience'] != PATIENT:
        raise ValueError('Original macro ES7/min-delta policy changed')
    for key,value in old_neural.items():
        # ES stopping fields are deliberately removed from the final no-ES
        # neural config. Their analysis policy is pinned above and in shadow_ES7.
        if key not in {'threads','minimum_improvement','early_stopping_patience'} and config['neural'].get(key) != value:
            raise ValueError('An undeclared neural comparison setting changed: '+key)
    receipts = {CONFIG:config_sha,RESULT:sha(root/RESULT),DATA_AUDIT:dataset_sha,
                OLD_CONFIG:sha(root/OLD_CONFIG),
                str(gate_path.relative_to(root)):sha(gate_path),
                'ops/build_r6_fixed100_report.py':sha(root/'ops/build_r6_fixed100_report.py')}
    freeze_path=run/'freeze.json';freeze=read_json(freeze_path)
    if freeze['config_sha256'] != config_sha or freeze['data_manifest_sha256'] != dataset_sha:
        raise ValueError('Startup freeze binding mismatch')
    if set(freeze['new_code_sha256']) != {'pipeline/fixed100_r6.py','ops/run_r6_fixed100.py'}:
        raise ValueError('Fixed100 executable source closure changed')
    for name,digest in {**old_freeze['code_sha256'],**freeze['new_code_sha256']}.items():
        if sha(root/name) != digest:
            raise ValueError('Frozen source changed: '+name)
        receipts[name]=digest
    original_core=config.get('original_frozen_core17_sha256',{})
    if original_core and len(original_core)!=17:
        raise ValueError('Declared original core closure must contain17 files')
    for name,digest in original_core.items():
        if sha(root/name)!=digest:
            raise ValueError('Original frozen core17 changed: '+name)
        receipts[name]=digest
    receipts[str(freeze_path.relative_to(root))]=sha(freeze_path)
    receipts[str(old_freeze_path.relative_to(root))]=sha(old_freeze_path)
    expected_inputs={f'data/synthetic/r6_cpu_guardrails_v3_schemafix/{role}_{law}.npz'
                     for role in ['train','inner'] for law in ['r4','r6']}
    expected_inputs|={'data/versions/source_split_r6_guardrails_v2/manifest.json',
                     'data/versions/hard_negatives_r6_v2/manifest.json'}
    if set(config['used_input_sha256']) != expected_inputs:
        raise ValueError('Exactly four TRAIN/INNER arrays and two source manifests may be hashed')
    for name,digest in config['used_input_sha256'].items():
        if sha(root/name) != digest:
            raise ValueError('Original TRAIN/INNER opaque file bytes changed: '+name)
        receipts[name]=digest
    old_gate_path = root/OLD_RUN/'all_checkpoints_frozen_before_heldout.json'
    old_gate = read_json(old_gate_path)
    if old_gate.get('status') != 'frozen' or old_gate.get('all_requested_arms_completed') is not True:
        raise ValueError('Former run freeze receipt is invalid')
    receipts[str(old_gate_path.relative_to(root))] = sha(old_gate_path)
    studies = {}
    for summary in final['summaries']:
        name = summary['arm']['name']; folder = run/name
        path = folder/'summary.json'
        if (sha(path) != gate['model_summary_sha256'][name] or read_json(path) != summary
                or summary.get('status') != 'completed' or summary['epochs_run'] != 100
                or summary.get('early_stopping_enabled') is not False
                or summary.get('heldout_evaluated') is not False
                or summary['stop_reason'] != 'completed_100_epochs'
                or summary['threads'] != config['neural']['threads']):
            raise ValueError('Incomplete or changed fixed100 arm summary: '+name)
        history_path = folder/'epochs.jsonl'
        rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        analysis = history_metrics(rows)
        cf=counterfactual_es7(rows)
        if (summary['shadow_ES7']['stop_epoch'] != cf['stop_epoch']
                or summary['shadow_ES7']['best_macro_epoch'] != cf['best_epoch']
                or summary['shadow_ES7']['best_macro_rmse'] != cf['best_site_macro_rmse']
                or summary['shadow_ES7']['stopping_used_for_training'] is not False
                or summary['shadow_ES7']['minimum_improvement'] != MIN_DELTA):
            raise ValueError('Saved shadow ES differs from independent replay: '+name)
        training_n=old_config['samples']['train']
        steps_per_epoch=(training_n+config['neural']['batch_size']-1)//config['neural']['batch_size']
        if (summary['optimizer_steps'] != 100*steps_per_epoch
                or any(row['optimizer_steps'] != row['epoch']*steps_per_epoch for row in rows)):
            raise ValueError('Optimizer-step coverage is not100 full epochs: '+name)
        if (rows[-1]['shadow_ES7_stop_epoch'] != cf['stop_epoch']
                or rows[-1]['shadow_ES7_best_macro_epoch'] != cf['best_epoch']):
            raise ValueError('Epoch shadow ES record mismatch: '+name)
        checkpoint_map=summary['epoch_checkpoint_sha256']
        if set(checkpoint_map) != {f'epoch_{e:03d}.pt' for e in range(1,101)}:
            raise ValueError('Every one of100 epoch checkpoints is required: '+name)
        if set(summary['artifact_sha256']) != ARTIFACTS:
            raise ValueError('Seven fixed snapshot/state/prediction artifacts required: '+name)
        for label,key,file in [('window','best_inner_window','best_window.pt'),
                               ('macro','best_inner_macro','best_macro.pt'),
                               ('last','final_epoch','last.pt')]:
            entry=summary[key];record=analysis[label];epoch=record['epoch']
            if entry['epoch'] != epoch or entry['checkpoint'] != file:
                raise ValueError('Summary/independent first-minimum epoch mismatch: '+name+'/'+label)
            for metric,rowkey in [('rmse','inner_rmse'),('site_macro_rmse','inner_site_macro_rmse')]:
                if abs(entry['metrics'][metric]-record[rowkey]) > 1e-9:
                    raise ValueError('Checkpoint reload/history metrics differ: '+name+'/'+label)
            if summary['artifact_sha256'][file] != checkpoint_map[f'epoch_{epoch:03d}.pt']:
                raise ValueError('Snapshot alias is not the corresponding saved epoch: '+name+'/'+label)
        for filename,digest in checkpoint_map.items():
            cp=folder/'checkpoints'/filename
            if cp.is_symlink() or sha(cp) != digest:
                raise ValueError('Saved epoch checkpoint changed: '+name+'/'+filename)
            receipts[str(cp.relative_to(root))]=digest
        for filename,digest in summary['artifact_sha256'].items():
            artifact=folder/filename
            if artifact.is_symlink() or sha(artifact) != digest:
                raise ValueError('Saved snapshot/state/predictions changed: '+name+'/'+filename)
            receipts[str(artifact.relative_to(root))]=digest
        requires_prior=summary['arm']['channels']=='legacy8'
        if requires_prior != (summary['prior_sha256'] is not None):
            raise ValueError('Legacy/relative TRAIN-prior declaration changed: '+name)
        if summary['prior_sha256'] is not None:
            prior=folder/'legacy_train_prior.json'
            if sha(prior) != summary['prior_sha256']:
                raise ValueError('Legacy own-TRAIN prior changed: '+name)
            receipts[str(prior.relative_to(root))]=summary['prior_sha256']
        elif (folder/'legacy_train_prior.json').exists():
            raise ValueError('Relative arm has unexpected fitted prior')
        old_path = root/OLD_RUN/name/'summary.json'
        if sha(old_path) != old_gate['model_summaries'][name]:
            raise ValueError('Former ES summary changed: '+name)
        old_summary = read_json(old_path)
        old_history_path = root/OLD_RUN/name/'epochs.jsonl'
        old_rows = [json.loads(line) for line in old_history_path.read_text().splitlines() if line.strip()]
        if old_summary['epochs_run'] != len(old_rows):
            raise ValueError('Former ES history length mismatch')
        for p in [path,history_path,old_path,old_history_path]:
            receipts[str(p.relative_to(root))] = sha(p)
        if old_summary['arm'] != summary['arm'] or old_summary['model_config'] != summary['model_config']:
            raise ValueError('Former/new architecture identity differs: '+name)
        studies[name] = {'summary':summary,'history':rows,'analysis':analysis,
                         'counterfactual_es7':cf,
                         'counterfactual_window_es7':counterfactual_es7(rows,monitor='inner_rmse'),
                         'macro_stopping_window_analysis':stopping_window_analysis(rows,cf),
                         'window_stopping_window_analysis':stopping_window_analysis(rows,counterfactual_es7(rows,monitor='inner_rmse')),
                         'former_summary':old_summary,'former_history':old_rows,
                         'prefix':compare_prefix(rows,old_rows)}
    return config,final,studies,receipts


def draw_figures(root, studies):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.facecolor':'white'})
    out = root/FIGURES; out.mkdir(parents=True,exist_ok=False)
    fig, axes = plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ax,name in zip(axes.flat,EXPECTED):
        st=studies[name]; rows=st['history']; a=st['analysis']
        epoch=[r['epoch'] for r in rows]
        ax.plot(epoch,[r['train_rmse'] for r in rows],label='TRAIN online',color='#5d88ac',lw=1.2)
        ax.plot(epoch,[r['inner_rmse'] for r in rows],label='INNER window',color='#41856d',lw=1.2)
        ax.plot(epoch,[r['inner_site_macro_rmse'] for r in rows],label='INNER site macro',color='#c38a46',lw=1.2)
        ax.scatter([a['window']['epoch']],[a['window']['inner_rmse']],s=30,color='#246b4e',marker='*',zorder=4)
        ax.scatter([a['macro']['epoch']],[a['macro']['inner_site_macro_rmse']],s=30,color='#9f601f',marker='*',zorder=4)
        cf=st['counterfactual_es7']
        if cf['stop_reached']:
            ax.axvline(cf['stop_epoch'],color='.55',ls='--',lw=.8,label='Macro ES7 replay stop')
        window_cf=st['counterfactual_window_es7']
        if window_cf['stop_reached']:
            ax.axvline(window_cf['stop_epoch'],color='#8a72a3',ls=':',lw=.9,label='Window ES7 replay stop')
        ax.set(title=SHORT[name],xlabel='Epoch',ylabel='RMSE',xlim=(1,100))
        ax.grid(alpha=.15); ax.legend(fontsize=6.5)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0,.94,'100 actual epochs per arm\nNo early stopping\n\nStars: separate recorded minima\nDashed: macro ES7 replay stop\nDotted: window ES7 replay stop\nBoth use actual macro-scheduler trajectory.\n\nOwn-law INNER is reused for monitoring.\nNo locked/Test evaluation here.\nNot a global-minimum guarantee.',
                       va='top',transform=axes.flat[-1].transAxes,linespacing=1.5)
    fig.suptitle('Fixed100 CPU study: complete unsmoothed training histories')
    fig.savefig(out/'01_fixed100_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.7),layout='constrained')
    fig.get_layout_engine().set(rect=(0,0,1,.82))
    loc=np.arange(len(EXPECTED));width=.18
    for ax,key,title in [(axes[0],'inner_rmse','INNER window RMSE'),
                         (axes[1],'inner_site_macro_rmse','INNER site-macro RMSE')]:
        variants=[[],[],[],[]]
        for name in EXPECTED:
            st=studies[name]; old=st['former_summary']['best_inner_metrics']; a=st['analysis']
            variants[0].append(old['rmse' if key=='inner_rmse' else 'site_macro_rmse'])
            variants[1].append(a['last'][key]);variants[2].append(a['window'][key]);variants[3].append(a['macro'][key])
        for offset,v,label,color in zip([-1.5,-.5,.5,1.5],variants,
               ['Former ES7 selected','Epoch100','Recorded best-window epoch','Recorded best-macro epoch'],
               ['#999fa4','#bf986a','#5a927b','#5c80a7']):
            ax.barh(loc+offset*width,v,height=width,label=label,color=color)
        ax.set_yticks(loc,[SHORT[n] for n in EXPECTED],fontsize=7);ax.invert_yaxis()
        ax.set(xlabel=title,title=title);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,
               fontsize=8,frameon=False)
    fig.suptitle('Same own-law INNER: selected checkpoint and separate epoch minima',y=.995)
    fig.savefig(out/'02_former_es_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    for name in EXPECTED:
        rows=studies[name]['history']
        ax.step([r['epoch'] for r in rows],[r['learning_rate'] for r in rows],where='post',label=SHORT[name])
    ax.set(xlabel='Epoch',ylabel='Recorded learning rate',title='Frozen scheduler policy; actual learning-rate paths',xlim=(1,100),yscale='log')
    ax.grid(alpha=.15);ax.legend(fontsize=7)
    fig.savefig(out/'03_learning_rates.png',dpi=180);plt.close(fig)
    return sorted(out.glob('*.png'))


def render(config,final,studies,receipts):
    rows=[];former=[];counter=[];snapshot_rows=[];window_counter=[]
    for name in EXPECTED:
        st=studies[name];s=st['summary'];a=st['analysis'];old=st['former_summary'];cf=st['counterfactual_es7']
        rows.append([name,fmt(a['last']['inner_rmse']),fmt(a['last']['inner_site_macro_rmse']),
                     f"{a['window']['epoch']} / {fmt(a['window']['inner_rmse'])}",
                     f"{a['macro']['epoch']} / {fmt(a['macro']['inner_site_macro_rmse'])}",
                     fmt(a['macro']['inner_rmse']),fmt(s['elapsed_seconds']/60,2)])
        former.append([name,f"{old['best_epoch']} / {old['epochs_run']}",fmt(old['best_inner_metrics']['rmse']),
                       fmt(old['best_inner_metrics']['site_macro_rmse']),
                       fmt(a['macro']['inner_rmse']-old['best_inner_metrics']['rmse'],signed=True),
                       fmt(a['macro']['inner_site_macro_rmse']-old['best_inner_metrics']['site_macro_rmse'],signed=True)])
        counter.append([name,old['epochs_run'],cf['stop_epoch'] if cf['stop_reached'] else '100epoch까지 미도달',
                        cf['best_epoch'],fmt(cf['best_site_macro_rmse']),
                        st['prefix']['epochs_compared'],fmt(max(st['prefix']['max_abs_by_metric'].values()),10)])
        m=s['best_inner_window']['metrics']
        snapshot_rows.append([name,fmt(s['best_window_TRAIN_eval']['rmse']),fmt(m['rmse']),
                              fmt(m['site_macro_rmse']),fmt(m['mae']),fmt(m['bias'],signed=True),
                              fmt(m['presence_auc'])])
        macro_stop=st['macro_stopping_window_analysis'];window_stop=st['window_stopping_window_analysis']
        window_counter.append([name,macro_stop['through_epoch'],fmt(macro_stop['best_window_rmse_through_stop']),
                               window_stop['through_epoch'],fmt(window_stop['best_window_rmse_through_stop']),
                               fmt(a['window']['inner_rmse']),
                               fmt(macro_stop['recorded_window_improvement_beyond_stop_prefix']),
                               fmt(window_stop['recorded_window_improvement_beyond_stop_prefix'])])
    # Full500-checkpoint SHA ledger lives in the JSON receipt, not a500-row MD dump.
    compact_receipts={n:d for n,d in receipts.items()
                      if '/checkpoints/' not in n and
                      (Path(n).name in {'summary.json','all_models_frozen.json','freeze.json'}
                       or n.startswith(('configs/','ops/','pipeline/')) or n in {RESULT,DATA_AUDIT})}
    neural=config['neural'];schedule=config['scheduler'];execution=config['execution']
    return f'''# R6 고정100epoch CPU 학습 결과

생성 시각: {datetime.now(timezone.utc).isoformat()}. 기존의 동일한 5개 CNN을 **early stopping 없이 실제100epochs** 학습한 TRAIN/INNER 실험이다. tree 재학습·대형 GPU 모델 재개·locked 패널 재평가·공식 Test 결과가 아니다. 기존 데이터와 결과를 덮어쓰지 않았다.

## 1. 비교 범위와 고정한 정책

동일한 immutable TRAIN/INNER, source split, seed, 8채널, 물리 정답, Adam 및 loss를 유지한다. epoch 종료를100으로 고정하고 ES 중단을 비활성화했다. 단, 하드웨어 실행 정책은 **arm당 CPU2 threads(이전4), 최대 {execution['maximum_parallel_arms']}개 subprocess 병렬**로 달라졌다. 이는 학습 결과를 보고 바꾼 설정은 아니지만 floating-point reduction/궤적에 차이를 만들 수 있어 bitwise prefix 일치를 약속하지 않는다.

Adam LR={neural['learning_rate']}, weight decay={neural['weight_decay']}, batch={neural['batch_size']}, gradient clip={neural['gradient_clip']}이다. loss=share MSE+{neural['cell_loss_weight']}·energy-weighted cell MSE+{neural['presence_loss_weight']}·derived presence BCE를 유지한다. 학습률 scheduler는 기존 **ReduceLROnPlateau(macro, factor{schedule['factor']}, patience{schedule['patience']}, minLR{schedule['min_lr']})** 정책을 그대로 두며 아래 실제 LR 경로를 공개한다. 이 보고서는 저장 기록을 읽기만 하며 fitting·checkpoint 교체·후처리·앙상블 선택을 하지 않는다. 전체 병렬 실행 경과는 **{final['elapsed_seconds']/60:.2f}분**, 다섯 arm의 fit 경과 합은 **{sum(studies[n]['summary']['elapsed_seconds'] for n in EXPECTED)/60:.2f}분**으로 서로 다른 시간 단위다.

R4 law와 R6 law의 INNER는 서로 다른 target population이다. 따라서 각 arm의 **자기 law 안에서** 이전 ES run과 비교한다. INNER는 이미 early stopping/epoch monitoring에 반복 사용되어 독립 OOF가 아니다. 5개 중 가장 낮은 값을 보고 공식 성능 순위나 대회 일반화를 주장하지 않는다.

## 2. 전체100epoch 곡선과 별도 최선 epoch

window RMSE 최소와 site-macro RMSE 최소를 **서로 독립적으로** 저장 이력에서 찾았다. 동일값이면 처음 관측한 epoch를 쓴다. 이는 기록한100개 epoch 중 최소이며, 최적화 전체의 global minimum 또는100epoch 이후에도 개선이 없다는 증거가 아니다. 실제100번째 epoch도 함께 보고해 사후 최소값만으로 결과를 제시하지 않는다.

{table(['Arm','epoch100 window','epoch100 macro','관측 best-window epoch / RMSE','관측 best-macro epoch / RMSE','best-macro의 window RMSE','분'],rows)}

주 선택 정책은 **strict lowest INNER window RMSE**이며, macro 최소는 별도 보조 기록이다. 두 기준 모두 min-delta를 적용하지 않고 동일값이면 처음 epoch를 저장한다. 아래는 best-window checkpoint에서 같은 eval mode로 계산한 TRAIN/INNER와 auxiliary presence metric이다. TRAIN와 INNER source 분포가 달라 gap 하나만으로 과적합의 원인을 확정할 수 없다.

{table(['Arm','TRAIN eval RMSE','INNER window','INNER macro','INNER MAE','INNER bias','고정 score AUC'],snapshot_rows)}

![100개 실제epoch 곡선](r6_fixed100_figures/01_fixed100_curves.png)

TRAIN online RMSE는 epoch 중 바뀌는 weight의 minibatch 오차다. INNER는 epoch 끝 eval이므로 단순한 동일조건 train/validation gap이 아니다. 학습 loss를 낮추기 위해 쓰는 derived presence score도 보정 확률은 아니며 share 회귀와 상충할 수 있다는 기존 수식적 한계는 유지된다.

## 3. 이전 ES7 결과와 own-law INNER 비교

이전 run은 **site-macro를 ES 중단과 checkpoint 선택 모두에 사용**했고, 사용자의 현재 관심은 **window RMSE 최소**다. 이번에는 ES 비활성화 외에 주 checkpoint 기준도 strict window minimum으로 달라졌다. 따라서 새 best-window의 향상을 전부 ES 제거 하나의 효과로 돌릴 수 없다. CPU threads4→2, min-delta 선택→strict first-minimum 선택도 비교 차이다.

아래 새 Δ는 **새 이력의 exact best-macro epoch−이전 선택 checkpoint**이며 음수는 해당 재사용 INNER에서 낮아졌음을 뜻한다. 이를 통해 checkpoint 목적 변경의 일부를 분리해 보여주지만, exact minimum과 min-delta 규칙·수치 궤적 차이가 있으므로 완전히 독립 평가나 순수한 인과 효과의 추정으로 읽지 않는다.

{table(['Arm','이전 best / stop epoch','이전 window','이전 macro','새 best-macro window Δ','새 best-macro macro Δ'],former)}

![이전 ES와 100epoch 체크포인트 비교](r6_fixed100_figures/02_former_es_comparison.png)

## 4. Macro/window 각각의 분석용 ES7과 scheduler

먼저 원래 macro 이력에 `score<best−1e-5`면 best 갱신, 그 외 stale+1, stale7에서 최초 중단하는 이전 규칙을 그대로 재생한다. 학습기에서 저장한 `shadow_ES7`와 보고서의 독립 replay를 대조한다. 다음으로 **동일한100epoch 이력의 window RMSE**에도 같은 ES7/min-delta 규칙을 별도로 적용한다. 두 counterfactual은 분석용이며 학습을 중단하거나 checkpoint를 바꾸지 않는다. 동일값은 best를 갱신하지 않아 처음을 유지한다.

이는 optimizer/scheduler를 다시 돌린 실험이 아니며, 두 기준 모두 **실제로 사용한 macro-driven LR 경로** 위에서 계산한다. window-driven scheduler로 재학습했을 결과나100epoch 이후 성능은 알 수 없다. 과거 ES run과의 접두 epoch metric/LR 차이를 함께 기록하므로 동일 궤적을 가정하지 않는다. 특히 threads2/4 차이 때문에 새 이력의 macro ES7 stop이 예전 실제 stop과 같을 필요는 없다.

{table(['Arm','이전 실제 stop','새 이력 ES7 최초 stop','counterfactual best epoch','counterfactual macro','접두 비교 epochs','접두 metric/LR maxabs'],counter)}

아래는 각 hypothetical stop까지 이미 관측할 수 있었던 **exact best-window RMSE**와 전체100epoch 최소를 비교한다. 개선량은 `stop까지 prefix minimum−100epoch minimum`으로, 양수면 그 prefix 뒤에 더 낮은 window 오차가 관측됐다는 뜻이다. 이를 해당 중단 정책을 실제로 바꾼 인과적 효과로 해석하지 않는다. ES7 stop이100epoch까지 생기지 않으면100으로 표기하며 개선량은0이다. min-delta로 갱신한 ES 내부 best와 prefix의 exact window minimum은 서로 다를 수 있다.

{table(['Arm','macro ES7 stop/상한','그때 prefix best-window','window ES7 stop/상한','그때 prefix best-window','전체100 best-window','macro stop 뒤 추가 개선','window stop 뒤 추가 개선'],window_counter)}

![실제 LR 경로](r6_fixed100_figures/03_learning_rates.png)

scheduler 정책을 바꾸지 않아도 ES를 제거하면 이후 validation 기록에 따라 LR이 추가로 낮아질 수 있다. 긴 학습의 효과는 그 고정 scheduler 아래의 실제 궤적에 대한 관측이다. 학습률·loss·augmentation·seed를 결과를 보고 재선택하지 않았다.

## 5. 해석 한계와 재현 근거

전체100epoch는 사용자가 요청한 epoch-budget 비교이며 성능 개선을 보장하지 않는다. epoch100이 더 나빠진 경우와 이전 ES보다 불리한 결과도 모두 보고한다. 같은 INNER에서 찾은 최소값에는 반복 관측·선택 편향이 남는다. 새로운 locked/OUTER/Test 평가 없이 이 결과를 LB 개선·공식 대표성·완전한 과적합 해소로 환산하지 않는다.

원본5arm의 context ON/OFF는 파라미터 수 차이가 있고, GroupNorm과 창별 상대 전처리 때문에 no-added-context를 엄밀한 국소 모델로 부를 수 없다. q05는 확인된 배경 B가 아니고 hard-negative의 A=0도 자연 선박 부재 라벨이 아니다. 이100epoch 비교는 그런 기존 생성·입력 정의를 검증된 실제 수중 ground truth로 바꾸지 않는다.

각 arm의100개 epoch weight와 best-window/best-macro/epoch100 alias, 마지막 optimizer·scheduler·Torch RNG state, 세 INNER prediction NPZ를 보존하고 SHA를 검증했다. 부분 학습 재개는 하지 않았다. 보고서 생성에서는 **모델 deserialization이나 NPZ array 로드 없이** 그 파일의 bytes만 무결성 hash로 읽었다. TRAIN/INNER 원본4파일도 opaque SHA만 확인했고, 과거 locked 결과·Test·GPU/SSH·공식 제출은 사용하지 않았다. 기존 dataset 감사 JSON은 locked 기술 요약을 포함하므로 내용을 파싱하지 않고 그 파일의 SHA만 검사했다. 과거 locked 패널은 이미 관찰되어 새 blind holdout이라고 부르지 않는다.

세 가지 기록 목적(epoch100, window 최소, macro 최소)을 구분해 보고하고, 이 단계에서 locked 성능에 맞춘 재학습을 하지 않는다. 모든 input/artifact SHA와 독립 이력 계산은 [재현 receipt](R6_FIXED100_TRAINING_REPORT.receipt.json)에 보존하며, 아래에는 핵심 JSON/code binding만 표시한다.

[새 설정](../{CONFIG}) · [전체 결과 JSON](R6_FIXED100_RESULTS.json) · [이전 guardrails 결과](R6_GUARDRAILS_AND_CPU_ABLATION_REPORT.md)

{table(['핵심 검증 입력','SHA256'],[[n,'`'+d+'`'] for n,d in compact_receipts.items()])}
'''


def self_test():
    rows=[]
    for epoch in range(1,101):
        # The late window minimum and late macro minimum are intentionally distinct.
        window=.3-.001*epoch
        macro=.2 if epoch>3 else .3-.01*epoch
        if epoch==90:window=.05
        if epoch==95:macro=.01
        rows.append({'epoch':epoch,'train_rmse':.4-.002*epoch,'inner_rmse':window,
                     'inner_site_macro_rmse':macro,'learning_rate':.001 if epoch<=20 else .0005})
    a=history_metrics(rows);cf=counterfactual_es7(rows);window_cf=counterfactual_es7(rows,monitor='inner_rmse')
    assert a['window']['epoch']==90 and a['macro']['epoch']==95 and a['last']['epoch']==100
    assert cf['stop_epoch']==11 and cf['best_epoch']==4  # first .2 at4, then seven stale
    assert window_cf['stop_epoch']==97 and window_cf['best_epoch']==90
    macro_window=stopping_window_analysis(rows,cf);window_window=stopping_window_analysis(rows,window_cf)
    assert macro_window['best_window_epoch_through_stop']==11
    assert macro_window['recorded_window_improvement_beyond_stop_prefix']>0
    assert window_window['best_window_epoch_through_stop']==90
    assert window_window['recorded_window_improvement_beyond_stop_prefix']==0
    assert compare_prefix(rows,rows[:11])['all_recorded_metrics_exactly_equal']
    tie=[dict(r) for r in rows];tie[9]['inner_rmse']=.05
    assert history_metrics(tie)['window']['epoch']==10
    short=False
    try:history_metrics(rows[:-1])
    except ValueError:short=True
    assert short
    # A complete temporary fixture uses deliberately NON-loadable PT/NPZ files:
    # passing proves the reporter verifies bytes without deserialization.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='r6_fixed100_report_fixture_',dir='/private/tmp') as directory:
        root=Path(directory)
        def write(name,value):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(value if isinstance(value,bytes) else (json.dumps(value,allow_nan=False)+'\n').encode())
            return sha(p)
        arms=[{'name':name,'generator':'r4' if name.startswith('r4') else 'r6',
               'channels':'legacy8' if 'legacy' in name else 'relative8',
               'global_context':not name.endswith('_local')} for name in EXPECTED]
        neural={'epochs':100,'optimizer':'adam','early_stopping_patience':7,
                'minimum_improvement':1e-5,'threads':2,'batch_size':128,
                'learning_rate':.001,'weight_decay':.0003,'gradient_clip':1.,
                'cell_loss_weight':.25,'presence_loss_weight':.05,'early_stopping_enabled':False}
        old_neural={k:v for k,v in neural.items() if k!='early_stopping_enabled'};old_neural['threads']=4
        old_config={'CNN_arms':arms,'seed':42,'neural':old_neural,'samples':{'train':256}}
        old_config_sha=write(OLD_CONFIG,old_config)
        data_sha=write(DATA_AUDIT,b'opaque fixture, deliberately not parseable JSON; no locked descriptors parsed')
        inputs={n:write(n,b'opaque synthetic TRAIN/INNER; not a valid NPZ') for n in
                [f'data/synthetic/r6_cpu_guardrails_v3_schemafix/{role}_{law}.npz'
                 for role in ['train','inner'] for law in ['r4','r6']]}
        for n in ['data/versions/source_split_r6_guardrails_v2/manifest.json',
                  'data/versions/hard_negatives_r6_v2/manifest.json']:
            inputs[n]=write(n,{'fixture_only':True})
        config={'CNN_arms':arms,'seed':42,'neural':neural,'data_regeneration_or_split_change':False,
                'global_minimum_guaranteed':False,'data_manifest_sha256':data_sha,'used_input_sha256':inputs,
                'scheduler':{'name':'ReduceLROnPlateau','metric':'INNER_site_macro_rmse','factor':.5,'patience':2,'min_lr':1e-6,'changed_from_original':False},
                'checkpoint_policy':{'keep_every_epoch':True,'last_epoch':100,'minimum_improvement_for_checkpoints':0.},
                'execution':{'maximum_parallel_arms':5}}
        config_sha=write(CONFIG,config)
        code={n:write(n,b'fixture source bytes only') for n in
              ['pipeline/fixed100_r6.py','ops/run_r6_fixed100.py']}
        write('ops/build_r6_fixed100_report.py',b'synthetic fixture reporter source identity')
        write(RUN+'/freeze.json',{'config_sha256':config_sha,'data_manifest_sha256':data_sha,'new_code_sha256':code})
        write(OLD_RUN+'/freeze.json',{'config_sha256':old_config_sha,'code_sha256':{}})
        summaries=[];new_hashes={};old_hashes={}
        def metric(row):
            return {'rmse':row['inner_rmse'],'site_macro_rmse':row['inner_site_macro_rmse'],
                    'mae':.1,'bias':0.,'presence_auc':None}
        for arm in arms:
            name=arm['name'];folder=RUN+'/'+name
            history=[{**r,'optimizer_steps':r['epoch']*2,
                      'shadow_ES7_stop_epoch':cf['stop_epoch'],'shadow_ES7_best_macro_epoch':cf['best_epoch']}
                     for r in rows]
            write(folder+'/epochs.jsonl',b''.join((json.dumps(r)+'\n').encode() for r in history))
            checkpoints={f'epoch_{e:03d}.pt':write(folder+f'/checkpoints/epoch_{e:03d}.pt',f'nonmodel fixture {e}'.encode())
                         for e in range(1,101)}
            artifacts={};selected={}
            for label,key,filename in [('window','best_inner_window','best_window.pt'),
                                        ('macro','best_inner_macro','best_macro.pt'),('last','final_epoch','last.pt')]:
                record=a[label];epoch=record['epoch']
                artifacts[filename]=write(folder+'/'+filename,(root/folder/f'checkpoints/epoch_{epoch:03d}.pt').read_bytes())
                selected[key]={'epoch':epoch,'metrics':metric(record),'checkpoint':filename}
            for filename in ARTIFACTS-set(artifacts):
                artifacts[filename]=write(folder+'/'+filename,b'opaque synthetic artifact; never deserialize')
            prior_sha=write(folder+'/legacy_train_prior.json',{'fixture_only':True}) if arm['channels']=='legacy8' else None
            summary={'status':'completed','arm':arm,'epochs_run':100,'early_stopping_enabled':False,
                     'heldout_evaluated':False,'stop_reason':'completed_100_epochs','threads':2,
                     'optimizer_steps':200,'shadow_ES7':{'stop_epoch':cf['stop_epoch'],'best_macro_epoch':cf['best_epoch'],
                     'best_macro_rmse':cf['best_site_macro_rmse'],'stopping_used_for_training':False,'minimum_improvement':1e-5},
                     'model_config':{'input_channels':8},'epoch_checkpoint_sha256':checkpoints,
                     'artifact_sha256':artifacts,'prior_sha256':prior_sha,'elapsed_seconds':1.,
                     'best_window_TRAIN_eval':{'rmse':.08},**selected}
            summaries.append(summary);new_hashes[name]=write(folder+'/summary.json',summary)
            old={'arm':arm,'epochs_run':11,'best_epoch':4,'model_config':{'input_channels':8},
                 'best_inner_metrics':metric(rows[3])}
            old_hashes[name]=write(OLD_RUN+'/'+name+'/summary.json',old)
            write(OLD_RUN+'/'+name+'/epochs.jsonl',b''.join((json.dumps(r)+'\n').encode() for r in rows[:11]))
        write(RUN+'/all_models_frozen.json',{'status':'frozen','config_sha256':config_sha,'heldout_evaluated':False,'model_summary_sha256':new_hashes})
        write(OLD_RUN+'/all_checkpoints_frozen_before_heldout.json',{'status':'frozen','all_requested_arms_completed':True,'model_summaries':old_hashes})
        final={'status':'completed','config_sha256':config_sha,'data_manifest_sha256':data_sha,
               'former_ES_run':OLD_RUN,'summaries':summaries,'heldout_evaluated':False,'elapsed_seconds':2.}
        write(RESULT,final);write(RUN+'/results.json',final)
        cfg,out,studies,receipts=load_inputs(root)
        assert len(studies)==5 and len(receipts)>500
        text=render(cfg,out,studies,receipts)
        assert 'CPU2 threads' in text and 'R6 고정100epoch' in text
        changed=root/RUN/EXPECTED[0]/'checkpoints/epoch_100.pt';changed.write_bytes(b'tampered')
        try:load_inputs(root)
        except ValueError:pass
        else:raise AssertionError('Changed checkpoint must fail closed')
    print('Fixed100 history/minima/ES7 + synthetic five-arm500-checkpoint schema/tamper fixtures passed; no real datasets/results opened.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--self-test',action='store_true')
    choice.add_argument('--generate',action='store_true',help='Only after parent authorizes final schema and generation')
    a=p.parse_args()
    if a.self_test:self_test();return
    output=ROOT/OUTPUT
    if output.exists() or (ROOT/FIGURES).exists() or output.with_suffix('.receipt.json').exists():
        raise FileExistsError('Fresh report/figures required; existing artifacts are preserved')
    config,final,studies,receipts=load_inputs(ROOT)
    text=render(config,final,studies,receipts)
    figures=draw_figures(ROOT,studies)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:f.write(text)
    analysis={name:{'recorded_minima':studies[name]['analysis'],
                    'counterfactual_ES7':studies[name]['counterfactual_es7'],
                    'counterfactual_window_ES7':studies[name]['counterfactual_window_es7'],
                    'macro_stopping_window_analysis':studies[name]['macro_stopping_window_analysis'],
                    'window_stopping_window_analysis':studies[name]['window_stopping_window_analysis'],
                    'former_new_prefix_comparison':studies[name]['prefix']} for name in EXPECTED}
    receipt={'format':'r6_fixed100_report_v1','status':'generated_pending_visual_QA',
             'input_sha256':receipts,'report':OUTPUT,'report_sha256':sha(output),
             'figure_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in figures},
             'independent_history_analysis':analysis,
             'model_or_array_deserialization_performed':False,'model_fit_or_selection_performed':False,
             'old_locked_results_or_Test_GPU_SSH_submission_used':False,
             'threads_comparison':{'former':4,'fixed100':2,'bitwise_prefix_promised':False}}
    with output.with_suffix('.receipt.json').open('x') as f:
        f.write(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'generated','report':OUTPUT,'report_sha256':sha(output),
                      'figures':[p.relative_to(ROOT).as_posix() for p in figures]}))


if __name__=='__main__':main()
