"""Build report-only figures/facts from frozen local evidence; never fit/read Test."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = REPORTS / 'r5_issue_figures'
    folder.mkdir(exist_ok=True)
    archive = ROOT / 'ops/server_backups/r5_gpu_r1/r5_snapshot_20261025T181356_426036Z_a349d40f.zip'
    with zipfile.ZipFile(archive) as z:
        summaries = {}
        for obs in ['raw', 'resolution_01', 'resolution_01_endpoint']:
            prefix = f'fold_00/preprocessing_pilot/{obs}/bg_cnn/'
            d = json.loads(z.read(prefix + 'summary.json'))
            summaries[obs] = {k: d[k] for k in ['best_epoch', 'epochs_run', 'best_inner_rmse', 'elapsed_seconds', 'stop_reason']}
        epochs = [json.loads(row) for row in z.read('fold_00/preprocessing_pilot/raw/bg_cnn/epochs.jsonl').decode().splitlines()]
    profile = json.loads((REPORTS / 'R5_CNN_MAC_INFERENCE_PROFILE.json').read_text())
    observed = json.loads((ROOT / 'ops/r5_lb_failure_observation.json').read_text())
    scale = json.loads((REPORTS / 'FINAL_DATASET_SCALE.json').read_text())
    cpu = profile['INNER_metrics']['CPU_FP32_RMSE']
    public_rmse = observed['public_share_rmse_user_screenshot']
    benchmark = 1 - .753867825
    facts = {
        'scope': 'Report-only local evidence aggregation. User screenshot public metrics; uploaded bytes unverified.',
        'dataset': scale,
        'completed_CNNs': summaries,
        'raw_CNN_epochs': [{k: e[k] for k in ['epoch', 'train_rmse', 'inner_rmse', 'best_epoch', 'patience_used', 'learning_rate']} for e in epochs],
        'CPU_INNER_RMSE': cpu,
        'public_RMSE_user_screenshot': public_rmse,
        'RMSE_arithmetic_gap_different_populations': public_rmse - cpu,
        'RMSE_ratio_different_populations': public_rmse / cpu,
        'MSE_ratio_different_populations': (public_rmse / cpu) ** 2,
        'benchmark_RMSE_from_user_score': benchmark,
        'public_RMSE_minus_benchmark': public_rmse - benchmark,
        'public_RMSE_relative_increase_vs_benchmark': public_rmse / benchmark - 1,
        'role_percentages': {k: v['samples'] / scale['samples_total'] * 100 for k, v in scale['roles'].items()},
        'Fold0_role_percentages': {'train': 53244 / 59244 * 100, 'inner': 3000 / 59244 * 100, 'outer': 3000 / 59244 * 100},
        'observation_receipt': observed,
        'no_Test_OUTER_GPU_SSH_or_fitting': True,
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [archive, REPORTS / 'R5_CNN_MAC_INFERENCE_PROFILE.json', ROOT / 'ops/r5_lb_failure_observation.json', REPORTS / 'FINAL_DATASET_SCALE.json']},
    }
    (REPORTS / 'R5_CURRENT_ISSUES_FACTS.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.facecolor': 'white'})
    fig, ax = plt.subplots(figsize=(8.4, 3.8), constrained_layout=True)
    ax.plot([e['epoch'] for e in epochs], [e['train_rmse'] for e in epochs], label='TRAIN (fresh mixtures; online)', color='#3e719b', lw=1.8)
    ax.plot([e['epoch'] for e in epochs], [e['inner_rmse'] for e in epochs], label='INNER (fixed 3,000; epoch selection)', color='#c37b39', lw=1.8)
    ax.scatter([19], [summaries['raw']['best_inner_rmse']], color='#c37b39', zorder=3)
    ax.axvline(19, color='#999999', ls=':', lw=1)
    ax.annotate('Best epoch 19: 0.090849', (19, summaries['raw']['best_inner_rmse']), xytext=(10, .15), arrowprops={'arrowstyle': '->', 'color': '#777777'}, fontsize=9)
    ax.set(xlabel='Epoch', ylabel='Share RMSE', title='Fold 0 raw CNN: actual training history', xlim=(1, 26), ylim=(0, .20))
    ax.grid(axis='y', alpha=.18)
    ax.legend(loc='lower left', fontsize=8)
    fig.savefig(folder / '01_raw_cnn_learning_curve.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.4, 3.8), constrained_layout=True)
    names = ['Own INNER\nCPU FP32', 'User benchmark\npublic', 'Submitted CNN\npublic']
    values = [cpu, benchmark, public_rmse]
    bars = ax.bar(names, values, color=['#3e719b', '#8b9b78', '#b85750'], width=.54)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .009, f'{value:.6f}', ha='center', fontsize=10)
    ax.set(ylabel='Share RMSE (lower is better)', title='Observed evaluation gap: different populations', ylim=(0, .4))
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    fig.text(.5, -.07, 'INNER is used for epoch selection. Public labels/sample membership are unavailable; bars are not paired evaluations.', ha='center', fontsize=7)
    fig.savefig(folder / '02_inner_public_gap.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({k: facts[k] for k in ['CPU_INNER_RMSE', 'RMSE_arithmetic_gap_different_populations', 'RMSE_ratio_different_populations', 'MSE_ratio_different_populations', 'public_RMSE_relative_increase_vs_benchmark', 'role_percentages', 'Fold0_role_percentages']}, indent=2))


if __name__ == '__main__':
    main()
