"""R15 saved-output-only audit. No models, raw Test, labels, or fitting."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranks(x):
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return (starts + (counts - 1) / 2.)[inverse]


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    q = np.quantile(x, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return dict(n=len(x), finite=int(np.isfinite(x).sum()), mean=float(x.mean()),
        std_ddof0=float(x.std()), rms=float(np.sqrt(np.mean(x*x))),
        unique_values=int(len(np.unique(x))),
        **{k: float(v) for k, v in zip(['min','p01','p05','p25','median','p75','p95','p99','max'], q)},
        fraction_le001=float(np.mean(x <= .01)), fraction_ge099=float(np.mean(x >= .99)),
        fraction_ge05=float(np.mean(x >= .5)), fraction_ge09=float(np.mean(x >= .9)))


def main():
    source_paths = [
        'models/r14_production_outputs_r2/hybrid_fast3/query_predictions.npz',
        'models/r14_production_outputs_r2/hybrid_fast3/policy.json',
        'models/r14_production_outputs_r1/tree_query_predictions.npz',
        'submissions/r9_fold0_all5_cpu_r1/query_member_predictions.npz',
        'submissions/r9_fold0_all5_cpu_r1/frozen_policy.json',
        'submissions/r9_lightgbm_5fold_cpu_r1/frozen_fold_predictions.npz',
        'pipeline/models_r14_production.py', 'pipeline/training_r14_production.py',
        'ops/run_r14_production_outputs.py', 'ops/run_r14_production_outputs_v2.py',
        'pipeline/relative_grid_r6.py', 'pipeline/relative_grid_r6_v2.py',
    ]
    csv_paths = {
        'R5_CNN': 'submissions/r5_fold0_raw_cnn_cpu_pilot_r1/submission.csv',
        'R9_tree5': 'submissions/r9_lightgbm_5fold_cpu_r1/submission.csv',
        'R9_all5': 'submissions/r9_fold0_all5_cpu_r1/submission.csv',
        'R14_step1': 'submissions/r14_main_r2/submission_r14_step1.csv',
        'R14_exact_Fold0': 'submissions/R14_r2/submission_r14_fold0.csv',
        'R14_tree5': 'submissions/R14/submission_r14_fulltree_5fold.csv',
    }
    source_paths += list(csv_paths.values())
    before = {p: sha(ROOT/p) for p in source_paths}
    with np.load(ROOT/source_paths[0], allow_pickle=False) as z:
        r14 = {k: z[k] for k in z.files}
    ids = r14['sample_id'].astype(str).tolist()
    assert len(ids) == len(set(ids)) == 1500
    series, descriptions, orders, checks = {}, {}, {}, []

    def align(values, other_ids):
        other_ids = np.asarray(other_ids).astype(str).tolist()
        assert len(other_ids) == len(set(other_ids)) == 1500 and set(other_ids) == set(ids)
        ix = {s:i for i,s in enumerate(other_ids)}
        return np.asarray(values)[[ix[s] for s in ids]]

    def add(name, x, origin, definition):
        x = np.asarray(x, dtype=np.float64)
        assert x.shape == (1500,) and np.isfinite(x).all() and ((x >= 0) & (x <= 1)).all()
        assert name not in series
        series[name] = x
        descriptions[name] = {'source': origin, 'definition': definition}

    for name, path in csv_paths.items():
        with (ROOT/path).open(newline='') as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ['ID','share_added_vessel','score_added_vessel']
            rows = list(reader)
        original_ids = [r['ID'] for r in rows]
        orders[name] = original_ids == ids
        for field, col in [('share','share_added_vessel'),('score','score_added_vessel')]:
            add(name+'.'+field, align([float(r[col]) for r in rows], original_ids), path,
                'Saved delivered CSV '+col+'; no new inference')

    members = r14['member_order'].astype(str).tolist()
    assert r14['member_share'].shape == (1500,len(members))
    for i, member in enumerate(members):
        name = 'R14.'+member
        add(name+'.share', r14['member_share'][:,i], source_paths[0],
            'Saved F0 neural share or five-fold-mean tree-family share')
        if member.startswith('nn_'):
            add(name+'.score', r14['member_share'][:,i], source_paths[0],
                'Not an independent classifier: exact R14 source policy score=share; copied saved share')
    for k, s in [('share','R14_step1.share'),('score','R14_step1.score'),
                 ('Fold0_share','R14_exact_Fold0.share'),('Fold0_score','R14_exact_Fold0.score')]:
        delta = float(np.max(np.abs(r14[k].astype(float)-series[s])))
        assert delta == 0
        checks.append({'check':'NPZ versus CSV '+s,'max_absolute_error':delta})

    with np.load(ROOT/source_paths[2], allow_pickle=False) as z:
        tree = {k:z[k] for k in z.files}
    tree_components = align(tree['component_share'], tree['sample_id'])
    tree_order = tree['component_order'].astype(str).tolist()
    for i, key in enumerate(tree_order):
        add('R14_tree_component.'+key+'.share', tree_components[:,i], source_paths[2],
            'Saved individual fold and family regression; independent classifier probability was not saved')
    for family in ['tree_baseline','tree_highshare']:
        cols = [tree_order.index('fold'+str(f)+'_'+family) for f in range(5)]
        mean = tree_components[:,cols].astype(float).mean(1).astype(np.float32)
        err = float(np.max(np.abs(mean-series['R14.'+family+'.share'])))
        assert err == 0
        checks.append({'check':'Exact five-fold mean for '+family,'max_absolute_error':err})

    policy = json.loads((ROOT/source_paths[1]).read_text())
    assert policy['member_order'] == members
    weights = np.asarray(policy['regression_weights'], float)
    recon = np.clip(r14['member_share'].astype(float) @ weights, 0, 1).astype(np.float32)
    err = float(np.max(np.abs(recon-r14['share'])))
    assert err == 0
    checks.append({'check':'Frozen regression policy reconstruction','max_absolute_error':err})

    r9policy = json.loads((ROOT/source_paths[4]).read_text())
    with np.load(ROOT/source_paths[3], allow_pickle=False) as z:
        r9 = {k:z[k] for k in z.files}
    r9names = r9policy['member_order']
    assert r9['share'].shape == r9['score'].shape == (1500,len(r9names))
    for j, member in enumerate(r9names):
        for field in ['share','score']:
            add('R9_member.'+member+'.'+field, align(r9[field][:,j],r9['ID']), source_paths[3],
                'Saved R9 member '+field+'; score is the historical policy, not assumed to equal share')
    for field in ['share','score']:
        assert np.max(np.abs(align(r9['ensemble_'+field],r9['ID'])-series['R9_all5.'+field])) == 0

    with np.load(ROOT/source_paths[5], allow_pickle=False) as z:
        r9tree = {k:z[k] for k in z.files}
    for key, values in r9tree.items():
        if key == 'ID': continue
        add('R9_tree_cache.'+key, align(values,r9tree['ID']), source_paths[5],
            'Stored raw classifier probability' if key.endswith('_score_classifier') else 'Stored fold/mean prediction')
    for field in ['share','score']:
        err = float(np.max(np.abs(align(r9tree['mean_'+field],r9tree['ID'])-series['R9_tree5.'+field])))
        assert err < 1e-15
        checks.append({'check':'R9 tree cache versus CSV '+field,'max_absolute_error':err})

    table = [{'series':name, **stats(x), **descriptions[name]} for name,x in series.items()]
    names = list(series)
    matrix = np.column_stack([series[n] for n in names])
    pearson = np.corrcoef(matrix, rowvar=False)
    spearman = np.corrcoef(np.column_stack([ranks(series[n]) for n in names]), rowvar=False)
    assert np.isfinite(pearson).all() and np.isfinite(spearman).all()
    for filename, mat in [('pearson.csv',pearson),('spearman.csv',spearman)]:
        with (OUT/filename).open('x',newline='') as f:
            writer=csv.writer(f);writer.writerow(['series']+names)
            for n,row in zip(names,mat):writer.writerow([n]+row.tolist())
    with (OUT/'distribution_summary.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    with (OUT/'aligned_saved_predictions_1500.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['ID']+names)
        for i,sample in enumerate(ids):writer.writerow([sample]+matrix[i].tolist())
    availability = {
        'R14_NN_independent_classifier': 'DOES_NOT_EXIST: score=share exactly in production model source',
        'R14_per_fold_tree_raw_classifier': 'NOT_SAVED in either approved query NPZ; no probabilities invented or reverse-engineered',
        'R14_tree_family_mean_classifier': 'NOT_SAVED; final blended score exists, but is not labelled raw classifier',
        'R9_per_fold_tree_raw_classifier': 'AVAILABLE: frozen_fold_predictions.npz fold_XX_score_classifier',
        'R9_NN_score': 'AVAILABLE as historical saved member scores; not R14 score policy',
        'R14_TFGrid': 'NOT_IN_THIS_FAST3_OUTPUT; no prediction fabricated',
    }
    def corr(a,b): return float(np.corrcoef(series[a],series[b])[0,1])
    deltas = {}
    for field in ['share','score']:
        a,b=series['R9_all5.'+field],series['R14_step1.'+field]
        deltas[field]={'mean_delta':float((b-a).mean()),'RMS_delta':float(np.sqrt(np.mean((b-a)**2))),
            'median_absolute_delta':float(np.median(np.abs(b-a))), 'maximum_absolute_delta':float(np.max(np.abs(b-a))),
            'Pearson':corr('R9_all5.'+field,'R14_step1.'+field)}
    after = {p:sha(ROOT/p) for p in source_paths}
    assert before == after
    result={'status':'PASS_saved_output_descriptive_audit','created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Saved public-query predictions ONLY; no raw acoustic query, hidden labels, fit, inference, calibration, or policy changes',
        'rows':1500,'series_count':len(names),'ID_set_and_alignment_verified':True,
        'original_CSV_order_matches_R14_saved_ID':orders,'source_sha256':before,'source_unchanged':True,
        'ID_order_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        'availability':availability,'frozen_R14_policy':{'member_order':members,'weights':weights.tolist(),
            'score_policy':policy['score_policy'],'gate_RMSE':policy['gate_proposed_RMSE']},
        'checks':checks,'R14_minus_R9':deltas,'statistics':table,
        'interpretation_limits':['Prediction distributions and correlations are not target distributions or error correlations.',
            'No hidden-label target moments or per-site official performance can be inferred.',
            'High prediction correlation does not imply high AUC or low RMSE.',
            'No new production weights, thresholds, or Test matching are selected.']}
    with (OUT/'RESULTS.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='OUTPUT_MANIFEST.json'}
    (OUT/'OUTPUT_MANIFEST.json').write_text(json.dumps({'source_sha256':before,'output_sha256':outputs},indent=2))
    print(json.dumps({'status':result['status'],'rows':1500,'series':len(names),'result_sha256':sha(OUT/'RESULTS.json'),
        'wide_table':str((OUT/'aligned_saved_predictions_1500.csv').relative_to(ROOT)),'deltas':deltas}))


if __name__ == '__main__':
    main()
