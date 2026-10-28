"""One frozen, non-query evaluation of the declared support x feature pilot."""
from pathlib import Path
import csv, datetime, hashlib, json, os, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'reports/r16_support_temporal_r1'
MANIFEST = OUT / 'data/manifest.json'
MODELS = ROOT / 'models/r16_support_temporal_r1'
ARMS = ('D0F0', 'D0F1', 'D1F0', 'D1F1')
OPS = ('control_eq6', 'mfhf_shelf6', 'mfhf_shelf12', 'mfhf_shelf12_lfminus6',
       'alpha_preEQ_inserted', 'independent_subband_snr')
METHODS = (('NN', 'nn_share', 'nn_classifier'), ('Tree', 'tree_share', 'tree_classifier'),
           ('fixed60T40N', 'balanced_share', 'balanced_classifier'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.tmp')
    q.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    q.replace(p)


def guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0]).replace('\\', '/').lower()
        if (Path(p).name in ('test.csv', 'sample_submission.csv') or '/eda_deep/' in p
                or '/submissions/' in p or '/saved_query_audit' in p or '/r15_band_comparison_' in p):
            raise PermissionError('Frozen pilot evaluation prohibits query/cohort-derived access')
    if event in ('socket.connect', 'subprocess.Popen', 'os.system'):
        raise PermissionError('Evaluation does not require network or subprocesses')


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def measure(y, p, c, weights=None):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, float); p = np.asarray(p, float); c = np.asarray(c, float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    if not len(y) or not np.isfinite(np.r_[y, p, c, w]).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Invalid evaluation values')
    if ((np.r_[p, c] < 0) | (np.r_[p, c] > 1)).any():
        raise ValueError('Prediction outside physical/probability range')
    w = w / w.sum(); e = p - y; positive = y > 0
    avg = lambda a: float(np.dot(w, np.asarray(a, float)))
    result = {'n': len(y), 'positive_n': int(positive.sum()), 'zero_n': int((~positive).sum()),
        'rmse': float(np.sqrt(avg(e * e))), 'mae': avg(abs(e)), 'bias': avg(e),
        'target_mean': avg(y), 'prediction_mean': avg(p), 'prediction_rms': float(np.sqrt(avg(p*p))),
        'classifier_mean': avg(c), 'classifier_ge099_fraction': avg(c >= .99),
        'auc_share': float(roc_auc_score(positive, p, sample_weight=w)) if positive.any() and (~positive).any() else None,
        'auc_classifier': float(roc_auc_score(positive, c, sample_weight=w)) if positive.any() and (~positive).any() else None}
    for name, ix in [('positive', positive), ('negative', ~positive), ('y_gt020', y > .2)]:
        if ix.any():
            ww = w[ix] / w[ix].sum()
            result[name] = {'n': int(ix.sum()), 'target_mean': float(ww @ y[ix]),
                'prediction_mean': float(ww @ p[ix]), 'bias': float(ww @ e[ix]),
                'rmse': float(np.sqrt(ww @ (e[ix]**2))), 'classifier_mean': float(ww @ c[ix]),
                'classifier_ge05_fraction': float(ww @ (c[ix] >= .5))}
    return result


def freeze():
    all_done = read(MODELS / 'ALL_COMPLETED.json')
    pins = {'ops/evaluate_r16_support_temporal.py': sha(__file__),
        'reports/r16_support_temporal_r1/PROTOCOL.md': sha(OUT / 'PROTOCOL.md'),
        str((MODELS / 'ALL_COMPLETED.json').relative_to(ROOT)): sha(MODELS / 'ALL_COMPLETED.json')}
    for arm in ARMS:
        folder = MODELS / arm; s = read(folder / 'COMPLETED.json')
        if s['status'] != 'completed' or s['epochs'] != 100 or s['NN_early_stopping'] is not False or s['fixture_only']:
            raise ValueError('All four declared 100epoch arms must complete')
        for name, h in s['artifacts_sha256'].items():
            p = folder / name
            if sha(p) != h: raise ValueError('Model artifact changed: ' + str(p))
            pins[str(p.relative_to(ROOT))] = h
        pins[str((folder / 'COMPLETED.json').relative_to(ROOT))] = sha(folder / 'COMPLETED.json')
        pins.update(s['source_code_sha256'])
    # The historical comparator and diagnostic arrays are frozen evidence too.
    historical = ROOT / 'models/r16_clean_pilot_r1/mixed'
    hs = read(historical / 'COMPLETED.json')
    for name, h in {**hs['checkpoint_sha256'], **hs['tree_sha256']}.items():
        p = historical / name
        if sha(p) != h: raise ValueError('Historical model changed: ' + str(p))
        pins[str(p.relative_to(ROOT))] = h
    pins[str((historical / 'COMPLETED.json').relative_to(ROOT))] = sha(historical / 'COMPLETED.json')
    old_plan = ROOT / 'reports/r16_r11_stress_r1/PLAN.json'
    old_pins = read(old_plan)['input_file_sha256']
    pins[str(old_plan.relative_to(ROOT))] = sha(old_plan)
    for name in ('r9_original', 'native_long15', 'hf_high', 'st_francis_dense'):
        rel = 'reports/r11_diagnostics/full_support_hf_r1/'+name+'_samples.npz'
        h = old_pins[rel]
        if sha(ROOT/rel) != h: raise ValueError('Fixed R11 sample changed: '+rel)
        pins[rel] = h
    pins['ops/r16_support_temporal_data.py'] = sha(ROOT/'ops/r16_support_temporal_data.py')
    policy = {'status': 'fixed_before_locked_evaluation', 'regression': {'Tree': .6, 'NN': .4},
        'score': {'Tree_classifier': .5, 'NN_classifier': .5, 'share': 0},
        'fit_on_any_eval_panel': False, 'Test_used': False}
    write(OUT / 'POLICY.json', policy)
    pins[str((OUT / 'POLICY.json').relative_to(ROOT))] = sha(OUT / 'POLICY.json')
    receipt = {'status': 'completed', 'all_models_frozen': True, 'ensemble_policy_frozen': True,
        'data_manifest_sha256': sha(MANIFEST), 'artifact_sha256': pins,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'Test_used': False,
        'scope': 'new source split; four small factorial arms only; not full production',
        'original_completion_receipt_sha256': sha(MODELS / 'ALL_COMPLETED.json')}
    write(OUT / 'GLOBAL_FREEZE.json', receipt)
    return receipt


def verify(receipt):
    if sha(MANIFEST) != receipt['data_manifest_sha256']: raise ValueError('Data manifest changed')
    for name, h in receipt['artifact_sha256'].items():
        if sha(ROOT / name) != h: raise ValueError('Frozen artifact changed: ' + name)


def main():
    import torch
    from ops.r16_support_temporal_data import load_panel, load_long_panel
    from ops.r16_support_temporal_models import predict_arm
    from ops.r16_clean_models import predict_arm as predict_historical
    sys.addaudithook(guard); torch.set_num_threads(2); torch.set_num_interop_threads(1)
    result_dir = OUT / 'evaluation'
    if result_dir.exists(): raise FileExistsError('Fresh frozen evaluation output required')
    result_dir.mkdir(); (result_dir / 'predictions').mkdir()
    began = time.perf_counter(); receipt = freeze(); rows = []; stored = {}; predictions = {}; cohorts = {}
    freeze_path = OUT / 'GLOBAL_FREEZE.json'

    def evaluate(name, d, *, role, operator='', interpretation='', weights=None):
        verify(receipt); y = np.asarray(d['share'], float)
        ids = d.get('sample_id', np.asarray([name + '_' + str(i) for i in range(len(y))])).astype(str)
        obs = np.isfinite(d['db']).all(2).sum(1)
        histogram = lambda x: {str(k): int(v) for k,v in zip(*np.unique(x,return_counts=True))}
        cohorts[name] = {'observed_minute_counts': histogram(obs),
            'positive_observed_minute_counts': histogram(obs[y>0]), 'zero_observed_minute_counts': histogram(obs[y==0])}
        if 'site' in d:
            cohorts[name].update(positive_site_counts=histogram(np.asarray(d['site'])[y>0]),
                zero_site_counts=histogram(np.asarray(d['site'])[y==0]))
        if 'cell_share' in d:
            support = (np.asarray(d['cell_share'])>0).any(2).sum(1)
            cohorts[name]['positive_actual_support_minute_counts'] = histogram(support[y>0])
        pr = {arm: predict_arm(MODELS / arm, d) for arm in ARMS}
        pr['historical_R16_mixed'] = predict_historical(ROOT / 'models/r16_clean_pilot_r1/mixed', d)
        names = list(pr); target = result_dir / 'predictions' / (name + '.npz')
        arrays = {'sample_id': ids, 'target_share': y, 'model_order': np.asarray(names),
            'sample_weight': np.ones(len(y)) if weights is None else weights}
        for key in ('tree_share', 'tree_classifier', 'nn_share', 'nn_classifier', 'balanced_share', 'balanced_classifier'):
            arrays[key] = np.column_stack([pr[a][key] for a in names])
        for key in ('site', 'base_window_index', 'reference_component', 'background_group'):
            if key in d: arrays[key] = d[key]
        np.savez_compressed(target, **arrays); stored[str(target.relative_to(result_dir))] = sha(target)
        predictions[name] = {'data': d, 'values': pr}
        for arm, v in pr.items():
            for method, pk, ck in METHODS:
                m = measure(y, v[pk], v[ck], weights)
                m.update(panel=name, role=role, operator=operator, arm=arm, model=method, interpretation=interpretation)
                if weights is not None: m['unweighted_rmse'] = measure(y, v[pk], v[ck])['rmse']
                if 'site' in d:
                    site = np.asarray(d['site']).astype(str)
                    m['site_slices'] = {s: measure(y[site==s], v[pk][site==s], v[ck][site==s]) for s in sorted(set(site))}
                if method=='fixed60T40N' and 'cell_share' in d:
                    m['positive_support_slices'] = {}
                    for low, high in ((1,4),(5,10),(11,15)):
                        ix = (support>=low)&(support<=high)&(y>0)
                        if ix.any(): m['positive_support_slices'][f'{low}to{high}'] = measure(y[ix],v[pk][ix],v[ck][ix])
                rows.append(m)
        print(json.dumps({'event': 'panel', 'name': name, 'n': len(y), 'seconds': time.perf_counter()-began}), flush=True)
        return pr

    # Primary roles are not fitted or used to choose new settings/mixture weights.
    for role in ('local_gate', 'local_report', 'domain_holdout'):
        ds = []
        for op in OPS:
            d = load_panel(MANIFEST, role, op, freeze_receipt=freeze_path); ds.append(d)
            evaluate(role+'__'+op, d, role=role, operator=op,
                interpretation='new C-prime source split' if role != 'local_gate' else 'same unchanged B gate')
        joined = {k: np.concatenate([d[k] for d in ds]) for k in ds[0] if all(k in d for d in ds)}
        # Reuse saved predictions rather than infer the pooled role again.
        for arm in (*ARMS, 'historical_R16_mixed'):
            for method, pk, ck in METHODS:
                p = np.concatenate([predictions[role+'__'+op]['values'][arm][pk] for op in OPS])
                c = np.concatenate([predictions[role+'__'+op]['values'][arm][ck] for op in OPS])
                rows.append({'panel': role+'__POOLED', 'role': role, 'operator': 'POOLED', 'arm': arm,
                    'model': method, **measure(joined['share'], p, c)})

    # Independently held out 12-minute reference; class balancing is explicit.
    long_pos = []
    for op in OPS:
        d = load_long_panel(MANIFEST, 'native12_positive', operator=op, freeze_receipt=freeze_path)
        if not np.all(d['selected_minutes']==12) or not np.all(np.asarray(d['share'])>0):
            raise ValueError('Independent long panel must contain only real native12 positives')
        long_pos.append(d)
    positive = {k: np.concatenate([d[k] for d in long_pos]) for k in long_pos[0] if all(k in d for d in long_pos)}
    for name in ('hf_high_negative', 'st_francis_dense_negative'):
        neg = load_long_panel(MANIFEST, name, freeze_receipt=freeze_path)
        common = set(positive) & set(neg)
        pair = {k: np.concatenate([positive[k], neg[k]]) for k in common}
        weights = np.r_[np.full(len(positive['share']), .5/len(positive['share'])),
                        np.full(len(neg['share']), .5/len(neg['share']))]
        evaluate('native12_VS_'+name, pair, role='independent_long',
            interpretation='one independent heldout12min reference component; no claim of unseen15min', weights=weights)

    # Known old R11 diagnostic inputs: no synthetic flat15 panel and no fit.
    legacy = ROOT / 'reports/r11_diagnostics/full_support_hf_r1'
    fixed = {name: load_npz(legacy / (name+'_samples.npz'))
             for name in ('r9_original', 'native_long15', 'hf_high', 'st_francis_dense')}
    for name, d in fixed.items():
        evaluate('legacy_R11__'+name, d, role='known_R11',
            interpretation='development-exposed stress; native15 reference seen after new split')
    for name in ('hf_high', 'st_francis_dense'):
        pos = fixed['native_long15']; neg = fixed[name]
        pair = {k: np.concatenate([pos[k], neg[k]]) for k in set(pos)&set(neg)}
        weights = np.r_[np.full(len(pos['share']), .5/len(pos['share'])), np.full(len(neg['share']), .5/len(neg['share']))]
        # Reuse individual predictions exactly, no second inference.
        for arm in (*ARMS, 'historical_R16_mixed'):
            for method, pk, ck in METHODS:
                p = np.r_[predictions['legacy_R11__native_long15']['values'][arm][pk], predictions['legacy_R11__'+name]['values'][arm][pk]]
                c = np.r_[predictions['legacy_R11__native_long15']['values'][arm][ck], predictions['legacy_R11__'+name]['values'][arm][ck]]
                rows.append({'panel': 'legacy_R11__native15_VS_'+name, 'role': 'known_R11', 'operator': '',
                    'arm': arm, 'model': method, 'interpretation': 'reference-seen diagnostic, not independent generalization',
                    **measure(pair['share'], p, c, weights), 'unweighted_rmse': measure(pair['share'], p, c)['rmse']})

    index = {(r['panel'], r['arm'], r['model']): r for r in rows}; checks = []; interactions = []
    for role in ('local_gate', 'local_report', 'domain_holdout'):
        panel = role+'__POOLED'; a = index[panel, 'D0F0', 'fixed60T40N']; b = index[panel, 'D1F1', 'fixed60T40N']
        checks.append({'name': role+'_primary_preservation', 'baseline_rmse': a['rmse'], 'corrected_rmse': b['rmse'],
            'rmse_tolerance': max(.005, .05*a['rmse']), 'auc_change': b['auc_classifier']-a['auc_classifier'],
            'passed': bool(b['rmse'] <= a['rmse']+max(.005, .05*a['rmse']) and b['auc_classifier'] >= a['auc_classifier']-.02)})
    for name in ('hf_high_negative', 'st_francis_dense_negative'):
        panel = 'native12_VS_'+name; a = index[panel, 'D0F0', 'fixed60T40N']; b = index[panel, 'D1F1', 'fixed60T40N']
        checks.append({'name': name+'_unseen_long_AUC', 'AUC': b['auc_classifier'], 'minimum': .75,
                       'passed': bool(b['auc_classifier'] >= .75)})
        if name == 'hf_high_negative':
            ca = a['negative']['classifier_mean']; cb = b['negative']['classifier_mean']
            checks.append({'name': 'matched_HF_false_positive_reduction', 'baseline_probability_mean': ca,
                'corrected_probability_mean': cb, 'negative_share_mean_change': b['negative']['prediction_mean']-a['negative']['prediction_mean'],
                'passed': bool((cb <= ca-.10 or cb <= .75*ca) and b['negative']['prediction_mean'] <= a['negative']['prediction_mean']+.01)})
    for panel in sorted({r['panel'] for r in rows}):
        if all((panel, arm, 'fixed60T40N') in index for arm in ARMS):
            arm_rows = [index[panel, arm, 'fixed60T40N'] for arm in ARMS]
            interactions.append({'panel': panel, 'rmse_interaction_DxF': arm_rows[3]['rmse']-arm_rows[2]['rmse']-arm_rows[1]['rmse']+arm_rows[0]['rmse'],
                'AUC_interaction_DxF': (arm_rows[3]['auc_classifier']-arm_rows[2]['auc_classifier']-arm_rows[1]['auc_classifier']+arm_rows[0]['auc_classifier']) if arm_rows[0]['auc_classifier'] is not None else None})
    verify(receipt)
    result = {'status': 'completed', 'rows': rows, 'checks': checks, 'small_gate_passed': all(c['passed'] for c in checks),
        'factorial_interactions': interactions, 'cohort_descriptions': cohorts, 'predictions_sha256': stored, 'global_freeze_sha256': sha(freeze_path),
        'Test_accessed': False, 'eval_fit_or_repeated_search': False, 'official_submission': False,
        'elapsed_seconds': time.perf_counter()-began,
        'limitations': ['one heldout12min component; cannot prove unseen>=15min reference transfer',
                        'selected119 conditional reference graph, not independent full509 corpus',
                        'known R11 native15 becomes reference-seen after declared split; not a selection panel']}
    write(result_dir/'RESULTS.json', result)
    cols = ['panel','role','operator','arm','model','n','rmse','bias','target_mean','prediction_mean','prediction_rms','auc_share','auc_classifier','classifier_mean']
    with (result_dir/'metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    md = ['# R16 자연 support·시간 구조 2×2 실측 결과', '',
          '네 조건의 CNN은 각각 100epoch/Adam, Tree 회귀·독립 분류는 local_fit만 사용했다. 결합은 사전 고정 share=60%Tree+40%CNN, score=독립 분류 확률 50%씩이다.', '',
          '| 평가 | 조건 | RMSE | 독립 분류 AUC | 예측 평균 |', '|---|---|---:|---:|---:|']
    for r in rows:
        if r['model']=='fixed60T40N' and (r['operator']=='POOLED' or r['role']=='independent_long'):
            md.append(f"| {r['panel']} | {r['arm']} | {r['rmse']:.6f} | {r['auc_classifier']:.6f} | {r['prediction_mean']:.6f} |")
    md += ['', '## 사전 운영 기준', '']
    md += [f"- {c['name']}: {'통과' if c['passed'] else '미달'}" for c in checks]
    md += ['', '**본체 연결 관문: '+('통과' if result['small_gate_passed'] else '미달')+'**', '',
        '원천 독립 장시간 평가는 12분 컴포넌트 하나다. R11 native15는 새 TRAIN과 참조가 겹치는 진단이며 미관측15분 전이로 주장하지 않는다. Test를 읽거나 공식 제출하지 않았다.', '',
        '[전 지표](evaluation/metrics.csv) · [수치/검증 기준](evaluation/RESULTS.json) · [사전 실험안](PROTOCOL.md)']
    (OUT/'R16_SUPPORT_TEMPORAL_ABLATION.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'status': 'completed', 'gate_passed': result['small_gate_passed'], 'seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
