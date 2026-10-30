#!/usr/bin/env python3
"""Train R18 production D0/H0 trees once; clean_ES chooses the auxiliary branch.

No new K-fold. No Test IO. Fresh-fit and fixed panels are scored only after the
regression policy is frozen, and never choose this module's model or features.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('DO_NOT_TRACK', '1')
from pathlib import Path
import argparse, gc, json, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.trees_r18_production import transform, schema, fit_background_prior, predict, SEED
from pipeline.trees_r16_production import sha, write, array_sha, token

DATA = ROOT/'data/synthetic/r18_production_v1'
MODEL = ROOT/'models/r18_production_r1/trees'
OUT = ROOT/'reports/r18_production_r1/trees'
CODE = ['pipeline/trees_r18_production.py', 'ops/train_r18_production_tree.py',
        'pipeline/training_r16_multisupport.py', 'pipeline/trees_r16_production.py',
        'pipeline/relative_grid_r6.py', 'pipeline/relative_grid_r6_v2.py',
        'pipeline/physics_features_r6_v2.py', 'ops/r16_temporal_features.py']
PANELS = ['local_gate', 'domain_holdout', 'native12_VS_hf_high_negative',
          'native12_VS_st_francis_dense_negative', 'local_report']


def guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        if Path(os.fsdecode(args[0])).name.lower() in ('test.csv', 'sample_submission.csv', 'query_predictions.npz'):
            raise PermissionError('R18 production tree training/evaluation prohibits query access')


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def progress(**value):
    value['unix_time'] = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT/'progress.json', value)
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def metrics(y, p, c, weights=None, site=None):
    from sklearn.metrics import roc_auc_score
    y, p, c = np.asarray(y, float), np.asarray(p, float), np.asarray(c, float)
    w = np.ones(len(y), float) if weights is None else np.asarray(weights, float)
    w /= w.sum()
    z = y > 0
    error = p-y
    result = dict(n=len(y), rmse=float(np.sqrt(w@(error*error))), bias=float(w@error),
                  target_mean=float(w@y), prediction_mean=float(w@p),
                  prediction_rms=float(np.sqrt(w@(p*p))),
                  share_auc=float(roc_auc_score(z, p, sample_weight=w)) if z.any() and (~z).any() else None,
                  classifier_auc=float(roc_auc_score(z, c, sample_weight=w)) if z.any() and (~z).any() else None,
                  classifier_mean=float(w@c), score_ge090=float(w@(c >= .9)))
    if site is not None:
        ss = np.asarray(site).astype(str)
        result['site_macro_rmse'] = float(np.mean([np.sqrt(np.mean(error[ss == s]**2)) for s in np.unique(ss)]))
    return result


def preflight():
    from ops.r18_classifier_audit import extra_features, names
    from ops.r17_clean_classifier_features import classifier_indices
    from pipeline.training_r16_multisupport import tree_features
    rng = np.random.default_rng(SEED)
    d = np.round(rng.normal(90, 7, (8, 15, 25)), 1).astype(np.float32)
    d[0, 0] = np.nan;d[1, 3:5] = np.nan
    new = transform(d, 'classification')
    old = np.concatenate([tree_features(d, 'classification')[:, classifier_indices('direct')], extra_features(d)[0]], 1)
    assert np.array_equal(new, old)
    assert schema()['classification_profile_energy'] == names('profile_energy')
    delta = float(abs(new-transform(d+20, 'classification')).max())
    assert delta < 1e-5
    assert transform(d, 'relative').shape == (8, 4667)
    dummy_prior = dict(centers=np.tile(np.arange(25, dtype=float), (16, 1)).tolist(),
                       templates_db=np.full((16, 25), 80.).tolist(), global_template_db=np.full(25, 80.).tolist())
    assert transform(d, 'auxiliary', dummy_prior).shape == (8, 4679)
    result = dict(status='passed', classifier_exact_R18_profile_energy_parity=True,
                  classifier_common_gain20_max_abs=delta,
                  classification_features=2408, regression_relative_features=4667,
                  regression_auxiliary_features=4679, pure_mask_regression_removed=493,
                  pure_mask_classifier_removed=25,
                  real_audio_lag_correlation_features_preserved=True,
                  warning='Removing explicit O columns does not make every feature mask-blind; full-mask presence pairing is verified in the data contract.')
    write(OUT/'PREFLIGHT.json', result)
    return result


def protocol(contract):
    return dict(format='r18_single_split_production_tree_v1', status='registered_before_fit',
                data_contract_sha256=sha(DATA/'DATA_CONTRACT.json'),
                data_manifest_sha256=contract['manifest_sha256'],
                new_KFold=False, regimes=['D0', 'H0'], CPU_threads=2, seed=SEED,
                full_train_rows_per_regime=60000, clean_ES_rows_per_regime=2000,
                maximum_trees=1500, early_stopping_patience=40,
                learning_rate=.035, num_leaves=31, min_child_samples=50, max_bin=127,
                regression_branches=['relative4667', 'relative_plus_level_TRAINprior4679'],
                classification='Exact R18 profile_energy2408: pure observed fractions25 removed; Ch4 profile25 and sqrtw375 restored',
                regression_selection=dict(role='clean_ES_only', minimum_RMSE_improvement=.0001,
                    maximum_site_macro_RMSE_degradation=.0005, maximum_abs_bias_degradation=.0005,
                    fallback='relative'),
                auxiliary_features=schema()['regression_auxiliary'][-12:],
                auxiliary_prior='TRAIN-only original B shape16 templates, no source-ID query lookup/PCA/LF anchor; descriptors only, never upper-bound masks',
                contextual_evidence='R18 small raw/level improvement is weak and does not prove the conditional TRAIN-background descriptors help; the branch must win this full-data clean_ES comparison.',
                early_stop_metric=dict(regression='clipped_RMSE', classification='binary_logloss'),
                evaluation_after_policy_freeze=['fresh_fit_D0', 'fresh_fit_H0']+PANELS,
                evaluation_selects_models=False, Test_used=False,
                source_code_sha256={n: sha(ROOT/n) for n in CODE})


def validate_roles(contract):
    if contract['status'] != 'completed':
        raise RuntimeError('Production data contract is not complete')
    if sha(DATA/'manifest.json') != contract['manifest_sha256']:
        raise ValueError('Production manifest hash differs')
    if contract.get('KFold') is not False or contract.get('Test_used') is not False:
        raise ValueError('Foreign production scope')
    for role in ('train', 'clean_ES', 'fresh_fit'):
        for regime in ('D0', 'H0'):
            rec = contract['roles'][role][regime]
            p = ROOT/rec['path']
            if sha(p) != rec['sha256']:
                raise ValueError('Dataset changed: '+str(p))
    # Read only source metadata, without evaluating fresh-fit labels yet.
    import pandas as pd
    source = {role: pd.read_parquet(DATA/(role+'_backgrounds.parquet')) for role in ('train','clean_ES','fresh_fit')}
    from pipeline.r16_production_data import _crossrole
    proof = []
    for i, (role, frame) in enumerate(source.items()):
        proof += [dict(primary=role, **r) for r in _crossrole(frame, dict(list(source.items())[i+1:]))]
    write(OUT/'TRAIN_ES_FIT_SOURCE_PURGE.json', proof)
    return source


def fit_one_regime(regime, contract, settings):
    import lightgbm as lgb
    folder = MODEL/regime
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'summary.json').exists():
        s = json.loads((folder/'summary.json').read_text())
        if s['status'] == 'completed':
            for n, h in s['artifact_sha256'].items():
                if sha(folder/n) != h:raise ValueError('Completed artifact differs')
            return s
    if any((folder/(b+'.txt')).exists() for b in ('relative','auxiliary','classification')):
        raise RuntimeError('Partial model run exists; do not silently reuse or overwrite')
    tr = load(ROOT/contract['roles']['train'][regime]['path'])
    es = load(ROOT/contract['roles']['clean_ES'][regime]['path'])
    assert len(tr['share']) == 60000 and len(es['share']) == 2000
    assert not set(tr['base_window_index']) & set(es['base_window_index'])
    for d in (tr, es):
        assert np.array_equal(d['share'] > 0, d['presence'].astype(bool))
        assert np.isfinite(d['share']).all()
    prior = fit_background_prior(DATA/'train_backgrounds.parquet', folder)
    write(folder/'feature_schema.json', schema())
    write(folder/'PROTOCOL.json', settings)
    predictions, histories, fits = {}, {}, {}
    start = time.time()
    for branch in ('relative', 'auxiliary', 'classification'):
        if branch == 'classification':
            target, valid = tr['presence'], es['presence']
        else:target, valid = tr['share'], es['share']
        began = time.time()
        progress(stage='feature_transform', regime=regime, branch=branch, rows=len(tr['db']))
        x = transform(tr['db'], branch, prior)
        v = transform(es['db'], branch, prior)
        progress(stage='fitting', regime=regime, branch=branch, feature_count=x.shape[1],
                 feature_seconds=time.time()-began, memory_feature_bytes=x.nbytes+v.nbytes)
        args = dict(n_estimators=1500, learning_rate=.035, num_leaves=31,
                    min_child_samples=50, max_bin=127, n_jobs=2,
                    random_state=SEED, verbosity=-1, colsample_bytree=.85,
                    subsample=.85, subsample_freq=1, deterministic=True,
                    force_col_wise=True, data_random_seed=SEED,
                    feature_fraction_seed=SEED, bagging_seed=SEED)
        classifier = branch == 'classification'
        model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', **args) if classifier else lgb.LGBMRegressor(objective='regression_l2', metric='None', **args)
        def clipped_rmse(y, p):
            return 'clipped_rmse', float(np.sqrt(np.mean((np.clip(p, 0, 1)-y)**2))), False
        history = {}
        def callback(env):
            if env.iteration == 0 or (env.iteration+1) % 50 == 0:
                progress(stage='boosting', regime=regime, branch=branch, round=env.iteration+1,
                         branch_elapsed_seconds=time.time()-began,
                         evaluation=[list(a) for a in env.evaluation_result_list])
        callback.order = 30
        model.fit(x, target, eval_set=[(v, valid)],
                  eval_metric='binary_logloss' if classifier else clipped_rmse,
                  callbacks=[lgb.early_stopping(40, verbose=False), lgb.record_evaluation(history), callback])
        booster = model.booster_
        predictions[branch] = np.clip(booster.predict(v, num_threads=2), 0, 1)
        booster.save_model(str(folder/(branch+'.txt')))
        histories[branch] = history
        write(folder/(branch+'_boosting_history.json'), history)
        fits[branch] = dict(best_iteration=int(model.best_iteration_), executed_rounds=len(next(iter(history['valid_0'].values()))),
                            feature_count=x.shape[1], train_feature_bytes=x.nbytes,
                            elapsed_seconds=time.time()-began,
                            clean_ES_rmse=float(np.sqrt(np.mean((predictions[branch]-valid)**2))) if not classifier else None)
        progress(stage='branch_completed', regime=regime, branch=branch, **fits[branch])
        del x, v, model, booster
        gc.collect()
    relative = metrics(es['share'], predictions['relative'], predictions['classification'], site=es['site'])
    auxiliary = metrics(es['share'], predictions['auxiliary'], predictions['classification'], site=es['site'])
    eligible = auxiliary['rmse'] <= relative['rmse']-.0001 and auxiliary['site_macro_rmse'] <= relative['site_macro_rmse']+.0005 and abs(auxiliary['bias']) <= abs(relative['bias'])+.0005
    selected = 'auxiliary' if eligible else 'relative'
    np.savez_compressed(folder/'clean_ES_predictions.npz',
                        sample_id=es['sample_id'], target_share=es['share'], target_presence=es['presence'],
                        site=es['site'], base_window_index=es['base_window_index'],
                        share_relative=predictions['relative'].astype(np.float32),
                        share_auxiliary=predictions['auxiliary'].astype(np.float32),
                        classifier_prob=predictions['classification'].astype(np.float32))
    write(folder/'SELECTION_FROZEN.json', dict(selected_regression=selected, role='clean_ES_only',
          relative=relative, auxiliary=auxiliary, eligible=bool(eligible),
          no_fresh_gate_report_or_Test_used=True, frozen_unix_time=time.time()))
    files=['relative.txt', 'auxiliary.txt', 'classification.txt', 'TRAIN_background_prior.json',
           'feature_schema.json', 'PROTOCOL.json', 'SELECTION_FROZEN.json', 'clean_ES_predictions.npz']
    files += [b+'_boosting_history.json' for b in ('relative','auxiliary','classification')]
    result = dict(format='r18_single_split_production_tree_v1', status='completed', regime=regime,
                  selected_regression=selected, fits=fits, clean_ES_metrics=dict(relative=relative, auxiliary=auxiliary),
                  train_rows=60000, clean_ES_rows=2000, new_KFold=False,
                  data_binding={role: contract['roles'][role][regime] for role in ('train','clean_ES')},
                  code_sha256=settings['source_code_sha256'],
                  artifact_sha256={n: sha(folder/n) for n in files}, elapsed_seconds=time.time()-start,
                  classifier_independent_of_regression=True, Test_used=False)
    write(folder/'summary.json', result)
    return result


def evaluate(contract):
    import pandas as pd
    from ops.run_r16_postprocess import panel_data
    if not (MODEL/'POLICY_FROZEN.json').exists():raise ValueError('Both regimes must be frozen before evaluation')
    rows=[]
    for name in ('fresh_fit_D0', 'fresh_fit_H0', *PANELS):
        if name.startswith('fresh_fit_'):
            r = name.split('_')[-1]
            d = load(ROOT/contract['roles']['fresh_fit'][r]['path']);weights=None
        else:
            d, weights = panel_data(name, ROOT/'models/r16_pretrained_cuda_bf16_r1/both/final/GLOBAL_FREEZE.json')
        preds = {r: predict(MODEL/r, d['db']) for r in ('D0','H0')}
        saved = dict(sample_id=d['sample_id'], target_share=d['share'], target_presence=d['presence'], site=d['site'])
        if weights is not None:saved['weights']=weights
        for r in ('D0','H0'):
            for key,value in preds[r].items():saved[key+'_'+r]=value
        for branch in ('selected','relative','auxiliary'):
            for regime in ('D0','H0','D0_H0_50_50'):
                if regime == 'D0_H0_50_50':
                    p=.5*(preds['D0']['share_'+branch]+preds['H0']['share_'+branch])
                    c=.5*(preds['D0']['classifier_prob']+preds['H0']['classifier_prob'])
                else:p=preds[regime]['share_'+branch];c=preds[regime]['classifier_prob']
                rows.append(dict(panel=name, regime=regime, branch=branch, **metrics(d['share'],p,c,weights,d.get('site'))))
        out=OUT/'predictions';out.mkdir(exist_ok=True)
        np.savez_compressed(out/(name+'.npz'),**saved)
        progress(stage='evaluation_completed',panel=name)
        del d,preds,saved;gc.collect()
    pd.DataFrame(rows).to_csv(OUT/'PANEL_METRICS.csv',index=False)
    combined=pd.DataFrame(rows).query("regime == 'D0_H0_50_50'")
    report='# R18 본체 LightGBM 실측\n\n'
    report+='각 D0/H0에 새 TRAIN 60,000행, clean_ES 2,000행으로 단일분할 학습했다. 새 K-fold는 없다. 독립 분류기는 정확한 profile_energy 2,408특징이며, 회귀는 명시적 마스크·관측count 493개를 제거한 상대 4,667특징과 12개 레벨/TRAIN배경 보조특징을 더한 4,679특징을 비교했다. 실제 음향 lag/correlation 특징은 유지했다.\n\n'
    report+='보조 branch 채택은 오직 clean_ES 사전등록 조건으로 결정했다. TRAIN 배경 템플릿은 원천 분리·purge된 TRAIN 고유 배경에서만 적합했으며, 이후 추론에는 db만 사용했다. 이 템플릿은 추정 보조값이고 물리 상한선이 아니다. 과거 R18 소형 레벨 이득 자체는 조건부 배경 보조의 효용을 입증하지 않는다.\n\n'
    report+=combined[['panel','branch','rmse','bias','classifier_auc','site_macro_rmse']].to_markdown(index=False,floatfmt='.6f')+'\n\n'
    report+='fresh_fit·고정 패널은 동결 이후 진단이며 이 결과로 본 모듈의 회귀 branch나 학습 설정을 바꾸지 않는다. 고정 패널은 과거 반복 검토된 개발 자료이고, 공유 VTUAD hull/day 관계 때문에 엄격한 미관측 참조라고 주장하지 않는다. Test는 읽지 않았다.\n'
    (OUT/'R18_PRODUCTION_TREE_RESULTS.md').write_text(report)
    write(OUT/'COMPLETED.json',dict(status='completed', policy_sha256=sha(MODEL/'POLICY_FROZEN.json'),
          artifacts={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='COMPLETED.json'}, Test_used=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--evaluate-only',action='store_true')
    args=ap.parse_args();sys.addaudithook(guard)
    OUT.mkdir(parents=True,exist_ok=True);MODEL.mkdir(parents=True,exist_ok=True)
    result=preflight();progress(stage='preflight_completed',**result)
    if args.preflight_only:return
    contract=json.loads((DATA/'DATA_CONTRACT.json').read_text())
    validate_roles(contract)
    settings=protocol(contract)
    p=MODEL/'REGISTERED_PROTOCOL.json'
    if p.exists():
        if json.loads(p.read_text())!=settings:raise ValueError('Production tree protocol changed')
    else:write(p,settings)
    if not args.evaluate_only:
        summaries={r:fit_one_regime(r,contract,settings) for r in ('D0','H0')}
        if any(sha(ROOT/n)!=h for n,h in settings['source_code_sha256'].items()):raise ValueError('Source changed during training')
        write(MODEL/'POLICY_FROZEN.json',dict(format='r18_production_tree_policy_v1',
            regression={r:s['selected_regression'] for r,s in summaries.items()}, classifier='profile_energy2408',
            D0_H0=[.5,.5], query_inputs=['db'], Test_used=False, new_KFold=False,
            selected_only_on_clean_ES=True, models={r:sha(MODEL/r/'summary.json') for r in ('D0','H0')},
            code_sha256=settings['source_code_sha256'], data_contract_sha256=sha(DATA/'DATA_CONTRACT.json')))
    evaluate(contract);progress(stage='all_completed')


if __name__=='__main__':main()
