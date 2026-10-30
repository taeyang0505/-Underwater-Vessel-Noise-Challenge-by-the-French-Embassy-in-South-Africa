#!/usr/bin/env python3
"""CPU frozen-model replay on fresh clean source-separated fit only.

Both trained D0/H0 models see EACH same input, before their predictions are
averaged. Targets of the two generated input panels are never interchanged.
No optimizer, training, query inputs, or policy fitting is used.
"""
from __future__ import annotations
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['DO_NOT_TRACK'] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/r17_mpl')
from pathlib import Path
import datetime
import argparse
import gc
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops.r17_audit_common import sha, load, jsave
from pipeline.training_r16_multisupport import features, tree_features

DATA = ROOT/'data/synthetic/r17_clean_freshfit_v1'
OUT = ROOT/'reports/r17_clean_pipeline_r1/fresh_fit_frozen'
FAMILIES = ('lightgbm5', 'catboost5', 'dasheng_base', 'dinov2_small')
FEATURE_FILES = ('pipeline/training_r16_multisupport.py', 'pipeline/trees_r16_production.py',
                 'pipeline/relative_grid_r6_v2.py', 'pipeline/relative_grid_r6.py',
                 'pipeline/physics_features_r6_v2.py', 'ops/r16_temporal_features.py')


def no_query(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        name = Path(os.fsdecode(args[0])).name.lower()
        if name in ('test.csv', 'sample_submission.csv', 'query_predictions.npz'):
            raise PermissionError('Fresh clean fit frozen inference has no query access')


def progress(**values):
    record = {'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), **values}
    jsave(OUT/'progress.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def feature_cache(panel, data, input_sha):
    directory = OUT/'features'/panel
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory/'receipt.json'
    binding = {'input_sha256': input_sha,
               'source_code_sha256': {f: sha(ROOT/f) for f in FEATURE_FILES}}
    if receipt.exists():
        r = json.loads(receipt.read_text())
        if r['binding'] != binding:
            raise ValueError('Fresh fit feature cache binding changed')
        for file, digest in r['artifact_sha256'].items():
            if sha(directory/file) != digest:
                raise ValueError('Fresh fit feature cache bytes changed')
    else:
        n = len(data['share'])
        arrays = {
            'x': np.lib.format.open_memmap(directory/'x.npy', mode='w+', dtype=np.float32, shape=(n,17,15,25)),
            'w': np.lib.format.open_memmap(directory/'w.npy', mode='w+', dtype=np.float32, shape=(n,15,25)),
            'regression': np.lib.format.open_memmap(directory/'regression.npy', mode='w+', dtype=np.float32, shape=(n,5160)),
            'classification': np.lib.format.open_memmap(directory/'classification.npy', mode='w+', dtype=np.float32, shape=(n,2033)),
        }
        for off in range(0, n, 128):
            sl = slice(off, off+128)
            arrays['x'][sl], arrays['w'][sl] = features(data['db'][sl])
            arrays['regression'][sl] = tree_features(data['db'][sl], 'regression')
            arrays['classification'][sl] = tree_features(data['db'][sl], 'classification')
        for value in arrays.values():
            value.flush()
        del arrays
        jsave(receipt, {'binding': binding,
                        'artifact_sha256': {p.name: sha(p) for p in directory.glob('*.npy')}})
    return {name: np.load(directory/(name+'.npy'), mmap_mode='r')
            for name in ('x', 'w', 'regression', 'classification')}


def metrics(target, share, score, site):
    from sklearn.metrics import roc_auc_score
    target = np.asarray(target, float)
    error = np.asarray(share, float)-target
    site = np.asarray(site, str)
    site_mse = [float(np.mean(error[site == s]**2)) for s in np.unique(site)]
    return {'rows': len(target), 'rmse': float(np.sqrt(np.mean(error**2))),
            'bias': float(np.mean(error)), 'classifier_auc': float(roc_auc_score(target>0, score)),
            'prediction_mean': float(np.mean(share)), 'prediction_rms': float(np.sqrt(np.mean(np.asarray(share)**2))),
            'site_macro_mean_RMSE': float(np.mean(np.sqrt(site_mse))),
            'site_equal_weight_RMSE': float(np.sqrt(np.mean(site_mse))),
            'negative_score_mean': float(np.mean(np.asarray(score)[target == 0])),
            'positive_score_mean': float(np.mean(np.asarray(score)[target > 0])),
            'score_ge090': float(np.mean(np.asarray(score) >= .9))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('all','trees','fold0trees','neural','final'), default='all')
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--neural-batch', type=int, default=32)
    args = parser.parse_args()
    phase = args.phase
    if phase == 'all':
        # Torch and the Homebrew OpenMP-backed tree runtimes are isolated.
        # The first combined-runtime attempt exited 139 before any prediction.
        import subprocess
        for part in ('trees','fold0trees','neural','final'):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--phase', part], check=True)
        return
    if phase in ('trees','fold0trees'):
        import lightgbm as lgb
        from catboost import CatBoostClassifier, CatBoostRegressor
    if phase == 'neural':
        import torch
        from pipeline.training_r16_pretrained_cuda_bf16 import branch, base_fingerprint, CLASSIFIER_CHANNELS
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
    sys.addaudithook(no_query)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = json.loads((DATA/'manifest.json').read_text())
    if manifest['status'] != 'completed' or manifest['all_old_local_fit_source_overlap'] != 0:
        raise ValueError('Fresh fit source separation not confirmed')
    data, cache = {}, {}
    input_pins = {}
    for regime in ('D0', 'H0'):
        panel = 'fresh_fit_'+regime
        entry = manifest['roles']['ensemble_fit_fresh'][regime]
        path = ROOT/entry['path']
        if sha(path) != entry['sha256']:
            raise ValueError('Fresh fit input changed')
        data[panel] = load(path)
        if len(data[panel]['share']) != 2000:
            raise ValueError('Expected 400 backgrounds × 5 examples')
        input_pins[str(path.relative_to(ROOT))] = entry['sha256']
        cache[panel] = feature_cache(panel, data[panel], entry['sha256'])
        progress(phase='feature_cache_complete', panel=panel, rows=2000)
    predictions = {panel: {} if phase in ('trees','fold0trees') else load(OUT/(panel+'_partial_predictions.npz'))
                   for panel in data}
    model_pins = {}
    frozen = json.loads((ROOT/'models/r16_pretrained_cuda_bf16_r1/both/final/GLOBAL_FREEZE.json').read_text())
    source_pins = json.loads((ROOT/'reports/r17_eda_audit_20261030/shortcut/tree_sources_sha256.json').read_text())
    for family in (('lightgbm5', 'catboost5') if phase in ('trees','fold0trees') else ()):
        for regime in ('D0', 'H0'):
            for task in ('regression', 'classification'):
                accum = {panel: np.zeros(2000, float) for panel in data}
                for fold in (range(1) if phase == 'fold0trees' else range(5)):
                    if family == 'lightgbm5':
                        path = ROOT/f'models/r16_multisupport_full_r1/trees/{regime}/fold_{fold:02d}/{task}.txt'
                        model = lgb.Booster(model_file=str(path))
                        if model.num_feature() != (5160 if task == 'regression' else 2033):
                            raise ValueError('LGB schema width differs')
                        predict = lambda a: model.predict(a, num_threads=2)
                    else:
                        path = ROOT/f'models/r16_catboost_peft_cpu8_r1/{regime}/fold_{fold:02d}/{task}.cbm'
                        model = CatBoostRegressor() if task == 'regression' else CatBoostClassifier()
                        model.load_model(str(path))
                        if task == 'regression':
                            predict = lambda a: model.predict(a, thread_count=2)
                        else:
                            predict = lambda a: model.predict_proba(a, thread_count=2)[:,1]
                    digest = sha(path)
                    if digest != source_pins[str(path.relative_to(ROOT))]:
                        raise ValueError('Frozen Tree bytes changed')
                    model_pins[str(path.relative_to(ROOT))] = digest
                    for panel in data:
                        p = np.asarray(predict(cache[panel][task]), float)
                        if task == 'regression':
                            p = np.clip(p, 0, 1)
                        accum[panel] += p/(1. if phase == 'fold0trees' else 5.)
                    del model
                for panel in data:
                    family_key = family.replace('5','_fold0') if phase == 'fold0trees' else family
                    predictions[panel][family_key+'__'+regime+'__'+task] = accum[panel]
                progress(phase='tree_branch_complete', family=family, trained_regime=regime, task=task,
                         elapsed_seconds=time.perf_counter()-started)
    if phase in ('trees','fold0trees'):
        for panel in data:
            name = panel+('_fold0_tree_predictions.npz' if phase == 'fold0trees' else '_partial_predictions.npz')
            np.savez_compressed(OUT/name, **predictions[panel])
        jsave(OUT/('fold0_tree_model_sha256.json' if phase == 'fold0trees' else 'tree_model_sha256.json'), model_pins)
        progress(phase=phase+'_completed', elapsed_seconds=time.perf_counter()-started)
        return
    for family in (('dinov2_small', 'dasheng_base') if phase == 'neural' else ()):
        for regime in ('D0', 'H0'):
            for task in ('regression', 'classification'):
                path = ROOT/f'models/r16_pretrained_cuda_bf16_r1/both/{family}/{regime}/{task}/best.pt'
                digest = sha(path)
                if digest != frozen['artifact_sha256'][str(path.relative_to(ROOT))]:
                    raise ValueError('Frozen neural checkpoint bytes changed')
                done_receipt = OUT/(family+'_'+regime+'_'+task+'_receipt.json')
                output_key = family+'__'+regime+'__'+task
                if done_receipt.exists() and all(output_key in predictions[panel] for panel in data):
                    receipt = json.loads(done_receipt.read_text())
                    if receipt['checkpoint_sha256'] != digest:
                        raise ValueError('Completed branch checkpoint binding changed')
                    if not all(predictions[panel][output_key].shape == (2000,) and
                               np.isfinite(predictions[panel][output_key]).all() for panel in data):
                        raise ValueError('Completed branch prediction dimensions changed')
                    model_pins[str(path.relative_to(ROOT))] = digest
                    progress(phase='verified_completed_branch_no_replay', family=family,
                             trained_regime=regime, task=task)
                    continue
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                if (checkpoint['family'], checkpoint['regime'], checkpoint['task']) != (family, regime, task):
                    raise ValueError('Wrong neural checkpoint identity')
                model, provenance = branch(family, task, 'cpu')
                if base_fingerprint(model) != checkpoint['permanent_base_fingerprint']:
                    raise ValueError('Permanent pretrained base fingerprint differs')
                model.load_adapter_state(checkpoint['adapter_state'])
                model.eval()
                model_pins[str(path.relative_to(ROOT))] = digest
                branch_start = time.perf_counter()
                with torch.inference_mode():
                    for panel in data:
                        parts = []
                        for off in range(0, 2000, args.neural_batch):
                            xx = torch.from_numpy(np.asarray(cache[panel]['x'][off:off+args.neural_batch]).copy())
                            if task == 'classification':
                                xx = xx[:, CLASSIFIER_CHANNELS]
                            value = model(xx).float().sigmoid().numpy()
                            if task == 'regression':
                                value = (value*np.asarray(cache[panel]['w'][off:off+args.neural_batch])).sum((1,2))
                            parts.append(np.asarray(value).reshape(-1))
                            if off and off % 512 == 0:
                                progress(phase='frozen_neural_forward', family=family, trained_regime=regime,
                                         task=task, panel=panel, rows=off, total=2000,
                                         elapsed_seconds=time.perf_counter()-started)
                        predictions[panel][family+'__'+regime+'__'+task] = np.concatenate(parts)
                if base_fingerprint(model) != checkpoint['permanent_base_fingerprint']:
                    raise ValueError('Permanent base mutated during inference')
                jsave(OUT/(family+'_'+regime+'_'+task+'_receipt.json'), {
                    'checkpoint_sha256': digest, 'permanent_base_fingerprint_verified_before_and_after': True,
                    'CPU_precision': 'FP32', 'original_training_precision': 'CUDA BF16',
                    'no_strict_numerical_equivalence_claimed': True, 'training_steps': 0,
                    'provenance': provenance, 'seconds': time.perf_counter()-branch_start,
                    'both_input_panels_rows': 4000, 'Test_used': False,
                    'CPU_threads': args.cpu_threads, 'neural_batch': args.neural_batch,
                })
                for panel in data:
                    np.savez_compressed(OUT/(panel+'_partial_predictions.npz'), **predictions[panel])
                progress(phase='neural_branch_complete', family=family, trained_regime=regime, task=task,
                         seconds=time.perf_counter()-branch_start, elapsed_seconds=time.perf_counter()-started)
                del model, checkpoint
                gc.collect()
    if phase == 'neural':
        jsave(OUT/'neural_model_sha256.json', model_pins)
        progress(phase='neural_completed', elapsed_seconds=time.perf_counter()-started)
        return
    model_pins = {**json.loads((OUT/'tree_model_sha256.json').read_text()),
                  **json.loads((OUT/'neural_model_sha256.json').read_text()),
                  **json.loads((OUT/'fold0_tree_model_sha256.json').read_text())}
    for panel in data:
        predictions[panel].update(load(OUT/(panel+'_fold0_tree_predictions.npz')))
    evaluated_families = FAMILIES + ('lightgbm_fold0','catboost_fold0')
    records, sites = [], []
    averaged = {}
    for panel, d in data.items():
        averaged[panel] = {}
        for family in evaluated_families:
            p = .5*(predictions[panel][family+'__D0__regression']+predictions[panel][family+'__H0__regression'])
            c = .5*(predictions[panel][family+'__D0__classification']+predictions[panel][family+'__H0__classification'])
            averaged[panel][family] = (p, c)
        for blend, weights in {'B1': [.6,0,.15,.25], 'B2': [.5,0,.2,.3]}.items():
            averaged[panel][blend] = tuple(sum(w*averaged[panel][family][j] for w, family in zip(weights, FAMILIES))
                                           for j in (0,1))
        for blend, weights in {'B1_LGBFold0': [.6,.15,.25], 'B2_LGBFold0': [.5,.2,.3]}.items():
            averaged[panel][blend] = tuple(sum(w*averaged[panel][family][j]
                for w, family in zip(weights, ('lightgbm_fold0','dasheng_base','dinov2_small')))
                for j in (0,1))
        for family, (p,c) in averaged[panel].items():
            records.append({'input_panel': panel, 'model': family, 'trained_regime_mix': 'D0_H0_50_50',
                            'unique_source_backgrounds': 400, **metrics(d['share'],p,c,d['site'])})
            for site in np.unique(d['site']):
                use = d['site'] == site
                sites.append({'input_panel': panel, 'model': family, 'site': str(site),
                              **metrics(d['share'][use],p[use],c[use],d['site'][use])})
        np.savez_compressed(OUT/(panel+'_predictions.npz'),
                            **predictions[panel], target_share=d['share'], site=d['site'],
                            base_window_index=d['base_window_index'], sample_id=d['sample_id'])
    for family in averaged['fresh_fit_D0']:
        p = np.concatenate([averaged[panel][family][0] for panel in data])
        c = np.concatenate([averaged[panel][family][1] for panel in data])
        y = np.concatenate([data[panel]['share'] for panel in data])
        s = np.concatenate([data[panel]['site'] for panel in data])
        records.append({'input_panel': 'D0_and_H0_generated_inputs_pooled', 'model': family,
                        'trained_regime_mix': 'D0_H0_50_50', 'unique_source_backgrounds': 400,
                        **metrics(y,p,c,s)})
    frame = pd.DataFrame(records)
    frame['evaluation_scope'] = np.select([
        frame.model.isin(['lightgbm5','catboost5','B1','B2']),
        frame.model.eq('catboost_fold0')], [
        'secondary_legacy5fold_TRAIN_source_exposure',
        'secondary_CatFold0_earlystop_source_exposure'],
        default='primary_Fold0_TRAIN_and_checkpoint_background_unexposed')
    frame.to_csv(OUT/'frozen_models_metrics.csv', index=False)
    compact = frame.rename(columns={'input_panel': 'dataset', 'classifier_auc': 'auc',
                                    'site_macro_mean_RMSE': 'site_macro_rmse'})
    compact['geometry'] = compact.dataset.map({'fresh_fit_D0':'D0_generated_input',
        'fresh_fit_H0':'H0_generated_input',
        'D0_and_H0_generated_inputs_pooled':'pooled_both_inputs_same400_backgrounds'})
    compact.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(sites).to_csv(OUT/'frozen_models_site_metrics.csv', index=False)
    (OUT/'FRESH_FIT_FROZEN_MODELS.md').write_text(
        '# 새 클린 fit에서 동결된 기존 Tree·신경망 비교\n\n'
        '재학습·정책 적합·Test 접근 0회. 각 입력 패널 D0/H0의 정답은 따로 유지하고, '
        '각각의 **동일 db**를 학습된 D0 모델과 H0 모델 모두에 넣어 예측을 50:50 평균했다. '
        '생성 패널끼리 서로 다른 정답을 평균하여 채점하지 않았다.\n\n'
        + frame.to_markdown(index=False, floatfmt='.6f') + '\n\n'
        '각 입력 패널은 400개 고유 배경 × 5개 합성 예제로 2,000행이다. '
        '두 패널을 합쳐도 같은 400개 원천 배경을 공유하므로 4,000개의 독립 관측이 아니다. '
        '모든 새 fit 배경은 기존 local_fit과 원천·그룹 중복 0이며 24h purge 검사를 통과했다. '
        '단, 이 출처는 Fold0의 CV 풀에서 선택했으므로 기존 Tree의 Fold1~4 TRAIN에 '
        '포함되었다. CatFold0의 기존 own-CV 조기 종료셋과도 55개 배경이 직접 겹치며 '
        '400개 전체가 같은 배경 그룹 및 24h 이내에 있다(별도 LEGACY_EXPOSURE_AUDIT 참조). '
        '따라서 **단일 Fold0 LGB/Dash/DINO와 LGBFold0 기반 B1/B2**를 배경 비노출 '
        '주 비교로 취급하고, CatFold0 및 기존 LGB5/Cat5와 원래 B1/B2는 별도 보조 통계로 표시했다. '
        '확대된 465개 참조는 새 학습·검증에서 공유되므로 미관측 참조 평가로 부르지 않는다. '
        '기존 신경망은 60개, 기존 5-Fold Tree의 합집합은 75개 원음으로 학습된 동결 모델이다.\n\n'
        '신경망은 CPU FP32로 새로 추론했다. 과거 제출은 CUDA BF16이어서 수치적 동일성을 주장하지 않는다. '
        '통계량은 새 fit에서의 기술적 검증이며, 기존 Public 점수의 원인을 인과적으로 확정하지 않는다.\n',
        encoding='utf-8')
    outputs = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ('COMPLETED.json','progress.json')]
    jsave(OUT/'COMPLETED.json', {'status':'completed', 'training_steps':0, 'Test_used':False,
          'models':list(evaluated_families), 'fixed_blends':['B1','B2','B1_LGBFold0','B2_LGBFold0'], 'frozen_neural_branches':8,
          'input_rows_per_panel':2000, 'unique_source_backgrounds':400,
          'fresh_data_manifest_sha256':sha(DATA/'manifest.json'), 'input_sha256':input_pins,
          'frozen_model_sha256':model_pins, 'code_sha256':sha(Path(__file__)),
          'artifact_sha256':{str(p.relative_to(OUT)):sha(p) for p in outputs},
          'final_aggregation_seconds':time.perf_counter()-started,
          'neural_branch_forward_seconds_total':sum(json.loads(p.read_text())['seconds']
              for p in OUT.glob('*_receipt.json')),
          'runtime_isolation':'Tree and neural replay executed in separate CPU Python processes',
          'first_combined_runtime_attempt_exit139_cause_confirmed':False})
    progress(phase='completed', elapsed_seconds=time.perf_counter()-started)


if __name__ == '__main__':
    main()
