"""Paired Fold-0 INNER diagnostic; no fitting, OUTER, Test or GPU access."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np
from pipeline.fold_data_r5 import FoldDataR5
from pipeline.grid_inputs_r5 import file_sha256, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def metrics(y, p):
    error = p.astype('float64') - y.astype('float64')
    return {'n': len(y), 'rmse': float(np.sqrt(np.mean(error ** 2))),
            'mae': float(np.mean(np.abs(error))),
            'bias_pred_minus_truth': float(error.mean()),
            'target_mean': float(y.mean()), 'prediction_mean': float(p.mean())}


def main():
    started = time.perf_counter()
    output = ROOT / 'reports/R5_CNN_STARTER_PAIRED_INNER'
    if output.exists():
        raise FileExistsError('Fresh diagnostic output required')
    folder = ROOT / 'data/synthetic/group_cv_passage_r4_v1/fold_00'
    model = ROOT / 'ops/server_backups/r5_gpu_r1/restored_first_cnn_v1/fold_00/preprocessing_pilot/raw/bg_cnn'
    context_file = model.parents[2] / 'context.json'
    context = json.loads(context_file.read_text())
    summary = json.loads((model / 'summary.json').read_text())
    gate = json.loads((folder / 'readiness_manifest.json').read_text())
    code_hashes = context['code_sha256']
    for name, expected in code_hashes.items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError('Prediction writer/model code changed: ' + name)
    if summary['status'] != 'completed' or summary['outer_or_Test_accessed']:
        raise ValueError('Expected completed TRAIN/INNER-only CNN')
    for name, key in [('best.pt', 'checkpoint_sha256'), ('inner_predictions.npz', 'prediction_sha256')]:
        if file_sha256(model / name) != summary[key]:
            raise ValueError('Verified backup model artifact differs')
    allowed_data = {folder / name for name in ['readiness_manifest.json', 'sources/manifest.json',
                    'train.npz', 'validation.npz', 'inner_validation_metadata.parquet']}
    allowed_data = {p.resolve() for p in allowed_data}
    data_root = (ROOT / 'data').resolve()

    def audit(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'}:
            raise RuntimeError('No external actions in paired diagnostic')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(data_root):
                if path not in allowed_data:
                    raise ValueError('Non-INNER array or undeclared data access: ' + str(path))
                if args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                    raise ValueError('Source writes forbidden')
    sys.addaudithook(audit)
    fold = FoldDataR5(folder, 42)
    if fold.binding != context['training_binding'] or fold.binding != summary['training_binding']:
        raise ValueError('CNN and current dataset binding differs')
    data, md = fold.load('inner')
    with np.load(model / 'inner_predictions.npz', allow_pickle=False) as z:
        prediction = {k: z[k] for k in ['share', 'score', 'cell_mask']}
    y = data['share'].astype('float64')
    ids = md.sample_id.astype(str).to_numpy(dtype=str)
    p = prediction['share'].astype('float64')
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid CNN prediction')
    comparators = {'original_cnn': p}
    inputs = {}
    for label, directory in [('official_starter', 'R5_OFFICIAL_STARTER_TRAIN_INNER'),
                             ('our_train_mean', 'R5_TRAIN_MEAN_BASELINE_TRAIN_INNER')]:
        path = ROOT / 'reports' / directory / 'fold_00_inner_predictions.npz'
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(ids, z['sample_id']):
                raise ValueError('Baseline row IDs differ from CNN fixed query order')
            comparators[label] = z['share'].astype('float64')
        inputs[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    obs = np.isfinite(data['db'])
    peak = np.nanmax(data['db'], axis=(1, 2), keepdims=True)
    energy = np.where(obs, 10 ** ((data['db'].astype('float64') - peak) / 10), 0)
    w = energy / energy.sum((1, 2), keepdims=True)
    pool_delta = float(np.max(np.abs((w * prediction['cell_mask']).sum((1, 2)) - p)))
    if pool_delta > 3e-6:
        raise ValueError('CNN masks do not reproduce physical pooling')
    if abs(metrics(y, p)['rmse'] - summary['best_inner_rmse']) > 1e-12:
        raise ValueError('Stored best INNER RMSE does not replay')

    def group_result(take):
        yy = y[take]
        if not len(yy):
            return {'n': 0}
        row = {k: metrics(yy, v[take]) for k, v in comparators.items()}
        cnn_mse = np.mean((p[take] - yy) ** 2)
        row['paired_mean_delta_MSE_cnn_minus_baseline'] = {
            k: float(cnn_mse - np.mean((v[take] - yy) ** 2))
            for k, v in comparators.items() if k != 'original_cnn'}
        return row

    bins = {'y=0': y == 0, '0<y<=.1': (y > 0) & (y <= .1),
            '.1<y<=.25': (y > .1) & (y <= .25), '.25<y<=.5': (y > .25) & (y <= .5),
            '.5<y<=.75': (y > .5) & (y <= .75), '.75<y<=.95': (y > .75) & (y <= .95),
            '.95<y<=1': y > .95}
    groups = {}
    for field in ['site', 'reference_group', 'background_group']:
        values = md[field].fillna('<missing>').astype(str).to_numpy()
        groups[field] = {name: group_result(values == name) for name in sorted(set(values))}
    report = {'format': 'r5_cnn_starter_paired_inner_v1', 'diagnostic_only': True,
              'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'fold': 0, 'role': 'inner_early_stopping', 'overall': group_result(np.ones(len(y), bool)),
              'by_target_bin': {k: group_result(v) for k, v in bins.items()}, 'by_group': groups,
              'alignment': {'cnn_file_contains_sample_ids': False,
                            'proof': 'Immutable FoldDataR5 INNER SHA and metadata ordering; SHA-pinned predict_neural uses consecutive slices and concatenation without shuffle; saved best predictions replay recorded RMSE.',
                            'query_file_sha256': file_sha256(folder / 'validation.npz'),
                            'expected_query_file_sha256': gate['file_sha256']['validation.npz'],
                            'metadata_file_sha256': file_sha256(folder / 'inner_validation_metadata.parquet'),
                            'sample_id_order_sha256': canonical_sha256(ids.tolist()),
                            'training_binding': fold.binding},
              'model': {'best_epoch': summary['best_epoch'], 'epochs_run': summary['epochs_run'],
                        'checkpoint_sha256': summary['checkpoint_sha256'],
                        'prediction_sha256': summary['prediction_sha256']},
              'postprocessing_diagnostic': {'method': 'clip(prediction,0,1)', 'fitted': False,
                                            'changed_predictions': int(np.count_nonzero(p != np.clip(p, 0, 1))),
                                            'rmse_after_clip': metrics(y, np.clip(p, 0, 1))['rmse'],
                                            'prediction_min': float(p.min()), 'prediction_max': float(p.max()),
                                            'mask_pool_max_absolute_error': pool_delta},
              'fitting_performed': False, 'outer_Test_or_audio_accessed': False,
              'official_submission_performed': False, 'LB_score_forecast': None,
              'code_sha256': code_hashes, 'baseline_prediction_inputs_sha256': inputs,
              'elapsed_seconds': time.perf_counter() - started}
    output.mkdir()
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    np.savez_compressed(output / 'paired_errors.npz', sample_id=ids, truth=y,
                        **comparators, **{k + '_squared_error': (v - y) ** 2 for k, v in comparators.items()})
    overall = report['overall']
    lines = ['# R5 원본 CNN과 공식 starter: 동일 INNER 샘플 비교', '',
             'Mac CPU에서 Fold 0의 동일한 INNER 3,000개를 비교했다. CNN은 저장된 최적 epoch 19의 예측을 사용했다. 재학습·후처리 파라미터 피팅·OUTER/Test 접근은 하지 않았다.', '',
             '| 방법 | RMSE | MAE | 편향(예측−정답) |', '|---|---:|---:|---:|']
    labels = {'official_starter': '공식 starter: 항상 0.1', 'our_train_mean': '별도 대조군: TRAIN 평균', 'original_cnn': '원본 Wide BG Mask CNN'}
    for key in ['official_starter', 'our_train_mean', 'original_cnn']:
        m = overall[key]
        lines.append(f"| {labels[key]} | {m['rmse']:.6f} | {m['mae']:.6f} | {m['bias_pred_minus_truth']:+.6f} |")
    lines += ['', '## 정답 구간별 CNN', '', '| 정답 구간 | 개수 | RMSE | 편향 |', '|---|---:|---:|---:|']
    for key, row in report['by_target_bin'].items():
        if 'original_cnn' in row:
            m = row['original_cnn']
            lines.append(f"| {key} | {m['n']} | {m['rmse']:.6f} | {m['bias_pred_minus_truth']:+.6f} |")
    lines += ['', '## 해석', '',
              '공식 notebook cell 6은 학습 없이 share=0.1, score=0.5를 예측한다. 공개 Benchmark 0.753867825가 이 알고리즘에서 나왔다는 공식 연결은 확인되지 않았다. 따라서 이를 Benchmark 재현이라고 부르지 않는다.', '',
              '상수 예측의 MSE = Var(y) + (mean(y) − constant)²이다. 합성 TRAIN/INNER의 share 평균이 약 0.382라서 0.1 고정 예측에는 큰 음의 편향이 생긴다. TRAIN 평균 대조군은 이 효과를 분리한다.', '',
              'CNN은 같은 INNER에서 상수 기준선보다 잘 예측한다. 하지만 이 INNER는 early stopping과 전처리 선택에 사용되어 선택 편향이 있으며, 이 비교는 공식 Test의 대표성을 증명하지 않는다. 단순 RMSE 비율·점수 차이로 LB를 보정할 수 없다.', '',
              f"[0,1] clipping은 {report['postprocessing_diagnostic']['changed_predictions']}개를 변경하여 RMSE가 그대로다. 마스크와 관측 에너지 가중합을 재계산한 최대 차이는 {pool_delta:.3g}이다. 추가 보정 후보는 INNER의 원천 그룹 교차 검증으로 identity 대비 이득을 검증하고 OUTER를 열기 전에 고정해야 한다.", '',
              'CNN 예측 파일에는 sample_id가 없다. SHA로 고정된 입력 파일·메타데이터 순서, 연속 배치 예측 후 순서대로 연결하는 코드, 기록 RMSE 재현을 함께 확인하여 대응시켰다.', '',
              '고에너지 선박 통과 사례를 값이 크다는 이유만으로 제거하지 않았다. 공식 제출이나 현재 GPU 학습 설정 변경은 하지 않았다.', '',
              '[전체 paired 지표·사이트/원천별 진단](metrics.json) · [원형 starter 5fold 결과](../R5_OFFICIAL_STARTER_TRAIN_INNER/README.md) · [TRAIN 평균 대조군](../R5_TRAIN_MEAN_BASELINE_TRAIN_INNER/README.md)', '']
    (output / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'overall': overall, 'postprocessing': report['postprocessing_diagnostic'],
                      'output': str(output), 'seconds': report['elapsed_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
