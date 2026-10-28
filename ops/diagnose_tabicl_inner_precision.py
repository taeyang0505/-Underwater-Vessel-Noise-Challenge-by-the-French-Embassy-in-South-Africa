"""Disposable CUDA TabICL numerical diagnostics on train/inner only.

No outer input/label, learned task checkpoint, frozen file or training run is
changed. This OPS helper refits frozen-weight context only, never gradients.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, datetime, gc, hashlib, json, sys, time
sys.dont_write_bytecode = True

BASE = Path('/workspace/underwater_vessel_noise_r1')
PROJECT = BASE / 'project'
OPS = BASE / 'ops'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OPS / ('tabicl_inner_precision_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'))
    a = p.parse_args(); output = a.output.resolve()
    if not output.is_relative_to(OPS) or output.exists():
        raise ValueError('Use a fresh OPS report path')
    blocked = {'outer_validation.npz', 'outer_validation_recipes.json', 'validation_metadata.parquet',
               'test.csv', 'calibration.npz', 'sealed_holdout.npz', 'domain_holdout.npz'}
    observed_dataset_files, refused = set(), []
    def audit(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
            if path.name.lower() in blocked:
                refused.append(str(path)); raise PermissionError('Forbidden evaluation/source input blocked')
            if path.name in {'train.npz', 'validation.npz'}: observed_dataset_files.add(str(path))
    sys.addaudithook(audit)
    sys.path.insert(0, str(PROJECT))
    import numpy as np
    import torch
    from pipeline.cv_foundations import CHECKPOINTS, make_predictor
    from pipeline.mixtures import foundation_features
    fold = PROJECT / 'data/synthetic/group_cv_physics_v2_r1/fold_00'
    gate = json.loads((fold / 'readiness_manifest.json').read_text())
    if gate.get('ready_to_train') is not True or gate.get('challenge_test_accessed') is not False:
        raise ValueError('Verified train/inner package required')
    expected = gate['file_sha256']
    for name in ['train.npz', 'validation.npz']:
        if sha(fold / name) != expected[name]: raise ValueError('Input SHA mismatch')
    filename, checksum = CHECKPOINTS['tabicl']
    checkpoint = PROJECT / 'models/pretrained' / filename
    if sha(checkpoint) != checksum: raise ValueError('Pinned original checkpoint mismatch')
    with np.load(fold / 'train.npz', allow_pickle=False) as z:
        ids = np.random.default_rng(42).choice(len(z['share']), 16000, replace=False)
        X, y = foundation_features(z['db'][ids], version=3), z['share'][ids]
    with np.load(fold / 'validation.npz', allow_pickle=False) as z:
        Q = foundation_features(z['db'], version=3)
    if len(Q) < 3000 or not np.isfinite(X).all() or not np.isfinite(Q).all():
        raise ValueError('Expected finite 3000-row inner features')
    pairs = [[0,1], [16,17], [128,129], [512,513], [1024,1025], [2048,2049], [2998,2999]]
    starts = sorted({min(pair[0] // 256 * 256, len(Q) - 256) for pair in pairs})
    code = {name: sha(PROJECT / 'pipeline' / name) for name in ['cv_foundations.py', 'mixtures.py']}
    report = {'status':'running', 'scope':'train context and inner features only; numerical diagnostic, no score/model selection by labels',
        'settings':{'context':16000,'members':4,'seed':42,'feature_version':3,'threads':4,
                    'batch_sizes':[256,2], 'atol':3e-5,'rtol':1e-5,'pairs':pairs},
        'checkpoint_sha256':checksum, 'context_indices_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),
        'train_sha256':expected['train.npz'],'inner_sha256':expected['validation.npz'],
        'code_sha256':code,'torch_version':torch.__version__, 'GPU':torch.cuda.get_device_name(0),
        'gradient_training':False,'outer_accessed':False,'Test_calibration_sealed_accessed':False,'precisions':{}}
    def save():
        tmp = output.with_suffix('.json.partial')
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False)); tmp.replace(output)
    def compare(first, second):
        delta = np.abs(first.astype('float64') - second.astype('float64'))
        return {'max_abs_delta':float(delta.max()),
                'max_ratio_to_fixed_tolerance':float((delta / (3e-5 + 1e-5 * np.abs(second))).max()),
                'passes_unchanged_tolerance':bool(np.allclose(first,second,atol=3e-5,rtol=1e-5))}
    def call(predictor, features):
        prediction = np.asarray(predictor(features)).reshape(-1)
        if prediction.shape != (len(features),) or not np.isfinite(prediction).all():
            raise ValueError('Nonfinite or malformed prediction')
        return prediction
    torch.set_num_threads(4)
    try:
        for precision in ['auto','fp32']:
            precision_started=time.perf_counter()
            info={'status':'fitting_primary_context','comparisons':[]}; report['precisions'][precision]=info; save()
            torch.manual_seed(42); t=time.perf_counter()
            spec=SimpleNamespace(kind='tabicl',precision=precision,members=4,threads=4,seed=42,device='cuda')
            predictor=make_predictor(spec,X,y,checkpoint)
            info['primary_fit_seconds']=time.perf_counter()-t
            bulk, small = {}, {}
            info['status']='primary_queries'
            for start in starts:
                first=call(predictor,Q[start:start+256]); second=call(predictor,Q[start:start+256])
                bulk[start]=first
                info.setdefault('same_context_bulk_repeats',[]).append({'start':start,**compare(first,second)})
            for pair in pairs:
                first=call(predictor,Q[pair]); second=call(predictor,Q[pair]); small[tuple(pair)]=first
                start=min(pair[0]//256*256,len(Q)-256)
                reference=bulk[start][np.asarray(pair)-start]
                info['comparisons'].append({'positions':pair,'bulk_start':start,'batch256_vs2':compare(reference,first),
                    'small_repeat':compare(first,second),'batch256_values':reference.tolist(),'batch2_values':first.tolist(),
                    'batch256_dtype':str(reference.dtype),'batch2_dtype':str(first.dtype)})
            info['max_batch_delta']=max(v['batch256_vs2']['max_abs_delta'] for v in info['comparisons'])
            info['all_primary_batch_checks_pass']=all(v['batch256_vs2']['passes_unchanged_tolerance'] for v in info['comparisons'])
            info['status']='refitting_fresh_context'; save()
            print(json.dumps({'precision':precision,'primary_max_batch_delta':info['max_batch_delta'],
                              'primary_pass':info['all_primary_batch_checks_pass']}),flush=True)
            del predictor; gc.collect(); torch.cuda.empty_cache()
            torch.manual_seed(42); t=time.perf_counter()
            predictor=make_predictor(spec,X,y,checkpoint)
            info['fresh_fit_seconds']=time.perf_counter()-t
            # Reverse query order: start with batch2 in this fresh context.
            info['fresh_context_query_order']='batch2 first, then batch256; primary context did batch256 first'
            fresh_small={tuple(pair):call(predictor,Q[pair]) for pair in pairs}
            fresh_bulk={start:call(predictor,Q[start:start+256]) for start in starts}
            info['fresh_context_checks']=[]
            for pair in pairs:
                start=min(pair[0]//256*256,len(Q)-256)
                fb=fresh_bulk[start][np.asarray(pair)-start]; fs=fresh_small[tuple(pair)]
                info['fresh_context_checks'].append({'positions':pair,
                    'fresh_batch256_vs2':compare(fb,fs), 'primary_vs_fresh_batch2':compare(small[tuple(pair)],fs),
                    'primary_vs_fresh_batch256':compare(bulk[start][np.asarray(pair)-start],fb)})
            info['all_fresh_checks_pass']=all(c['passes_unchanged_tolerance'] for row in info['fresh_context_checks'] for k,c in row.items() if k!='positions')
            info['status']='completed'; info['seconds_total']=time.perf_counter()-precision_started
            info['float32_matmul_precision']=torch.get_float32_matmul_precision()
            info['allow_tf32']=torch.backends.cuda.matmul.allow_tf32
            del predictor; gc.collect(); torch.cuda.empty_cache(); save()
        for name in ['train.npz','validation.npz']:
            if sha(fold/name)!=expected[name]: raise ValueError('Input changed during diagnostic')
        if sha(checkpoint)!=checksum or any(sha(PROJECT/'pipeline'/k)!=v for k,v in code.items()):
            raise ValueError('Frozen code/original checkpoint changed')
        report['status']='completed'; report['dataset_files_opened']=sorted(observed_dataset_files)
        report['forbidden_open_attempts']=refused; save()
        print(json.dumps({'status':report['status'],'report':str(output),
            'summary':{k:{'max_batch_delta':v['max_batch_delta'],'primary_pass':v['all_primary_batch_checks_pass'],
                          'fresh_pass':v['all_fresh_checks_pass']} for k,v in report['precisions'].items()}},indent=2),flush=True)
    except BaseException as exc:
        report['status']='failed'; report['error']={'type':type(exc).__name__,'message':str(exc)[:2000]}
        report['dataset_files_opened']=sorted(observed_dataset_files); report['forbidden_open_attempts']=refused; save(); raise


if __name__=='__main__': main()
