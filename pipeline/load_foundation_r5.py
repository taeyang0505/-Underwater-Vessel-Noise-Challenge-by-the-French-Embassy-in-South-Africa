"""Pinned public R5 checkpoint loader; local-only unless --download is explicit.

No HF implicit credentials or from_pretrained remote-code execution are used.
Official reviewed source files are verified before local import. Model tensors
are constructed on meta, then assigned from safetensors. CPU smoke runs use
synthetic grids only and instantiate exactly one large object per process.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import types
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT/'models/pretrained_r5'
SPECS = {
    'timemoe_public': {
        'model_id': 'Maple728/TimeMoE-200M', 'revision': '794591bfeb1225fdf742cec0f4c71f20c3f3b87e',
        'family': 'timemoe_public_200m_active_453m_total',
        'weight_bytes': 906450104,
        'weight_sha256': '28d1cbf480c1f63d4ad56d8b7649d9860b9afa9f31abeedff9decea48d9aee57',
        'files': {
            'README.md': 'c0ab1a3140946b09c2e10f62194eda89db391bde05576fbfcf664f703829f1f6',
            'config.json': 'cd582cc3dd64decc15c088735d18dcae00ad2bdc013ec6a814af7999e16952bd',
            'configuration_time_moe.py': 'd9b2e91a18fcc79753af7504a93382d43fa87d168318db01eb8431ef8d1b1125',
            'modeling_time_moe.py': 'afd34fc6799799ef7784ae55425da7e58f4dabe49f2873744af201c06ebcfae6',
            'ts_generation_mixin.py': 'b8ed9708bb450dda5443093ce46fa84890ae119823bd897e7f86c4f50dd2814d',
        },
        'configuration_module': 'configuration_time_moe', 'configuration_class': 'TimeMoeConfig',
        'modeling_module': 'modeling_time_moe', 'model_class': 'TimeMoeForPrediction',
        'license': 'Apache-2.0',
        'license_repository': 'Time-MoE/Time-MoE',
        'license_revision': '915bfda4c78a544d62a2bec6ab22948423059236',
        'license_sha256': '7cd9951c712ed85dc4e03d25fff036a925ee7d1a0fceb5aa42a7e8788b40eac9',
    },
    'dasheng': {
        'model_id': 'mispeech/dasheng-1.2B', 'revision': '4ae6b990f87bf40579b97bab065f4e47920b28dc',
        'family': 'dasheng_1.2b', 'weight_bytes': 4536241360,
        'weight_sha256': 'b3b9316fe0ea051eefafff51ee2341629914b8db5f78cbe991803dc86f5669df',
        'files': {
            'README.md': 'cdc8c6a1b114b22202a2c140c8daae3caf37cab151750082cc1c558036408855',
            'config.json': '16d0439c565428d2de2bc2cd64b734bbc886a6d367d16652a4080d1b19902913',
            'configuration_dasheng.py': '7a524b9d37c0e0392b75bbfb0fe5e2d1f60f62a8f2c46917bb186b638580966b',
            'modeling_dasheng.py': '480d07bf805abae43ebea7f32d1ea321041ab17266329d1eac999da12dec3a8e',
        },
        'configuration_module': 'configuration_dasheng', 'configuration_class': 'DashengConfig',
        'modeling_module': 'modeling_dasheng', 'model_class': 'DashengModel',
        'license': 'Apache-2.0',
        'license_repository': 'RicherMans/Dasheng',
        'license_revision': '91bc15c728f08e8d10f741311d8afc42fb33d497',
        'license_sha256': '249c4ef6ef22f69a8a1076d4eb04eb8560af2d4fc48440749f51477efacbaa8f',
    },
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4*1024**2), b''): h.update(data)
    return h.hexdigest()


def model_directory(kind, cache_dir=DEFAULT_CACHE):
    return Path(cache_dir)/SPECS[kind]['model_id'].replace('/', '--')


def _url(spec, name):
    return 'https://huggingface.co/'+spec['model_id']+'/resolve/'+spec['revision']+'/'+name


def _download(url, destination, expected_sha, expected_bytes=None):
    """Public anonymous GET; never reads an HF token or a private source."""
    destination = Path(destination)
    if destination.exists():
        if sha(destination) != expected_sha: raise RuntimeError('Existing artifact has unexpected SHA; refusing overwrite: '+destination.name)
        if expected_bytes is not None and destination.stat().st_size != expected_bytes: raise RuntimeError('Existing size mismatch')
        return
    partial = destination.with_name(destination.name+'.part')
    if partial.exists(): raise RuntimeError('Partial artifact already exists; inspect it explicitly before retry: '+partial.name)
    req = urllib.request.Request(url, headers={'User-Agent': 'underwater-r5-public-checkpoint-audit'})
    h = hashlib.sha256(); n = 0; last = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=90) as response, partial.open('xb') as f:
        while True:
            data = response.read(4*1024**2)
            if not data: break
            f.write(data); h.update(data); n += len(data)
            if n-last >= 256*1024**2:
                print(json.dumps({'download': destination.name, 'bytes': n}), flush=True); last = n
    if h.hexdigest() != expected_sha or (expected_bytes is not None and n != expected_bytes):
        raise RuntimeError('Downloaded artifact SHA/size mismatch; partial preserved, never imported')
    partial.rename(destination)


def fetch_public_checkpoint(kind, cache_dir=DEFAULT_CACHE):
    spec = SPECS[kind]; directory = model_directory(kind, cache_dir)
    for name, expected in spec['files'].items(): _download(_url(spec, name), directory/name, expected)
    _download(_url(spec, 'model.safetensors'), directory/'model.safetensors', spec['weight_sha256'], spec['weight_bytes'])
    license_info = _license_info(spec)
    _download(license_info['url'], directory/'LICENSE', spec['license_sha256'])
    lp = directory/'license_receipt.json'
    if lp.exists():
        if json.loads(lp.read_text()) != license_info: raise RuntimeError('Existing license receipt differs from pin; refusing overwrite')
    else: lp.write_text(json.dumps(license_info, indent=2)+'\n')
    return verify_public_checkpoint(kind, cache_dir)


def _license_info(spec):
    return {'repository': spec['license_repository'], 'revision': spec['license_revision'],
            'url': 'https://raw.githubusercontent.com/'+spec['license_repository']+'/'+spec['license_revision']+'/LICENSE',
            'sha256': spec['license_sha256'], 'license': spec['license']}


def verify_public_checkpoint(kind, cache_dir=DEFAULT_CACHE):
    spec = SPECS[kind]; directory = model_directory(kind, cache_dir)
    for name, expected in spec['files'].items():
        path = directory/name
        if not path.is_file() or path.is_symlink() or sha(path) != expected: raise RuntimeError('Missing/changed pinned source/config: '+name)
    path = directory/'model.safetensors'
    if not path.is_file() or path.is_symlink() or path.stat().st_size != spec['weight_bytes'] or sha(path) != spec['weight_sha256']:
        raise RuntimeError('Exact public pretrained safetensors unavailable or hash mismatch')
    # Separately fetched official repository LICENSE is provenance, not inferred
    # from a third-party mirror or a dataset's terms.
    license_receipt = directory/'license_receipt.json'
    if not license_receipt.is_file(): raise RuntimeError('Pinned official license receipt required')
    license_info = json.loads(license_receipt.read_text())
    if license_info != _license_info(spec) or sha(directory/'LICENSE') != spec['license_sha256']:
        raise RuntimeError('Official license snapshot mismatch')
    return {'kind': kind, 'model_id': spec['model_id'], 'revision': spec['revision'],
            'weight_bytes': spec['weight_bytes'], 'weight_sha256': spec['weight_sha256'],
            'source_sha256': spec['files'], 'license': license_info,
            'credential_policy': 'anonymous urllib; no HF token lookup; no from_pretrained',
            'verified': True}


def _official_runtime(cache_dir):
    # Transformers4.40 imports perform a cache migration check even though the
    # reviewed loader never calls from_pretrained. Keep that IO in this project.
    os.environ['HF_HOME'] = str(Path(cache_dir)/'hf_cache')
    os.environ['HF_HUB_CACHE'] = str(Path(cache_dir)/'hf_cache/hub')
    os.environ['TRANSFORMERS_CACHE'] = str(Path(cache_dir)/'hf_cache/transformers')
    runtime = Path(cache_dir)/'runtime'
    if 'transformers' in sys.modules:
        if sys.modules['transformers'].__version__ != '4.40.1': raise RuntimeError('Run this loader in its separate pinned Transformers4.40.1 process')
    elif runtime.is_dir(): sys.path.insert(0, str(runtime))
    import transformers
    if transformers.__version__ != '4.40.1': raise RuntimeError('Pinned Transformers4.40.1 runtime required; never silently patch vendor API')
    return transformers.__version__


def _import_pinned_module(directory, package, name):
    if package not in sys.modules:
        parent = types.ModuleType(package); parent.__path__ = [str(directory)]; sys.modules[package] = parent
    fullname = package+'.'+name
    if fullname in sys.modules: return sys.modules[fullname]
    spec = importlib.util.spec_from_file_location(fullname, directory/(name+'.py'))
    module = importlib.util.module_from_spec(spec); sys.modules[fullname] = module; spec.loader.exec_module(module)
    return module


def load_public_foundation(kind, cache_dir=DEFAULT_CACHE, *, dtype='float32', adapt_router=False):
    """Return (grid_adapter,receipt), loading exactly one verified public object.

    CPU defaultFP32 is explicit. Caller may later move this wrapper to a GPU;
    this loader does not probe CUDA/MPS or make that move itself.
    """
    if dtype not in {'float32', 'bfloat16', 'float16'}: raise ValueError('Unsupported explicit load dtype')
    receipt = verify_public_checkpoint(kind, cache_dir)
    receipt['transformers'] = _official_runtime(cache_dir)
    import torch
    from safetensors.torch import load_file
    from pipeline.foundation_masks_r5 import PretrainedEvidence, TimeMoEPublicMask, Dasheng12BMask
    spec = SPECS[kind]; directory = model_directory(kind, cache_dir)
    package = 'r5_verified_'+kind+'_'+spec['revision'][:12]
    cm = _import_pinned_module(directory, package, spec['configuration_module'])
    mm = _import_pinned_module(directory, package, spec['modeling_module'])
    cfg = getattr(cm, spec['configuration_class'])(**json.loads((directory/'config.json').read_text()))
    if kind == 'timemoe_public': cfg._attn_implementation = 'eager'
    with torch.device('meta'):
        backbone = getattr(mm, spec['model_class'])(cfg)
    tensors = load_file(str(directory/'model.safetensors'), device='cpu')
    tensor_parameters = sum(t.numel() for t in tensors.values())
    native_dtypes = sorted({str(t.dtype) for t in tensors.values()})
    backbone.load_state_dict(tensors, strict=True, assign=True)
    del tensors; gc.collect()
    if kind == 'timemoe_public':
        # Nonpersistent official RoPE caches are absent from the checkpoint.
        # Reconstruct just those caches with the pinned official constructor.
        for layer in backbone.model.layers:
            old = layer.self_attn.rotary_emb
            layer.self_attn.rotary_emb = type(old)(old.dim, max_position_embeddings=old.max_position_embeddings,
                                                  base=old.base, device='cpu')
    remaining_meta = [n for n, t in list(backbone.named_parameters())+list(backbone.named_buffers()) if t.is_meta]
    if remaining_meta: raise RuntimeError('Unmaterialized official tensors remain: '+str(remaining_meta[:5]))
    backbone = backbone.to(device='cpu', dtype=getattr(torch, dtype))
    evidence = PretrainedEvidence(spec['family'], _url(spec, 'model.safetensors'), spec['revision'], spec['weight_sha256'], True)
    adapter = (TimeMoEPublicMask(backbone, evidence=evidence, adapt_router=adapt_router)
               if kind == 'timemoe_public' else Dasheng12BMask(backbone, evidence=evidence))
    receipt.update({'checkpoint_tensor_elements': tensor_parameters, 'native_tensor_dtypes': native_dtypes,
                    'load_dtype': dtype, 'strict_assign_load': True, 'meta_then_assign': True,
                    'adapter_manifest': adapter.adapter_manifest(), 'adapter_counts': adapter.trainable_parameter_counts(),
                    'is_Ultra_2_4B': False, 'official_source_edited': False,
                    'loader_sha256': sha(Path(__file__)),
                    'adapter_module_sha256': sha(ROOT/'pipeline/foundation_masks_r5.py')})
    return adapter, receipt


def _parameter_digest(model, *, frozen_only):
    h = hashlib.sha256(); count = 0
    for name, p in model.named_parameters():
        if frozen_only and p.requires_grad: continue
        h.update(name.encode()+b'\0'); h.update(str((tuple(p.shape), str(p.dtype))).encode())
        h.update(p.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()); count += p.numel()
    return h.hexdigest(), count


def cpu_smoke(kind, cache_dir=DEFAULT_CACHE, threads=2, smoke_tag=''):
    """One synthetic window+one adapter SGD step, no real examples/labels."""
    # Do not let an unexpected library dependency perform network IO at load.
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    def no_network(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo'}: raise RuntimeError('Network access forbidden during actual-object CPU smoke')
    sys.addaudithook(no_network)
    import torch
    if smoke_tag and (len(smoke_tag) > 32 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in smoke_tag)):
        raise ValueError('Use a short safe smoke tag to preserve earlier results')
    torch.set_num_threads(threads); torch.manual_seed(20261025)
    begin = time.time(); model, receipt = load_public_foundation(kind, cache_dir)
    loaded = time.time(); before, frozen_count = _parameter_digest(model, frozen_only=True)
    x = torch.randn(1, 8, 15, 25, dtype=torch.float32)
    w = torch.rand(1, 15, 25); w[:, [1, 7]] = 0; w[:, :, 0] = 0
    w /= w.sum((1, 2), keepdim=True); old_w = w.clone(); w.requires_grad_(True)
    model.train(); optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.0001)
    out = model(x, w); forward = time.time()
    share, presence, cell = out
    assert share.shape == presence.shape == (1,) and cell.shape == (1, 15, 25)
    assert all(torch.isfinite(t).all() for t in out) and (cell[w == 0] == 0).all()
    reconstructed = (cell.float()*w.detach()).sum((1, 2))
    torch.testing.assert_close(share, reconstructed, rtol=0, atol=0)
    loss = (share-.35).square().mean()+.05*torch.nn.functional.binary_cross_entropy_with_logits(presence, torch.ones_like(presence))
    loss.backward(); backward = time.time()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for n, p in trainable:
        if p.grad is None or not torch.isfinite(p.grad).all(): raise RuntimeError('Missing/nonfinite adapter gradient: '+n)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert w.grad is None and torch.equal(w.detach(), old_w)
    gradient_l2 = {n: float(p.grad.norm()) for n, p in trainable if n.startswith('stem.') or n.endswith('lora_B')}
    assert any(v > 0 for n, v in gradient_l2.items() if n.endswith('lora_B'))
    assert any(v > 0 for n, v in gradient_l2.items() if n.startswith('stem.'))
    optimizer.step(); after, _ = _parameter_digest(model, frozen_only=True)
    assert before == after, 'Frozen official base parameter bytes changed'
    model.eval()
    with torch.no_grad(): predictions = tuple(t.detach().clone() for t in model(x, w))
    checkpoint = model.adapter_checkpoint()
    suffix = '_'+smoke_tag if smoke_tag else ''
    adapter_file = model_directory(kind, cache_dir)/('cpu_smoke_adapter'+suffix+'.pt')
    if adapter_file.exists(): raise RuntimeError('Previous CPU smoke adapter exists; preserve it and choose an explicit new run')
    torch.save(checkpoint, adapter_file)
    restored = torch.load(adapter_file, map_location='cpu', weights_only=True)
    with torch.no_grad():
        for _, p in trainable: p.zero_()
    model.load_adapter_checkpoint(restored)
    with torch.no_grad(): replay = model(x, w)
    for a, b in zip(predictions, replay): torch.testing.assert_close(a, b, rtol=0, atol=0)
    receipt.update({'status': 'passed_actual_public_backbone_CPU_grid_adapter', 'synthetic_one_window_only': True,
        'pretrained_original_waveform_forward_tested': False, 'backbone_grid_forward_tested': True,
        'adapter_one_step_backward_passed': True, 'frozen_base_parameter_elements': frozen_count,
        'frozen_base_sha256_before': before, 'frozen_base_sha256_after': after,
        'frozen_expert_base_gradients_absent': True, 'all_trainable_gradients_finite': True,
        'gradient_l2_selected': gradient_l2, 'original_energy_weights_unchanged_no_gradient': True,
        'physical_pool_error': float((share-reconstructed).detach().abs().max()),
        'finite_synthetic_smoke_loss_not_validation_performance': float(loss.detach()),
        'adapter_path': str(adapter_file.relative_to(ROOT)), 'adapter_sha256': sha(adapter_file),
        'adapter_reload_same_verified_backbone_bitwise': True,
        'adapter_reload_fresh_second_B_object_tested': False,
        'timing_seconds': {'load_and_verify': loaded-begin, 'forward_with_base_hash': forward-loaded,
                           'backward': backward-forward, 'total': time.time()-begin},
        'network_disabled_during_load_and_compute': True, 'GPU_used': False,
        'platform': platform.platform(),
        'maximum_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
        'real_training_data_or_Test_accessed': False, 'Fold0_performance_measured': False,
        'production_adoption_decision': 'pending_actual_Fold0_training_and_source_validation'})
    out_path = ROOT/'reports'/('R5_ACTUAL_'+kind.upper()+'_CPU'+suffix.upper()+'.json')
    if out_path.exists(): raise RuntimeError('Previous smoke report exists; preserve it and choose a fresh tag')
    out_path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': receipt['status'], 'report': str(out_path), 'counts': receipt['adapter_counts'],
                      'timing_seconds': receipt['timing_seconds']}, indent=2), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=SPECS)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--cpu-smoke', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--smoke-tag', default='')
    args = parser.parse_args()
    if args.download: print(json.dumps(fetch_public_checkpoint(args.kind, args.cache_dir), indent=2), flush=True)
    if args.cpu_smoke: cpu_smoke(args.kind, args.cache_dir, args.threads, args.smoke_tag)
    if not args.download and not args.cpu_smoke: print(json.dumps(verify_public_checkpoint(args.kind, args.cache_dir), indent=2))


if __name__ == '__main__': main()
