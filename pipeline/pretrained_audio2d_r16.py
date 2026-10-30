"""R16 separately adapted public Dasheng / DINOv2 token backbones.

No network access occurs in the factory. Use the verification CLI's --download
explicitly first. Public weights are loaded strictly and anonymously, with SHA
pins. The new stride-one grid frontend is NOT the pretrained audio/image
frontend; successful transfer must be established by held-out experiments.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / 'models/pretrained_r16'
OLD_CACHE = ROOT / 'models/pretrained_r5'
PUBLIC_SPECS = {
    'dasheng_base': {
        'model_id': 'mispeech/dasheng-base',
        'revision': 'e0b0a27818689e59010bdd7739475e33090dff11',
        'weight_bytes': 341807392,
        'weight_sha256': 'adaa439ebec13933501242364a29b7912c2695d0354061b278c873438b2736c3',
        'files': {
            'config.json': 'b9ff40da4167dd776cb2fe084c9a80c0fdd6568c3b995e0255b46a9d435e7b80',
            'configuration_dasheng.py': '7a524b9d37c0e0392b75bbfb0fe5e2d1f60f62a8f2c46917bb186b638580966b',
            'modeling_dasheng.py': 'a7908ca04b9e3ec3c6fb04b3719d5ae180d555383cbe8ab4bd2f35be272cbe99',
            'README.md': '03f667e8e1d7745c71ec288ea80426235dfd0b4cfaa2de1be0f2b587a2243e28',
        },
        'license': 'Apache-2.0',
        'license_url': 'https://raw.githubusercontent.com/RicherMans/Dasheng/91bc15c728f08e8d10f741311d8afc42fb33d497/LICENSE',
        'license_sha256': '249c4ef6ef22f69a8a1076d4eb04eb8560af2d4fc48440749f51477efacbaa8f',
    },
    'dinov2_small': {
        'model_id': 'facebook/dinov2-small',
        'revision': 'ed25f3a31f01632728cabb09d1542f84ab7b0056',
        'weight_bytes': 88249960,
        'weight_sha256': 'ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1',
        'files': {
            'config.json': '1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d',
            'README.md': '4c20dca454a8e5c670e8de5c7e6040f512aeca5438516f7623eedc4e3b00599c',
        },
        'license': 'Apache-2.0',
        'license_url': 'https://raw.githubusercontent.com/facebookresearch/dinov2/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE',
        'license_sha256': '600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7',
    },
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            h.update(part)
    return h.hexdigest()


def _directory(kind, cache_dir):
    return Path(cache_dir) / PUBLIC_SPECS[kind]['model_id'].replace('/', '--')


def fetch_audio2d_checkpoint(kind, cache_dir=DEFAULT_CACHE):
    """Explicit public download, with no HF login/token or remote code runner."""
    from pipeline.load_foundation_r5 import _download
    spec = PUBLIC_SPECS[kind]
    directory = _directory(kind, cache_dir)
    for name, expected in spec['files'].items():
        url = f"https://huggingface.co/{spec['model_id']}/resolve/{spec['revision']}/{name}"
        _download(url, directory / name, expected)
    url = f"https://huggingface.co/{spec['model_id']}/resolve/{spec['revision']}/model.safetensors"
    _download(url, directory / 'model.safetensors', spec['weight_sha256'], spec['weight_bytes'])
    _download(spec['license_url'], directory / 'LICENSE', spec['license_sha256'])
    return verify_audio2d_checkpoint(kind, cache_dir)


def verify_audio2d_checkpoint(kind, cache_dir=DEFAULT_CACHE):
    spec = PUBLIC_SPECS[kind]
    directory = _directory(kind, cache_dir)
    expected_files = {**spec['files'], 'model.safetensors': spec['weight_sha256'],
                      'LICENSE': spec['license_sha256']}
    for name, expected in expected_files.items():
        path = directory / name
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise RuntimeError(f'Missing/changed public checkpoint artifact: {path}')
    if (directory / 'model.safetensors').stat().st_size != spec['weight_bytes']:
        raise RuntimeError('Checkpoint byte-size mismatch')
    return {'kind': kind, **spec, 'verified': True, 'directory': str(directory),
            'credential_policy': 'anonymous public URL; no token lookup or implicit from_pretrained'}


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, dropout=.1):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1:
            raise ValueError('LoRA requires a Linear and positive rank')
        self.base = base
        self.base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device,
                                               dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                              dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** .5)
        self.dropout = nn.Dropout(dropout)
        self.scaling = 2.0

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scaling


class AttentivePool(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.attention = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, h, valid):
        attention = self.attention(h).squeeze(-1).masked_fill(~valid, torch.finfo(h.dtype).min)
        return self.output((attention.softmax(-1).unsqueeze(-1) * h).sum(1)).squeeze(-1)


class Audio2DTaskBranch(nn.Module):
    """One task owns its stem, LoRA, norms and head; never shares adapted state."""
    def __init__(self, backbone, family, task, width, rank=8, dropout=.1,
                 gradient_checkpointing=False):
        super().__init__()
        aliases = {'regression': 'regression', 'reg': 'regression', 'classification': 'classification',
                   'classifier': 'classification', 'cls': 'classification'}
        if task not in aliases:
            raise ValueError('task must be regression/reg or classification/cls')
        self.task = aliases[task]
        self.family = family
        self.in_channels = 17 if self.task == 'regression' else 15
        self.mask_channel = 5
        self.width = width
        self.gradient_checkpointing = gradient_checkpointing
        self.backbone = backbone.requires_grad_(False)
        self.stem = nn.Conv2d(self.in_channels, width, 3, stride=1, padding=1)
        self.time_position = nn.Parameter(torch.zeros(1, 15, 1, width))
        self.band_position = nn.Parameter(torch.zeros(1, 1, 25, width))
        nn.init.trunc_normal_(self.time_position, std=.02)
        nn.init.trunc_normal_(self.band_position, std=.02)
        self.stem_dropout = nn.Dropout(dropout)
        self.head = (nn.Conv2d(width, 1, 1) if self.task == 'regression'
                     else AttentivePool(width, dropout))
        if family == 'dasheng':
            for block in self.backbone.blocks:
                block.attn.qkv = LoRALinear(block.attn.qkv, rank, dropout)
            active = [self.backbone.blocks, self.backbone.norm]
        elif family == 'dinov2':
            for block in self.backbone.encoder.layer:
                attention = block.attention.attention
                attention.query = LoRALinear(attention.query, rank, dropout)
                attention.value = LoRALinear(attention.value, rank, dropout)
            active = [self.backbone.encoder, self.backbone.layernorm]
        else:
            raise ValueError('Unknown verified backbone family')
        adapter_ids = {id(p) for name, p in self.named_parameters() if not name.startswith('backbone.')}
        for module in active:
            for child in module.modules():
                if isinstance(child, LoRALinear):
                    adapter_ids.update([id(child.lora_a), id(child.lora_b)])
                if isinstance(child, nn.LayerNorm):
                    adapter_ids.update(id(p) for p in child.parameters(recurse=False))
        self._adapter_names = tuple(name for name, p in self.named_parameters() if id(p) in adapter_ids)
        self.set_stage(1)

    def set_stage(self, stage):
        if stage in ('warmup',):
            stage = 1
        if stage in ('peft',):
            stage = 2
        if stage not in (1, 2):
            raise ValueError('stage must be 1 (warmup) or 2 (PEFT)')
        self.requires_grad_(False)
        for name, p in self.named_parameters():
            if name in self._adapter_names and (stage == 2 or not name.startswith('backbone.')):
                p.requires_grad_(True)
        self.stage = stage
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        # Stochastic pretrained layers remain in eval. Only our LoRA dropout
        # is enabled in stage2; gradient flow through the frozen body is kept.
        if hasattr(self, 'backbone'):
            self.backbone.eval()
            for module in self.backbone.modules():
                if isinstance(module, LoRALinear):
                    module.dropout.train(mode and getattr(self, 'stage', 1) == 2)
        return self

    def adapter_state(self):
        state = self.state_dict()
        return {name: state[name].detach().cpu().clone() for name in self._adapter_names}

    def load_adapter_state(self, state):
        if set(state) != set(self._adapter_names):
            raise RuntimeError('Adapter state keys differ (strict task/backbone match required)')
        current = self.state_dict()
        for name in self._adapter_names:
            if state[name].shape != current[name].shape or not torch.isfinite(state[name]).all():
                raise RuntimeError('Invalid adapter tensor: ' + name)
        with torch.no_grad():
            for name in self._adapter_names:
                current[name].copy_(state[name].to(device=current[name].device, dtype=current[name].dtype))
        return self

    def frozen_named_parameters(self):
        return ((name, p) for name, p in self.named_parameters() if name not in self._adapter_names)

    def _tokens(self, h):
        if self.family == 'dasheng':
            for block in self.backbone.blocks:
                h = checkpoint(block, h, use_reentrant=False) if self.gradient_checkpointing and self.training else block(h)
            return self.backbone.norm(h)
        for block in self.backbone.encoder.layer:
            def run(z, layer=block):
                return layer(z, head_mask=None, output_attentions=False)[0]
            h = checkpoint(run, h, use_reentrant=False) if self.gradient_checkpointing and self.training else run(h)
        return self.backbone.layernorm(h)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.in_channels, 15, 25):
            raise ValueError(f'Expected B,{self.in_channels},15,25; got {tuple(x.shape)}')
        # Mask index is5 in both the ordered17reg and ordered15cls contracts.
        valid = x[:, self.mask_channel].reshape(x.shape[0], 375) > 0
        if not bool(valid.any(-1).all()):
            raise ValueError('Every window must contain an observed cell')
        h = self.stem(x).permute(0, 2, 3, 1)
        h = self.stem_dropout(h + self.time_position + self.band_position).reshape(x.shape[0], 375, self.width)
        h = self._tokens(h)
        if self.task == 'classification':
            return self.head(h, valid)
        return self.head(h.reshape(x.shape[0], 15, 25, self.width).permute(0, 3, 1, 2)).squeeze(1)


def create_audio2d_branch(kind, task, rank=8, dropout=.1, device='cpu', dtype='float32',
                          cache_dir=DEFAULT_CACHE, gradient_checkpointing=False, **kwargs):
    """Strictly load public weights; return a newly independent task branch."""
    aliases = {'dasheng': 'dasheng_base', 'dasheng-base': 'dasheng_base',
               'dinov2': 'dinov2_small', 'dinov2-small': 'dinov2_small',
               'dasheng_1.2b': 'dasheng_1.2b', 'dasheng-1.2b': 'dasheng_1.2b'}
    kind = aliases.get(kind, kind)
    if 'convnext' in kind.lower():
        raise RuntimeError('ConvNeXt-V2 public pretrained weights are CC-BY-NC; this run uses Apache-2.0 DINOv2-small instead. No silent substitution.')
    if dtype not in ('float32', 'float16', 'bfloat16'):
        raise ValueError('Explicit supported dtype required')
    from pipeline.load_foundation_r5 import _official_runtime, _import_pinned_module
    _official_runtime(OLD_CACHE)
    from safetensors.torch import load_file
    if kind == 'dasheng_1.2b':
        from pipeline.load_foundation_r5 import verify_public_checkpoint, model_directory
        provenance = verify_public_checkpoint('dasheng', OLD_CACHE)
        directory = model_directory('dasheng', OLD_CACHE)
    else:
        if kind not in PUBLIC_SPECS:
            raise ValueError('Unsupported explicit public kind: ' + str(kind))
        provenance = verify_audio2d_checkpoint(kind, cache_dir)
        directory = _directory(kind, cache_dir)
    cfg_json = json.loads((directory / 'config.json').read_text())
    if kind.startswith('dasheng'):
        package = '_underwater_r16_' + kind.replace('.', '_')
        cm = _import_pinned_module(directory, package, 'configuration_dasheng')
        mm = _import_pinned_module(directory, package, 'modeling_dasheng')
        config = cm.DashengConfig(**cfg_json)
        with torch.device('meta'):
            complete = mm.DashengModel(config)
        family, width = 'dasheng', config.encoder_kwargs['embed_dim']
    else:
        from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
        from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
        config = Dinov2Config(**cfg_json)
        with torch.device('meta'):
            complete = Dinov2Model(config)
        family, width = 'dinov2', config.hidden_size
    tensors = load_file(str(directory / 'model.safetensors'), device='cpu')
    provenance['strict_checkpoint_keys'] = len(tensors)
    provenance['checkpoint_parameter_elements'] = sum(t.numel() for t in tensors.values())
    complete.load_state_dict(tensors, strict=True, assign=True)
    del tensors
    if any(p.is_meta for p in complete.parameters()):
        raise RuntimeError('Pretrained parameters remained on meta')
    backbone = complete.encoder if family == 'dasheng' else complete
    active_parts = ([backbone.blocks, backbone.norm] if family == 'dasheng'
                    else [backbone.encoder, backbone.layernorm])
    active_pretrained = sum(p.numel() for part in active_parts for p in part.parameters())
    module = Audio2DTaskBranch(backbone, family, task, width, rank, dropout, gradient_checkpointing)
    module = module.to(device=device, dtype=getattr(torch, dtype))
    provenance.update({'architecture': kind, 'task': module.task, 'width': width, 'tokens': 375,
                       'stem_stride': 1, 'grid': [15, 25], 'LoRA_rank': rank,
                       'strict_checkpoint_load': True, 'classification_parameter_sharing': False,
                       'pretrained_frontend_used': False,
                       'pretrained_forward_parameter_elements': active_pretrained,
                       'pretrained_bypassed_parameter_elements': provenance['checkpoint_parameter_elements'] - active_pretrained,
                       'pretrained_forward_scope': 'all Transformer blocks and final normalization',
                       'transfer_limitation': 'Original audio/image frontend, absolute positions and CLS/readout are bypassed. New grid stem/positions require supervised alignment; source-task SOTA is not competition SOTA.',
                       'total_parameters': sum(p.numel() for p in module.parameters()),
                       'adaptation_parameters': sum(p.numel() for n, p in module.named_parameters() if n in module._adapter_names)})
    module.provenance = provenance
    return module, provenance
