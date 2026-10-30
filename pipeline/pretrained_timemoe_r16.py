"""R16 TimeMoE per-band PEFT branches, separate from all frozen R5/R16 code.

The public checkpoint is a forecasting model (~453M total / ~200M active),
not a pretrained underwater source separator.  Shape-compatible adaptation is
not evidence of transfer benefit.  Each task owns its own decoder/LoRA/norms.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from torch import nn

from pipeline.foundation_masks_r5 import LoRALinear
from pipeline.load_foundation_r5 import (
    DEFAULT_CACHE, SPECS, _import_pinned_module, _official_runtime,
    model_directory, verify_public_checkpoint,
)

CLASSIFIER_CHANNELS = (0, 1, 2, 3, 6, 5) + tuple(range(8, 17))


def _bare_decoder(cache_dir, dtype):
    """Local pinned files only; no old R5 task adapters/checkpoints are reused."""
    from safetensors.torch import load_file
    receipt = verify_public_checkpoint('timemoe_public', cache_dir)
    receipt['transformers'] = _official_runtime(cache_dir)
    spec = SPECS['timemoe_public']
    directory = model_directory('timemoe_public', cache_dir)
    package = 'r16_peft_timemoe_' + spec['revision'][:12]
    cm = _import_pinned_module(directory, package, spec['configuration_module'])
    mm = _import_pinned_module(directory, package, spec['modeling_module'])
    cfg = getattr(cm, spec['configuration_class'])(**json.loads((directory / 'config.json').read_text()))
    cfg._attn_implementation = 'eager'
    with torch.device('meta'):
        full = getattr(mm, spec['model_class'])(cfg)
    state = load_file(str(directory / 'model.safetensors'), device='cpu')
    full.load_state_dict(state, strict=True, assign=True)
    del state
    core = full.model
    del full
    for layer in core.layers:
        old = layer.self_attn.rotary_emb
        layer.self_attn.rotary_emb = type(old)(old.dim, max_position_embeddings=old.max_position_embeddings,
                                             base=old.base, device='cpu')
    if any(p.is_meta for p in list(core.parameters()) + list(core.buffers())):
        raise RuntimeError('Unmaterialized TimeMoE tensor')
    core = core.to(dtype=getattr(torch, dtype))
    gc.collect()
    receipt.update({'strict_pretrained_load': True, 'official_source_edited': False,
                    'retained': 'official TimeMoeModel decoder',
                    'bypassed': ['original scalar input embedding', 'forecast heads'],
                    'legacy_R5_task_adapter_reused': False,
                    'task_transfer_performance_proven': False})
    return core, receipt


class TimeMoER16Branch(nn.Module):
    """17- or 15-channel input -> 25 independent 15-point decoder sequences.

    A small cross-band decoder mixes frequency information *after* the causal
    pretrained time decoder. Classification uses attentive pooling and never
    receives energy weights, share predictions or regression features.
    """
    def __init__(self, backbone, task, rank=8, dropout=.1, *, fixture_mode=False):
        super().__init__()
        if task not in ('regression', 'classification'):
            raise ValueError('Expected regression or classification task')
        if not 0 <= dropout < 1 or rank not in (8, 16):
            raise ValueError('Use rank8/16 and valid dropout')
        self.task, self.rank, self.dropout_probability = task, rank, float(dropout)
        self.fixture_mode = bool(fixture_mode)
        self.input_channels = 17 if task == 'regression' else 15
        self.backbone = backbone
        self.width = int(backbone.config.hidden_size)
        self.depth = len(backbone.layers)
        if not fixture_mode and (self.width, self.depth) != (768, 12):
            raise ValueError('Requires exact public TimeMoE-200M width768/depth12')
        backbone.requires_grad_(False)
        for layer in backbone.layers:
            for name in ('q_proj', 'v_proj'):
                base = getattr(layer.self_attn, name)
                if type(base) is not nn.Linear:
                    raise TypeError('Bare official linear projections required')
                setattr(layer.self_attn, name, LoRALinear(base, rank=rank, alpha=2*rank, dropout=dropout))
        self.stem = nn.Linear(self.input_channels, self.width)
        self.stem_norm = nn.LayerNorm(self.width)
        self.time_position = nn.Parameter(torch.zeros(1, 1, 15, self.width))
        self.band_position = nn.Parameter(torch.zeros(1, 25, 1, self.width))
        decoder_width = min(128, self.width)
        if decoder_width % 4:
            raise ValueError('Cross-band width must divide four heads')
        self.cross_in = nn.Linear(self.width, decoder_width)
        self.cross_norm = nn.LayerNorm(decoder_width)
        self.cross_attention = nn.MultiheadAttention(decoder_width, 4, dropout=dropout, batch_first=True)
        self.cross_ff = nn.Sequential(nn.LayerNorm(decoder_width), nn.Linear(decoder_width, decoder_width*2),
                                      nn.GELU(), nn.Dropout(dropout), nn.Linear(decoder_width*2, decoder_width))
        if task == 'regression':
            self.mask_head = nn.Linear(decoder_width, 1)
        else:
            self.attention_pool = nn.Sequential(nn.Linear(decoder_width, decoder_width//2), nn.Tanh(),
                                                nn.Linear(decoder_width//2, 1))
            self.presence_head = nn.Sequential(nn.LayerNorm(decoder_width), nn.Dropout(dropout),
                                               nn.Linear(decoder_width, 1))
        # Include trainable affine norms, including official TimeMoeRMSNorm,
        # but do not unfreeze expert/router/attention-base weights or biases.
        self._norm_names = set()
        for module_name, module in backbone.named_modules():
            if 'norm' in type(module).__name__.lower():
                for name, _ in module.named_parameters(recurse=False):
                    self._norm_names.add('backbone.' + module_name + '.' + name)
        self._adapter_names = frozenset(n for n, _ in self.named_parameters()
                                       if not n.startswith('backbone.')
                                       or n.endswith(('lora_A', 'lora_B')) or n in self._norm_names)
        self.stage = 0
        self.set_stage(1)

    def set_stage(self, stage):
        if stage in ('stage1', 'stage2'):
            stage = int(stage[-1])
        if stage not in (1, 2):
            raise ValueError('Stage must be1 or2')
        for name, p in self.named_parameters():
            p.requires_grad_(name in self._adapter_names and (stage == 2 or not name.startswith('backbone.')))
        self.stage = stage
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for module in self.backbone.modules():
            if isinstance(module, LoRALinear):
                module.dropout.train(mode and self.stage == 2)
        return self

    def _validate(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.input_channels, 15, 25):
            raise ValueError(f'Expected [N,{self.input_channels},15,25]')
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('Finite floating features required')
        observed = x[:, 5] > .5
        if not observed.any((1, 2)).all():
            raise ValueError('Each window must contain an observed cell')
        return observed

    def forward_features(self, x):
        observed = self._validate(x)
        # Frequency-major independent temporal sequences; no chronology
        # compression, interpolation, token downsampling or dropped minutes.
        h = x.permute(0, 3, 2, 1)
        h = self.stem_norm(self.stem(h)) + self.time_position + self.band_position
        h = h.reshape(len(x)*25, 15, self.width)
        h = h.to(next(self.backbone.parameters()).dtype)
        output = self.backbone(inputs_embeds=h, attention_mask=None, use_cache=False,
                               return_dict=True)
        h = output.last_hidden_state
        if h.shape != (len(x)*25, 15, self.width):
            raise RuntimeError('Unexpected official decoder output')
        h = h.reshape(len(x), 25, 15, self.width).permute(0, 2, 1, 3)
        h = self.cross_in(h.to(self.cross_in.weight.dtype)).reshape(len(x)*15, 25, -1)
        valid = observed.reshape(len(x)*15, 25)
        safe_valid = valid.clone()
        safe_valid[~valid.any(1), 0] = True  # Avoid all-masked softmax NaNs.
        normalized = self.cross_norm(h)
        attended, _ = self.cross_attention(normalized, normalized, normalized,
                                            key_padding_mask=~safe_valid, need_weights=False)
        h = h + attended
        h = h + self.cross_ff(h)
        h = h.reshape(len(x), 15, 25, -1)
        return h.masked_fill(~observed[..., None], 0)

    def forward(self, x):
        h = self.forward_features(x)
        if self.task == 'regression':
            return self.mask_head(h)[..., 0]
        observed = x[:, 5] > .5
        attention = self.attention_pool(h)[..., 0].masked_fill(~observed, -torch.inf)
        attention = attention.flatten(1).softmax(1).reshape(len(x), 15, 25)
        pooled = (h * attention[..., None]).sum((1, 2))
        return self.presence_head(pooled)[:, 0]

    def adapter_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if n in self._adapter_names}

    def load_adapter_state(self, state):
        expected = {n: p for n, p in self.named_parameters() if n in self._adapter_names}
        if set(state) != set(expected):
            raise ValueError('Adapter keys differ')
        for n, p in expected.items():
            t = state[n]
            if not isinstance(t, torch.Tensor) or t.shape != p.shape or not torch.isfinite(t).all():
                raise ValueError('Invalid adaptation tensor: ' + n)
        with torch.no_grad():
            for n, p in expected.items():
                p.copy_(state[n].to(p.device, p.dtype))
        return self

    def frozen_named_parameters(self):
        return ((n, p) for n, p in self.named_parameters() if n not in self._adapter_names)

    def architecture(self):
        return {'format': 'r16_timemoe_perband_peft_v1', 'task': self.task,
                'input_channels': self.input_channels, 'input_grid': [15, 25],
                'sequence_layout': 'N*25 bands,15 temporal tokens,width768',
                'rank': self.rank, 'dropout': self.dropout_probability,
                'backbone_width': self.width, 'backbone_layers': self.depth,
                'cross_band_decoder_width': min(128, self.width),
                'total_parameters': sum(p.numel() for p in self.parameters()),
                'adaptation_parameters': sum(p.numel() for n, p in self.named_parameters() if n in self._adapter_names),
                'current_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'stage': self.stage, 'fixture_mode': self.fixture_mode,
                'pretrained_performance_proven': False, 'energy_weights_in_classifier': False}


def create_timemoe_branch(task, rank=8, dropout=.1, cache_dir=DEFAULT_CACHE,
                          device='cpu', dtype='float32'):
    if dtype not in ('float32', 'float16', 'bfloat16'):
        raise ValueError('Unsupported explicit dtype')
    backbone, receipt = _bare_decoder(Path(cache_dir), dtype)
    model = TimeMoER16Branch(backbone, task, rank, dropout).to(device)
    receipt.update({'architecture': model.architecture(),
                    'classifier_channel_order': list(CLASSIFIER_CHANNELS),
                    'regression_classification_trainable_state_shared': False,
                    'normalization_note': 'new input stem; original scalar forecasting input projection bypassed'})
    return model, receipt
