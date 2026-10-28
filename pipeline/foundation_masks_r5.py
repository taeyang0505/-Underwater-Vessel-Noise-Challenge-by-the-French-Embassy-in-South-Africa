"""R5 coarse-grid adaptations of injected Time-MoE/Dasheng backbones.

This module never downloads or constructs a foundation backbone. Callers must
inject the real, separately verified object. The unverified Ultra checkpoint
availability gate is CLOSED. Public TimeMoE-200M is explicitly a different
453M-total checkpoint. Tiny protocol fixtures require fixture_mode=True
and are explicitly not pretrained models or evidence of model performance.

Official APIs inspected 2026-10-25:
https://github.com/Time-MoE/Time-MoE/blob/main/time_moe/models/modeling_time_moe.py
https://huggingface.co/mispeech/dasheng-1.2B/blob/main/modeling_dasheng.py
https://github.com/RicherMans/Dasheng/blob/main/dasheng/pretrained/pretrained.py
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


ULTRA_PRETRAINED_CHECKPOINT_AVAILABLE = False
OFFICIAL_SOURCES = {
    'timemoe_code': 'https://github.com/Time-MoE/Time-MoE/blob/main/time_moe/models/modeling_time_moe.py',
    'timemoe_paper': 'https://arxiv.org/html/2409.16040v4',
    'timemoe_public_checkpoint': 'https://huggingface.co/Maple728/TimeMoE-200M',
    'dasheng_hf_code': 'https://huggingface.co/mispeech/dasheng-1.2B/blob/main/modeling_dasheng.py',
    'dasheng_hf_config': 'https://huggingface.co/mispeech/dasheng-1.2B/blob/main/config.json',
    'dasheng_weight_url': 'https://huggingface.co/mispeech/dasheng-1.2B/resolve/main/model.safetensors',
    'dasheng_author_weight_url': 'https://zenodo.org/records/11511780/files/dasheng_12b.pt?download=1',
}


class FoundationAvailabilityError(RuntimeError):
    """Requested verified pretrained artifact is not available/established."""


@dataclass(frozen=True)
class PretrainedEvidence:
    """Caller-supplied local artifact verification, not an object-weight proof.

    The loading layer must actually verify its file hash before construction.
    This wrapper does not reinterpret random or substituted models as pretrained.
    """
    family: str
    checkpoint_url: str
    revision: str
    checkpoint_sha256: str
    weights_verified: bool

    def validate(self, family: str):
        if self.family != family or self.weights_verified is not True:
            raise FoundationAvailabilityError('Verified pretrained evidence for the exact family is required')
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise FoundationAvailabilityError('Pin the source revision before using a real backbone')
        if len(self.checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.checkpoint_sha256):
            raise FoundationAvailabilityError('Verified SHA-256 of the exact checkpoint is required')
        if family == 'dasheng_1.2b' and self.checkpoint_url not in {
            OFFICIAL_SOURCES['dasheng_weight_url'], OFFICIAL_SOURCES['dasheng_author_weight_url'],
        } and not self.checkpoint_url.startswith('https://huggingface.co/mispeech/dasheng-1.2B/resolve/'+self.revision+'/'):
            raise FoundationAvailabilityError('Unsupported Dasheng checkpoint source; review provenance first')
        if family == 'timemoe_public_200m_active_453m_total' and not self.checkpoint_url.startswith(
            'https://huggingface.co/Maple728/TimeMoE-200M/resolve/'+self.revision+'/'
        ):
            raise FoundationAvailabilityError('Public Time-MoE source/revision must be exact; never label it Ultra')


class LoRALinear(nn.Module):
    """Frozen standard Linear plus alpha/r * B(A(dropout(x))); no PEFT needed."""
    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None, dropout: float = .1):
        super().__init__()
        if type(base) is not nn.Linear:
            raise TypeError('Only an unwrapped standard torch.nn.Linear is supported')
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1 or not 0 <= dropout < 1:
            raise ValueError('Invalid LoRA rank/dropout')
        alpha = float(2*rank if alpha is None else alpha)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('Positive finite LoRA alpha required')
        self.base = base
        self.base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.alpha, self.dropout_probability = rank, alpha, float(dropout)
        self.scaling = alpha/rank
        self.dropout = nn.Dropout(dropout)
        factory = {'device': base.weight.device, 'dtype': base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty((rank, base.in_features), **factory))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, rank), **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        # Official Time-MoE flash path reads q_proj.weight.dtype.
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x)+F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)*self.scaling


def _require_linear(parent, name, in_features=None, out_features=None):
    layer = getattr(parent, name, None)
    if type(layer) is not nn.Linear:
        raise TypeError(f'Official protocol expected an unwrapped Linear at {name}')
    if in_features is not None and layer.in_features != in_features:
        raise ValueError(f'Unexpected input width for {name}')
    if out_features is not None and layer.out_features != out_features:
        raise ValueError(f'Unexpected output width for {name}')
    return layer


def _weights(x, weights):
    if x.ndim != 4 or len(x) < 1 or tuple(x.shape[1:]) != (8, 15, 25) or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('Expected finite floating input [B,8,15,25] with explicit missingness features')
    if weights.shape != (len(x), 15, 25) or weights.device != x.device or not weights.is_floating_point():
        raise ValueError('Energy weights must have shape [B,15,25] on the input device')
    w = weights.detach()
    if not torch.isfinite(w).all() or (w < 0).any() or not torch.allclose(w.sum((1, 2)), torch.ones(len(x), device=w.device, dtype=w.dtype), atol=2e-5, rtol=0):
        raise ValueError('Use original nonnegative observed-cell energy weights normalized per window; do not learn/re-normalize them here')
    return w


def _output(logits, presence, w):
    if logits.shape != w.shape or not torch.isfinite(logits).all() or not torch.isfinite(presence).all():
        raise RuntimeError('Backbone/head produced invalid finite grid logits')
    cell = torch.sigmoid(logits)*(w > 0).to(logits.dtype)
    # FP32 accumulation when the body uses FP16/BF16. w is detached, unchanged.
    share = (cell.float()*w.float()).sum((1, 2))
    return share, presence[:, 0], cell


class _AdapterBase(nn.Module):
    feature_version = 3
    input_channels = 8
    architecture_version = 6
    requires_background_prior = True

    def train(self, mode=True):
        # Keep frozen vendor dropout/BN buffers stable; only new LoRA dropout trains.
        super().train(mode)
        self.backbone.eval()
        for module in self.backbone.modules():
            if isinstance(module, LoRALinear): module.dropout.train(mode)
        return self

    def trainable_parameter_counts(self):
        counts = {'total': sum(p.numel() for p in self.parameters()),
                  'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
                  'frozen': sum(p.numel() for p in self.parameters() if not p.requires_grad)}
        counts['lora'] = sum(p.numel() for n, p in self.named_parameters() if p.requires_grad and (n.endswith('lora_A') or n.endswith('lora_B')))
        return counts

    def adapter_checkpoint(self):
        """Only trainable adapters/new stem/heads/positions, never vendor weights."""
        return {'format': 'r5_foundation_adapter_v1', 'manifest': self.adapter_manifest(),
                'state': {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}}

    def load_adapter_checkpoint(self, checkpoint):
        if checkpoint.get('format') != 'r5_foundation_adapter_v1' or checkpoint.get('manifest') != self.adapter_manifest():
            raise ValueError('Adapter identity/config/pretrained provenance mismatch')
        expected = {n: p for n, p in self.named_parameters() if p.requires_grad}
        state = checkpoint.get('state', {})
        if set(state) != set(expected): raise ValueError('Missing/unexpected adapter parameters')
        # Validate all before any mutation.
        for name, p in expected.items():
            t = state[name]
            if not isinstance(t, torch.Tensor) or t.shape != p.shape or not torch.isfinite(t).all():
                raise ValueError(f'Invalid adapter tensor: {name}')
        with torch.no_grad():
            for name, p in expected.items(): p.copy_(state[name].to(device=p.device, dtype=p.dtype))
        return self

    def save_adapter(self, path):
        torch.save(self.adapter_checkpoint(), Path(path))

    def _common_manifest(self):
        return {'family': self.family, 'fixture_mode': self.fixture_mode, 'pretrained_claim': not self.fixture_mode,
                'provenance': None if self.evidence is None else asdict(self.evidence),
                'width': self.width, 'layers': self.depth, 'rank': self.rank, 'alpha': 2*self.rank,
                'lora_dropout': self.lora_dropout, 'lora_paths': self.lora_paths,
                'input_grid': [8, 15, 25], 'pool': 'sum(original_detached_energy_weights * sigmoid_grid_mask)',
                'presence': 'independent learned logit head', 'base_frozen': True}


class TimeMoEUltraMask(_AdapterBase):
    """15 time tokens; 200-feature stem -> official decoder -> 25 logits.

    Ultra pretrained availability remains unverified: production construction
    always fails closed. fixture_mode only admits a tiny protocol fixture and
    explicitly cannot be labelled as a pretrained Ultra run.
    """
    family = 'timemoe_ultra_2.4b'

    def _validate_evidence(self, evidence, fixture_mode):
        if not fixture_mode:
            if not ULTRA_PRETRAINED_CHECKPOINT_AVAILABLE:
                raise FoundationAvailabilityError('Time-MoE Ultra 2.4B pretrained checkpoint availability is unverified. No 453M substitution/random pretrained fallback is permitted.')
            if evidence is None: raise FoundationAvailabilityError('Exact Ultra checkpoint evidence required')
            evidence.validate(self.family)
        elif evidence is not None:
            raise ValueError('Fixture mode cannot carry pretrained evidence/claim')

    def _validate_production_shape(self, config, width, depth):
        if (width, depth, config.num_experts, config.num_experts_per_tok) != (1024, 36, 8, 2):
            raise ValueError('Requested Ultra shape does not match the official 2.4B configuration')

    def __init__(self, backbone: nn.Module, *, evidence: PretrainedEvidence | None = None,
                 fixture_mode=False, rank=16, lora_dropout=.1, adapt_router=False):
        super().__init__()
        if type(fixture_mode) is not bool or type(adapt_router) is not bool:
            raise TypeError('fixture_mode/adapt_router must be explicit booleans')
        self._validate_evidence(evidence, fixture_mode)
        if not isinstance(backbone, nn.Module): raise TypeError('Inject a backbone object, not a name/download instruction')
        # Accept official TimeMoeForPrediction.model or bare TimeMoeModel.
        decoder = getattr(backbone, 'model', backbone)
        config = getattr(decoder, 'config', None)
        if config is None or not hasattr(decoder, 'layers') or not hasattr(decoder, 'norm'):
            raise TypeError('Expected official TimeMoeModel layers/norm/config protocol')
        params = inspect.signature(decoder.forward).parameters
        if 'inputs_embeds' not in params or 'use_cache' not in params or 'return_dict' not in params:
            raise TypeError('Time-MoE decoder must explicitly accept inputs_embeds/use_cache/return_dict')
        self.width, self.depth = int(config.hidden_size), len(decoder.layers)
        if self.width < 1 or self.depth < 1: raise ValueError('Empty decoder')
        if not fixture_mode: self._validate_production_shape(config, self.width, self.depth)
        self.backbone = decoder; self.fixture_mode = bool(fixture_mode); self.evidence = evidence
        self.rank, self.lora_dropout, self.adapt_router = rank, float(lora_dropout), bool(adapt_router)
        # Validate the whole API before wrapping any layer.
        targets = []
        for i, layer in enumerate(decoder.layers):
            attention = getattr(layer, 'self_attn', None)
            if attention is None: raise TypeError('Missing official self_attn protocol')
            for name in ['q_proj', 'v_proj']:
                linear = _require_linear(attention, name, self.width)
                targets.append((attention, name, linear, f'layers.{i}.self_attn.{name}'))
            if adapt_router:
                router = getattr(layer, 'ffn_layer', None)
                if router is None: raise TypeError('Router adaptation requested without ffn_layer')
                linear = _require_linear(router, 'gate', self.width)
                targets.append((router, 'gate', linear, f'layers.{i}.ffn_layer.gate'))
        decoder.requires_grad_(False)
        self.lora_paths = [path for _, _, _, path in targets]
        for parent, name, linear, _ in targets:
            setattr(parent, name, LoRALinear(linear, self.rank, dropout=self.lora_dropout))
        self.stem = nn.Linear(8*25, self.width)
        self.time_position = nn.Parameter(torch.zeros(1, 15, self.width))
        self.mask_head = nn.Linear(self.width, 25)
        self.presence_head = nn.Linear(self.width, 1)
        self.train(True)

    def _hidden(self, x):
        h = self.stem(x.permute(0, 2, 1, 3).reshape(len(x), 15, 200))+self.time_position
        body_dtype = next(self.backbone.parameters()).dtype
        # All fixed grid positions exist; missingness is a feature, not padding.
        # Avoid deleting timeline positions or generating a cache during training.
        out = self.backbone(inputs_embeds=h.to(body_dtype), attention_mask=None,
                            use_cache=False, return_dict=True)
        h = getattr(out, 'last_hidden_state', None)
        if h is None or h.shape != (len(x), 15, self.width):
            raise RuntimeError('Official decoder output last_hidden_state protocol mismatch')
        return h.to(self.mask_head.weight.dtype)

    def forward(self, x, energy_weights):
        w = _weights(x, energy_weights); h = self._hidden(x)
        time_weights = w.sum(-1).to(h.dtype)
        pooled = (h*time_weights[..., None]).sum(1)
        return _output(self.mask_head(h), self.presence_head(pooled), w)

    def representation(self, x):
        return self._hidden(x).mean(1)

    def adapter_manifest(self):
        return {**self._common_manifest(), 'tokens': 15, 'position': 'new learned 15-time positions plus official RoPE',
                'router_lora': self.adapt_router, 'experts_trainable': False,
                'availability_gate': 'fixture_only_until_exact_Ultra_pretrained_checkpoint_verified'}


class TimeMoEPublicMask(TimeMoEUltraMask):
    """Exact public Maple728/TimeMoE-200M: ~200M active, ~453M TOTAL.

    This is a separately named public checkpoint, never an Ultra replacement
    hidden behind the Ultra family label. It shares the reviewed decoder API.
    """
    family = 'timemoe_public_200m_active_453m_total'

    def _validate_evidence(self, evidence, fixture_mode):
        if fixture_mode:
            if evidence is not None: raise ValueError('Fixture cannot claim pretrained evidence')
        else:
            if evidence is None: raise FoundationAvailabilityError('Verified exact public TimeMoE-200M checkpoint evidence required')
            evidence.validate(self.family)

    def _validate_production_shape(self, config, width, depth):
        if (width, depth, config.num_experts, config.num_experts_per_tok) != (768, 12, 8, 2):
            raise ValueError('Public TimeMoE-200M requires width768/depth12/8experts/top2')

    def adapter_manifest(self):
        return {**super().adapter_manifest(), 'availability_gate': 'verified_exact_public_checkpoint',
                'checkpoint_brand': 'Maple728/TimeMoE-200M',
                'paper_approximate_active_parameters': 200_000_000,
                'paper_approximate_total_parameters': 453_000_000,
                'is_Ultra_2_4B': False}


class Dasheng12BMask(_AdapterBase):
    """8->D 3x3 stem -> 375 grid tokens -> official blocks/norm -> mask.

    Bypasses waveform/mel patch frontend and old mel positional parameters.
    New 2D time/band positions are trainable; this is a task adaptation, not the
    original audio-encoder input distribution or a claimed pretrained regressor.
    """
    family = 'dasheng_1.2b'

    def __init__(self, backbone: nn.Module, *, evidence: PretrainedEvidence | None = None,
                 fixture_mode=False, rank=8, lora_dropout=.1):
        super().__init__()
        if type(fixture_mode) is not bool:
            raise TypeError('fixture_mode must be an explicit boolean')
        if not fixture_mode:
            if evidence is None: raise FoundationAvailabilityError('Inject a verified Dasheng-1.2B pretrained object with exact checkpoint evidence')
            evidence.validate(self.family)
        elif evidence is not None: raise ValueError('Fixture mode cannot carry pretrained evidence/claim')
        if not isinstance(backbone, nn.Module): raise TypeError('Inject a backbone object')
        encoder = getattr(backbone, 'encoder', backbone)
        if not all(hasattr(encoder, key) for key in ['embed_dim', 'blocks', 'norm']):
            raise TypeError('Expected official Dasheng encoder embed_dim/blocks/norm protocol')
        self.width, self.depth = int(encoder.embed_dim), len(encoder.blocks)
        if self.width < 1 or self.depth < 1: raise ValueError('Empty Dasheng encoder')
        if not fixture_mode and (self.width, self.depth) != (1536, 40):
            raise ValueError('Dasheng-1.2B requires width1536/depth40; no silent Base/0.6B substitution')
        if not fixture_mode and encoder.__class__.__name__ not in {'AudioTransformerMAE_Encoder', 'Dasheng'}:
            raise TypeError('Unreviewed Dasheng encoder class; review its official token/block API before use')
        targets = []
        for i, block in enumerate(encoder.blocks):
            attention = getattr(block, 'attn', None)
            if attention is None: raise TypeError('Expected Dasheng blocks[i].attn')
            linear = _require_linear(attention, 'qkv', self.width, 3*self.width)
            if not fixture_mode and getattr(attention, 'num_heads', None) != 24:
                raise ValueError('Dasheng-1.2B requires24heads')
            targets.append((attention, linear, f'blocks.{i}.attn.qkv'))
        self.backbone = encoder; self.fixture_mode = bool(fixture_mode); self.evidence = evidence
        self.rank, self.lora_dropout = rank, float(lora_dropout)
        encoder.requires_grad_(False)
        self.lora_paths = [path for _, _, path in targets]
        for attention, linear, _ in targets:
            attention.qkv = LoRALinear(linear, self.rank, dropout=self.lora_dropout)
        self.stem = nn.Conv2d(8, self.width, kernel_size=3, padding=1)
        self.time_position = nn.Parameter(torch.zeros(1, 15, 1, self.width))
        self.band_position = nn.Parameter(torch.zeros(1, 1, 25, self.width))
        self.mask_head = nn.Linear(self.width, 1)
        self.presence_head = nn.Linear(self.width, 1)
        self.train(True)

    def _hidden(self, x):
        h = self.stem(x).permute(0, 2, 3, 1)+self.time_position+self.band_position
        h = h.reshape(len(x), 375, self.width)
        h = h.to(next(self.backbone.parameters()).dtype)
        # The verified HF API blocks are Sequential; author API also accepts x.
        h = self.backbone.norm(self.backbone.blocks(h))
        if h.shape != (len(x), 375, self.width): raise RuntimeError('Dasheng blocks/norm grid-token protocol mismatch')
        return h.to(self.mask_head.weight.dtype).reshape(len(x), 15, 25, self.width)

    def forward(self, x, energy_weights):
        w = _weights(x, energy_weights); h = self._hidden(x)
        pooled = (h*w.to(h.dtype)[..., None]).sum((1, 2))
        return _output(self.mask_head(h)[..., 0], self.presence_head(pooled), w)

    def representation(self, x):
        return self._hidden(x).mean((1, 2))

    def adapter_manifest(self):
        return {**self._common_manifest(), 'tokens': 375, 'token_order': 'time-major then band',
                'position': 'new learned factorized time15 plus band25 positions',
                'bypassed': ['waveform_frontend', 'mel_patch_embed', 'original_mel_positions', 'original_outputlayer'],
                'router_lora': False, 'self_attention_lora': 'all rows of official fused qkv'}
