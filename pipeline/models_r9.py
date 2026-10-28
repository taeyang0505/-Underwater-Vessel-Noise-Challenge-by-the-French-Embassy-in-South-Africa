"""R9 fresh-weight adapters over four existing energy-grid backbones.

The caller supplies ONLY ``relative_grid_v2(db)``'s eight R6 channels and
rounded observed-cell energy weights. There is no absolute-dB input,
background-prior fit/query, source identifier, or pretrained checkpoint here.
These are task adaptations, not complete reproductions of published SOTA.
WideCNN additionally receives a parameter-free observed-frequency global
mean residual per channel/minute before its q head; this is new R9 wiring.

The existing q backbones and q heads are retained. Their separate learned
presence heads are removed; score is the unchanged R6 mask-derived diagnostic
``.5 * energy_weighted_share + .5 * observed_mid/high_mask_contrast``.
All four branches consequently use the same physical pooling/score contract.

Training policy belongs to the runner: fresh initialization, Adam, dropout
0.1, 100 epochs without early stopping. This module never trains or loads
weights and does not encode a performance expectation.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn

from pipeline.relative_grid_r6_v2 import FEATURE_NAMES
from pipeline.relative_models_r6_v2 import fixed_presence_score, SCORE_EPSILON


MODEL_DEFAULTS = {
    'bg_cnn': {'width': 128, 'blocks': 5, 'dropout': .1},
    'tfgrid_fdy_conformer': {
        'embed_dim': 64, 'hidden_size': 128, 'grid_blocks': 5,
        'dropout': .1, 'conformer_blocks': 1,
    },
    'fdy': {'width': 32, 'blocks': 3, 'dropout': .1},
    'itransformer': {'width': 96, 'blocks': 3, 'dropout': .1},
}
INPUT_POLICY = 'r6_relative_grid_v2_8channels_rounded_weights_v1'
SCORE_POLICY = 'r6_fixed_half_share_half_observed_mask_contrast_bands10_22_v1'


def _settings(kind, overrides):
    if kind not in MODEL_DEFAULTS:
        raise ValueError('Unsupported R9 neural kind: ' + str(kind))
    if set(overrides) - set(MODEL_DEFAULTS[kind]):
        raise ValueError('Unknown or input-changing model settings')
    resolved = dict(MODEL_DEFAULTS[kind], **overrides)
    for name, value in resolved.items():
        if name == 'dropout':
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError('Finite dropout in [0,1) required')
        elif type(value) is not int or value < 1:
            raise ValueError('Positive integer model dimensions required')
    if kind in {'bg_cnn', 'fdy', 'itransformer'} and resolved['width'] % 4:
        raise ValueError('Width must be divisible by four')
    if kind == 'tfgrid_fdy_conformer' and resolved['embed_dim'] % 4:
        raise ValueError('Embedding width must be divisible by four')
    return resolved


class R9RelativeMaskAdapter(nn.Module):
    """Strict ``(x8,w) -> (share, derived_score_logit, observed_q)`` API.

    x[5] must be the observation mask, and x[7] must be sqrt(w). These
    checks reject a legacy R5 eight-channel tensor despite its equal shape.
    No normalization, fitting, prior loading, or input reinterpretation is
    performed. q is exactly zero at cells whose observed energy weight is 0.
    """
    def __init__(self, kind, **settings):
        super().__init__()
        resolved = _settings(kind, settings)
        if kind in {'bg_cnn', 'tfgrid_fdy_conformer'}:
            from pipeline.hybrid_models_r5 import make_hybrid_model
            self.backbone = make_hybrid_model(kind, **resolved)
            head_parent = self.backbone.network if kind == 'bg_cnn' else self.backbone.grid
        else:
            from pipeline.models import FDYMaskCRNN, InvertedMaskTransformer
            cls = FDYMaskCRNN if kind == 'fdy' else InvertedMaskTransformer
            self.backbone = cls(input_channels=8, **resolved)
            head_parent = self.backbone
        self.base_total_parameters = sum(p.numel() for p in self.backbone.parameters())
        self.removed_presence_parameters = sum(p.numel() for p in head_parent.presence_head.parameters())
        del head_parent.presence_head
        self.kind = kind
        self.model_config = resolved
        self.architecture_version = 9
        self.input_channels = 8
        self.input_policy = INPUT_POLICY
        self.score_policy = SCORE_POLICY
        self.requires_background_prior = False
        self.pretrained = False

    @staticmethod
    def _validate_x(x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or tuple(x.shape[1:]) != (8, 15, 25) or len(x) < 1:
            raise ValueError('Expected nonempty R6 x[N,8,15,25]')
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('Finite floating-point R6 features required')
        if not ((x[:, 5] == 0) | (x[:, 5] == 1)).all():
            raise ValueError('R6 feature channel 5 must be a binary observed-cell mask')
        if (x.masked_select((x[:, 5] == 0)[:, None].expand_as(x)) != 0).any():
            raise ValueError('All R6 feature channels must be zero at missing cells')

    def _validate(self, x, w):
        self._validate_x(x)
        if not isinstance(w, torch.Tensor) or w.shape != (len(x), 15, 25) or w.device != x.device:
            raise ValueError('Energy weights must match x batch/grid/device')
        if not w.is_floating_point() or not torch.isfinite(w).all() or (w < 0).any():
            raise ValueError('Finite nonnegative energy weights required')
        if not torch.equal(x[:, 5] > 0, w > 0):
            raise ValueError('Observed-cell mask and positive energy weights disagree')
        if not torch.allclose(w.float().sum((1, 2)), torch.ones(len(x), device=w.device), atol=3e-6, rtol=0):
            raise ValueError('Per-window energy weights must sum to one')
        if not torch.allclose(x[:, 7].float(), w.float().sqrt(), atol=3e-6, rtol=0):
            raise ValueError('R6 feature channel 7 must equal sqrt of the SAME pooling weights')

    def _cell_mask(self, x):
        if self.kind == 'bg_cnn':
            return self.backbone.network.mask_head(self.encode(x))[:, 0].sigmoid()
        if self.kind == 'tfgrid_fdy_conformer':
            return self.backbone.grid.mask_head(self.backbone.encode(x))[:, 0].sigmoid()
        h = self.backbone._hidden(x)
        if self.kind == 'fdy':
            return self.backbone.mask_head(h).squeeze(-1).transpose(1, 2).sigmoid()
        return self.backbone.mask_head(h).transpose(1, 2).sigmoid()

    def encode(self, x):
        """WideCNN body plus masked global-frequency residual, same 25 bands.

        c[b,c,t,1]=sum_f O[b,t,f]*h[b,c,t,f]/sum_f O[b,t,f].
        An unobserved minute contributes c=0. Broadcasting c to every band
        gives an explicit all-frequency dependency without a new parameter.
        The five 2-convolution residual blocks + stem have a theoretical
        temporal receptive field of23, intersected with the15-minute input;
        this is not a promise of all15times at every edge output location.
        """
        self._validate_x(x)
        if self.kind != 'bg_cnn':
            raise ValueError('R9 encode grid API is currently the WideCNN branch only')
        h = self.backbone.encode(x)
        observed = x[:, 5:6].to(dtype=h.dtype)
        denominator = observed.sum(dim=3, keepdim=True).clamp_min(1)
        context = (h*observed).sum(dim=3, keepdim=True)/denominator
        return h+context

    def forward(self, x, energy_weights):
        self._validate(x, energy_weights)
        # FP32 pooling keeps the physical constraint stable under CUDA BF16
        # autocast, while convolution/recurrent operators may use autocast.
        q = self._cell_mask(x).float()
        q = torch.where(energy_weights > 0, q, torch.zeros_like(q))
        score, share, _ = fixed_presence_score(q, energy_weights.float())
        return share, torch.logit(score.clamp(SCORE_EPSILON, 1-SCORE_EPSILON)), q

    def representation(self, x):
        """Optional hidden embedding; no labels/prior/context are accepted."""
        self._validate_x(x)
        if self.kind == 'bg_cnn':
            return self.encode(x).mean(dim=(2, 3))
        return self.backbone.representation(x)

    def architecture(self):
        """JSON-safe reconstruction identity; weights are supplied separately."""
        value = {
            'format': 'r9_relative_mask_adapter_v1', 'architecture_version': 9,
            'kind': self.kind, 'model_settings': deepcopy(self.model_config),
            'input_policy': INPUT_POLICY, 'feature_names': list(FEATURE_NAMES),
            'score_policy': SCORE_POLICY, 'pretrained': False,
            'requires_background_prior': False,
            'base_total_parameters': self.base_total_parameters,
            'removed_presence_parameters': self.removed_presence_parameters,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        if self.kind == 'bg_cnn':
            value['frequency_context_policy'] = 'per_minute_observed_frequency_mean_additive_residual_v1'
            value['frequency_context_new_parameters'] = 0
            value['frequency_context_uses_labels_or_prior'] = False
            value['body_theoretical_temporal_receptive_field'] = 1+2*(1+2*self.model_config['blocks'])
            value['body_receptive_field_clipped_to_input'] = True
        return value


def make_r9_model(kind, **settings):
    """Construct a new random-weight adapter; no checkpoint/data reads occur."""
    return R9RelativeMaskAdapter(kind, **settings)
