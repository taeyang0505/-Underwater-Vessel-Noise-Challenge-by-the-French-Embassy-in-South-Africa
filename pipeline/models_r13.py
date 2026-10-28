"""Fresh full-size R13 mask adapters, without changing R9 checkpoints/code.

The R9 backbones are retained; their input stems accept11 channels. P0 uses
the same11-input parameter budget with the three shape maps zero padded.
These are energy-grid task adaptations, not official pretrained SOTA models.
No contrast-derived classifier/BCE path exists: score is the pooled share.
"""
from copy import deepcopy
import torch
from torch import nn
from pipeline.models_r9 import MODEL_DEFAULTS, _settings
from pipeline.features_r12 import FEATURE_NAMES

INPUT_POLICY = 'r13_relative8_plus_TRAIN_safe23_PCA3_11channels_v1'
SCORE_POLICY = 'pooled_share_probability_no_contrast_classifier_v1'


class R13MaskAdapter(nn.Module):
    def __init__(self, kind, **settings):
        super().__init__()
        resolved = _settings(kind, settings)
        self.kind, self.model_config = kind, resolved
        if kind == 'bg_cnn':
            from pipeline.models import EnergyMaskCNN
            self.backbone = EnergyMaskCNN(input_channels=11, **resolved)
            del self.backbone.presence_head
        elif kind == 'tfgrid_fdy_conformer':
            from pipeline.hybrid_models_r5 import TFGridFDYConformerMask
            self.backbone = TFGridFDYConformerMask(**resolved)
            c = resolved['embed_dim']
            self.backbone.grid.stem[0] = nn.Conv2d(11, c, 3, padding=1)
            del self.backbone.grid.presence_head
            # encode below explicitly executes the unchanged body, avoiding
            # the legacy8 input validator without patching its source/methods.
        else:
            from pipeline.models import FDYMaskCRNN, InvertedMaskTransformer
            cls = FDYMaskCRNN if kind == 'fdy' else InvertedMaskTransformer
            self.backbone = cls(input_channels=11, **resolved)
            del self.backbone.presence_head
        self.architecture_version = 13
        self.input_channels = 11
        self.input_policy = INPUT_POLICY
        self.score_policy = SCORE_POLICY
        self.pretrained = False
        self.requires_background_prior = False

    def _validate(self, x, w=None):
        if (not isinstance(x, torch.Tensor) or x.ndim != 4
                or tuple(x.shape[1:]) != (11, 15, 25) or not len(x)
                or not x.is_floating_point() or not torch.isfinite(x).all()):
            raise ValueError('Finite x[N,11,15,25] required')
        mask = x[:, 5] > 0
        if not ((x[:, 5] == 0) | (x[:, 5] == 1)).all():
            raise ValueError('Channel5 is a binary observation mask')
        if (x.masked_select((~mask)[:, None].expand_as(x)) != 0).any():
            raise ValueError('Every channel must be zero at missing cells')
        if w is not None:
            if (w.shape != (len(x), 15, 25) or w.device != x.device
                    or not torch.isfinite(w).all() or (w < 0).any()
                    or not torch.equal(mask, w > 0)):
                raise ValueError('Finite matching observed energy weights required')
            if not torch.allclose(w.float().sum((1, 2)), torch.ones(len(x), device=x.device), atol=3e-6, rtol=0):
                raise ValueError('Weights must sum to1 perwindow')
            if not torch.allclose(x[:, 7].float(), w.float().sqrt(), atol=3e-6, rtol=0):
                raise ValueError('Channel7 must equal sqrt of the same physical weights')

    def encode(self, x):
        self._validate(x)
        if self.kind == 'bg_cnn':
            h = self.backbone.body(x)
            observed = x[:, 5:6].to(h.dtype)
            context = (h*observed).sum(3, keepdim=True)/observed.sum(3, keepdim=True).clamp_min(1)
            return h+context
        if self.kind == 'tfgrid_fdy_conformer':
            h = self.backbone.grid.stem(x)
            for block in self.backbone.grid.blocks:
                h = block(h)
            h = self.backbone.dynamic(h)
            n, c, t, f = h.shape
            temporal = h.permute(0, 3, 2, 1).reshape(n*f, t, c)
            return self.backbone.conformer(temporal).reshape(n, f, t, c).permute(0, 3, 2, 1)
        return self.backbone._hidden(x)

    def forward(self, x, energy_weights):
        self._validate(x, energy_weights)
        h = self.encode(x)
        if self.kind == 'bg_cnn':
            q = self.backbone.mask_head(h)[:, 0].sigmoid()
        elif self.kind == 'tfgrid_fdy_conformer':
            q = self.backbone.grid.mask_head(h)[:, 0].sigmoid()
        elif self.kind == 'fdy':
            q = self.backbone.mask_head(h).squeeze(-1).transpose(1, 2).sigmoid()
        else:
            q = self.backbone.mask_head(h).transpose(1, 2).sigmoid()
        q = torch.where(energy_weights > 0, q.float(), torch.zeros_like(energy_weights.float()))
        p = (q*energy_weights.float()).sum((1, 2))
        return p, p, q

    def architecture(self):
        return {'format': 'r13_full_mask_adapter_v1', 'architecture_version': 13,
            'kind': self.kind, 'model_settings': deepcopy(self.model_config),
            'input_channels': 11, 'input_policy': INPUT_POLICY,
            'feature_names': list(FEATURE_NAMES), 'score_policy': SCORE_POLICY,
            'pretrained': False, 'requires_background_prior': False,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'R9_code_and_old_checkpoint_shapes_unchanged': True,
            'R13_new_input_stem': True,
            'frequency_context': 'masked all-band mean residual retained fromR9' if self.kind == 'bg_cnn' else None}


def make_r13_model(kind, **settings):
    return R13MaskAdapter(kind, **settings)
