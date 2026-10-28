"""Fresh, versioned F1 compatibility adapter; not a production training runner.

The old relative8 checkpoints stay immutable. These four task-adapted
backbones accept relative8 plus nine ordered F1 channels. Regression and
classification have disjoint parameters; score never uses share or weights.
"""
from copy import deepcopy
import torch
from torch import nn
from pipeline.models_r9 import R9RelativeMaskAdapter
from pipeline.relative_grid_r6_v2 import FEATURE_NAMES as BASE_NAMES
from ops.r16_temporal_features import GRID_NAMES

FEATURE_NAMES = tuple(BASE_NAMES) + tuple(GRID_NAMES)
CLASSIFIER_CHANNELS = (0, 1, 2, 3, 6, 5) + tuple(range(8, 17))
INPUT_POLICY = 'r16_f1_relative17_fresh_adapter_preflight_v1'


def expand_conv(old):
    new = nn.Conv2d(17, old.out_channels, old.kernel_size, old.stride,
                    old.padding, old.dilation, old.groups,
                    bias=old.bias is not None, padding_mode=old.padding_mode)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :8].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


class F1MaskAdapter(R9RelativeMaskAdapter):
    def __init__(self, kind, **settings):
        super().__init__(kind, **settings)
        b = self.backbone
        if kind == 'bg_cnn':
            b.network.body[0] = expand_conv(b.network.body[0])
        elif kind == 'tfgrid_fdy_conformer':
            b.grid.stem[0] = expand_conv(b.grid.stem[0])
            b.grid.input_channels = 17
        elif kind == 'fdy':
            b.stem = expand_conv(b.stem)
        elif kind == 'itransformer':
            old = b.embed
            new = nn.Linear(17 * 15, old.out_features,
                            bias=old.bias is not None)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, :8 * 15].copy_(old.weight)
                if old.bias is not None:
                    new.bias.copy_(old.bias)
            b.embed = new
            b.input_channels = 17
        self.input_channels = 17
        self.input_policy = INPUT_POLICY

    @staticmethod
    def _validate_x(x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or tuple(x.shape[1:]) != (17, 15, 25):
            raise ValueError('Expected F1 x[N,17,15,25]')
        R9RelativeMaskAdapter._validate_x(x[:, :8])
        if not torch.isfinite(x[:, 8:]).all():
            raise ValueError('F1 channels must be finite')
        if (x[:, 8:].masked_select((x[:, 5] == 0)[:, None].expand_as(x[:, 8:])) != 0).any():
            raise ValueError('F1 features must be zero at missing cells')

    def encode(self, x):
        self._validate_x(x)
        if self.kind != 'bg_cnn':
            raise ValueError('Grid encoding entry point is the CNN only')
        h = self.backbone.network.body(x)
        observed = x[:, 5:6].to(h.dtype)
        context = (h * observed).sum(3, keepdim=True) / observed.sum(3, keepdim=True).clamp_min(1)
        return h + context

    def _cell_mask(self, x):
        if self.kind != 'tfgrid_fdy_conformer':
            return super()._cell_mask(x)
        b = self.backbone
        # Bypass only the legacy fixed-8 shape check, retain all grid blocks.
        h = b.grid.stem(x)
        for block in b.grid.blocks:
            h = block(h)
        h = b.dynamic(h)
        n, c, t, f = h.shape
        temporal = h.permute(0, 3, 2, 1).reshape(n * f, t, c)
        h = b.conformer(temporal).reshape(n, f, t, c).permute(0, 3, 2, 1)
        return b.grid.mask_head(h)[:, 0].sigmoid()


class IndependentF1Classifier(nn.Module):
    def __init__(self, dropout=.1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(15, 16, 3, padding=1), nn.GELU(),
                                  nn.Dropout2d(dropout), nn.Conv2d(16, 16, 3, padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Dropout(dropout), nn.Linear(16, 1))

    def forward(self, x):
        observed = x[:, 5] > 0
        h = self.body(x[:, list(CLASSIFIER_CHANNELS)])
        o = observed[:, None].to(h.dtype)
        avg = (h * o).sum((2, 3)) / o.sum((2, 3)).clamp_min(1)
        maximum = h.masked_fill(~observed[:, None], -torch.inf).amax((2, 3))
        return self.head(torch.cat([avg, maximum], 1))[:, 0]


class R16F1PreflightModel(nn.Module):
    def __init__(self, kind, **settings):
        super().__init__()
        self.regressor = F1MaskAdapter(kind, **settings)
        self.classifier = IndependentF1Classifier(self.regressor.model_config['dropout'])
        self.kind = kind

    def forward(self, x, weights):
        self.regressor._validate(x, weights)
        q = self.regressor._cell_mask(x).float()
        q = torch.where(weights > 0, q, torch.zeros_like(q))
        return (q * weights.float()).sum((1, 2)), self.classifier(x), q

    def architecture(self):
        return {'format': INPUT_POLICY, 'kind': self.kind,
                'settings': deepcopy(self.regressor.model_config),
                'feature_names': list(FEATURE_NAMES),
                'classifier_channels': list(CLASSIFIER_CHANNELS),
                'regression_parameters': sum(p.numel() for p in self.regressor.parameters()),
                'classification_parameters': sum(p.numel() for p in self.classifier.parameters()),
                'shared_parameters': 0, 'fresh_initialization': True,
                'production_training_started': False}
