"""R6 small CNN plus backwards-compatible relative-grid re-exports.

Pre-fit engine correction: pure NumPy feature definitions now live in
pipeline.relative_grid_r6 so CPU tree workers need not import Torch.
This wrapper keeps all previous public names and the CNN architecture.
No numerical feature, label, loss or model setting changed in this split.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pipeline.relative_grid_r6 import (
    FEATURE_NAMES, LOW_BAND_COUNT, MID_HIGH_BAND_START, QUANTIZATION_DECIMALS,
    quantization_pooling_bound, quantized_db, relative_grid,
)


class _ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(4, width), nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(4, width),
        )

    def forward(self, x):
        return torch.nn.functional.gelu(x + self.body(x))


class TinyMaskCNN(nn.Module):
    """Small residual mask CNN; input_channels=8 permits a width-matched control.

    forward(x,w) returns (physically_weighted_share, presence_logit, cell_q),
    matching the existing training tuple order. q is zero outside positive w.
    w must come from the same db representation as the desired pooling rule.
    This network receives neither labels nor source tags in its forward call.
    """
    def __init__(self, input_channels=6, width=32, blocks=3, dropout=.1):
        super().__init__()
        if (isinstance(input_channels, bool) or not isinstance(input_channels, int)
                or input_channels <= 0 or isinstance(width, bool)
                or not isinstance(width, int) or width <= 0 or width % 4
                or isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1
                or not np.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError('Invalid channels/width/blocks/dropout')
        self.input_channels = input_channels
        self.width = width
        self.blocks = blocks
        self.dropout = float(dropout)
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, padding=1), nn.GELU(),
            *[_ResidualBlock(width, dropout) for _ in range(blocks)],
        )
        self.mask_head = nn.Conv2d(width, 1, 1)
        self.presence_head = nn.Linear(width, 1)

    def forward(self, x, energy_weights):
        if (x.ndim != 4 or x.shape[1:] != (self.input_channels, 15, 25)
                or energy_weights.shape != (x.shape[0], 15, 25)):
            raise ValueError('Expected x[N,C,15,25] and w[N,15,25]')
        h = self.body(x)
        cell = torch.sigmoid(self.mask_head(h)[:, 0])
        cell = torch.where(energy_weights > 0, cell, torch.zeros_like(cell))
        share = (cell * energy_weights).sum(dim=(1, 2))
        presence_logit = self.presence_head(h.mean(dim=(2, 3)))[:, 0]
        return share, presence_logit, cell

    def representation(self, x):
        return self.body(x).mean(dim=(2, 3))
