"""R6-v2 small mask CNN with cross-frequency context and a fixed score.

This is a pre-fit architecture definition, not a claim of SOTA or improved
presence AUC. Its score was not calibrated against query/LB statistics.
The forward call accepts x,w only; q/y labels are never inference inputs.
The same model and derived score support both relative8 and legacy8 arms.
"""
from __future__ import annotations

import math
import torch
from torch import nn


SCORE_BAND_START = 10
SCORE_BAND_STOP = 23  # 10..22 inclusive
SCORE_EPSILON = 1e-6
CONTEXT_BAND_SLICES = ((10, 20), (20, 25), (15, 23))


def fixed_presence_score(cell_q, energy_weights):
    """Return (score,share,contrast); observed-only, no learned presence head.

    share=sum(w*q). For f in 10..22, b_f=min_{observed t}q[t,f].
    contrast=max_{observed t,f in 10..22} relu(q[t,f]-b_f), or zero if absent.
    A band with only one observed minute contributes zero contrast.
    score=.5*share+.5*contrast, hence [0,1] for bounded q and normalized w.
    """
    if cell_q.ndim != 3 or cell_q.shape[1:] != (15, 25) or energy_weights.shape != cell_q.shape:
        raise ValueError('Expected matching q,w[N,15,25]')
    observed = energy_weights > 0
    share = (cell_q * energy_weights).sum(dim=(1, 2))
    q = cell_q[:, :, SCORE_BAND_START:SCORE_BAND_STOP]
    mask = observed[:, :, SCORE_BAND_START:SCORE_BAND_STOP]
    floor = q.masked_fill(~mask, float('inf')).amin(dim=1, keepdim=True)
    floor = torch.where(mask.any(dim=1, keepdim=True), floor, torch.zeros_like(floor))
    difference = torch.where(mask, torch.relu(q-floor), torch.zeros_like(q))
    contrast = difference.amax(dim=(1, 2))
    return .5*share + .5*contrast, share, contrast


class _ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(4, width), nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(4, width),
        )

    def forward(self, x):
        return torch.nn.functional.gelu(x+self.body(x))


class TinyMaskCNNv2(nn.Module):
    """Original tiny body, optional per-minute global frequency residual.

    At each minute, concatenate hidden-band means from 10:20, 20:25 and
    15:23. Context=GELU(project(concat))*sigmoid(gate(concat)), broadcast to
    every one of the 25 bands and add to the body output before the q head.
    This is a hidden representation, not an estimated acoustic component.

    forward(x,w) -> (share, logit(clamp(fixed_score,1e-6,1-1e-6)), cell_q).
    BCE auxiliary loss acts on this DERIVED logit; no free classifier exists.
    Sigmoid(logit) returns the clamped score; inspect fixed_presence_score()
    if the exact unclamped endpoint is needed for a numeric audit.
    """
    def __init__(self, input_channels=8, width=32, blocks=3, dropout=.1, global_context=True):
        super().__init__()
        if (isinstance(input_channels, bool) or not isinstance(input_channels, int) or input_channels < 1
                or isinstance(width, bool) or not isinstance(width, int) or width < 1 or width % 4
                or isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1
                or not math.isfinite(dropout) or not 0 <= dropout < 1
                or type(global_context) is not bool):
            raise ValueError('Invalid channels/width/blocks/dropout/context')
        self.input_channels = input_channels
        self.width = width
        self.blocks = blocks
        self.dropout = float(dropout)
        self.global_context = global_context
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, padding=1), nn.GELU(),
            *[_ResidualBlock(width, dropout) for _ in range(blocks)],
        )
        self.mask_head = nn.Conv2d(width, 1, 1)
        if global_context:
            self.context_projection = nn.Conv1d(3*width, width, 1)
            self.context_gate = nn.Conv1d(3*width, width, 1)

    def add_global_context(self, hidden):
        if not self.global_context:
            return hidden
        pooled = torch.cat([hidden[:, :, :, a:b].mean(dim=3)
                            for a,b in CONTEXT_BAND_SLICES], dim=1)
        context = torch.nn.functional.gelu(self.context_projection(pooled)) * torch.sigmoid(self.context_gate(pooled))
        return hidden + context.unsqueeze(3)

    def forward(self, x, energy_weights):
        if (x.ndim != 4 or x.shape[1:] != (self.input_channels, 15, 25)
                or energy_weights.shape != (x.shape[0], 15, 25)):
            raise ValueError('Expected x[N,C,15,25],w[N,15,25]')
        h = self.add_global_context(self.body(x))
        q = torch.sigmoid(self.mask_head(h)[:, 0])
        q = torch.where(energy_weights > 0, q, torch.zeros_like(q))
        score, share, _ = fixed_presence_score(q, energy_weights)
        return share, torch.logit(score.clamp(SCORE_EPSILON, 1-SCORE_EPSILON)), q

    def representation(self, x):
        return self.add_global_context(self.body(x)).mean(dim=(2, 3))
