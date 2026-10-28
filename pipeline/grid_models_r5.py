"""R5 energy-mask adaptations of published time/frequency operators.

These are independently written task adaptations, not reproductions of the
papers' complex-STFT/waveform separators and not a claim of current SOTA.
No vendor code, pretrained weights, downloads, STFT, or waveform decoder is
used. Inputs are eight real feature channels on the fixed 15 x 25 energy grid.

TF-grid sources:
  https://arxiv.org/abs/2211.12433
  https://github.com/espnet/espnet/pull/5395
Band-split sources:
  https://arxiv.org/abs/2209.15174

Adaptation deltas: the TF-grid front end is 8->E rather than complex RI input;
frequency unfolding uses the paper's I=J=1 option; the output is a direct
sigmoid energy-fraction mask. Frequency BLSTM, temporal BLSTM, and full-band
cross-frame attention remain distinct operators. Band-split uses five groups
of five already aggregated deci-decade bands, rather than variable-width FFT
bins. Its band-specific MLP/GLU decoders emit five real logits each. Both
models pool their 25-band masks with caller-supplied normalized observed-cell
energy weights; they never force an output prior or apply a residual penalty.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import time

import torch
from torch import nn


GRID_FRAMES = 15
GRID_BANDS = 25
GRID_CHANNELS = 8
BAND_GROUP_SLICES = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25))
HEADER_CENTRES_HZ = (
    31.6, 39.8, 50.1, 63.1, 79.4, 100., 125.9, 158.5, 199.5, 251.2,
    316.2, 398.1, 501.2, 631., 794.3, 1000., 1258.9, 1584.9, 1995.3,
    2511.9, 3162.3, 3981.1, 5011.9, 6309.6, 7943.3,
)


def band_group_metadata():
    """Exact zero-based half-open indices and base-10 support edges in Hz."""
    return [
        {
            'group': i, 'indices_half_open': [start, end],
            'header_centres_hz': list(HEADER_CENTRES_HZ[start:end]),
            'exact_nominal_centres_hz': [10 ** (1.5 + .1 * j) for j in range(start, end)],
            'support_edges_hz': [10 ** (1.5 + .1 * start - .05),
                                 10 ** (1.5 + .1 * (end - 1) + .05)],
        }
        for i, (start, end) in enumerate(BAND_GROUP_SLICES)
    ]


def _validate_dimensions(input_channels, n_frames, n_bands, embed_dim, hidden_size, blocks, dropout):
    if (input_channels, n_frames, n_bands) != (GRID_CHANNELS, GRID_FRAMES, GRID_BANDS):
        raise ValueError('R5 models require exactly eight channels, 15 frames and 25 bands')
    if any(type(v) is not int or v < 1 for v in (embed_dim, hidden_size, blocks)):
        raise ValueError('Embedding, recurrent width and block count must be positive integers')
    if not 0 <= dropout < 1:
        raise ValueError('dropout must be in [0,1)')


def _validate_input(x):
    if x.ndim != 4 or tuple(x.shape[1:]) != (GRID_CHANNELS, GRID_FRAMES, GRID_BANDS) or x.shape[0] < 1:
        raise ValueError('Expected x with shape [B,8,15,25] and a nonempty batch')
    if not x.is_floating_point():
        raise ValueError('Real floating-point feature inputs are required')


def _pool_mask(cell_mask, energy_weights):
    """The caller supplies nonnegative observed-cell weights summing to one.

    Weight normalization and target/mask validity are dataset/trainer gates.
    No re-normalization, population statistic or query fitting occurs here.
    """
    if energy_weights.shape != cell_mask.shape or energy_weights.device != cell_mask.device:
        raise ValueError('Energy weights must match the [B,15,25] mask and device')
    return (cell_mask * energy_weights).sum(dim=(1, 2))


class _AxisBLSTM(nn.Module):
    """Pre-LN BLSTM -> linear projection -> dropout -> residual on one axis."""
    def __init__(self, embed_dim, hidden_size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(2 * hidden_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        update, _ = self.rnn(self.norm(sequence))
        return sequence + self.dropout(self.projection(update))


class FullBandCrossFrameAttention(nn.Module):
    """Attend across frames using all 25 bands in each head's Q, K and V.

    Attention scores have shape [B,H,T,T]. Each frame's Q/K concatenates
    per-band projections, rather than averaging the frequency axis. V also
    keeps all band positions, so output resolution stays [B,E,T,25].
    """
    def __init__(self, embed_dim=64, heads=4, n_bands=25, approx_qk_dim=512, dropout=.1):
        super().__init__()
        if type(heads) is not int or heads < 1 or embed_dim % heads:
            raise ValueError('Embedding dimension must be divisible by the attention head count')
        if type(approx_qk_dim) is not int or approx_qk_dim < 1:
            raise ValueError('Attention Q/K dimension must be a positive integer')
        self.heads = heads
        self.n_bands = n_bands
        self.qk_per_band = math.ceil(approx_qk_dim / n_bands)
        self.value_per_band = embed_dim // heads
        self.query_projection = nn.Conv2d(embed_dim, heads * self.qk_per_band, 1)
        self.key_projection = nn.Conv2d(embed_dim, heads * self.qk_per_band, 1)
        self.value_projection = nn.Conv2d(embed_dim, embed_dim, 1)
        self.query_activation = nn.PReLU(heads * self.qk_per_band)
        self.key_activation = nn.PReLU(heads * self.qk_per_band)
        self.value_activation = nn.PReLU(embed_dim)
        self.query_norm = nn.ModuleList([nn.LayerNorm((self.qk_per_band, n_bands)) for _ in range(heads)])
        self.key_norm = nn.ModuleList([nn.LayerNorm((self.qk_per_band, n_bands)) for _ in range(heads)])
        self.value_norm = nn.ModuleList([nn.LayerNorm((self.value_per_band, n_bands)) for _ in range(heads)])
        self.attention_dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 1), nn.GroupNorm(1, embed_dim), nn.Dropout(dropout))

    def _heads(self, value, width, norms):
        batch, _, frames, bands = value.shape
        value = value.reshape(batch, self.heads, width, frames, bands).permute(0, 1, 3, 2, 4)
        value = torch.stack([norms[h](value[:, h]) for h in range(self.heads)], dim=1)
        return value.reshape(batch, self.heads, frames, width * bands)

    def forward(self, x):
        batch, embedding, frames, bands = x.shape
        if bands != self.n_bands:
            raise ValueError('Full-band attention frequency resolution differs')
        q = self._heads(self.query_activation(self.query_projection(x)), self.qk_per_band, self.query_norm)
        k = self._heads(self.key_activation(self.key_projection(x)), self.qk_per_band, self.key_norm)
        v = self._heads(self.value_activation(self.value_projection(x)), self.value_per_band, self.value_norm)
        attention = (q @ k.transpose(-2, -1) / math.sqrt(self.qk_per_band * bands)).softmax(dim=-1)
        update = self.attention_dropout(attention) @ v
        update = update.reshape(batch, self.heads, frames, self.value_per_band, bands)
        update = update.permute(0, 1, 3, 2, 4).reshape(batch, embedding, frames, bands)
        return x + self.output(update)


class TFGridBlock(nn.Module):
    """Full-band frequency BLSTM, sub-band temporal BLSTM, then attention."""
    def __init__(self, embed_dim=64, hidden_size=128, dropout=.1, heads=4, approx_qk_dim=512):
        super().__init__()
        self.fullband_frequency = _AxisBLSTM(embed_dim, hidden_size, dropout)
        self.subband_temporal = _AxisBLSTM(embed_dim, hidden_size, dropout)
        self.cross_frame_attention = FullBandCrossFrameAttention(embed_dim, heads, GRID_BANDS, approx_qk_dim, dropout)

    def forward(self, x):
        batch, embedding, frames, bands = x.shape
        frequency = x.permute(0, 2, 3, 1).reshape(batch * frames, bands, embedding)
        frequency = self.fullband_frequency(frequency)
        x = frequency.reshape(batch, frames, bands, embedding).permute(0, 3, 1, 2)
        temporal = x.permute(0, 3, 2, 1).reshape(batch * bands, frames, embedding)
        temporal = self.subband_temporal(temporal)
        x = temporal.reshape(batch, bands, frames, embedding).permute(0, 3, 2, 1)
        return self.cross_frame_attention(x)


class TFGridEnergyMask(nn.Module):
    """Real-energy TF-grid core adaptation; no official waveform weights."""
    def __init__(self, embed_dim=64, hidden_size=128, blocks=6, dropout=.1,
                 input_channels=8, n_frames=15, n_bands=25,
                 attention_heads=4, attention_approx_qk_dim=512):
        super().__init__()
        _validate_dimensions(input_channels, n_frames, n_bands, embed_dim, hidden_size, blocks, dropout)
        self.input_channels = input_channels
        self.feature_version = self.preprocessing_version = 3
        self.grid_kind = 'tfgrid_mask'
        self.grid_config = dict(embed_dim=embed_dim, hidden_size=hidden_size, blocks=blocks, dropout=dropout,
            input_channels=input_channels, n_frames=n_frames, n_bands=n_bands,
            attention_heads=attention_heads, attention_approx_qk_dim=attention_approx_qk_dim)
        self.stem = nn.Sequential(nn.Conv2d(input_channels, embed_dim, 3, padding=1), nn.GroupNorm(1, embed_dim))
        self.blocks = nn.ModuleList([TFGridBlock(embed_dim, hidden_size, dropout, attention_heads, attention_approx_qk_dim) for _ in range(blocks)])
        self.mask_head = nn.Conv2d(embed_dim, 1, 1)
        self.presence_head = nn.Linear(embed_dim, 1)

    def encode(self, x):
        _validate_input(x)
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return h

    def representation(self, x):
        return self.encode(x).mean(dim=(2, 3))

    def forward(self, x, energy_weights):
        h = self.encode(x)
        cell = self.mask_head(h)[:, 0].sigmoid()
        share = _pool_mask(cell, energy_weights)
        presence = self.presence_head(h.mean(dim=(2, 3)))[:, 0]
        return share, presence, cell


class BandSplitBlock(nn.Module):
    """Shared time BLSTM then ordered group-axis BLSTM, with residuals."""
    def __init__(self, embed_dim=128, hidden_size=128, dropout=.1):
        super().__init__()
        self.temporal = _AxisBLSTM(embed_dim, hidden_size, dropout)
        self.band = _AxisBLSTM(embed_dim, hidden_size, dropout)

    def forward(self, groups):
        batch, count, frames, embedding = groups.shape
        groups = self.temporal(groups.reshape(batch * count, frames, embedding))
        groups = groups.reshape(batch, count, frames, embedding)
        bands = groups.permute(0, 2, 1, 3).reshape(batch * frames, count, embedding)
        bands = self.band(bands)
        return bands.reshape(batch, frames, count, embedding).permute(0, 2, 1, 3)


class _BandMaskDecoder(nn.Module):
    """Band-specific LN/MLP/tanh/GLU producing real, per-cell logits."""
    def __init__(self, embed_dim, bands, dropout):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 4 * embed_dim),
            nn.Tanh(), nn.Dropout(dropout), nn.Linear(4 * embed_dim, 2 * bands), nn.GLU(dim=-1))

    def forward(self, x):
        return self.network(x)


class BandSplitEnergyMask(nn.Module):
    """Five physical groups -> recurrent core -> five five-band decoders.

    Group slices are [0:5], [5:10], [10:15], [15:20], [20:25]. Each is half a
    decade of frequency support, not five uniformly spaced linear-Hz bins.
    ``encode`` broadcasts each group embedding to its five associated band
    positions for feature/SSL interfaces; output masks still have distinct
    learned logits for every one of the original 25 physical bands.
    """
    def __init__(self, embed_dim=128, hidden_size=128, blocks=4, dropout=.1,
                 input_channels=8, n_frames=15, n_bands=25):
        super().__init__()
        _validate_dimensions(input_channels, n_frames, n_bands, embed_dim, hidden_size, blocks, dropout)
        self.input_channels = input_channels
        self.feature_version = self.preprocessing_version = 3
        self.grid_kind = 'bandsplit_mask'
        self.grid_config = dict(embed_dim=embed_dim, hidden_size=hidden_size, blocks=blocks,
            dropout=dropout, input_channels=input_channels, n_frames=n_frames, n_bands=n_bands)
        self.band_groups = BAND_GROUP_SLICES
        self.group_metadata = band_group_metadata()
        self.group_projections = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(input_channels * (end - start)), nn.Linear(input_channels * (end - start), embed_dim))
            for start, end in self.band_groups
        ])
        self.blocks = nn.ModuleList([BandSplitBlock(embed_dim, hidden_size, dropout) for _ in range(blocks)])
        self.mask_decoders = nn.ModuleList([_BandMaskDecoder(embed_dim, end - start, dropout) for start, end in self.band_groups])
        self.presence_head = nn.Linear(embed_dim, 1)

    def _encode_groups(self, x):
        _validate_input(x)
        tokens = []
        for projection, (start, end) in zip(self.group_projections, self.band_groups):
            part = x[..., start:end].permute(0, 2, 1, 3).reshape(x.shape[0], GRID_FRAMES, -1)
            tokens.append(projection(part))
        h = torch.stack(tokens, dim=1)
        for block in self.blocks:
            h = block(h)
        return h

    def encode(self, x):
        h = self._encode_groups(x)
        return torch.cat([h[:, i, :, :, None].expand(-1, -1, -1, end - start)
                          for i, (start, end) in enumerate(self.band_groups)], dim=-1).permute(0, 2, 1, 3)

    def representation(self, x):
        return self._encode_groups(x).mean(dim=(1, 2))

    def forward(self, x, energy_weights):
        h = self._encode_groups(x)
        logits = torch.cat([decoder(h[:, i]) for i, decoder in enumerate(self.mask_decoders)], dim=-1)
        cell = logits.sigmoid()
        share = _pool_mask(cell, energy_weights)
        presence = self.presence_head(h.mean(dim=(1, 2)))[:, 0]
        return share, presence, cell


def make_grid_model(kind, *, embed_dim=None, hidden_size=128, blocks=None, dropout=.1,
                    input_channels=8, n_frames=15, n_bands=25,
                    attention_heads=4, attention_approx_qk_dim=512):
    """Build one explicitly sized adaptation. No weights or fitted inputs load."""
    common = dict(hidden_size=hidden_size, dropout=dropout, input_channels=input_channels,
                  n_frames=n_frames, n_bands=n_bands)
    if kind == 'tfgrid_mask':
        return TFGridEnergyMask(embed_dim=64 if embed_dim is None else embed_dim,
            blocks=6 if blocks is None else blocks, attention_heads=attention_heads,
            attention_approx_qk_dim=attention_approx_qk_dim, **common)
    if kind == 'bandsplit_mask':
        if attention_heads != 4 or attention_approx_qk_dim != 512:
            raise ValueError('Band-split recurrent adaptation has no attention settings')
        return BandSplitEnergyMask(embed_dim=128 if embed_dim is None else embed_dim,
                                  blocks=4 if blocks is None else blocks, **common)
    raise ValueError('Unknown R5 grid model kind: ' + str(kind))


def _fixture(report_path):
    """Generated CPU tensors only; tests actual default models and all paths."""
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(20261025)
    observed = torch.rand(3, GRID_FRAMES, GRID_BANDS) > .12
    x = torch.randn(3, GRID_CHANNELS, GRID_FRAMES, GRID_BANDS) * observed[:, None]
    x[:, 3] = observed.float()
    raw = torch.rand(3, GRID_FRAMES, GRID_BANDS) * observed
    weights = raw / raw.sum(dim=(1, 2), keepdim=True)
    truth = torch.rand_like(weights) * observed
    y = (truth * weights).sum(dim=(1, 2))
    rows = []
    for kind in ('tfgrid_mask', 'bandsplit_mask'):
        torch.manual_seed(42)
        model = make_grid_model(kind)
        assert all(p.device.type == 'cpu' for p in model.parameters())
        model.train()
        p, presence, cell = model(x, weights)
        assert p.shape == presence.shape == (3,) and cell.shape == (3, 15, 25)
        assert all(torch.isfinite(a).all() for a in (p, presence, cell))
        assert bool(((cell >= 0) & (cell <= 1)).all()) and bool(((p >= 0) & (p <= 1)).all())
        assert torch.equal(p, (cell * weights).sum(dim=(1, 2)))
        loss = (p - y).square().mean() + .25 * ((cell - truth).square() * weights).sum((1, 2)).mean()
        loss = loss + .05 * nn.functional.binary_cross_entropy_with_logits(presence, torch.ones_like(presence))
        loss.backward()
        missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
        assert not missing, missing
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        path_gradients = {}
        for name, parameter in model.named_parameters():
            if any(part in name for part in ('fullband_frequency.rnn.weight_ih', 'subband_temporal.rnn.weight_ih',
                'cross_frame_attention.query_projection.weight', 'temporal.rnn.weight_ih', 'band.rnn.weight_ih')):
                norm = float(parameter.grad.norm())
                assert norm > 0, name
                path_gradients[name] = norm
        if kind == 'tfgrid_mask':
            assert len([k for k in path_gradients if 'query_projection' in k]) == 6
        else:
            assert len(model.mask_decoders) == 5
            assert all(bool(cell[..., begin:end].std(dim=-1).gt(0).all()) for begin, end in model.band_groups)
        model.eval()
        with torch.no_grad():
            encoded = model.encode(x)
            representation = model.representation(x)
            reference = model(x, weights)
            assert encoded.shape == (3, model.grid_config['embed_dim'], 15, 25)
            assert representation.shape == (3, model.grid_config['embed_dim'])
            alone = model(x[:1], weights[:1])
            changed = x.clone(); changed[1:] = 7 * changed[1:] + 3
            companions_changed = model(changed, weights)
            batch_error = max(float((a[:1] - b).abs().max()) for a, b in zip(reference, alone))
            companion_error = max(float((a[:1] - b[:1]).abs().max()) for a, b in zip(reference, companions_changed))
            assert batch_error < 3e-6 and companion_error == 0
            buffer = io.BytesIO()
            torch.save({'kind': kind, 'grid_config': model.grid_config, 'state_dict': model.state_dict()}, buffer)
            buffer.seek(0); saved = torch.load(buffer, map_location='cpu', weights_only=True)
            restored = make_grid_model(saved['kind'], **saved['grid_config'])
            restored.load_state_dict(saved['state_dict'], strict=True); restored.eval()
            replay = restored(x, weights)
            assert all(torch.equal(a, b) for a, b in zip(reference, replay))
        rows.append({'kind': kind, 'parameters': sum(p.numel() for p in model.parameters()),
            'config': model.grid_config, 'input_shape': list(x.shape), 'encoded_shape': list(encoded.shape),
            'share_shape': list(p.shape), 'mask_shape': list(cell.shape), 'loss': float(loss.detach()),
            'all_parameters_have_finite_gradients': True, 'operator_path_gradient_norms': path_gradients,
            'energy_weighted_pooling_exact': True, 'bounded_share_and_cell_mask': True,
            'eval_single_vs_batch_max_abs_error': batch_error,
            'eval_changed_companions_max_abs_error': companion_error,
            'checkpoint_reload_bitwise_equal': True})
    for kwargs in ({'kind': 'unknown'}, {'kind': 'tfgrid_mask', 'input_channels': 6},
                   {'kind': 'tfgrid_mask', 'embed_dim': 63}, {'kind': 'bandsplit_mask', 'n_bands': 24}):
        try:
            make_grid_model(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid model contract accepted')
    report = {'status': 'PASS', 'scope': 'generated CPU numerical fixture; no actual project datasets',
        'torch_version': str(torch.__version__), 'GPU_accessed': False, 'pretrained_weights_downloaded': False,
        'training_dataset_or_Test_or_sealed_accessed': False, 'full_training_executed': False,
        'fixture_backward_is_not_model_training': True, 'model_results': rows,
        'band_groups': band_group_metadata(), 'source_file_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'elapsed_seconds': time.perf_counter() - start}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--report', type=Path, default=Path('reports/R5_GRID_MODELS_FIXTURE.json'))
    arguments = parser.parse_args()
    if arguments.self_test:
        _fixture(arguments.report)
    else:
        for name in ('tfgrid_mask', 'bandsplit_mask'):
            network = make_grid_model(name)
            print(json.dumps({'kind': name, 'grid_config': network.grid_config,
                'parameters': sum(p.numel() for p in network.parameters()), 'training_executed': False}))
