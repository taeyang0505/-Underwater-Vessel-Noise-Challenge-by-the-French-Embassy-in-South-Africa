"""Training-only feature hiding for R16 pretrained adapters.

This is context reconstruction, NOT a new observation mask. The original
observation channel remains unchanged; physical pooling weights and targets are
not accepted by this API and must remain separate in the caller. Apply this to
the 17-channel tensor BEFORE selecting the ordered 15 classifier channels.

A required CPU torch.Generator produces only small random stripe plans. Plans
are identical across CPU/MPS/CUDA. Mask application, observed-cell budgeting and
all feature tensors remain on the input device. This explicit control RNG is
not a fallback for unsupported accelerator operations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch

MASK_CHANNEL = 5
PROFILE_CHANNEL = 4
CHANNELS = 17
HEIGHT = 15
WIDTH = 25
VERSION = 'r16_feature_context_mask_v1'


@dataclass(frozen=True)
class FeatureMaskingConfig:
    block_probability: float = .5
    profile_probability: float = .4
    max_observed_fraction: float = .3
    min_frequency_stripes: int = 1
    max_frequency_stripes: int = 2
    min_frequency_width: int = 1
    max_frequency_width: int = 2
    min_time_width: int = 1
    max_time_width: int = 2
    variant: str = 'both'

    def __post_init__(self):
        if not (0 <= self.block_probability <= 1 and 0 <= self.profile_probability <= 1):
            raise ValueError('Mask probabilities must be in [0,1]')
        if not 0 <= self.max_observed_fraction <= .3:
            raise ValueError('Feature masking may hide at most 30% of observed cells')
        for low, high in ((self.min_frequency_stripes, self.max_frequency_stripes),
                          (self.min_frequency_width, self.max_frequency_width),
                          (self.min_time_width, self.max_time_width)):
            if not (1 <= low <= high <= 2):
                raise ValueError('Use one or two stripes, bands and minutes')

    @classmethod
    def for_variant(cls, variant='both'):
        settings = {
            'off': (0., 0.),
            'spec': (.5, 0.),
            'profile': (0., .4),
            'both': (.5, .4),
        }
        if variant not in settings:
            raise ValueError('Expected off/spec/profile/both variant')
        block, profile = settings[variant]
        return cls(block_probability=block, profile_probability=profile, variant=variant)

    def state_dict(self):
        return {'version': VERSION, **asdict(self)}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.state_dict(), sort_keys=True).encode()).hexdigest()


def make_feature_masking_generator(seed: int, *, epoch: int = 0, stream: str = 'train'):
    """Explicit independent RNG stream; persist generator.get_state() to resume."""
    payload = json.dumps([int(seed), int(epoch), str(stream)], separators=(',', ':')).encode()
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2 ** 63 - 1)
    return torch.Generator(device='cpu').manual_seed(derived_seed)


@dataclass
class FeatureMaskingMetadata:
    context_mask: torch.Tensor
    block_requested: torch.Tensor
    profile_dropped: torch.Tensor
    observed_cells: torch.Tensor
    masked_cells: torch.Tensor
    candidate_cells: torch.Tensor
    visible_minutes: torch.Tensor
    training: bool
    task: str
    config: FeatureMaskingConfig

    def log_dict(self):
        """Small explicit CPU scalar transfer for batch/epoch logging only."""
        obs = self.observed_cells.detach().cpu().double()
        masked = self.masked_cells.detach().cpu().double()
        candidates = self.candidate_cells.detach().cpu().double()
        n = int(obs.numel())
        return {
            'version': VERSION, 'variant': self.config.variant,
            'config_sha256': self.config.fingerprint, 'training': self.training,
            'task': self.task, 'n': n,
            'block_requested_samples': int(self.block_requested.sum().detach().cpu()),
            'context_masked_samples': int((masked > 0).sum()),
            'profile_dropped_samples': int(self.profile_dropped.sum().detach().cpu()),
            'observed_cells': int(obs.sum()), 'masked_observed_cells': int(masked.sum()),
            'masked_fraction_observed': float(masked.sum() / obs.sum()) if n else 0.,
            'max_sample_masked_fraction': float((masked / obs.clamp_min(1)).max()) if n else 0.,
            'budget_clipped_samples': int((candidates > masked).sum()),
            'minimum_visible_minutes': int(self.visible_minutes.min().detach().cpu()) if n else 0,
            'original_observation_channel_preserved': True,
            'physical_weights_or_targets_modified': False,
        }


def _empty_metadata(x, observed, *, task, training, config):
    n = x.shape[0]
    zero = torch.zeros(n, dtype=torch.bool, device=x.device)
    count = observed.sum((1, 2))
    return FeatureMaskingMetadata(
        context_mask=torch.zeros_like(observed), block_requested=zero,
        profile_dropped=zero.clone(), observed_cells=count,
        masked_cells=torch.zeros_like(count), candidate_cells=torch.zeros_like(count),
        visible_minutes=observed.any(-1).sum(-1), training=training,
        task=task, config=config)


def apply_r16_feature_masking(x, *, task, training, generator,
                              config=None):
    """Return (augmented17, metadata), never mutating the original tensor.

    All semantic feature channels, including sqrt(w), are hidden at selected
    context cells. Channel5 remains original O. The caller must continue using
    the original separate w/q/y arrays for supervision and pooling. Ch4 profile
    dropout is regression-only; classification has zero profile-drop probability.
    """
    config = FeatureMaskingConfig() if config is None else config
    if not isinstance(config, FeatureMaskingConfig):
        raise TypeError('FeatureMaskingConfig required')
    if task not in ('regression', 'classification'):
        raise ValueError('Expected regression or classification task')
    if not isinstance(x, torch.Tensor) or x.ndim != 4 or tuple(x.shape[1:]) != (CHANNELS, HEIGHT, WIDTH):
        raise ValueError('Apply to [N,17,15,25] before task_input; direct 15-channel input is forbidden')
    if not x.is_floating_point():
        raise TypeError('Floating feature tensor required')
    if x.device.type == 'mps' and __import__('os').environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('Implicit MPS CPU fallback is forbidden')
    if not torch.isfinite(x).all():
        raise ValueError('R16 feature tensor must be finite; missingness belongs to O')
    original_o = x[:, MASK_CHANNEL]
    if not ((original_o == 0) | (original_o == 1)).all():
        raise ValueError('Observation channel must contain original binary O')
    observed = original_o > 0
    if not observed.any((1, 2)).all():
        raise ValueError('All-missing windows cannot be reconstructed')
    n = x.shape[0]
    if n == 0 or not training or (config.block_probability == 0 and
                        (config.profile_probability == 0 or task == 'classification')):
        return x, _empty_metadata(x, observed, task=task, training=bool(training), config=config)
    if not isinstance(generator, torch.Generator) or generator.device.type != 'cpu':
        raise ValueError('An explicit CPU torch.Generator is required for portable control plans')

    # Generate control plans on CPU by design. No input feature is copied to CPU.
    block_cpu = torch.rand(n, generator=generator) < config.block_probability
    profile_cpu = torch.rand(n, generator=generator) < config.profile_probability
    if task == 'classification':
        profile_cpu.zero_()
    candidate = torch.zeros_like(observed)
    for i in range(n):
        if not bool(block_cpu[i]):
            continue
        tw = int(torch.randint(config.min_time_width, config.max_time_width + 1, (), generator=generator))
        ts = int(torch.randint(0, HEIGHT - tw + 1, (), generator=generator))
        candidate[i, ts:ts+tw, :] = True
        stripes = int(torch.randint(config.min_frequency_stripes, config.max_frequency_stripes + 1, (), generator=generator))
        for _ in range(stripes):
            fw = int(torch.randint(config.min_frequency_width, config.max_frequency_width + 1, (), generator=generator))
            fs = int(torch.randint(0, WIDTH - fw + 1, (), generator=generator))
            candidate[i, :, fs:fs+fw] = True
    candidate &= observed
    candidate_cells = candidate.sum((1, 2))
    observed_cells = observed.sum((1, 2))

    # Prune the stripe union only when sparse observations would violate the
    # 30% cap. A deterministic random permutation avoids preferring early time
    # or low frequency cells. FP32 cumsum is exact for <=375 cells and works on
    # native MPS; no argsort/device fallback is required.
    order_cpu = torch.stack([torch.randperm(HEIGHT * WIDTH, generator=generator) for _ in range(n)])
    order = order_cpu.to(x.device)
    ordered = candidate.reshape(n, -1).gather(1, order)
    allowed = (observed_cells.float() * config.max_observed_fraction).floor()
    keep_ordered = ordered & (ordered.float().cumsum(1) <= allowed[:, None])
    context_mask = torch.zeros_like(ordered).scatter(1, order, keep_ordered).reshape_as(observed)
    masked_cells = context_mask.sum((1, 2))
    remaining = observed & ~context_mask
    visible_minutes = remaining.any(-1).sum(-1)
    if (masked_cells > allowed).any() or not (visible_minutes >= 1).all():
        raise AssertionError('Observed-cell cap or visible-context invariant failed')

    semantic_channels = torch.arange(CHANNELS, device=x.device) != MASK_CHANNEL
    output = x.masked_fill(context_mask[:, None] & semantic_channels[None, :, None, None], 0.)
    profile = profile_cpu.to(x.device)
    # out-of-place multiplication keeps both input ownership and autograd safe.
    if task == 'regression':
        profile_keep = torch.ones((n, CHANNELS, 1, 1), dtype=x.dtype, device=x.device)
        profile_keep[:, PROFILE_CHANNEL, 0, 0] = (~profile).to(x.dtype)
        output = output * profile_keep
    return output, FeatureMaskingMetadata(
        context_mask=context_mask.detach(), block_requested=block_cpu.to(x.device),
        profile_dropped=profile, observed_cells=observed_cells,
        masked_cells=masked_cells, candidate_cells=candidate_cells,
        visible_minutes=visible_minutes, training=True, task=task, config=config)
