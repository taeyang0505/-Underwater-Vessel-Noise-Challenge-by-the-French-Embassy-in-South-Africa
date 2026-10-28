"""Opt-in training-only NT-Xent regularization for 15x25 energy grids.

This is an identity-projection ablation, not a reproduction of SimCLR. The
caller must pass training examples only and keep its weight zero until a
source-disjoint validation experiment supports enabling it. No data are read.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F
from pipeline.mixtures import cnn_channels

SUPPORTED_KINDS = frozenset({'cnn', 'fdy', 'itransformer'})
COMMON_GAIN_DB = 3.0


def common_gain_views(batch_db, seed=0):
    """Two conservative views; retain every measured cell and missing position.

    A scalar gain represents the SAME multiplier on both known components:
    sum(g*R)/sum(g*(B+R)) == sum(R)/sum(B+R). No crop, frequency response,
    cell masking, mixup or untracked additive noise is applied. The relative
    and energy-weight channels therefore retain their original meaning.
    """
    x = np.array(batch_db, dtype=np.float32, copy=True)
    if x.ndim != 3 or x.shape[1:] != (15, 25) or len(x) == 0:
        raise ValueError('Expected a nonempty [N,15,25] dB batch')
    if np.isinf(x).any() or not np.isfinite(x).any(axis=(1, 2)).all():
        raise ValueError('Only NaN missingness is supported; each window needs observations')
    valid = np.isfinite(x)
    rng = np.random.default_rng(int(seed))
    gains = rng.uniform(-COMMON_GAIN_DB, COMMON_GAIN_DB, (2, len(x), 1, 1)).astype(np.float32)
    views = tuple(np.where(valid, x + gain, np.nan).astype(np.float32) for gain in gains)
    if not all(np.isfinite(v[valid]).all() for v in views):
        raise ValueError('Common gain overflowed the observed dB values')
    return views


def _unique_gain_equivalent_rows(batch_db):
    """Conservatively retain one grid per mask and relative pattern (1 mdB).

    This removes within-batch exact/gain-equivalent copies, not all related
    source events. No semantic class or source identity is inferred here.
    """
    rows = np.asarray(batch_db, dtype=np.float64)
    selected = []
    seen = set()
    for i, row in enumerate(rows):
        valid = np.isfinite(row)
        relative = np.zeros(row.shape, dtype=np.float64)
        relative[valid] = np.round(row[valid] - row[valid].mean(), 3)
        relative[relative == 0] = 0  # canonicalize negative zero
        key = valid.tobytes() + relative.tobytes()
        if key not in seen:
            seen.add(key)
            selected.append(i)
    return np.asarray(selected, dtype=np.int64)


def nt_xent(first, second, temperature=0.2, source_groups=None):
    """Symmetric normalized temperature-scaled CE, with self pairs excluded.

    Optional equal source-group IDs are excluded from negative candidates;
    the paired positive is always retained. Anchors without a valid negative
    contribute no loss. Group IDs do not certify that other groups are true
    semantic negatives. The returned zero, when needed, stays differentiable.
    """
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError('temperature must be finite and positive')
    if first.ndim != 2 or first.shape != second.shape or first.shape[1] == 0:
        raise ValueError('Expected matching [N,D] representations')
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError('Nonfinite representation')
    n = len(first)
    if n < 2:
        return (first.sum() + second.sum()) * 0
    z = F.normalize(torch.cat([first.float(), second.float()]), p=2, dim=1, eps=1e-12)
    logits = (z @ z.T) / float(temperature)
    if not torch.isfinite(logits).all():
        raise ValueError('temperature produced nonfinite contrastive logits')
    device = logits.device
    ids = torch.arange(2 * n, device=device)
    positive = (ids + n) % (2 * n)
    allowed = ~torch.eye(2 * n, dtype=torch.bool, device=device)
    if source_groups is not None:
        groups = list(source_groups)
        if len(groups) != n or any(g is None for g in groups):
            raise ValueError('One nonmissing source-group ID per representation is required')
        same = torch.tensor([[a == b for b in groups] for a in groups], dtype=torch.bool, device=device)
        allowed &= ~same.repeat(2, 2)
    allowed[ids, positive] = True
    eligible = allowed.sum(1) > 1
    if not eligible.any():
        return z.sum() * 0
    logits = logits.masked_fill(~allowed, -torch.inf)
    return F.cross_entropy(logits[eligible], positive[eligible])


def contrastive_regularizer(model, kind, batch_db, device, temperature=0.2,
                            seed=0, source_groups=None):
    """Return a scalar Tensor without creating parameters or changing inputs.

    model.representation accepts cnn_channels(db,version=model.feature_version)[0],
    [N,4,15,25] for legacy models or [N,6,15,25] for V3. The caller
    owns any supervised objective, optimizer, training-source restrictions,
    and auxiliary loss weight. Calling in eval mode or with TabM is rejected.
    source_groups=None cannot prevent same-source false negatives; default
    experiment weight must therefore remain zero until justified by ablation.
    """
    if kind not in SUPPORTED_KINDS:
        raise ValueError('Contrastive representation unsupported for kind: ' + str(kind))
    if not model.training or not callable(getattr(model, 'representation', None)):
        raise ValueError('A training-mode model with representation(x) is required')
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError('temperature must be finite and positive')
    device = torch.device(device)
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('This verified RNG-isolated ablation supports CPU or CUDA only')
    # Validate before deduplication, including singleton and all-missing rows.
    common_gain_views(batch_db, seed)
    selected = _unique_gain_equivalent_rows(batch_db)
    x = np.asarray(batch_db, dtype=np.float32)[selected]
    groups = None
    if source_groups is not None:
        incoming = list(source_groups)
        if len(incoming) != len(batch_db) or any(g is None for g in incoming):
            raise ValueError('One nonmissing source-group ID per input window is required')
        groups = [incoming[i] for i in selected]
    views = common_gain_views(x, seed)
    channels = np.concatenate([cnn_channels(v,version=getattr(model,'feature_version',1))[0] for v in views])
    if not np.isfinite(channels).all():
        raise ValueError('Nonfinite transformed channels')
    tensor = torch.from_numpy(channels).to(device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    # Preserve the caller's RNG stream, including stochastic encoder layers.
    with torch.random.fork_rng(devices=cuda_devices):
        local_cpu = torch.Generator(device='cpu').manual_seed(int(seed))
        torch.set_rng_state(local_cpu.get_state())
        if cuda_devices:
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(int(seed))
        h = model.representation(tensor)
    if h.ndim != 2 or len(h) != 2 * len(x):
        raise ValueError('representation(x) must return [N,D]')
    return nt_xent(h[:len(x)], h[len(x):], temperature, groups)
