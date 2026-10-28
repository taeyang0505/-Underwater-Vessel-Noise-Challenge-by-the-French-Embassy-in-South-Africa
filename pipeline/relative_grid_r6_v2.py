"""Pure NumPy R6-v2 query grid; no targets, prior fitting, or Torch imports.

Channels 0--5 are the unchanged R6-v1 relative grid. Channel 6 is
(Q(db)-observed_temporal_band_quantile_0.05)/15. Channel 7 is sqrt(w), with
w from the SAME 0.1-dB rounded observations Q. q05 is an observed-window
statistic, not certified background truth or source separation.

Raw-precision target q/y are not accepted or redefined. Quantized weights
can pool a fixed raw q to a different share; the v1 bound still applies.
All channels/weights are common-gain invariant after quantization; arbitrary
gain before rounding can cross boundaries. Static band offsets cancel in
channels 0--3 and 6 when rounding commutes, but NOT in profile channel 4,
weights, or sqrt(w). No per-query population fitting is performed.
"""
from __future__ import annotations

import warnings
import numpy as np
from pipeline.relative_grid_r6 import (
    FEATURE_NAMES as V1_FEATURE_NAMES,
    quantized_db, quantization_pooling_bound, relative_grid,
)


FEATURE_NAMES = V1_FEATURE_NAMES + (
    'band_temporal_q05_relative_div15', 'sqrt_rounded_energy_weight',
)


def relative_grid_v2(db) -> tuple[np.ndarray, np.ndarray]:
    """Return finite float32 x8[N,8,15,25], w[N,15,25] from db ONLY."""
    base, weights = relative_grid(db)
    x = quantized_db(db)
    observed = np.isfinite(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        floor = np.nanquantile(x, .05, axis=1, keepdims=True, method='linear')
    q05_relative = np.where(observed, (x-floor)/15., 0.).astype(np.float32)
    energy_channel = np.sqrt(weights)
    result = np.concatenate([base, q05_relative[:, None], energy_channel[:, None]], axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite R6-v2 relative grid')
    return result, weights
