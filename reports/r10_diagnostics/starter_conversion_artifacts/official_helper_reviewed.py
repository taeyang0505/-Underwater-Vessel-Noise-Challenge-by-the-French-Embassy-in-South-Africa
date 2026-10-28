def minute_band_power(samples, sample_rate_hz):
    """One-channel audio -> complete-minute mean-square values, shape (minutes, 25)."""
    from scipy.signal import periodogram

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("Use one channel with finite samples.")
    fs = float(sample_rate_hz)
    if not np.isfinite(fs) or fs <= 0 or not fs.is_integer():
        raise ValueError("Use the actual integer sample rate from the recording.")
    segment_size = int(2 * fs)
    minutes = samples.size // (30 * segment_size)
    if minutes == 0:
        raise ValueError("At least one complete minute is needed.")
    segments = samples[:minutes * 30 * segment_size].reshape(-1, segment_size)
    frequency, psd = periodogram(
        segments, fs=fs, window="hann", detrend="constant",
        return_onesided=True, scaling="density", axis=1,
    )
    centres = 10.0 ** (1.5 + 0.1 * np.arange(25))
    power = np.full((len(segments), 25), np.nan)
    df = frequency[1] - frequency[0]
    for band, centre in enumerate(centres):
        low, high = centre / 10**0.05, centre * 10**0.05
        bins = (frequency >= low) & (frequency < high)
        if high <= fs / 2 and bins.any():
            power[:, band] = psd[:, bins].sum(axis=1) * df
    return power.reshape(minutes, 30, 25).mean(axis=1)
