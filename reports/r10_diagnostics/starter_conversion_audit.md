# R10: official starter conversion parity audit

**The reviewed optional starter conversion and our R9 native-minute conversion agree in their PSD/integration/averaging equations.** The measured120-second VTUAD example differs by at most **1.884e-7 relative power / 8.182e-7 dB** from the official float64 helper. Its newly computed float32 two-second powers are **bitwise identical to the existing bank**. This is conversion parity, not proof that SA exports or the hidden generator use the same calibration or acoustics.

## Source and execution scope

Only the `minute_band_power` FunctionDef from notebook cell11 was extracted after source inspection. Its only external global is NumPy; its local import is scipy.signal.periodogram. No other notebook cell, Test loading, submission code, model fitting, report array or network operation was executed. Notebook SHA: `08242f4771c38de43ab9022d460b537b5eaa90a687bb1ddb532061b512dd6b83`; helper SHA: `38241ef029c3605119f260678f6e518fcfc81d87110cabdf2ceed5dc6380dde3`.

23 synthetic in-memory WAV cases cover32/48/96kHz,16kHz, the17,825/17,826Hz Nyquist boundary,60/120/fractional-minute durations, tones adjacent to band edges, white noise, DC offset, changing level, stereo and silence. Every generated case remains in the JSON/CSV, including expected rejections.

## Equations and numerical checks

For each non-overlapping two-second block of actual sampling rate fs, N=2fs and nfft=N. Both paths subtract the block mean, use the periodic Hann window, and request one-sided density scaling. Interior positive-frequency bins are doubled; DC and Nyquist are not.

`S[k] = abs(rfft((x-mean(x))*Hann))[k]^2 / (fs*sum(Hann^2))`

`P_band = sum(S[k] where centre/10^.05 <= f[k] < centre*10^.05) * df`

`P_minute = mean(P_band for30 complete two-second blocks)`

Band centres are `10^(1.5+.1*j)`, j=0..24. This is integrated band power, not the arithmetic mean of PSD bins and not sound exposure level. No logarithm is taken inside either helper. Uncalibrated input remains uncalibrated digital mean-square energy.

Independent FFT reconstruction: df=0.5Hz, sum(Hann²)/N=0.375; manual versus scipy density max absolute error=1.372e-19. Full-frequency integration=0.0099767880317804883, matching window-weighted detrended mean square=0.0099767880317804866.

At df=.5Hz the first band has effective discrete integration width7.0Hz and the last1833.5Hz. Replacing `sum*df` with `mean` would therefore create a frequency-dependent factor, not an innocuous common gain. Omitting df would cause a factor2 (+3.0103dB); that factor is absent in the comparisons.

## Observed comparisons

| Case | Official / R9 native output | Max relative power error* | Max absolute dB error* |
|---|---|---:|---:|
|tone100_60s|1×25 / same|6.967e-08|3.026e-07|
|tone100_120s|2×25 / same|6.967e-08|3.026e-07|
|tone100_125p5s|2×25 / same|6.967e-08|3.026e-07|
|noise_119p9s|1×25 / same|1.578e-07|6.854e-07|
|white_noise_32k|1×25 / same|1.07e-07|4.647e-07|
|white_noise_48k|1×25 / same|1.339e-07|5.814e-07|
|white_noise_96k|1×25 / same|2.452e-07|1.065e-06|
|DC_plus_noise|1×25 / same|2.242e-07|9.738e-07|
|linear_then_log|1×25 / same|6.458e-09|2.804e-08|
|stereo_first_channel|1×25 / same|6.967e-08|3.026e-07|
|actual_VTUAD_original_TRAIN_120s|2×25 / same|1.884e-07|8.182e-07|

*Relative/dB comparisons use official power greater than max(1e-8×that minute maximum,1e-25). The12kHz out-of-all-bands tone is retained separately: its largest absolute power difference is3.041e-24; numerical leakage near zero is not a reliable relative-error denominator. The JSON retains the ratio ranges and significance rule; the NPZ artifacts retain every output cell for recomputation.

## Actual reference and existing-cache check

The metadata-only selection was the first clip_id among177 verified, original-TRAIN, non-background-class,120..300s candidates. The selected cargo clip is `046492f97124201a`, `strict_interval_wav/inclusion_4000_exclusion_6000/vessel/357.wav`: PCM16 mono, actual32,000Hz,3,840,000 frames,120seconds. Its full WAV SHA `046492f97124201ab60fb6afbe111b5de3d7340b7ed919a8e87264f2b1982974` matched the manifest before decoding. No reserved recording was restored or source/cached waveform changed.

The new two-second matrix equals all60 existing bank rows bitwise (max difference0). Official versus existing-cache→native-minute power ratio is [0.9999998421615011,1.000000188392467] over all50 compared band-minute cells. The maximum absolute error is2.565e-15. The small residual is consistent with the verified dtype difference: officialfloat64 samples/PSD versus ourfloat32 samples/PSD and stored band powers, followed byfloat64 R9 minute averaging. This is not a demonstrated large normalization defect.

## Real policy/dtype differences, preserved rather than silently fixed

| Condition | Official helper | Current external/R9 path |
|---|---|---|
|Sample/PSD precision|Converts samples tofloat64; PSDfloat64|WAV decodefloat32; PSDfloat32; stored segmentsfloat32; native_minutes averages in float64|
|Unsupported upper band|Returns NaN for that band|Rejects the entire source lacking any of25 supported bands|
|Nyquist threshold|Last upper edge=8912.5093813375Hz;16k and17,825Hz yield NaN in band24|Same band edge, but whole-source rejection;17,826Hz supports all25|
|Less than60s|Rejects, no complete minute|Can retain complete2s segments; R9 native_minutes still rejects fewer than30segments. No padding or repetition|
|Incomplete final minute|Discards it|Can retain its complete2s rows in the segment bank; R9 native_minutes discards them for minute-first data|
|Stereo|Rejects a2D input; caller must select one channel|Explicitly selects channel0; the comparison selects the same channel. No stereo averaging|
|Silence|Returns zero power|Rejects zero-power segments/minutes|

For119.9seconds, the external bank can retain59two-second rows, but both complete-minute outputs contain only the first60seconds. The external `trailing_seconds_dropped` field describes the incomplete two-second tail(1.9s), not all59.9seconds omitted from the minute representation. For125.5seconds, both minute outputs contain120seconds.

The changing-amplitude fixture gives linear mean power0.205000001: correct10log10(mean power)=-6.882461dB, whereas mean(10log10 per-block power)=-13.467875dB, a6.585413dB difference. Both actual helpers use the former order; this deliberately wrong calculation is only an arithmetic contrast.

## Conclusion and limits

This audit finds no evidence here for a missing df, PSD-bin-mean, Hann normalization, actual-fs or log-before-linear-averaging error in the admitted reference conversion. It does find the explicit dtype and source-admission differences above. Original notebook, conversion sources, numeric bank and archive size/mtime checks remained unchanged. No original implementation was fixed.

The starter explicitly calls its function an optional external-audio example rather than a verified reconstruction of SA exports. Passing parity therefore does not establish correct hydrophone/ADC calibration, the hidden target construction, natural vessel-only labels, domain alignment, or the cause of a leaderboard gap.

[Full audit JSON](starter_conversion_audit.json) · [Case table CSV](starter_conversion_cases.csv) · [Band bounds CSV](starter_conversion_band_bounds.csv) · [Derived summary JSON](starter_conversion_summary.json) · [Extracted helper](starter_conversion_artifacts/official_helper_reviewed.py)
