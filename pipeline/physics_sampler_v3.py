"""Opt-in support-aware physics mixtures; frozen v2 code remains unchanged.

Targets measure known added-reference energy, including its recorded ambience.
They are not pure-vessel labels or new independent field observations.
"""
from dataclasses import dataclass
import numpy as np
from pipeline.native_sampler_v2 import NativeFirstSampler, AugmentationConfig
from pipeline.augmentation_v2 import (solve_reference_scale as _legacy_solve,
    average_complete_minutes, blend_backgrounds, UnobservedReferenceError, EnergyComponents)


@dataclass(frozen=True)
class PhysicsAugmentationConfig(AugmentationConfig):
    multi_reference_probability: float = .5
    background_blend_probability: float = .25
    reference_eq_enabled: bool = True
    reference_eq_max_db: float = 4.
    zero_addition_probability: float = .2
    high_share_threshold: float = .4
    min_positive_share: float = .005
    max_positive_share: float = .98
    max_minute_snr_db: float = 20.

    def __post_init__(self):
        # Check only inherited fields with the original schema.
        AugmentationConfig(**{k: getattr(self, k) for k in AugmentationConfig.__dataclass_fields__})
        if type(self.reference_eq_enabled) is not bool:
            raise ValueError('reference_eq_enabled must be a boolean')
        if not np.isfinite(self.reference_eq_max_db) or not 0 <= self.reference_eq_max_db <= 4:
            raise ValueError('Reference EQ must be bounded within +/-4 dB')
        if not 0 <= self.zero_addition_probability <= 1 or not 0 < self.high_share_threshold < 1:
            raise ValueError('Invalid target probabilities or high-share threshold')
        if not 0 < self.min_positive_share <= self.max_positive_share <= .98:
            raise ValueError('Positive share range must be within (0,.98]')
        if self.common_response_range_db or self.measured_ambient_probability:
            raise ValueError('V3 uses reference-only EQ; common response and extra ambient are not enabled')
        if not np.isfinite(self.max_minute_snr_db) or not -300 <= self.max_minute_snr_db <= 300:
            raise ValueError('A finite representable minute SNR policy in [-300,300] dB is required')


def reference_support_audit(reference, observed, single_short=False):
    r = np.asarray(reference, dtype=np.float64)
    o = np.asarray(observed, dtype=bool)
    if r.ndim != 2 or r.shape != o.shape or not o.any():
        raise ValueError('Reference and nonempty observed mask must align')
    if not np.isfinite(r[o]).all() or (r[o] < 0).any():
        raise ValueError('Observed reference energy must be finite/nonnegative')
    observed_minutes = int(o.any(1).sum())
    active = np.where(o, r, 0).sum(1) > 0
    active_minutes = int(active.sum())
    coverage = active_minutes / observed_minutes
    eligible = bool(single_short and active_minutes <= 2)
    cap = min(.98, .25 + .73 * coverage) if eligible else None
    return {'observed_minutes': observed_minutes, 'active_ref_minutes': active_minutes,
            'support_coverage': coverage, 'support_cap': cap,
            'single_short_reference': bool(single_short),
            'support_definition': 'Count observed minute-grid rows with positive added-reference energy, divided by observed minute rows.'}


def solve_reference_scale(background, reference, observed, desired_share, *,
                          support_cap=False, single_short=False, return_audit=False,
                          max_minute_snr_db=None):
    """Legacy four-argument calls retain exactly the original two-value result."""
    if not support_cap and not return_audit and max_minute_snr_db is None:
        return _legacy_solve(background, reference, observed, desired_share)
    requested = float(desired_share)
    if not 0 <= requested < 1:
        raise ValueError('A finite added scale requires 0 <= desired share < 1')
    audit = reference_support_audit(reference, observed, single_short)
    cap = audit['support_cap'] if support_cap else None
    effective = min(requested, cap) if cap is not None else requested
    alpha, mixture = _legacy_solve(background, reference, observed, effective)
    original_alpha = alpha
    alpha_upper, snr_share_cap = None, None
    if max_minute_snr_db is not None:
        limit = float(max_minute_snr_db)
        if not np.isfinite(limit) or not -300 <= limit <= 300:
            raise ValueError('Minute SNR policy must be finite and within [-300,300] dB')
        if requested > 0:
            bminute = mixture.background.sum(1)
            rminute = np.where(mixture.observed, reference, 0).sum(1)
            active_ref = rminute > 0
            if not active_ref.any() or (bminute[active_ref] <= 0).any():
                raise UnobservedReferenceError('Positive reference meets zero minute background; resample, never silently label zero')
            # A/R <= 10**(limit/10) in EVERY positive-reference observed minute.
            bounds = (bminute[active_ref] / rminute[active_ref]) * 10 ** (limit / 10)
            alpha_upper = float(np.min(bounds))
            if not np.isfinite(alpha_upper) or alpha_upper <= 0:
                raise UnobservedReferenceError('Minute SNR scale is not representable; resample without a zero label')
            alpha_upper = float(np.nextafter(alpha_upper, 0.))
            if alpha_upper <= 0:
                raise UnobservedReferenceError('Minute SNR scale underflowed; resample without a zero label')
            alpha = min(alpha, alpha_upper)
            mixture = EnergyComponents(background, alpha * np.where(observed, reference, 0), observed)
            # The implied global ceiling is separate from the support heuristic.
            cap_added = alpha_upper * float(rminute.sum())
            snr_share_cap = float(cap_added / (float(bminute.sum()) + cap_added))
            if mixture.share <= 0 or np.float32(mixture.share) <= 0:
                raise UnobservedReferenceError('Positive capped share would quantize to zero; resample explicitly')
    active = np.where(mixture.observed, reference, 0).sum(1) > 0
    bmass = float(mixture.background.sum())
    amass = float(mixture.added.sum())
    active_bmass = float(mixture.background[active].sum())
    active_amass = float(mixture.added[active].sum())
    bm, am = mixture.background.sum(1), mixture.added.sum(1)
    minute_snr = [float(10 * np.log10(a / b)) if a > 0 and b > 0 else None for a, b in zip(am, bm)]
    actual_max = max((v for v in minute_snr if v is not None), default=None)
    if max_minute_snr_db is not None and actual_max is not None and actual_max > float(max_minute_snr_db) + 1e-10:
        raise RuntimeError('Per-minute known-component SNR bound failed')
    ceilings = [v for v in [cap, snr_share_cap] if v is not None]
    audit.update({'requested_share': requested, 'effective_share': mixture.share,
        'support_cap': min(ceilings) if ceilings else None, 'single_short_support_cap': cap,
        'snr_induced_share_cap': snr_share_cap, 'max_minute_snr_db': max_minute_snr_db,
        'minute_alpha_upper': alpha_upper, 'minute_snr_db': minute_snr,
        'max_minute_snr_db_actual': actual_max,
        'minute_snr_cap_applied': bool(alpha < original_alpha), 'snr_cap_applied': bool(alpha < original_alpha),
        'cap_applied': bool(mixture.share < requested - 1e-12),
        'global_snr_db': float(10 * np.log10(amass / bmass)) if amass > 0 else None,
        'active_snr_db': float(10 * np.log10(active_amass / active_bmass)) if active_amass > 0 and active_bmass > 0 else None,
        'snr_definition': '10log10(sum(O*A)/sum(O*B)); active_snr restricts both sums to reference-active minute rows.'})
    return (alpha, mixture, audit) if return_audit else (alpha, mixture)


def smooth_reference_eq(rng, max_db=4., bands=25):
    if bands < 2 or not np.isfinite(max_db) or not 0 <= max_db <= 4:
        raise ValueError('Invalid smooth reference EQ range')
    x = np.linspace(-1., 1., bands)
    tilt, quadratic = rng.uniform(-max_db, max_db, 2)
    curve = tilt * x + quadratic * (x * x - np.mean(x * x))
    scale = min(1., max_db / max(float(np.abs(curve).max()), 1e-30))
    return curve * scale, {'tilt_coefficient_db': float(tilt),
        'quadratic_coefficient_db': float(quadratic), 'bound_scale': float(scale)}


def apply_reference_eq(reference, gain_db):
    r, gains = np.asarray(reference, dtype=np.float64), np.asarray(gain_db, dtype=np.float64)
    if r.ndim != 2 or gains.shape != (r.shape[1],) or not np.isfinite(gains).all() or not np.isfinite(r).all() or (r < 0).any():
        raise ValueError('Reference EQ requires finite nonnegative energies and one gain per band')
    return r * 10 ** (gains[None, :] / 10)


class PhysicsAwareSampler(NativeFirstSampler):
    def __init__(self, background_db, background_manifest, references, split='train', seed=20261024,
                 config=PhysicsAugmentationConfig(), allow_sealed_holdout=False, require_native_long=True):
        if split not in ['train', 'validation', 'outer_validation'] or allow_sealed_holdout:
            raise ValueError('Physics development sampling excludes calibration and sealed partitions')
        if not isinstance(config, PhysicsAugmentationConfig):
            raise TypeError('PhysicsAugmentationConfig is required for this opt-in sampler')
        super().__init__(background_db, background_manifest, references, split, seed, config,
                         allow_sealed_holdout=False, require_native_long=require_native_long)
        if 'boundary_candidate_flag' in self.bmeta:
            self.bi = self.bi[self.bmeta.boundary_candidate_flag.iloc[self.bi].eq(False).to_numpy()]
        if not len(self.bi):
            raise ValueError('No clean current-split background remains')
        self.site_pools = {site: g.index.to_numpy() for site, g in self.bmeta.iloc[self.bi].groupby('site')}
        sites = self.bmeta.site.iloc[self.bi].to_numpy()
        self._blend_pools = {site: self.bi[sites != site] if (sites != site).any() else self.bi
                             for site in self.site_pools}
        self._eligible_backgrounds = set(self.bi.tolist())
        self._seen_explicit = set()
        self._draw_count = 0

    def _read_background(self, i):
        x = np.array(self.db[int(self.bmeta.base_window_index.iloc[i])], dtype=np.float64)
        if x.shape != (15, 25) or np.isinf(x).any() or np.isfinite(x).all(1).sum() < 3:
            raise ValueError('A clean background needs at least three observed full-band minutes')
        o = np.isfinite(x)
        level = float(np.nanmax(x))
        return np.where(o, 10 ** ((x - level) / 10), 0), o, level

    def _physics_background(self, explicit=None):
        rng, cfg = self.rng, self.config
        if explicit is None:
            site = rng.choice(list(self.site_pools))
            pool = self.site_pools[site]
            high = pool[self.bmeta.observed_minutes.iloc[pool].to_numpy() >= 12]
            sparse = pool[self.bmeta.observed_minutes.iloc[pool].to_numpy() < 12]
            preferred = high if rng.random() < cfg.high_background_coverage_fraction else sparse
            i = int(rng.choice(preferred if len(preferred) else pool))
        else:
            i = int(explicit)
        b, observed, level = self._read_background(i)
        recipe = {'secondary_background_index': -1, 'background_blend_beta': 0.,
            'background_secondary_digital_scale': 1., 'background_blend_applied': False,
            'background_blend_skip_reason': 'not_requested', 'dropped_minutes': [],
            'primary_observed_minutes': int(observed.all(1).sum()),
            'primary_nominal_band_warning': bool(self.bmeta.urn_nominal_band_warning.iloc[i]) if 'urn_nominal_band_warning' in self.bmeta else False}
        if self.split == 'train' and rng.random() < cfg.background_blend_probability:
            candidates = self._blend_pools[self.bmeta.site.iloc[i]]
            recipe['background_blend_skip_reason'] = 'no_compatible_clean_train_background'
            for j in rng.choice(candidates, size=min(12, len(candidates)), replace=False):
                j = int(j)
                if j == i: continue
                if self.bmeta.base_window_index.iloc[j] == self.bmeta.base_window_index.iloc[i]: continue
                other, other_mask, _ = self._read_background(j)
                if (observed & other_mask).all(1).sum() < 3: continue
                beta = float(rng.uniform(.25, .75))
                try:
                    mixed, joint, scale = blend_backgrounds(b, other, observed, other_mask, beta)
                except ValueError:
                    continue  # incompatible or zero joint energy; retain the primary
                b, observed = mixed, joint
                recipe.update({'secondary_background_index': j, 'background_blend_beta': beta,
                    'background_secondary_digital_scale': scale, 'background_blend_applied': True,
                    'background_blend_skip_reason': None, 'secondary_split': self.split,
                    'secondary_background_group': str(self.bmeta.background_group.iloc[j]) if 'background_group' in self.bmeta else None})
                break
        if rng.random() < cfg.whole_minute_dropout_probability:
            minutes = np.flatnonzero(observed.all(1))
            if len(minutes) > 3:
                k = int(rng.choice(minutes)); observed[k] = False; b[k] = 0
                recipe['dropped_minutes'] = [k]
        return i, b, observed, level, recipe

    def _reference_plan(self, requested):
        rng, cfg = self.rng, self.config
        if requested <= 0: return [], 'zero_addition'
        multi = bool(rng.random() < cfg.multi_reference_probability)
        count = min(int(rng.integers(2, 4)) if multi else 1, len(self.ei))
        high = requested > cfg.high_share_threshold
        native_route = high and len(self.long) and (len(self.short) < 2 or rng.random() < cfg.native_long_fraction)
        if high and not native_route and len(self.ei) >= 2: count = min(3, len(self.ei))
        chosen, groups = [], set()
        for k in range(count):
            if native_route and k == 0:
                pool = self.long
            elif high and not native_route and len(self.short) >= 2:
                pool = self.short
            else:
                pool = self.long if len(self.long) and (not len(self.short) or rng.random() < cfg.native_long_fraction) else self.short
            pool = np.asarray([j for j in pool if j not in chosen], dtype=int)
            if not len(pool): pool = np.asarray([j for j in self.ei if j not in chosen], dtype=int)
            fresh_group = np.asarray([j for j in pool if (self.references[j].source, self.references[j].group_key) not in groups], dtype=int)
            j = self._hierarchical_reference(fresh_group if len(fresh_group) else pool)
            chosen.append(j); groups.add((self.references[j].source, self.references[j].group_key))
        route = 'high_share_native' if high and any(j in self.long for j in chosen) else 'high_share_dispersed_multi' if high and len(chosen) >= 2 else 'high_share_fallback_cap' if high else 'ordinary'
        return chosen, route

    def draw(self, n, background_indices=None):
        if int(n) != n or n < 1: raise ValueError('Positive integer batch size required')
        explicit = None
        if background_indices is not None:
            explicit = np.asarray(background_indices)
            if explicit.shape != (n,) or explicit.dtype.kind not in 'iu': raise ValueError('Expected n absolute integer bmeta indices')
            if len(set(explicit.tolist())) != n or any(int(i) not in self._eligible_backgrounds or int(i) in self._seen_explicit for i in explicit):
                raise ValueError('Explicit primary indices must be clean current-split and used exactly once per sampler')
        db, targets, presence, cells, items = [], [], [], [], []
        for k in range(n):
            bi, b, observed, level, item = self._physics_background(None if explicit is None else explicit[k])
            requested = 0. if self.rng.random() < self.config.zero_addition_probability else float(self.rng.uniform(self.config.min_positive_share, self.config.max_positive_share))
            selected, route = self._reference_plan(requested)
            total = np.zeros((15, 25)); refs = []
            observed_ids = np.flatnonzero(observed.all(1))
            bins = np.array_split(observed_ids, len(selected)) if len(selected) > 1 else []
            for jn, j in enumerate(selected):
                anchor_mask = observed.copy()
                anchor_minute = None
                if len(selected) > 1 and len(self.references[j].segments) < 450:
                    anchor_minute = int(self.rng.choice(bins[jn]))
                    anchor_mask[:] = False; anchor_mask[anchor_minute] = observed[anchor_minute]
                else:
                    anchor_mask[~observed.all(1)] = False
                piece, placement = super()._placed_reference(j, anchor_mask)
                response, eq = smooth_reference_eq(self.rng, self.config.reference_eq_max_db) if self.config.reference_eq_enabled else (np.zeros(25), {})
                minute = apply_reference_eq(average_complete_minutes(piece), response)
                if float(minute[observed].sum()) <= 0: raise UnobservedReferenceError('Chosen reference has no measured observed overlap')
                relative = float(10 ** self.rng.uniform(-.3, .3)); total += minute * relative
                record = self.references[j]
                refs.append({'reference_index': j, 'clip_id': record.clip_id, 'source': record.source,
                    'group_key': record.group_key, 'relative_scale': relative, **placement,
                    'requested_anchor_minute': anchor_minute, 'reference_eq_db': response.tolist(),
                    'reference_eq_parameters': eq, 'reference_eq_applied_before_alpha': True,
                    'actual_active_minutes': np.flatnonzero(np.where(observed, minute, 0).sum(1) > 0).tolist()})
            single_short = len(selected) == 1 and len(self.references[selected[0]].segments) < 450
            routed_request = min(requested, self.config.high_share_threshold) if route == 'high_share_fallback_cap' else requested
            alpha, mixture, audit = solve_reference_scale(b, total, observed, routed_request,
                support_cap=True, single_short=single_short, return_audit=True,
                max_minute_snr_db=self.config.max_minute_snr_db)
            effective = mixture.share
            single_support_cap = audit['single_short_support_cap']
            ceilings = [v for v in [audit['support_cap'], self.config.high_share_threshold if route == 'high_share_fallback_cap' else None] if v is not None]
            applied_ceiling = min(ceilings) if ceilings else None
            audit.update({'requested_share': requested, 'cap_applied': bool(effective < requested - 1e-12),
                'single_short_support_cap': single_support_cap, 'support_cap': applied_ceiling,
                'high_share_route_cap': self.config.high_share_threshold if route == 'high_share_fallback_cap' else None})
            reverse = bool(self.rng.random() < self.config.time_reverse_probability)
            if reverse:
                mixture = mixture.reverse_time()
                audit['minute_snr_db'] = list(reversed(audit['minute_snr_db']))
            gain = float(self.rng.uniform(-self.config.common_gain_range_db, self.config.common_gain_range_db))
            mixture = mixture.common_gain(gain)
            if not np.isclose((mixture.cell_share * mixture.energy_weights).sum(), effective, atol=1e-10):
                raise RuntimeError('Physics target violates the energy identity')
            db.append((mixture.to_db() + level).astype('float32')); targets.append(effective)
            presence.append(effective > 0); cells.append(mixture.cell_share.astype('float32'))
            items.append({'background_index': bi, 'base_background_index': int(self.bmeta.base_window_index.iloc[bi]),
                'references': refs, 'desired_share': effective, **audit, 'addition_scale': alpha,
                'route': route, 'seed': self.seed, 'draw_index': self._draw_count + k,
                'common_gain_db': gain, 'common_response_db': [0.] * 25,
                'time_reversed': reverse, **item})
        self._draw_count += n
        if explicit is not None: self._seen_explicit.update(explicit.tolist())
        recipes = {'version': 'physics_aware_v3', 'split': self.split, 'seed': self.seed,
            'background_index': np.asarray([i['background_index'] for i in items]),
            'items': items, 'config': vars(self.config), 'explicit_primary_order': explicit is not None,
            'surrogate_envelopes_used': False, 'reference_repetition_used': False,
            'reference_semantics': 'Known added measured reference including ambience, not isolated natural vessel truth.'}
        return np.stack(db), np.asarray(targets, dtype='float32'), np.asarray(presence, dtype='float32'), np.stack(cells), recipes
