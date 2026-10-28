"""Opt-in transformations of KNOWN linear-energy components.

No learned labels, challenge Test statistics, waveform repetition or claim of
new independent recordings is used. These functions leave every v1 interface
unchanged. Target invariance is used only when the mathematics guarantees it;
otherwise the target is recomputed from transformed components and mask.
"""
from dataclasses import dataclass
import numpy as np

class UnobservedReferenceError(ValueError):
    pass

@dataclass(frozen=True)
class EnergyComponents:
    background: np.ndarray
    added: np.ndarray
    observed: np.ndarray

    def __post_init__(self):
        b=np.asarray(self.background,dtype=np.float64)
        r=np.asarray(self.added,dtype=np.float64)
        o=np.asarray(self.observed,dtype=bool)
        if b.shape!=r.shape or b.shape!=o.shape or b.ndim!=2:
            raise ValueError('Components and observed mask must share a two-dimensional shape')
        if not o.any():raise ValueError('No observed cells')
        if not (np.isfinite(b[o]).all() and np.isfinite(r[o]).all()):
            raise ValueError('Observed component energies must be finite')
        if (b[o]<0).any() or (r[o]<0).any():raise ValueError('Linear energy must be nonnegative')
        # Unobserved values are undefined and never silently counted as silence.
        object.__setattr__(self,'background',np.where(o,b,0).copy())
        object.__setattr__(self,'added',np.where(o,r,0).copy())
        object.__setattr__(self,'observed',o.copy())
        if not np.isfinite(self.total_energy) or self.total_energy<=0:
            raise ValueError('Observed total energy must be finite and positive')

    @property
    def total_energy(self):return float((self.background+self.added)[self.observed].sum())
    @property
    def share(self):return float(self.added[self.observed].sum()/self.total_energy)
    @property
    def cell_share(self):
        e=self.background+self.added
        return np.divide(self.added,e,out=np.zeros_like(e),where=self.observed&(e>0))
    @property
    def energy_weights(self):return (self.background+self.added)/self.total_energy

    def to_db(self):
        e=self.background+self.added
        result=np.full(e.shape,np.nan)
        if (e[self.observed]<=0).any():raise ValueError('Observed zero energy has no finite dB level')
        result[self.observed]=10*np.log10(e[self.observed])
        return result

    def common_gain(self,gain_db):
        gain=10**(float(gain_db)/10)
        return EnergyComponents(self.background*gain,self.added*gain,self.observed)

    def reverse_time(self):
        return EnergyComponents(self.background[::-1],self.added[::-1],self.observed[::-1])

    def with_mask(self,observed):
        # A mask may remove observations; it cannot create new measured cells.
        incoming=np.asarray(observed,dtype=bool)
        if incoming.shape!=self.observed.shape:raise ValueError('New observation mask must have the same shape')
        mask=self.observed&incoming
        return EnergyComponents(self.background,self.added,mask)

    def add_measured_ambient(self,ambient,scale=1):
        a=np.asarray(ambient,dtype=np.float64)
        if a.shape!=self.background.shape or scale<0 or not np.isfinite(scale):
            raise ValueError('Ambient energy must match the component shape and scale must be finite/nonnegative')
        return EnergyComponents(self.background+float(scale)*a,self.added,self.observed)

    def frequency_response(self,gain_db):
        """A known common receiver response changes frequency weights, hence y.

        This is distinct from a scalar common gain. The new share is computed
        after the response, rather than copying the old scalar target.
        """
        response=10**(np.asarray(gain_db,dtype=np.float64)/10)
        if response.shape!=(self.background.shape[1],):raise ValueError('One response value per frequency band required')
        return EnergyComponents(self.background*response,self.added*response,self.observed)

def solve_reference_scale(background,reference,observed,desired_share):
    b=np.asarray(background,dtype=np.float64);r=np.asarray(reference,dtype=np.float64)
    o=np.asarray(observed,dtype=bool);u=float(desired_share)
    if b.shape!=r.shape or b.shape!=o.shape or not o.any():raise ValueError('Invalid component shapes or empty observation mask')
    if not 0<=u<1:raise ValueError('A finite added scale requires 0 <= desired share < 1')
    if not np.isfinite(b[o]).all() or not np.isfinite(r[o]).all() or (b[o]<0).any() or (r[o]<0).any():
        raise ValueError('Known observed component energies must be finite/nonnegative')
    be=float(b[o].sum());re=float(r[o].sum())
    if be<=0:raise ValueError('Positive background energy required')
    if u>0 and re<=0:raise UnobservedReferenceError('Positive requested share but no reference energy intersects the observed mask; resample placement')
    alpha=0. if u==0 else u*be/((1-u)*re)
    if not np.isfinite(alpha):raise ValueError('Requested component scaling overflowed')
    return alpha,EnergyComponents(b,alpha*np.where(o,r,0),o)

def place_measured_segments(segments,start_segment,window_segments=450):
    """Place a native measured 2-s sequence ONCE; no interpolation or cycling.

    The zero extension defines the controlled added component outside its
    chosen support. It is not a claim that unrecorded natural sound was zero.
    Negative start values crop a recorded leading portion from the window.
    """
    s=np.asarray(segments,dtype=np.float64)
    if s.ndim!=2 or not len(s) or not np.isfinite(s).all() or (s<0).any():
        raise ValueError('A finite nonnegative measured segment sequence is required')
    start=int(start_segment)
    result=np.zeros((window_segments,s.shape[1]),dtype=np.float64)
    lo=max(0,start);hi=min(window_segments,start+len(s))
    if hi>lo:result[lo:hi]=s[lo-start:hi-start]
    return result

def average_complete_minutes(segments):
    s=np.asarray(segments,dtype=np.float64)
    if s.ndim!=2 or len(s)%30:raise ValueError('Use complete blocks of thirty measured/modelled 2-s segments')
    return s.reshape(-1,30,s.shape[1]).mean(1)

def blend_backgrounds(first,second,mask_first,mask_second,beta):
    """A declared synthetic soundscape, not a newly measured recording.

    A single relative digital-level normalization is applied to the second
    source. Intersected masks prevent filling unmeasured cells with silence.
    """
    a=np.asarray(first,dtype=np.float64);b=np.asarray(second,dtype=np.float64)
    o=np.asarray(mask_first,dtype=bool)&np.asarray(mask_second,dtype=bool)
    if a.shape!=b.shape or o.shape!=a.shape or not 0<=beta<=1 or not o.any():raise ValueError('Invalid blend or no jointly observed cells')
    if not np.isfinite(a[o]).all() or not np.isfinite(b[o]).all() or (a[o]<0).any() or (b[o]<0).any():
        raise ValueError('Observed background energy must be finite/nonnegative')
    scale=float(a[o].sum()/b[o].sum()) if b[o].sum()>0 else 0.
    if not np.isfinite(scale) or scale<=0:raise ValueError('Both observed backgrounds require finite positive energy')
    return np.where(o,(1-beta)*a+beta*scale*b,0),o,scale

def complete_audio_tail_audit(duration_seconds,segments,minutes):
    """Name the different tails explicitly, including complete 2-s pieces.

    Existing v1 ``trailing_seconds_dropped`` is a segment-level field, not the
    duration absent from the complete-minute array.
    """
    return {'trailing_seconds_below_two_second_segment':float(duration_seconds-2*len(segments)),
            'complete_segments_not_in_minute_array':int(len(segments)%30),
            'seconds_not_in_complete_minute_array':float(duration_seconds-60*len(minutes))}
