"""R4 controlled VT passage additions with one policy in every source split.

This is a declared band-power simulator, not the unreleased challenge generator.
No dB interpolation, clip repetition, Test inputs or natural-vessel labels.
Bounded cumulative-linear-power interpolation is explicitly recorded below.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class PassageConfigR4:
    zero_probability: float = .2
    minimum_share: float = .005
    maximum_share: float = .95
    reference_eq_max_db: float = 4.
    fade_seconds: float = 30.
    duration_edges_seconds: tuple = (120., 300., 1380.)
    rendered_minimum_segments: int = 60
    rendered_maximum_segments: int = 450
    minimum_stretch: float = 1/3
    maximum_stretch: float = 3.

    def __post_init__(self):
        if not 0 <= self.zero_probability <= 1 or not 0 < self.minimum_share < self.maximum_share < 1:
            raise ValueError('Invalid target prior')
        if not 0 <= self.reference_eq_max_db <= 4 or not 0 < self.fade_seconds <= 60:
            raise ValueError('Unapproved EQ/fade bound')
        e=np.asarray(self.duration_edges_seconds,float)
        if e[0]!=120 or e[-1]!=1380 or not (np.diff(e)>0).all():raise ValueError('Invalid diagnostic duration edges')
        if (self.rendered_minimum_segments,self.rendered_maximum_segments)!=(60,450) or self.minimum_stretch!=1/3 or self.maximum_stretch!=3:
            raise ValueError('Rendered support and stretch bounds must be predeclared')

    def to_dict(self):return asdict(self)


@dataclass(frozen=True)
class PassageReference:
    metadata: dict
    segments: np.ndarray

    def validate(self):
        m=self.metadata;s=np.asarray(self.segments)
        if m['source']!='VTUAD_v1.3.2' or m.get('split') not in ['train','validation','outer_validation']:
            raise ValueError('Only the permitted VT development pool is allowed')
        for flag in ['reference_eligible','verified','measured','quality_full_band_good']:
            if not bool(m.get(flag)):raise ValueError('Reference quality gate failed: '+flag)
        if s.ndim!=2 or s.shape[1]!=25 or not 60<=len(s)<=690 or not np.isfinite(s).all() or (s<0).any() or not (s.sum(1)>0).all():
            raise ValueError('Expected 120–1380 measured seconds of complete 2s × 25 powers')
        if m.get('sealed',False):raise ValueError('Reserved reference is forbidden')
        if m.get('original_split','train') not in ['train','validation'] or m.get('role','vessel_passage')!='vessel_passage':raise ValueError('Forbidden original reference role/split')
        if 'segment_count' in m and len(s)!=int(m['segment_count']):raise ValueError('Truncated reference bank slice')
        if not 0<=float(m['duration_seconds'])-2*len(s)<2+1e-6:raise ValueError('Header duration and complete 2s source bins disagree')


def duration_bin(seconds, config):
    e=np.asarray(config.duration_edges_seconds,float)
    if not e[0]<=seconds<=e[-1]:raise ValueError('Reference duration outside declared bounds')
    return min(int(np.searchsorted(e,seconds,side='right')-1),len(e)-2)


def warp_band_power(segments,target_segments):
    """Conservative monotone rebinning of the WHOLE measured power timeline.

    Each original 2s source bin has constant power between its edges. Linear
    interpolation of its cumulative integral, followed by differencing,
    preserves each band's mean power. Total power×duration changes by M/K.
    This is a derived synthetic time axis, never a changed original UTC range.
    """
    s=np.asarray(segments,dtype=np.float64);k=len(s);m=int(target_segments)
    if s.ndim!=2 or s.shape[1]!=25 or k<1 or m!=target_segments or not 60<=m<=450 or not 1/3<=m/k<=3:
        raise ValueError('Unapproved timewarp support/stretch')
    if not np.isfinite(s).all() or (s<0).any():raise ValueError('Finite nonnegative linear powers required')
    edges=np.linspace(0,k,m+1);lo=np.floor(edges).astype(int);fraction=edges-lo
    cumulative=np.vstack([np.zeros((1,25)),np.cumsum(s,axis=0)])
    integral=cumulative[lo]+fraction[:,None]*s[np.minimum(lo,k-1)]
    result=np.diff(integral,axis=0)/(k/m)
    if (result < -1e-12*max(float(s.max()),1)).any():raise RuntimeError('Negative conservative rebinning power')
    result=np.maximum(result,0);mean=s.mean(0);error=float(np.max(np.abs(result.mean(0)-mean)/np.maximum(mean,1e-300)))
    if error>1e-10 or not np.isfinite(result).all():raise RuntimeError('Timewarp mean power identity failed')
    return result,{'source_coordinate_begin':0.,'source_coordinate_end':float(k),'source_coordinate_step':float(k/m),
        'mapping':'source_edge_j = j*K/M; monotone whole-source single pass',
        'timewarp_ratio':float(m/k),'power_mean_relative_error':error}


def make_passage_schedule(n,seed,config=PassageConfigR4()):
    """Full-split LHS marginals; independent u/duration shuffles, no retries."""
    if int(n)!=n or n<1:raise ValueError('Positive integer split size required')
    n=int(n)
    streams=[np.random.default_rng(s) for s in np.random.SeedSequence(int(seed)).spawn(5)]
    count_zero=int(round(config.zero_probability*n));positive=n-count_zero
    share=np.zeros(n,dtype=np.float64);duration=np.zeros(n,dtype=np.int64)
    positions=streams[0].permutation(n)[count_zero:]
    if positive:
        u=(np.arange(positive)+streams[1].random(positive))/positive
        d=(np.arange(positive)+streams[3].random(positive))/positive
        u=u[streams[2].permutation(positive)];d=d[streams[4].permutation(positive)]
        share[positions]=config.minimum_share+(config.maximum_share-config.minimum_share)*u
        width=config.rendered_maximum_segments-config.rendered_minimum_segments+1
        duration[positions]=config.rendered_minimum_segments+np.minimum((d*width).astype(int),width-1)
    return {'requested_share':share,'rendered_segments':duration,'schedule_seed':int(seed),'zero_count':count_zero,
        'scheme':'full-split independently shuffled LHS u and discrete duration; exact round(p_zero*N); no seed retry'}


def schedule_slice(schedule,begin,end):
    return {**schedule,'requested_share':schedule['requested_share'][begin:end],
        'rendered_segments':schedule['rendered_segments'][begin:end]}


def raised_cosine_power_weights(segments, fade_seconds=30.):
    """Average a(t)^2 over each 2s bin, using 16-point Gauss quadrature.

    a(t) has raised-cosine entry/exit and a flat middle. Multiplication of
    measured band summaries is controlled trajectory modulation, not a claim
    to reproduce the PSD of a newly filtered waveform exactly.
    """
    duration=2.*int(segments);fade=min(float(fade_seconds),duration/4)
    if segments<1 or fade<=0:raise ValueError('Positive duration/fade required')
    nodes,weights=np.polynomial.legendre.leggauss(16)
    t=2*np.arange(segments)[:,None]+1+nodes[None,:]
    ramp=np.minimum(t,duration-t)/fade
    amplitude=np.where(ramp<1,.5*(1-np.cos(np.pi*np.clip(ramp,0,1))),1.)
    result=(amplitude**2)@weights/2
    if not np.isfinite(result).all() or not ((result>0)&(result<=1+1e-12)).all():raise ValueError('Invalid fade weights')
    return result


def smooth_eq(rng, maximum_db):
    x=np.linspace(-1,1,25);a,b=rng.uniform(-maximum_db,maximum_db,2)
    curve=a*x+b*(x*x-np.mean(x*x));scale=min(1.,maximum_db/max(float(np.max(np.abs(curve))),1e-30))
    return curve*scale,{'tilt_db':float(a),'quadratic_db':float(b),'bound_scale':float(scale)}


class PassageSamplerR4:
    def __init__(self,background_db,background_manifest,references,split,seed=42,config=PassageConfigR4()):
        if split not in ['train','validation','outer_validation']:raise ValueError('Reserved/Test split forbidden')
        self.db=background_db;self.b=background_manifest.reset_index(drop=True).copy();self.refs=list(references)
        self.split=split;self.seed=int(seed);self.rng=np.random.default_rng(seed);self.config=config
        self.background_indices=np.flatnonzero(self.b.split.eq(split).to_numpy())
        if not len(self.background_indices):raise ValueError('Empty background split')
        self.pools={m:{} for m in range(config.rendered_minimum_segments,config.rendered_maximum_segments+1)}
        for j,r in enumerate(self.refs):
            r.validate()
            if r.metadata['split']!=split:raise ValueError('Cross-split reference access')
            group=r.metadata['reference_component']
            for m in self.pools:
                if config.minimum_stretch<=m/len(r.segments)<=config.maximum_stretch:
                    self.pools[m].setdefault(group,[]).append(j)
        missing=[m for m,pool in self.pools.items() if not pool]
        if missing:raise ValueError(f'Desired duration has no eligible real reference; never redraw it: {missing}')
        self._background_set=set(self.background_indices.tolist())

    def _reference(self,observed,rendered_segments):
        rng=self.rng;cfg=self.config;m=int(rendered_segments)
        groups=sorted(self.pools[m]);group=groups[int(rng.integers(len(groups)))];j=int(rng.choice(self.pools[m][group]))
        ref=self.refs[j];s=np.asarray(ref.segments,dtype=np.float64)
        source_segments=len(s);normalizer=float(s.sum(1).mean());s=s/normalizer
        s,warp=warp_band_power(s,m)
        fade=raised_cosine_power_weights(m,cfg.fade_seconds);s=s*fade[:,None]
        curve,parameters=smooth_eq(rng,cfg.reference_eq_max_db);s=s*10**(curve[None,:]/10)
        anchors=np.flatnonzero(np.repeat(observed.any(1),30));anchor=int(rng.choice(anchors))
        low=max(0,anchor-m+1);high=min(anchor,450-m)
        start=int(rng.integers(low,high+1));placed=np.zeros((450,25));placed[start:start+m]=s
        power=placed.reshape(15,30,25).mean(1)
        m=ref.metadata
        recipe={'clip_id':m['clip_id'],'source':m['source'],'group_key':m['reference_component'],
            'reference_component':m['reference_component'],'source_sha256':m['sha256'],
            'kind':'controlled_monotonic_band_power_timewarp','duration_bin':duration_bin(2*rendered_segments,cfg),
            'source_segments':source_segments,'rendered_segments':int(rendered_segments),'measured_used_seconds':2*source_segments,
            'derived_duration_seconds':2*int(rendered_segments),'original_duration_seconds':float(m['duration_seconds']),
            'source_segment_begin':0,'source_utc_begin':str(m.get('utc_begin','')),'source_utc_end':str(m.get('utc_end','')),
            'placement_segment':start,'fade_seconds':min(cfg.fade_seconds,int(rendered_segments)/2),
            'fade_definition':'mean squared raised-cosine amplitude per derived 2s bin, after conservative warp',**warp,
            'digital_power_normalization':normalizer,'reference_eq_db':curve.tolist(),
            'reference_eq_parameters':parameters,'reference_eq_applied_before_alpha':True,
            'waveform_repetition':False,'dB_interpolation':False,'linear_power_timewarp_applied':True,
            'original_UTC_modified':False}
        return power,recipe

    def draw(self,n,background_indices=None,schedule=None):
        if int(n)!=n or n<1:raise ValueError('Positive integer batch required')
        if background_indices is None:indices=self.rng.choice(self.background_indices,size=n,replace=True)
        else:
            indices=np.asarray(background_indices,dtype=np.int64)
            if indices.shape!=(n,) or any(int(i) not in self._background_set for i in indices):raise ValueError('Primary indices must belong to this source split')
        if schedule is None:schedule=make_passage_schedule(n,int(self.rng.integers(2**31)),self.config)
        targets=np.asarray(schedule['requested_share'],dtype=np.float64);durations=np.asarray(schedule['rendered_segments'],dtype=np.float64)
        if targets.shape!=(n,) or durations.shape!=(n,):raise ValueError('Full split schedule slice shape mismatch')
        if not np.isfinite(targets).all() or not np.isfinite(durations).all() or not np.equal(durations,np.floor(durations)).all() or (durations<0).any() or (durations>450).any():
            raise ValueError('Schedule requires finite targets and integer rendered segments, without truncation')
        durations=durations.astype(np.int64)
        arrays=[[],[],[],[]];items=[]
        for j,bi in enumerate(indices):
            row=self.b.iloc[int(bi)];x=np.array(self.db[int(row.base_window_index)],dtype=np.float64)
            o=np.isfinite(x)
            if o.any(1).sum()<3:raise ValueError('Insufficient natural observed support')
            peak=float(np.nanmax(x));b=np.where(o,10**((x-peak)/10),0)
            u=float(targets[j]);derived=int(durations[j])
            if not (u==0 and derived==0 or self.config.minimum_share<=u<self.config.maximum_share and self.config.rendered_minimum_segments<=derived<=self.config.rendered_maximum_segments):raise ValueError('Schedule outside declared prior')
            refs=[];r=np.zeros((15,25))
            if u>0:r,ref=self._reference(o,derived);refs=[ref]
            re=float(r[o].sum());be=float(b[o].sum())
            if u>0 and re<=0:raise RuntimeError('Positive passage failed observed overlap')
            alpha=0. if u==0 else u*be/((1-u)*re);added=alpha*np.where(o,r,0);e=b+added
            q=np.divide(added,e,out=np.zeros_like(e),where=o&(e>0));actual=float(added.sum()/e.sum())
            result=np.full((15,25),np.nan);result[o]=10*np.log10(e[o])+peak
            if not np.isclose(actual,u,atol=1e-12) or not np.isfinite(result[o]).all():raise RuntimeError('Known component identity failed')
            active=(added>0).any(1);bm=b.sum(1);am=added.sum(1)
            snr=[float(10*np.log10(a/z)) if a>0 and z>0 else None for a,z in zip(am,bm)]
            arrays[0].append(result.astype('float32'));arrays[1].append(np.float32(actual));arrays[2].append(np.float32(u>0));arrays[3].append(q.astype('float32'))
            items.append({'background_index':int(bi),'base_background_index':int(row.base_window_index),
                'background_group':str(row.background_group),'references':refs,'requested_share':u,'desired_share':actual,
                'addition_scale':alpha,'background_peak_db':peak,'observed_minutes':int(o.any(1).sum()),
                'active_ref_minutes':int(active.sum()),'support_coverage':float(active.sum()/o.any(1).sum()),
                'global_snr_db':float(10*np.log10(actual/(1-actual))) if actual>0 else None,
                'minute_snr_db':snr,'hard_snr_or_support_cap_applied':False,
                'background_blending_applied':False,'additional_dropout_applied':False,
                'common_gain_db':0.,'time_reversed':False})
        recipe={'version':'passage_r4','split':self.split,'seed':self.seed,'config':self.config.to_dict(),
            'background_index':np.asarray(indices).tolist(),'items':items,
            'schedule_seed':int(schedule['schedule_seed']),'schedule_scheme':schedule['scheme'],
            'same_transformation_policy_all_splits':True,'natural_observation_mask_only':True}
        return *(np.stack(v) for v in arrays),recipe
