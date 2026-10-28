"""Opt-in native-first generator; it never alters the frozen v1 generator.

Callers must provide a versioned background manifest and verified reference
records. New references can be attached after the external acquisition audit.
The sealed source split requires explicit opt-in and is never a default pool.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.augmentation_v2 import (EnergyComponents,solve_reference_scale,
    UnobservedReferenceError,place_measured_segments,average_complete_minutes,
    blend_backgrounds)

@dataclass(frozen=True)
class ReferenceRecord:
    clip_id: str
    source: str
    group_key: str
    split: str
    segments: np.ndarray
    measured: bool = True
    verified: bool = False
    role: str = 'vessel_passage'
    quality_full_band_good: bool = False
    provenance: str = ''

    def validate(self):
        s=np.asarray(self.segments)
        if not self.measured or not self.verified:raise ValueError('Only verified measured references are eligible')
        if not self.quality_full_band_good:raise ValueError('Full-band quality gate failed')
        if s.ndim!=2 or s.shape[1]!=25 or len(s)==0:raise ValueError('Expected measured native 2-s × 25 band energies')
        if not np.isfinite(s).all() or (s<0).any() or not (s.sum(1)>0).all():
            raise ValueError('Invalid measured reference band energies')

@dataclass(frozen=True)
class AugmentationConfig:
    native_long_fraction: float = .5
    high_background_coverage_fraction: float = .8
    multi_reference_probability: float = .2
    time_reverse_probability: float = .5
    common_gain_range_db: float = 8.
    whole_minute_dropout_probability: float = .1
    common_response_range_db: float = 0.
    background_blend_probability: float = 0.
    measured_ambient_probability: float = 0.

    def __post_init__(self):
        for key,value in vars(self).items():
            if not np.isfinite(value):raise ValueError(f'{key} must be finite')
            if key.endswith('_db'):
                if value<0:raise ValueError(f'{key} must be nonnegative')
            elif not 0<=value<=1:raise ValueError(f'{key} must be a probability/fraction in [0,1]')

class NativeFirstSampler:
    def __init__(self,background_db,background_manifest,references,split='train',seed=20261024,
                 config=AugmentationConfig(),allow_sealed_holdout=False,require_native_long=True):
        if split=='sealed_holdout' and not allow_sealed_holdout:
            raise ValueError('Sealed source access requires an explicit final-evaluation opt-in')
        self.db=background_db;self.bmeta=background_manifest.reset_index(drop=True).copy()
        self.references=list(references);self.config=config;self.rng=np.random.default_rng(seed);self.split=split;self.seed=seed
        # Do not touch sealed or other partition arrays when initializing a
        # training sampler. Only metadata are inspected for global isolation.
        group_folds={}
        for r in self.references:
            if r.split==split:r.validate()
            group_folds.setdefault((r.source,r.group_key),set()).add(r.split)
        if any(len(folds)>1 for folds in group_folds.values()):
            raise ValueError('A source group crosses reference partitions')
        bsplit='domain_holdout' if split=='sealed_holdout' else split
        self.bi=np.flatnonzero(self.bmeta.split.to_numpy()==bsplit)
        self.ei=np.array([i for i,r in enumerate(self.references) if r.split==split and r.role=='vessel_passage'],dtype=int)
        self.ai=np.array([i for i,r in enumerate(self.references) if r.split==split and r.role=='ambient_control'],dtype=int)
        self.long=self.ei[[len(self.references[i].segments)>=450 for i in self.ei]]
        self.short=self.ei[[len(self.references[i].segments)<450 for i in self.ei]]
        if not len(self.bi) or not len(self.ei):raise ValueError(f'No eligible source-isolated {split} pool')
        if require_native_long and not len(self.long):
            raise RuntimeError(f'{split} has no verified native 15-minute positive reference; acquire one before native-first training')
        self.site_pools={site:g.index.to_numpy() for site,g in self.bmeta.iloc[self.bi].groupby('site')}

    def _background(self):
        rng=self.rng;cfg=self.config
        site=rng.choice(list(self.site_pools));pool=self.site_pools[site]
        high=pool[self.bmeta.observed_minutes.iloc[pool].to_numpy()>=12]
        sparse=pool[self.bmeta.observed_minutes.iloc[pool].to_numpy()<12]
        preferred=high if rng.random()<cfg.high_background_coverage_fraction else sparse
        if not len(preferred):preferred=pool
        i=int(rng.choice(preferred));x=np.array(self.db[int(self.bmeta.base_window_index.iloc[i])],dtype=np.float64)
        observed=np.isfinite(x);level=float(np.nanmax(x))
        b=np.where(observed,10**((x-level)/10),0)
        second=-1;beta=0.;second_gain=1.
        if rng.random()<cfg.background_blend_probability:
            other_sites=[s for s in self.site_pools if s!=site]
            for _ in range(12):
                if not other_sites:break
                j=int(rng.choice(self.site_pools[rng.choice(other_sites)]))
                y=np.array(self.db[int(self.bmeta.base_window_index.iloc[j])],dtype=np.float64)
                joint=observed&np.isfinite(y)
                if joint.all(1).sum()<3:continue
                ypeak=float(np.nanmax(y));b2=np.where(np.isfinite(y),10**((y-ypeak)/10),0)
                beta=float(rng.uniform(.25,.75));b,observed,second_gain=blend_backgrounds(b,b2,observed,np.isfinite(y),beta)
                second=j;break
        dropped=[]
        if rng.random()<cfg.whole_minute_dropout_probability:
            minutes=np.flatnonzero(observed.all(1))
            if len(minutes)>3:
                k=int(rng.choice(minutes));observed[k]=False;b[k]=0;dropped=[k]
        return i,b,observed,level,{'secondary_background_index':second,'background_blend_beta':beta,
            'background_secondary_digital_scale':second_gain,'dropped_minutes':dropped}

    def _hierarchical_reference(self,pool):
        """Balance corpus -> source group -> recording, not raw file counts."""
        rng=self.rng
        sources=sorted({self.references[i].source for i in pool})
        source=rng.choice(sources);indices=[i for i in pool if self.references[i].source==source]
        groups=sorted({self.references[i].group_key for i in indices});group=rng.choice(groups)
        return int(rng.choice([i for i in indices if self.references[i].group_key==group]))

    def _placed_reference(self,j,observed):
        rng=self.rng;s=np.asarray(self.references[j].segments,dtype=np.float64)
        # One digital level normalization preserves measured time/frequency ratios.
        s=s/max(float(s.sum(1).mean()),1e-30)
        if len(s)>=450:
            begin=int(rng.integers(len(s)-450+1));placed=s[begin:begin+450]
            return placed,{'kind':'native_contiguous_15min_crop','source_segment_begin':begin,'placement_segment':0}
        # Condition the chosen event's overlap on a real observed cell, avoiding
        # a missingness-dependent positive-draw -> zero-target shortcut.
        anchor=int(rng.choice(np.flatnonzero(np.repeat(observed.any(1),30))))
        source_anchor=int(rng.integers(len(s)));start=anchor-source_anchor
        placed=place_measured_segments(s,start)
        return placed,{'kind':'native_short_once_with_declared_controlled_support','source_segment_begin':0,'placement_segment':start}

    def draw(self,n):
        if int(n)!=n or n<1:raise ValueError('Positive integer batch size required')
        rng=self.rng;cfg=self.config;db=[];targets=[];presence=[];cells=[];items=[]
        for _ in range(n):
            bi,b,observed,level,item=self._background()
            if cfg.measured_ambient_probability and len(self.ai) and rng.random()<cfg.measured_ambient_probability:
                j=self._hierarchical_reference(self.ai)
                noise,noise_recipe=self._placed_reference(j,observed)
                ambient=average_complete_minutes(noise)
                mass=float(ambient[observed].sum())
                noise_scale=float(b[observed].sum()/mass*10**rng.uniform(-1.5,-.3))
                b=b+noise_scale*ambient
                item['ambient_reference']={'reference_index':j,'scale':noise_scale,**noise_recipe}
            u=0. if rng.random()<.2 else float(rng.uniform(.005,.98))
            total_ref=np.zeros((15,25));refs=[]
            if u>0:
                count=1 if rng.random()>=cfg.multi_reference_probability else int(rng.integers(2,4))
                count=min(count,len(self.ei));used=set()
                for _ in range(count):
                    pool=self.long if len(self.long) and (not len(self.short) or rng.random()<cfg.native_long_fraction) else self.short
                    pool=np.array([j for j in pool if j not in used],dtype=int)
                    if not len(pool):pool=np.array([j for j in self.ei if j not in used],dtype=int)
                    j=self._hierarchical_reference(pool);piece,placement=self._placed_reference(j,observed)
                    used.add(j)
                    relative_scale=float(10**rng.uniform(-.3,.3))
                    total_ref+=average_complete_minutes(piece)*relative_scale
                    refs.append({'reference_index':j,'clip_id':self.references[j].clip_id,
                        'source':self.references[j].source,'group_key':self.references[j].group_key,
                        'relative_scale':relative_scale,**placement})
            response=np.zeros(25)
            if cfg.common_response_range_db:
                response=np.interp(np.arange(25),[0,12,24],rng.uniform(-cfg.common_response_range_db,cfg.common_response_range_db,3))
                factor=10**(response/10);b=b*factor;total_ref=total_ref*factor
            alpha,mixture=solve_reference_scale(b,total_ref,observed,u)
            reverse=bool(rng.random()<cfg.time_reverse_probability)
            if reverse:mixture=mixture.reverse_time()
            gain=float(rng.uniform(-cfg.common_gain_range_db,cfg.common_gain_range_db))
            mixture=mixture.common_gain(gain)
            sample=mixture.to_db()+level
            assert np.isclose(mixture.share,u,atol=1e-10)
            assert np.isclose((mixture.cell_share*mixture.energy_weights).sum(),u,atol=1e-10)
            db.append(sample.astype('float32'));targets.append(mixture.share);presence.append(u>0);cells.append(mixture.cell_share.astype('float32'))
            items.append({'background_index':bi,'base_background_index':int(self.bmeta.base_window_index.iloc[bi]),
                'references':refs,'desired_share':u,'addition_scale':alpha,'common_gain_db':gain,
                'common_response_db':response.tolist(),'time_reversed':reverse,
                'observed_minutes':int(mixture.observed.all(1).sum()),**item})
        recipes={'version':'native_first_v2','split':self.split,'seed':self.seed,
            'background_index':np.array([x['background_index'] for x in items]),'items':items,
            'config':vars(cfg),'surrogate_envelopes_used':False,'reference_repetition_used':False}
        return np.stack(db),np.asarray(targets,dtype='float32'),np.asarray(presence,dtype='float32'),np.stack(cells),recipes
