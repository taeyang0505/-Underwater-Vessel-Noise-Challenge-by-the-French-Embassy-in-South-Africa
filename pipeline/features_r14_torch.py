"""Batched Torch counterpart of the frozen R12 relative8 + safe23 PCA maps.

Only db enters the transform. PCA is an already fitted fold-TRAIN nuisance
basis, stored as float64 buffers; no fitting or source/target query inputs.
FP64 internal arithmetic preserves the NumPy rounding/quantile contract.
CPU parity does not establish CUDA numerical parity, memory use or speed.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from pipeline.features_r12 import (validate_spectral_basis,FEATURE_NAMES,
                                  BASE_FEATURE_NAMES)

VERSION='r14_torch_relative11_safe23_v1'


class TorchFeatureFrontend(nn.Module):
    def __init__(self,basis=None,*,include_shape=True,pad_to_eleven=False):
        super().__init__()
        if type(include_shape)is not bool or type(pad_to_eleven)is not bool:raise TypeError('Explicit bool feature policies required')
        self.include_shape=include_shape;self.pad_to_eleven=pad_to_eleven
        if include_shape:
            validate_spectral_basis(basis);self.basis=copy.deepcopy(basis)
            mean=torch.tensor(basis['mean_db'],dtype=torch.float64)
            components=torch.tensor(basis['components'],dtype=torch.float64)
        else:
            self.basis=None;mean=torch.empty(0,dtype=torch.float64);components=torch.empty((0,23),dtype=torch.float64)
        self.register_buffer('mean_db',mean);self.register_buffer('components',components)
    @property
    def feature_names(self):
        return FEATURE_NAMES if self.include_shape or self.pad_to_eleven else BASE_FEATURE_NAMES
    def metadata(self):
        return {'version':VERSION,'include_shape':self.include_shape,'pad_to_eleven':self.pad_to_eleven,
            'basis':copy.deepcopy(self.basis),'internal_dtype':'float64','output_dtype':'float32',
            'input_contract':'db-only; optional finite-cell mask must match NaNs exactly',
            'quantization_decimals':1,'query_population_fit':False,'feature_names':list(self.feature_names)}
    def get_extra_state(self):return self.metadata()
    def set_extra_state(self,state):
        if state!=self.metadata():raise ValueError('Foreign frontend/checkpoint metadata')
    def verify_frozen_buffers(self):
        if self.mean_db.dtype!=torch.float64 or self.components.dtype!=torch.float64:
            raise ValueError('FP64 frontend buffers must not be cast by model.half()/float(); use autocast for the backbone')
        if self.include_shape:
            if not torch.equal(self.mean_db.detach().cpu(),torch.tensor(self.basis['mean_db'],dtype=torch.float64)) or not torch.equal(self.components.detach().cpu(),torch.tensor(self.basis['components'],dtype=torch.float64)):
                raise ValueError('PCA buffers differ from their pinned TRAIN metadata')
    def _shape(self,x,observed):
        safe=x[:,:,:23].reshape(-1,23);obs=observed[:,:,:23].reshape(-1,23)
        counts=obs.sum(1,keepdim=True)
        center=torch.where(obs,safe,0).sum(1,keepdim=True)/counts.clamp_min(1)
        values=torch.where(obs,safe-center-self.mean_db,0.)
        coeff=values@self.components.T
        errors=(values-coeff@self.components)/10.
        residual=torch.where(obs & obs.all(1,keepdim=True),errors,0.)
        partial=obs.any(1)&~obs.all(1)
        # Production full-band minutes use the batched matmul above. This
        # explicit availability-pattern path preserves the old observed-only
        # least-squares semantics for genuinely partial safe-band spectra.
        if bool(partial.any()):
            ids=torch.nonzero(partial,as_tuple=False).flatten()
            patterns,inverse=torch.unique(obs[ids],dim=0,return_inverse=True)
            for j,mask in enumerate(patterns):
                rows=ids[inverse==j];cols=torch.nonzero(mask,as_tuple=False).flatten()
                local=values[rows][:,cols];axes=self.components[:,cols]
                projection=torch.linalg.pinv(axes.T,atol=0.,rtol=1e-15).T
                err=(local-(local@projection)@axes)/10.
                # Avoid any per-row/window PCA fitting: this is only projection
                # onto the same frozen axes over measured frequencies.
                residual[rows[:,None],cols[None,:]]=err
        signed=torch.zeros_like(x);signed[:,:,:23]=residual.reshape(-1,15,23)
        curvature=torch.zeros_like(x)
        valid=observed[:,:,9:21]&observed[:,:,10:22]&observed[:,:,11:23]
        curve=(2*x[:,:,10:22]-x[:,:,9:21]-x[:,:,11:23])/10.
        curvature[:,:,10:22]=torch.where(valid,curve,0.)
        return torch.stack((signed,signed.clamp_min(0),curvature),1).float()
    def forward(self,db,observed_mask=None):
        if not isinstance(db,torch.Tensor) or db.ndim!=3 or tuple(db.shape[1:])!=(15,25) or not len(db) or not db.is_floating_point():
            raise ValueError('Floating Tensor db[B,15,25] required; labels/dictionaries are forbidden')
        if self.components.device!=db.device:raise ValueError('Frontend and raw db must share the same device')
        if self.components.dtype!=torch.float64 or self.mean_db.dtype!=torch.float64:raise ValueError('Frontend FP64 buffers were cast')
        finite=torch.isfinite(db)
        if bool(torch.isinf(db).any()) or not bool(finite.any((1,2)).all()):raise ValueError('Inf or wholly unobserved window')
        if observed_mask is not None:
            if (not isinstance(observed_mask,torch.Tensor) or observed_mask.dtype!=torch.bool
                or observed_mask.device!=db.device or observed_mask.shape!=db.shape or not torch.equal(observed_mask,finite)):
                raise ValueError('Explicit observed mask differs from finite raw cells')
        # NumPy np.round(float64,1) uses scaled nearest-even rounding.
        x=torch.round(db.to(torch.float64)*10.)/10.
        observed=torch.isfinite(x)
        if not torch.equal(observed,finite):raise ValueError('Quantization overflow')
        quantiles=torch.nanquantile(x,x.new_tensor([.5,.2,.05]),dim=1,keepdim=True,interpolation='linear')
        med,p20,q05=quantiles.unbind(0)
        win=torch.nanquantile(x.flatten(1),.5,dim=1,keepdim=True,interpolation='linear')[:,:,None]
        delta=torch.zeros_like(x);pairs=observed[:,1:]&observed[:,:-1]
        delta[:,1:]=torch.where(pairs,x[:,1:]-x[:,:-1],0.)/10.
        lowpairs=pairs[:,:,:9];count=lowpairs.sum(2,keepdim=True)
        lowpos=torch.where(lowpairs,(delta[:,1:,:9]*10.).clamp_min(0),0.)
        lowmean=lowpos.sum(2,keepdim=True)/count.clamp_min(1)
        corise=torch.zeros_like(x[:,:,:1]);corise[:,1:]=torch.tanh(lowmean/3.)
        coherence=delta.clamp_min(0)*corise;coherence[:,:,:10]=0.
        zero=torch.zeros_like(x)
        base=torch.stack((torch.where(observed,(x-med)/10.,zero),
            torch.where(observed,(x-p20)/15.,zero),delta,coherence,
            torch.where(observed,(med-win).expand_as(x)/20.,zero),observed.to(x.dtype)),1).float()
        peak=torch.where(observed,x,-torch.inf).amax((1,2),keepdim=True)
        energy=torch.where(observed,torch.pow(10.,(x-peak)/10.),0.)
        if not bool(torch.isfinite(energy).all()) or not bool((energy[observed]>0).all()):raise ValueError('Observed energy underflow/overflow unsupported')
        w=(energy/energy.sum((1,2),keepdim=True)).float()
        extra=torch.where(observed,(x-q05)/15.,zero).float()
        out=torch.cat((base,extra[:,None],torch.sqrt(w)[:,None]),1)
        if self.include_shape:out=torch.cat((out,self._shape(x,observed)),1)
        elif self.pad_to_eleven:out=torch.cat((out,torch.zeros_like(out[:,:3])),1)
        if not bool(torch.isfinite(out).all()) or not torch.equal(out[:,5],(w>0).float()):raise ValueError('Finite feature/mask invariant failed')
        return out,w


def transform(db,basis=None,*,include_shape=True):
    """Functional parity API; reuse the Module for repeated training batches."""
    if not isinstance(db,torch.Tensor):raise TypeError('Torch Tensor db required')
    frontend=TorchFeatureFrontend(basis,include_shape=include_shape).to(device=db.device)
    return frontend(db)
