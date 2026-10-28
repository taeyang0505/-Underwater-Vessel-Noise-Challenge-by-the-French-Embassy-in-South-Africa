"""Production14: exact R12 relative11 plus the frozen R14 anchor3 formula.

FP64 CPU frontend, FP32 output. The MPS backbone receives only x/w; the
frontend must remain on CPU. No fitting or source/target input is accepted.
"""
import torch
from pipeline.features_r14_torch import TorchFeatureFrontend
from pipeline.features_r12 import FEATURE_NAMES as BASE_NAMES
from pipeline.features_r14_anchor import ANCHOR_NAMES
FEATURE_NAMES=BASE_NAMES+ANCHOR_NAMES
VERSION='r14_production_anchor14_cpuFP64_v1'

class ProductionFrontend(TorchFeatureFrontend):
    def __init__(self,basis):super().__init__(basis,include_shape=True)
    @property
    def feature_names(self):return FEATURE_NAMES
    def metadata(self):return {**super().metadata(),'version':VERSION,'output_channels':14,'frontend_device':'cpu'}
    def forward(self,db,observed_mask=None):
        if not isinstance(db,torch.Tensor) or db.device.type!='cpu':raise ValueError('Production FP64 frontend accepts CPU raw db only')
        base,w=super().forward(db,observed_mask)
        d=torch.round(db.double()*10.)/10.;o=torch.isfinite(d)
        low=torch.nanmean(d[:,:,:8],2,keepdim=True);mid=torch.nanmean(d[:,:,9:21],2,keepdim=True)
        offset=self.mean_db-self.mean_db[:8].mean();valid=o&torch.isfinite(low)
        lift=(mid-low)/20.;lift=torch.where(torch.isfinite(lift),lift,0.)
        a=torch.where(valid,lift.expand_as(d),0.);a[:,:,23:]=0.
        residual=d[:,:,:23]-(low+offset[None,None,:])
        b=torch.zeros_like(d);c=torch.zeros_like(d)
        b[:,:,:23]=torch.where(valid[:,:,:23],residual/20.,0.)
        frac=(1-torch.pow(10.,(-residual/10.).clamp(-30,30))).clamp_min(0)
        c[:,:,:23]=torch.where(valid[:,:,:23],frac,0.)
        output=torch.cat((base,torch.stack((a,b,c),1).float()),1)
        if not bool(torch.isfinite(output).all()) or bool((output.masked_select((~o)[:,None].expand_as(output))!=0).any()):raise ValueError('Production14 mask/finite contract')
        return output,w

def transform(db,basis):return ProductionFrontend(basis)(db)
