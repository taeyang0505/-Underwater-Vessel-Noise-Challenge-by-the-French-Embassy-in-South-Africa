import torch
from torch import nn
from tabm import TabM

class TabMRegressor(nn.Module):
    def __init__(self,n_features=1254,width=256,blocks=3,k=16):
        super().__init__()
        self.backbone=TabM.make(n_num_features=n_features,d_out=2,d_block=width,n_blocks=blocks,
                               k=k,dropout=.1,arch_type='tabm')
    def forward(self,x):return self.backbone(x)

class ResidualBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(4,width),nn.GELU(),
                                nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(4,width))
    def forward(self,x):return torch.nn.functional.gelu(x+self.body(x))

class EnergyMaskCNN(nn.Module):
    """Predict a bounded component fraction in each cell, then conserve energy."""
    def __init__(self,width=32,blocks=3):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(4,width,3,padding=1),nn.GELU(),
                                *[ResidualBlock(width) for _ in range(blocks)])
        self.mask_head=nn.Conv2d(width,1,1)
        self.presence_head=nn.Linear(width,1)
    def forward(self,x,energy_weights):
        h=self.body(x);cell=torch.sigmoid(self.mask_head(h)[:,0])
        share=(cell*energy_weights).sum((1,2))
        presence=self.presence_head(h.mean((2,3)))[:,0]
        return share,presence,cell

class FrequencyDynamicBlock(nn.Module):
    """Independent adaptation of FDY: frequency-conditioned mixtures of kernels.

    Preserves all 25 physical bands; it does not reproduce the original SED model.
    """
    def __init__(self,width,bases=4):
        super().__init__();self.width=width;self.bases=bases
        self.kernels=nn.Conv2d(width,width*bases,3,padding=1)
        self.gate=nn.Sequential(nn.Conv1d(width+1,width,1),nn.GELU(),nn.Conv1d(width,bases,1))
        self.norm=nn.GroupNorm(4,width)
    def forward(self,x):
        n,c,t,f=x.shape
        pos=torch.linspace(-1,1,f,device=x.device,dtype=x.dtype).expand(n,1,f)
        weights=self.gate(torch.cat([x.mean(2),pos],1)).softmax(1)
        h=self.kernels(x).reshape(n,self.bases,c,t,f)
        h=(h*weights[:,:,None,None,:]).sum(1)
        return torch.nn.functional.gelu(x+self.norm(h))

class FDYMaskCRNN(nn.Module):
    def __init__(self,width=32,blocks=3):
        super().__init__()
        self.stem=nn.Conv2d(4,width,3,padding=1)
        self.blocks=nn.Sequential(*[FrequencyDynamicBlock(width) for _ in range(blocks)])
        self.temporal=nn.GRU(width,width,batch_first=True,bidirectional=True)
        self.mask_head=nn.Linear(2*width,1);self.presence_head=nn.Linear(2*width,1)
    def forward(self,x,energy_weights):
        h=self.blocks(torch.nn.functional.gelu(self.stem(x)))
        n,c,t,f=h.shape
        h,_=self.temporal(h.permute(0,3,2,1).reshape(n*f,t,c))
        h=h.reshape(n,f,t,2*c)
        cell=self.mask_head(h).squeeze(-1).transpose(1,2).sigmoid()
        share=(cell*energy_weights).sum((1,2))
        presence=self.presence_head(h.mean((1,2))).squeeze(-1)
        return share,presence,cell

class InvertedMaskTransformer(nn.Module):
    """iTransformer-inspired band tokens, adapted for energy-fraction regression."""
    def __init__(self,width=96,blocks=3):
        super().__init__()
        self.embed=nn.Linear(4*15,width)
        self.band_embedding=nn.Parameter(torch.randn(1,25,width)*.02)
        layer=nn.TransformerEncoderLayer(width,4,width*3,.1,batch_first=True,norm_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,blocks,enable_nested_tensor=False)
        self.mask_head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,15))
        self.presence_head=nn.Linear(width,1)
    def forward(self,x,energy_weights):
        n=x.shape[0];tokens=x.permute(0,3,1,2).reshape(n,25,60)
        h=self.encoder(self.embed(tokens)+self.band_embedding)
        cell=self.mask_head(h).transpose(1,2).sigmoid()
        return (cell*energy_weights).sum((1,2)),self.presence_head(h.mean(1)).squeeze(-1),cell

def make_model(kind,width=256,k=16):
    if kind=='tabm':return TabMRegressor(width=width,k=k)
    if kind=='cnn':return EnergyMaskCNN()
    if kind=='fdy':return FDYMaskCRNN()
    if kind=='itransformer':return InvertedMaskTransformer()
    raise ValueError(kind)
