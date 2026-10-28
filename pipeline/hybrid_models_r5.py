"""The two M-scale branches of the requested R5 five-branch ensemble.

TF-grid + frequency-dynamic convolution + a Macaron Conformer is a new
energy-grid composition, NOT an official 'Full FDY-Conformer' reproduction.
TF-grid source: https://arxiv.org/abs/2211.12433
FDY source: https://www.isca-archive.org/interspeech_2022/nam22_interspeech.html
Conformer source: https://arxiv.org/abs/2005.08100

Conformer differences: short 15-step sequences, batch-independent GroupNorm
inside its convolution module, standard noncausal attention, energy mask head.
All eight channels are supplied by the SAME fold-TRAIN-fitted input adapter.
"""
import torch
from torch import nn
from pipeline.grid_models_r5 import TFGridEnergyMask
from pipeline.models import EnergyMaskCNN,FrequencyDynamicBlock


class _FeedForward(nn.Module):
    def __init__(self,width,dropout):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,4*width),nn.SiLU(),
            nn.Dropout(dropout),nn.Linear(4*width,width),nn.Dropout(dropout))
    def forward(self,x):return self.net(x)


class _ConvolutionModule(nn.Module):
    def __init__(self,width,dropout,kernel=7):
        super().__init__();self.norm=nn.LayerNorm(width)
        self.net=nn.Sequential(nn.Conv1d(width,2*width,1),nn.GLU(dim=1),
            nn.Conv1d(width,width,kernel,padding=kernel//2,groups=width),
            nn.GroupNorm(1,width),nn.SiLU(),nn.Conv1d(width,width,1),nn.Dropout(dropout))
    def forward(self,x):return self.net(self.norm(x).transpose(1,2)).transpose(1,2)


class EnergyConformerBlock(nn.Module):
    def __init__(self,width=64,heads=4,dropout=.1):
        super().__init__();self.first=_FeedForward(width,dropout)
        self.attention_norm=nn.LayerNorm(width)
        self.attention=nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
        self.dropout=nn.Dropout(dropout);self.convolution=_ConvolutionModule(width,dropout)
        self.second=_FeedForward(width,dropout);self.final=nn.LayerNorm(width)
    def forward(self,x):
        x=x+.5*self.first(x);h=self.attention_norm(x)
        x=x+self.dropout(self.attention(h,h,h,need_weights=False)[0])
        x=x+self.convolution(x);return self.final(x+.5*self.second(x))


class TFGridFDYConformerMask(nn.Module):
    def __init__(self,embed_dim=64,hidden_size=128,grid_blocks=5,dropout=.1,conformer_blocks=1):
        super().__init__()
        if type(conformer_blocks) is not int or conformer_blocks<1:raise ValueError('Positive Conformer depth required')
        self.grid=TFGridEnergyMask(embed_dim=embed_dim,hidden_size=hidden_size,blocks=grid_blocks,dropout=dropout)
        self.dynamic=FrequencyDynamicBlock(embed_dim,bases=4,dropout=dropout)
        self.conformer=nn.Sequential(*[EnergyConformerBlock(embed_dim,4,dropout) for _ in range(conformer_blocks)])
        self.model_config=dict(embed_dim=embed_dim,hidden_size=hidden_size,grid_blocks=grid_blocks,
            dropout=dropout,conformer_blocks=conformer_blocks)
        self.input_channels=8;self.feature_version=3;self.kind='tfgrid_fdy_conformer'
    def encode(self,x):
        h=self.dynamic(self.grid.encode(x));n,c,t,f=h.shape
        temporal=h.permute(0,3,2,1).reshape(n*f,t,c)
        return self.conformer(temporal).reshape(n,f,t,c).permute(0,3,2,1)
    def representation(self,x):return self.encode(x).mean((2,3))
    def forward(self,x,energy_weights):
        h=self.encode(x);q=self.grid.mask_head(h)[:,0].sigmoid()
        if q.shape!=energy_weights.shape:raise ValueError('Energy-grid shape mismatch')
        return (q*energy_weights).sum((1,2)),self.grid.presence_head(h.mean((2,3)))[:,0],q


class WideBackgroundCNN(nn.Module):
    """New ~1.5M direct-mask model, not the 58k-parameter R2 checkpoint."""
    def __init__(self,width=128,blocks=5,dropout=.1):
        super().__init__()
        if width<4 or width%4 or blocks<1 or not 0<=dropout<1:raise ValueError('Invalid wide CNN dimensions')
        self.network=EnergyMaskCNN(width=width,blocks=blocks,dropout=dropout,input_channels=8)
        self.model_config=dict(width=width,blocks=blocks,dropout=dropout)
        self.input_channels=8;self.feature_version=3;self.kind='bg_cnn'
    def encode(self,x):
        if x.ndim!=4 or tuple(x.shape[1:])!=(8,15,25):raise ValueError('Expected shared [B,8,15,25] input')
        return self.network.body(x)
    def representation(self,x):return self.encode(x).mean((2,3))
    def forward(self,x,energy_weights):return self.network(x,energy_weights)


def make_hybrid_model(kind,**settings):
    if kind=='tfgrid_fdy_conformer':return TFGridFDYConformerMask(**settings)
    if kind=='bg_cnn':return WideBackgroundCNN(**settings)
    raise ValueError('Unsupported R5 M-scale branch: '+str(kind))
