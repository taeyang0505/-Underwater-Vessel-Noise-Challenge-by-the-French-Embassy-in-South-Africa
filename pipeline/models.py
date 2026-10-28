import torch
from torch import nn
from tabm import TabM

class TabMRegressor(nn.Module):
    def __init__(self,n_features=1254,width=256,blocks=3,k=16,dropout=.1):
        super().__init__()
        self.backbone=TabM.make(n_num_features=n_features,d_out=2,d_block=width,n_blocks=blocks,
                               k=k,dropout=dropout,arch_type='tabm')
    def forward(self,x):return self.backbone(x)

class ResidualBlock(nn.Module):
    def __init__(self,width,dropout=0.):
        super().__init__()
        layers=[nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(4,width),nn.GELU()]
        if dropout:layers.append(nn.Dropout2d(dropout))
        self.body=nn.Sequential(*layers,nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(4,width))
    def forward(self,x):return torch.nn.functional.gelu(x+self.body(x))

class EnergyMaskCNN(nn.Module):
    """Predict a bounded component fraction in each cell, then conserve energy."""
    def __init__(self,width=32,blocks=3,dropout=0.,input_channels=4):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(input_channels,width,3,padding=1),nn.GELU(),
                                *[ResidualBlock(width,dropout) for _ in range(blocks)])
        self.mask_head=nn.Conv2d(width,1,1)
        self.presence_head=nn.Linear(width,1)
    def forward(self,x,energy_weights):
        h=self.body(x);cell=torch.sigmoid(self.mask_head(h)[:,0])
        share=(cell*energy_weights).sum((1,2))
        presence=self.presence_head(h.mean((2,3)))[:,0]
        return share,presence,cell
    def representation(self,x):return self.body(x).mean((2,3))

class BackgroundPriorResidualCNN(nn.Module):
    """Validated OPS hybrid, kept separate from all historical architectures.

    The caller builds eight channels from db and a TRAIN-fitted aggregate
    prior. The prior is softened before a zero-initialized logit residual.
    ``last_residual`` retains the current forward graph for its explicit loss.
    """
    def __init__(self,width=32,dropout=.1,prior_clip=.01,prior_strength=.5):
        super().__init__()
        if width<=0 or width%4 or not 0<=dropout<1:raise ValueError('Invalid hybrid width/dropout')
        if not 0<prior_clip<.5 or not 0<prior_strength<=1:raise ValueError('Invalid softened background prior')
        self.network=EnergyMaskCNN(width=width,dropout=dropout,input_channels=8)
        nn.init.zeros_(self.network.mask_head.weight);nn.init.zeros_(self.network.mask_head.bias)
        self.prior_clip=prior_clip;self.prior_strength=prior_strength
        self.feature_version=3;self.preprocessing_version=3;self.architecture_version=4
        self.input_channels=8;self.requires_background_prior=True
        self.background_prior=None;self.last_residual=None
    def forward(self,x,energy_weights,return_residual=False):
        h=self.network.body(x);residual=self.network.mask_head(h)[:,0]
        self.last_residual=residual
        q0=x[:,6].clamp(self.prior_clip,1-self.prior_clip)
        cell=torch.sigmoid(self.prior_strength*torch.logit(q0)+residual)
        share=(cell*energy_weights).sum((1,2))
        presence=self.network.presence_head(h.mean((2,3)))[:,0]
        return (share,presence,cell,residual) if return_residual else (share,presence,cell)
    def representation(self,x):return self.network.body(x).mean((2,3))


class BackgroundPriorInputCNN(nn.Module):
    """R4: eight input channels, freely learned sigmoid energy mask.

    Aggregate TRAIN-only priors are features, never an additive output logit.
    Architecture 4 remains unchanged for all historical R2 checkpoints.
    """
    def __init__(self,width=32,dropout=.1):
        super().__init__()
        self.network=EnergyMaskCNN(width=width,dropout=dropout,input_channels=8)
        self.feature_version=3;self.preprocessing_version=3;self.architecture_version=5
        self.input_channels=8;self.requires_background_prior=True
        self.background_prior=None
    def forward(self,x,energy_weights):return self.network(x,energy_weights)
    def representation(self,x):return self.network.representation(x)


class FrequencyDynamicBlock(nn.Module):
    """Independent adaptation of FDY: frequency-conditioned mixtures of kernels.

    Preserves all 25 physical bands; it does not reproduce the original SED model.
    """
    def __init__(self,width,bases=4,dropout=0.):
        super().__init__();self.width=width;self.bases=bases
        self.kernels=nn.Conv2d(width,width*bases,3,padding=1)
        self.gate=nn.Sequential(nn.Conv1d(width+1,width,1),nn.GELU(),nn.Conv1d(width,bases,1))
        self.norm=nn.GroupNorm(4,width)
        self.dropout=nn.Dropout2d(dropout)
    def forward(self,x):
        n,c,t,f=x.shape
        pos=torch.linspace(-1,1,f,device=x.device,dtype=x.dtype).expand(n,1,f)
        weights=self.gate(torch.cat([x.mean(2),pos],1)).softmax(1)
        h=self.kernels(x).reshape(n,self.bases,c,t,f)
        h=(h*weights[:,:,None,None,:]).sum(1)
        return torch.nn.functional.gelu(x+self.dropout(self.norm(h)))

class FDYMaskCRNN(nn.Module):
    def __init__(self,width=32,blocks=3,dropout=0.,input_channels=4):
        super().__init__()
        self.stem=nn.Conv2d(input_channels,width,3,padding=1)
        self.blocks=nn.Sequential(*[FrequencyDynamicBlock(width,dropout=dropout) for _ in range(blocks)])
        self.temporal=nn.GRU(width,width,batch_first=True,bidirectional=True)
        self.dropout=nn.Dropout(dropout)
        self.mask_head=nn.Linear(2*width,1);self.presence_head=nn.Linear(2*width,1)
    def _hidden(self,x):
        h=self.blocks(torch.nn.functional.gelu(self.stem(x)))
        n,c,t,f=h.shape
        h,_=self.temporal(h.permute(0,3,2,1).reshape(n*f,t,c))
        return self.dropout(h.reshape(n,f,t,2*c))
    def representation(self,x):return self._hidden(x).mean((1,2))
    def forward(self,x,energy_weights):
        h=self._hidden(x)
        cell=self.mask_head(h).squeeze(-1).transpose(1,2).sigmoid()
        share=(cell*energy_weights).sum((1,2))
        presence=self.presence_head(h.mean((1,2))).squeeze(-1)
        return share,presence,cell

class InvertedMaskTransformer(nn.Module):
    """iTransformer-inspired band tokens, adapted for energy-fraction regression."""
    def __init__(self,width=96,blocks=3,dropout=.1,input_channels=4):
        super().__init__()
        self.input_channels=input_channels
        self.embed=nn.Linear(input_channels*15,width)
        self.band_embedding=nn.Parameter(torch.randn(1,25,width)*.02)
        layer=nn.TransformerEncoderLayer(width,4,width*3,dropout,batch_first=True,norm_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,blocks,enable_nested_tensor=False)
        self.mask_head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,15))
        self.presence_head=nn.Linear(width,1)
    def _hidden(self,x):
        n=x.shape[0];tokens=x.permute(0,3,1,2).reshape(n,25,self.input_channels*15)
        return self.encoder(self.embed(tokens)+self.band_embedding)
    def representation(self,x):return self._hidden(x).mean(1)
    def forward(self,x,energy_weights):
        h=self._hidden(x)
        cell=self.mask_head(h).transpose(1,2).sigmoid()
        return (cell*energy_weights).sum((1,2)),self.presence_head(h.mean(1)).squeeze(-1),cell

def make_model(kind,width=256,k=16,*,architecture_version=1,dropout=.1,model_width=None,input_channels=None,n_features=None,prior_clip=.01,prior_strength=.5):
    """Version 1 exactly preserves historical checkpoint shapes and defaults.

    New trainers save version 2/3 and explicit dropout/width in checkpoint
    ``architecture`` metadata. Dropout is an experiment, not an optimum claim.
    """
    if architecture_version not in [1,2,3,4,5]:raise ValueError('Unsupported architecture version')
    if not 0<=dropout<1:raise ValueError('dropout must be in [0,1)')
    if architecture_version==5:
        if kind!='cnn' or input_channels not in [None,8] or n_features is not None:
            raise ValueError('Architecture 5 requires the eight-channel background-feature CNN')
        return BackgroundPriorInputCNN(width=model_width or 32,dropout=dropout)
    if architecture_version==4:
        if kind!='cnn':raise ValueError('Architecture 4 is the CNN background-prior replacement only')
        if input_channels not in [None,8] or n_features is not None:raise ValueError('Hybrid architecture requires eight spatial channels')
        return BackgroundPriorResidualCNN(width=model_width or 32,dropout=dropout,prior_clip=prior_clip,prior_strength=prior_strength)
    channels=6 if architecture_version==3 else 4
    features=1258 if architecture_version==3 else 1254
    if input_channels is not None and input_channels!=channels:raise ValueError('Input channels differ from architecture version')
    if n_features is not None and n_features!=features:raise ValueError('Feature count differs from architecture version')
    if kind=='tabm':model=TabMRegressor(n_features=features,width=model_width or width,k=k,dropout=.1 if architecture_version==1 else dropout)
    effective_width=(model_width or {'cnn':32,'fdy':32,'itransformer':96}.get(kind,0)) if architecture_version in [2,3] else {'cnn':32,'fdy':32,'itransformer':96}.get(kind,0)
    if kind in ['cnn','fdy','itransformer'] and (effective_width<=0 or effective_width%4):
        raise ValueError('Convolution/attention width must be a positive multiple of four')
    effective_dropout=(.1 if kind=='itransformer' else 0.) if architecture_version==1 else dropout
    if kind=='cnn':model=EnergyMaskCNN(width=effective_width,dropout=effective_dropout,input_channels=channels)
    elif kind=='fdy':model=FDYMaskCRNN(width=effective_width,dropout=effective_dropout,input_channels=channels)
    elif kind=='itransformer':model=InvertedMaskTransformer(width=effective_width,dropout=effective_dropout,input_channels=channels)
    elif kind!='tabm':raise ValueError(kind)
    model.feature_version=3 if architecture_version==3 else 1
    model.preprocessing_version=model.feature_version
    return model
