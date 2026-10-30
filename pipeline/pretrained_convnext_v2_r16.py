"""Version-2 ConvNeXt-V2 Base FCMAE adapter; V1 sources stay immutable.

The checkpoint is an ImageNet-1k FCMAE encoder, NOT an underwater separator.
All checkpoint tensors load strictly before the RGB stem is bypassed and the
three transition strides are changed.  Even-kernel asymmetric padding keeps
every stage at 15x25; changing only the first stem would not do that.
Factories never download weights or silently substitute random initialization.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / 'models/pretrained_r16/timm--convnextv2_base.fcmae'
SPEC = {
    'model_id': 'timm/convnextv2_base.fcmae',
    'revision': '4a8b02c2230370906f2b6ea42e18c553e50fbf57',
    'weight_bytes': 350809374,
    'weight_sha256': '4c2c0ab209f18d9dc1d4cf85e92dda8c834f377d27c5485c284e38daa9c060db',
    'files': {'config.json': '9fabc8a46c737330f47f77757713ee76661e395d0b14fcd924ed57694ccc408e',
              'README.md': '47b2a6d70bd219ac02054efade8c9100ea18ccde692c06c62be15574806640ba'},
    'weights_license': 'CC-BY-NC-4.0',
    'weights_license_url': 'https://huggingface.co/timm/convnextv2_base.fcmae',
    'paper_url': 'https://arxiv.org/abs/2301.00808',
    'official_source_url': 'https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(b)
    return digest.hexdigest()


def verify_checkpoint(cache_dir=DEFAULT_CACHE):
    directory = Path(cache_dir)
    for name, expected in {**SPEC['files'], 'model.safetensors': SPEC['weight_sha256']}.items():
        p = directory / name
        if not p.is_file() or p.is_symlink() or sha256(p) != expected:
            raise RuntimeError('BLOCKED: missing/changed public ConvNeXt checkpoint: ' + str(p))
    if (directory / 'model.safetensors').stat().st_size != SPEC['weight_bytes']:
        raise RuntimeError('BLOCKED: public checkpoint byte-size differs')
    return {**SPEC, 'verified': True, 'directory': str(directory)}


def _safe_f32_state(path):
    """Read only pinned F32 safetensors; no pickle or downloaded Python code.

    A deliberately limited loader avoids installing/replacing a V1 dependency.
    Bounds, exact tensor bytes, dtype, and overlapping storage are checked.
    """
    with Path(path).open('rb') as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise RuntimeError('Truncated safetensors prefix')
        length = struct.unpack('<Q', length_bytes)[0]
        if length > 1024**2:
            raise RuntimeError('Unexpected safetensors header length')
        entries = json.loads(stream.read(length))
        base, size = 8 + length, Path(path).stat().st_size
        state, ranges = {}, []
        for name, meta in entries.items():
            if name == '__metadata__':
                continue
            if meta['dtype'] != 'F32' or any(type(v) is not int or v < 0 for v in meta['shape']):
                raise RuntimeError('Only the pinned F32 checkpoint is supported')
            start, end = meta['data_offsets']
            count = 1
            for dim in meta['shape']:
                count *= dim
            if not (0 <= start <= end <= size-base) or end-start != count*4:
                raise RuntimeError('Invalid tensor offsets: ' + name)
            ranges.append((start, end))
            stream.seek(base+start)
            raw = bytearray(stream.read(end-start))
            if len(raw) != end-start:
                raise RuntimeError('Truncated tensor: ' + name)
            state[name] = torch.frombuffer(raw, dtype=torch.float32).clone().reshape(meta['shape'])
        ordered = sorted(ranges)
        if any(a[1] > b[0] for a,b in zip(ordered, ordered[1:])):
            raise RuntimeError('Overlapping safetensors tensor offsets')
    return state


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(x.permute(0,2,3,1), self.normalized_shape,
                            self.weight, self.bias, self.eps).permute(0,3,1,2)


class GlobalResponseNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        magnitude = x.norm(p=2, dim=(1,2), keepdim=True)
        response = magnitude / (magnitude.mean(dim=-1, keepdim=True) + 1e-6)
        return x + self.weight * (x * response) + self.bias


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, 4*width)
        self.grn = GlobalResponseNorm(4*width)
        self.fc2 = nn.Linear(4*width, width)

    def forward(self, x):
        return self.fc2(self.grn(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv_dw = nn.Conv2d(width,width,7,padding=3,groups=width)
        self.norm = nn.LayerNorm(width,eps=1e-6)
        self.mlp = MLP(width)

    def forward(self,x):
        h = self.norm(self.conv_dw(x).permute(0,2,3,1))
        return x + self.mlp(h).permute(0,3,1,2)


class Stage(nn.Module):
    def __init__(self, incoming, width, depth):
        super().__init__()
        self.downsample = (nn.Identity() if incoming==width else nn.Sequential(
            LayerNorm2d(incoming,eps=1e-6), nn.Conv2d(incoming,width,2,stride=2)))
        self.blocks = nn.Sequential(*(Block(width) for _ in range(depth)))


class CheckpointEncoder(nn.Module):
    """State names and shapes match timm/convnextv2_base.fcmae exactly."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3,128,4,stride=4),LayerNorm2d(128,eps=1e-6))
        dims, depths = (128,256,512,1024),(3,3,27,3)
        self.stages = nn.ModuleList(Stage(dims[max(0,i-1)],dim,depth)
                                   for i,(dim,depth) in enumerate(zip(dims,depths)))
        self.head = nn.Module()
        self.head.norm = nn.LayerNorm(1024,eps=1e-6)


class ResidualAdapter(nn.Module):
    def __init__(self,width,rank,dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(width,rank,1),nn.GELU(),nn.Dropout(dropout),nn.Conv2d(rank,width,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        return x+self.net(x)


class ConvNeXtV2TaskBranch(nn.Module):
    def __init__(self, backbone, task, rank=16, dropout=.1, gradient_checkpointing=False):
        super().__init__()
        if task not in ('regression','classification') or rank not in (8,16):
            raise ValueError('Expected separate regression/classification branch and rank8/16')
        self.task, self.family, self.mask_channel = task,'convnextv2_base_fcmae',5
        self.in_channels = 17 if task=='regression' else 15
        self.backbone = backbone.requires_grad_(False)
        for stage in backbone.stages[1:]:
            stage.downsample[1].stride = (1,1)
        self.stem = nn.Conv2d(self.in_channels,128,3,padding=1,stride=1)
        self.stem_norm = LayerNorm2d(128,eps=1e-6)
        self.adapters = nn.ModuleList(ResidualAdapter(c,rank,dropout) for c in (128,256,512,1024))
        self.head = nn.Conv2d(1024,1,1)
        self.pool_attention = nn.Conv2d(1024,1,1) if task=='classification' else None
        self.gradient_checkpointing = gradient_checkpointing
        self._adapter_names = tuple(n for n,_ in self.named_parameters()
            if not n.startswith('backbone.') or n.startswith('backbone.head.norm.')
            or (n.startswith('backbone.stages.3.') and '.norm.' in n))
        self.set_stage(1)

    def set_stage(self,stage):
        stage = {'warmup':1,'peft':2}.get(stage,stage)
        if stage not in (1,2):raise ValueError('Expected stage1/2')
        for n,p in self.named_parameters():
            p.requires_grad_(n in self._adapter_names and (stage==2 or (not n.startswith(('backbone.','adapters.')))))
        self.stage=stage
        self.train(self.training)
        return self

    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'backbone'):self.backbone.eval()
        if hasattr(self,'adapters'):self.adapters.train(mode and getattr(self,'stage',1)==2)
        return self

    def forward(self,x):
        if x.ndim!=4 or tuple(x.shape[1:])!=(self.in_channels,15,25):
            raise ValueError(f'Expected [N,{self.in_channels},15,25]')
        valid=x[:,self.mask_channel]>.5
        if not valid.any((1,2)).all():raise ValueError('Every row needs observed cells')
        h=self.stem_norm(self.stem(x))
        for index,stage in enumerate(self.backbone.stages):
            if index:
                # Preserve original 2x2 weights; explicit right/bottom SAME pad.
                h=stage.downsample(F.pad(h,(0,1,0,1)))
            for block in stage.blocks:
                h=checkpoint(block,h,use_reentrant=False) if self.gradient_checkpointing and self.training else block(h)
            h=self.adapters[index](h)
            if tuple(h.shape[-2:])!=(15,25):raise RuntimeError('Spatial information was downsampled')
        h=self.backbone.head.norm(h.permute(0,2,3,1)).permute(0,3,1,2)
        logits=self.head(h).squeeze(1)
        if self.task=='regression':return logits
        a=self.pool_attention(h).squeeze(1).masked_fill(~valid,torch.finfo(h.dtype).min)
        return (a.flatten(1).softmax(1)*logits.flatten(1)).sum(1)

    def frozen_named_parameters(self):
        return ((n,p) for n,p in self.named_parameters() if n not in self._adapter_names)

    def adapter_state(self):
        state=self.state_dict()
        return {n:state[n].detach().cpu().clone() for n in self._adapter_names}

    def load_adapter_state(self,state):
        if set(state)!=set(self._adapter_names):raise RuntimeError('Adapter keys differ')
        current=self.state_dict()
        with torch.no_grad():
            for n,v in state.items():
                if v.shape!=current[n].shape or not torch.isfinite(v).all():raise RuntimeError('Invalid '+n)
                current[n].copy_(v.to(current[n]))
        return self


def create_convnext_v2_branch(task,rank=16,dropout=.1,device='cpu',dtype='float32',
                             cache_dir=DEFAULT_CACHE,gradient_checkpointing=False):
    receipt=verify_checkpoint(cache_dir)
    with torch.device('meta'):base=CheckpointEncoder()
    state=_safe_f32_state(Path(cache_dir)/'model.safetensors')
    base.load_state_dict(state,strict=True,assign=True)
    del state
    base=base.to(dtype=getattr(torch,dtype))
    model=ConvNeXtV2TaskBranch(base,task,rank,dropout,gradient_checkpointing).to(device)
    receipt.update({'strict_pretrained_load':True,'checkpoint_parameter_elements':sum(p.numel() for p in base.parameters()),
       'total_parameters':sum(p.numel() for p in model.parameters()),'stage1_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
       'task':task,'input_channels':model.in_channels,'output_grid':[15,25],'downsampling_strides':[1,1,1,1],
       'pretrained_frontend_used':False,'bypassed_checkpoint_tensors':['stem.0.weight','stem.0.bias','stem.1.weight','stem.1.bias'],
       'changes':['new 3x3 stride1 input stem','all three pretrained 2x2 transitions stride1 with asymmetric SAME padding',
                  'new rank16 residual adapters and task head','top-stage LayerNorm affine + final norm stage2 trainable'],
       'regression_classifier_parameter_sharing':False,'task_transfer_performance_proven':False})
    model.set_stage(2);receipt['stage2_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.set_stage(1)
    return model,receipt


def create_r16_v2_branch(family,task,device='cpu',dtype='float32',dropout=.1,
                         gradient_checkpointing=False):
    """Explicit V2 registry; never mutates the V1 global factory or run plan."""
    if family=='convnextv2_base_fcmae':
        return create_convnext_v2_branch(task,rank=16,dropout=dropout,device=device,
            dtype=dtype,gradient_checkpointing=gradient_checkpointing)
    if family=='dasheng_base':
        from pipeline.pretrained_audio2d_r16 import create_audio2d_branch
        return create_audio2d_branch(family,task,rank=8,dropout=dropout,device=device,
            dtype=dtype,gradient_checkpointing=gradient_checkpointing)
    if family=='timemoe':
        from pipeline.pretrained_timemoe_r16 import create_timemoe_branch
        return create_timemoe_branch(task,rank=8,dropout=dropout,device=device,dtype=dtype)
    raise ValueError('V2 family outside frozen registry: '+str(family))
