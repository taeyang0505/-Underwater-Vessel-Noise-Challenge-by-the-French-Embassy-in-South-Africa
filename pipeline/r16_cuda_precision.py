"""Explicit numerical policy for the new R16 CUDA BF16 run.

The old frozen training/model sources are never edited. Permanent parameters
stay FP32. Only supported forward operations use BF16 autocast; physical
pooling, labels, regression/BCE losses and the Audio2D classifier pool stay
FP32. This file does not access inputs, checkpoints, the network or devices at
import time.
"""
from contextlib import nullcontext
import torch
from pipeline.pretrained_audio2d_r16 import AttentivePool


class FP32AttentivePool(AttentivePool):
    """Same parameters/state keys, with explicit FP32 pooling execution."""
    def forward(self, h, valid):
        with torch.autocast(device_type=h.device.type, enabled=False):
            return super().forward(h.float(), valid)


def install_fp32_classifier_pool(model, family, task):
    policy = {'applied': False, 'parameter_values_or_state_keys_changed': False,
              'implementation': 'FP32AttentivePool', 'regression_pooling': 'FP32 outside autocast'}
    if family in ('dasheng_base', 'dinov2_small') and task == 'classification':
        if type(model.head) is not AttentivePool:
            raise TypeError('Exact original Audio2D attentive classifier head required')
        original = model.head
        parameter = next(original.parameters())
        # Keep initialization RNG unchanged: the new head immediately receives
        # the original values, so random temporary values must not alter later
        # dropout/masking streams.
        with torch.random.fork_rng(devices=[]):
            replacement = FP32AttentivePool(model.width, original.output[1].p)
        replacement = replacement.to(device=parameter.device, dtype=parameter.dtype)
        replacement.load_state_dict(original.state_dict(), strict=True)
        if list(replacement.state_dict()) != list(original.state_dict()):
            raise AssertionError('Classifier head state keys changed')
        for key, value in original.state_dict().items():
            if not torch.equal(value, replacement.state_dict()[key]):
                raise AssertionError('Classifier parameter values changed')
        model.head = replacement
        # _adapter_names is name based and remains exact. Restage trainability
        # after the parameter object replacement.
        model.set_stage(model.stage)
        policy['applied'] = True
    return model, policy


def autocast_context(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()


def require_cuda_bf16():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Actual BF16 CUDA device required; no CPU/MPS fallback')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    return {'device': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
            'torch': torch.__version__, 'CUDA': torch.version.cuda,
            'arch_list': torch.cuda.get_arch_list(), 'BF16': True}
