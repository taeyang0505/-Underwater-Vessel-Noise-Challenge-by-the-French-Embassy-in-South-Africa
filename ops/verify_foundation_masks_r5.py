"""CPU protocol fixtures, never vendor/B-sized models or a pretrained smoke run."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from pipeline.foundation_masks_r5 import (
    TimeMoEUltraMask, Dasheng12BMask, LoRALinear, FoundationAvailabilityError,
    PretrainedEvidence, OFFICIAL_SOURCES,
)

ROOT = Path(__file__).resolve().parents[1]


class TinyTimeAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q_proj = nn.Linear(width, width); self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width); self.o_proj = nn.Linear(width, width, bias=False)
    def forward(self, x):
        def heads(layer): return layer(x).reshape(len(x), 15, 4, -1).transpose(1, 2)
        h = F.scaled_dot_product_attention(heads(self.q_proj), heads(self.k_proj), heads(self.v_proj), is_causal=True)
        return self.o_proj(h.transpose(1, 2).reshape(len(x), 15, -1))


class TinyExperts(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate = nn.Linear(width, 4, bias=False)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(width, width*2), nn.SiLU(), nn.Linear(width*2, width)) for _ in range(4)])
        self.shared_expert = nn.Linear(width, width)
    def forward(self, x):
        # Protocol fixture: differentiable gate/expert path, not a replica of
        # the vendor sparse top-k dispatch or its exact numerical implementation.
        w = self.gate(x).softmax(-1)
        return (torch.stack([m(x) for m in self.experts], -2)*w[..., None]).sum(-2)+self.shared_expert(x)


class TinyTimeLayer(nn.Module):
    def __init__(self, width):
        super().__init__(); self.self_attn = TinyTimeAttention(width); self.ffn_layer = TinyExperts(width)
        self.input_layernorm = nn.LayerNorm(width); self.post_attention_layernorm = nn.LayerNorm(width)
    def forward(self, x):
        x = x+self.self_attn(self.input_layernorm(x))
        return x+self.ffn_layer(self.post_attention_layernorm(x))


class TinyTimeMoeProtocol(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width, num_hidden_layers=2, num_experts=4, num_experts_per_tok=2)
        self.embed_layer = nn.Linear(1, width); self.layers = nn.ModuleList([TinyTimeLayer(width) for _ in range(2)])
        self.norm = nn.LayerNorm(width); self.calls = 0
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, use_cache=None, return_dict=None):
        assert input_ids is None and attention_mask is None and use_cache is False and return_dict is True
        assert inputs_embeds.shape[1:] == (15, self.config.hidden_size)
        self.calls += 1; h = inputs_embeds
        for layer in self.layers: h = layer(h)
        return SimpleNamespace(last_hidden_state=self.norm(h))


class TinyDashengAttention(nn.Module):
    def __init__(self, width):
        super().__init__(); self.num_heads = 4
        self.qkv = nn.Linear(width, 3*width); self.proj = nn.Linear(width, width)
    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, 4, d//4).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(b, n, d))


class TinyDashengBlock(nn.Module):
    def __init__(self, width):
        super().__init__(); self.attn = TinyDashengAttention(width)
        self.norm1 = nn.LayerNorm(width); self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width*4), nn.GELU(), nn.Linear(width*4, width))
    def forward(self, x):
        assert x.shape[1] == 375
        x = x+self.attn(self.norm1(x))
        return x+self.mlp(self.norm2(x))


class TinyDashengProtocol(nn.Module):
    def __init__(self, width=32):
        super().__init__(); self.embed_dim = width
        self.blocks = nn.Sequential(*[TinyDashengBlock(width) for _ in range(2)])
        self.norm = nn.LayerNorm(width)
        self.patch_embed = nn.Conv2d(1, width, (64, 4), stride=(64, 4))
        self.time_pos_embed = nn.Parameter(torch.randn(1, width, 1, 252))
        self.freq_pos_embed = nn.Parameter(torch.randn(1, width, 1, 1))
    def forward(self, _):
        raise AssertionError('Original waveform/mel frontend must not run in the grid wrapper')


def tensor_sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def frozen_snapshot(model):
    return {name: tensor_sha(p) for name, p in model.named_parameters() if not p.requires_grad}


def one_case(kind, router=False):
    torch.manual_seed(1972)
    original = TinyTimeMoeProtocol() if kind == 'timemoe' else TinyDashengProtocol()
    pristine = deepcopy(original)
    constructor = TimeMoEUltraMask if kind == 'timemoe' else Dasheng12BMask
    extra = {'adapt_router': router} if kind == 'timemoe' else {}
    model = constructor(original, fixture_mode=True, **extra)
    before = frozen_snapshot(model)
    x = torch.randn(2, 8, 15, 25, requires_grad=True)
    weights = torch.rand(2, 15, 25); weights[:, 1:3] = 0; weights[:, :, 0] = 0
    weights = (weights/weights.sum((1, 2), keepdim=True)).requires_grad_(True)
    w_before = weights.detach().clone()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.002)
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        share, presence, cell = model(x, weights)
        assert share.shape == presence.shape == (2,) and cell.shape == (2, 15, 25)
        assert torch.isfinite(share).all() and torch.isfinite(presence).all() and torch.isfinite(cell).all()
        assert torch.all((cell >= 0) & (cell <= 1)) and (cell[weights == 0] == 0).all()
        torch.testing.assert_close(share, (cell.float()*weights.detach().float()).sum((1, 2)), rtol=0, atol=0)
        loss = ((share-torch.tensor([.2, .8]))**2).mean()+.1*F.binary_cross_entropy_with_logits(presence, torch.tensor([0., 1.]))+.01*(cell-.3).square().mean()
        loss.backward()
        assert weights.grad is None and torch.equal(weights.detach(), w_before)
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
            else: assert p.grad is None, name
        optimizer.step(); losses.append(float(loss.detach()))
    assert frozen_snapshot(model) == before
    allowed_trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    for name in allowed_trainable:
        assert not name.startswith('backbone.') or name.endswith(('lora_A', 'lora_B')), name
    assert any(n.startswith('stem.') for n in allowed_trainable)
    assert model.time_position.grad is not None and model.time_position.grad.abs().sum() > 0
    assert any(p.abs().sum() > 0 for n, p in model.named_parameters() if n.endswith('lora_B'))
    model.eval()
    with torch.no_grad(): expected = model(x, weights)
    checkpoint = model.adapter_checkpoint()
    assert not any(name in before for name in checkpoint['state'])
    stream = io.BytesIO(); torch.save(checkpoint, stream); stream.seek(0)
    loaded = torch.load(stream, weights_only=True, map_location='cpu')
    restored = constructor(pristine, fixture_mode=True, **extra).load_adapter_checkpoint(loaded).eval()
    with torch.no_grad(): actual = restored(x, weights)
    for a, b in zip(actual, expected): torch.testing.assert_close(a, b, rtol=0, atol=0)
    mismatch = deepcopy(loaded); mismatch['manifest']['width'] += 1
    try: restored.load_adapter_checkpoint(mismatch)
    except ValueError: pass
    else: raise AssertionError('Mismatched adapter identity accepted')
    missing = deepcopy(loaded); missing['state'].pop(next(iter(missing['state'])))
    try: restored.load_adapter_checkpoint(missing)
    except ValueError: pass
    else: raise AssertionError('Missing adapter state accepted')
    # Physical limiting cases preserve the provided energy sum exactly.
    with torch.no_grad():
        restored.mask_head.weight.zero_(); restored.mask_head.bias.fill_(40)
        one, _, _ = restored(x, weights)
        restored.mask_head.bias.fill_(-40); zero, _, _ = restored(x, weights)
    torch.testing.assert_close(one, torch.ones(2), rtol=0, atol=2e-6)
    assert (zero < 1e-15).all()
    rejected = 0
    for xx, ww in [(x[:, :7], weights), (x, weights*2), (x, torch.zeros_like(weights)), (x.detach().clone().fill_(float('nan')), weights)]:
        try: restored(xx, ww)
        except ValueError: rejected += 1
        else: raise AssertionError('Invalid grid/energy weights accepted')
    return {'kind': kind, 'router_lora': router, 'passed': True, 'counts': model.trainable_parameter_counts(),
            'manifest': model.adapter_manifest(), 'finite_fixture_loss_values_not_performance': losses,
            'frozen_parameters_bytes_unchanged': True, 'frozen_experts_no_grad': True,
            'only_LoRA_inside_backbone_trainable': True, 'original_energy_weights_no_grad_unchanged': True,
            'original_missing_cells_zero_mask': True, 'adapter_bytes': len(stream.getvalue()),
            'adapter_excludes_frozen_vendor_weights': True, 'adapter_roundtrip_bitwise': True,
            'adapter_identity_missing_tensor_rejected': True, 'invalid_inputs_rejected': rejected,
            'physical_zero_one_limits_passed': True}


def main():
    torch.set_num_threads(2)
    gate = []
    for constructor, backbone in [(TimeMoEUltraMask, TinyTimeMoeProtocol()), (Dasheng12BMask, TinyDashengProtocol())]:
        try: constructor(backbone)
        except FoundationAvailabilityError as error: gate.append(str(error))
        else: raise AssertionError('Unverified/default pretrained construction allowed')
    try:
        Dasheng12BMask(TinyDashengProtocol(), evidence=PretrainedEvidence('dasheng_1.2b', OFFICIAL_SOURCES['dasheng_weight_url'], 'fake_fixture', '0'*64, True))
    except ValueError: pass  # Reject wrong small shape even with purported evidence.
    else: raise AssertionError('Small substitute advertised as 1.2B')
    torch.manual_seed(23); linear = nn.Linear(12, 16); x = torch.randn(2, 3, 12)
    expected = linear(x); lora = LoRALinear(linear, 8)
    torch.testing.assert_close(lora(x), expected, rtol=0, atol=0)
    cases = [one_case('timemoe'), one_case('timemoe', router=True), one_case('dasheng')]
    code = ROOT/'pipeline/foundation_masks_r5.py'
    report = {'passed': True, 'scope': 'CPU tiny protocol fixtures only; NOT actual pretrained inference or validation of real Ultra/Dasheng weights.',
              'real_B_sized_models_instantiated': False, 'pretrained_weights_downloaded': False,
              'GPU_used': False, 'Test_or_real_acoustic_arrays_read': False,
              'source_code_sha256': hashlib.sha256(code.read_bytes()).hexdigest(),
              'fixture_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'availability_default_rejections': gate, 'small_Dasheng_substitution_rejected': True,
              'LoRA_zero_initialization_bitwise_base_parity': True, 'cases': cases,
              'official_sources': OFFICIAL_SOURCES,
              'remaining_production_gates': ['Ultra exact pretrained checkpoint remains unavailable/unverified',
                  'Real Dasheng artifact SHA/revision and audited code loading must be supplied by caller',
                  'Real backbone numeric/memory/adapter reload smoke and source-held-out regression validation not run']}
    out = ROOT/'reports/R5_FOUNDATION_ADAPTER_FIXTURE.json'
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'passed': True, 'report': str(out), 'code_sha256': report['source_code_sha256'],
                      'case_counts': [c['counts'] for c in cases], 'production_models_ready': False}, indent=2))


if __name__ == '__main__': main()
