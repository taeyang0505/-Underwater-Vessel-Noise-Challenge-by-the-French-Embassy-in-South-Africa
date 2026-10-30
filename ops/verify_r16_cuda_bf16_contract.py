"""CPU-only fixtures for the new CUDA runner. Never loads pretrained weights.

This is a structural/numerical fixture, not a CUDA/MPS performance or full-data
validation result. A passed fixture does not create a production RUN_PLAN.
"""
from pathlib import Path
import argparse, ast, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(output):
    import torch
    from torch import nn
    from pipeline.pretrained_audio2d_r16 import AttentivePool
    from pipeline.r16_cuda_precision import FP32AttentivePool, install_fp32_classifier_pool
    from pipeline.r16_feature_masking import FeatureMaskingConfig, apply_r16_feature_masking, make_feature_masking_generator
    from pipeline.training_r16_pretrained_cuda_bf16 import loss_and_values, NUMERICAL_POLICY, MICROBATCH, source_pins, seed_training_rngs, SEED
    from pipeline.r16_multisupport_data import write
    import numpy as np
    import random
    torch.set_num_threads(2)
    seed_policy=seed_training_rngs()
    first=(torch.rand(4),np.random.rand(4),[random.random() for _ in range(4)])
    assert SEED>2**32-1 and seed_policy['numpy_legacy_seed_u32']==SEED%(2**32)
    assert seed_training_rngs()==seed_policy
    second=(torch.rand(4),np.random.rand(4),[random.random() for _ in range(4)])
    assert torch.equal(first[0],second[0]) and np.array_equal(first[1],second[1]) and first[2]==second[2]
    assert np.array_equal(np.random.default_rng([SEED,1]).permutation(32),np.random.default_rng([SEED,1]).permutation(32))
    torch.manual_seed(1337)
    class Toy(nn.Module):
        def __init__(self):
            super().__init__();self.width=12;self.head=AttentivePool(12,.1);self.stage=2
        def set_stage(self,stage):self.stage=stage;return self
    model=Toy(); original={key:value.clone() for key,value in model.head.state_dict().items()}
    rng=torch.get_rng_state().clone();model,policy=install_fp32_classifier_pool(model,'dinov2_small','classification')
    assert torch.equal(rng,torch.get_rng_state()) and type(model.head) is FP32AttentivePool
    assert all(torch.equal(value,model.head.state_dict()[key]) for key,value in original.items())
    tokens=torch.randn(4,375,12,requires_grad=True); valid=torch.ones(4,375,dtype=torch.bool);valid[:,::4]=False
    with torch.autocast('cpu',dtype=torch.bfloat16):logits=model.head(tokens,valid)
    assert logits.dtype==torch.float32 and torch.isfinite(logits).all()
    torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.,0.,1.])).backward()
    assert torch.isfinite(tokens.grad).all() and tokens.grad.abs().sum()>0
    x=torch.randn(4,17,15,25);x[:,5]=1
    w=torch.full((4,15,25),1/375);q=torch.rand_like(w);y=(w*q).sum((1,2));z=(y>0).float()
    original_values=[v.clone() for v in (x,w,q,y,z)]
    masked,metadata=apply_r16_feature_masking(x,task='regression',training=True,
        generator=make_feature_masking_generator(42),config=FeatureMaskingConfig.for_variant('both'))
    assert all(torch.equal(a,b) for a,b in zip((x,w,q,y,z),original_values))
    assert torch.equal(masked[:,5],x[:,5])
    mask_logits=torch.randn(4,15,25,dtype=torch.bfloat16,requires_grad=True)
    loss,pred,cell=loss_and_values(mask_logits,'regression',w,y,q,z)
    assert loss.dtype==pred.dtype==torch.float32 and torch.isfinite(loss)
    expected=(pred-y).square().mean()+.25*((mask_logits.float().sigmoid()-q).square()*w).sum((1,2)).mean()
    assert torch.allclose(loss,expected)
    loss.backward(); assert torch.isfinite(mask_logits.grad).all()
    pins=source_pins()
    for name in ('pipeline/training_r16_pretrained_cuda_bf16.py','pipeline/r16_cuda_precision.py',
                 'ops/run_r16_pretrained_cuda_bf16.py','ops/finalize_r16_pretrained_cuda_bf16.py'):
        assert name in pins;ast.parse((ROOT/name).read_text())
    report={'status':'passed','CPU_fixture_only':True,'CUDA_or_MPS_executed':False,
       'pretrained_weights_loaded':False,'Test_or_locked_panels_opened':False,
       'classifier_pool_state_values_keys_and_initialization_RNG_preserved':True,
       'FP32_classifier_pool_inside_CPU_BF16_autocast_passed':True,
       'physical_loss_FP32_and_targets_weights_observation_mask_preserved':True,
       'nonzero_finite_gradient_through_FP32_pool':True,'numerical_policy':NUMERICAL_POLICY,
       'uint32_legacy_numpy_seed_compatibility_and_rng_replay_passed':True,'seed_policy':seed_policy,
       'microbatch_policy':MICROBATCH,'source_sha256':pins,'production_plan_created':False,
       'remaining':['Real GPU preflight for all three families, both tasks, both stages',
                    'Completed Mac off/both control before production plan freeze',
                    'First full production epoch speed/memory/convergence observation']}
    path=Path(output)
    if path.exists():raise FileExistsError('Fresh fixture receipt required')
    write(path,report);print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    main(p.parse_args().output)
