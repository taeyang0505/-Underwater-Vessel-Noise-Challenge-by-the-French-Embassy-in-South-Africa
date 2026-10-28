"""R6-v2 pre-fit numeric fixtures only. No source datasets or model selection."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from pipeline.relative_grid_r6 import relative_grid, quantized_db
from pipeline.relative_grid_r6_v2 import FEATURE_NAMES, relative_grid_v2
from pipeline.relative_features_r6 import TinyMaskCNN
from pipeline.relative_models_r6_v2 import TinyMaskCNNv2, fixed_presence_score


V1_FILES = ('pipeline/relative_grid_r6.py', 'pipeline/relative_features_r6.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    start=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(73)
    original_v1={p:sha(p) for p in V1_FILES}
    probe=subprocess.run([sys.executable,'-c',
        'import sys,json,numpy as np;from pipeline.relative_grid_r6_v2 import relative_grid_v2;'
        'x,w=relative_grid_v2(np.full((1,15,25),100.));'
        'assert not any(k=="torch" or k.startswith("torch.") for k in sys.modules);'
        'print(json.dumps({"torch_loaded":False,"shape":list(x.shape)}))'],
        capture_output=True,text=True,check=True,timeout=20)
    rng=np.random.default_rng(73);db=rng.normal(100,5,(6,15,25))
    db[0,0]=np.nan;db[1,:,4]=np.nan;db[2,2:5]=np.nan
    original=db.copy();x,w=relative_grid_v2(db);x1,w1=relative_grid(db)
    assert x.shape==(6,8,15,25) and x.dtype==w.dtype==np.float32
    assert np.array_equal(x[:,:6],x1) and np.array_equal(w,w1)
    assert np.array_equal(x[:,7],np.sqrt(w))
    mask=np.isfinite(db)
    assert (x[:,6][~mask]==0).all() and (x[:,7][~mask]==0).all()
    assert np.array_equal(db,original,equal_nan=True)
    ramp=np.broadcast_to(np.arange(15)[None,:,None]+100.,(1,15,25)).copy()
    rx,rw=relative_grid_v2(ramp)
    assert abs(float(rx[0,6,0,0])-(-.7/15))<1e-7
    assert abs(float(rx[0,6,14,0])-(13.3/15))<1e-7
    changed,changed_w=relative_grid_v2(db+12.)
    assert np.array_equal(x,changed) and np.array_equal(w,changed_w)
    offset=np.linspace(-6,6,25)[None,None,:]
    ox,ow=relative_grid_v2(db+offset)
    assert np.allclose(ox[:,[0,1,2,3,6]],x[:,[0,1,2,3,6]],atol=2e-7)
    assert np.max(abs(ox[:,4]-x[:,4]))>.01 and np.max(abs(ox[:,7]-x[:,7]))>1e-3
    small_x,small_w=relative_grid_v2(db+.037)
    chunks=[relative_grid_v2(db[a:b]) for a,b in [(0,1),(1,4),(4,6)]]
    assert np.array_equal(np.concatenate([v[0] for v in chunks]),x)
    assert np.array_equal(np.concatenate([v[1] for v in chunks]),w)
    # Exact observed-only temporal-contrast score, independent hand oracle.
    cases=[]
    cw=torch.full((1,15,25),1/375.)
    cq=torch.full((1,15,25),.4)
    score,share,contrast=fixed_presence_score(cq,cw)
    assert torch.allclose(share,torch.tensor([.4]),atol=1e-7)
    assert contrast.item()==0 and abs(score.item()-.2)<1e-7
    cases.append({'case':'static_q','share':share.item(),'contrast':contrast.item(),'score':score.item()})
    cq.zero_();cq[0,7,10]=.8
    score,share,contrast=fixed_presence_score(cq,cw)
    assert abs(score.item()-(.8/375+.8)*.5)<1e-7
    cq.zero_();cq[0,7,23]=.8
    score,share,contrast=fixed_presence_score(cq,cw)
    assert contrast.item()==0  # index23 is excluded from the fixed 10..22 score
    sparse_w=torch.zeros((1,15,25));sparse_q=torch.zeros_like(sparse_w)
    sparse_w[0,2,10]=.5;sparse_w[0,3,10]=.5
    sparse_q[0,2,10]=.6;sparse_q[0,3,10]=.8
    score,share,contrast=fixed_presence_score(sparse_q,sparse_w)
    assert abs(share.item()-.7)<1e-7 and abs(contrast.item()-.2)<1e-7
    assert abs(score.item()-.45)<1e-7
    cases.append({'case':'missing_minutes_excluded_from_min','share':share.item(),
                  'contrast':contrast.item(),'score':score.item()})
    sparse_w.zero_();sparse_q.zero_();sparse_w[0,2,10]=1.;sparse_q[0,2,10]=.7
    score,share,contrast=fixed_presence_score(sparse_q,sparse_w)
    assert contrast.item()==0 and abs(score.item()-.35)<1e-7
    sparse_w.zero_();sparse_q.zero_();sparse_w[0,2,0]=1.;sparse_q[0,2,0]=.9
    score,share,contrast=fixed_presence_score(sparse_q,sparse_w)
    assert contrast.item()==0 and abs(score.item()-.45)<1e-7
    for seed in [0,1,2]:
        g=torch.Generator().manual_seed(seed);q=torch.rand((4,15,25),generator=g)
        ew=torch.rand((4,15,25),generator=g);ew[ew<.2]=0;ew/=ew.sum((1,2),keepdim=True)
        s,y,c=fixed_presence_score(q,ew)
        assert ((s>=0)&(s<=1)).all() and ((c>=0)&(c<=1)).all()
    # Unchanged local body without the optional context path.
    torch.manual_seed(101);old=TinyMaskCNN(input_channels=8).eval()
    torch.manual_seed(101);without=TinyMaskCNNv2(global_context=False).eval()
    assert all(torch.equal(v,without.state_dict()[k]) for k,v in old.state_dict().items()
               if k.startswith(('body.','mask_head.')))
    inp=torch.from_numpy(x);tw=torch.from_numpy(w)
    with torch.no_grad():
        old_outputs=old(inp,tw);no_outputs=without(inp,tw)
        assert torch.equal(old_outputs[0],no_outputs[0]) and torch.equal(old_outputs[2],no_outputs[2])
    # Isolate the NEW explicit context path from GroupNorm's existing global statistics.
    with_context=TinyMaskCNNv2(global_context=True).eval()
    hidden=torch.randn((1,32,15,25),requires_grad=True)
    ctx=with_context.add_global_context(hidden)
    ctx[:,:,:,0].square().mean().backward()
    high_gradient=float(hidden.grad[:,:,:,20:25].abs().max())
    assert high_gradient>1e-8
    assert with_context.context_projection.weight.grad.abs().max()>0
    assert with_context.context_gate.weight.grad.abs().max()>0
    for v in with_context.parameters():
        if v.grad is not None:assert torch.isfinite(v.grad).all()
    h2=hidden.detach().clone().requires_grad_(True)
    without.add_global_context(h2)[:,:,:,0].square().mean().backward()
    assert (h2.grad[:,:,:,20:25]==0).all()
    high_perturbed=hidden.detach().clone();high_perturbed[:,:,:,20:25]+=.3
    with torch.no_grad():
        low_change=float((with_context.add_global_context(high_perturbed)[:,:,:,0]-
                          with_context.add_global_context(hidden.detach())[:,:,:,0]).abs().max())
        assert low_change>1e-6
        assert torch.equal(without.add_global_context(high_perturbed)[:,:,:,0],
                           without.add_global_context(hidden.detach())[:,:,:,0])
    models=[]
    for context in [False,True]:
        torch.manual_seed(202);model=TinyMaskCNNv2(global_context=context)
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        losses=[]
        for _ in range(2):
            model.train();opt.zero_grad();pred,logit,q=model(inp,tw)
            target=torch.linspace(.05,.85,len(inp))
            loss=(pred-target).square().mean()+.05*torch.nn.functional.binary_cross_entropy_with_logits(logit,(target>.1).float())
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert model.mask_head.weight.grad.abs().max()>0
            opt.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            result=model(inp,tw);s,y,c=fixed_presence_score(result[2],tw)
            assert torch.equal(result[0],y)
            assert torch.allclose(result[1].sigmoid(),s.clamp(1e-6,1-1e-6),atol=1e-7)
            assert (result[2][tw==0]==0).all()
            small=[model(inp[i:i+1],tw[i:i+1]) for i in range(len(inp))]
            maxdiff=max(float((torch.cat([v[j] for v in small])-result[j]).abs().max()) for j in range(3))
            assert maxdiff<1e-5
        b=io.BytesIO();torch.save(model.state_dict(),b);b.seek(0)
        reloaded=TinyMaskCNNv2(global_context=context).eval()
        reloaded.load_state_dict(torch.load(b,weights_only=True,map_location='cpu'))
        with torch.no_grad():assert all(torch.equal(a,b) for a,b in zip(result,reloaded(inp,tw)))
        assert not any('presence_head' in n for n,_ in model.named_parameters())
        models.append({'global_context':context,'input_channels':8,'parameters':sum(p.numel() for p in model.parameters()),
                       'numeric_Adam_steps':2,'fixture_losses':losses,'reload_bitwise':True,
                       'singleton_vs_batch_max_abs':maxdiff,'free_presence_head':False})
    assert original_v1=={p:sha(p) for p in V1_FILES}
    return {'format':'r6_v2_relative_grid_context_fixed_score_fixture_v1','status':'passed','fixture_only':True,
            'formal_training_or_dataset_model_selection_performed':False,
            'TRAIN_INNER_OUTER_Test_or_other_source_data_opened':False,'GPU_SSH_external_submission_used':False,
            'code_sha256':{p:sha(p) for p in ['pipeline/relative_grid_r6_v2.py','pipeline/relative_models_r6_v2.py',__file__]},
            'v1_files_unchanged':original_v1,'fresh_pure_process_probe':json.loads(probe.stdout),
            'feature_names':list(FEATURE_NAMES),'first6_and_weights_bitwise_v1_equal':True,
            'q05_hand_quantile_check':True,'q05_is_not_verified_background':True,'db_input_unchanged':True,
            'common_grid_aligned_gain_bitwise':True,'nongrid_common_gain_max_feature_difference':float(abs(small_x-x).max()),
            'nongrid_common_gain_max_weight_difference':float(abs(small_w-w).max()),
            'static_band_offset_invariant_channels':[0,1,2,3,6],
            'static_band_offset_noninvariant_channels':[4,7],'mask_unchanged':True,
            'chunk_bitwise':True,'fixed_score_hand_oracles':cases,'score_safe_bands':'indices10..22 inclusive',
            'score_formula':'.5*sum(q*w)+.5*max_observed_safe(relu(q-min_observed_time_per_band(q)))',
            'BCE_logit_epsilon':1e-6,'score_does_not_guarantee_AUC_improvement':True,
            'explicit_context':{'band_slices':[[10,20],[20,25],[15,23]],
                'formula':'broadcast(GELU(project(concat_band_means))*sigmoid(gate(concat_band_means))))',
                'high_hidden_gradient_from_low_band_loss':high_gradient,
                'high_perturbation_changes_low_context_output':low_change,
                'global_context_false_path_high_gradient':0.,
                'isolated_path_test_excludes_GroupNorm_global_statistics':True},
            'models':models,'same_8channel_architecture_for_legacy_and_relative':True,
            'v1_body_and_mask_head_equal_when_context_off':True,
            'environment':{'numpy':np.__version__,'torch':torch.__version__,'CPU_threads':torch.get_num_threads()},
            'elapsed_seconds':time.perf_counter()-start}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/R6_V2_RELATIVE_FIXTURE.json')
    args=p.parse_args();path=Path(args.output)
    if path.exists() or path.with_suffix('.md').exists():raise FileExistsError('Fresh output required')
    result=run();path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    path.with_suffix('.md').write_text('# R6 v2 pre-fit numeric fixture\n\n'
        '**PASS — numeric fixture only.** No competition source data, model selection, GPU, SSH or external transmission. '
        'Two Adam updates on generated numbers check gradients only; no model checkpoint is saved.\n\n'
        'The [pure eight-channel grid](../pipeline/relative_grid_r6_v2.py) appends temporal q05-relative dB/15 and sqrt(w) '
        'to the unchanged six-channel v1 grid. q05 is not a background label. Features and weights use rounded 0.1-dB input; '
        'original precision labels are not changed. Arbitrary gains can cross quantization bins. Profile and energy channels '
        'are not invariant to per-band offsets.\n\n'
        'The [CNN](../pipeline/relative_models_r6_v2.py) uses the unchanged tiny local body plus optional per-minute '
        'frequency context. Means over bands 10:20, 20:25 and 15:23 are concatenated, projected and sigmoid-gated, '
        'then broadcast to all bands. An isolated hidden-tensor test verifies an explicit high-to-low gradient and perturbation '
        'path; it does not claim older GroupNorm had no frequency-global effects.\n\n'
        'There is no learned presence head. Fixed score is 0.5×pooled share + 0.5×maximum observed temporal q contrast '
        'over bands 10–22. The temporal minimum excludes missing cells. Its clamped logit drives the BCE auxiliary term. '
        'This predeclared formula is not evidence of improved AUC or calibration. Legacy8 and relative8 use the same '
        'architecture and score; context-off is a separate supported ablation.\n\n'
        'Tests cover pure-module Torch absence, old-grid bitwise equality, q05 arithmetic, mask/chunk/gain behavior, '
        'hand score cases, context-path gradients, finite CNN gradients, physical pooling, batch tolerance and bitwise reload. '
        'V1 source SHAs remained unchanged.\n\n'
        f'[Detailed JSON]({path.name})\n')
    print(json.dumps({'status':result['status'],'report':str(path),'report_sha256':sha(path),'models':result['models']}))


if __name__=='__main__':main()
