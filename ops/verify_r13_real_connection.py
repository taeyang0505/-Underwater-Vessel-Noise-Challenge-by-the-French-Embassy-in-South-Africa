"""Default-size four neural models, actual R13 TRAIN, one Adam step each."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,sys,tempfile,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from pipeline.full_data_r13 import R13Sources,DynamicR13Dataset,validate_full_bundle,sha
from pipeline.models_r13 import make_r13_model
from pipeline.training_r13 import physical_multitask_loss

def main(output='reports/R13_REAL_FULL_MODEL_CONNECTION_V3.json'):
    out=ROOT/output
    if out.exists():raise FileExistsError('Fresh verification receipt required')
    torch.set_num_threads(1);sources=R13Sources(bundle=ROOT/'data/synthetic/group_cv_physics_r13_variants_v2');data=DynamicR13Dataset(sources,0)
    sch=data.schedule;order=sch['primary_order'];counts=data.metadata.qc_observed_minutes.to_numpy()[order]
    conditions=[sch['zero'],(~sch['zero'])&(~sch['highshare'])&(counts==15),
        sch['highshare']&np.char.startswith(sch['family'],'long_'),sch['highshare']&~np.char.startswith(sch['family'],'long_')]
    indices=[int(np.flatnonzero(k)[0]) for k in conditions];rows=[data[i] for i in indices]
    x=torch.from_numpy(np.stack([r['x'] for r in rows]));w=torch.from_numpy(np.stack([r['w'] for r in rows]))
    y=torch.tensor([r['share'] for r in rows]);z=torch.tensor([r['presence'] for r in rows],dtype=torch.float32);q=torch.from_numpy(np.stack([r['cell_share'] for r in rows]))
    if not x[:,8:].abs().sum()>0:raise AssertionError('Real P1 channels missing')
    temp=Path(tempfile.mkdtemp(prefix='r13-real-one-step-',dir='/private/tmp'));reports=[]
    codes={n:sha(ROOT/n) for n in ['pipeline/models_r13.py','pipeline/training_r13.py','pipeline/full_data_r13.py']}
    for kind in ('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer'):
        torch.manual_seed(2026102713);model=make_r13_model(kind);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
        model.train();p,s,m=model(x,w);loss,parts=physical_multitask_loss((p,s,m),(y,z,q),w)
        if not torch.equal(loss,parts['share_mse']+.25*parts['weighted_cell_mse']):raise AssertionError('Unapproved loss path')
        opt.zero_grad();loss.mean().backward()
        if not all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None):raise AssertionError('Nonfinite gradients')
        if kind=='bg_cnn':extra=model.backbone.body[0].weight.grad[:,8:]
        elif kind=='tfgrid_fdy_conformer':extra=model.backbone.grid.stem[0].weight.grad[:,8:]
        elif kind=='fdy':extra=model.backbone.stem.weight.grad[:,8:]
        else:extra=model.backbone.embed.weight.grad[:,120:]
        extra_grad=float(extra.abs().sum());assert extra_grad>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();model.eval()
        with torch.inference_mode():before=model(x,w)
        path=temp/(kind+'.pt');torch.save({'state_dict':model.state_dict(),'architecture':model.architecture(),'fixture_only':True},path)
        saved=torch.load(path,map_location='cpu',weights_only=True);new=make_r13_model(kind);new.load_state_dict(saved['state_dict'],strict=True);new.eval()
        with torch.inference_mode():after=new(x,w)
        assert all(torch.equal(a,b) for a,b in zip(before,after))
        reports.append({'kind':kind,'parameters':sum(t.numel() for t in model.parameters()),'actual_Adam_steps':1,
            'real_P1_stem_gradient_L1':extra_grad,'strict_reload_bitwise':True,'physical_share_score_identical':torch.equal(before[0],before[1]),
            'physical_pool_max_error':float(abs((before[2]*w).sum((1,2))-before[0]).max()),'no_presence_head':not any('presence_head' in n for n,_ in model.named_parameters())})
        del model,new,opt
    sources.assert_unchanged()
    for n,h in codes.items():assert sha(ROOT/n)==h
    value={'format':'r13_real_fullsize_one_step_connection_v1','status':'passed','fixture_only':True,
        'data_hash':validate_full_bundle(sources.bundle,verify_all=False)['data_hash'],'actual_TRAIN_indices':indices,
        'scenarios':['knownA0','ordinary dense positive','highshare long','highshare standard'],
        'model_checks':reports,'code_sha256':codes,'fixture_weights_never_production':str(temp),
        'CUDA_SSH_Test_new_validation_gate_report_or_full100training':False,'original_sources_unchanged':True}
    out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'passed','receipt':str(out),'models':reports}))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/R13_REAL_FULL_MODEL_CONNECTION_V3.json')
    main(p.parse_args().output)
