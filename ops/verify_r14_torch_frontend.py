"""CPU-only R14 frontend parity, raw-loader and checkpoint diagnostics."""
from pathlib import Path
import json,sys,time,tempfile,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from pipeline.features_r12 import load_spectral_basis,transform_channels
from pipeline.features_r14_torch import TorchFeatureFrontend,transform
from pipeline.raw_frontend_r14 import RawR14Dataset,RawFeatureMaskModel

OUT=ROOT/'reports/R14_TORCH_FRONTEND'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    if OUT.exists():raise FileExistsError('Fresh diagnostic output required')
    OUT.mkdir(parents=True);torch.set_num_threads(2)
    protocol={'version':'r14_torch_frontend_CPU_v1','threads':2,'feature_atol':3e-6,'weight_atol':3e-8,
        'relative_tolerance':0.,'model_prediction_atol':3e-6,'actual_TRAIN_benchmark_rows':4096,
        'benchmark_repetitions':3,'no_GPU_Test_network_model_performance_selection':True,
        'frozen_R13_code_data_policy_must_not_change':True}
    write(OUT/'protocol.json',protocol)
    bundle=ROOT/'data/synthetic/group_cv_physics_r13_variants_v2';m=json.loads((bundle/'manifest.json').read_text());f=m['folds']['0']
    basis_path=bundle/f['basis_path'];basis=load_spectral_basis(basis_path,expected_sha256=f['basis_sha256'],expected_training_binding=f['basis_training_binding'])
    from pipeline.runner_r13 import CODE_NAMES
    snapshot={n:sha(ROOT/n) for n in CODE_NAMES}
    frontend=TorchFeatureFrontend(basis)
    rng=np.random.default_rng(2026102714);synthetic=rng.normal(100,15,(37,15,25))
    synthetic[0,:10]=np.nan;synthetic[1,:,23:]=np.nan;synthetic[2,:,3]=np.nan
    synthetic[3,:,:22]=np.nan;synthetic[4,:,1::2]=np.nan
    synthetic[5,3:8]=np.nan;synthetic[6,:,1:22]=np.nan
    synthetic[7]=100.;synthetic[8]=np.asarray([-.15,-.05,.05,.15,99.95]*5)[None,:]
    checks=[]
    def parity(name,db):
        expected,w=transform_channels(db,basis,True);a,b=frontend(torch.from_numpy(db))
        actual=a.detach().numpy();weights=b.detach().numpy()
        per=[float(np.max(np.abs(actual[:,i]-expected[:,i]))) for i in range(11)]
        error=float(np.max(np.abs(weights-w)))
        if max(per)>protocol['feature_atol'] or error>protocol['weight_atol']:raise AssertionError((name,per,error))
        assert np.array_equal(actual[:,5],np.isfinite(db).astype(np.float32))
        checks.append({'case':name,'channel_max_abs_error':per,'weights_max_abs_error':error})
    parity('synthetic_float64_ties_partialbands_allmissingband_constant',synthetic)
    parity('synthetic_float32',synthetic.astype(np.float32))
    # URN upper two bands do not affect new safe23 shape maps; original8 and
    # full25 energy weights deliberately retain those measured bands.
    changed=synthetic.copy();changed[:,:,23:]+=40
    a,_=frontend(torch.from_numpy(synthetic));b,_=frontend(torch.from_numpy(changed))
    assert torch.equal(a[:,8:],b[:,8:])
    base,w=transform(torch.from_numpy(synthetic),include_shape=False)
    e,v=transform_channels(synthetic,None,False)
    assert np.max(abs(base.numpy()-e))<=3e-6 and np.max(abs(w.numpy()-v))<=3e-8
    p0,_=TorchFeatureFrontend(include_shape=False,pad_to_eleven=True)(torch.from_numpy(synthetic))
    assert p0.shape[1]==11 and not bool(p0[:,8:].any())
    rejected=[]
    for name,bad,mask in [('labels_dict',{'db':synthetic},None),('inf',torch.full((1,15,25),torch.inf),None),
                          ('all_missing',torch.full((1,15,25),torch.nan),None),('wrong_mask',torch.from_numpy(synthetic),torch.zeros(synthetic.shape,dtype=torch.bool))]:
        try:frontend(bad,mask)
        except (TypeError,ValueError):rejected.append(name)
        else:raise AssertionError('Invalid input accepted: '+name)
    from pipeline.full_data_r13 import R13Sources,DynamicR13Dataset
    runtime=json.loads((ROOT/'configs/r13_runtime_candidate_highshare_P1_v2.json').read_text())
    sources=R13Sources(bundle=bundle,runtime_policy=runtime);parent=DynamicR13Dataset(sources,0);raw=RawR14Dataset(parent)
    start=time.perf_counter();items=[raw[i] for i in range(protocol['actual_TRAIN_benchmark_rows'])];synthesis=time.perf_counter()-start
    db=np.stack([r['db'] for r in items]);parity('actual_TRAIN4096_all_source_types',db)
    for i in [0,1,17,63]:
        original=parent.raw_item(i)
        for k in ('share','cell_share','presence'):
            assert np.array_equal(np.asarray(items[i][k]),np.asarray(original[k]))
    # Small actual batch for independent MPS hardware diagnostic; q/y are
    # untouched loss targets and are not transform inputs.
    np.savez_compressed(OUT/'actual_TRAIN_fixture.npz',db=db[:64],observed_mask=np.isfinite(db[:64]),
        cell_share=np.stack([r['cell_share'] for r in items[:64]]),share=np.asarray([r['share'] for r in items[:64]],np.float32),
        presence=np.asarray([r['presence'] for r in items[:64]],np.float32))
    timers={}
    start=time.perf_counter()
    for row in db:transform_channels(row[None],basis,True)
    timers['NumPy_per_item_seconds']=time.perf_counter()-start
    def measure(fn):
        times=[]
        for repeat in range(3):
            start=time.perf_counter()
            for begin in range(0,len(db),128):fn(db[begin:begin+128])
            times.append(time.perf_counter()-start)
        return {'repetitions':times,'median_seconds':float(np.median(times))}
    timers['NumPy_batched128']=measure(lambda x:transform_channels(x,basis,True))
    with torch.inference_mode():timers['Torch_batched128']=measure(lambda x:frontend(torch.from_numpy(x)))
    from pipeline.models_r13 import make_r13_model
    from pipeline.training_r13 import physical_multitask_loss
    x,w=transform_channels(db[:8],basis,True);tx=torch.from_numpy(x);tw=torch.from_numpy(w)
    batch=torch.from_numpy(db[:8]);mask=torch.isfinite(batch)
    y=torch.tensor([r['share'] for r in items[:8]]);z=torch.tensor([r['presence'] for r in items[:8]],dtype=torch.float32)
    q=torch.from_numpy(np.stack([r['cell_share'] for r in items[:8]]));model_checks=[]
    temp=Path(tempfile.mkdtemp(prefix='r14-frontend-checkpoint-',dir='/private/tmp'))
    for kind in ('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer'):
        torch.manual_seed(2026102714);backbone=make_r13_model(kind);model=RawFeatureMaskModel(backbone,basis);model.eval()
        with torch.no_grad():expected=backbone(tx,tw);actual,weights=model(batch,mask,return_weights=True)
        error=max(float(abs(a-b).max()) for a,b in zip(expected,actual));assert error<=3e-6
        model.train();opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003);opt.zero_grad()
        pred,weights=model(batch,mask,return_weights=True);loss,_=physical_multitask_loss(pred,(y,z,q),weights)
        loss.mean().backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert not any(True for _ in model.frontend.parameters());gradient=sum(float(p.grad.abs().sum()) for p in model.backbone.parameters() if p.grad is not None);assert gradient>0
        opt.step();model.eval()
        with torch.no_grad():before=model(batch,mask)
        path=temp/(kind+'.pt');torch.save({'state_dict':model.state_dict(),'metadata':model.metadata(),'fixture_only':True},path)
        saved=torch.load(path,weights_only=True);new=RawFeatureMaskModel(make_r13_model(kind),basis);new.load_state_dict(saved['state_dict'],strict=True);new.eval()
        with torch.no_grad():after=new(batch,mask)
        assert all(torch.equal(a,b) for a,b in zip(before,after))
        model_checks.append({'kind':kind,'old_vs_raw_frontend_max_abs':error,'actual_Adam_steps':1,
            'backbone_gradient_L1':gradient,'frontend_parameters':0,'strict_checkpoint_reload_bitwise':True})
    sources.assert_unchanged();assert all(sha(ROOT/n)==h for n,h in snapshot.items())
    result={'status':'passed','protocol_sha256':sha(OUT/'protocol.json'),'CPU_threads':2,'data_hash':m['data_hash'],
        'basis_file_sha256':sha(basis_path),'basis_training_binding':f['basis_training_binding'],
        'feature_checks':checks,'URNsafe23_shape_invariance':True,'P0_zero_padded11':True,'rejected_inputs':rejected,
        'model_checks':model_checks,'benchmark':{'rows':len(db),'synthesis_only_seconds':synthesis,**timers,
            'GPU_latency_or_speedup_measured':False,'not_full_epoch_or_full_training_timing':True},
        'original_R13_code_sha256':snapshot,'original_R13_sources_unchanged':True,
        'code_sha256':{n:sha(ROOT/n) for n in ('pipeline/features_r14_torch.py','pipeline/raw_frontend_r14.py','ops/verify_r14_torch_frontend.py')},
        'actual_TRAIN_fixture_sha256':sha(OUT/'actual_TRAIN_fixture.npz'),
        'Test_report_arrays_external_network_GPU_accessed':False,'fixture_weights_not_production':str(temp)}
    write(OUT/'RESULTS.json',result);print(json.dumps({'status':'passed','result':str(OUT/'RESULTS.json'),'benchmark':result['benchmark']}))

if __name__=='__main__':main()
