"""Numerical fixture only: validated OPS hybrid parity and frozen R1 regression."""
from pathlib import Path
import hashlib,io,json,sys,types,zipfile
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'ops'))
from pipeline import models,mixtures
from pipeline.background_prior_features import fit_background_prior,background_prior_maps
from train_background_hybrid_pilot import BackgroundPriorResidualCNN as OpsHybrid

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def same(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b):assert torch.equal(x,y),float((x-y).abs().max())

def main():
    torch.set_num_threads(4);torch.manual_seed(42);rng=np.random.default_rng(42)
    db=(75+rng.normal(0,3,(24,15,25))+np.linspace(8,-8,25)).astype('float32')
    db[rng.random(db.shape)<.04]=np.nan
    q=np.where(np.isfinite(db),rng.uniform(.05,.7,db.shape),0).astype('float32')
    before=db.copy();prior=fit_background_prior(db,q,seed=42,clusters=4)
    prior=json.loads(json.dumps(prior));x,w=mixtures.hybrid_cnn_channels(db,prior)
    base,ow=mixtures.cnn_channels(db,version=3)
    q0,bfrac,_=background_prior_maps(db,prior)
    ops_x=np.concatenate([base,q0[:,None],(np.log1p(np.clip(bfrac,0,10))/np.log(11.))[:,None]],1).astype('float32')
    assert np.array_equal(x,ops_x) and np.array_equal(w,ow)
    assert np.array_equal(db,before,equal_nan=True) and x.shape==(24,8,15,25)
    assert np.all(x.transpose(0,2,3,1)[~np.isfinite(db)]==0)
    canonical=models.make_model('cnn',architecture_version=4,input_channels=8,model_width=32,dropout=.1,prior_clip=.01,prior_strength=.5)
    canonical.background_prior=prior;ops=OpsHybrid()
    ops.load_state_dict(canonical.state_dict(),strict=True)
    xt,wt=torch.from_numpy(x[:8]),torch.from_numpy(w[:8])
    canonical.eval();ops.eval()
    with torch.no_grad():same(canonical(xt,wt,True),ops(xt,wt,True))
    assert torch.count_nonzero(canonical.last_residual)==0
    # Nonzero residual weights make parity exercise body gradients as well.
    with torch.no_grad():canonical.network.mask_head.weight.normal_(0,.03)
    ops.load_state_dict(canonical.state_dict(),strict=True)
    canonical.train();ops.train();qt=torch.from_numpy(q[:8]);y=(qt*wt).sum((1,2));z=torch.ones_like(y)
    losses=[];outs=[]
    for model in [canonical,ops]:
        torch.manual_seed(170);p,logit,cell,residual=model(xt,wt,True)
        loss=F.mse_loss(p,y)+.03*F.binary_cross_entropy_with_logits(logit,z)+.03*((cell-qt).square()*wt).sum((1,2)).mean()+.001*(residual.square()*wt).sum((1,2)).mean()
        loss.backward();losses.append(loss);outs.append((p,logit,cell,residual))
    same(outs[0],outs[1]);assert torch.equal(*losses)
    same([p.grad for p in canonical.parameters()],[p.grad for p in ops.parameters()])
    assert canonical.last_residual is outs[0][-1] and canonical.last_residual.requires_grad
    assert sum(float(p.grad.abs().sum()) for p in canonical.network.body.parameters())>0
    arch={'architecture_version':4,'input_channels':8,'model_width':32,'dropout':.1,'prior_clip':.01,'prior_strength':.5}
    buf=io.BytesIO();torch.save({'state':canonical.state_dict(),'architecture':arch,'prior':prior},buf);buf.seek(0)
    checkpoint=torch.load(buf,weights_only=True);restored=models.make_model('cnn',**checkpoint['architecture']);restored.background_prior=checkpoint['prior'];restored.load_state_dict(checkpoint['state'],strict=True)
    xr,wr=mixtures.hybrid_cnn_channels(db,restored.background_prior)
    assert np.array_equal(x,xr) and np.array_equal(w,wr)
    canonical.eval();restored.eval()
    with torch.no_grad():same(canonical(xt,wt),restored(xt,wt))
    for kwargs in [{'kind':'fdy'},{'kind':'itransformer'},{'kind':'tabm'},{'kind':'cnn','input_channels':6}]:
        try:models.make_model(architecture_version=4,**kwargs)
        except ValueError:pass
        else:raise AssertionError('Unsupported architecture 4 accepted')
    try:mixtures.hybrid_cnn_channels(db,None)
    except ValueError:pass
    else:raise AssertionError('Missing prior accepted')
    # Independent immutable source from the R1 GPU delivery, never its data.
    archive=ROOT/'delivery/group_cv_physics_v2_r1_gpu.zip'
    with zipfile.ZipFile(archive) as package:
        modules={}
        for name in ['models','mixtures']:
            source=package.read('pipeline/'+name+'.py').decode()
            module=types.ModuleType('immutable_r1_'+name);module.__file__=str(ROOT/'pipeline'/name)+'.py'
            exec(compile(source,'immutable-r1/'+name+'.py','exec'),module.__dict__);modules[name]=module
    regression=[]
    for version in [1,2,3]:
        assert np.array_equal(mixtures.fixed_features(db,version),modules['mixtures'].fixed_features(db,version))
        newc,neww=mixtures.cnn_channels(db,version);oldc,oldw=modules['mixtures'].cnn_channels(db,version)
        assert np.array_equal(newc,oldc) and np.array_equal(neww,oldw)
        for kind in ['tabm','cnn','fdy','itransformer']:
            torch.manual_seed(270);old=modules['models'].make_model(kind,architecture_version=version)
            torch.manual_seed(270);new=models.make_model(kind,architecture_version=version)
            assert old.state_dict().keys()==new.state_dict().keys()
            same(list(old.state_dict().values()),list(new.state_dict().values()))
            old.eval();new.eval()
            with torch.no_grad():
                if kind=='tabm':
                    inp=torch.from_numpy(mixtures.fixed_features(db[:3],version));assert torch.equal(old(inp),new(inp))
                else:
                    inp=torch.from_numpy(newc[:3]);weight=torch.from_numpy(neww[:3]);same(old(inp,weight),new(inp,weight))
            regression.append({'architecture_version':version,'kind':kind,'state_and_output_bitwise_equal':True})
    report={'status':'PASS','fixture_only':True,'actual_train_or_heldout_data_opened':False,'ops_hybrid_output_loss_gradient_bitwise_equal':True,'checkpoint_and_prior_reload_bitwise_equal':True,'eight_channel_transform_matches_OPS_bitwise':True,'missing_prior_and_wrong_architecture_rejected':True,'immutable_r1_regression':regression,'hybrid_parameters':sum(p.numel() for p in canonical.parameters()),'architecture':arch,'residual_loss':'0.001 * mean(sum(energy_weights * residual_logits**2))','prior_fitted_on':'generated numerical fixture only','files_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'pipeline/models.py',ROOT/'pipeline/mixtures.py',ROOT/'pipeline/background_prior_features.py',ROOT/'ops/train_background_hybrid_pilot.py',Path(__file__)]},'immutable_r1_archive_sha256':sha(archive)}
    target=ROOT/'reports/background_hybrid_parity_verification.json';target.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
