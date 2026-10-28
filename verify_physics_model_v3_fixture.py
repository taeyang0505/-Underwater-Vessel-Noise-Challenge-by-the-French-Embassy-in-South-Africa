"""V3 analytical/tiny development fixtures; never full competition training."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,warnings
from unittest import mock
import numpy as np
import torch
from torch import nn
from pipeline.mixtures import (fixed_features,physics_features,foundation_features,cnn_channels,
    high_frequency_anomaly_features,per_band_baseline_db,baseline_excess_spectrum,retrieval_features)
from pipeline.models import make_model
from pipeline.train import load_prebuilt_data,predict_deep,build_parser,main as training_main
from pipeline.infer import predict
from pipeline.contrastive import contrastive_regularizer
from pipeline import gpu_preflight
from verify_training_policy_fixture import fixture,digest
ROOT=Path(__file__).resolve().parent

def main():
    warnings.filterwarnings('ignore',category=RuntimeWarning);torch.set_num_threads(1);checks={};details={}
    rng=np.random.default_rng(701)
    x=(80+np.arange(15)[None,:,None]+np.arange(25)[None,None,:]/10+np.arange(3)[:,None,None]).astype('float32')
    x[0,5:7,0]=np.nan;x[1,:,[23,24]]=np.nan
    source=x.copy();old_c,old_w=cnn_channels(x);c,w=cnn_channels(x,version=3)
    assert c.shape==(3,6,15,25) and np.array_equal(c[:,:4],old_c) and np.array_equal(w,old_w)
    assert not c[:,5,0].any() and not c[0,5,5:8,0].any()
    floor=np.nanquantile(x,.2,axis=1,keepdims=True)
    assert np.allclose(per_band_baseline_db(x),floor,equal_nan=True,atol=1e-7)
    assert np.allclose(c[:,4],np.where(np.isfinite(x),(x-floor)/20,0),atol=1e-7)
    assert (c[:,4][np.isfinite(x)&(x<floor)]<0).all() and (c[:,4]<0).any()
    assert np.all(c[:,4:][~np.isfinite(x)[:,None].repeat(2,axis=1)]==0)
    assert np.array_equal(source,x,equal_nan=True) and np.all(w[~np.isfinite(x)]==0)
    checks['six_channels_old_four_exact_mask_conservation_and_gap_aware_derivative']='passed'
    checks['signed_per_band_baseline_channel_preserves_bottom_quantile_negatives']='passed'
    assert fixed_features(x).shape==(3,1254) and fixed_features(x,version=3).shape==(3,1258)
    assert foundation_features(x).shape==(3,96) and foundation_features(x,version=3).shape==(3,100)
    assert physics_features(x).shape==(3,45)
    h=high_frequency_anomaly_features(x)
    assert np.allclose(h[2],[.005,.005,.7,.7],atol=1e-6) and not h[1].any()
    assert np.allclose(h,high_frequency_anomaly_features(x+31),atol=1e-6)
    assert np.allclose(baseline_excess_spectrum(x),baseline_excess_spectrum(x+31),atol=1e-10)
    checks['1258_45_100_feature_counts_analytic_highband_features_and_common_gain']='passed'
    widths={'cnn':8,'fdy':8,'itransformer':12,'tabm':16};counts={}
    for kind in ['cnn','fdy','itransformer']:
        model=make_model(kind,width=16,k=2,architecture_version=3,model_width=widths[kind]).eval()
        assert model.feature_version==3 and model.preprocessing_version==3
        p,logit,q=model(torch.from_numpy(c),torch.from_numpy(w))
        assert torch.isfinite(p).all() and torch.isfinite(logit).all() and (q>=0).all() and (q<=1).all()
        assert torch.allclose(p,(q*torch.from_numpy(w)).sum((1,2)),atol=1e-7)
        restored=make_model(kind,width=16,k=2,architecture_version=3,model_width=widths[kind]);restored.load_state_dict(model.state_dict(),strict=True)
        model.train();extra=contrastive_regularizer(model,kind,rng.normal(90,3,(4,15,25)).astype('float32'),'cpu',seed=5);extra.backward()
        assert torch.isfinite(extra) and any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.parameters())
    checks['three_spectral_architectures_require_bounded_q_energy_pooling_and_v3_contrastive']='passed'
    for kind in ['tabm','cnn','fdy','itransformer']:
        model=make_model(kind,architecture_version=3);counts[kind]=sum(p.numel() for p in model.parameters())
    details['default_width_v3_parameter_counts']=counts
    with tempfile.TemporaryDirectory(prefix='uvn-physics-v3-fixture-') as temp:
        temp=Path(temp);data_dir=temp/'data';g=fixture(data_dir,cv=True)
        for split in ['train','validation']:
            path=data_dir/(split+'.npz')
            with np.load(path) as z:d={k:z[k] for k in z.files}
            d['db'][::2,:,23:]=np.nan;d['cell_share'][~np.isfinite(d['db'])]=0
            np.savez_compressed(path,**d);g['file_sha256'][path.name]=digest(path)
        g['version']='quality_v2_physics_group_cv_prebuilt';g['feature_version']=3
        g['data_package_hash']=hashlib.sha256(json.dumps(g['file_sha256'],sort_keys=True).encode()).hexdigest()
        (data_dir/'readiness_manifest.json').write_text(json.dumps(g))
        data,_,_=load_prebuilt_data(data_dir,cv_mode=True)
        args=build_parser().parse_args(['--data-dir',str(data_dir),'--run-dir',str(temp/'refused'),'--cv-mode','--validate-only'])
        try:training_main(args)
        except ValueError as e:assert 'feature-version 3' in str(e)
        else:raise AssertionError('Physics package silently accepted legacy features')
        assert not (temp/'refused').exists();checks['partial_band_observations_allowed_and_wrong_feature_version_rejected']='passed'
        output=temp/'run'
        command=[sys.executable,'-m','pipeline.train','--data-dir',str(data_dir),'--run-dir',str(output),'--cv-mode','--feature-version','3','--models','lgbm','lgbm_physics','lgbm_retrieval','tabm','cnn','fdy','itransformer','--epochs','1','--patience','7','--batch','8','--width','16','--k','2','--cnn-width','8','--fdy-width','8','--itransformer-width','12','--threads','1','--trees','2','--optimizer','adam','--scheduler','none']
        result=subprocess.run(command,cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True)
        if result.returncode:raise AssertionError(result.stdout+'\n'+result.stderr)
        manifest=json.loads((output/'run_manifest.json').read_text());assert manifest['status'].startswith('completed') and manifest['feature_version']==3
        assert not any((output/(s+'_predictions.npz')).exists() for s in ['outer_validation','calibration','domain_holdout'])
        members=['lgbm','lgbm_physics','lgbm_retrieval','tabm','cnn','fdy','itransformer'];weight=np.full(len(members),1/len(members))
        with np.load(output/'validation_predictions.npz') as z:stored={k:z[k] for k in z.files}
        policy={'models':members,'weights':weight.tolist(),'affine':[1.,0.]}
        p,s=predict(data['validation']['db'],output,policy,torch.device('cpu'))
        expected=np.stack([stored[k+'_share'] for k in members],1)@weight
        assert np.allclose(p,expected,atol=3e-6);details['V3_full_inference_replay_max_share_error']=float(np.max(np.abs(p-expected)))
        for kind in ['tabm','cnn','fdy','itransformer']:
            saved=torch.load(output/(kind+'.pt'),weights_only=True)
            assert saved['feature_version']==3 and saved['architecture']['architecture_version']==3
            assert saved['n_features']==(1258 if kind=='tabm' else None)
            model=make_model(kind,width=saved['width'],k=saved['k'],**saved['architecture']);model.load_state_dict(saved['state_dict'])
            rp,_=predict_deep(model,kind,data['validation'],torch.device('cpu'))
            assert np.allclose(rp,stored[kind+'_share'],atol=3e-6)
        pm=json.loads((output/'prototype_manifest.json').read_text())
        assert pm['feature_mode']==pm['query_feature_mode']=='per_band20db_excess_shape_v3' and pm['whole_mixture_spectrum_used'] is False
        proto=output/'prototypes.npz';features=retrieval_features(data['validation']['db'],proto,version=3)
        assert np.isfinite(features).all() and np.allclose(features,retrieval_features(data['validation']['db']+22,proto,version=3),atol=1e-5)
        try:retrieval_features(data['validation']['db'],proto,version=1)
        except ValueError:pass
        else:raise AssertionError('V3 prototype was silently queried with legacy features')
        checks['seven_candidate_tiny_CLI_checkpoint_reload_and_full_inference_replay']='passed'
        checks['train_only_excess_prototypes_same_frozen_query_mode_and_legacy_mismatch_guard']='passed'
        details['tiny_CLI_epochs']=1;details['tiny_CLI_rows']={s:len(d['share']) for s,d in data.items()};details['tiny_tree_feature_counts']={k:manifest['training_summaries'][k]['n_features'] for k in members[:3]}
        class TinyTab(nn.Module):
            def __init__(self):super().__init__();self.fc=nn.Linear(1258,2)
            def forward(self,x):return self.fc(x)[:,None,:]
        a=gpu_preflight.build_parser().parse_args(['--data-dir',str(data_dir),'--output',str(temp/'gpu.json'),'--feature-version','3','--batch','4','--warmup','1','--steps','2'])
        with mock.patch('pipeline.gpu_preflight.make_model',side_effect=lambda *a,**kw:TinyTab()):
            timing=gpu_preflight.benchmark_neural('tabm',data['train'],data['validation'],a,torch.device('cpu'),development_fixture=True)
        assert timing['feature_version']==3 and timing['architecture']['n_features']==1258 and timing['CUDA_measured'] is False
        checks['GPU_preflight_V3_schema_CPU_fixture_only']='passed'
    files=['mixtures.py','models.py','train.py','train_lgbm.py','infer.py','predict_lgbm.py','contrastive.py','gpu_preflight.py']
    report={'status':'passed','full_competition_training_executed':False,'real_GPU_used':False,'test_outer_calibration_sealed_opened':False,
        'checks':checks,'details':details,'feature_contract':{'fixed':1258,'physics_only':45,'foundation':100,'spectral_channels':6,'spectral_mask_energy_weights_preserved':True,'prototype_mode':'per_band20db_excess_shape_v3'},
        'code_sha256':{f:hashlib.sha256((ROOT/'pipeline'/f).read_bytes()).hexdigest() for f in files},
        'runtime':{'torch':torch.__version__,'target_GPU_torch_2_8_verified':False},'limits':['Tiny fixtures validate wiring, conservation, versioning and reproducibility, not competition performance.','Per-band lower-baseline excess is a fixed feature and does not prove a separated natural-vessel component.','Target Ubuntu/PyTorch 2.8/CUDA 12.8 execution still requires a server smoke test.']}
    (ROOT/'reports/physics_model_v3_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'checks':checks,'details':details},indent=2))
if __name__=='__main__':main()
