"""Independent tiny fixtures only: no competition training or GPU execution."""
from pathlib import Path
import contextlib,copy,hashlib,io,json,os,subprocess,sys,tempfile
from unittest import mock
import numpy as np
import torch
from torch import nn
from pipeline import train
from pipeline.models import make_model
ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fixture(folder,cv=False,domain=False):
    folder.mkdir();source=folder/'source';source.mkdir();(source/'manifest.json').write_text('{"development_fixture":true}')
    partitions={}
    splits=['train','validation']+([] if cv else ['calibration'])+(['domain_holdout'] if domain else [])
    for index,split in enumerate(splits):
        n=24 if split=='train' else 12;rng=np.random.default_rng(9000+index)
        db=rng.normal(90,4,(n,15,25)).astype('float32');db[::2,-2:]=np.nan
        y=np.resize(np.array([0.,.2,.6,.9],dtype='float32'),n);q=np.where(np.isfinite(db),y[:,None,None],0).astype('float32')
        np.savez_compressed(folder/(split+'.npz'),db=db,share=y,presence=(y>0).astype('float32'),cell_share=q)
        partitions[split]={'examples':n,'positive_draw_became_zero_count':0,'frozen_generation_replayed':True}
    if cv:(folder/'outer_validation.npz').write_bytes(b'OUTER INPUT MUST NEVER BE OPENED BY THE TRAINER')
    files={p.name:digest(p) for p in folder.glob('*.npz')}
    gate={'version':'quality_v2_development_fixture','ready_to_train':True,'source_checks_passed':True,'reference_group_and_duplicate_isolation_passed':True,'background_week_and_purge_isolation_passed':True,'sealed_holdout_locked':True,'challenge_test_accessed':False,'existing_sealed_NOAA_accessed':False,'sealed_holdout_materialized':False,'frozen_v1_data_modified':False,'source_hash':'fixture-only','source_version_dir':str(source),'source_version_manifest_sha256':digest(source/'manifest.json'),'file_sha256':files,'data_package_hash':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),'partitions':partitions}
    if cv:gate['cv_protocol']={'kind':'development_group_cv_fixture','fold':0,'n_splits':3}
    (folder/'readiness_manifest.json').write_text(json.dumps(gate));return gate

def args_for(data,output,*extra):
    return train.build_parser().parse_args(['--data-dir',str(data),'--run-dir',str(output),'--threads','1','--batch','8',*extra])

def must_fail(call,exc):
    try:call()
    except exc:return
    raise AssertionError('Expected '+exc.__name__)

class TinyTab(nn.Module):
    def __init__(self):super().__init__();self.fc=nn.Linear(1254,2)
    def forward(self,x):return self.fc(x)[:,None,:]

def controlled_training(data,folder,optimizer='adam',scheduler='none',early=True):
    args=args_for(Path('.'),folder,'--models','tabm','--optimizer',optimizer,'--scheduler',scheduler,'--epochs','100' if early else '1','--patience','2')
    records=[];values=[.3,.1,.2,.2] if early else [.1];val_calls=0
    def predict(model,kind,part,device,**kwargs):
        nonlocal val_calls
        if part is data['train']:return part['share'].copy(),part['presence'].copy()
        if val_calls<len(values):
            records.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()});value=values[val_calls]
        else:
            best_index=int(np.argmin(values))
            assert all(torch.equal(v,records[best_index][k]) for k,v in model.state_dict().items()),'Best state was not restored'
            value=min(values)
        val_calls+=1;return part['share']+value,part['presence'].copy()
    with mock.patch('pipeline.train.make_model',side_effect=lambda *a,**k:TinyTab()),mock.patch('pipeline.train.predict_deep',side_effect=predict),contextlib.redirect_stdout(io.StringIO()):
        model=train.train_deep('tabm',data['train'],data['validation'],torch.device('cpu'),args,folder)
    saved=torch.load(folder/'tabm.pt',weights_only=True);summary=saved['training_summary']
    assert summary['best_checkpoint_restored'] and summary['optimizer']=={'adam':'Adam','nadam':'NAdam','sgd_nesterov':'SGD'}[optimizer]
    assert all(torch.equal(model.state_dict()[k],saved['state_dict'][k]) for k in saved['state_dict'])
    if early:
        assert summary['epochs_run']==4 and summary['best_epoch']==2 and summary['stop_reason']=='early_stopping_patience_reached'
        assert any(not torch.equal(records[1][k],records[3][k]) for k in records[1]),'Fixture did not change weights after best epoch'
    else:assert summary['epochs_run']==1 and summary['stop_reason']=='max_epochs_reached'
    history=json.loads((folder/'tabm_history.json').read_text())
    assert all({'train_rmse','validation_rmse','best_epoch','epochs_run','stop_reason','learning_rate'}.issubset(h) for h in history)
    return summary

def main():
    torch.set_num_threads(1);checks={};artifacts=[]
    base=ROOT/'models/fullsources_cpu_a6d7f3a5_seed42'
    paths=[p for pattern in ['*.pt','*.txt','prototypes.npz'] for p in base.glob(pattern)]
    paths+=[ROOT/'models/ensemble_foundation/postprocess.json',ROOT/'reports/frozen_inference_audit.json']
    paths+=[ROOT/'data/synthetic/fullsources_cpu_a6d7f3a5_seed42'/(s+'.npz') for s in ['train','validation','calibration','domain_holdout']]
    before={str(p.relative_to(ROOT)):digest(p) for p in paths}
    with tempfile.TemporaryDirectory(prefix='uvn-training-fixture-') as tmp:
        tmp=Path(tmp);data_dir=tmp/'prebuilt';fixture(data_dir)
        data,gate,_=train.load_prebuilt_data(data_dir);checks['prebuilt_train_val_cal_without_domain']='passed'
        defaults=args_for(data_dir,tmp/'unused')
        assert (defaults.epochs,defaults.patience,defaults.optimizer,defaults.dropout)==(100,7,'adam',.1)
        must_fail(lambda:train.validate_arguments(args_for(data_dir,tmp/'unused','--epochs','101')),ValueError)
        checks['defaults_and_100_epoch_ceiling']='passed'
        with contextlib.redirect_stdout(io.StringIO()):train.main(args_for(data_dir,tmp/'validate_only','--validate-only'))
        assert not (tmp/'validate_only').exists();checks['validate_only_creates_no_run']='passed'
        existing=tmp/'existing';existing.mkdir();must_fail(lambda:train.main(args_for(data_dir,existing)),FileExistsError)
        checks['existing_run_refused']='passed'
        bad=tmp/'bad_gate';fixture(bad);g=json.loads((bad/'readiness_manifest.json').read_text());g['ready_to_train']=False;(bad/'readiness_manifest.json').write_text(json.dumps(g))
        must_fail(lambda:train.load_prebuilt_data(bad),ValueError)
        altered=tmp/'bad_hash';fixture(altered);(altered/'train.npz').write_bytes((altered/'train.npz').read_bytes()+b'changed')
        must_fail(lambda:train.load_prebuilt_data(altered),ValueError);checks['false_readiness_and_changed_array_rejected']='passed'
        invalid=tmp/'bad_target';g=fixture(invalid)
        with np.load(invalid/'train.npz') as z:d={k:z[k] for k in z.files}
        d['cell_share']=np.zeros_like(d['cell_share']);np.savez_compressed(invalid/'train.npz',**d)
        g['file_sha256']['train.npz']=digest(invalid/'train.npz');g['data_package_hash']=hashlib.sha256(json.dumps(g['file_sha256'],sort_keys=True).encode()).hexdigest();(invalid/'readiness_manifest.json').write_text(json.dumps(g))
        must_fail(lambda:train.load_prebuilt_data(invalid),ValueError);checks['energy_target_mismatch_rejected_even_with_valid_hash']='passed'
        optional=tmp/'with_domain';fixture(optional,domain=True);assert 'domain_holdout' in train.load_prebuilt_data(optional)[0]
        checks['optional_domain_supported']='passed'
        cv=tmp/'cv';fixture(cv,cv=True);real_sha=train.sha256;opened=[]
        def guarded_sha(path):
            opened.append(Path(path).name)
            if Path(path).name=='outer_validation.npz':raise AssertionError('Outer bytes opened')
            return real_sha(path)
        with mock.patch('pipeline.train.sha256',side_effect=guarded_sha):cvdata,_,v=train.load_prebuilt_data(cv,cv_mode=True)
        assert set(cvdata)=={'train','validation'} and 'outer_validation.npz' not in opened
        must_fail(lambda:train.load_prebuilt_data(cv),ValueError);checks['cv_reads_only_train_inner_validation_never_outer_bytes']='passed'
        for optimizer,scheduler,early in [('adam','none',True),('nadam','cosine',False),('sgd_nesterov','plateau',False)]:
            out=tmp/(optimizer+'_controlled');out.mkdir();artifacts.append(controlled_training(data,out,optimizer,scheduler,early))
        checks['controlled_curve_early_stop_best_state_and_three_optimizers']='passed'
        out=tmp/'actual_cnn_and_tree_fixture'
        command=[sys.executable,'-m','pipeline.train','--data-dir',str(data_dir),'--run-dir',str(out),'--models','lgbm','lgbm_physics','lgbm_retrieval','cnn','--device','cpu','--epochs','1','--patience','7','--optimizer','adam','--scheduler','plateau','--dropout','.1','--contrastive-weight','.05','--batch','8','--threads','1','--trees','2']
        result=subprocess.run(command,cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True)
        if result.returncode:raise AssertionError(result.stdout+'\n'+result.stderr)
        m=json.loads((out/'run_manifest.json').read_text());assert m['status'].startswith('completed') and not m['validation_protocol']['is_k_fold']
        assert (out/'validation_predictions.npz').exists() and (out/'calibration_predictions.npz').exists() and not (out/'domain_holdout_predictions.npz').exists()
        saved=torch.load(out/'cnn.pt',weights_only=True);model=make_model('cnn',width=saved['width'],k=saved['k'],**saved['architecture']);model.load_state_dict(saved['state_dict'])
        pred,_=train.predict_deep(model,'cnn',data['validation'],torch.device('cpu'))
        with np.load(out/'validation_predictions.npz') as z:assert np.allclose(pred,z['cnn_share'],atol=1e-6)
        assert saved['training_summary']['optimizer']=='Adam' and saved['training_summary']['contrastive_weight']==.05
        assert json.loads((out/'prototype_manifest.json').read_text())['v1_source_manifest_read'] is False
        checks['actual_tiny_CLI_four_candidates_metrics_contrastive_checkpoint_reload']='passed'
        artifacts.append({'actual_CLI_examples':{'train':24,'validation':12,'calibration':12},'CNN_epochs':1,'LightGBM_trees':2,'contrastive_weight':.05,'result_keys':sorted(m['results'])})
        cv_out=tmp/'actual_CV_fixture'
        cv_command=[sys.executable,'-m','pipeline.train','--data-dir',str(cv),'--run-dir',str(cv_out),'--cv-mode','--models','cnn','--epochs','1','--optimizer','adam','--scheduler','cosine','--batch','8','--threads','1']
        cv_result=subprocess.run(cv_command,cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True)
        if cv_result.returncode:raise AssertionError(cv_result.stdout+'\n'+cv_result.stderr)
        cv_manifest=json.loads((cv_out/'run_manifest.json').read_text())
        assert cv_manifest['status'].startswith('completed') and cv_manifest['arguments']['cv_mode'] is True
        assert (cv_out/'validation_predictions.npz').exists()
        assert not any((cv_out/(s+'_predictions.npz')).exists() for s in ['calibration','domain_holdout','outer_validation'])
        checks['actual_tiny_CV_CLI_inner_only_outer_canary_unread']='passed'
    # Legacy model defaults must still load and reproduce own validation output.
    with np.load(ROOT/'data/synthetic/fullsources_cpu_a6d7f3a5_seed42/validation.npz') as z:val={k:z[k][:2] for k in z.files}
    with np.load(base/'validation_predictions.npz') as z:stored={k:z[k] for k in z.files}
    errors={}
    for kind in ['tabm','cnn','fdy','itransformer']:
        saved=torch.load(base/(kind+'.pt'),weights_only=True);model=make_model(kind,width=saved['width'],k=saved['k']);model.load_state_dict(saved['state_dict'],strict=True)
        pred,_=train.predict_deep(model,kind,val,torch.device('cpu'));errors[kind]=float(np.max(np.abs(pred-stored[kind+'_share'][:2])))
        assert errors[kind]<2e-6
    checks['legacy_V1_checkpoint_shapes_and_predictions_unchanged']=errors
    after={str(p.relative_to(ROOT)):digest(p) for p in paths};assert before==after
    checks['V1_model_policy_audit_and_synthetic_array_hashes_unchanged']=True
    shell=subprocess.run(['bash','-n',str(ROOT/'run_gpu.sh')],capture_output=True,text=True);assert shell.returncode==0
    checks['GPU_runner_shell_syntax']='passed'
    implementation_sha256={p:digest(ROOT/p) for p in ['pipeline/train.py','pipeline/train_lgbm.py','pipeline/models.py','run_gpu.sh']}
    report={'status':'passed','actual_competition_training_executed':False,'GPU_used':False,'challenge_Test_read':False,'sealed_NOAA_read':False,'checks':checks,'controlled_training_summaries_and_fixture':artifacts,'implementation_sha256':implementation_sha256,'immutable_V1_hashes':before,'limits':'Tiny development fixtures validate execution and isolation, not convergence, GPU memory, real-data generalization or a global optimum.'}
    (ROOT/'reports/training_policy_v2_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':'passed','checks':checks},indent=2))

if __name__=='__main__':main()
