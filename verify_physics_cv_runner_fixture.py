"""Independent frozen-code V3 five-fold numerical integration fixture.

Only generated 24/12/12-row train/inner/outer arrays are fitted or evaluated.
Original pinned foundation weights are read, never modified or fine-tuned.
"""
from pathlib import Path
import copy,hashlib,json,os,subprocess,sys,tempfile,time
import numpy as np
import pandas as pd
from pipeline.train_group_cv import preflight,execute,digest,immutable_step,inner_weights
from pipeline.mixtures import foundation_features
from verify_training_policy_fixture import fixture
ROOT=Path(__file__).resolve().parent
PROTOCOL='source_group_nested5_week24h_v1'


def command(arguments,log):
    env=os.environ.copy();env['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    r=subprocess.run([sys.executable,'-m',*arguments],cwd=ROOT,env=env,capture_output=True,text=True)
    log.write_text(r.stdout+'\n'+r.stderr)
    if r.returncode:raise AssertionError('Fixture subprocess failed: '+str(log)+'\n'+log.read_text())


def expect_value_error(action):
    try:action()
    except ValueError as exc:return str(exc)
    raise AssertionError('An invalid resume/artifact was accepted')


def flags_metadata(fold,role,n=12):
    outer=role=='outer'
    return pd.DataFrame({'sample_id':[f'physics-numeric-{fold}-{role}-{i}' for i in range(n)],
        'fold':fold,'is_outer_oof':outer,'validation_id':np.arange(n),
        'background_group':[f'fixture-background-{fold}-{i%2}' for i in range(n)],
        'reference_group':[f'fixture-reference-{fold}-{i%3}' for i in range(n)],
        'boundary_artifact_candidate':(np.arange(n)%3==0).astype(bool),
        'urn_8321_nominal_warning':(np.arange(n)%3==1).astype(bool)})


def main():
    started=time.perf_counter();checks={};details={}
    before={p.name:digest(p) for p in (ROOT/'pipeline').glob('*.py')}
    original_weights={name:digest(ROOT/'models/pretrained'/name) for name in ['tabicl-regressor-v2-20260212.ckpt','kumo_large_regressor.pt']}
    with tempfile.TemporaryDirectory(prefix='uvn-physics-cv-runner-') as temporary:
        tmp=Path(temporary);cv_dir=tmp/'cv';cv_dir.mkdir();folds=[]
        for f in range(5):
            directory=cv_dir/f'fold_{f:02d}';gate=fixture(directory,cv=True);parts={}
            for i,split in enumerate(['train','validation','outer_validation']):
                n=24 if split=='train' else 12;rng=np.random.default_rng(7100+1000*f+i)
                db=rng.normal(90,4,(n,15,25)).astype('float32');db[::2,-2:]=np.nan;db[::3,:,23:]=np.nan
                y=np.resize(np.array([0.,.2,.6,.9],dtype='float32'),n)
                q=np.where(np.isfinite(db),y[:,None,None],0).astype('float32')
                np.savez_compressed(directory/(split+'.npz'),db=db,share=y,presence=(y>0).astype('float32'),cell_share=q)
                parts[split]={'examples':n,'positive_draw_became_zero_count':0,'frozen_generation_replayed':True}
            flags_metadata(f,'inner').to_parquet(directory/'inner_validation_metadata.parquet',index=False)
            flags_metadata(f,'outer').to_parquet(directory/'validation_metadata.parquet',index=False)
            files={p.name:digest(p) for p in directory.iterdir() if p.suffix in ['.npz','.parquet']}
            gate.update(version='quality_v2_physics_group_cv_prebuilt',feature_version=3,preprocessing_version=3,
                protocol=PROTOCOL,cv_protocol=PROTOCOL,fold=f,parent_cv_source_hash='physics-numerical-fixture-only',
                source_hash=f'physics-fixture-source-{f}',nested_early_stopping=True,requires_fresh_task_fit=True,
                sealed_arrays_accessed=False,model_evaluated_on_sealed=False,model_training_executed=False,
                minimum_cross_partition_support_gap_hours=24.,validation_role='inner_early_stopping',
                outer_validation_role='outer_oof_reporting_only',partitions=parts,file_sha256=files,
                data_package_hash=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
            (directory/'readiness_manifest.json').write_text(json.dumps(gate))
            folds.append({'fold':f,'data_dir':str(directory),'data_package_hash':gate['data_package_hash'],
                'source_hash':gate['source_hash'],'outer_validation_file':'outer_validation.npz',
                'validation_metadata_file':'validation_metadata.parquet','validation_examples':12})
        cv={'protocol':PROTOCOL,'ready_to_train':True,'n_splits':5,'folds':folds,'source_hash':'physics-numerical-fixture-only',
            'challenge_test_accessed':False,'nested_early_stopping':True,'outer_validation_used_for_early_stopping':False}
        (cv_dir/'cv_manifest.json').write_text(json.dumps(cv))
        config=json.loads((ROOT/'configs/group_cv_physics_training.json').read_text())
        config.update(cv_dir=str(cv_dir),output_dir=str(tmp/'models'),models=['cnn'],foundation_models={},seeds=[42],status='Numerical fixture only; one epoch')
        config['training'].update(epochs=1,batch=8,threads=1,trees=2,feature_version=3)
        _,ready=preflight(config,cv_dir);assert ready['ready'] and ready['oof_examples']==60
        checks['V3_five_fold_training_preflight']='passed; generated arrays only'
        output=tmp/'models';execute(config,cv,cv_dir,output,'cpu',False)
        report=json.loads((output/'cv_report.json').read_text())
        assert report['n_splits']==5 and report['oof_examples']==60
        assert report['outer_used_for_early_stopping'] is False and report['outer_used_for_ensemble_weights'] is False
        with np.load(output/'oof_predictions.npz') as z:
            oof={k:z[k] for k in z.files}
        meta=pd.read_parquet(output/'oof_metadata.parquet')
        assert len(np.unique(oof['sample_id']))==60 and np.array_equal(oof['sample_id'].astype(str),meta.sample_id.to_numpy(dtype=str))
        error=oof['ensemble_share'].astype('float64')-oof['share_true'].astype('float64')
        boundary=meta.boundary_artifact_candidate.to_numpy();urn=meta.urn_8321_nominal_warning.to_numpy()
        assert boundary.dtype==bool and urn.dtype==bool and boundary.sum()==20 and urn.sum()==20
        expected_all=float(np.sqrt(np.mean(error**2)));expected_clean=float(np.sqrt(np.mean(error[~boundary]**2)))
        metrics=report['pooled_oof']['ensemble']
        assert np.isclose(metrics['oof_rmse_all'],expected_all,rtol=0,atol=1e-12)
        assert np.isclose(metrics['oof_rmse_excluding_boundary'],expected_clean,rtol=0,atol=1e-12)
        assert metrics['quality_slices']['boundary_artifact_candidate_flagged']['n']==20
        assert metrics['quality_slices']['boundary_artifact_candidate_excluded']['n']==40
        assert metrics['quality_slices']['urn_8321_nominal_warning_flagged']['n']==20
        details['numerical_fixture_OOF_metrics']={'rows':60,'boundary_flagged':20,'boundary_excluded':40,'urn_warning_flagged':20,
            'rmse_all':expected_all,'rmse_excluding_boundary':expected_clean,'not_competition_performance':True}
        checks['pooled_all_boundary_excluded_and_URN_quality_slices_match_independent_math']='passed'
        for fold in folds:
            f=fold['fold'];target=output/f'fold_{f:02d}';policy=json.loads((target/'ensemble_policy.json').read_text())
            assessed=json.loads((target/'seed_42/outer_validation_predictions.metrics.json').read_text())
            task=json.loads((target/'seed_42/run_manifest.json').read_text())
            assert policy['fit_split']=='inner validation only' and policy['outer_used_for_fit'] is False
            assert assessed['policy_sha256_before_outer']==digest(target/'ensemble_policy.json')
            assert task['arguments']['feature_version']==3 and task['training_summaries']['cnn']['architecture']['input_channels']==6
            assert task['training_summaries']['cnn']['epochs_run']==1
        checks['five_fold_CNN_1epoch_inner_freeze_then_outer']='passed; baseline width, six channels'
        checkpoints={f: digest(output/f'fold_{f:02d}/seed_42/cnn.pt') for f in range(5)}
        execute(config,cv,cv_dir,output,'cpu',True)
        assert checkpoints=={f:digest(output/f'fold_{f:02d}/seed_42/cnn.pt') for f in range(5)}
        checks['same_config_resume_reuses_recorded_checkpoints']='passed'
        altered=copy.deepcopy(config);altered['training']['dropout']=.2
        details['changed_config_rejection']=expect_value_error(lambda:execute(altered,cv,cv_dir,output,'cpu',True))
        checks['changed_configuration_resume_rejected']=True
        # Run each real pinned foundation API on fold 0, 24 context rows and
        # one member. Fit the combined diagnostic policy on inner predictions.
        directory=Path(folds[0]['data_dir']);gate=json.loads((directory/'readiness_manifest.json').read_text())
        with np.load(directory/'train.npz') as z:context_shape=foundation_features(z['db'],version=3).shape
        assert context_shape==(24,100);details['foundation_context_feature_shape']=list(context_shape)
        inner_runs={};common_by_kind={};foundation_details={}
        with np.load(output/'fold_00/seed_42/validation_predictions.npz') as z:
            y=z['share_true'];inner_predictions=[z['cnn_share']]
        for kind in ['tabicl','kumo_large']:
            inside=tmp/(kind+'_inner');inner_runs[kind]=inside
            common=['pipeline.cv_foundations','--data',str(directory),'--kind',kind,'--context','24','--members','1','--feature-version','3','--device','cpu','--threads','1','--seed','42','--batch','8']
            common_by_kind[kind]=common
            command([*common,'--phase','inner','--output',str(inside)],tmp/(kind+'_inner.log'))
            im=json.loads((inside/'manifest.json').read_text())
            assert im['feature_version']==3 and im['context_rows']==24 and im['members']==1 and im['outer_accessed'] is False
            assert im['weight_training'] is False and im['epochs'] is None and im['optimizer'] is None
            with np.load(inside/'validation_predictions.npz') as z:
                assert np.array_equal(z['share_true'],y);inner_predictions.append(z[kind+'_share'])
        weights=inner_weights(np.stack(inner_predictions,1),y)
        foundation_policy=tmp/'foundation_inner_fitted_policy.json'
        foundation_policy.write_text(json.dumps({'models':['cnn','tabicl','kumo_large'],'weights':weights.tolist(),
            'affine':[1.,0.],'fit_split':'inner validation only','outer_used_for_fit':False,'seeds':[42],
            'data_package_hash':gate['data_package_hash'],'source_hash':gate['source_hash'],'fold':0,'protocol':PROTOCOL}))
        for kind in ['tabicl','kumo_large']:
            outside=tmp/(kind+'_outer')
            command([*common_by_kind[kind],'--phase','outer','--output',str(outside),'--inner-result',str(inner_runs[kind]),'--policy',str(foundation_policy)],tmp/(kind+'_outer.log'))
            om=json.loads((outside/'manifest.json').read_text())
            assert om['phase']=='outer' and om['feature_version']==3 and om['weight_training'] is False
            assert om['policy_sha256_before_outer']==digest(foundation_policy) and om['inner_replay_max_abs_error']<4e-5
            outer_metrics=om['results']['outer_validation']
            assert 'quality_slices' in outer_metrics and outer_metrics['quality_slices']['boundary_artifact_candidate_flagged']['n']==4
            foundation_details[kind]={'context_rows':24,'feature_columns':100,'members':1,'feature_version':3,
                'inner_replay_max_abs_error':om['inner_replay_max_abs_error'],'original_weights_sha256':om['checkpoint_sha256'],
                'weight_training':False,'policy_fitted_on':'generated inner validation only'}
            checks[kind+'_actual_pinned_API_100feature_inner_outer_replay']='passed; numerical context only'
        details['actual_foundation_API']=foundation_details
        checkpoint=output/'fold_00/seed_42/cnn.pt';original=checkpoint.read_bytes();checkpoint.write_bytes(original+b'changed-artifact')
        details['changed_checkpoint_rejection']=expect_value_error(lambda:execute(config,cv,cv_dir,output,'cpu',True));checkpoint.write_bytes(original)
        checks['checkpoint_artifact_tamper_resume_rejected']=True
        policy_path=output/'fold_00/ensemble_policy.json';original_policy=policy_path.read_bytes();policy_path.write_bytes(original_policy+b' ')
        details['changed_policy_rejection']=expect_value_error(lambda:execute(config,cv,cv_dir,output,'cpu',True));policy_path.write_bytes(original_policy)
        checks['frozen_policy_artifact_tamper_resume_rejected']=True
        unrecorded=tmp/'unrecorded';unrecorded.mkdir();payload=unrecorded/'partial';payload.mkdir()
        expect_value_error(lambda:immutable_step(unrecorded,'unknown',[payload],lambda:None));checks['unrecorded_partial_artifacts_rejected']=True
    after={p.name:digest(p) for p in (ROOT/'pipeline').glob('*.py')}
    assert before==after,'Frozen core pipeline code changed during integration fixture'
    assert original_weights=={name:digest(ROOT/'models/pretrained'/name) for name in original_weights}
    checks['all_core_pipeline_and_original_foundation_weights_unchanged']=True
    result={'passed':True,'checks':checks,'details':details,'elapsed_seconds':time.perf_counter()-started,
        'real_competition_arrays_opened':False,'full_competition_training_executed':False,'GPU_used':False,
        'challenge_Test_calibration_sealed_opened':False,'generated_numerical_outer_rows_evaluated':60,
        'scope':'Five tiny CNN folds and one tiny context per original foundation validate execution, not competition quality, convergence or performance.',
        'code_sha256':after,'pinned_original_weight_sha256':original_weights}
    (ROOT/'reports/physics_cv_runner_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['code_sha256','pinned_original_weight_sha256']},indent=2))
if __name__=='__main__':main()
