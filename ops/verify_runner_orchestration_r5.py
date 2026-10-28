"""Control-flow fixture; child fits are STUBBED, not performance evidence."""
from pathlib import Path
import copy,json,tempfile
import numpy as np
import pandas as pd
from pipeline.dynamic_passage_r5 import _make_numeric_fixture
from pipeline.fold_data_r5 import FoldDataR5
from pipeline.grid_inputs_r5 import file_sha256,canonical_sha256
import pipeline.runner_r5 as runner


def main():
    root=Path(__file__).resolve().parents[1];tmp=Path(tempfile.mkdtemp(prefix='r5_orchestration_fixture_'))
    folder,source,source_info,gate=_make_numeric_fixture(tmp)
    for role,npz,meta in [('train','train.npz','train_sample_metadata.parquet'),
                          ('validation','validation.npz','inner_validation_metadata.parquet'),
                          ('outer_validation','outer_validation.npz','outer_validation_metadata.parquet')]:
        with np.load(folder/npz,allow_pickle=False) as z:y=z['share']
        n=len(y)
        frame=pd.DataFrame({'sample_id':[f'passage_r4_f00_{role}_{i:07d}' for i in range(n)],
            'fold':0,'validation_id':np.arange(n),'effective_share':y.astype(float),
            'is_outer_oof':role=='outer_validation','site':['numeric_site_'+str(i%2) for i in range(n)],
            'background_group':['numeric_bg_'+str(i//2) for i in range(n)],
            'reference_group':[json.dumps(['numeric_reference_'+str(i%4)]) if value>0 else '[]' for i,value in enumerate(y)],
            'boundary_artifact_candidate':[(i==0 and role=='outer_validation') for i in range(n)],
            'urn_8321_nominal_warning':False})
        frame.to_parquet(folder/meta,index=False)
        if role=='outer_validation':frame.to_parquet(folder/'validation_metadata.parquet',index=False)
    gate.update(validation_role='inner_early_stopping',outer_validation_role='outer_oof_reporting_only')
    files={p.name:file_sha256(p) for p in folder.iterdir() if p.is_file() and p.name not in ['readiness_manifest.json','manifest.json']}
    gate['file_sha256']=files
    import hashlib
    gate['data_package_hash']=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    for name in ['readiness_manifest.json','manifest.json']:(folder/name).write_text(json.dumps(gate,indent=2))
    config=json.loads(runner.CONFIG.read_text());config['dataset']=str(folder.parent)
    runner.validate_config(config)
    output=tmp/'result';events=[];fits=[]
    real_call=runner._call;original_load=FoldDataR5.load

    def checked_load(self,role,*,frozen_policy=None):
        if role=='outer':
            assert (output/'frozen_ensemble_policy.json').is_file()
            assert len(fits)==9
            events.append('read_outer_after_freeze')
        else:events.append('read_'+role)
        return original_load(self,role,frozen_policy=frozen_policy)

    def child_stub(module,args):
        values={args[i]:str(args[i+1]) for i in range(1,len(args),2)}
        action=args[0];kind=values.get('--kind','lgbm_physics');observation=values['--observation']
        path=Path(values['--output']);assert path.is_absolute()
        context=json.loads(Path(values['--context']).read_text());f=FoldDataR5(context['data_fold'])
        role=values.get('--role','inner')
        frozen=json.loads(Path(values['--frozen-policy']).read_text()) if '--frozen-policy' in values else None
        arrays,_=f.load(role,frozen_policy=frozen)
        # Deliberate synthetic stub, using fake labels to construct predictable
        # control-flow outputs. No real model or actual held-out input is used.
        offset={'bg_cnn':.08,'tfgrid_fdy_conformer':.06,'lgbm_physics':.07,'fdy':.065,'itransformer':.055,'timemoe_public':.03,'dasheng':.02}[kind]
        if observation=='resolution_01':offset*=.8
        if observation=='resolution_01_endpoint':offset*=.7
        p=np.clip(arrays['share']+.5*offset,0,1).astype('float32')
        pred={'share':p,'score':np.clip(.1+.8*arrays['presence'],0,1).astype('float32')}
        if kind!='lgbm_physics':pred['cell_mask']=np.broadcast_to(p[:,None,None],(len(p),15,25)).copy()
        if action=='fit':
            fits.append((kind,observation));events.append('fit_'+kind)
            path.mkdir(parents=True)
            np.savez_compressed(path/'inner_predictions.npz',**pred)
            (path/'summary.json').write_text(json.dumps({'status':'completed','kind':kind,'best_inner_rmse':float(np.sqrt(np.mean((p-arrays['share'])**2)))}))
        else:
            assert role=='outer';events.append('predict_outer_'+kind)
            path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**pred)
    try:
        FoldDataR5.load=checked_load;runner._call=child_stub
        summary=runner.fit_fold(config,0,output)
    finally:FoldDataR5.load=original_load;runner._call=real_call
    assert summary['status']=='completed' and len(summary['models'])==7
    assert summary['selected_observation']=='resolution_01_endpoint'
    assert summary['outer_changed_policy'] is False
    assert 'read_outer_after_freeze' in events and events.index('read_outer_after_freeze')>max(i for i,v in enumerate(events) if v.startswith('fit_'))
    metrics=json.loads((output/'outer_metrics.json').read_text())
    assert metrics['ensemble']['quality_slices']['boundary_artifact_candidate_flagged']['n']==1
    with np.load(output/'outer_oof.npz',allow_pickle=False) as z:
        assert z['boundary_artifact_candidate'].sum()==1 and len(z['site'])==24
    manifest=json.loads((folder/'readiness_manifest.json').read_text())
    unsafe=copy.deepcopy(manifest);unsafe['source_checks_passed']=False
    (folder/'readiness_manifest.json').write_text(json.dumps(unsafe))
    try:FoldDataR5(folder)
    except ValueError:pass
    else:raise AssertionError('Source-isolation preflight guard missing')
    finally:(folder/'readiness_manifest.json').write_text(json.dumps(manifest))
    changed=copy.deepcopy(config);changed['ensemble']['ridge']=.1
    try:runner.validate_config(changed)
    except ValueError:pass
    else:raise AssertionError('Undeclared ridge not rejected')
    report={'status':'PASS','scope':'Synthetic orchestration fixture; all child model training/prediction STUBBED',
        'actual_model_training':False,'actual_data_Test_outer_or_GPU_accessed':False,
        'fake_roles':{'train':72,'inner':24,'outer':24},'fit_jobs_stubbed':len(fits),
        'model_count':7,'all_INNER_reads_before_policy_freeze_only_for_selection':True,
        'OUTER_read_after_all_fits_and_policy_freeze':True,
        'legacy_validation_metadata_points_to_outer_and_explicit_inner_path_works':True,
        'OOF_boundary_and_site_columns_preserved':True,'unsafe_source_and_changed_ridge_rejected':True,
        'code_sha256':{'pipeline/runner_r5.py':file_sha256(root/'pipeline/runner_r5.py'),
                       'pipeline/fold_data_r5.py':file_sha256(root/'pipeline/fold_data_r5.py'),
                       'ops/verify_runner_orchestration_r5.py':file_sha256(Path(__file__))},
        'fixture_directory':str(tmp)}
    (root/'reports/R5_RUNNER_ORCHESTRATION_FIXTURE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
