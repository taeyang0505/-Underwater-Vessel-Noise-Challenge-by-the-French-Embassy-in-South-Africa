"""Synthetic/mock singlefold inference fixture. No real models/data/Test IO."""
from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
import csv,hashlib,json,sys
import numpy as np
from pipeline.inference_r5 import FrozenInferenceR5,FEATURE_COLUMNS,REQUIRED_CODE,read_query_csv,original_energy_weights
from pipeline.hierarchical_ensemble_r5 import select_inner_policy,canonical_sha256

ROOT=Path(__file__).resolve().parents[1]
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def reject(fn,contains=None):
    try:fn()
    except (ValueError,TypeError,FileNotFoundError) as e:
        if contains is not None:assert contains.lower() in str(e).lower(),str(e)
        return str(e)
    raise AssertionError('Invalid/forbidden inference operation accepted')

class MockBackend:
    fixture_backend=True
    def __init__(self):self.active=0;self.maximum_active=0;self.loads=[];self.fit_calls=0
    def load_prior(self,path,context):return json.loads(path.read_text())
    def checkpoint(self,path):return json.loads(path.read_text())
    def load_neural(self,kind,settings,cache_dir,device):
        assert self.active==0;self.active+=1;self.maximum_active=max(self.maximum_active,self.active);self.loads.append(kind)
        return {'kind':kind}
    def predict_neural(self,model,checkpoint,kind,fold,db,device,batch_size):
        assert model['kind']==kind and checkpoint['kind']==kind
        # Per-window-only generated prediction; no fit/normalization across rows.
        q=1/(1+np.exp(-(.015*np.nanmean(db,axis=(1,2))+{'timemoe_public':.1,'dasheng':-.1,'bg_cnn':0}[kind])))
        mask=np.broadcast_to(q[:,None,None],db.shape).copy();w=original_energy_weights(db)
        return {'share':(mask*w).sum((1,2)),'score':np.clip(q+.05,0,1),'cell_mask':mask}
    def predict_tree(self,db,fold,model_dir,batch_size):
        p=1/(1+np.exp(-.015*np.nanmean(db,axis=(1,2))))
        return {'share':p,'score':np.clip(p-.05,0,1)}
    def release(self,model,device):self.active-=1;assert self.active==0

def make_fixture(base,nfolds=1):
    project=base/'project';run=base/'run';project.mkdir(parents=True);run.mkdir()
    hashes={}
    for name in REQUIRED_CODE:
        p=project/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# synthetic frozen code identity; never executed\n');hashes[name]=sha(p)
    entries=[]
    for number in range(nfolds):
        folder=run/f'fold_{number:02d}';folder.mkdir()
        binding={'fold':number,'seed':42,'source_hash':'a'*64,'data_package_hash':'b'*64,'train_npz_sha256':'c'*64,'train_examples':100}
        n=48;t=np.tile(np.linspace(.2,.8,16),3);error=np.tile(.06*np.sin(np.linspace(0,2*np.pi,16)),3);groups=np.repeat(['A','B','C'],16)
        shares={'timemoe_public':t+error/6,'dasheng':t-error/6,'bg_cnn':t+error,'lgbm_physics':t-error}
        policy=select_inner_policy(t,shares,groups,binding=binding,explicit_exclusions={'tfgrid_fdy_conformer':'not part of this mock-only fixture'})
        assert set(policy['weights'])==set(shares)
        dump(folder/'frozen_ensemble_policy.json',policy)
        observation={'quantization_decimals':None,'endpoint_dropout_probability':0.}
        config={'version':'r5_training_integration_v1','protocol':'source_group_nested5_vtuad_passage_r4','seed':42,
            'model_settings':{k:{} for k in shares},'observation_candidates':{'raw':observation}}
        prior={'fit_split':'train','training_binding':binding,'mock_only':True}
        prior['shared_prior_payload_sha256']=canonical_sha256(prior);dump(folder/'shared_background_prior.json',prior)
        prior_record={'filename':'shared_background_prior.json','file_sha256':sha(folder/'shared_background_prior.json'),'payload_sha256':prior['shared_prior_payload_sha256']}
        context={'format':'r5_fold_context_v1','config':config,'config_sha256':canonical_sha256(config),
            'training_binding':binding,'prior':prior_record,'code_sha256':hashes}
        dump(folder/'context.json',context);models={}
        for kind in shares:
            directory=folder/'models'/kind;directory.mkdir(parents=True);models[kind]='models/'+kind
            summary={'status':'completed','kind':kind,'training_binding':binding,'observation_policy':observation,'fixture_only':True}
            if kind=='lgbm_physics':
                (directory/'share.txt').write_text('mock tree; not a real trained model\n')
                (directory/'presence.txt').write_text('mock diagnostic; not a real trained model\n')
                summary.update({'artifact_sha256':{p.name:sha(p) for p in directory.iterdir()},'code_sha256':hashes})
            else:
                settings={'fixture':True,'evaluation_batch':2 if kind in {'timemoe_public','dasheng'} else 4}
                weights={'adapter_only':False,'state_dict':{}}
                if kind in {'timemoe_public','dasheng'}:
                    weights={'adapter_only':True,'adapter':{'manifest':{'fixture_mode':True,'pretrained_claim':False,
                        'family':'dasheng_1.2b' if kind=='dasheng' else 'timemoe_public_200m_active_453m_total'},'state':{}}}
                checkpoint={'format':'r5_neural_checkpoint_v1','kind':kind,'training_binding':binding,
                    'shared_prior_sha256':prior_record['payload_sha256'],'observation_policy':observation,
                    'model_code_sha256':hashes,'weights':weights,'training_settings':settings,'epoch':0}
                dump(directory/'best.pt',checkpoint)
                summary.update({'settings':settings,'best_epoch':0,'checkpoint_sha256':sha(directory/'best.pt')})
            dump(directory/'summary.json',summary)
        summary={'status':'completed','fold':number,'training_binding':binding,'selected_observation':'raw','models':models,
            'frozen_policy_sha256':sha(folder/'frozen_ensemble_policy.json'),'context_sha256':sha(folder/'context.json'),
            'outer_changed_policy':False,'Test_accessed':False,'official_submission':False,
            'artifact_sha256':{p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}}
        dump(folder/'summary.json',summary);entries.append(summary)
    dump(run/'run_state.json',{'status':'completed','folds':entries,'outer_selected_any_policy':False,'fixture_only':nfolds==1,
        'config_sha256':canonical_sha256(config),'Test_accessed':False,'official_submission':False})
    return project,run

def refresh(run):
    state=json.loads((run/'run_state.json').read_text());rows=[]
    for entry in state['folds']:
        folder=run/f'fold_{entry["fold"]:02d}';s=json.loads((folder/'summary.json').read_text())
        s['artifact_sha256']={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and p!=folder/'summary.json'}
        s['context_sha256']=sha(folder/'context.json');s['frozen_policy_sha256']=sha(folder/'frozen_ensemble_policy.json')
        dump(folder/'summary.json',s);rows.append(s)
    state['folds']=rows;dump(run/'run_state.json',state)

def main():
    # No actual project CSV/array or checkpoint may be read by this fixture.
    def protect(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            p=Path(args[0]).resolve()
            if str(p).startswith(str(ROOT)) and p.suffix.lower() in {'.csv','.npy','.npz','.pt','.parquet','.wav','.flac'}:
                raise RuntimeError('Actual project input/model access forbidden in this mock fixture')
    sys.addaudithook(protect)
    with TemporaryDirectory(prefix='r5_frozen_inference_') as tmp:
        base=Path(tmp);project,run=make_fixture(base/'single');backend=MockBackend()
        engine=FrozenInferenceR5(run,project_root=project,backend=backend,_fixture_mode=True)
        db=np.tile(np.linspace(30,60,25),(7,15,1)).astype('float32');db[1,2]=np.nan;db[2,:,::4]=np.nan
        frozen=db.copy();result=engine.predict_db(db,batch_size=3)
        assert result['receipt']['fixture_only'] and len(result['fold_predictions'])==1
        assert backend.maximum_active==1 and backend.loads==['timemoe_public','dasheng','bg_cnn'] and backend.fit_calls==0
        assert [v['batch_size'] for v in result['receipt']['neural_query_batches']]==[2,2,3]
        one=np.concatenate([engine.predict_db(db[i:i+1])['share'] for i in range(len(db))])
        assert np.array_equal(one,result['share']) and np.array_equal(db,frozen,equal_nan=True)
        reversed_result=engine.predict_db(db[::-1].copy(),batch_size=1)
        assert np.array_equal(reversed_result['share'],result['share'][::-1])
        assert np.array_equal(reversed_result['score'],result['score'][::-1])
        reject(lambda:engine.predict_db({'db':db,'share':np.zeros(len(db))}),'db only')
        reject(lambda:engine.predict_db(np.full_like(db,np.nan)))
        reject(lambda:FrozenInferenceR5(run,project_root=project),'fixture run')
        state_path=run/'run_state.json';old_state=state_path.read_bytes();state=json.loads(old_state);state['fixture_only']=False;dump(state_path,state)
        reject(lambda:FrozenInferenceR5(run,project_root=project),'exactly completed folds');state_path.write_bytes(old_state)
        reject(lambda:FrozenInferenceR5(run,project_root=project,backend=MockBackend()),'cannot inject')
        state=json.loads(old_state);state['config_sha256']='0'*64;dump(state_path,state)
        reject(lambda:FrozenInferenceR5(run,project_root=project,backend=MockBackend(),_fixture_mode=True),'same frozen configuration')
        state_path.write_bytes(old_state)
        # Full fivefold metadata still rejects mock/fixture model summaries.
        pp,rr=make_fixture(base/'five',nfolds=5)
        reject(lambda:FrozenInferenceR5(rr,project_root=pp),'production models')
        # Binary SHA tampering fails before any predictions.
        target=run/'fold_00/models/bg_cnn/best.pt';old=target.read_bytes();target.write_bytes(old+b' ')
        reject(lambda:engine.predict_db(db),'changed');target.write_bytes(old)
        # Rehash the outer file ledger but leave the policy's internal SHA bad.
        p=run/'fold_00/frozen_ensemble_policy.json';old=p.read_bytes();v=json.loads(old);v['weights']['bg_cnn']+=.01;dump(p,v);refresh(run)
        reject(lambda:FrozenInferenceR5(run,project_root=project,backend=MockBackend(),_fixture_mode=True),'policy sha')
        p.write_bytes(old);refresh(run)
        # A rehashed ledger cannot hide an invalid prior payload or context bind.
        p=run/'fold_00/shared_background_prior.json';old=p.read_bytes();v=json.loads(old);v['mock_only']=False;dump(p,v)
        cp=run/'fold_00/context.json';old_context=cp.read_bytes();c=json.loads(old_context);c['prior']['file_sha256']=sha(p);dump(cp,c);refresh(run)
        reject(lambda:FrozenInferenceR5(run,project_root=project,backend=MockBackend(),_fixture_mode=True),'prior payload')
        p.write_bytes(old);cp.write_bytes(old_context);refresh(run)
        c=json.loads(old_context);c['training_binding']['fold']=1;dump(cp,c);refresh(run)
        reject(lambda:FrozenInferenceR5(run,project_root=project,backend=MockBackend(),_fixture_mode=True),'binding')
        cp.write_bytes(old_context);refresh(run)
        # Synthetic completed-epoch receipt must match the best checkpoint.
        dynamic_dir=base/'dynamic';dynamic_dir.mkdir();binding=engine.folds[0]['context']['training_binding']
        identity={'binding':binding,'fixture_only':True}
        epoch_receipt={'epoch':0,'examples':100,'clean_primary_once':True,'heldout_arrays_opened':False,'source_identity_sha256':canonical_sha256(identity)}
        dpath=dynamic_dir/'dynamic_source_receipts.json'
        dump(dpath,{'source_identity':identity,'completed_epochs':{'0':epoch_receipt}})
        checkpoint={'dynamic_source_identity':identity,'best_epoch_source_receipt':epoch_receipt}
        dynamic_summary={'dynamic_source_identity':identity,'dynamic_source_receipts_file':dpath.name,'best_epoch':0}
        engine._dynamic_receipt(checkpoint,dynamic_summary,dynamic_dir,binding)
        bad=deepcopy(checkpoint);bad['best_epoch_source_receipt']['examples']=99
        reject(lambda:engine._dynamic_receipt(bad,dynamic_summary,dynamic_dir,binding),'complete matching')
        # Explicit failed/partial state never falls back to a usable model.
        sp=run/'run_state.json';old=sp.read_bytes();state=json.loads(old);state['status']='running';dump(sp,state)
        reject(lambda:FrozenInferenceR5(run,project_root=project,backend=MockBackend(),_fixture_mode=True),'complete frozen')
        sp.write_bytes(old);(run/'fold_00/FAILED.json').write_text('{}')
        reject(lambda:FrozenInferenceR5(run,project_root=project,backend=MockBackend(),_fixture_mode=True),'failed')
        (run/'fold_00/FAILED.json').unlink()
        # Safe path guard remains effective even if summaries are rehashed.
        fs=run/'fold_00/summary.json';old=fs.read_bytes();s=json.loads(old);s['models']['bg_cnn']='../escape';dump(fs,s);refresh(run)
        reject(lambda:FrozenInferenceR5(run,project_root=project,backend=MockBackend(),_fixture_mode=True),'escaping')
        fs.write_bytes(old);refresh(run)
        # CSV schema only: the file below is generated here, never public Test.
        path=base/'synthetic_query.csv';ids=['001','vessel_x','03','D','e','f','g']
        with path.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['ID',*FEATURE_COLUMNS])
            for name,row in zip(ids,db.reshape(len(db),-1)):writer.writerow([name,*['' if not np.isfinite(v) else str(float(v)) for v in row]])
        got,x,receipt=read_query_csv(path,expected_rows=None)
        assert got==ids and np.array_equal(x,db,equal_nan=True) and receipt['raw_query_csv_sha256']==sha(path)
        text=path.read_text();path.write_text(text.replace('m01_TOL_31.6','unexpected_feature',1))
        reject(lambda:read_query_csv(path,expected_rows=None),'375');path.write_text(text)
        rows=list(csv.reader(path.open()));rows[2][0]=rows[1][0]
        with path.open('w',newline='') as f:csv.writer(f).writerows(rows)
        reject(lambda:read_query_csv(path,expected_rows=None),'unique')
        report={'passed':True,'scope':'singlefold synthetic/mock backend only; no actual models/training data/public Test read',
            'code_sha256':{'pipeline/inference_r5.py':sha(ROOT/'pipeline/inference_r5.py'),'ops/verify_frozen_inference_r5.py':sha(Path(__file__))},
            'query_shape':[7,15,25],'singlefold_fixture_only':True,'production_exactly_fivefold_required':True,
            'production_fixture_models_rejected':True,'production_backend_injection_rejected':True,
            'completed_artifact_SHA_policy_SHA_binding_context_prior_checkpoint_checks_exercised':True,
            'runstate_common_config_SHA_guard_exercised':True,
            'changed_checkpoint_and_internal_policy_rejected':True,'failed_partial_and_escaping_path_rejected':True,
            'query_db_only_and_no_label_dictionary':True,'query_chunk_single_row_and_permutation_bitwise':True,
            'original_query_array_and_NaN_unchanged':True,'selected_positive_weight_models_only':True,
            'maximum_simultaneous_mock_backbones':backend.maximum_active,'actual_large_backbones_loaded':False,
            'saved_evaluation_batch_cap_preserved':True,'dynamic_best_epoch_TRAIN_receipt_checks':True,
            'mock_model_loads_first_query':['timemoe_public','dasheng','bg_cnn'],'fit_calls':0,
            'canonical_375_header_schema_ID_order_unique_and_NaN_CSV_checks':True,
            'raw_query_CSV_SHA_recorded':True,'public_Test_accessed':False,'GPU_used':False,
            'production_fivefold_inference_or_numeric_accuracy_proven':False,
            'real_run_integration_remaining':'Requires completed production fivefold run; parent trainer currently under integration testing.'}
        out=ROOT/'reports/R5_FROZEN_INFERENCE_FIXTURE.json';dump(out,report)
        print(json.dumps({'passed':True,'report':str(out),'code_sha256':report['code_sha256'],'public_Test_accessed':False},indent=2))

if __name__=='__main__':main()
