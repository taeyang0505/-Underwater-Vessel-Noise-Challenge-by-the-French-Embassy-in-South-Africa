"""Frozen five-fold R5 inference. CSV IO happens only in explicit CLI `infer`.

No training arrays, query labels, cohort statistics, fitting or submissions.
Production validates a completed five-fold run before opening a query CSV.
Mock backends exist only behind explicit single-fold fixture mode (not CLI).
"""
from pathlib import Path,PurePosixPath
import argparse,csv,gc,hashlib,json
import numpy as np
from pipeline.hierarchical_ensemble_r5 import ALL_CANDIDATES,canonical_sha256,validate_policy,apply_policy

ROOT=Path(__file__).resolve().parents[1]
FREQUENCY_TAGS=('31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5','251.2','316.2','398.1','501.2','631','794.3','1000','1258.9','1584.9','1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3')
FEATURE_COLUMNS=tuple(f'm{m:02d}_TOL_{f}' for m in range(1,16) for f in FREQUENCY_TAGS)
FOUNDATIONS={'timemoe_public','dasheng'}
CHECKPOINT_CODE={'pipeline/'+name for name in ('neural_r5.py','hybrid_models_r5.py','grid_models_r5.py','models.py','foundation_masks_r5.py',
    'grid_inputs_r5.py','mixtures.py','background_prior_features.py','observation_augmentation_r5.py',
    'load_foundation_r5.py','dynamic_passage_r5.py')}
REQUIRED_CODE={'pipeline/runner_r5.py','pipeline/neural_r5.py','pipeline/lgbm_r5.py','pipeline/inference_r5.py',
    'pipeline/fold_data_r5.py','pipeline/grid_inputs_r5.py','pipeline/dynamic_passage_r5.py',
    'pipeline/observation_augmentation_r5.py','pipeline/physics_features_r5.py',
    'pipeline/hierarchical_ensemble_r5.py','pipeline/hybrid_models_r5.py',
    'pipeline/grid_models_r5.py','pipeline/foundation_masks_r5.py','pipeline/load_foundation_r5.py',
    'pipeline/mixtures.py','pipeline/models.py','pipeline/background_prior_features.py'}

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def _json(path):
    def bad(value):raise ValueError('Nonfinite JSON constant forbidden: '+value)
    return json.loads(Path(path).read_text(),parse_constant=bad)

def _safe(root,relative):
    if Path(root).is_symlink():raise ValueError('Symlink artifact root forbidden')
    if not isinstance(relative,str) or not relative or '\\' in relative:raise ValueError('Plain project-relative artifact path required')
    parts=PurePosixPath(relative)
    if parts.is_absolute() or '..' in parts.parts or any(x in {'','.'} for x in parts.parts):raise ValueError('Escaping artifact path forbidden')
    path=Path(root)
    for part in parts.parts:
        path=path/part
        if path.is_symlink():raise ValueError('Symlink artifact forbidden: '+relative)
    if not path.is_file():raise ValueError('Missing regular artifact: '+relative)
    return path

def _query(db):
    if isinstance(db,dict):raise TypeError('Query accepts db only; labels/provenance dictionaries are forbidden')
    x=np.asarray(db)
    if x.ndim!=3 or x.shape[1:]!=(15,25) or not len(x) or not np.issubdtype(x.dtype,np.floating):raise ValueError('Nonempty floating query [N,15,25] required')
    if np.isinf(x).any() or not (np.isfinite(x).any(2).sum(1)>=3).all():raise ValueError('Query needs finite or NaN cells and at least3 observed minutes')
    result=np.ascontiguousarray(x,dtype='float32')
    if np.isinf(result).any():raise ValueError('Query exceeds float32 representable input range')
    return result

def original_energy_weights(db):
    x=db.astype('float64');mask=np.isfinite(x);peak=np.nanmax(x,(1,2),keepdims=True)
    e=np.where(mask,10**((x-peak)/10),0)
    if not np.isfinite(e).all() or not (e[mask]>0).all():raise ValueError('Unsupported observed energy underflow/overflow')
    return (e/e.sum((1,2),keepdims=True)).astype('float32')

class ProductionBackend:
    fixture_backend=False
    def load_prior(self,path,context):
        from pipeline.grid_inputs_r5 import load_shared_prior
        r=context['prior']
        return load_shared_prior(path,expected_binding=context['training_binding'],expected_file_sha256=r['file_sha256'],expected_prior_sha256=r['payload_sha256'])
    def checkpoint(self,path):
        import torch
        return torch.load(path,map_location='cpu',weights_only=True)
    def load_neural(self,kind,settings,cache_dir,device):
        import torch
        from pipeline.neural_r5 import make_neural
        model,_=make_neural(kind,settings,cache_dir=cache_dir)
        return model.to(torch.device(device))
    def predict_neural(self,model,checkpoint,kind,fold,db,device,batch_size):
        from pipeline.neural_r5 import restore_into,predict_neural
        context=fold['context'];prior=fold['prior'];binding=context['training_binding']
        restore_into(model,checkpoint,kind=kind,binding=binding,prior_sha=prior['shared_prior_payload_sha256'],policy=fold['observation'])
        return predict_neural(model,db,prior,binding,fold['observation'],device=device,batch_size=batch_size)
    def predict_tree(self,db,fold,model_dir,batch_size):
        from pipeline.lgbm_r5 import predict_tree
        return predict_tree(db,fold['context']['training_binding'],fold['observation'],model_dir,allow_fixture=False,batch_size=batch_size)
    def release(self,model,device):
        # The engine drops its last live reference before cleanup/empty_cache.
        return None
    def cleanup(self,device):
        gc.collect()
        if str(device).startswith('cuda'):
            import torch
            torch.cuda.empty_cache()

class FrozenInferenceR5:
    def __init__(self,run_dir,*,project_root=ROOT,device='cpu',cache_dir=None,backend=None,_fixture_mode=False):
        if type(_fixture_mode) is not bool:raise TypeError('Explicit fixture boolean required')
        if backend is not None and not _fixture_mode:raise ValueError('Production cannot inject a mock backend')
        if _fixture_mode and (backend is None or getattr(backend,'fixture_backend',False) is not True):raise ValueError('Fixture mode requires an explicit synthetic backend')
        self.run=Path(run_dir);self.root=Path(project_root).resolve();self.device=device;self.cache_dir=cache_dir
        if self.run.is_symlink():raise ValueError('Symlink run forbidden')
        self.run=self.run.resolve();self.fixture_mode=_fixture_mode;self.backend=backend or ProductionBackend();self.snapshot={};self.folds=[]
        state_path=_safe(self.run,'run_state.json');self._remember(state_path);state=_json(state_path)
        if state.get('status')!='completed' or state.get('outer_selected_any_policy') is not False:raise ValueError('Complete frozen run required; no failed/partial run inference')
        if not _fixture_mode and state.get('fixture_only'):raise ValueError('Fixture run is not a production model')
        if not _fixture_mode and (state.get('Test_accessed') is not False or state.get('official_submission') is not False):raise ValueError('Production training run must declare no Test access/submission')
        entries=state.get('folds',[]);numbers=[r.get('fold') for r in entries]
        if any(type(n) is not int for n in numbers):raise ValueError('Integer fold identities required')
        if _fixture_mode:
            if numbers!=[0] or state.get('fixture_only') is not True:raise ValueError('Only declared singlefold0 synthetic fixture supported')
        elif len(numbers)!=5 or set(numbers)!=set(range(5)):raise ValueError('Production requires exactly completed folds0..4')
        if len(set(numbers))!=len(numbers):raise ValueError('Duplicate fold identity')
        if any(self.run.rglob('FAILED.json')):raise ValueError('Failed run artifacts cannot be used for inference')
        for entry in sorted(entries,key=lambda x:x['fold']):self.folds.append(self._fold(entry))
        config_hashes={f['context']['config_sha256'] for f in self.folds}
        if len(config_hashes)!=1 or state.get('config_sha256')!=next(iter(config_hashes)):raise ValueError('Run state and all folds must share the same frozen configuration/seed/dataset/settings')
        self.run_state_sha256=self.snapshot[str(state_path)]

    def _remember(self,path,expected=None):
        actual=file_sha256(path)
        if expected is not None and actual!=expected:raise ValueError('Frozen artifact SHA mismatch: '+str(path))
        self.snapshot[str(path)]=actual;return actual

    def _code(self,mapping):
        if not isinstance(mapping,dict) or not mapping:raise ValueError('Frozen code identity required')
        for name,h in mapping.items():self._remember(_safe(self.root,name),h)

    def _fold(self,entry):
        number=entry['fold'];folder=self.run/f'fold_{number:02d}'
        summary_path=_safe(folder,'summary.json');self._remember(summary_path);summary=_json(summary_path)
        if summary!=entry or summary.get('status')!='completed' or summary.get('outer_changed_policy') is not False:raise ValueError('Fold summary differs from completed run state')
        if summary.get('fixture_only') and not self.fixture_mode:raise ValueError('Fixture fold forbidden')
        ledger=summary.get('artifact_sha256')
        if not isinstance(ledger,dict) or not ledger:raise ValueError('Completed fold artifact ledger required')
        for name,h in ledger.items():self._remember(_safe(folder,name),h)
        for path in folder.rglob('*'):
            if path.is_symlink():raise ValueError('Symlink inside frozen fold')
            if path.is_file() and path.name.endswith(('.tmp','.part','.incomplete')):raise ValueError('Partial artifacts inside completed fold')
            if path.is_file() and path.suffix in {'.pt','.json','.jsonl','.npz','.txt'} and path!=summary_path and path.relative_to(folder).as_posix() not in ledger:raise ValueError('Untracked completed-fold artifact: '+path.name)
        cp=_safe(folder,'context.json');pp=_safe(folder,'frozen_ensemble_policy.json')
        self._remember(cp,summary['context_sha256']);self._remember(pp,summary['frozen_policy_sha256'])
        context=_json(cp);policy=_json(pp);config=context.get('config',{})
        if context.get('format')!='r5_fold_context_v1' or canonical_sha256(config)!=context.get('config_sha256'):raise ValueError('Frozen config/context mismatch')
        if config.get('version')!='r5_training_integration_v1' or config.get('protocol')!='source_group_nested5_vtuad_passage_r4':raise ValueError('Unknown production R5 protocol')
        binding=context.get('training_binding')
        if binding!=summary.get('training_binding') or binding.get('fold')!=number or binding.get('seed')!=config.get('seed'):raise ValueError('Fold/TRAIN binding mismatch')
        validate_policy(policy,binding=binding)
        if not REQUIRED_CODE.issubset(context.get('code_sha256',{})):raise ValueError('Incomplete frozen preprocessing/model code closure')
        self._code(context['code_sha256'])
        observation=config['observation_candidates'][summary['selected_observation']]
        prior_record=context['prior'];prior_path=_safe(folder,prior_record['filename']);self._remember(prior_path,prior_record['file_sha256'])
        prior=_json(prior_path)
        if prior.get('training_binding')!=binding or prior.get('fit_split')!='train':raise ValueError('Shared prior is not from the same frozen TRAIN')
        payload=canonical_sha256({k:v for k,v in prior.items() if k!='shared_prior_payload_sha256'})
        if payload!=prior.get('shared_prior_payload_sha256') or payload!=prior_record['payload_sha256']:raise ValueError('Shared prior payload SHA mismatch')
        selected={}
        for kind,weight in policy['weights'].items():
            if weight<=0:continue
            if kind not in ALL_CANDIDATES or kind not in summary['models']:raise ValueError('Missing selected model path: '+kind)
            directory=_safe(folder,summary['models'][kind]+'/summary.json').parent
            model_summary=_json(directory/'summary.json')
            if model_summary.get('status')!='completed' or model_summary.get('kind')!=kind or model_summary.get('training_binding')!=binding or model_summary.get('observation_policy')!=observation:raise ValueError('Selected model summary/binding mismatch')
            if model_summary.get('fixture_only') is not False and not self.fixture_mode:raise ValueError('Only production models may predict Test')
            if not self.fixture_mode and model_summary.get('outer_or_Test_accessed') is not False:raise ValueError('Selected model training may not access OUTER/Test')
            if kind=='lgbm_physics':
                for name,h in model_summary['artifact_sha256'].items():self._remember(_safe(directory,name),h)
                self._code(model_summary['code_sha256'])
            else:
                path=_safe(directory,'best.pt');self._remember(path,model_summary['checkpoint_sha256'])
                checkpoint=self.backend.checkpoint(path)
                self._inspect_checkpoint(checkpoint,kind,binding,prior_record['payload_sha256'],observation,model_summary)
                if not self.fixture_mode:
                    settings=dict(config['training'])
                    if kind in FOUNDATIONS:settings.update(config['foundation_training_overrides'])
                    if settings!=model_summary.get('settings'):raise ValueError('Selected model training settings differ from frozen config')
                    self._dynamic_receipt(checkpoint,model_summary,directory,binding)
                del checkpoint
            selected[kind]={'directory':directory,'summary':model_summary}
        prior=self.backend.load_prior(prior_path,context)
        return {'fold':number,'folder':folder,'context':context,'policy':policy,'prior':prior,'observation':observation,'selected':selected}

    def _inspect_checkpoint(self,c,kind,binding,prior_sha,observation,summary):
        if c.get('format')!='r5_neural_checkpoint_v1' or c.get('kind')!=kind or c.get('training_binding')!=binding or c.get('shared_prior_sha256')!=prior_sha or c.get('observation_policy')!=observation:raise ValueError('Checkpoint format/kind/TRAIN/prior/preprocessing mismatch')
        if c.get('training_settings')!=summary.get('settings') or c.get('epoch')!=summary.get('best_epoch'):raise ValueError('Checkpoint settings/best epoch mismatch')
        if not self.fixture_mode and (set(c.get('model_code_sha256',{}))!=CHECKPOINT_CODE or c.get('receipt')!=summary.get('receipt')):raise ValueError('Complete checkpoint code/provenance receipt required')
        self._code(c['model_code_sha256'])
        weights=c.get('weights',{})
        if kind in FOUNDATIONS:
            manifest=weights.get('adapter',{}).get('manifest',{})
            wanted_fixture=self.fixture_mode
            if weights.get('adapter_only') is not True or manifest.get('fixture_mode') is not wanted_fixture or manifest.get('pretrained_claim') is not (not wanted_fixture):raise ValueError('Foundation requires an explicitly real pretrained adapter in production; mock adapter only in declared fixture mode')
            expected='dasheng_1.2b' if kind=='dasheng' else 'timemoe_public_200m_active_453m_total'
            if manifest.get('family')!=expected:raise ValueError('Wrong pretrained family/Ultra substitution')
        elif weights.get('adapter_only') is not False:raise ValueError('Invalid core neural checkpoint')

    def _dynamic_receipt(self,checkpoint,summary,directory,binding):
        identity=checkpoint.get('dynamic_source_identity')
        if not isinstance(identity,dict) or identity.get('binding')!=binding or identity!=summary.get('dynamic_source_identity'):raise ValueError('Checkpoint dynamic TRAIN source identity mismatch')
        path=_safe(directory,summary.get('dynamic_source_receipts_file'));self._remember(path);record=_json(path)
        if record.get('source_identity')!=identity:raise ValueError('Dynamic epoch receipt source mismatch')
        epoch=summary['best_epoch'];receipt=record.get('completed_epochs',{}).get(str(epoch))
        if not isinstance(receipt,dict) or receipt!=checkpoint.get('best_epoch_source_receipt') or receipt.get('epoch')!=epoch or receipt.get('examples')!=binding.get('train_examples') or receipt.get('clean_primary_once') is not True or receipt.get('heldout_arrays_opened') is not False or receipt.get('source_identity_sha256')!=canonical_sha256(identity):raise ValueError('Best checkpoint epoch lacks a complete matching TRAIN source receipt')

    def _unchanged(self):
        for name,h in self.snapshot.items():
            if Path(name).is_symlink() or not Path(name).is_file() or file_sha256(name)!=h:raise ValueError('Frozen artifact/code changed during inference: '+name)

    def predict_db(self,db_only,*,batch_size=128):
        if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:raise ValueError('Positive integer query batch required')
        self._unchanged();db=_query(db_only);w=original_energy_weights(db);parts={f['fold']:{} for f in self.folds};loads=[];batches=[]
        for kind in ALL_CANDIDATES:
            selected=[f for f in self.folds if kind in f['selected']]
            if not selected:continue
            if kind=='lgbm_physics':
                for f in selected:parts[f['fold']][kind]=self.backend.predict_tree(db,f,f['selected'][kind]['directory'],batch_size)
                continue
            # Reuse one backbone across fold adapters with identical architecture.
            settings_groups={}
            for f in selected:
                settings=f['context']['config']['model_settings'][kind];key=canonical_sha256(settings)
                settings_groups.setdefault(key,[]).append(f)
            for key,folds in sorted(settings_groups.items()):
                settings=folds[0]['context']['config']['model_settings'][kind]
                model=self.backend.load_neural(kind,settings,self.cache_dir,self.device)
                loads.append({'kind':kind,'folds':[f['fold'] for f in folds],'architecture_sha256':key})
                try:
                    for f in folds:
                        path=f['selected'][kind]['directory']/'best.pt';self._remember(path,f['selected'][kind]['summary']['checkpoint_sha256'])
                        checkpoint=self.backend.checkpoint(path)
                        saved_batch=f['selected'][kind]['summary']['settings'].get('evaluation_batch')
                        if type(saved_batch) is not int or saved_batch<1:raise ValueError('Saved positive evaluation batch required')
                        actual_batch=min(batch_size,saved_batch);batches.append({'fold':f['fold'],'kind':kind,'batch_size':actual_batch,'saved_evaluation_batch':saved_batch})
                        parts[f['fold']][kind]=self.backend.predict_neural(model,checkpoint,kind,f,db,self.device,actual_batch)
                        del checkpoint
                finally:
                    self.backend.release(model,self.device);del model;gc.collect()
                    if hasattr(self.backend,'cleanup'):self.backend.cleanup(self.device)
        fold_predictions=[]
        for f in self.folds:
            branches=parts[f['fold']];shares={};masks={};score=np.zeros(len(db))
            for kind,weight in f['policy']['weights'].items():
                if weight<=0:continue
                pred=branches[kind]
                for name in ['share','score']:
                    a=np.asarray(pred[name])
                    if a.shape!=(len(db),) or not np.isfinite(a).all() or (a<0).any() or (a>1).any():raise ValueError('Invalid branch query prediction: '+kind+'/'+name)
                shares[kind]=pred['share'];score+=weight*pred['score']
                if 'cell_mask' in pred:masks[kind]=pred['cell_mask']
            share=apply_policy(f['policy'],shares,evaluation_split='inference',binding=f['context']['training_binding'],neural_masks=masks,energy_weights=w)
            fold_predictions.append({'fold':f['fold'],'share':share,'score':np.clip(score,0,1)})
        share=np.clip(np.mean([p['share'] for p in fold_predictions],axis=0),0,1).astype('float32')
        score=np.clip(np.mean([p['score'] for p in fold_predictions],axis=0),0,1).astype('float32');self._unchanged()
        return {'share':share,'score':score,'fold_predictions':fold_predictions,'receipt':{
            'format':'frozen_r5_inference_receipt_v1','run_state_sha256':self.run_state_sha256,
            'folds':[f['fold'] for f in self.folds],'fold_policy_sha256':{str(f['fold']):f['policy']['policy_sha256'] for f in self.folds},
            'verified_artifact_sha256':dict(self.snapshot),'model_loads':loads,'neural_query_batches':batches,'maximum_simultaneous_neural_backbones':1 if loads else 0,
            'fold_combination':'singlefold synthetic fixture only' if self.fixture_mode else 'fixed arithmetic mean of all completed folds0..4','fixture_only':self.fixture_mode,
            'query_db_only':True,'query_labels_or_population_statistics_used':False,'fit_executed':False,'official_submission':False,
            'diagnostic_score':'same frozen within-fold model weights, then fixed fold mean; controlled-addition evidence only',
            'runtime_note':'pretrained_r5/runtime on Mac contains macOS binaries; install native runtime_requirements.txt dependencies on Linux'}}

def read_query_csv(path,*,expected_rows=1500):
    """Public schema copied from pipeline.infer/prepare_sa, never discovered by reading Test during implementation."""
    path=Path(path);before=file_sha256(path)
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.reader(f);header=next(reader,None)
        if header!=['ID',*FEATURE_COLUMNS]:raise ValueError('Exactly ID plus375 canonical minute-major band columns required')
        ids=[];values=[]
        for row in reader:
            if len(row)!=376 or not row[0] or row[0].strip()!=row[0]:raise ValueError('Malformed row/ID')
            ids.append(row[0]);values.append([float(v) if v.strip() else np.nan for v in row[1:]])
    if not ids or len(ids)!=len(set(ids)) or expected_rows is not None and len(ids)!=expected_rows:raise ValueError('Expected unique nonempty IDs and the declared query row count')
    db=_query(np.asarray(values,dtype='float32').reshape(-1,15,25))
    if file_sha256(path)!=before:raise ValueError('Query file changed while reading')
    return ids,db,{'raw_query_csv_sha256':before,'rows':len(ids),'canonical_features':375,'ID_order_preserved':True,'NaN_kept_missing':True}

def infer_csv(run_dir,input_csv,output_csv,*,project_root=ROOT,device='cpu',cache_dir=None,batch_size=128):
    output=Path(output_csv);manifest=output.with_suffix('.manifest.json')
    if output.exists() or manifest.exists():raise FileExistsError('Fresh local prediction/manifest paths required')
    engine=FrozenInferenceR5(run_dir,project_root=project_root,device=device,cache_dir=cache_dir)
    ids,db,input_receipt=read_query_csv(input_csv);result=engine.predict_db(db,batch_size=batch_size)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['ID','share_added_vessel','score_added_vessel'])
        writer.writerows(zip(ids,result['share'].tolist(),result['score'].tolist()))
    receipt={**result['receipt'],**input_receipt,'output_sha256':file_sha256(output),
        'inference_code_sha256':file_sha256(Path(__file__)),'official_submission':False}
    manifest.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['infer'])
    p.add_argument('--run-dir',required=True,type=Path);p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',default='cpu');p.add_argument('--cache-dir',type=Path);p.add_argument('--batch-size',type=int,default=128);a=p.parse_args()
    result=infer_csv(a.run_dir,a.input,a.output,device=a.device,cache_dir=a.cache_dir,batch_size=a.batch_size)
    print(json.dumps({'saved':str(a.output),'rows':result['rows'],'official_submission':False}))

if __name__=='__main__':main()
