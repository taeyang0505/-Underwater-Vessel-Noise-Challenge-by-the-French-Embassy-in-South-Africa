"""Build a separate Mac-primary R5 variant from a COMPLETED original fold.

No training/restart/source mutation/GPU/Test IO. Original server-tree results
are retained as provenance only. INNER selects a new policy before any OUTER
array is opened. Reused NN OUTER predictions require the same native package,
checkpoint, prior and observation policy; original ensemble OOF is never reused.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, sys, tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pipeline.inference_r5 import _safe, _json, file_sha256, REQUIRED_CODE, FrozenInferenceR5, original_energy_weights
from pipeline.hierarchical_ensemble_r5 import canonical_sha256, select_inner_policy, validate_policy, apply_policy

NEURAL=('bg_cnn','tfgrid_fdy_conformer','timemoe_public','dasheng','fdy','itransformer')
MODELS=set(NEURAL)|{'lgbm_physics'}
VARIANT='r5_mac_primary_completed_fold_v1'

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')

def array_sha256(a):
    a=np.asarray(a);h=hashlib.sha256(json.dumps({'shape':list(a.shape),'dtype':str(a.dtype)},sort_keys=True,separators=(',',':'),allow_nan=False).encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()

def identity(arrays,metadata,w):
    result={'examples':len(arrays['db']),'sample_id_order_sha256':canonical_sha256(metadata.sample_id.astype(str).tolist()),'raw_db_sha256':array_sha256(arrays['db']),'raw_energy_weights_sha256':array_sha256(w)}
    for name in ['share','presence','cell_share']:result[name+'_sha256']=array_sha256(arrays[name])
    return result

class NativeBackend:
    fixture_backend=False
    def fold(self,path,seed):
        from pipeline.fold_data_r5 import FoldDataR5
        return FoldDataR5(path,seed)
    def checkpoint(self,path):
        import torch
        return torch.load(path,map_location='cpu',weights_only=True)
    def predictions(self,path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    def tree(self,db,binding,observation,path):
        from pipeline.lgbm_r5 import predict_tree
        return predict_tree(db,binding,observation,path,allow_fixture=False)
    def metrics(self,arrays,metadata,predictions):
        from pipeline.runner_r5 import _metrics
        return _metrics(arrays,metadata,predictions)
    def aggregate_metrics(self,parts):
        from pipeline.metrics import evaluate_regression
        return evaluate_regression(np.concatenate([p['truth'] for p in parts]),np.concatenate([p['share'] for p in parts]),np.concatenate([p['presence'] for p in parts]),np.concatenate([p['score'] for p in parts]),
            groups={k:np.concatenate([p[k] for p in parts]) for k in ['site','background_group','reference_group']},
            quality_flags={k:np.concatenate([p[k] for p in parts]) for k in ['boundary_artifact_candidate','urn_8321_nominal_warning']})

def verify_ledger(folder,summary):
    files=summary.get('artifact_sha256')
    if summary.get('status')!='completed' or not isinstance(files,dict) or not files:raise ValueError('A fully completed original fold and full artifact ledger are required; completed children of a partial fold are insufficient')
    checked={}
    for name,h in files.items():
        p=_safe(folder,name)
        if file_sha256(p)!=h:raise ValueError('Completed fold artifact changed: '+name)
        checked[str(p)]=h
    for p in Path(folder).rglob('*'):
        if p.is_symlink():raise ValueError('Symlink inside source fold forbidden')
        if p.name=='FAILED.json':raise ValueError('Failed/partial source fold cannot be promoted')
        if p.is_file() and p.suffix in {'.pt','.json','.jsonl','.npz','.txt'} and p!=Path(folder)/'summary.json' and p.relative_to(folder).as_posix() not in files:raise ValueError('Untracked source fold artifact')
    return checked

def unchanged(checked):
    for name,h in checked.items():
        p=Path(name)
        if p.is_symlink() or not p.is_file() or file_sha256(p)!=h:raise ValueError('Frozen source changed: '+name)

def validate_source(folder,project,backend,fixture):
    folder=Path(folder)
    if folder.is_symlink():raise ValueError('Symlink source fold forbidden')
    folder=folder.resolve();summary_path=_safe(folder,'summary.json');summary=_json(summary_path);checked=verify_ledger(folder,summary);checked[str(summary_path)]=file_sha256(summary_path)
    context_path=_safe(folder,'context.json');context=_json(context_path)
    if file_sha256(context_path)!=summary.get('context_sha256') or context.get('format')!='r5_fold_context_v1' or canonical_sha256(context.get('config'))!=context.get('config_sha256'):raise ValueError('Original context/config identity mismatch')
    cfg=context['config'];binding=context['training_binding'];number=summary['fold']
    if type(number) is not int or number not in range(5) or binding!=summary.get('training_binding') or binding.get('fold')!=number or binding.get('seed')!=cfg.get('seed'):raise ValueError('Original fold/config/TRAIN binding mismatch')
    if summary.get('outer_changed_policy') is not False or summary.get('Test_accessed') is not False or summary.get('official_submission') is not False:raise ValueError('Unsafe original fold flags')
    if set(summary.get('models',{}))!=MODELS:raise ValueError('All six completed NN branches and the original server tree are required')
    engine=object.__new__(FrozenInferenceR5);engine.root=Path(project).resolve();engine.snapshot={};engine.fixture_mode=fixture
    if set(context.get('code_sha256',{}))!=REQUIRED_CODE:raise ValueError('Exact original 17-file R5 code closure required')
    engine._code(context['code_sha256'])
    selection_path=_safe(folder,'preprocessing_selection.json');selection=_json(selection_path);obs=summary['selected_observation']
    if selection.get('fit_split')!='inner' or selection.get('outer_or_Test_used') is not False or selection.get('training_binding')!=binding or selection.get('selected')!=obs:raise ValueError('Original CNN-only preprocessing winner is not frozen')
    candidates=selection['candidates']
    if {p['observation'] for p in candidates}!=set(cfg['observation_candidates']) or min(candidates,key=lambda p:p['inner_rmse'])['observation']!=obs:raise ValueError('Original preprocessing winner differs from predeclared CNN INNER selection')
    winner=next(p for p in candidates if p['observation']==obs)
    if winner['model_path']!=summary['models']['bg_cnn']:raise ValueError('Selected CNN path differs from original CNN pilot winner')
    for item in candidates:
        if file_sha256(_safe(folder,item['model_path']+'/summary.json'))!=item['summary_sha256']:raise ValueError('Original CNN pilot summary changed')
    observation=cfg['observation_candidates'][obs];prior=context['prior'];prior_path=_safe(folder,prior['filename']);p=_json(prior_path)
    if file_sha256(prior_path)!=prior['file_sha256'] or p.get('training_binding')!=binding or p.get('fit_split')!='train' or p.get('shared_prior_payload_sha256')!=prior['payload_sha256'] or canonical_sha256({k:v for k,v in p.items() if k!='shared_prior_payload_sha256'})!=prior['payload_sha256']:raise ValueError('Original TRAIN prior is invalid')
    policy_path=_safe(folder,'frozen_ensemble_policy.json');old_policy=_json(policy_path);validate_policy(old_policy,binding=binding)
    if file_sha256(policy_path)!=summary['frozen_policy_sha256']:raise ValueError('Original policy file changed')
    for kind in NEURAL:
        directory=_safe(folder,summary['models'][kind]+'/summary.json').parent;s=_json(directory/'summary.json')
        if s.get('status')!='completed' or s.get('kind')!=kind or s.get('fixture_only') is not fixture or s.get('training_binding')!=binding or s.get('observation_policy')!=observation or s.get('outer_or_Test_accessed') is not False:raise ValueError('Original NN model is incomplete/unbound')
        for name,key in [('best.pt','checkpoint_sha256'),('inner_predictions.npz','prediction_sha256')]:
            if file_sha256(_safe(directory,name))!=s[key]:raise ValueError('Original NN checkpoint/INNER prediction changed')
        c=backend.checkpoint(directory/'best.pt');engine._inspect_checkpoint(c,kind,binding,prior['payload_sha256'],observation,s)
        if not fixture:engine._dynamic_receipt(c,s,directory,binding)
        # OUTER is hashed here; it is not opened as an array before new freeze.
        _safe(folder,'outer_predictions/'+kind+'.npz')
    checked.update(engine.snapshot)
    return {'folder':folder,'summary':summary,'context':context,'binding':binding,'config':cfg,'observation':observation,'selected_observation':obs,'selection_sha256':file_sha256(selection_path),'checked':checked}

def validate_mac(path,receipt_path,source,project,fixture):
    path=Path(path).resolve();summary=_json(_safe(path,'summary.json'));receipt_path=Path(receipt_path) if receipt_path else path/'producer_summary.json';receipt=_json(receipt_path)
    if receipt.get('format')!='mac_lgbm_r5_producer_v1' or receipt.get('status')!='completed':raise ValueError('A completed explicit Mac producer receipt is required')
    expected={'context_sha256':source['summary']['context_sha256'],'config_sha256':source['context']['config_sha256'],'selection_sha256':source['selection_sha256'],'training_binding':source['binding'],'selected_observation':source['selected_observation'],'observation_policy':source['observation']}
    if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('Mac producer context/selection/observation/TRAIN binding differs')
    if summary.get('status')!='completed' or summary.get('kind')!='lgbm_physics' or summary.get('fixture_only') is not fixture or summary.get('training_binding')!=source['binding'] or summary.get('observation_policy')!=source['observation'] or summary.get('outer_or_Test_accessed') is not False:raise ValueError('Invalid Mac production tree summary')
    server=_json(_safe(source['folder'],source['summary']['models']['lgbm_physics']+'/summary.json'))
    if summary.get('code_sha256')!=server.get('code_sha256'):raise ValueError('Mac/server tree preprocessing code closures differ')
    checked={str(_safe(path,'summary.json')):file_sha256(path/'summary.json'),str(receipt_path):file_sha256(receipt_path)}
    for name,h in summary['artifact_sha256'].items():
        p=_safe(path,name)
        if file_sha256(p)!=h:raise ValueError('Mac tree artifact SHA mismatch')
        checked[str(p)]=h
    for name,h in summary['code_sha256'].items():
        p=_safe(project,name)
        if file_sha256(p)!=h:raise ValueError('Mac tree code changed')
        checked[str(p)]=h
    if summary.get('maximum_trees')!=source['config']['tree']['maximum_trees']:raise ValueError('Mac tree hyperparameter policy differs')
    if receipt.get('output_summary_sha256') not in {None,checked[str(path/'summary.json')]}:raise ValueError('Mac output summary receipt differs')
    return path,summary,receipt,receipt_path,checked

def predictions_ok(pred,n,w,neural=False):
    for k in ['share','score']:
        a=np.asarray(pred[k])
        if a.shape!=(n,) or not np.isfinite(a).all() or (a<0).any() or (a>1).any():raise ValueError('Invalid saved branch prediction')
    if neural:
        q=np.asarray(pred.get('cell_mask'))
        if q.shape!=w.shape or not np.isfinite(q).all() or (q<0).any() or (q>1).any() or not np.allclose((q*w).sum((1,2)),pred['share'],atol=3e-6,rtol=0):raise ValueError('Reused NN mask/share does not match the same raw physical query weights/order')

def ensemble(policy,branches,w,binding,role):
    shares={k:v['share'] for k,v in branches.items()};masks={k:v['cell_mask'] for k,v in branches.items() if 'cell_mask' in v}
    y=apply_policy(policy,shares,evaluation_split=role,binding=binding,neural_masks=masks,energy_weights=w)
    score=np.zeros(len(y))
    for k,v in policy['weights'].items():
        if v>0:score+=v*branches[k]['score']
    return {'share':y.astype('float32'),'score':np.clip(score,0,1).astype('float32')}

def finalize(original_fold,mac_tree,data_fold,output,*,producer_receipt=None,project_root=ROOT,_backend=None,_fixture_mode=False):
    if _backend is not None and not _fixture_mode:raise ValueError('Backend injection is restricted to explicit synthetic fixtures')
    backend=_backend or NativeBackend()
    if _fixture_mode and getattr(backend,'fixture_backend',False) is not True:raise ValueError('Synthetic backend required')
    source=validate_source(original_fold,project_root,backend,_fixture_mode)
    mac,mac_summary,producer,receipt_path,mac_checked=validate_mac(mac_tree,producer_receipt,source,project_root,_fixture_mode)
    f=backend.fold(data_fold,source['config']['seed'])
    if f.binding!=source['binding']:raise ValueError('Native local TRAIN/package/source/fold binding differs from original run')
    inner,meta=f.load('inner');w=original_energy_weights(inner['db']);inner_id=identity(inner,meta,w)
    if producer.get('inner_identity')!=inner_id:raise ValueError('Mac INNER input/ID/order/energy/target identity mismatch')
    branches={k:backend.predictions(source['folder']/source['summary']['models'][k]/'inner_predictions.npz') for k in NEURAL}
    branches['lgbm_physics']=backend.predictions(mac/'inner_predictions.npz')
    for k,p in branches.items():predictions_ok(p,len(inner['share']),w,k in NEURAL)
    cfg=source['config'];policy=select_inner_policy(inner['share'],{k:v['share'] for k,v in branches.items()},meta.reference_group.to_numpy(),fit_split='inner',binding=f.binding,explicit_exclusions={},enable_logit_mask_candidate=cfg['ensemble']['enable_logit_mask_candidate'],neural_masks={k:p['cell_mask'] for k,p in branches.items() if k in NEURAL},energy_weights=w)
    output=Path(output)
    if output.exists() or output.is_symlink():raise FileExistsError('Fresh Mac-primary fold output required')
    if output.name!=f"fold_{source['summary']['fold']:02d}":raise ValueError('Output folder must preserve fold_XX identity')
    if output.resolve().is_relative_to(source['folder']) or output.resolve().is_relative_to(mac):raise ValueError('Variant must remain outside source folders')
    output.mkdir(parents=True)
    try:
        old_tree=source['summary']['models']['lgbm_physics'];exclude={'summary.json','frozen_ensemble_policy.json','inner_metrics.json','outer_metrics.json','outer_oof.npz','outer_predictions/lgbm_physics.npz'}
        for name in source['summary']['artifact_sha256']:
            if name in exclude:continue
            target=output/('provenance/original_server_tree/'+name[len(old_tree)+1:] if name.startswith(old_tree+'/') else name)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(_safe(source['folder'],name),target)
        dump(output/'provenance/original_fold_summary.json',source['summary'])
        tree_out=output/'models/lgbm_physics';tree_out.mkdir(parents=True)
        for name in ['summary.json',*mac_summary['artifact_sha256']]:shutil.copyfile(_safe(mac,name),tree_out/name)
        shutil.copyfile(receipt_path,tree_out/'producer_summary.json')
        # This exclusive on-disk commit is the OUTER access boundary.
        dump(output/'frozen_ensemble_policy.json',policy);policy_file_sha=file_sha256(output/'frozen_ensemble_policy.json')
        dump(output/'policy_freeze_receipt.json',{'fit_split':'inner','training_binding':f.binding,'policy_sha256':policy['policy_sha256'],'policy_file_sha256':policy_file_sha,'outer_arrays_opened_before_freeze':False,'original_CNN_observation_selection_reused_unchanged':True,'Mac_tree_used_for_observation_selection':False})
        inner_predictions=ensemble(policy,branches,w,f.binding,'inner');inner_metrics={k:backend.metrics(inner,meta,p) for k,p in branches.items()};inner_metrics['ensemble']=backend.metrics(inner,meta,inner_predictions);dump(output/'inner_metrics.json',inner_metrics)
        if file_sha256(output/'frozen_ensemble_policy.json')!=policy_file_sha:raise ValueError('New INNER policy changed before OUTER access')
        # No OUTER NPZ is loaded before this line, including original NN files.
        outer,outer_meta=f.load('outer',frozen_policy=policy);ow=original_energy_weights(outer['db'])
        outer_branches={k:backend.predictions(output/'outer_predictions'/f'{k}.npz') for k in NEURAL}
        outer_branches['lgbm_physics']=backend.tree(outer['db'],f.binding,source['observation'],tree_out)
        for k,p in outer_branches.items():predictions_ok(p,len(outer['share']),ow,k in NEURAL)
        np.savez_compressed(output/'outer_predictions/lgbm_physics.npz',**outer_branches['lgbm_physics'])
        out_pred=ensemble(policy,outer_branches,ow,f.binding,'outer');outer_metrics={k:backend.metrics(outer,outer_meta,p) for k,p in outer_branches.items()};outer_metrics['ensemble']=backend.metrics(outer,outer_meta,out_pred);dump(output/'outer_metrics.json',outer_metrics)
        np.savez_compressed(output/'outer_oof.npz',**out_pred,truth=outer['share'],presence=outer['presence'],sample_id=outer_meta.sample_id.to_numpy(dtype=str),**{k:outer_meta[k].to_numpy(dtype=bool if k in {'boundary_artifact_candidate','urn_8321_nominal_warning'} else str) for k in ['boundary_artifact_candidate','urn_8321_nominal_warning','site','background_group','reference_group']})
        dump(output/'executor_provenance.json',{'format':VARIANT,'status':'completed','original_fold_summary_sha256':file_sha256(source['folder']/'summary.json'),'original_context_sha256':source['summary']['context_sha256'],'Mac_producer_receipt_sha256':file_sha256(receipt_path),'Mac_tree_summary_sha256':file_sha256(mac/'summary.json'),'source_NN_retrained':False,'original_server_tree_final_candidate':False,'original_server_tree_baseline_preserved':'provenance/original_server_tree','native_inner_identity':inner_id,'native_outer_identity':identity(outer,outer_meta,ow),'NN_prediction_row_order_evidence':'Original prediction NPZ has no embedded IDs; native package/metadata SHA plus frozen runner/predictor preserves order, additionally checked by physical mask/share pooling.','outer_used_for_policy_selection':False,'Test_accessed':False,'official_submission':False,'finalizer_sha256':file_sha256(__file__),'fixture_only':_fixture_mode})
        unchanged(source['checked']);unchanged(mac_checked)
        summary={'status':'completed','fold':source['summary']['fold'],'training_binding':f.binding,'selected_observation':source['selected_observation'],'models':{**{k:source['summary']['models'][k] for k in NEURAL},'lgbm_physics':'models/lgbm_physics'},'frozen_policy_sha256':policy_file_sha,'context_sha256':file_sha256(output/'context.json'),'inner_rmse':inner_metrics['ensemble']['rmse'],'outer_rmse':outer_metrics['ensemble']['rmse'],'outer_rmse_excluding_boundary':outer_metrics['ensemble'].get('oof_rmse_excluding_boundary'),'outer_changed_policy':False,'Test_accessed':False,'official_submission':False,'executor_variant':VARIANT,'fixture_only':_fixture_mode}
        summary['artifact_sha256']={p.relative_to(output).as_posix():file_sha256(p) for p in output.rglob('*') if p.is_file()}
        dump(output/'summary.json',summary);return summary
    except BaseException as e:
        dump(output/'FAILED.json',{'status':'failed','error':type(e).__name__,'message':str(e),'partial_reuse_allowed':False});raise

def aggregate(run,*,_backend=None,_fixture_mode=False):
    if _backend is not None and not _fixture_mode:raise ValueError('Synthetic aggregate backend only')
    backend=_backend or NativeBackend();run=Path(run)
    if (run/'run_state.json').exists():raise FileExistsError('Fresh aggregate run_state required')
    summaries=[];contexts=[];parts=[]
    for i in range(5):
        folder=run/f'fold_{i:02d}';s=_json(_safe(folder,'summary.json'));verify_ledger(folder,s)
        if s.get('fold')!=i or s.get('executor_variant')!=VARIANT or s.get('fixture_only') is not _fixture_mode:raise ValueError('Five completed matching Mac-primary folds required')
        ctx=_json(_safe(folder,'context.json'));policy=_json(_safe(folder,'frozen_ensemble_policy.json'));validate_policy(policy,binding=s['training_binding'])
        if file_sha256(folder/'context.json')!=s['context_sha256'] or file_sha256(folder/'frozen_ensemble_policy.json')!=s['frozen_policy_sha256'] or ctx['config_sha256']!=canonical_sha256(ctx['config']):raise ValueError('Fold context/policy changed')
        summaries.append(s);contexts.append(ctx);parts.append(backend.predictions(folder/'outer_oof.npz'))
    if len({c['config_sha256'] for c in contexts})!=1:raise ValueError('All five folds must share original frozen config')
    state={'status':'completed','config_sha256':contexts[0]['config_sha256'],'folds':summaries,'outer_selected_any_policy':False,'Test_accessed':False,'official_submission':False,'executor_variant':VARIANT,'fixture_only':_fixture_mode,'pooled_oof':backend.aggregate_metrics(parts)}
    dump(run/'run_state.json',state);return state

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    f=sub.add_parser('finalize');f.add_argument('--original-fold',type=Path,required=True);f.add_argument('--mac-tree',type=Path,required=True);f.add_argument('--data-fold',type=Path,required=True);f.add_argument('--output',type=Path,required=True);f.add_argument('--producer-receipt',type=Path)
    a=sub.add_parser('aggregate');a.add_argument('--run-dir',type=Path,required=True)
    sub.add_parser('self-test');args=p.parse_args()
    if args.action=='finalize':result=finalize(args.original_fold,args.mac_tree,args.data_fold,args.output,producer_receipt=args.producer_receipt)
    elif args.action=='aggregate':result=aggregate(args.run_dir)
    else:result=self_test()
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
