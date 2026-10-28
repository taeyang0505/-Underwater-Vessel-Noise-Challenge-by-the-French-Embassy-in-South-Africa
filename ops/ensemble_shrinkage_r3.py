"""One predeclared CPU-only R3 candidate. Freeze all INNER policies before OOF."""
from pathlib import Path
import argparse,hashlib,io,json,time,zipfile
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'ops/server_backups/r2_snapshot_20261024T203941_205747Z_b215c38d.zip'
EXPECTED_ARCHIVE_SHA='7d45b9e6cc1c34e2a753cca269944780de09a886241f13462ab9c3782739bc9a'
OUT=ROOT/'models/ensemble_shrinkage_r3'
MODELS=['lgbm','lgbm_physics','lgbm_retrieval','tabm','cnn','fdy','itransformer','tabicl','kumo_large']
DECLARATION={'experiment':'ensemble_shrinkage_r3','candidate_count':1,'lambda':.05,
 'objective':'mean((P@w-y)**2) + 0.05*sum((w-uniform9)**2)',
 'constraints':'w>=0; sum(w)=1','affine':[1.,0.],'models':MODELS,'folds':list(range(5)),
 'seeds':[42],'fit_split':'each fold INNER validation only','outer_used_for_fit':False,
 'lambda_search':False,'GPU_used':False,'R2_or_dataset_modifications':False,
 'comparison_scope':'post-baseline development OOF comparison; not a new untouched holdout'}

def digest_bytes(b):return hashlib.sha256(b).hexdigest()
def digest(p):return digest_bytes(Path(p).read_bytes())
def write(p,m):p.write_text(json.dumps(m,indent=2,allow_nan=False))
def array_sha(x):return digest_bytes(np.ascontiguousarray(x).tobytes())

def fit():
 if OUT.exists():raise FileExistsError('One fresh candidate only; existing policies are immutable')
 OUT.mkdir(parents=True);write(OUT/'predeclared_experiment.json',dict(DECLARATION,declared_unix=time.time(),code_sha256=digest(__file__)))
 if digest(ARCHIVE)!=EXPECTED_ARCHIVE_SHA:raise ValueError('Archive identity differs')
 records=[];policy_hashes={};opened=[]
 with zipfile.ZipFile(ARCHIVE) as z:
  snapshot=json.loads(z.read('__snapshot_manifest.json'));index={r['archive_path']:r for r in snapshot['files']}
  if len(index)!=207:raise ValueError('Expected independently verified 207-file snapshot')
  def read(name):
   if any(s in name.lower() for s in ['outer','oof','test.csv','calibration','sealed']):raise PermissionError('OOF/outer blocked until all five policies freeze')
   b=z.read(name)
   if name in index and digest_bytes(b)!=index[name]['sha256']:raise ValueError('Snapshot SHA mismatch '+name)
   opened.append(name);return b
  for fold in range(5):
   prefix=f'fold_{fold:02d}/';baseline_b=read(prefix+'ensemble_policy.json');baseline=json.loads(baseline_b)
   if baseline['models']!=MODELS or baseline['seeds']!=[42] or baseline['affine']!=[1.,0.] or baseline['outer_used_for_fit']:raise ValueError('R2 policy protocol differs')
   ledger=json.loads(read(prefix+'artifact_ledger.json'))
   expected={prefix+n:s for group in ledger.values() for n,s in group.items()}
   run_name=prefix+'seed_42/run_manifest.json';run_b=read(run_name);run=json.loads(run_b)
   if digest_bytes(run_b)!=expected[run_name]:raise ValueError('Task ledger binding differs')
   gate=run['prebuilt_readiness_manifest'];package=run['data_package'];source=baseline['source_hash']
   if package['data_package_hash']!=baseline['data_package_hash'] or gate['source_hash']!=source or gate['data_package_hash']!=package['data_package_hash']:raise ValueError('Task source/package mismatch')
   if run['arguments']['seed']!=42 or run['test_accessed'] or run['sealed_holdout_accessed'] or not gate['ready_to_train']:raise ValueError('Task run provenance differs')
   data=ROOT/'data/synthetic/group_cv_physics_v2_r1'/f'fold_{fold:02d}'
   if digest(data/'readiness_manifest.json')!=package['readiness_manifest_sha256']:raise ValueError('Local readiness binding differs')
   val_file=data/'validation.npz'
   if digest(val_file)!=gate['file_sha256']['validation.npz']:raise ValueError('INNER truth package SHA differs')
   with np.load(val_file) as a:truth=a['share'].copy();presence=a['presence'].copy()
   base_name=prefix+'seed_42/validation_predictions.npz';base_b=read(base_name)
   if digest_bytes(base_b)!=expected[base_name]:raise ValueError('Task prediction ledger differs')
   with np.load(io.BytesIO(base_b)) as a:inside={k:a[k] for k in a.files}
   if not np.array_equal(truth,inside['share_true']) or not np.array_equal(presence,inside['presence_true']):raise ValueError('Task INNER truth order mismatch')
   # Foundation IDs additionally bind the otherwise ID-free task prediction order.
   meta_file=data/'inner_validation_metadata.parquet'
   if digest(meta_file)!=gate['file_sha256'][meta_file.name]:raise ValueError('INNER metadata SHA differs')
   ids=pd.read_parquet(meta_file,columns=['sample_id']).sample_id.to_numpy(dtype=str)
   inputs={base_name:digest_bytes(base_b),run_name:digest_bytes(run_b),prefix+'ensemble_policy.json':digest_bytes(baseline_b)}
   for kind in ['tabicl','kumo_large']:
    fm_name=prefix+kind+'_seed_42_inner/manifest.json';fm_b=read(fm_name);fm=json.loads(fm_b)
    fn=prefix+kind+'_seed_42_inner/validation_predictions.npz';fb=read(fn)
    if any([fm['phase']!='inner',fm['seed']!=42,fm['fold']!=fold,fm['source_hash']!=source,fm['data_package_hash']!=package['data_package_hash'],fm['readiness_manifest_sha256']!=package['readiness_manifest_sha256'],fm['outer_accessed'],not fm['query_independence_checks']['passed']]):raise ValueError('Foundation INNER provenance mismatch')
    if digest_bytes(fm_b)!=expected[fm_name] or digest_bytes(fb)!=expected[fn] or digest_bytes(fb)!=fm['file_sha256']['validation_predictions.npz']:raise ValueError('Foundation ledger/SHA mismatch')
    with np.load(io.BytesIO(fb)) as a:
     if not np.array_equal(a['share_true'],truth) or not np.array_equal(a['presence_true'],presence) or not np.array_equal(a['sample_id'].astype(str),ids):raise ValueError('Foundation INNER truth/ID order mismatch')
     inside[kind+'_share']=a[kind+'_share'].copy()
    inputs.update({fm_name:digest_bytes(fm_b),fn:digest_bytes(fb)})
   P=np.stack([inside[k+'_share'] for k in MODELS],1).astype('float64');y=truth.astype('float64');u=np.full(9,1/9)
   if P.shape!=(len(y),9) or not np.isfinite(P).all() or not np.isfinite(y).all():raise ValueError('Invalid INNER arrays')
   def objective(w):return np.mean((P@w-y)**2)+.05*np.sum((w-u)**2)
   def gradient(w):return 2*P.T@(P@w-y)/len(y)+.1*(w-u)
   result=minimize(objective,u,jac=gradient,method='SLSQP',bounds=[(0.,1.)]*9,constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(9)},options={'maxiter':1000,'ftol':1e-14})
   w=result.x
   if not result.success or w.min()<-1e-10 or abs(w.sum()-1)>1e-10:raise RuntimeError('Simplex optimization failed')
   grad=gradient(w);active=w>1e-8;nu=float(grad[active].mean());kkt=max(float(np.max(abs(grad[active]-nu))),float(np.max(np.maximum(nu-grad[~active],0))) if (~active).any() else 0.)
   if kkt>1e-6:raise RuntimeError('QP KKT check failed')
   policy=dict(DECLARATION,fold=fold,weights=w.tolist(),data_package_hash=package['data_package_hash'],source_hash=source,baseline_policy_sha256=digest_bytes(baseline_b),protocol=baseline['protocol'])
   folder=OUT/f'fold_{fold:02d}';folder.mkdir();path=folder/'ensemble_policy.json';write(path,policy);policy_hashes[str(path.relative_to(OUT))]=digest(path)
   records.append({'fold':fold,'rows':len(y),'data_package_hash':package['data_package_hash'],'source_hash':source,'readiness_sha256':package['readiness_manifest_sha256'],'inner_truth_sha256':array_sha(truth),'inner_sample_ids_sha256':array_sha(ids),'files_sha256':inputs,'inner_rmse':float(np.sqrt(np.mean((P@w-y)**2))),'objective':float(result.fun),'KKT_residual':kkt,'optimizer_success':bool(result.success),'weights':w.tolist()})
 write(OUT/'fit_inputs_manifest.json',{'status':'ALL_FIVE_POLICIES_FROZEN_BEFORE_OOF_OPEN','frozen_unix':time.time(),'declaration_sha256':digest(OUT/'predeclared_experiment.json'),'code_sha256':digest(__file__),'archive_sha256':EXPECTED_ARCHIVE_SHA,'archive_entries':207,'policies_sha256':policy_hashes,'folds':records,'archive_files_opened_during_fit':opened,'outer_OOF_arrays_opened_during_fit':False})
 print(json.dumps({'status':'all five policies frozen','policy_sha256':policy_hashes}))

def metrics(y,p,clean):
 def one(mask):return {'n':int(mask.sum()),'rmse':float(np.sqrt(np.mean((p[mask]-y[mask])**2))),'bias':float(np.mean(p[mask]-y[mask]))} if mask.any() else {'n':0,'rmse':None,'bias':None}
 return {'all':one(np.ones(len(y),dtype=bool)),'excluding_boundary':one(clean)}

def evaluate():
 manifest=json.loads((OUT/'fit_inputs_manifest.json').read_text())
 if manifest['status']!='ALL_FIVE_POLICIES_FROZEN_BEFORE_OOF_OPEN' or len(manifest['policies_sha256'])!=5:raise ValueError('All policies must be frozen first')
 if manifest['code_sha256']!=digest(__file__):raise ValueError('Code changed after policy freeze')
 if (OUT/'evaluation_report.json').exists():raise FileExistsError('Single frozen candidate evaluation already performed')
 policies={}
 for name,value in manifest['policies_sha256'].items():
  if digest(OUT/name)!=value:raise ValueError('Frozen policy tampered')
  policy=json.loads((OUT/name).read_text());policies[policy['fold']]=policy
 write(OUT/'evaluation_started.json',{'started_unix':time.time(),'frozen_fit_manifest_sha256':digest(OUT/'fit_inputs_manifest.json'),'policy_hashes_checked_before_outer_open':True})
 oof=ROOT/'ops/r2_progress/oof_predictions.npz';meta_path=ROOT/'ops/r2_progress/oof_metadata.parquet'
 with np.load(oof) as a:arrays={k:a[k] for k in a.files}
 meta=pd.read_parquet(meta_path)
 if not np.array_equal(arrays['sample_id'].astype(str),meta.sample_id.to_numpy(dtype=str)) or not meta.sample_id.is_unique:raise ValueError('OOF metadata order/ID mismatch')
 column='outer_fold' if 'outer_fold' in meta else 'fold';folds=meta[column].to_numpy(dtype=int)
 if set(folds)!=set(range(5)):raise ValueError('OOF fold membership differs')
 y=arrays['share_true'].astype('float64');P=np.stack([arrays[k+'_share'] for k in MODELS],1).astype('float64');pred=np.empty_like(y)
 clean=~meta.boundary_artifact_candidate.to_numpy(dtype=bool)
 for fold,policy in policies.items():pred[folds==fold]=P[folds==fold]@np.array(policy['weights'])
 baseline=arrays['ensemble_share'].astype('float64');cnn=arrays['cnn_share'].astype('float64')
 if not np.isfinite(pred).all():raise ValueError('Nonfinite R3 predictions')
 results={'R2_baseline':metrics(y,baseline,clean),'R2_CNN':metrics(y,cnn,clean),'R3_shrinkage':metrics(y,pred,clean)}
 per_fold={str(f):{name:metrics(y[folds==f],p[folds==f],clean[folds==f]) for name,p in [('R2_baseline',baseline),('R2_CNN',cnn),('R3_shrinkage',pred)]} for f in range(5)}
 delta=results['R3_shrinkage']['all']['rmse']-results['R2_baseline']['all']['rmse']
 report={'status':'one frozen R3 development comparison completed','results':results,'per_fold':per_fold,'R3_minus_R2_RMSE':delta,'improved_over_R2':delta<0,'automatic_promotion':False,'retry_or_lambda_search':False,'comparison_scope':DECLARATION['comparison_scope'],'fit_manifest_sha256':digest(OUT/'fit_inputs_manifest.json'),'OOF_opened_after_all_five_policies_frozen':True,'oof_prediction_sha256':digest(oof),'oof_metadata_sha256':digest(meta_path),'evaluation_finished_unix':time.time()}
 np.savez_compressed(OUT/'r3_oof_predictions.npz',sample_id=arrays['sample_id'],fold=folds,share_true=y,R3_share=pred)
 write(OUT/'evaluation_report.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['fit','evaluate','all'],default='all');args=parser.parse_args()
 if args.stage in ['fit','all']:fit()
 if args.stage in ['evaluate','all']:evaluate()
