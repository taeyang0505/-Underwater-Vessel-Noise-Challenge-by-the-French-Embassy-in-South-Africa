"""CPU LightGBM on fingerprinted V2 arrays, isolated from Torch/OpenMP.

Retrieval prototypes are rebuilt from this package's training mixtures only.
No V1 source manifest or external held-out reference bank is read.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from lightgbm import LGBMRegressor,LGBMClassifier,early_stopping,log_evaluation
from pipeline.mixtures import fixed_features,physics_features
from pipeline.prototypes import normalize_shape,features as prototype_features
ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def training_prototypes(data,path,package_hash,seed):
    # Positive controlled mixtures include measured background; these centres
    # are mixture shapes, never claims of isolated vessel or independent hulls.
    x=data['db'][data['presence']>0].astype('float64')
    if not len(x):raise ValueError('Training mixture prototypes need positive training examples')
    peak=np.nanmax(x,(1,2),keepdims=True)
    e=np.where(np.isfinite(x),10**((x-peak)/10),0).sum(1)
    shapes=normalize_shape(e);rng=np.random.default_rng(seed)
    k=min(8,max(1,len(shapes)//3));centres=shapes[rng.choice(len(shapes),k,replace=False)].copy()
    for _ in range(30):
        ids=np.argmax(shapes@centres.T,1)
        centres=np.stack([shapes[ids==j].mean(0) if (ids==j).any() else centres[j] for j in range(k)])
        centres/=np.maximum(np.linalg.norm(centres,axis=1,keepdims=True),1e-12)
    ids=np.argmax(shapes@centres.T,1);counts=np.bincount(ids,minlength=k)
    np.savez_compressed(path,centres=centres.astype('float32'),mixture_counts=counts,manifest_hash=package_hash)
    return {'prototype_mode':'training_positive_mixture_shapes','prototypes':k,'training_examples_used':len(shapes),'data_package_hash':package_hash,'training_only':True,'v1_source_manifest_read':False,'independent_source_group_count':None,'no_individual_clip_retrieval':True,'test_accessed':False}

def main(a):
    gate=json.loads((a.data/'readiness_manifest.json').read_text())
    if gate.get('ready_to_train') is not True:raise ValueError('Data readiness gate is not true')
    if not str(gate.get('version','')).startswith('quality_v2'):raise ValueError('Prepared V2 package required')
    if any(gate.get(k) is not False for k in ['challenge_test_accessed','existing_sealed_NOAA_accessed','sealed_holdout_materialized','frozen_v1_data_modified']):raise ValueError('Package provenance gate failed')
    if any(gate.get(k) is not True for k in ['source_checks_passed','reference_group_and_duplicate_isolation_passed','background_week_and_purge_isolation_passed','sealed_holdout_locked']):raise ValueError('Source isolation gate failed')
    files=gate['file_sha256'];package_hash=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    if package_hash!=gate['data_package_hash'] or (a.data_package_hash and a.data_package_hash!=package_hash):raise ValueError('Prebuilt data package hash mismatch')
    version_dir=Path(gate['source_version_dir'])
    if not version_dir.is_absolute():version_dir=ROOT/version_dir
    if sha256(version_dir/'manifest.json')!=gate['source_version_manifest_sha256']:raise ValueError('Source-version manifest changed')
    splits=['train','validation'] if a.cv_mode else ['train','validation','calibration']
    if a.cv_mode:
        if not gate.get('cv_protocol'):raise ValueError('Explicit cv_protocol required')
    elif (a.data/'domain_holdout.npz').exists():splits.append('domain_holdout')
    if (a.data/'sealed_holdout.npz').exists():raise ValueError('Sealed data is excluded')
    data={}
    for split in splits:
        name=split+'.npz';path=a.data/name
        if name not in files or path.is_symlink() or sha256(path)!=files[name]:raise ValueError('Split SHA-256 mismatch: '+name)
        with np.load(path,allow_pickle=False) as z:data[split]={k:z[k] for k in ['db','share','presence']}
    if len(np.unique(data['train']['presence']))<2:raise ValueError('Presence classifier requires both training classes')
    kind='lgbm' if a.variant=='base' else 'lgbm_'+a.variant
    if any(a.output.glob(kind+'_*')):raise FileExistsError('Model artifacts already exist; never overwrite a trained variant')
    a.output.mkdir(parents=True,exist_ok=True);prototype_path=a.output/'prototypes.npz'
    if a.variant=='retrieval':
        if prototype_path.exists():raise FileExistsError('V2 prototype file already exists')
        info=training_prototypes(data['train'],prototype_path,package_hash,a.seed)
        (a.output/'prototype_manifest.json').write_text(json.dumps(info,indent=2))
    def transform(db):
        parts=[fixed_features(db)]
        if a.variant!='base':parts.append(physics_features(db))
        if a.variant=='retrieval':parts.append(prototype_features(db,prototype_path))
        return np.concatenate(parts,1)
    X=transform(data['train']['db']);V=transform(data['validation']['db'])
    reg=LGBMRegressor(objective='regression_l2',n_estimators=a.trees,learning_rate=.035,num_leaves=31,min_child_samples=50,colsample_bytree=.85,subsample=.85,subsample_freq=1,max_bin=127,n_jobs=a.threads,random_state=a.seed,verbosity=-1)
    reg.fit(X,data['train']['share'],eval_set=[(V,data['validation']['share'])],callbacks=[early_stopping(40,verbose=False),log_evaluation(100)])
    clf=LGBMClassifier(n_estimators=min(300,a.trees),num_leaves=15,max_bin=127,min_child_samples=50,n_jobs=a.threads,random_state=a.seed,verbosity=-1)
    clf.fit(X,data['train']['presence'])
    reg.booster_.save_model(str(a.output/(kind+'_share.txt')));clf.booster_.save_model(str(a.output/(kind+'_presence.txt')))
    for split,d in data.items():
        if split=='train':continue
        x=transform(d['db']);np.savez_compressed(a.output/f'{kind}_{split}.npz',share=np.clip(reg.predict(x),0,1),score=clf.predict_proba(x)[:,1])
    (a.output/f'{kind}_training_summary.json').write_text(json.dumps({'kind':kind,'max_trees':a.trees,'best_iteration':reg.best_iteration_,'selection_metric':'validation_l2','data_package_hash':package_hash,'evaluation_splits':splits[1:],'cv_mode':a.cv_mode,'test_accessed':False,'sealed_holdout_accessed':False},indent=2))
    print('LightGBM complete',reg.best_iteration_,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--trees',type=int,default=700);p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--data-package-hash');p.add_argument('--cv-mode',action='store_true')
    p.add_argument('--variant',choices=['base','physics','retrieval'],default='base');main(p.parse_args())
