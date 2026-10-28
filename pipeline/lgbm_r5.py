"""R5 physics/retrieval tree branch in a process isolated from Torch."""
from pathlib import Path
import json
import time
import numpy as np
from pipeline.grid_inputs_r5 import canonical_sha256,file_sha256
from pipeline.observation_augmentation_r5 import prepare_query_observations
from pipeline.mixtures import fixed_features,physics_features,retrieval_features
from pipeline.physics_features_r5 import (fit_physics_pca,physics_features_r5,
    save_physics_pca,load_physics_pca)
CODE_NAMES=('lgbm_r5.py','physics_features_r5.py','mixtures.py','observation_augmentation_r5.py','train_lgbm.py','prototypes.py','grid_inputs_r5.py')


def feature_binding(binding,policy):
    return {**binding,'quantization_decimals':policy['quantization_decimals'],
            'preprocessing_policy_sha256':canonical_sha256(policy)}


def transform(db,policy,pca,binding,prototypes):
    feature=prepare_query_observations(db,quantization_decimals=policy['quantization_decimals'])['feature_db']
    parts=[fixed_features(feature,version=3),physics_features(feature),
           retrieval_features(feature,prototypes,version=3),
           physics_features_r5(feature,pca,expected_binding=feature_binding(binding,policy))]
    result=np.concatenate(parts,axis=1)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite LightGBM features')
    return result


def fit_tree(train,inner,binding,policy,settings,output,*,fixture=False):
    from lightgbm import LGBMRegressor,LGBMClassifier,early_stopping,log_evaluation
    from pipeline.train_lgbm import training_prototypes
    output=Path(output)
    if output.exists():raise FileExistsError('Fresh tree output required')
    output.mkdir(parents=True);begin=time.monotonic()
    b=feature_binding(binding,policy)
    feature=prepare_query_observations(train['db'],quantization_decimals=policy['quantization_decimals'])['feature_db']
    pca=fit_physics_pca(feature,b,split='train');pca_record=save_physics_pca(output/'physics_pca.json',pca,expected_binding=b)
    # Use positive TRAIN excess-energy shapes, never whole low-share mixtures.
    proto_record=training_prototypes({'db':feature,'presence':train['presence']},output/'prototypes.npz',
                                     binding['data_package_hash'],binding['seed'],feature_version=3)
    X=transform(train['db'],policy,pca,binding,output/'prototypes.npz')
    V=transform(inner['db'],policy,pca,binding,output/'prototypes.npz')
    minimum=3 if fixture else 50
    reg=LGBMRegressor(objective='regression_l2',n_estimators=settings['maximum_trees'],learning_rate=.035,
        num_leaves=31,min_child_samples=minimum,colsample_bytree=.85,subsample=.85,subsample_freq=1,
        max_bin=127,n_jobs=settings['threads'],random_state=binding['seed'],verbosity=-1)
    reg.fit(X,train['share'],eval_set=[(V,inner['share'])],
            callbacks=[early_stopping(settings['patience'],verbose=False),log_evaluation(100)])
    clf=LGBMClassifier(n_estimators=min(300,settings['maximum_trees']),num_leaves=15,
        min_child_samples=minimum,max_bin=127,n_jobs=settings['threads'],random_state=binding['seed'],verbosity=-1)
    clf.fit(X,train['presence'])
    reg.booster_.save_model(str(output/'share.txt'));clf.booster_.save_model(str(output/'presence.txt'))
    prediction={'share':np.clip(reg.predict(V),0,1).astype('float32'),
                'score':clf.predict_proba(V)[:,1].astype('float32')}
    np.savez_compressed(output/'inner_predictions.npz',**prediction)
    summary={'status':'completed','kind':'lgbm_physics','training_binding':binding,'observation_policy':policy,
             'n_features':X.shape[1],'best_iteration':int(reg.best_iteration_),'maximum_trees':settings['maximum_trees'],
             'best_inner_rmse':float(np.sqrt(np.mean((prediction['share']-inner['share'])**2))),
             'physics_pca':pca_record,'prototype':proto_record,'fixture_only':fixture,
             'training_observation_dropout':False,'test_site_proportions_used':False,
             'outer_or_Test_accessed':False,'elapsed_seconds':time.monotonic()-begin,
             'artifact_sha256':{name:file_sha256(output/name) for name in ['share.txt','presence.txt','prototypes.npz','physics_pca.json','inner_predictions.npz']},
             'code_sha256':{'pipeline/'+name:file_sha256(Path(__file__).with_name(name)) for name in CODE_NAMES}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary,prediction


def predict_tree(db,binding,policy,model_dir,*,allow_fixture=False,batch_size=256):
    from lightgbm import Booster
    if isinstance(db,dict):raise TypeError('Tree query accepts db only, never a labelled dictionary')
    db=np.asarray(db)
    if db.ndim!=3 or db.shape[1:]!=(15,25) or not len(db) or not isinstance(batch_size,int) or batch_size<1:
        raise ValueError('Nonempty [N,15,25] observations and a positive prediction batch required')
    root=Path(__file__).resolve().parents[1];folder=Path(model_dir)
    summary=json.loads((folder/'summary.json').read_text())
    if summary.get('status')!='completed' or summary.get('kind')!='lgbm_physics' or summary.get('fixture_only') and not allow_fixture:
        raise ValueError('Completed production tree required')
    if summary['training_binding']!=binding or summary['observation_policy']!=policy:raise ValueError('Tree data/preprocessing binding differs')
    if set(summary.get('code_sha256',{}))!={'pipeline/'+name for name in CODE_NAMES}:raise ValueError('Complete tree preprocessing code closure required')
    for name,h in summary['artifact_sha256'].items():
        if file_sha256(folder/name)!=h:raise ValueError('Tree artifact changed: '+name)
    for name,h in summary['code_sha256'].items():
        if file_sha256(root/name)!=h:raise ValueError('Tree feature code changed: '+name)
    record=summary['physics_pca']
    pca=load_physics_pca(folder/record['filename'],expected_binding=feature_binding(binding,policy),
        expected_file_sha256=record['file_sha256'],expected_payload_sha256=record['payload_sha256'])
    reg=Booster(model_file=str(folder/'share.txt'));clf=Booster(model_file=str(folder/'presence.txt'))
    result={'share':[],'score':[]}
    for begin in range(0,len(db),batch_size):
        x=transform(db[begin:begin+batch_size],policy,pca,binding,folder/'prototypes.npz')
        result['share'].append(np.clip(reg.predict(x),0,1).astype('float32'))
        result['score'].append(np.clip(clf.predict(x),0,1).astype('float32'))
    return {k:np.concatenate(v) for k,v in result.items()}


def main():
    import argparse
    from pipeline.fold_data_r5 import FoldDataR5
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['fit','predict'])
    parser.add_argument('--context',required=True,type=Path);parser.add_argument('--observation',required=True)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--role',choices=['inner','outer'],default='inner');parser.add_argument('--frozen-policy',type=Path)
    a=parser.parse_args();context=json.loads(a.context.read_text());config=context['config']
    if canonical_sha256(config)!=context['config_sha256']:raise ValueError('Run configuration changed')
    for name,h in context['code_sha256'].items():
        if file_sha256(Path(__file__).resolve().parents[1]/name)!=h:raise ValueError('Run code snapshot changed: '+name)
    f=FoldDataR5(context['data_fold'],config['seed'])
    if f.binding!=context['training_binding']:raise ValueError('Data binding differs')
    policy=config['observation_candidates'][a.observation]
    if a.action=='fit':
        train,_=f.load('train');inner,_=f.load('inner')
        fit_tree(train,inner,f.binding,policy,config['tree'],a.output)
    else:
        if a.model_dir is None:raise ValueError('Explicit tree model required')
        frozen=json.loads(a.frozen_policy.read_text()) if a.frozen_policy else None
        query,_=f.load(a.role,frozen_policy=frozen)
        prediction=predict_tree(query['db'],f.binding,policy,a.model_dir)
        if a.output.exists():raise FileExistsError('Prediction output already exists')
        a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**prediction)


if __name__=='__main__':main()
