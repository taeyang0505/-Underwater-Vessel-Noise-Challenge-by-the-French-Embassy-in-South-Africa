"""Completed-fivefold R13 query-only inference. No fitting/submission action.

Only explicit CLI infer opens the public query CSV, after the whole run has
passed completed-model/artifact/policy checks. Old8 checkpoints, prior-q0,
partial runs and a different tree feature schema are never substituted.
"""
from __future__ import annotations
from pathlib import Path
import argparse,csv,gc,json
import numpy as np
from pipeline.runner_r13 import ROOT,NEURAL,sha,canonical,restore_neural
from pipeline.trees_ensemble_r13 import PhysicsLGBM,apply_frozen_ensemble
from pipeline.features_r12 import load_spectral_basis,transform_channels
from pipeline.inference_r5 import read_query_csv


def _query(db):
    if isinstance(db,dict):raise TypeError('db-only query, never labels/source IDs')
    x=np.asarray(db)
    if (x.ndim!=3 or x.shape[1:]!=(15,25) or not len(x) or x.dtype.kind!='f'
            or np.isinf(x).any() or (np.isfinite(x).all(2).sum(1)<3).any()
            or (np.isfinite(x).any(2)!=np.isfinite(x).all(2)).any()):
        raise ValueError('Finite/NaN full-band minutes, minimum3 observed, required')
    return np.ascontiguousarray(x,np.float32)


class ProductionBackend:
    fixture_backend=False
    def open(self,run):
        from ops.run_r13_training import verify_run
        verified=verify_run(run,allow_fixture=False)
        result=json.loads((run/'results.json').read_text());identity=json.loads((run/'execution_identity.json').read_text())
        if (result.get('all5_background_folds_completed') is not True or identity.get('folds')!=list(range(5))
                or result.get('folds')!=list(range(5))):
            raise ValueError('Five complete production folds0..4 required')
        if identity.get('fixture_only') or (run/'FAILED.json').exists():raise ValueError('No fixture/failed inference')
        cfg=identity['config'];bundle=ROOT/identity['prepared_bundle_dir'];m=json.loads((bundle/'manifest.json').read_text())
        runtime=json.loads((run/'runtime_policy.json').read_text())
        if sha(run/'runtime_policy.json')!=identity['runtime_policy_sha256']:raise ValueError('Runtime recipe changed')
        if m['data_hash']!=identity['data_bundle_hash']:raise ValueError('R13 prepared preprocessing state differs')
        snapshots={str(bundle/'manifest.json'):sha(bundle/'manifest.json')}
        views=[]
        for fold in range(5):
            folder=run/f'fold_{fold:02d}';context=json.loads((folder/'context.json').read_text())
            if context['binding']['config_sha256']!=identity['config_sha256'] or context['binding']['fold']!=fold:
                raise ValueError('Fold/config binding differs')
            basis_entry=m['folds'][str(fold)];bp=bundle/basis_entry['basis_path']
            basis=load_spectral_basis(bp,expected_sha256=basis_entry['basis_sha256'],expected_training_binding=basis_entry['basis_training_binding'])
            policy=json.loads((folder/'ensemble_policy.json').read_text())
            if policy['training_binding']!=context['binding'] or not isinstance(policy.get('score_policy'),dict):
                raise ValueError('Frozen R13 regression/independent-score policy required')
            for kind in NEURAL:
                s=json.loads((folder/'models'/kind/'summary.json').read_text())
                if s['train_identity']['basis_sha256']!=basis_entry['basis_sha256'] or s['architecture']['architecture_version']!=13:
                    raise ValueError('Wrong-fold PCA/legacy neural adapter; no fallback')
            snapshots[str(bp)]=sha(bp);snapshots[str(folder/'ensemble_policy.json')]=sha(folder/'ensemble_policy.json')
            views.append({'fold':fold,'folder':folder,'basis':basis,'include_shape':runtime['include_shape'],
                          'basis_training_binding':basis_entry['basis_training_binding'],
                          'basis_sha256':basis_entry['basis_sha256'],
                          'policy':policy,'binding':context['binding'],'settings':cfg['training']})
        return views,snapshots,verified
    def predict(self,view,db,device,batch):
        import torch
        from torch.utils.data import DataLoader
        from pipeline.training_r13 import predict_loader
        x,w=transform_channels(db,view['basis'],include_shape=view['include_shape'])
        if x.shape[1]==8:x=np.pad(x,((0,0),(0,3),(0,0),(0,0)))
        class Query:
            def __len__(self):return len(x)
            def __getitem__(self,i):return {'x':x[i],'w':w[i]}
        pred={};used=[]
        for kind in NEURAL:
            model,summary=restore_neural(view['folder']/'models'/kind,device=device)
            actual_batch=min(int(batch),summary['settings']['evaluation_batch_size'])
            loader=DataLoader(Query(),batch_size=actual_batch,shuffle=False,num_workers=0)
            pred[kind]=predict_loader(model,loader,device,bf16=summary['settings']['bf16'],collect_targets=False)
            used.append({'kind':kind,'batch_size':actual_batch,'checkpoint_sha256':sha(view['folder']/'models'/kind/'best_window.pt')})
            del model;gc.collect()
            if str(device).startswith('cuda'):torch.cuda.empty_cache()
        tree=PhysicsLGBM.load(view['folder']/'models/lgbm_physics',expected_binding=view['binding'])
        if (tree.summary['include_shape'] is not view['include_shape']
                or tree.summary['basis_training_binding']!=view['basis_training_binding']
                or (tree.include_shape and sha(view['folder']/'models/lgbm_physics/basis.npz')!=view['basis_sha256'])):
            raise ValueError('Tree runtime feature/PCA state differs from the same-fold frozen neural view')
        pred['lgbm_physics']=tree.predict(db,batch_size=batch)
        if tree.summary['n_features']!=5252:raise ValueError('Full R13 tree requires5252 columns; no compact/old fallback')
        result=apply_frozen_ensemble(view['policy'],pred)
        return result,{'fold':view['fold'],'neural_loads':used,'maximum_simultaneous_neural_models':1,
                       'tree_features':5252,'selected_lambda':view['policy']['score_policy']['selected_lambda']}
    def verify_after(self,run):
        from ops.run_r13_training import verify_run
        return verify_run(run,allow_fixture=False)


class FrozenInferenceR13:
    def __init__(self,run,*,device='cpu',backend=None,_fixture=False):
        self.run=Path(run).resolve();self.device=device;self.fixture=_fixture
        if backend is not None and (not _fixture or getattr(backend,'fixture_backend',False) is not True):
            raise ValueError('Production backend injection is forbidden')
        self.backend=backend or ProductionBackend()
        self.views,self.snapshots,self.verified=self.backend.open(self.run)
        if [v['fold'] for v in self.views]!=list(range(5)):raise ValueError('All five frozen fold policies required')
    def predict_db(self,db,*,batch_size=128):
        x=_query(db)
        if type(batch_size)is not int or batch_size<1:raise ValueError('Positive integer query batch required')
        results=[];details=[]
        for view in self.views:
            p,receipt=self.backend.predict(view,x,self.device,batch_size)
            for key in ('share','score'):
                a=np.asarray(p[key])
                if a.shape!=(len(x),) or not np.isfinite(a).all() or ((a<0)|(a>1)).any():raise ValueError('Frozen fold prediction alignment/range failed')
            results.append(p);details.append(receipt)
        share=np.clip(np.mean([p['share'].astype(float) for p in results],axis=0),0,1).astype(np.float32)
        score=np.clip(np.mean([p['score'].astype(float) for p in results],axis=0),0,1).astype(np.float32)
        for name,h in self.snapshots.items():
            if sha(name)!=h:raise ValueError('Frozen query state changed:'+name)
        after=self.backend.verify_after(self.run)
        if after!=self.verified:raise ValueError('Completed training artifacts changed during inference')
        return {'share':share,'score':score,'receipt':{'folds':details,'fivefold_equal_arithmetic_mean':True,
            'extra_calibration_or_query_fit':False,'postprocessing':'numericalclip[0,1] only',
            'query_db_only':True,'fixture_only':self.fixture,'official_submission':False}}


def infer_csv(run,input_csv,output_csv,*,device='cpu',batch_size=128):
    out=Path(output_csv);receipt=out.with_suffix('.manifest.json')
    if out.exists() or receipt.exists():raise FileExistsError('Fresh CSV/receipt required')
    engine=FrozenInferenceR13(run,device=device)  # COMPLETE barrier before query file open.
    ids,db,inp=read_query_csv(input_csv,expected_rows=1500)
    result=engine.predict_db(db,batch_size=batch_size)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['ID','share_added_vessel','score_added_vessel'])
        writer.writerows(zip(ids,result['share'].tolist(),result['score'].tolist()))
    if sha(input_csv)!=inp['raw_query_csv_sha256']:raise ValueError('Query bytes changed')
    record={**inp,**result['receipt'],'output_sha256':sha(out),'source_code_sha256':sha(__file__),
        'completed_run_verification':engine.verified,'ID_order_preserved':True}
    receipt.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['infer'])
    p.add_argument('--run-dir',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--batch-size',type=int,default=128)
    a=p.parse_args();r=infer_csv(a.run_dir,a.input,a.output,device=a.device,batch_size=a.batch_size)
    print(json.dumps({'rows':r['rows'],'output':a.output,'official_submission':False}))
