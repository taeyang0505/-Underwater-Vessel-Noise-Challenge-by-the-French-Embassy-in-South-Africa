"""R13 four frozen recipes: gate selection, locked reporting, no Test I/O.

Regressor and diagnostic score are selected separately on the fixed gate.
All R10/R11 panels are exposed developmental stress tests, not blind OOF.
Original R12 artifacts are read, never modified or retrospectively reranked.
"""
from __future__ import annotations
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
ARMS=('baselineP0','baselineP1','highshareP0','highshareP1')
LAMBDAS=(0.,.25,.5,1.)

def mix_score(share,classifier,weight):
    import numpy as np
    if weight not in LAMBDAS:raise ValueError('Non-predeclared classifier weight')
    return ((1-weight)*np.asarray(share,np.float64)+weight*np.asarray(classifier,np.float64)).astype(np.float32)

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--cnn',default='models/r13_cpu_cnn_r1');a.add_argument('--tree',default='models/r13_compact_trees_v1')
    a.add_argument('--output',default='reports/r13_cpu_correction_r1');a.add_argument('--threads',type=int,default=2)
    a.add_argument('--batch',type=int,default=128);args=a.parse_args()
    from ops import evaluate_r12_ablation as parent
    parent.environment(args.threads)
    import numpy as np,pandas as pd,torch
    from pipeline.training_r12 import R12MaskCNN,channels,predict,sha,write
    from pipeline.features_r12 import load_spectral_basis
    from ops.train_r12_lightgbm import CompactPhysicsLGBM
    from ops.verify_r12_evaluation import cluster_ci
    from pipeline.dataset_r13 import validate_r13_bundle
    config=ROOT/'configs/r13_cpu_correction_v1.json';cfg=json.loads(config.read_text())
    if cfg['training_arms']!=list(ARMS) or cfg['score_candidates_classifier_weight']!=list(LAMBDAS):raise ValueError('Prospective candidate set changed')
    cr=ROOT/args.cnn;tr=ROOT/args.tree
    c=json.loads((cr/'ALL_CNN_FROZEN.json').read_text());t=json.loads((tr/'ALL_TREES_FROZEN.json').read_text())
    if any(v.get('status')!='completed' or not v.get('all_models_frozen') for v in (c,t)) or any((v/'FAILED.json').exists() for v in (cr,tr)):
        raise RuntimeError('All CNN/tree fits must be complete, no partial reuse')
    if set(c['models'])!=set(ARMS) or set(t['models'])!=set(ARMS):raise ValueError('Four paired recipes required')
    data=ROOT/'data/synthetic/r13_cpu_correction_v1';m=validate_r13_bundle(data)
    if c['binding']['data_hash']!=m['data_hash'] or t['data_hash']!=m['data_hash']:raise ValueError('Model TRAIN identity differs')
    pins={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),config,cr/'ALL_CNN_FROZEN.json',tr/'ALL_TREES_FROZEN.json',data/'manifest.json']}
    pins.update(c['source_pins']);pins.update(t['binding']['source_pins'])
    comparator_dir=ROOT/'reports/r12_mask_matched_evaluation_r2'
    oldextra=json.loads((comparator_dir/'RESULTS.json').read_text())
    if oldextra.get('status')!='completed':raise ValueError('Completed R12 comparator required')
    for name,h in oldextra['output_sha256'].items():
        if sha(comparator_dir/name)!=h:raise ValueError('R12 comparator changed:'+name)
        pins[str((comparator_dir/name).relative_to(ROOT))]=h
    pins[str((comparator_dir/'RESULTS.json').relative_to(ROOT))]=sha(comparator_dir/'RESULTS.json')
    oldplan,oldf=parent.check_freeze('reports/R12_ALL17_GLOBAL_FREEZE.json')
    oldpolicy_path=ROOT/'reports/r12_cpu_ablation_r1/selected_policy.json'
    parent.check_policy(oldpolicy_path,oldf['freeze_sha256'])
    basis=load_spectral_basis(ROOT/oldplan['basis_path'],expected_sha256=oldf['basis_sha256'],expected_training_binding=m['basis_training_binding'])
    nns={};trees={};torch.set_num_threads(args.threads)
    for arm in ARMS:
        info=c['models'][arm];sp=ROOT/info.get('summary_path',info.get('reused_summary_path',''))
        if sha(sp)!=info.get('summary_sha256',info.get('reused_summary_sha256')):raise ValueError('CNN summary changed')
        nnsummary=json.loads(sp.read_text());ckpath=ROOT/info['checkpoint_path']
        if nnsummary['epochs_run']!=100 or nnsummary['optimizer']!='Adam' or nnsummary['early_stopping'] or sha(ckpath)!=info['checkpoint_sha256']:
            raise ValueError('CNN fixed100/checkpoint policy changed')
        if nnsummary['arm']['shape']!=arm.endswith('1'):raise ValueError('CNN spectral feature toggle differs')
        ck=torch.load(ckpath,map_location='cpu',weights_only=True)
        history=[json.loads(v) for v in (sp.parent/'epochs.jsonl').read_text().splitlines()]
        if ([v['epoch'] for v in history]!=list(range(1,101)) or ck['epoch']!=nnsummary['best_epoch']
            or ck['arm']!=nnsummary['arm'] or ck['parameters']!=26953
            or nnsummary['best_epoch']!=int(np.argmin([v['validation_fit_rmse'] for v in history]))+1):
            raise ValueError('CNN epoch/checkpoint identity differs')
        nn=R12MaskCNN();nn.load_state_dict(ck['state_dict'],strict=True);nn.eval();nns[arm]=nn
        if not all(torch.isfinite(v).all() for v in nn.parameters()):raise ValueError('Nonfinite CNN checkpoint')
        summary=ROOT/t['models'][arm]['summary_path']
        if sha(summary)!=t['models'][arm]['summary_sha256']:raise ValueError('Tree summary changed')
        ts=json.loads(summary.read_text());shape=arm.endswith('1');data_arm='baseline' if arm.startswith('baseline') else 'highshare'
        expected={'R13_data_hash':m['data_hash'],'R13_manifest_sha256':sha(data/'manifest.json'),'data_arm':data_arm,
            'include_shape':shape,'TRAIN_npz_sha256':m['arms'][data_arm]['sha256'],
            'parent_R12_manifest_sha256':m['parent_R12_manifest_sha256'],'parent_R12_data_hash':m['parent_R12_data_hash'],
            'validation_fit_npz_sha256':m['validation_references']['validation_fit']['sha256'],
            'basis_sha256':oldf['basis_sha256'] if shape else None,'basis_training_binding':m['basis_training_binding'] if shape else None}
        if any(ts['training_binding'].get(k)!=v for k,v in expected.items()):raise ValueError('Tree data/feature binding differs')
        trees[arm]=CompactPhysicsLGBM.load(summary.parent,expected_binding=ts['training_binding'])
        for p in [sp,ckpath,summary,summary.parent/'share.txt',summary.parent/'presence.txt']:
            pins[str(p.relative_to(ROOT))]=sha(p)
    def verify():
        for name,h in pins.items():
            if sha(ROOT/name)!=h:raise ValueError('Frozen R13 source changed:'+name)
    verify();out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir()
    freeze={'format':'r13_cpu_global_freeze_v1','status':'completed','source_pins':pins,'data_hash':m['data_hash'],
        'CNN_new_fits':3,'completed_control_reused':1,'tree_new_fits':4,'numeric_gate_read':False,'development_exposed':True}
    freeze['freeze_sha256']=parent.token(freeze);write(out/'FROZEN_BEFORE_GATE.json',freeze)
    def infer(db):
        cache={};r={}
        for shape in (False,True):
            x,w=channels(db,basis,shape);cache[shape]=(torch.from_numpy(x),torch.from_numpy(w),w)
        for arm in ARMS:
            x,w,ww=cache[arm.endswith('1')];n=predict(nns[arm],x,w,args.batch);tree=trees[arm].predict(db,batch_size=args.batch)
            if (float(abs((ww*n['cell_mask']).sum((1,2))-n['share']).max())>3e-6
                or np.any(n['cell_mask'][ww==0]!=0) or not np.isfinite(n['cell_mask']).all()
                or np.any((n['cell_mask']<0)|(n['cell_mask']>1))):raise ValueError('Physical pooling/missing-mask mismatch')
            share=(.5*n['share'].astype(float)+.5*tree['share'].astype(float)).astype(np.float32)
            r[arm]={'CNN':n,'tree':tree,'share':share,'classifier':tree['score_classifier']}
        return r
    gate=parent.load_R12(oldplan,'validation_gate');gpred=infer(gate['db'])
    gate_rmse={arm:float(np.sqrt(np.mean((gpred[arm]['share'].astype(float)-gate['y'])**2))) for arm in ARMS}
    winner=min(ARMS,key=lambda arm:gate_rmse[arm])
    gate_auc={str(v):parent.auc(gate['y'],mix_score(gpred[winner]['share'],gpred[winner]['classifier'],v)) for v in LAMBDAS}
    lam=max(LAMBDAS,key=lambda v:gate_auc[str(v)])
    policy={'format':'r13_cpu_selected_policy_v1','selected_arm':winner,'regression_weights':{'CNN':.5,'tree':.5},
        'score_classifier_weight':lam,'gate_rmse':gate_rmse,'selected_arm_gate_score_AUC':gate_auc,
        'selection':'regression lowest gate RMSE; selected regression arm score highest gate AUC, first canonical tie',
        'freeze_sha256':freeze['freeze_sha256'],'development_not_blind':True,'Test_or_LB_tuning':False}
    policy['policy_sha256']=parent.token(policy);write(out/'selected_policy.json',policy);policysha=sha(out/'selected_policy.json')
    def check_policy():
        if sha(out/'selected_policy.json')!=policysha:raise ValueError('R13 policy changed after gate')
    check_policy()
    panels=[gate]
    for role in ('eval','high_snr_long'):
        check_policy();panels.append(parent.load_R12(oldplan,role,oldpolicy_path,oldf['freeze_sha256']))
    dev,r11=parent.load_development();panels.extend(dev)
    # New evaluator explicitly connects the previously omitted unseen sensor/week groups.
    for p in panels:
        if p['name'].startswith('R10_unseen__'):
            meta=pd.read_parquet(ROOT/parent.R10/'unseen_source_manifest.parquet')
            if len(meta)!=len(p['y']):raise ValueError('Unseen metadata order/count mismatch')
            if 'sample_id' in meta and not np.array_equal(meta.sample_id.astype(str),p['sample_id']):raise ValueError('Unseen IDs differ')
            p['groups']=(meta.sensor_group.astype(str)+'__'+meta.recording_week_start.astype(str)).to_numpy()
            p['cluster_key']='sensor_group_x_recording_week_start'
    for pos in parent.DEV_R11_POS:
        for neg in parent.DEV_R11_NEG:
            x,y=r11[pos],r11[neg];panels.append({'name':x['name']+'_VS_'+neg,'y':np.r_[x['y'],y['y']],
                'db':np.concatenate([x['db'],y['db']]),'sample_id':np.r_[x['sample_id'],y['sample_id']],
                'groups':np.r_[x['groups'],y['groups']],'cluster_key':x['cluster_key'],'site':np.r_[x['site'],y['site']],
                'stage':'exposed_development'})
    if len(panels)!=17:raise ValueError('Expected same17 evaluation panels')
    rows=[];cis=[];start=time.perf_counter()
    def enrich(p,r):
        m=parent.metrics(p,r);z=np.asarray(p['y'])>0;e=np.asarray(r['share'],float)-p['y'];high=p['y']>.2;low=z&(p['y']<=.03)
        m.update(score_brier=float(np.mean((np.asarray(r['score'],float)-z.astype(float))**2)),score_ge099=float(np.mean(r['score']>=.99)),
            zero_prediction_mean=float(np.mean(r['share'][~z])) if (~z).any() else None,
            high_y_bias=float(e[high].mean()) if high.any() else None,high_y_rmse=float(np.sqrt(np.mean(e[high]**2))) if high.any() else None,
            low_positive_n=int(low.sum()),classifier_ge099=float(np.mean(r['score_classifier']>=.99)) if 'score_classifier' in r else None)
        if low.any() and (~z).any():
            q=low|(~z);m['low_positive_AUC']=parent.auc(np.asarray(p['y'])[q],np.asarray(r['score'])[q])
        else:m['low_positive_AUC']=None
        return m
    for panel in panels:
        check_policy();name=panel['name'];pred=gpred if name==gate['name'] else infer(panel['db']);output={};raw={}
        for arm,v in pred.items():
            output['CNN_'+arm]={k:v['CNN'][k] for k in ['share','score','contrast']}
            output['Tree_'+arm]=v['tree']
            for value in LAMBDAS:
                output['Blend50_'+arm+'_lambda'+str(value)]={'share':v['share'],
                    'score':mix_score(v['share'],v['classifier'],value),
                    'score_classifier':v['classifier']}
            raw[arm+'_cell_mask']=v['CNN']['cell_mask'];raw[arm+'_contrast']=v['CNN']['contrast'];raw[arm+'_classifier']=v['classifier']
        primary=output['Blend50_'+winner+'_lambda'+str(lam)]
        output['R13_selected']=primary
        saved=ROOT/'reports/r12_mask_matched_evaluation_r2/predictions'/(name+'.npz')
        with np.load(saved,allow_pickle=False) as z:
            if not np.array_equal(z['target_share'],panel['y']) or not np.array_equal(z['sample_id'].astype(str),panel['sample_id'].astype(str)):
                raise ValueError('R12 comparator ID/target order differs')
            order=z['model_order'].astype(str).tolist();idx=order.index('Blend50_L1HM1P1')
            output['R12_matched_share_only']={'share':z['share'][:,idx],'score':z['score'][:,idx]}
            if 'R9_frozen_all5' in order:
                j=order.index('R9_frozen_all5');output['R9_frozen_all5']={'share':z['share'][:,j],'score':z['score'][:,j],'mixed_R9_component':True}
        for model,value in output.items():rows.append({'panel':name,'model':model,'development_not_blind':True,**enrich(panel,value)})
        for other in ['R12_matched_share_only','R9_frozen_all5']:
            if other not in output:continue
            co=cluster_ci(name,panel['y'],panel['groups'],panel['cluster_key'],(primary['share'],primary['score']),
                (output[other]['share'],output[other]['score']))
            cis.extend({'panel':name,'comparison':'selected_minus_'+other,'metric':metric,**value} for metric,value in co.items())
        arrays={'sample_id':panel['sample_id'],'target_share':panel['y'],'model_order':np.asarray(list(output)),
            'share':np.column_stack([v['share'] for v in output.values()]),'score':np.column_stack([v['score'] for v in output.values()]),
            'observed_weights':channels(panel['db'],basis,False)[1],**raw}
        np.savez_compressed(out/'predictions'/(name+'.npz'),**arrays)
        print(json.dumps({'panel':name,'selected_arm':winner,'lambda':lam,**enrich(panel,primary)}),flush=True)
    parent.csv_rows(out/'results.csv',rows);parent.csv_rows(out/'paired_cluster_CI.csv',cis)
    verify();check_policy();parent.check_freeze('reports/R12_ALL17_GLOBAL_FREEZE.json')
    result={'format':'r13_cpu_evaluation_v1','status':'completed','selected_policy':policy,'panels':17,'metric_rows':len(rows),
        'source_pins':pins,'no_Test_or_GPU_or_refit':True,'raw_masks_weights_classifiers_saved':True,
        'elapsed_evaluation_seconds':time.perf_counter()-start,
        'limitations':['Shared references and development-exposed panels; not blind OOF','Single seed and four recipe comparison',
            'Conditional source-cluster bootstrap, not a multiple-comparison adjustment','AUC/Brier use presence labels; regression y remains physical energy share']}
    result['output_sha256']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()};write(out/'RESULTS.json',result)

if __name__=='__main__':main()
