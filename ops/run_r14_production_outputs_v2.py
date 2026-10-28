"""R2 output-only repair: detached frozen policies and fresh hybrid outputs.

Existing R1 Tree-only CSV/OOF and failed hybrid artifacts stay unchanged. Hybrid
selection uses NN0+Tree0 only; Test-time other-fold Tree averaging is an
explicit bagging extension, never called exact hybrid5fold OOF.
"""
from pathlib import Path
import argparse,csv,json,sys,time,hashlib,gc,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
CONFIG='configs/r14_production_mac_v1.json';PROTOCOL='configs/r14_production_ensemble_protocol_v1.json'
TREE='models/r14_production_trees_r1';NNROOT='models/r14_production_nn_fold0_r1'
OUT='models/r14_production_outputs_r2';REPORT='reports/r14_production_r2'
TEST_SHA='3ef819192c05c4308b66564e6307e6f4f26fe806afe96ec93e2d15f25a3474d4'
OLD17_RESULT_SHA='18fc5095a1ff3bf493325fbad7ab9331d40f6ef0efcfd04704b3d9501e9da0ea'
LEGACY_DRIVER='ops/run_r14_production_outputs.py'
LEGACY_DRIVER_SHA='beb6765591dd8f142129038b2d525e8565da896ac3a1b684a8d87fce11056da0'
LEGACY_FAST3_POLICY='models/r14_production_outputs_r1/hybrid_fast3/policy.json'
LEGACY_FAST3_POLICY_SHA='59c624e196ed4bf842c9e1c7ac00aa8abd2b501189a6d96a35cca773ee1240d5'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def npz(p):
    import numpy as np
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def frozen(path,value):
    p=Path(path)
    if p.exists():
        if read(p)!=value:raise FileExistsError('Different existing frozen policy:'+str(p))
        return sha(p)
    write(p,value);return sha(p)

def detached_binding(binding):
    """Fit receives an owned snapshot, never caller-owned mutable pin maps."""
    if not isinstance(binding,dict):raise TypeError('Named binding required')
    return copy.deepcopy(binding)

def freeze_policy_snapshot(path,policy,source_pins):
    """Return disk-owned policy plus a separate extended execution pin map.

    A policy cannot include its own file hash. Its binding holds only pre-fit
    source pins; the external execution map adds the frozen policy afterward.
    Neither returned object aliases the caller or the other returned object.
    """
    from pipeline.ensemble_r14_production import token
    p=Path(path)
    if p.exists():raise FileExistsError('Fresh R2 policy path required:'+str(p))
    payload=copy.deepcopy(policy);pins=copy.deepcopy(source_pins)
    if token({k:v for k,v in payload.items() if k!='policy_sha256'})!=payload.get('policy_sha256'):
        raise ValueError('Policy content changed before disk freeze')
    if payload.get('binding',{}).get('source_pins')!=pins:
        raise ValueError('Policy pre-fit source pins differ from execution pins')
    frozen(p,payload);saved=read(p)
    if saved!=payload or token({k:v for k,v in saved.items() if k!='policy_sha256'})!=saved['policy_sha256']:
        raise ValueError('Frozen policy readback differs')
    key=str(p.relative_to(ROOT))
    if key in pins:raise ValueError('Policy cannot contain its own file hash')
    extended={**pins,key:sha(p)}
    return saved,extended

def reserve_scope(scope):
    if scope not in ('fast3','full4'):raise ValueError('Explicit completed neural scope required')
    suffix='' if scope=='fast3' else '_with_tfgrid'
    root=ROOT/OUT/('hybrid_'+scope)
    paths=[root,ROOT/REPORT/('hybrid_'+scope),
      ROOT/REPORT/('HYBRID_'+scope+'_17_PANEL_METRICS.csv'),
      ROOT/REPORT/('HYBRID_'+scope+'_QUERY_DISTRIBUTION_AUDIT.json')]
    for name in ('submissions/R14_r2/submission_r14_fold0','submissions/r14_main_r2/submission_r14_step1'):
        p=ROOT/(name+suffix+'.csv');paths.extend((p,p.with_suffix('.manifest.json')))
    if any(p.exists() for p in paths):raise FileExistsError('Fresh R2 scope required; no partial reuse or overwrite')
    root.mkdir(parents=True,exist_ok=False)
    return root

def verify_legacy_decisions(policy,scope):
    """The fast3 repair must preserve every original fitted decision."""
    if scope!='fast3':return
    p=ROOT/LEGACY_FAST3_POLICY
    if sha(p)!=LEGACY_FAST3_POLICY_SHA:raise ValueError('Original failed-stage frozen policy changed')
    original=read(p)
    from pipeline.ensemble_r14_production import token
    if token({k:v for k,v in original.items() if k!='policy_sha256'})!=original['policy_sha256']:
        raise ValueError('Original on-disk policy is not valid')
    decisions=lambda v:{k:x for k,x in v.items() if k not in ('binding','policy_sha256')}
    if decisions(policy)!=decisions(original):
        raise ValueError('R2 output repair changed an original fit/gate decision')

def approved():
    c=read(ROOT/CONFIG);a=read(ROOT/c['approval_path']);protocol=read(ROOT/PROTOCOL)
    if c['production_recipe_approved'] is not True or a['production_recipe_approved'] is not True or a['official_upload']:
        raise ValueError('Explicit local production approval required')
    if sha(a['user_instruction_path'])!=a['user_instruction_sha256']:raise ValueError('Approval instruction changed')
    bundle=ROOT/c['bundle_path'];m=read(bundle/'manifest.json')
    if m['data_hash']!=c['bundle_data_hash']:raise ValueError('Data version changed')
    return c,a,protocol,bundle,m

def role(bundle,m,fold,name,*,report_gate=None):
    if name not in ('ensemble_fit','ensemble_gate','report'):raise ValueError('Explicit R14 role required')
    if name=='report':
        if (report_gate is None or report_gate.get('status')!='completed' or report_gate.get('scope')!='tree_only_all5_complete'
            or report_gate.get('folds')!=list(range(5)) or set(report_gate.get('policies',{}))!=set(map(str,range(5)))):
            raise PermissionError('New scoped R14 tree report-access receipt required')
        if sha(ROOT/TREE/'ALL_FIVEFOLD_TREES_FROZEN.json')!=report_gate['tree_registry_sha256']:
            raise PermissionError('Tree report receipt registry differs')
        for entry in report_gate['policies'].values():
            if sha(ROOT/entry['path'])!=entry['sha256']:raise PermissionError('Frozen Tree policy changed')
    e=m['folds'][str(fold)]['variant_roles']['baseline'][name];p=bundle/e['path']
    if sha(p)!=e['sha256']:raise ValueError('Fixed role bytes differ')
    d=npz(p)
    if len(d['share'])!=(1000 if name=='report' else 500):raise ValueError('Fixed role row count differs')
    return d,{'path':str(p.relative_to(ROOT)),'sha256':e['sha256'],'role':name,'fold':fold,'baseline_common_observations':True}

def tree_registry():
    from ops.train_r14_production_trees import load_plan,completed
    p=load_plan(ROOT/TREE);models=completed(ROOT/TREE,p,persist=False)
    r=read(ROOT/TREE/'ALL_FIVEFOLD_TREES_FROZEN.json')
    if r.get('status')!='completed' or len(r['models'])!=5:raise ValueError('Five complete Tree folds required')
    # completed() revalidates source/model/approval/Fold0 exact-reuse binding.
    return r

def tree_pair(registry,fold,db):
    import numpy as np
    from pipeline.trees_ensemble_r13 import PhysicsLGBM
    shares={};clfs={}
    for v in ('baseline','highshare'):
        e=registry['models'][str(fold)][v+'P1'];folder=ROOT/e['model_dir'];s=read(folder/'summary.json')
        if sha(folder/'summary.json')!=e['summary_sha256'] or s['n_features']!=5252:raise ValueError('Full Tree entry changed')
        model=PhysicsLGBM.load(folder,expected_binding=s['training_binding']);p=model.predict(db,batch_size=256)
        shares['tree_'+v]=p['share'];clfs[v]=p['score_classifier'];del model;gc.collect()
    clfs['mean']=(.5*clfs['baseline'].astype(float)+.5*clfs['highshare'].astype(float)).astype(np.float32)
    return shares,clfs

def metrics(y,p):
    import numpy as np
    from pipeline.ensemble_r14_production import auc
    y=np.asarray(y,float);s=np.asarray(p['share'],float);score=np.asarray(p['score'],float);e=s-y;hi=y>.2
    return {'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(abs(e).mean()),'bias':float(e.mean()),
      'auc_share':auc(y,s),'auc_score':auc(y,score),'prediction_mean':float(s.mean()),'prediction_rms':float(np.sqrt(np.mean(s*s))),
      'target_mean':float(y.mean()),'high_y_n':int(hi.sum()),'high_y_bias':float(e[hi].mean()) if hi.any() else None,
      'score_ge099_fraction':float((score>=.99).mean())}

def verify_pins(pins):
    for p,h in pins.items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen execution source changed:'+p)

def base_pins(c):
    from pipeline.ensemble_r14_production import token
    if sha(ROOT/LEGACY_DRIVER)!=LEGACY_DRIVER_SHA:raise ValueError('Preserved original output driver changed')
    names=[CONFIG,PROTOCOL,c['approval_path'],'pipeline/ensemble_r14_production.py',str(Path(__file__).relative_to(ROOT)),LEGACY_DRIVER,
      TREE+'/ALL_FIVEFOLD_TREES_FROZEN.json',c['bundle_path']+'/manifest.json']
    return {n:sha(ROOT/n) for n in names}

def write_csv(path,ids,p,receipt):
    import numpy as np
    path=Path(path);manifest=path.with_suffix('.manifest.json')
    if path.exists() or manifest.exists():raise FileExistsError('Never overwrite a submission CSV/receipt')
    if len(ids)!=1500 or len(set(ids))!=1500:raise ValueError('Exactly1500 unique query IDs required')
    for k in ('share','score'):
        a=np.asarray(p[k])
        if a.shape!=(1500,) or not np.isfinite(a).all() or ((a<0)|(a>1)).any():raise ValueError('Invalid submission values')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','share_added_vessel','score_added_vessel']);w.writerows(zip(ids,p['share'].tolist(),p['score'].tolist()))
    with path.open(newline='') as f:
        rr=list(csv.reader(f))
    if rr[0]!=['ID','share_added_vessel','score_added_vessel'] or [v[0] for v in rr[1:]]!=list(ids):raise ValueError('CSV reread ID/schema mismatch')
    reread=np.asarray([[float(x) for x in row[1:]] for row in rr[1:]],dtype=np.float64)
    if reread.shape!=(1500,2) or not np.array_equal(reread,np.column_stack((p['share'],p['score'])).astype(np.float64)):
        raise ValueError('CSV numeric roundtrip differs')
    record={**receipt,'path':str(path.relative_to(ROOT)),'output_sha256':sha(path),'rows':1500,'ID_order_preserved':True,'numeric_roundtrip_exact':True,'official_upload':False}
    write(manifest,record);return record

def stats(values):
    import numpy as np
    a=np.asarray(values,float)
    return {'mean':float(a.mean()),'std_ddof0':float(a.std()),'rms':float(np.sqrt(np.mean(a*a))),
      'quantiles':dict(zip(('min','p01','p05','p25','median','p75','p95','p99','max'),np.quantile(a,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist())),
      'ge099_fraction':float((a>=.99).mean()),'le001_fraction':float((a<=.01).mean())}

def compare_R9(ids,p,components,out):
    import numpy as np
    old=ROOT/'submissions/r9_fold0_all5_cpu_r1/submission.csv'
    if sha(old)!='f016265ae9a1702dbb7121bfb0c852f5900f85a17194881056316768300fb39d':raise ValueError('Original R9 submitted CSV bytes differ')
    with old.open(newline='') as f:rows=list(csv.DictReader(f))
    index={r['ID']:r for r in rows}
    if len(index)!=1500 or set(index)!=set(ids):raise ValueError('R9 query ID set differs')
    oldp={k:np.asarray([float(index[i][col]) for i in ids]) for k,col in [('share','share_added_vessel'),('score','score_added_vessel')]}
    def corr(a,b):
        v=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else float('nan')
        return v if np.isfinite(v) else None
    result={'R9_official_reference':{'score':.782464193,'RMSE':.217535806,'AUC':.555066666},
      'R9_reference_CSV_sha256':sha(old),'R9_uploaded_bytes_independently_verified':False,
      'new_official_or_Test_label_metrics_available':False,'R9':{k:stats(v) for k,v in oldp.items()},'new':{k:stats(p[k]) for k in ('share','score')},
      'new_vs_R9_Pearson':{k:corr(oldp[k],p[k]) for k in ('share','score')},
      'new_minus_R9_mean':{k:float((p[k]-oldp[k]).mean()) for k in ('share','score')},'components':{k:stats(v) for k,v in components.items()}}
    names=list(components);matrix=np.column_stack(list(components.values()))
    corr=np.corrcoef(matrix,rowvar=False);result['member_share_correlation_order']=names
    result['member_share_Pearson']= [[None if not np.isfinite(v) else float(v) for v in row] for row in corr]
    write(out,result);return result

def neural_registry(scope):
    from pipeline.training_r14_production import source_codes,restore_completed_model
    run=ROOT/NNROOT;filename='FAST_MODELS_FROZEN.json' if scope=='fast3' else 'ALL_FOLD0_MODELS_FROZEN.json'
    p=run/filename;r=read(p);identity=read(run/'execution_identity.json')
    expected={'itransformer','fdy','bg_cnn'} if scope=='fast3' else {'itransformer','fdy','bg_cnn','tfgrid_fdy_conformer'}
    if (r['status']!='completed' or r['fold']!=0 or set(r['models'])!=expected or r['source_code_sha256']!=source_codes()
        or r['config_sha256']!=sha(ROOT/CONFIG) or identity['approval_sha256']!=sha(ROOT/'ops/r14_production_approval.json')):
        raise ValueError('Complete approved Fold0 neural stage required')
    pins={str(p.relative_to(ROOT)):sha(p),str((run/'execution_identity.json').relative_to(ROOT)):sha(run/'execution_identity.json')}
    for kind,entry in r['models'].items():
        folder=run/kind
        if sha(folder/'summary.json')!=entry['summary_sha256'] or sha(folder/'best.pt')!=entry['best_checkpoint_sha256']:
            raise ValueError('Neural catalog artifacts changed')
        model,frontend,summary=restore_completed_model(folder,device='cpu')
        if summary['kind']!=kind or summary['data_hash']!=r['data_hash']:raise ValueError('Neural restored stage identity differs')
        pins[str((folder/'summary.json').relative_to(ROOT))]=entry['summary_sha256'];pins[str((folder/'best.pt').relative_to(ROOT))]=entry['best_checkpoint_sha256']
        del model,frontend;gc.collect()
    return r,pins

def neural_predictions(registry,db):
    from pipeline.training_r14_production import predict_completed_model
    pred={}
    for kind in registry['models']:
        p=predict_completed_model(ROOT/NNROOT/kind,db,batch_size=128,device='cpu')
        pred['nn_'+kind]=p['share'];del p;gc.collect()
    return pred

def development_panels():
    import numpy as np
    from ops import evaluate_r12_ablation as parent
    p,f=parent.check_freeze('reports/R12_ALL17_GLOBAL_FREEZE.json');policy=ROOT/'reports/r12_cpu_ablation_r1/selected_policy.json'
    panels=[parent.load_R12(p,r,policy,f['freeze_sha256']) for r in ('validation_gate','eval','high_snr_long')]
    dev,r11=parent.load_development();panels.extend(dev)
    for pos in parent.DEV_R11_POS:
        for neg in parent.DEV_R11_NEG:
            x,z=r11[pos],r11[neg];panels.append({'name':x['name']+'_VS_'+neg,'y':np.r_[x['y'],z['y']],
              'db':np.concatenate((x['db'],z['db'])),'sample_id':np.r_[x['sample_id'],z['sample_id']]})
    if len(panels)!=17:raise ValueError('Expected same17 development panels')
    return panels

def hybrid(scope='fast3'):
    import numpy as np
    from pipeline.ensemble_r14_production import fit_policy,apply_policy
    from pipeline.inference_r5 import read_query_csv
    c,a,protocol,bundle,m=approved();trees=tree_registry();nn,npins=neural_registry(scope)
    pins={**base_pins(c),**npins};root=reserve_scope(scope)
    fit,fr=role(bundle,m,0,'ensemble_fit');gate,gr=role(bundle,m,0,'ensemble_gate')
    fs,fc=tree_pair(trees,0,fit['db']);gs,gc_=tree_pair(trees,0,gate['db'])
    fs.update(neural_predictions(nn,fit['db']));gs.update(neural_predictions(nn,gate['db']))
    policy=fit_policy(fit['share'],fs,fc,gate['share'],gs,gc_,binding=detached_binding({'fold':0,'scope':'Fold0_'+scope+'_hybrid',
       'fit':fr,'gate':gr,'source_pins':pins,'data_hash':m['data_hash'],'Test_tree_bagging_is_inference_extension':True}))
    verify_legacy_decisions(policy,scope)
    pp=root/'policy.json';policy,pins=freeze_policy_snapshot(pp,policy,pins)
    barrier={'status':'completed','scope':'Fold0_'+scope+'_policy_frozen','source_pins':copy.deepcopy(pins),'production_recipe_approved':True,
       'neural_members':list(nn['models']),'neural_folds':[0],'tree_folds_available':list(range(5)),
       'full_neural5fold_OOF':False,'report17_for_fitting':False,'Test_for_fitting':False}
    frozen(root/'FROZEN_BEFORE_DEVELOPMENT_AND_QUERY.json',barrier)
    oldroot=ROOT/'reports/r14_fulltree_decoupled_v1';oldresult=oldroot/'RESULTS.json'
    if sha(oldresult)!=OLD17_RESULT_SHA:raise ValueError('Historical17 comparison registry changed')
    oldhashes=read(oldresult)['output_sha256']
    if sha(oldroot/'selected_results.csv')!=oldhashes['selected_results.csv']:raise ValueError('Historical panel comparison CSV changed')
    reportrows=[]
    for panel in development_panels():
        verify_pins(pins);s,cl=tree_pair(trees,0,panel['db']);n=neural_predictions(nn,panel['db']);s.update(n)
        pred=apply_policy(policy,s,cl)
        saved=ROOT/'reports/r14_fulltree_decoupled_v1/predictions'/(panel['name']+'.npz')
        oldname=str(saved.relative_to(oldroot))
        if oldname not in oldhashes or sha(saved)!=oldhashes[oldname]:raise ValueError('Historical panel artifact changed')
        oldpred=npz(saved)
        if not np.array_equal(panel['sample_id'].astype(str),oldpred['sample_id'].astype(str)) or not np.array_equal(panel['y'],oldpred['target_share']):
            raise ValueError('Historical panel comparison identity differs:'+panel['name'])
        row={'panel':panel['name'],'model':'R14_production_'+scope+'_exact_Fold0_proxy',**metrics(panel['y'],pred),
          'development_exposed':True,'other_fold_tree_bagging_used':False,'historical_comparison_npz_sha256':sha(saved)}
        reportrows.append(row)
        folder=ROOT/REPORT/('hybrid_'+scope);folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/(panel['name']+'.npz'),sample_id=panel['sample_id'],target_share=panel['y'],
           member_order=np.asarray(list(s)),member_share=np.column_stack(list(s.values())),**pred)
        old=ROOT/'reports/r14_fulltree_decoupled_v1/selected_results.csv'
        for oldrow in csv.DictReader(old.open()):
            if oldrow['panel']==panel['name']:
                row['R14_smallCNN_fullTree_previous_rmse']=float(oldrow['rmse']);row['rmse_minus_previous']=row['rmse']-float(oldrow['rmse'])
                row['R14_smallCNN_fullTree_previous_score_AUC']=float(oldrow['auc_score']);break
        print(json.dumps(row),flush=True)
    rp=ROOT/REPORT/('HYBRID_'+scope+'_17_PANEL_METRICS.csv');rp.parent.mkdir(parents=True,exist_ok=True)
    with rp.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(reportrows[0]));w.writeheader();w.writerows(reportrows)
    verify_pins(pins)
    if sha(ROOT/'data/raw/Test.csv')!=TEST_SHA:raise ValueError('Approved official query bytes differ')
    ids,db,query=read_query_csv(ROOT/'data/raw/Test.csv',expected_rows=1500)
    neural=neural_predictions(nn,db);family=[];classifiers=[]
    for fold in range(5):
        s,cl=tree_pair(trees,fold,db);family.append(s);classifiers.append(cl)
    fold0shares={**family[0],**neural};fold0pred=apply_policy(policy,fold0shares,classifiers[0])
    bagshares={k:np.mean([v[k].astype(float) for v in family],axis=0).astype(np.float32) for k in family[0]}
    bagcl={k:np.mean([v[k].astype(float) for v in classifiers],axis=0).astype(np.float32) for k in ('baseline','highshare')}
    bagcl['mean']=(.5*bagcl['baseline'].astype(float)+.5*bagcl['highshare'].astype(float)).astype(np.float32)
    bagshares.update(neural);bagpred=apply_policy(policy,bagshares,bagcl)
    verify_pins(pins)
    if sha(ROOT/'data/raw/Test.csv')!=query['raw_query_csv_sha256']:raise ValueError('Query changed during inference')
    suffix='' if scope=='fast3' else '_with_tfgrid'
    common={**query,'source_pins':pins,'policy_sha256':sha(pp),'neural_members':list(nn['models']),
       'NN_folds':[0],'full_neural5fold_OOF':False,'score_weight':policy['score_policy']['lambda']}
    r0=write_csv(ROOT/('submissions/R14_r2/submission_r14_fold0'+suffix+'.csv'),ids,fold0pred,{**common,'scope':'exact_Fold0_hybrid','Tree_folds':[0]})
    rb=write_csv(ROOT/('submissions/r14_main_r2/submission_r14_step1'+suffix+'.csv'),ids,bagpred,{**common,
      'scope':'Fold0_policy_with_Test_time_5fold_Tree_bagging_extension','Tree_folds':list(range(5)),
      'exact_hybrid_5fold_validation_measured':False,'fitting_on_averaged_Tree5_predictions':False})
    audit=compare_R9(ids,bagpred,bagshares,ROOT/REPORT/('HYBRID_'+scope+'_QUERY_DISTRIBUTION_AUDIT.json'))
    np.savez_compressed(root/'query_predictions.npz',sample_id=np.asarray(ids),**bagpred,
       member_order=np.asarray(list(bagshares)),member_share=np.column_stack(list(bagshares.values())),
       Fold0_share=fold0pred['share'],Fold0_score=fold0pred['score'])
    write(root/'COMPLETED.json',{'status':'completed','local_CSVs':[r0,rb],'panel_metrics':str(rp.relative_to(ROOT)),
       'source_pins':pins,'official_upload':False,'exact_hybrid5fold_OOF':False})
    print(json.dumps({'status':'hybrid_CSVs_completed','scope':scope,'receipts':[r0,rb]}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['hybrid-fast3','hybrid-full4']);a=p.parse_args()
    from ops.train_r12_lightgbm import environment
    environment(2)
    if a.action=='hybrid-fast3':hybrid('fast3')
    elif a.action=='hybrid-full4':hybrid('full4')

if __name__=='__main__':main()
