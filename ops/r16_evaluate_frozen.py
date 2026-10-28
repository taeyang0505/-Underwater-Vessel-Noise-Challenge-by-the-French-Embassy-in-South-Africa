"""R16 frozen CPU evaluation; no Test IO, no fitting or policy selection.

Two stages keep local_report/domain_holdout behind the data provider's global
freeze guard. TF epoch37 is explicitly an incomplete-budget diagnostic model,
not a completed production model. Labels enter saved-array metrics only.
"""
from __future__ import annotations
import argparse,csv,gc,hashlib,json,os,resource,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT='reports/r16_frozen_evaluation_r1'
KINDS=('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer')
OPERATORS=('control_eq6','mfhf_shelf6','mfhf_shelf12','mfhf_shelf12_lfminus6','alpha_preEQ_inserted','independent_subband_snr')
ROLES={'development':('local_fit',),'locked':('local_gate','local_report','domain_holdout')}
ATTACHMENT=Path('/Users/hongtaeyang/.codex/attachments/1fc448b5-90b6-4ff8-b790-a20fe8ecf1ee/붙여넣은 텍스트.txt')
TF_SHA='172b81179f9fc7f0355b5ccc40bc4a6d43d627e01908652dfd61e593caf759c9'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        n=os.fsdecode(args[0]).replace('\\','/')
        if '/data/raw/Test.csv' in n or n.endswith('/Test.csv') or '/r15_' in n and any(x in n for x in ('joined1500','Test_','test_')):
            raise PermissionError('R16 prohibits challenge query and previous query-derived audit arrays')

def pins_model_catalog():
    pre=read(ROOT/'submissions/r9_fold0_all5_cpu_r1/prequery_plan.json')
    registry=read(ROOT/'models/r14_production_trees_r1/ALL_FIVEFOLD_TREES_FROZEN.json')
    catalog=[];pins={}
    def pin(p,expected=None):
        p=Path(p);p=p if p.is_absolute() else ROOT/p
        if not p.is_relative_to(ROOT):raise ValueError('Project-contained models required')
        h=sha(p)
        if expected is not None and h!=expected:raise ValueError('Model/source changed:'+str(p))
        pins[str(p.relative_to(ROOT))]=h;return h
    for p in ['ops/r16_evaluate_frozen.py','ops/r15_training_pause_receipt.json',
              'submissions/r9_fold0_all5_cpu_r1/prequery_plan.json',
              'submissions/r9_fold0_all5_cpu_r1/frozen_policy.json',
              'models/r14_production_outputs_r2/hybrid_fast3/policy.json',
              'models/r14_production_trees_r1/ALL_FIVEFOLD_TREES_FROZEN.json',
              'models/r14_production_nn_fold0_r1/execution_identity.json']:
        pin(p)
    for k in KINDS:
        folder=Path(pre['nn_dirs'][k]);s=read(folder/'summary.json')
        pin(folder/'summary.json')
        for n,h in s['artifact_sha256'].items():pin(folder/n,h)
        for n,h in s['code_sha256'].items():pin(n,h)
        if s['status']!='completed' or s['epochs_run']!=100 or s['fixture_only'] is not False:raise ValueError('Completed old R9 required')
        catalog.append({'name':'R9_'+k,'family':'R9_NN','kind':k,'folder':str(folder.relative_to(ROOT)),
                        'checkpoint':'best_window.pt','epoch':s['best_window_epoch'],
                        'own_score':'frozen derived share-plus-temporal-contrast, not independent presence probability',
                        'independent_classifier':False})
    for fold in range(5):
        folder=ROOT/f'models/r9_mac_trees_r2/fold_{fold:02d}/models/lgbm_physics';s=read(folder/'summary.json');pin(folder/'summary.json')
        for n,h in s['artifact_sha256'].items():pin(folder/n,h)
        for n,h in s['code_sha256'].items():pin(n,h)
        catalog.append({'name':f'R9_tree_f{fold}','family':'R9_tree','fold':fold,'folder':str(folder.relative_to(ROOT)),
                        'own_score':'.5 clipped_share+.5 independent_classifier','independent_classifier':True})
    identity=read(ROOT/'models/r14_production_nn_fold0_r1/execution_identity.json')
    for k in KINDS:
        folder=ROOT/'models/r14_production_nn_fold0_r1'/k
        if k=='tfgrid_fdy_conformer':
            pin(folder/'latest.pt',TF_SHA);epoch=37;summary=None;code=identity['source_code_sha256'];checkpoint='latest.pt'
        else:
            s=read(folder/'summary.json');pin(folder/'summary.json')
            for n,h in s['artifact_sha256'].items():pin(folder/n,h)
            if s['status']!='completed' or s['epochs_run']!=100:raise ValueError('R14 fast NN complete budget required')
            epoch=s['best_epoch'];code=s['source_code_sha256'];checkpoint='best.pt'
        for n,h in code.items():pin(n,h)
        catalog.append({'name':'R14_'+k,'family':'R14_NN','kind':k,'folder':str(folder.relative_to(ROOT)),
                        'checkpoint':checkpoint,'epoch':epoch,'budget_complete':k!='tfgrid_fdy_conformer',
                        'own_score':'share only; independent presence head unavailable','independent_classifier':False})
    for fold in range(5):
        for v in ('baseline','highshare'):
            e=registry['models'][str(fold)][v+'P1'];folder=ROOT/e['model_dir'];s=read(folder/'summary.json');pin(folder/'summary.json',e['summary_sha256'])
            for n,h in e['artifact_sha256'].items():pin(n,h)
            for n,h in s['code_sha256'].items():pin(n,h)
            catalog.append({'name':f'R14_tree_{v}_f{fold}','family':'R14_tree','fold':fold,'variant':v,'folder':str(folder.relative_to(ROOT)),
                            'own_score':'default share only; independent raw classifier separately saved','independent_classifier':True})
    return catalog,pins,pre,identity

def verify(pins):
    for n,h in pins.items():
        if sha(ROOT/n)!=h:raise ValueError('Frozen evaluation source/artifact changed:'+n)

def load(entry,pre,identity):
    folder=ROOT/entry['folder']
    if entry['family']=='R9_NN':
        from pipeline.runner_r9 import restore_neural
        m,_=restore_neural(folder,device='cpu',fixture=False,expected_identity=pre['binding']['selection_identity']);return m,None
    if entry['family']=='R14_NN':
        from pipeline.training_r14_production import restore_completed_model,restore_checkpoint
        if entry['kind']!='tfgrid_fdy_conformer':m,front,_=restore_completed_model(folder,device='cpu')
        else:
            m,front,ck=restore_checkpoint(folder/'latest.pt',expected_config_sha256=identity['config_sha256'],expected_fold=0,
                expected_data_hash=identity['data_hash'],expected_basis_file_sha256='8cc263b3a43f0715daf755bed142b2e45b4373030a1e89fa7c499d77283d2d8d',
                expected_approval_sha256=identity['approval_sha256'],device='cpu')
            if ck['epoch']!=37:raise ValueError('Only paused complete epoch37, not active epoch38')
        return m,front
    if entry['family']=='R9_tree':from pipeline.trees_ensemble_r9 import PhysicsLGBM
    else:from pipeline.trees_ensemble_r13 import PhysicsLGBM
    s=read(folder/'summary.json');m=PhysicsLGBM.load(folder,expected_binding=s['training_binding'])
    # Runtime parallelism only; no change to saved weights/settings or source.
    m.settings=dict(m.settings,threads=2);return m,None

def predict(entry,model,frontend,db,batch=64):
    import torch
    from pipeline.relative_grid_r6_v2 import relative_grid_v2
    out={'share':[],'score':[]};maxgap=0.;maxmissing=0.
    if entry['family'].endswith('tree'):
        pr=model.predict(db,batch_size=batch)
        return {'share':pr['share'],'score':pr['score'],'raw_classifier':pr['score_classifier']},{'physical_pool_gap':None,'presence_independent':True}
    for start in range(0,len(db),batch):
        b=np.asarray(db[start:start+batch],np.float32)
        if entry['family']=='R9_NN':
            x,w=relative_grid_v2(b);tx=torch.from_numpy(x);tw=torch.from_numpy(w)
        else:
            with torch.no_grad():tx,tw=frontend(torch.from_numpy(b))
        with torch.no_grad():p,z,q=model(tx,tw)
        score=z.sigmoid() if entry['family']=='R9_NN' else z
        gap=float(abs(p-(q.float()*tw.float()).sum((1,2))).max());maxgap=max(maxgap,gap)
        if (~torch.isfinite(torch.from_numpy(b))).any():maxmissing=max(maxmissing,float(q[~torch.isfinite(torch.from_numpy(b))].abs().max()))
        if gap>3e-6 or maxmissing!=0:raise ValueError('Frozen physical pooling/missing contract failed')
        out['share'].append(p.cpu().numpy().astype(np.float32));out['score'].append(score.cpu().numpy().astype(np.float32))
    out={k:np.concatenate(v) for k,v in out.items()}
    if any(not np.isfinite(v).all() or ((v<0)|(v>1)).any() for v in out.values()):raise ValueError('Invalid frozen predictions')
    return out,{'physical_pool_max_abs':maxgap,'missing_q_max_abs':maxmissing,'presence_independent':False}

def auc(y,p):
    y=np.asarray(y)>0;p=np.asarray(p,float);n1=int(y.sum());n0=len(y)-n1
    if not n1 or not n0:return None
    ix=np.argsort(p,kind='stable');x=p[ix];rank=np.empty(len(x),float);lo=np.r_[0,np.flatnonzero(x[1:]!=x[:-1])+1];hi=np.r_[lo[1:],len(x)]
    for a,b in zip(lo,hi):rank[a:b]=(a+1+b)/2
    return float((rank[y[ix]].sum()-n1*(n1+1)/2)/(n1*n0))
def metrics(y,pr):
    y=np.asarray(y,float);s=pr['share'].astype(float);e=s-y
    return {'n':len(y),'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(abs(e).mean()),'bias':float(e.mean()),
            'share_mean':float(s.mean()),'target_mean':float(y.mean()),'share_AUC':auc(y,s),'own_score_AUC':auc(y,pr['score']),
            'raw_classifier_AUC':auc(y,pr['raw_classifier']) if 'raw_classifier'in pr else None}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['development','locked']);p.add_argument('--manifest',default='reports/r16_cross_operator_r1/data/manifest.json');p.add_argument('--freeze-receipt');p.add_argument('--output',default=OUT);p.add_argument('--batch',type=int,default=64);a=p.parse_args()
    if a.batch<1 or a.batch>128:raise ValueError('Bounded CPU microbatch required')
    if a.stage=='locked' and not a.freeze_receipt:raise PermissionError('Gate/report/domain require explicitly frozen new global policy receipt')
    if not os.environ.get('DYLD_LIBRARY_PATH','').startswith('/opt/homebrew/opt/libomp/lib'):raise ValueError('Pinned single OpenMP environment required before imports')
    sys.addaudithook(guard)
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    out=(ROOT/a.output/a.stage).resolve()
    if not out.is_relative_to(ROOT/'reports') or out.exists():raise FileExistsError('Fresh stage folder required; no failed partial reuse')
    catalog,pins,pre,identity=pins_model_catalog();manifest=(ROOT/a.manifest).resolve();pins[str(manifest.relative_to(ROOT))]=sha(manifest)
    dm=read(manifest)
    pins['ops/r16_forward_data.py']=sha(ROOT/'ops/r16_forward_data.py')
    if dm.get('code_sha256')!=pins['ops/r16_forward_data.py']:raise ValueError('Prepared data builder source differs')
    pins[str((manifest.parent/'readiness.json').relative_to(ROOT))]=sha(manifest.parent/'readiness.json')
    for n,h in dm['source_pins'].items():
        if sha(ROOT/n)!=h:raise ValueError('Data source provenance changed:'+n)
        pins[n]=h
    if a.stage=='locked':
        receipt=read(ROOT/a.freeze_receipt)
        if (receipt.get('status')!='completed' or receipt.get('all_models_frozen')is not True
            or receipt.get('ensemble_policy_frozen')is not True or receipt.get('data_manifest_sha256')!=sha(manifest)
            or not receipt.get('artifact_sha256')):raise PermissionError('Global model/policy freeze certificate invalid')
        for n,h in receipt['artifact_sha256'].items():
            path=(ROOT/n).resolve()
            if not path.is_relative_to(ROOT) or sha(path)!=h:raise PermissionError('Global frozen artifact changed')
            pins[n]=h
    for role in ROLES[a.stage]:
        for operator in OPERATORS:
            e=dm['panels'][role][operator];path=manifest.parent/e['path']
            if sha(path)!=e['sha256']:raise ValueError('Declared panel bytes differ')
            pins[str(path.relative_to(ROOT))]=e['sha256']
    if a.stage=='locked' and not a.freeze_receipt:raise PermissionError('Gate/report/domain require explicitly frozen new global policy receipt')
    if a.freeze_receipt:pins[a.freeze_receipt]=sha(ROOT/a.freeze_receipt)
    out.mkdir(parents=True)
    plan={'format':'r16_frozen_crossoperator_evaluation_v1','stage':a.stage,'roles':list(ROLES[a.stage]),'operators':list(OPERATORS),
          'manifest':str(manifest.relative_to(ROOT)),'manifest_sha256':sha(manifest),'model_catalog':catalog,'source_artifact_sha256':pins,
          'request_sha256':sha(ATTACHMENT),'CPU_threads':2,'FP32_backbone':True,'batch':a.batch,'label_inputs_to_model':False,
          'Test_read_fit_calibration_policy_selection':False,'TF_epoch37_is_not_completed50_budget':True,
          'known_reference_limit':'Old models already used all509references; new A/B/C partitions cannot make them reference-unseen',
          'tree_runtime_threads':'loaded memory object4→2 for R9 only, saved weights/settings unchanged',
          'score_semantics':'R9NN=derivedcontrast+share;R14NN=share only;R9Tree=.5share+.5rawclf;R14Tree default=share;rawclf separately stored',
          'created_unix':time.time()}
    write(out/'plan.json',plan);started=time.time()
    try:
        from ops.r16_forward_data import load_panel
        panels=[]
        for role in ROLES[a.stage]:
            for operator in OPERATORS:
                data=load_panel(manifest,role,operator,freeze_receipt=ROOT/a.freeze_receipt if a.freeze_receipt else None)
                if len(data['share'])!=400 or data['db'].shape!=(400,15,25):raise ValueError('Declared400panel shape')
                if any(np.asarray(v).dtype.kind=='O' for v in data.values() if isinstance(v,np.ndarray)):raise TypeError('Numeric/Unicode arrays only, no object')
                panels.append((role,operator,data));print(json.dumps({'stage':'panel_loaded','role':role,'operator':operator}),flush=True)
        db=np.concatenate([d['db'] for _,_,d in panels]);predictions={};timings={};pool={}
        for entry in catalog:
            verify(pins);begin=time.perf_counter();model,front=load(entry,pre,identity);loaded=time.perf_counter();pr,qa=predict(entry,model,front,db,a.batch)
            predictions[entry['name']]=pr;timings[entry['name']]={'load_seconds':loaded-begin,'predict_seconds':time.perf_counter()-loaded};pool[entry['name']]=qa
            del model,front;gc.collect();verify(pins)
            print(json.dumps({'stage':'model_complete','model':entry['name'],**timings[entry['name']]}),flush=True)
            write(out/'progress.json',{'status':'running','completed_models':list(predictions),'total_models':len(catalog),'elapsed_seconds':time.time()-started})
        def avg(name,names):
            keys=set.intersection(*[set(predictions[n]) for n in names]);predictions[name]={k:np.mean([predictions[n][k].astype(float) for n in names],axis=0).astype(np.float32) for k in keys}
        avg('R9_tree5',[f'R9_tree_f{i}'for i in range(5)])
        for v in ('baseline','highshare'):avg('R14_tree5_'+v,[f'R14_tree_{v}_f{i}'for i in range(5)])
        # Exactly the existing R9 frozen all5 combines fourNN plus Fold0tree.
        w9=read(ROOT/'submissions/r9_fold0_all5_cpu_r1/frozen_policy.json')['weights'];members=['R9_'+k for k in KINDS]+['R9_tree_f0']
        predictions['R9_all5_frozen_Fold0']={k:np.stack([predictions[n][k].astype(float) for n in members],1).dot(w9).astype(np.float32) for k in ('share','score')}
        # Existing R14fast3 policy is applied unchanged to exactFold0 and declaredTree5extension.
        from pipeline.ensemble_r14_production import apply_policy
        pol=read(ROOT/'models/r14_production_outputs_r2/hybrid_fast3/policy.json')
        for suffix,tn in [('exactFold0',lambda v:f'R14_tree_{v}_f0'),('tree5extension',lambda v:'R14_tree5_'+v)]:
            shares={'tree_'+v:predictions[tn(v)]['share'] for v in ('baseline','highshare')}
            shares.update({'nn_'+k:predictions['R14_'+k]['share'] for k in ('itransformer','fdy','bg_cnn')})
            cl={v:predictions[tn(v)]['raw_classifier']for v in ('baseline','highshare')};cl['mean']=(.5*cl['baseline'].astype(float)+.5*cl['highshare'].astype(float)).astype(np.float32)
            predictions['R14_fast3_frozen_'+suffix]=apply_policy(pol,shares,cl)
        rows=[];filemap={};modelnames=list(predictions)
        for i,(role,operator,data) in enumerate(panels):
            sl=slice(i*400,(i+1)*400);fd=out/'predictions';fd.mkdir(exist_ok=True);path=fd/(role+'__'+operator+'.npz')
            payload={k:np.asarray(data[k],dtype=str)for k in ('sample_id','site','background_group') if k in data};payload.update(target_share=np.asarray(data['share']),target_presence=np.asarray(data['presence']),model_order=np.asarray(modelnames,dtype=str),share=np.column_stack([predictions[n]['share'][sl]for n in modelnames]),own_score=np.column_stack([predictions[n]['score'][sl]for n in modelnames]))
            raw_names=[n for n in modelnames if 'raw_classifier'in predictions[n]];payload['classifier_model_order']=np.asarray(raw_names,dtype=str);payload['raw_classifier']=np.column_stack([predictions[n]['raw_classifier'][sl]for n in raw_names])
            for k in ('base_window_index','reference_index','reference_component','family','explicit_no_source','pair_id'):
                if k in data:
                    x=np.asarray(data[k]);payload[k]=x.astype(str)if x.dtype.kind in 'OUS' else x
            if any(v.dtype.kind=='O'for v in payload.values()):raise TypeError('Object payload forbidden')
            np.savez_compressed(path,**payload);filemap[str(path.relative_to(out))]=sha(path)
            for name in modelnames:rows.append({'role':role,'operator':operator,'model':name,**metrics(data['share'],{k:v[sl]for k,v in predictions[name].items()})})
        with(out/'metrics.csv').open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        verify(pins)
        result={'status':'completed_frozen_prediction_evaluation','stage':a.stage,'panels':len(panels),'rows_total':len(db),'individual_models':len(catalog),'reported_model_and_frozen_aggregate_count':len(modelnames),'model_order':modelnames,'source_artifact_sha256':pins,'model_catalog':catalog,'timings':timings,'physical_mask_QA':pool,'metrics_file':'metrics.csv','output_sha256':{**filemap,'metrics.csv':sha(out/'metrics.csv')},'elapsed_seconds':time.time()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'new_fit_or_policy_selection_or_Test_read':False,'TF37_diagnostic_only':True,'reference_unseen_for_old_models':False,'R14_NN_independent_presence_available':False,'frozen_R14_score_formula':'fast3=.75share+.25highshareclassifier; preserved prior policy, notnewlyfit'}
        write(out/'RESULTS.json',result);write(out/'progress.json',{'status':'completed','elapsed_seconds':time.time()-started});print(json.dumps({'status':'completed','results':str(out/'RESULTS.json'),'seconds':result['elapsed_seconds']}),flush=True)
    except BaseException as e:
        write(out/'FAILED.json',{'status':'failed','error':repr(e),'no_partial_resume_or_source_patch':True});raise

if __name__=='__main__':main()
