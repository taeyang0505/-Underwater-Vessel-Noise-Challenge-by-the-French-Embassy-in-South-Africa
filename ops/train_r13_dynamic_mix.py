"""One prospective paired-epoch mixture condition; old R12/R13 are immutable.

CNN sees each primary row once/epoch; each changed row sees baseline/high
exactly once in every epoch pair. Tree sees both 12000-row variants (24000
augmentation rows; common rows are intentionally duplicated). Only fixed
validation_fit is used before both learners freeze. No Test I/O.
"""
from __future__ import annotations
import argparse,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SEED=2026102712
DATA='data/synthetic/r13_cpu_correction_v1'
OUT='models/r13_dynamic_mix_r1'
REPORT='reports/r13_dynamic_mix_r1'
BASIS='models/r12_shared_basis_v1/basis.npz'
EXPECTED_DATA='5c30c3fec031afbae5d97de46d9e1cc4857976410e11a6be780a60b4bc752662'
BASIS_SHA='a176bedfd772e552bb3fb310449b699b23d2fb9445af071ed0f72b05a1205113'

def imports(threads):
    from ops.train_r12_lightgbm import environment,guard
    environment(threads)
    sys.addaudithook(guard)
    import numpy as np,torch
    torch.set_num_threads(threads)
    return np,torch

def write(path,v):
    from pipeline.training_r12 import write as atomic
    atomic(path,v)

def sha(path):
    from pipeline.training_r12 import sha as digest
    return digest(path)

def token(v):
    from ops.train_r12_lightgbm import token as digest
    return digest(v)

def arrays():
    import numpy as np
    from pipeline.dataset_r13 import validate_r13_bundle,load_r13_arrays
    from pipeline.features_r12 import load_spectral_basis
    m=validate_r13_bundle(ROOT/DATA)
    if m['data_hash']!=EXPECTED_DATA or sha(ROOT/BASIS)!=BASIS_SHA:raise ValueError('Approved data/basis changed')
    baseline=load_r13_arrays(ROOT/DATA,'baseline');high=load_r13_arrays(ROOT/DATA,'highshare')
    subset=np.load(ROOT/'data/synthetic/r12_cpu_ablation_v3/cnn_subset_indices.npy',allow_pickle=False)
    e=m['validation_references']['validation_fit']
    if sha(ROOT/e['path'])!=e['sha256']:raise ValueError('Validation fit identity changed')
    with np.load(ROOT/e['path'],allow_pickle=False) as z:val={k:z[k] for k in z.files}
    basis=load_spectral_basis(ROOT/BASIS,expected_sha256=BASIS_SHA,expected_training_binding=m['basis_training_binding'])
    return m,baseline,high,subset,val,basis

def schedule(changed,n,seed=SEED):
    import numpy as np
    changed=np.asarray(changed,np.int64)
    if len(changed)%2 or len(set(changed.tolist()))!=len(changed):raise ValueError('Even unique changed indices required')
    choices=np.zeros((100,n),bool)
    for pair in range(50):
        order=np.random.default_rng(np.random.SeedSequence([seed,13001,pair])).permutation(changed)
        choices[2*pair,order[:len(changed)//2]]=True
        choices[2*pair+1,order[len(changed)//2:]]=True
    if not np.all(choices[:,changed].sum(0)==50):raise AssertionError('Each changed row must see high50/baseline50')
    if any(not np.array_equal(choices[e,changed]^choices[e+1,changed],np.ones(len(changed),bool)) for e in range(0,100,2)):
        raise AssertionError('Paired complements required')
    return choices

def verify_sources(plan):
    for n,h in plan['source_pins'].items():
        if sha(ROOT/n)!=h:raise ValueError('Frozen source changed:'+n)

def prepare(out):
    import numpy as np
    from ops.train_r12_lightgbm import array_sha,dataset
    m,b,h,subset,val,basis=arrays()
    dataset(b,'train',expected_rows=12000);dataset(h,'train',expected_rows=12000);dataset(val,'validation_fit',expected_rows=1000)
    if set(b)!=set(h):raise ValueError('Variant fields differ')
    for k in b:
        if k not in ('db','unrounded_db','share','cell_share'):
            equal=np.array_equal(b[k],h[k],equal_nan=True) if b[k].dtype.kind in 'fc' else np.array_equal(b[k],h[k])
            if not equal:raise ValueError('Paired source/geometry/presence changed:'+k)
    changed_full=np.flatnonzero(b['share']!=h['share'])
    changed=np.flatnonzero(b['share'][subset]!=h['share'][subset])
    if len(subset)!=6000 or len(changed_full)!=2400 or len(changed)!=1200:raise ValueError('Expected6000 primary/1200 changed/2400 full changed')
    if not np.all((b['share'][subset][changed]>0)&(h['share'][subset][changed]>0)):raise ValueError('Only positive amplitudes may change')
    # Values of all nonselected rows and original zero windows are identical.
    unchanged=np.ones(len(b['share']),bool);unchanged[changed_full]=False
    for k in ('db','unrounded_db','share','cell_share'):
        if not np.array_equal(b[k][unchanged],h[k][unchanged],equal_nan=True):raise ValueError('Untouched rows changed:'+k)
    choices=schedule(changed,6000)
    rows={k:v[subset] for k,v in b.items()}
    zero=int((rows['share']==0).sum());long=int(np.char.startswith(rows['family'].astype(str),'long_').sum())
    if zero!=1200 or long!=1440:raise ValueError('Original zero/long schedule differs')
    paths=[Path(__file__),ROOT/'pipeline/training_r12.py',ROOT/'pipeline/features_r12.py',ROOT/'ops/train_r12_lightgbm.py',
           ROOT/'pipeline/dataset_r13.py',ROOT/'pipeline/dataset_r12.py',ROOT/'pipeline/relative_models_r6_v2.py',
           ROOT/'pipeline/relative_grid_r6_v2.py',ROOT/'pipeline/relative_grid_r6.py',ROOT/BASIS,
           ROOT/DATA/'manifest.json',ROOT/DATA/'readiness.json',ROOT/'data/synthetic/r12_cpu_ablation_v3/cnn_subset_indices.npy',
           ROOT/e['path'] if (e:=m['validation_references']['validation_fit']) else None]
    paths += [ROOT/m['arms'][a][k] for a in ('baseline','highshare') for k in ('path','metadata_path','recipes_path')]
    paths += [ROOT/'ops/evaluate_r13_cpu.py',ROOT/'ops/evaluate_r12_ablation.py',ROOT/'ops/verify_r12_evaluation.py',
              ROOT/'reports/r13_cpu_correction_r1/RESULTS.json',ROOT/'reports/r13_cpu_correction_r1/selected_policy.json']
    plan={'format':'r13_paired_epoch_mixture_single_condition_plan_v1','condition':'R13_dynamic_mixP1',
        'created_before_fitting':True,'seed':SEED,'data_hash':m['data_hash'],'manifest_sha256':sha(ROOT/DATA/'manifest.json'),
        'basis_path':BASIS,'basis_sha256':BASIS_SHA,'basis_training_binding':m['basis_training_binding'],
        'source_pins':{str(p.relative_to(ROOT)):sha(p) for p in paths},
        'CNN':{'primary_rows':6000,'changed_positive_rows':1200,'high_rows_per_epoch':600,'epochs':100,'early_stopping':False,
            'pair_policy':'per deterministic pair,600 changed rows high then complementary600; every changed row baseline/high exactly once per2epochs',
            'zero_rows_each_epoch':zero,'long_rows_each_epoch':long,'all_primary_rows_once_per_epoch':True,'threads':4,
            'model_parameters':26953,'dropout':.1,'optimizer':'Adam','loss':'shareMSE+.25energyWeightedCellMSE',
            'batch_size':128,'lr':.001,'weight_decay':.0003,'scheduler':'cosine100 to1e-5','clip_grad_norm':1.,
            'choices_typed_sha256':array_sha(choices),'changed_subset_indices_sha256':array_sha(changed)},
        'tree':{'rows':24000,'unique_primary_rows':12000,'baseline_rows':12000,'highshare_rows':12000,'features':673,
            'unchanged_primary_rows_duplicated':9600,'changed_primary_rows_two_variants':2400,
            'sample_id_policy':'append __baseline or __highshare; retain original sample ID in paired_original_sample_id',
            'maximum_trees':1500,'early_stopping':40,'threads':2,'seed':SEED,'independent_classifier':True},
        'evaluation':{'fixed_regression_weights':{'CNN':.5,'tree':.5},'classifier_lambda_candidates':[0.,.25,.5,1.],
            'gate':'R12 validation_gate500; choose score lambda highest AUC first tie after both models frozen',
            'recipe_selection':'none; one prospective condition only','panels':17,'development_not_blind':True,
            'comparators':'saved R12 matched share only / R13 selected / R9 if available; no reranking originals'},
        'original_policy_models_data_unchanged':True,'Test_or_GPU_or_new_source_used':False}
    plan['plan_sha256']=token(plan)
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'epoch_choices.npz',high_choice=choices,changed_subset_index=changed,primary_index=subset)
    plan['epoch_choices_file_sha256']=sha(out/'epoch_choices.npz')
    # The plan identity includes its serialized choice artifact.
    plan['plan_sha256']=token({k:v for k,v in plan.items() if k!='plan_sha256'})
    write(out/'PRE_FIT_FREEZE.json',plan)
    return plan

def load_plan(out):
    p=json.loads((out/'PRE_FIT_FREEZE.json').read_text())
    if token({k:v for k,v in p.items() if k!='plan_sha256'})!=p['plan_sha256']:raise ValueError('Plan changed')
    if sha(out/'epoch_choices.npz')!=p['epoch_choices_file_sha256']:raise ValueError('Schedule changed')
    verify_sources(p);return p

def fit_cnn(out):
    np,torch=imports(4)
    from pipeline.training_r12 import R12MaskCNN,channels,objective,predict,metrics
    from torch import nn
    plan=load_plan(out);m,b,h,subset,val,basis=arrays()
    with np.load(out/'epoch_choices.npz',allow_pickle=False) as z:choices=z['high_choice']
    bx,bw=channels(b['db'][subset],basis,True);hx,hw=channels(h['db'][subset],basis,True);vx,vw=channels(val['db'],basis,True)
    bx,bw,hx,hw,vx,vw=[torch.from_numpy(v) for v in (bx,bw,hx,hw,vx,vw)]
    by,hy,bq,hq=[torch.from_numpy(np.asarray(v,np.float32)) for v in (b['share'][subset],h['share'][subset],b['cell_share'][subset],h['cell_share'][subset])]
    if max(float(abs((w*q).sum((1,2))-y).max()) for w,q,y in [(bw,bq,by),(hw,hq,hy)])>.0058:raise ValueError('Physical rounded pooling bound')
    folder=out/'cnn';folder.mkdir();torch.manual_seed(SEED)
    model=R12MaskCNN().to(memory_format=torch.channels_last)
    if sum(t.numel() for t in model.parameters())!=26953:raise ValueError('Same26953bodyrequired')
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=1e-5)
    arm={'name':'R13_dynamic_mixP1','shape':True,'legacy_control':False,'paired_epoch_mixture':True}
    write(folder/'STARTED.json',{'arm':arm,'epochs':100,'optimizer':'Adam','early_stopping':False,'parameters':26953,'plan_sha256':plan['plan_sha256']})
    best=float('inf');best_epoch=0;beg=time.perf_counter()
    for epoch in range(1,101):
        model.train();mask=torch.from_numpy(choices[epoch-1]);order=np.random.default_rng(np.random.SeedSequence([SEED,epoch])).permutation(6000)
        ss=total=0.;count=0
        for start in range(0,6000,128):
            ix=order[start:start+128];a=mask[ix];xx=torch.where(a[:,None,None,None],hx[ix],bx[ix]).contiguous(memory_format=torch.channels_last)
            ww=torch.where(a[:,None,None],hw[ix],bw[ix]);y=torch.where(a,hy[ix],by[ix]);q=torch.where(a[:,None,None],hq[ix],bq[ix])
            optimizer.zero_grad(set_to_none=True);p,guess=model(xx,ww);rows,mse,_=objective(p,guess,ww,y,q);loss=rows.mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            ss+=float(mse.detach().sum());total+=float(rows.detach().sum());count+=len(ix)
        pred=predict(model,vx,vw,128);vm=metrics(val['share'],pred)
        if vm['rmse']<best:
            best=vm['rmse'];best_epoch=epoch
            tmp=folder/'best.tmp.pt';torch.save({'state_dict':model.state_dict(),'arm':arm,'epoch':epoch,'parameters':26953,'plan_sha256':plan['plan_sha256']},tmp);tmp.replace(folder/'best.pt')
        row={'epoch':epoch,'train_online_rmse':float(np.sqrt(ss/count)),'train_loss':total/count,'validation_fit_rmse':vm['rmse'],
             'validation_fit_auc_share':vm['auc_share'],'best_rmse':best,'best_epoch':best_epoch,'learning_rate':optimizer.param_groups[0]['lr'],
             'high_rows':int(mask.sum()),'primary_rows':count,'elapsed_seconds':time.perf_counter()-beg}
        with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        write(folder/'progress.json',row)
        if epoch%10==0 or epoch==1:print(json.dumps({'stage':'cnn',**row}),flush=True)
        scheduler.step()
    ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict'],strict=True)
    pred=predict(model,vx,vw,128);np.savez_compressed(folder/'validation_fit_predictions.npz',**pred)
    summary={'format':'r13_dynamic_mix_cnn_v1','status':'completed','arm':arm,'epochs_run':100,'best_epoch':best_epoch,
        'early_stopping':False,'optimizer':'Adam','dropout':.1,'parameters':26953,'loss':'shareMSE+.25weightedCellMSE',
        'learning_rate_schedule':'cosine .001->.00001 fixed100','plan_sha256':plan['plan_sha256'],'score':'share',
        'validation_fit_metrics':metrics(val['share'],pred,val.get('site')),'checkpoint_sha256':sha(folder/'best.pt'),
        'elapsed_seconds':time.perf_counter()-beg,'Test_gate_eval_read':False,'source_pins':plan['source_pins']}
    verify_sources(plan);write(folder/'summary.json',summary);return summary

def fit_tree(out):
    np,torch=imports(2)
    # The explicit24k subclass below is copied from the immutable parent fit;
    # only its train-row requirement differs. No fixture label is used for fits.
    plan=load_plan(out);m,b,h,subset,val,basis=arrays()
    train={k:np.concatenate([b[k],h[k]]) for k in b}
    original=train['sample_id'].astype(str).copy();train['paired_original_sample_id']=original
    train['augmentation_variant']=np.repeat(np.asarray(['baseline','highshare']),12000)
    train['sample_id']=np.char.add(original,np.char.add('__',train['augmentation_variant']))
    binding={'format':'r13_dynamic_mix_24k_tree_binding_v1','plan_sha256':plan['plan_sha256'],'data_hash':m['data_hash'],
        'baseline_TRAIN_sha256':m['arms']['baseline']['sha256'],'highshare_TRAIN_sha256':m['arms']['highshare']['sha256'],
        'validation_fit_sha256':m['validation_references']['validation_fit']['sha256'],'basis_sha256':BASIS_SHA,
        'two_augmentation_variants_same_source':True,'independent_source_rows':12000,'augmentation_rows':24000,
        'execution_source_sha256':sha(__file__),'development_not_blind':True}
    tree=DynamicMixTree(include_shape=True,basis_path=ROOT/BASIS,basis_sha256=BASIS_SHA,
        basis_training_binding=m['basis_training_binding'],settings={'threads':2},seed=SEED)
    s=tree.fit(train,val,out/'tree',binding,fixture=False)
    restored=DynamicMixTree.load(out/'tree',expected_binding=binding)
    p=tree.predict(val['db'][:7],batch_size=7);r=restored.predict(val['db'][:7],batch_size=1)
    if any(not np.array_equal(p[k],r[k]) for k in p):raise AssertionError('Tree reload1/7 parity')
    verify_sources(plan);write(out/'TREE_COMPLETED.json',{'status':'completed','summary_sha256':sha(out/'tree/summary.json'),'binding':binding})
    print(json.dumps({'stage':'tree_completed','rmse':s['validation_fit_metrics']['rmse'],'best_iteration':s['best_iteration'],'elapsed_seconds':s['elapsed_seconds']}),flush=True)
    return s

def fixture():
    np,torch=imports(2)
    c=schedule(np.arange(2,10),12)
    assert c.shape==(100,12) and np.all(c.sum(1)==4) and np.all(c[:,2:10].sum(0)==50)
    assert not c[:,:2].any() and not c[:,10:].any()
    from pipeline.training_r12 import R12MaskCNN,objective
    torch.manual_seed(SEED);model=R12MaskCNN();opt=torch.optim.Adam(model.parameters(),lr=.001)
    x=torch.randn(4,11,15,25);w=torch.rand(4,15,25);w/=w.sum((1,2),keepdim=True);q=torch.rand_like(w);y=(w*q).sum((1,2))
    p,guess=model(x,w);loss=objective(p,guess,w,y,q)[0].mean();loss.backward();opt.step()
    assert torch.isfinite(loss) and all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    return {'status':'passed','synthetic_only':True,'schedule_complement_pairs':50,'unchanged_rows_never_high':True,
            'parameter_count':sum(p.numel() for p in model.parameters()),'one_Adam_step_finite':True,'source_values_or_Test_used':False}

def freeze(out):
    plan=load_plan(out)
    if (out/'FAILED.json').exists():raise ValueError('Failed run not reusable')
    c=json.loads((out/'cnn/summary.json').read_text());t=json.loads((out/'tree/summary.json').read_text())
    if c['status']!='completed' or c['epochs_run']!=100 or t['status']!='completed' or t['train_rows']!=24000:raise ValueError('Both complete models required')
    paths=[p for sub in ('cnn','tree') for p in (out/sub).iterdir() if p.is_file()]
    frozen={'format':'r13_dynamic_mix_all2_frozen_v1','status':'completed','all_models_frozen':True,'plan_sha256':plan['plan_sha256'],
        'source_pins':plan['source_pins'],'artifact_sha256':{str(p.relative_to(out)):sha(p) for p in paths},
        'gate_eval_or_Test_read':False}
    frozen['freeze_sha256']=token(frozen);write(out/'ALL_MODELS_FROZEN.json',frozen);return frozen

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['fixture','prepare','cnn','tree','fit','evaluate'])
    p.add_argument('--output',default=OUT);p.add_argument('--report',default=REPORT);a=p.parse_args();out=(ROOT/a.output).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Project-contained fresh outputs only')
    if a.action=='fixture':print(json.dumps(fixture()));return
    if a.action=='prepare':imports(2);plan=prepare(out);print(json.dumps({'status':'plan_frozen','plan_sha256':plan['plan_sha256']}));return
    if a.action=='cnn':fit_cnn(out);return
    if a.action=='tree':fit_tree(out);return
    if a.action=='evaluate':evaluate(out,(ROOT/a.report).resolve());return
    if a.action=='fit':
        imports(2);load_plan(out)
        if any((out/k).exists() for k in ('cnn','tree','FAILED.json','ALL_MODELS_FROZEN.json')):raise FileExistsError('Fresh prepared run only')
        handles=[];processes=[]
        try:
            for action,threads in [('cnn',4),('tree',2)]:
                env=os.environ.copy()
                for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]=str(threads)
                log=(out/(action+'.log')).open('x');handles.append(log)
                processes.append(subprocess.Popen([sys.executable,'-m','ops.train_r13_dynamic_mix',action,'--output',str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
            write(out/'RUNNING.json',{'started_unix':time.time(),'pids':{'cnn':processes[0].pid,'tree':processes[1].pid},'threads':[4,2]})
            codes=[p.wait() for p in processes]
            if codes!=[0,0]:raise RuntimeError('Preserved partial failure:'+repr(codes))
            frozen=freeze(out);print(json.dumps({'status':'all2_frozen','freeze_sha256':frozen['freeze_sha256']}),flush=True)
        except BaseException as e:
            write(out/'FAILED.json',{'error':repr(e),'partial_reuse_allowed':False});raise
        finally:
            for f in handles:f.close()

def evaluate(out,report):
    np,torch=imports(2)
    import pandas as pd
    from ops import evaluate_r12_ablation as parent
    from ops.evaluate_r13_cpu import mix_score
    from ops.verify_r12_evaluation import cluster_ci
    from pipeline.training_r12 import R12MaskCNN,channels,predict
    plan=load_plan(out);f=json.loads((out/'ALL_MODELS_FROZEN.json').read_text())
    if ((out/'FAILED.json').exists() or f.get('all_models_frozen') is not True
        or token({k:v for k,v in f.items() if k!='freeze_sha256'})!=f['freeze_sha256']):raise ValueError('Both new models must freeze first')
    for n,h in f['artifact_sha256'].items():
        if sha(out/n)!=h:raise ValueError('Frozen model changed:'+n)
    m,b,h,subset,val,basis=arrays()
    cs=json.loads((out/'cnn/summary.json').read_text());ts=json.loads((out/'tree/summary.json').read_text())
    history=[json.loads(x) for x in (out/'cnn/epochs.jsonl').read_text().splitlines()]
    if (cs['epochs_run']!=100 or [r['epoch'] for r in history]!=list(range(1,101))
        or cs['best_epoch']!=int(np.argmin([r['validation_fit_rmse'] for r in history]))+1
        or any(r['high_rows']!=600 or r['primary_rows']!=6000 for r in history)):
        raise ValueError('Actual fixed100 paired schedule incomplete')
    ck=torch.load(out/'cnn/best.pt',map_location='cpu',weights_only=True)
    if ck['epoch']!=cs['best_epoch'] or ck['plan_sha256']!=plan['plan_sha256'] or sha(out/'cnn/best.pt')!=cs['checkpoint_sha256']:
        raise ValueError('Best checkpoint binding changed')
    nn=R12MaskCNN();nn.load_state_dict(ck['state_dict'],strict=True);nn.eval()
    tree=DynamicMixTree.load(out/'tree',expected_binding=ts['training_binding'])
    if ts['train_rows']!=24000 or ts['feature_count']!=673 or ts['fixture_only']:raise ValueError('Actual24k/P1tree required')
    oldplan,oldf=parent.check_freeze('reports/R12_ALL17_GLOBAL_FREEZE.json')
    oldpolicy=ROOT/'reports/r12_cpu_ablation_r1/selected_policy.json'
    parent.check_policy(oldpolicy,oldf['freeze_sha256'])
    prior=ROOT/'reports/r13_cpu_correction_r1';previous=json.loads((prior/'RESULTS.json').read_text())
    for n,d in previous['output_sha256'].items():
        if sha(prior/n)!=d:raise ValueError('Original R13 output changed:'+n)
    report.mkdir(parents=True,exist_ok=False);(report/'predictions').mkdir()
    write(report/'FROZEN_BEFORE_GATE.json',{'all2_frozen_sha256':sha(out/'ALL_MODELS_FROZEN.json'),
        'freeze_sha256':f['freeze_sha256'],'plan_sha256':plan['plan_sha256'],'numeric_gate_opened':False,
        'single_condition_only':True,'original_R13_policy_unchanged':True})

    def infer(db):
        x,w=channels(db,basis,True);n=predict(nn,torch.from_numpy(x),torch.from_numpy(w),128);t=tree.predict(db,batch_size=128)
        gap=float(abs((w*n['cell_mask']).sum((1,2))-n['share']).max())
        if gap>3e-6 or np.any(n['cell_mask'][w==0]!=0):raise ValueError('Query physical pooling/missing mismatch')
        reg=(.5*n['share'].astype(float)+.5*t['share'].astype(float)).astype(np.float32)
        return n,t,reg,w,gap

    gate=parent.load_R12(oldplan,'validation_gate');gp=infer(gate['db'])
    lambdas=(0.,.25,.5,1.);aucs={str(v):parent.auc(gate['y'],mix_score(gp[2],gp[1]['score_classifier'],v)) for v in lambdas}
    lam=max(lambdas,key=lambda v:aucs[str(v)])
    policy={'format':'r13_dynamic_mix_single_condition_policy_v1','condition':'R13_dynamic_mixP1',
        'regression_weights':{'CNN':.5,'tree':.5},'score_classifier_weight':lam,'gate_score_AUC':aucs,
        'selection':'Only classifier lambda highest gate AUC, first canonical tie; one recipe, no recipe reranking',
        'freeze_sha256':f['freeze_sha256'],'plan_sha256':plan['plan_sha256'],'development_not_blind':True,'Test_or_LB_used':False}
    policy['policy_sha256']=token(policy);write(report/'selected_policy.json',policy);policyfile=sha(report/'selected_policy.json')
    panels=[gate]
    for role in ('eval','high_snr_long'):panels.append(parent.load_R12(oldplan,role,oldpolicy,oldf['freeze_sha256']))
    dev,r11=parent.load_development();panels.extend(dev)
    for p in panels:
        if p['name'].startswith('R10_unseen__'):
            meta=pd.read_parquet(ROOT/parent.R10/'unseen_source_manifest.parquet')
            if len(meta)!=len(p['y']):raise ValueError('Unseen metadata count')
            p['groups']=(meta.sensor_group.astype(str)+'__'+meta.recording_week_start.astype(str)).to_numpy()
            p['cluster_key']='sensor_group_x_recording_week_start'
    for pos in parent.DEV_R11_POS:
        for neg in parent.DEV_R11_NEG:
            x,y=r11[pos],r11[neg]
            panels.append({'name':x['name']+'_VS_'+neg,'y':np.r_[x['y'],y['y']],
                'db':np.concatenate([x['db'],y['db']]),'sample_id':np.r_[x['sample_id'],y['sample_id']],
                'groups':np.r_[x['groups'],y['groups']],'cluster_key':x['cluster_key'],'site':np.r_[x['site'],y['site']],
                'stage':'exposed_development'})
    if len(panels)!=17:raise ValueError('Exactly same17 panels required')
    rows=[];cis=[];pool_max=0.;started=time.perf_counter()

    def metric(p,r):
        result=parent.metrics(p,r);z=np.asarray(p['y'])>0;err=np.asarray(r['share'],float)-p['y'];high=p['y']>.2
        result.update(score_brier=float(np.mean((np.asarray(r['score'],float)-z)**2)),
            zero_prediction_mean=float(np.asarray(r['share'],float)[~z].mean()) if (~z).any() else None,
            high_y_bias=float(err[high].mean()) if high.any() else None,
            high_y_rmse=float(np.sqrt(np.mean(err[high]**2))) if high.any() else None,
            classifier_auc=parent.auc(p['y'],r['score_classifier']) if 'score_classifier' in r else None,
            classifier_brier=float(np.mean((np.asarray(r['score_classifier'],float)-z)**2)) if 'score_classifier' in r else None,
            share_ge099=float(np.mean(r['share']>=.99)),score_ge099=float(np.mean(r['score']>=.99)))
        return result

    for panel in panels:
        if sha(report/'selected_policy.json')!=policyfile:raise ValueError('New policy changed')
        name=panel['name'];n,t,reg,w,gap=gp if name==gate['name'] else infer(panel['db']);pool_max=max(pool_max,gap)
        predictions={'CNN_dynamic_mix':n,'Tree_dynamic_mix':t}
        for v in lambdas:predictions['Blend50_lambda'+str(v)]={'share':reg,'score':mix_score(reg,t['score_classifier'],v),'score_classifier':t['score_classifier']}
        predictions['DynamicMix_selected']=predictions['Blend50_lambda'+str(lam)]
        original=prior/'predictions'/(name+'.npz')
        with np.load(original,allow_pickle=False) as z:
            if not np.array_equal(z['target_share'],panel['y']) or not np.array_equal(z['sample_id'].astype(str),panel['sample_id'].astype(str)):
                raise ValueError('Comparator row/target identity changed')
            order=z['model_order'].astype(str).tolist()
            for key in ('R13_selected','R12_matched_share_only','R9_frozen_all5'):
                if key in order:
                    k=order.index(key);predictions[key]={'share':z['share'][:,k],'score':z['score'][:,k]}
        for key,value in predictions.items():rows.append({'panel':name,'model':key,'development_not_blind':True,**metric(panel,value)})
        for key in ('R13_selected','R12_matched_share_only','R9_frozen_all5'):
            if key in predictions:
                a=predictions['DynamicMix_selected'];b=predictions[key]
                ci=cluster_ci(name,panel['y'],panel['groups'],panel['cluster_key'],(a['share'],a['score']),(b['share'],b['score']))
                cis.extend({'panel':name,'comparison':'dynamic_minus_'+key,'metric':k,**v} for k,v in ci.items())
        np.savez_compressed(report/'predictions'/(name+'.npz'),sample_id=panel['sample_id'],target_share=panel['y'],
            model_order=np.asarray(list(predictions)),share=np.column_stack([v['share'] for v in predictions.values()]),
            score=np.column_stack([v['score'] for v in predictions.values()]),observed_weights=w,
            cnn_cell_mask=n['cell_mask'],cnn_contrast=n['contrast'],tree_classifier=t['score_classifier'])
        print(json.dumps({'stage':'evaluation','panel':name,**metric(panel,predictions['DynamicMix_selected'])}),flush=True)
    parent.csv_rows(report/'results.csv',rows);parent.csv_rows(report/'paired_cluster_CI.csv',cis)
    verify_sources(plan)
    for n,d in previous['output_sha256'].items():
        if sha(prior/n)!=d:raise ValueError('Original R13 outputs changed')
    for n,d in f['artifact_sha256'].items():
        if sha(out/n)!=d:raise ValueError('New frozen model changed')
    result={'format':'r13_single_dynamic_mix_evaluation_v1','status':'completed','panels':17,'metric_rows':len(rows),
        'selected_policy':policy,'plan_sha256':plan['plan_sha256'],'models_freeze_sha256':f['freeze_sha256'],
        'models_freeze_file_sha256':sha(out/'ALL_MODELS_FROZEN.json'),'source_pins':plan['source_pins'],
        'physical_q_pool_max_abs_error':pool_max,'raw_classifier_contrast_q_weights_saved':True,
        'original_R13_policy_outputs_unchanged':True,'original_R12_R13_code_data_models_unchanged':True,
        'new_policy_scope':'this single diagnostic condition only; no production promotion',
        'elapsed_evaluation_seconds':time.perf_counter()-started,'Test_or_GPU_used':False,
        'limitations':['Development-exposed17panels and shared references, not blind evaluation',
            'One fixed seed and one new condition; no post-result repeat permitted',
            'CNN6000once/epoch paired variants; tree24000augmentation rows from12000sources; not24000independent examples',
            'R10sensor-week bootstrap groups may have15min adjacentweek gaps; not24h-independent bootstrap blocks']}
    result['output_sha256']={str(p.relative_to(report)):sha(p) for p in report.rglob('*') if p.is_file()}
    write(report/'RESULTS.json',result)

# The literal fit implementation below preserves the original learner operators.

import shutil
from ops.train_r12_lightgbm import (CompactPhysicsLGBM,CODE_NAMES,dataset,transform,metrics,array_sha,utc)

class DynamicMixTree(CompactPhysicsLGBM):
    """Same immutable compact learner with explicit24000 augmentation rows."""
    def fit(self,train,validation_fit,output,binding,*,fixture=False):
        import numpy as np,lightgbm as lgb
        from pipeline.features_r12 import compact_feature_names
        td,ty,tz=dataset(train,'train',expected_rows=None if fixture else 24000)
        vd,vy,vz=dataset(validation_fit,'validation_fit',expected_rows=None if fixture else 1000)
        if len(np.unique(tz))!=2 or len(np.unique(vz))!=2:raise ValueError('BothpresenceclassesrequiredTRAINandvalidationfit')
        for key in ('sample_id','base_window_index','source_id','background_group'):
            if key in train and key in validation_fit and set(np.asarray(train[key]).astype(str))&set(np.asarray(validation_fit[key]).astype(str)):
                raise ValueError('TRAIN/validationfitsourceseparationfailed:'+key)
        if not isinstance(binding,Mapping) or not binding:raise ValueError('Explicitdataset/sourcebindingrequired')
        token(binding);folder=Path(output)
        if folder.exists():raise FileExistsError('Freshmodeloutputrequirednoreuse')
        folder.mkdir(parents=True);started=time.perf_counter()
        pins={n:sha(ROOT/n) for n in CODE_NAMES}
        plan={'format':'r12_compact_tree_fit_plan_v1','created_utc':utc(),'training_binding':dict(binding),'settings':self.settings,'seed':self.seed,
              'include_shape':self.include_shape,'basis_file_sha256':self.basis_sha256,'basis_training_binding':self.basis_training_binding,'feature_fit_on_synthetic_train':False,
              'scaler':'none;fixedrelativefeatures','PCA':'sharedTRAIN-onlypureSAbackgroundbasis;loadedfrozen' if self.include_shape else 'disabled',
              'input_identity':{'train':{k:array_sha(v) for k,v in train.items()},'validation_fit':{k:array_sha(v) for k,v in validation_fit.items()}},
              'source_code_sha256':pins,'fixture_only':bool(fixture),'gate_eval_Test_used_for_ES':False}
        write(folder/'fit_plan.json',plan)
        try:
            x=transform(td,basis=self.basis,include_shape=self.include_shape,batch_size=self.settings['feature_batch_size'])
            v=transform(vd,basis=self.basis,include_shape=self.include_shape,batch_size=self.settings['feature_batch_size'])
            cfg=self.settings;args=dict(n_estimators=cfg['maximum_trees'],learning_rate=cfg['learning_rate'],num_leaves=cfg['num_leaves'],
                min_child_samples=cfg['minimum_leaf_rows'],max_bin=127,n_jobs=cfg['threads'],random_state=self.seed,verbosity=-1,
                colsample_bytree=.85,subsample=.85,subsample_freq=1,deterministic=True,force_col_wise=True,
                data_random_seed=self.seed,feature_fraction_seed=self.seed,bagging_seed=self.seed)
            reg=lgb.LGBMRegressor(objective='regression_l2',metric='None',**args);clf=lgb.LGBMClassifier(objective='binary',metric='binary_logloss',**args)
            def clipped_rmse(truth,p):return 'clipped_share_rmse',float(np.sqrt(np.mean((np.clip(p,0,1)-truth)**2))),False
            reg.fit(x,ty,eval_set=[(v,vy)],eval_metric=clipped_rmse,callbacks=[lgb.early_stopping(cfg['early_stopping_patience'],verbose=False),lgb.log_evaluation(0)])
            clf.fit(x,tz,eval_set=[(v,vz)],callbacks=[lgb.early_stopping(cfg['early_stopping_patience'],verbose=False),lgb.log_evaluation(0)])
            self.regressor,self.classifier=reg.booster_,clf.booster_;self.regressor.save_model(str(folder/'share.txt'));self.classifier.save_model(str(folder/'presence.txt'))
            schema={'format':'r12_compact_relative_feature_schema_v1','feature_count':len(compact_feature_names(include_shape=self.include_shape)),
                'feature_names':list(compact_feature_names(include_shape=self.include_shape)),'include_shape':self.include_shape,
                'basis_sha256':self.basis_sha256,'query_inputs':['db'],'TRAIN_fitted_scaler':False,'source_code_sha256':pins}
            write(folder/'feature_schema.json',schema)
            if self.include_shape:shutil.copyfile(self.basis_path,folder/'spectral_basis.npz')
            pred=self._predict_features(v);np.savez_compressed(folder/'validation_fit_predictions.npz',**pred,
                sample_id=np.asarray(validation_fit.get('sample_id',np.arange(len(v)))),target_share=vy,target_presence=vz)
            names=['share.txt','presence.txt','feature_schema.json','validation_fit_predictions.npz','fit_plan.json']+(['spectral_basis.npz'] if self.include_shape else [])
            summary={'format':'r12_compact_physics_lgbm_v1','kind':'lgbm_physics_compact','status':'completed','fixture_only':bool(fixture),
                'settings':cfg,'seed':self.seed,'include_shape':self.include_shape,'feature_count':x.shape[1],'feature_schema_sha256':token(schema),
                'basis_sha256':self.basis_sha256,'basis_training_binding':self.basis_training_binding,'training_binding':dict(binding),'fit_plan_sha256':sha(folder/'fit_plan.json'),
                'best_iteration':int(reg.best_iteration_),'classifier_best_iteration':int(clf.best_iteration_),
                'validation_fit_metrics':metrics(validation_fit,pred),'train_rows':len(td),'validation_fit_rows':len(vd),
                'feature_matrix_bytes':{'train':x.nbytes,'validation_fit':v.nbytes},'lightgbm_version':lgb.__version__,
                'score_policy':'.5clipped_share+.5presence_classifier; no calibration','NN_epoch_policy_applicable':False,
                'artifact_sha256':{n:sha(folder/n) for n in names},'source_code_sha256':pins,
                'elapsed_seconds':time.perf_counter()-started,'gate_eval_Test_used_for_fitting':False}
            self.summary=summary;write(folder/'summary.json',summary)
            write(folder/'FROZEN_BEFORE_HELDOUT.json',{'summary_sha256':sha(folder/'summary.json'),'created_utc':utc(),
                'binding_sha256':token(binding),'no_refit_after_heldout':True,'heldout_opened':False})
            return summary
        except BaseException as e:
            write(folder/'FAILED.json',{'type':type(e).__name__,'error':str(e),'partial_reuse_allowed':False});raise

if __name__=='__main__':main()
