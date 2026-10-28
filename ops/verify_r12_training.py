"""Numeric-only R12 CPU trainer verification, no acoustic-data access.

Exercises all nine 26,953-parameter arm settings and an actual 100-epoch
miniature production loop. Only fixture validation RMSE is scripted to test
strict minimum checkpoint selection; no performance is asserted. Temporary
numeric-model files are deleted, never reused as production checkpoints.
"""
from __future__ import annotations
import copy,hashlib,json,math,sys,tempfile
from pathlib import Path
from unittest import mock
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from pipeline import training_r12 as engine
from pipeline.features_r12 import fit_spectral_basis


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def a_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def forbidden_access(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,Path)):
        p=Path(str(args[0]));s=str(p)
        if p.name=='Test.csv' or '/data.nosync/' in s or '/data/synthetic/' in s or '/data/processed/' in s:
            raise PermissionError('R12 numeric fixture forbids all acoustic data/Test files')
    if event in ('socket.connect','subprocess.Popen','os.system'):raise PermissionError('No process/network actions')


def main():
    out=ROOT/'reports/R12_TRAINING_ENGINE_VERIFICATION.json'
    if out.exists():raise FileExistsError('Fresh fixture report required')
    torch.set_num_threads(1)
    pins={n:sha(ROOT/n) for n in ['pipeline/training_r12.py','ops/train_r12_cnn.py','pipeline/features_r12.py',
        'pipeline/relative_models_r6_v2.py','pipeline/relative_grid_r6_v2.py','pipeline/relative_grid_r6.py']}
    sys.addaudithook(forbidden_access)
    (ROOT/'reports/R12_TRAINING_ENGINE_VERIFICATION_PLAN.json').write_text(json.dumps({
        'format':'r12_numeric_training_engine_plan_v1','source_sha256':pins,
        'scope':['nine settings one Adam step each','pure/legacy gradient decomposition','six objective finite counterexamples','actual tiny100epoch loop','scripted equal-min checkpoint tie'],
        'temporary_only_numeric_models':True,'no_real_data_Test_or_heldout_read':True},indent=2)+'\n')
    rng=np.random.default_rng(202610271201)
    pure=rng.normal(75,4,(16,15,25));ids=np.asarray(['numeric_SA_TRAIN_'+str(i) for i in range(len(pure))])
    basis=fit_spectral_basis(pure,ids,training_binding={'fixture_only':True,'role':'train_A0_numeric_analogue'})
    def dataset(n,seed):
        r=np.random.default_rng(seed);db=np.round(r.normal(74,3,(n,15,25)),1).astype(np.float32)
        if n>1:db[1,[0,5],:]=np.nan;db[1,7,23]=np.nan
        y=np.resize(np.array([0,.2,.5,.05],np.float32),n)
        truth=np.broadcast_to(y[:,None,None],db.shape).copy();truth[~np.isfinite(db)]=0
        return {'db':db,'share':y,'cell_share':truth,'site':np.asarray(['numeric_site_'+str(i%2) for i in range(n)])}
    train=dataset(4,1);val=dataset(4,2)
    hashes={role:{k:a_sha(v) for k,v in data.items()} for role,data in [('train',train),('validation_fit',val)]}
    arms=[{'name':'legacy_contrast_control','long':False,'HF':False,'shape':False,'legacy_control':True}]+[
        {'name':f'L{l}H{h}P{s}','long':bool(l),'HF':bool(h),'shape':bool(s),'legacy_control':False}
        for l in [0,1] for h in [0,1] for s in [0,1]]
    steps=[]
    for arm in arms:
        torch.manual_seed(2026102712)
        model=engine.R12MaskCNN();optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
        x,w=engine.channels(train['db'],basis,arm['shape'])
        if not arm['shape']:assert not x[:,8:].any()
        x,w=torch.from_numpy(x),torch.from_numpy(w);y=torch.from_numpy(train['share']);truth=torch.from_numpy(train['cell_share'])
        p,q=model(x,w);loss,_,_=engine.objective(p,q,w,y,truth,legacy_control=arm['legacy_control'])
        assert torch.isfinite(loss).all() and torch.allclose(p,(q*w).sum((1,2)),atol=1e-7)
        assert not q[w==0].any()
        loss.mean().backward()
        grad=[v.grad for v in model.parameters() if v.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grad)
        optimizer.step();assert all(torch.isfinite(t).all() for t in model.parameters())
        count=sum(t.numel() for t in model.parameters());assert count==26953
        prediction=engine.predict(model,x,w,batch=3,legacy_score=arm['legacy_control'])
        if not arm['legacy_control']:assert np.array_equal(prediction['score'],prediction['share'])
        steps.append({'arm':arm,'parameters':count,'one_Adam_step_all_gradients_finite':True,'channels':list(x.shape),
                      'physical_pool_and_missing_q':True,'score_share_identity_unless_legacy_control':True})
    # Pure loss cannot call contrast/presence at all; verify its exact gradient.
    y=torch.tensor([.2]);w=torch.full((1,15,25),1/375.);truth=torch.full_like(w,.2)
    q=truth.clone();q[0,0,22]-=.001;q[0,1,22]+=.001;q.requires_grad_();p=(q*w).sum((1,2))
    with mock.patch.object(engine,'fixed_presence_score',side_effect=AssertionError('contrast must not enter pure objective')):
        pure_loss,_,_=engine.objective(p,q,w,y,truth)
        pure_grad=torch.autograd.grad(pure_loss.sum(),q,retain_graph=True)[0]
    expected=2*w*((p-y)[:,None,None]+.25*(q-truth))
    error=float(abs(pure_grad-expected).max());assert error<1e-9
    legacy_loss,_,_=engine.objective(p,q,w,y,truth,legacy_control=True)
    legacy_grad=torch.autograd.grad(legacy_loss.sum(),q)[0]
    assert pure_grad[0,0,22]<0 and pure_grad[0,1,22]>0
    assert legacy_grad[0,0,22]>0 and legacy_grad[0,1,22]<0
    counter=[]
    for weightkind in ['uniform375','LF_dominant']:
        ww=torch.ones(1,15,25)/375 if weightkind=='uniform375' else torch.pow(10.,-torch.arange(25,dtype=torch.float32)/5.)[None,None,:].repeat(1,15,1)
        if weightkind!='uniform375':ww/=ww.sum()
        for target in [.05,.2,.5]:
            yy=torch.tensor([target]);tt=torch.full_like(ww,target);wrong=tt.clone();wrong[0,0,22]=1e-4;wrong[0,1,22]=1-1e-4
            values={}
            for name,qq in [('correct',tt),('wrong',wrong)]:
                pp=(ww*qq).sum((1,2));a=engine.objective(pp,qq,ww,yy,tt)[0];b=engine.objective(pp,qq,ww,yy,tt,legacy_control=True)[0]
                values[name]={'pure_loss':float(a[0]),'legacy_loss':float(b[0])}
            assert values['correct']['pure_loss']<values['wrong']['pure_loss']
            assert values['wrong']['legacy_loss']<values['correct']['legacy_loss']
            counter.append({'weights':weightkind,'target':target,'values':values})
    # Actual100-loop test: no stopping despite a validation plateau. A tie at
    # epoch4 must NOT replace the strict first minimum at epoch3. This mocks
    # only numeric-fixture metrics, never an acoustic validation metric.
    original_metrics=engine.metrics;original_predict=engine.predict;original_adam=torch.optim.Adam
    state={'metric_calls':0,'predict_calls':0,'optimizer_steps':0};captured={}
    def controlled_metrics(y,pred,site=None):
        result=original_metrics(y,pred,site);state['metric_calls']+=1
        k=state['metric_calls'];result['rmse']=.4 if k==1 else .3 if k==2 else .2
        result['fixture_scripted_RMSE_not_performance']=True
        return result
    def tracked_predict(model,*args,**kwargs):
        state['predict_calls']+=1
        if state['predict_calls']==3:captured['epoch3_state']={k:v.detach().clone() for k,v in model.state_dict().items()}
        return original_predict(model,*args,**kwargs)
    class CountingAdam(original_adam):
        def step(self,*args,**kwargs):state['optimizer_steps']+=1;return super().step(*args,**kwargs)
    with tempfile.TemporaryDirectory(prefix='R12_numeric_loop_ONLY_',dir='/private/tmp') as tmp:
        folder=Path(tmp)/'numeric_fixture_ONLY'
        with mock.patch.object(engine,'metrics',controlled_metrics),mock.patch.object(engine,'predict',tracked_predict),mock.patch.object(torch.optim,'Adam',CountingAdam):
            summary=engine.fit(train,val,basis,arms[1],folder,threads=1,batch=3)
        history=[json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
        assert len(history)==100 and [r['epoch'] for r in history]==list(range(1,101))
        assert summary['epochs_run']==100 and not summary['early_stopping'] and summary['best_epoch']==3
        assert state['optimizer_steps']==200 and state['predict_calls']==101
        best=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        assert best['epoch']==3 and best['parameters']==26953
        assert all(torch.equal(v,captured['epoch3_state'][k]) for k,v in best['state_dict'].items())
        model=engine.R12MaskCNN();model.load_state_dict(best['state_dict'],strict=True)
        vx,vw=engine.channels(val['db'],basis,False);vx,vw=torch.from_numpy(vx),torch.from_numpy(vw)
        a=original_predict(model,vx,vw,batch=1);b=original_predict(model,vx,vw,batch=3)
        batch_error=max(float(abs(a[k]-b[k]).max()) for k in a)
        assert batch_error<3e-6
        with np.load(folder/'validation_fit_predictions.npz',allow_pickle=False) as z:
            for k in b:assert np.array_equal(z[k],b[k])
        rates=[r['learning_rate'] for r in history]
        assert abs(rates[0]-.001)<1e-12 and rates[-1]>1e-5 and rates[-1]<1.1e-5 and np.all(np.diff(rates)<0)
        loop={'actual_epochs':100,'actual_Adam_steps':200,'train_rows':4,'batch3_unequal_last_batch':True,
            'scripted_fit_validation_first_strict_min_epoch':3,'saved_best_state_equals_captured_epoch3_bitwise':True,
            'validation_gate_or_eval_not_supplied':True,'no_early_stop_on_plateau':True,
            'saved_validation_predictions_best_checkpoint_exact':True,'batch1_vs3_max_error':batch_error,
            'first_lr':rates[0],'epoch100_lr':rates[-1],'checkpoint_removed_with_tempdir':True,
            'warning':'Mock validation RMSE tests checkpoint selection only; it is not predictive performance.'}
    assert not folder.exists()
    assert all(a_sha(v)==hashes[role][k] for role,data in [('train',train),('validation_fit',val)] for k,v in data.items())
    assert all(sha(ROOT/n)==v for n,v in pins.items())
    report={'status':'passed','fixture_only':True,'source_sha256':pins,'script_sha256':sha(__file__),
        'nine_arm_real_Adam_steps':steps,'pure_objective_gradient':{'max_formula_error':error,
            'fixed_contrast_function_not_called':True,'pure_gradient_restores_truth_while_legacy_distorts_tested_cells':True},
        'six_counterexamples_pure_vs_legacy':counter,'actual_numeric100epoch_loop':loop,
        'actual_acoustic_data_heldout_Test_fit_inference':False,'input_arrays_and_frozen_source_unchanged':True,
        'limits':['Numeric fixture tests implementation, not model quality or Public-score improvement.',
            'Long/HF flags here test all9 model/loss configurations; acoustic factorial source construction is tested elsewhere.',
            'No production weights are saved or reused. The temporary core STARTED fixture:false field is not evidence of a real-data run.'],
        'review':['Pure loss is MSE+.25 weighted cell MSE, with no presence BCE.',
                  'All nine arms use identical11channel26953parameter architecture; shape-off maps are three zero channels.',
                  'Checkpoint selection only uses validation_fit; gate/eval arrays are not accepted by fit.']}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','report':str(out),'sha256':sha(out),'source_sha256':pins,'loop':loop}),flush=True)

if __name__=='__main__':main()
