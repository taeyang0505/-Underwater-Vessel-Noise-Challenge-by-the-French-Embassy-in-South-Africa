"""Independent, synthetic-only QA; never fit a model or open official queries."""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ops.r16_temporal_features import temporal_grid,temporal_statistics,GRID_NAMES
from ops.r16_support_temporal_models import make_model,inputs,tree_features,guard
from ops.r16_clean_models import SmallDecoupledCNN
from ops.evaluate_r16_support_temporal import measure

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    started=time.perf_counter();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    paths=['ops/r16_support_temporal_models.py','ops/r16_temporal_features.py',
           'ops/evaluate_r16_support_temporal.py','reports/r16_support_temporal_r1/PROTOCOL.md']
    before={p:sha(ROOT/p)for p in paths};sys.addaudithook(guard)
    rng=np.random.default_rng(2026102817)
    db=np.round(rng.normal(85,4,(8,15,25)),1)
    db[0,0]=np.nan;db[1,3:6]=np.nan;db[2,:,24]=np.nan;db[3]=81.
    db[4]=np.nan;db[4,7]=75.
    shift=np.round(rng.normal(0,15,(len(db),1,25)),1)
    g,o=temporal_grid(db);s,names=temporal_statistics(db)
    gg,_=temporal_grid(db+shift);ss,_=temporal_statistics(db+shift)
    assert g.shape==(len(db),len(GRID_NAMES),15,25)
    assert np.isfinite(g).all() and np.isfinite(s).all()
    assert np.all(g*np.logical_not(o[:,None])==0)
    grid_gap=float(abs(g-gg).max());stats_gap=float(abs(s-ss).max())
    assert grid_gap<3e-5 and stats_gap<3e-5
    assert np.array_equal(s,temporal_statistics(db)[0])
    shuffled=db[:,[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11]]
    order_delta=float(abs(s-temporal_statistics(shuffled)[0]).max())
    assert order_delta>.01
    # Lag1 must not silently bridge a missing original minute.
    missing=np.full((1,15,25),np.nan);missing[:,0]=60.;missing[:,2]=80.
    mg,_=temporal_grid(missing)
    lag1_index=GRID_NAMES.index('observed_contiguous_lag1_div10')
    assert np.all(mg[:,lag1_index]==0)
    clf_gaps={};parity={};gradient={};counts={}
    for f in (0,1):
        x,w=inputs(db,f);shift_x,shift_w=inputs(db+shift,f)
        model=make_model(f).eval();counts[str(f)]=model.parameter_counts()
        p,q,c=model(torch.from_numpy(x),torch.from_numpy(w))
        pp,qq,cc=model(torch.from_numpy(shift_x),torch.from_numpy(shift_w))
        clf_gaps[str(f)]=float(abs(c-cc).max())
        assert clf_gaps[str(f)]<3e-5
        assert float(abs(p-(q*torch.from_numpy(w)).sum((1,2))).max())<1e-7
        assert torch.isfinite(p).all() and torch.isfinite(c).all()
        model.zero_grad(set_to_none=True);c.sum().backward()
        a=not any(v.grad is not None for v in model.reg.parameters())
        assert a
        model.zero_grad(set_to_none=True)
        p,q,c=model(torch.from_numpy(x),torch.from_numpy(w));(p.sum()+q.sum()).backward()
        b=not any(v.grad is not None for v in list(model.clf.parameters())+list(model.clf_head.parameters()))
        assert b;gradient[str(f)]={'classifier_to_regressor_blocked':a,'regressor_to_classifier_blocked':b}
        r,c,n=tree_features(db,f);rr,ccc,_=tree_features(db+shift,f)
        assert np.isfinite(r).all() and np.isfinite(c).all()
        clf_gaps['tree_features_'+str(f)]=float(abs(c-ccc).max())
        assert clf_gaps['tree_features_'+str(f)]<3e-5
        counts[str(f)].update(tree_reg_features=r.shape[1],tree_clf_features=c.shape[1])
    x0,w=inputs(db,0);x1,_=inputs(db,1)
    with torch.no_grad():
        a=make_model(0).eval()(torch.from_numpy(x0),torch.from_numpy(w))
        b=make_model(1).eval()(torch.from_numpy(x1),torch.from_numpy(w))
        parity['F0_F1_common_initial_function']=max(float(abs(v-u).max())for v,u in zip(a,b))
        assert parity['F0_F1_common_initial_function']<2e-6
        torch.manual_seed(2026102817);old=SmallDecoupledCNN().eval()
        c=old(torch.from_numpy(x0),torch.from_numpy(w))
        parity['F0_legacy_architecture_and_initialization']=max(float(abs(v-u).max())for v,u in zip(a,c))
        assert parity['F0_legacy_architecture_and_initialization']==0
    y=np.array([.1,.5,0.,0.,0.]);p=np.array([.2,.3,.1,.2,.3]);c=np.array([.8,.9,.2,.3,.4]);weights=np.array([.25,.25,1/6,1/6,1/6])
    measured=measure(y,p,c,weights);manual=float(np.sqrt(np.dot(weights,(p-y)**2)))
    assert abs(measured['rmse']-manual)<1e-12
    denied=False
    try:
        with open('/private/tmp/r16-independent-qa-never-created/Test.csv','r'):pass
    except PermissionError:denied=True
    assert denied
    assert before=={p:sha(ROOT/p)for p in paths}
    result={'status':'passed','synthetic_rows':len(db),'new_temporal_grid_channels':len(GRID_NAMES),
            'new_temporal_statistics_columns':len(names),'additional_grid_static_band_offset_max_abs':grid_gap,
            'additional_stats_static_band_offset_max_abs':stats_gap,'classifier_static_band_offset_checks':clf_gaps,
            'time_shuffle_feature_change_max_abs':order_delta,'missing_lag1_no_time_compression':True,
            'missing_all_band_and_constant_windows_finite':True,'parameter_gradient_isolation':gradient,
            'forward_parity_max_abs':parity,'counts':counts,'weighted_measure_rmse_error':abs(measured['rmse']-manual),
            'query_open_guard_rejected_synthetic_path':True,'source_hash_unchanged_during_QA':before,
            'model_training_performed':False,'official_query_accessed':False,'elapsed_seconds':time.perf_counter()-started}
    out=Path(__file__).with_name('INDEPENDENT_CODE_QA.json');out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
