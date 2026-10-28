"""Numeric and fail-closed checks for the fit/gate policy, no real data I/O."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from pipeline.ensemble_r14_production import fit_policy,apply_policy,auc,token,RIDGE
    rng=np.random.default_rng(2026102714)
    fy=rng.beta(1.5,4,500);fy[:100]=0;gy=rng.beta(1.5,4,500);gy[:100]=0
    fs={'baseline':np.clip(fy+.06+rng.normal(0,.03,500),0,1),'neural':np.clip(fy-.05+rng.normal(0,.03,500),0,1)}
    gs={'baseline':np.clip(gy+.06+rng.normal(0,.03,500),0,1),'neural':np.clip(gy-.05+rng.normal(0,.03,500),0,1)}
    def clf(y):
        a=np.clip(.1+.65*(y>0)+rng.normal(0,.18,len(y)),0,1);b=np.clip(.2+.6*(y>0)+rng.normal(0,.2,len(y)),0,1)
        return {'baseline':a,'highshare':b,'mean':.5*(a+b)}
    fc,gc=clf(fy),clf(gy)
    p=fit_policy(fy,fs,fc,gy,gs,gc,binding={'fixture':'numeric_only','same_fold':True})
    out=apply_policy(p,gs,gc)
    x=np.column_stack(list(fs.values()));weights=np.asarray(p['proposed_regression_weights'])
    grid=np.linspace(0,1,10001);errors=((x[:,0,None]*grid+x[:,1,None]*(1-grid)-fy[:,None])**2).mean(0)+RIDGE*2*(grid-.5)**2
    objective=float(np.mean((x@weights-fy)**2)+RIDGE*np.sum((weights-.5)**2))
    assert objective<=float(errors.min())+1e-9
    assert out['share'].shape==(500,) and np.isfinite(out['score']).all() and (out['score']>=0).all() and (out['score']<=1).all()
    deltas=[]
    for v in (np.zeros(500),np.round(rng.random(500),1),rng.random(500)):
        delta=abs(auc(fy,v)-roc_auc_score(fy>0,v));assert delta<1e-12;deltas.append(delta)
    rejected=[]
    def rejection(name,fn):
        try:fn()
        except (ValueError,TypeError,RuntimeError):rejected.append(name);return
        raise AssertionError('Invalid input accepted:'+name)
    tampered=json.loads(json.dumps(p));tampered['regression_weights'][0]+=.1
    rejection('tampered_policy',lambda:apply_policy(tampered,gs,gc))
    rejection('missing_member',lambda:apply_policy(p,{'baseline':gs['baseline']},gc))
    bad=dict(gs);bad['baseline']=bad['baseline'].copy();bad['baseline'][0]=float('nan')
    rejection('NaN_prediction',lambda:apply_policy(p,bad,gc))
    badc=dict(gc);badc['mean']=np.zeros(500)
    rejection('classifier_mean_tamper',lambda:apply_policy(p,gs,badc))
    rejection('wrong_production_rows',lambda:fit_policy(fy[:20],{k:v[:20] for k,v in fs.items()},{k:v[:20] for k,v in fc.items()},gy,gs,gc,binding={'x':1}))
    rejection('single_class_targets',lambda:fit_policy(np.zeros(500),fs,fc,gy,gs,gc,binding={'x':1}))
    fp=dict(p);fp['fixture_only']=True;fp.pop('policy_sha256');fp['policy_sha256']=token(fp)
    rejection('fixture_policy_production',lambda:apply_policy(fp,gs,gc))
    # Same fitted candidate is rejected if its gate differs; no gate refitting.
    shifted={name:(gy.copy() if name==p['fit_best_single'] else np.clip(gy+.3,0,1)) for name in fs}
    fallback=fit_policy(fy,fs,fc,gy,shifted,gc,binding={'fixture':'gate_shift'})
    assert not fallback['regression_accepted_on_gate']
    assert np.count_nonzero(fallback['regression_weights'])==1
    result={'status':'passed','synthetic_fixture_only':True,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'pipeline/ensemble_r14_production.py')},
       'simplex_objective':objective,'dense_grid_minimum':float(errors.min()),'AUC_max_abs_error':max(deltas),'rejected_cases':rejected,
       'regression_gate_fallback_test':True,'real_TRAIN_Test_or_report_read':False,'actual_model_fitting':False}
    path=ROOT/'reports/R14_PRODUCTION_ENSEMBLE_NUMERIC_QA.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
