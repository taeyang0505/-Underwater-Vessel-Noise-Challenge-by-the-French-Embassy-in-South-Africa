"""Finite monotone scalar calibration, fit1000 only, saved predictions only.

No model inference, Test, dataset construction or original policy changes.
Scalar calibration does not claim equality with the original cell-q pool.
"""
from __future__ import annotations
import argparse,os,json,time
from pathlib import Path
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='2'
import numpy as np
from ops.analyze_r14_decoupled import ROOT,sha,token,read,write,npz,csv_write,metrics,auc

def candidates():
    result=[dict(name='identity',kind='identity')]
    result += [dict(name='power'+str(g),kind='power',gamma=g) for g in (.75,.85,.95,1.05,1.15,1.25)]
    result += [dict(name='temperature'+str(t),kind='temperature',temperature=t) for t in (.7,.85,1.15,1.3)]
    result += [dict(name=f'logit_a{a}_b{b}',kind='logit',a=a,b=b) for a in (.75,1.,1.25) for b in (-.25,0.,.25) if (a,b)!=(1.,0.)]
    result += [dict(name=f'piece_low{a}_high{b}',kind='piecewise',knee=.2,low_slope=a,high_slope=b)
        for a in (.75,1.,1.25) for b in (.75,1.,1.25,1.5,2.)]
    return result

def calibrate(p,c):
    p=np.asarray(p,np.float64)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Finite 1D probability/share predictions only')
    if c['kind']=='identity':out=p.copy()
    elif c['kind']=='power':out=p**c['gamma']
    elif c['kind'] in ('temperature','logit'):
        eps=np.finfo(float).eps;s=np.clip(p,eps,1-eps);logit=np.log(s)-np.log1p(-s)
        a=1/c['temperature'] if c['kind']=='temperature' else c['a'];b=0. if c['kind']=='temperature' else c['b']
        out=1/(1+np.exp(-np.clip(a*logit+b,-100,100)))
        out[p==0]=0.;out[p==1]=1.
    elif c['kind']=='piecewise':out=c['low_slope']*np.minimum(p,c['knee'])+c['high_slope']*np.maximum(p-c['knee'],0)
    else:raise ValueError('Unknown predeclared transform')
    return np.clip(out,0,1).astype(np.float32)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',default='reports/r14_fit_calibration_v1');a=ap.parse_args()
    out=(ROOT/a.output).resolve();source=ROOT/'reports/r14_decoupled_v1'
    if not out.is_relative_to(ROOT) or out.exists():raise FileExistsError('Fresh diagnostic output required')
    inputs=[Path(__file__),ROOT/'ops/analyze_r14_decoupled.py',source/'predeclared_plan.json',source/'selected_policy.json',source/'RESULTS.json',source/'validation_fit_saved_inputs.npz']
    inputs += list((source/'predictions').glob('*.npz'))
    pins={str(p.relative_to(ROOT)):sha(p) for p in inputs};out.mkdir();(out/'predictions').mkdir()
    plan=dict(format='r14_fit_only_monotone_calibration_plan_v1',created_unix=time.time(),source_sha256=pins,
        candidates=candidates(),fit_role='validation_fit1000 saved baseline CNN/Tree50:50 predictions only',
        objectives=['ordinary_MSE','normalized_weighted_MSE_weights_1plus2y'],
        score_source='unchanged strict baseline classifier probability',score_lambda_candidates=[0.,.25,.5,.75,1.],
        choice='Each scalar transform lambda maxfitAUC first tie; scalar calibration chosen minfitobjective first tie',
        panel_role='same17 development panels after all policies frozen; no selection on their labels',
        development_exposure='17 panels known from previous studies; not blind evidence',
        physical_limit='Postprocessed scalar share need not equal original q-weighted pool; no q/y/weights redefinition',
        new_fit='scalar finite-candidate selection only; model fit0',model_inference=0,Test=False,production_approved=False)
    plan['plan_sha256']=token(plan);write(out/'predeclared_plan.json',plan)
    grid=np.linspace(0,1,1001)
    for c in candidates():
        z=calibrate(grid,c);assert np.isfinite(z).all() and ((z>=0)&(z<=1)).all() and np.all(z[1:]>=z[:-1])
    write(out/'NUMERIC_FIXTURE.json',dict(status='passed',synthetic_only=True,monotonic_bounded_grid1001_all34=True))
    fit=npz(source/'validation_fit_saved_inputs.npz');y=fit['target_share'].astype(float);base=fit['base_share'];clf=fit['classifier_base'];fitrows=[];settings=[]
    for c in candidates():
        p=calibrate(base,c);sse=(p.astype(float)-y)**2
        weights=1+2*y;weighted=float((weights*sse).sum()/weights.sum())
        ac=[auc(y,((1-l)*p.astype(float)+l*clf.astype(float)).astype(np.float32)) for l in plan['score_lambda_candidates']]
        k=int(np.argmax(ac));lam=plan['score_lambda_candidates'][k]
        settings.append(dict(calibration=c,classifier_weight=lam))
        fitrows.append(dict(name=c['name'],MSE=float(sse.mean()),weighted_MSE=weighted,score_lambda=lam,fit_auc=ac[k]))
    ordinary=int(np.argmin([v['MSE'] for v in fitrows]));weighted=int(np.argmin([v['weighted_MSE'] for v in fitrows]))
    policy=dict(format='r14_frozen_fit_calibration_policies_v1',plan_sha256=plan['plan_sha256'],all_candidate_settings=settings,
        primary_ordinary=settings[ordinary],secondary_weighted_1plus2y=settings[weighted],
        primary_candidate_index=ordinary,secondary_candidate_index=weighted,fit_only=True,panels_used_for_parameter_choice=False)
    policy['policy_sha256']=token(policy);write(out/'selected_policies.json',policy);policysha=sha(out/'selected_policies.json');csv_write(out/'validation_fit_candidates.csv',fitrows)
    records=[];selected=[]
    for path in sorted((source/'predictions').glob('*.npz')):
        assert sha(out/'selected_policies.json')==policysha
        d=npz(path);names=d['regression_order'].astype(str).tolist();base=d['regression_share'][:,names.index('linear_high0.0')];clf=d['classifier_probability'][:,0]
        ps=[];ss=[]
        for s in settings:
            p=calibrate(base,s['calibration']);lam=s['classifier_weight'];sc=((1-lam)*p.astype(float)+lam*clf.astype(float)).astype(np.float32)
            ps.append(p);ss.append(sc);m=metrics(d['target_share'],p,sc)
            records.append(dict(panel=path.stem,candidate=s['calibration']['name'],classifier_weight=lam,**m))
        for label,k in [('primary_fit_MSE',ordinary),('secondary_fit_weighted_MSE',weighted)]:
            selected.append(dict(panel=path.stem,policy=label,candidate=settings[k]['calibration']['name'],**metrics(d['target_share'],ps[k],ss[k])))
        np.savez_compressed(out/'predictions'/path.name,sample_id=d['sample_id'],target_share=d['target_share'],candidate_order=np.asarray([s['calibration']['name'] for s in settings]),
            calibrated_share=np.column_stack(ps),score=np.column_stack(ss),original_baseline_share=base,baseline_classifier=clf)
    assert len(selected)==34
    csv_write(out/'results.csv',records);csv_write(out/'selected_results.csv',selected)
    checks={}
    for label in ('primary_fit_MSE','secondary_fit_weighted_MSE'):
        r={v['panel']:v for v in selected if v['policy']==label}
        checks[label]=dict(high_snr_rmse_lt145=r['R12_high_snr_long']['rmse']<.145,unseen_rmse_le0527=r['R10_unseen__r9_complete']['rmse']<=.0527,
            unseen_score_auc_ge92=r['R10_unseen__r9_complete']['auc_score']>=.92,full15_sf_score_auc_ge82=r['R11_full_smooth15_VS_st_francis_dense']['auc_score']>=.82,
            native15_sf_score_auc_ge80=r['R11_native_long15_VS_st_francis_dense']['auc_score']>=.80)
    for p,h in pins.items():assert sha(ROOT/p)==h,p
    result=dict(format='r14_fit_only_scalar_calibration_results_v1',status='completed',plan_sha256=plan['plan_sha256'],policy=policy,
        target_checks=checks,simultaneous_target_passed={k:all(v.values()) for k,v in checks.items()},selected_results=selected,
        candidate_count=34,metric_rows=len(records),model_fit=0,model_inference=0,Test=False,original_inputs_unchanged=True,
        physical_limit=plan['physical_limit'],development_not_blind=True,
        output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    write(out/'RESULTS.json',result)
    print(json.dumps(dict(status='completed',primary=policy['primary_ordinary'],secondary=policy['secondary_weighted_1plus2y'],checks=checks),ensure_ascii=False))

if __name__=='__main__':main()
