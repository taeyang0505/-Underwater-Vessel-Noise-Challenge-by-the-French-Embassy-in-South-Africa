"""Independent saved-array arithmetic QA; no model inference or data fitting."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def auc(y,s):
    z=np.asarray(y)>0;positive=np.asarray(s,float)[z];negative=np.sort(np.asarray(s,float)[~z])
    if not len(positive) or not len(negative):return None
    lo=np.searchsorted(negative,positive,'left');hi=np.searchsorted(negative,positive,'right')
    return float(np.sum(lo+.5*(hi-lo))/(len(positive)*len(negative)))

def run(folder):
    r=json.loads((folder/'RESULTS.json').read_text());policy=json.loads((folder/'selected_policy.json').read_text())
    if r['status']!='completed' or r['panels']!=17:raise ValueError('Complete17panel evaluation required')
    for n,h in r['source_pins'].items():
        if sha(ROOT/n)!=h:raise ValueError('Original source changed:'+n)
    for n,h in r['output_sha256'].items():
        if sha(folder/n)!=h:raise ValueError('Output changed:'+n)
    rows=list(csv.DictReader((folder/'results.csv').open()));index={(v['panel'],v['model']):v for v in rows}
    checks=0;formula=0;pool_error=0.;max_error=0.;panels=[];gateauc=None
    def close(name,actual,expected,tol=2e-10):
        nonlocal checks,max_error
        checks+=1
        if expected is None:
            if actual not in (None,''):raise AssertionError(name+': expected undefined')
            return
        error=abs(float(actual)-float(expected));max_error=max(max_error,error)
        if error>tol:raise AssertionError(name+': '+str(error))
    for path in sorted((folder/'predictions').glob('*.npz')):
        name=path.stem;panels.append(name)
        with np.load(path,allow_pickle=False) as z:
            y=z['target_share'].astype(float);order=z['model_order'].astype(str).tolist();p=z['share'];s=z['score']
            q=z['cnn_cell_mask'];w=z['observed_weights'];clf=z['tree_classifier'];contrast=z['cnn_contrast']
            if p.shape!=s.shape or p.shape!=(len(y),len(order)) or not np.isfinite(p).all() or not np.isfinite(s).all():raise ValueError('Prediction alignment')
            if np.any((q<0)|(q>1)) or not np.isfinite(q).all() or np.any(q[w==0]!=0):raise ValueError('Physical mask')
            gap=float(np.max(np.abs(np.sum(q*w,axis=(1,2))-p[:,order.index('CNN_dynamic_mix')])));pool_error=max(pool_error,gap)
            if gap>3e-6:raise ValueError('Physical q pool')
            blend=(.5*p[:,order.index('CNN_dynamic_mix')].astype(float)+.5*p[:,order.index('Tree_dynamic_mix')].astype(float)).astype(np.float32)
            for lam in (0.,.25,.5,1.):
                key='Blend50_lambda'+str(lam);k=order.index(key)
                expected=((1-lam)*blend.astype(float)+lam*clf.astype(float)).astype(np.float32)
                if not np.array_equal(p[:,k],blend) or not np.array_equal(s[:,k],expected):raise ValueError('Frozen blend formula')
                formula+=2
            selected=order.index('DynamicMix_selected');chosen=order.index('Blend50_lambda'+str(policy['score_classifier_weight']))
            if not np.array_equal(p[:,selected],p[:,chosen]) or not np.array_equal(s[:,selected],s[:,chosen]):raise ValueError('Selected lambda changed')
            with np.load(ROOT/'reports/r13_cpu_correction_r1/predictions'/path.name,allow_pickle=False) as old:
                if not np.array_equal(z['sample_id'],old['sample_id']) or not np.array_equal(z['target_share'],old['target_share']):raise ValueError('Old comparator ID/target changed')
                names=old['model_order'].astype(str).tolist()
                for key in ('R13_selected','R12_matched_share_only','R9_frozen_all5'):
                    if key in order:
                        if not np.array_equal(p[:,order.index(key)],old['share'][:,names.index(key)]) or not np.array_equal(s[:,order.index(key)],old['score'][:,names.index(key)]):raise ValueError('Old comparator changed')
                        formula+=2
            if name=='R12_validation_gate':gateauc={str(v):auc(y,s[:,order.index('Blend50_lambda'+str(v))]) for v in (0.,.25,.5,1.)}
            zero=y==0;high=y>.2;enriched_high=np.asarray(y,np.float32)>.2;present=y>0
            for j,key in enumerate(order):
                row=index[(name,key)];pred=p[:,j].astype(float);score=s[:,j].astype(float);err=pred-y;SSE=float(np.sum(err**2))
                metrics={'n':len(y),'positive_n':int(present.sum()),'zero_n':int(zero.sum()),'rmse':float(np.sqrt(np.mean(err**2))),
                    'mae':float(np.mean(abs(err))),'bias':float(err.mean()),'prediction_mean':float(pred.mean()),'prediction_rms':float(np.sqrt(np.mean(pred**2))),
                    'target_mean':float(y.mean()),'target_rms':float(np.sqrt(np.mean(y**2))),'auc_share':auc(y,pred),'auc_score':auc(y,score),
                    'high_y_n':int(high.sum()),'high_y_SSE':float(np.sum(err[high]**2)),'high_y_SSE_fraction':float(np.sum(err[high]**2)/SSE) if SSE else None,
                    'score_brier':float(np.mean((score-present)**2)),'zero_prediction_mean':float(pred[zero].mean()) if zero.any() else None,
                    'high_y_bias':float(err[enriched_high].mean()) if enriched_high.any() else None,'high_y_rmse':float(np.sqrt(np.mean(err[enriched_high]**2))) if enriched_high.any() else None,
                    'share_ge099':float(np.mean(pred>=.99)),'score_ge099':float(np.mean(score>=.99))}
                if key=='CNN_dynamic_mix':metrics['contrast_ge099_fraction']=float(np.mean(contrast>=.99))
                if key=='Tree_dynamic_mix' or key.startswith('Blend50_') or key=='DynamicMix_selected':
                    metrics['classifier_auc']=auc(y,clf);metrics['classifier_brier']=float(np.mean((clf.astype(float)-present)**2));metrics['auc_classifier']=auc(y,clf)
                for metric,value in metrics.items():close(name+'/'+key+'/'+metric,row[metric],value)
    if len(panels)!=17 or len(rows)!=r['metric_rows']:raise ValueError('Panel/row count')
    chosen=max((0.,.25,.5,1.),key=lambda v:gateauc[str(v)])
    if chosen!=policy['score_classifier_weight']:raise ValueError('Gate lambda first max differs')
    for k,v in gateauc.items():close('gate_score_AUC/'+k,policy['gate_score_AUC'][k],v)
    result={'format':'r13_dynamic_mix_saved_prediction_QA_v1','status':'passed','panels':17,'metric_rows':len(rows),
        'scalar_numeric_checks':checks,'exact_formula_checks':formula,'scalar_max_abs_error':max_error,
        'physical_q_pool_max_abs_error':pool_error,'selected_lambda_gate_first_argmax':chosen,
        'original_comparator_targets_IDs_predictions_exact':True,'original_source_pins_unchanged':True,
        'source_RESULT_sha256':sha(folder/'RESULTS.json'),'verification_source_sha256':sha(__file__),
        'tree_fit_count_clarification':'One tree learner.fit call, two booster fits: regressor1 plus independent classifier1; actual_tree_booster_fits:1 in executor receipt denotes learner call, not booster count.',
        'coverage_limits':['Point metrics/formulas/IDs/q pooling only; site-macro and bootstrap CI endpoints not independently recalculated here',
            'R10sensor-week clusters have previously documented short adjacentweek gaps; not independent24h bootstrap blocks',
            'All17panels development-exposed; no blind result or Test-loss claim'],
        'new_model_inference_or_fitting_Test_GPU':False}
    out=folder/'INDEPENDENT_QA.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','checks':checks,'sha256':sha(out)}))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',default='reports/r13_dynamic_mix_r1');a=p.parse_args();run(ROOT/a.folder)
