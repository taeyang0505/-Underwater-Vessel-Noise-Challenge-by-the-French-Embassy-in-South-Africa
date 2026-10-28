"""Independent saved-number verification, no model fit or inference."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/r14_fulltree_decoupled_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def auc(y,s):
    positive=s[y>0];negative=np.sort(s[y==0])
    return float(np.sum(np.searchsorted(negative,positive,'left')+.5*(np.searchsorted(negative,positive,'right')-np.searchsorted(negative,positive,'left')))/(len(positive)*len(negative)))
def calc(y,p,s):
    error=p.astype(float)-y.astype(float);hi=y>.2
    return {'rmse':float(np.linalg.norm(error)/np.sqrt(len(y))),'mae':float(abs(error).mean()),'bias':float(error.mean()),
        'auc_share':auc(y,p),'auc_score':auc(y,s),'prediction_mean':float(p.mean(dtype=float)),'target_mean':float(y.mean(dtype=float)),
        'high_y_bias':float(error[hi].mean()) if hi.any() else None}

def main():
    target=OUT/'INDEPENDENT_QA.json'
    if target.exists():raise FileExistsError('Fresh QA required')
    result=read(OUT/'RESULTS.json');policy=read(OUT/'selected_policy.json');plan=read(OUT/'plan.json')
    for n,h in plan['source_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in result['output_sha256'].items():assert sha(OUT/n)==h,n
    fit=arrays(OUT/'fit_inputs.npz');y=fit['target_share'];regs=list(csv.DictReader((OUT/'fit_regression_candidates.csv').open()))
    errors=[];regcheck=[]
    for row in regs:
        a=float(row['CNN_weight']);pred=(a*fit['CNN'].astype(float)+(1-a)*fit['fullTree'].astype(float)).astype(np.float32)
        rmse=float(np.linalg.norm(pred.astype(float)-y)/np.sqrt(len(y)));errors.append(abs(rmse-float(row['rmse'])));regcheck.append((rmse,a))
    assert min(regcheck,key=lambda v:v[0])[1]==policy['CNN_weight']
    c={'full5252':fit['full_classifier'],'strict673':fit['strict_classifier']};c['average']=(.5*c['full5252'].astype(float)+.5*c['strict673'].astype(float)).astype(np.float32)
    score_rows=list(csv.DictReader((OUT/'fit_score_candidates.csv').open()));checks=[]
    for row in score_rows:
        l=float(row['lambda']);pred=((1-l)*fit['selected_share'].astype(float)+l*c[row['classifier']].astype(float)).astype(np.float32)
        value=auc(y,pred);errors.append(abs(value-float(row['AUC'])));checks.append((value,row['classifier'],l))
    winner=max(checks,key=lambda r:r[0]);assert winner[1:]==(policy['classifier'],policy['classifier_weight'])
    rows=list(csv.DictReader((OUT/'selected_results.csv').open()));allrows=list(csv.DictReader((OUT/'all75_descriptive_results.csv').open()));checked=0
    for path in sorted((OUT/'predictions').glob('*.npz')):
        a=arrays(path);y=a['target_share'];ids=a['sample_id'];assert len(set(ids.astype(str)))==len(ids)
        origin={
            'full':ROOT/'reports/r14_tree_full_r1/predictions'/path.name,
            'anchor':ROOT/'reports/r14_anchor_cpu_r1/predictions'/path.name,
            'strict':ROOT/'reports/r12_mask_matched_evaluation_r2/predictions'/path.name}
        original={k:arrays(v) for k,v in origin.items()}
        for v in original.values():assert np.array_equal(v['sample_id'].astype(str),ids.astype(str)) and np.array_equal(v['target_share'],y)
        full=original['full'];anchor=original['anchor']
        assert np.array_equal(a['fullTree'],full['share'][:,full['model_order'].astype(str).tolist().index('FullTree_baselineP1')])
        assert np.array_equal(a['CNN'],anchor['share'][:,anchor['model_order'].astype(str).tolist().index('CNN_c0')])
        assert np.array_equal(a['full_classifier'],full['baselineP1_classifier']) and np.array_equal(a['strict_classifier'],original['strict']['extra_tree_classifier'])
        chosen=[r for r in rows if r['panel']==path.stem];assert len(chosen)==1
        for key,val in calc(y,a['share'],a['score']).items():
            if val is not None:errors.append(abs(val-float(chosen[0][key])));checked+=1
        cls={'full5252':a['full_classifier'],'strict673':a['strict_classifier']};cls['average']=(.5*cls['full5252'].astype(float)+.5*cls['strict673'].astype(float)).astype(np.float32)
        panelrows=[r for r in allrows if r['panel']==path.stem];assert len(panelrows)==75
        for row in panelrows:
            w=float(row['CNN_weight']);l=float(row['lambda']);p=(w*a['CNN'].astype(float)+(1-w)*a['fullTree'].astype(float)).astype(np.float32)
            s=((1-l)*p.astype(float)+l*cls[row['classifier']].astype(float)).astype(np.float32)
            for key,val in calc(y,p,s).items():
                if val is not None:errors.append(abs(val-float(row[key])));checked+=1
    feasibility=list(csv.DictReader((OUT/'all75_feasibility.csv').open()));assert sum(r['all_pass']=='True' for r in feasibility)==result['descriptive_all75_pass_count']==3
    assert result['selected_policy_all5_pass'] is False and len(rows)==17 and len(allrows)==1275
    assert max(errors)<1e-12
    output={'status':'passed','metrics_recomputed':checked,'maximum_absolute_difference':max(errors),
        'fit5_regression_and15_score_winners_recomputed':True,'all17_source_ID_target_member_vectors_exact':True,
        '75_combinations_count_each_panel':True,'selected_policy_passes_all5':False,'descriptive_all75_pass_count':3,
        'selection_not_changed_after_panels':True,'model_fit_or_inference':False,'source_pins_unchanged':True,
        'results_sha256':sha(OUT/'RESULTS.json'),'source_sha256':sha(__file__)}
    target.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))

if __name__=='__main__':main()
