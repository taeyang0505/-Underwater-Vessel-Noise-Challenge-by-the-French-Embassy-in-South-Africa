"""Independent frozen evaluation recomputation. Run only after root DONE."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from ops.r16_support_temporal_data import load_long_panel,guard
ARMS=('D0F0','D0F1','D1F0','D1F1','historical_R16_mixed')
OPS=('control_eq6','mfhf_shelf6','mfhf_shelf12','mfhf_shelf12_lfminus6','alpha_preEQ_inserted','independent_subband_snr')
METHODS={'NN':('nn_share','nn_classifier'),'Tree':('tree_share','tree_classifier'),'fixed60T40N':('balanced_share','balanced_classifier')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def metric(y,p,c,w=None):
    y=np.asarray(y,float);p=np.asarray(p,float);c=np.asarray(c,float);w=np.ones(len(y))if w is None else np.asarray(w,float)
    w=w/w.sum();e=p-y;z=y>0
    assert np.isfinite(np.r_[y,p,c,w]).all() and (w>=0).all()
    return {'n':len(y),'positive_n':int(z.sum()),'zero_n':int((~z).sum()),
            'rmse':float(np.sqrt(w@(e*e))),'mae':float(w@abs(e)),'bias':float(w@e),
            'target_mean':float(w@y),'prediction_mean':float(w@p),'prediction_rms':float(np.sqrt(w@(p*p))),
            'classifier_mean':float(w@c),'classifier_ge099_fraction':float(w@(c>=.99)),
            'auc_share':float(roc_auc_score(z,p,sample_weight=w))if len(set(z))==2 else None,
            'auc_classifier':float(roc_auc_score(z,c,sample_weight=w))if len(set(z))==2 else None}

def main():
    began=time.perf_counter();sys.addaudithook(guard)
    freeze_path=OUT/'GLOBAL_FREEZE.json';fr=read(freeze_path);result_path=OUT/'evaluation/RESULTS.json';result=read(result_path)
    assert fr['all_models_frozen']is True and fr['ensemble_policy_frozen']is True and result['status']=='completed'
    assert result['global_freeze_sha256']==sha(freeze_path)
    assert fr['data_manifest_sha256']==sha(OUT/'data/manifest.json')
    for p,h in fr['artifact_sha256'].items():assert sha(ROOT/p)==h,p
    policy=read(OUT/'POLICY.json');assert policy['score']=={'Tree_classifier':.5,'NN_classifier':.5,'share':0}
    assert policy['regression']=={'Tree':.6,'NN':.4}and policy['Test_used']is False
    predictions={};pred_verified=0
    for p,h in result['predictions_sha256'].items():
        path=OUT/'evaluation'/p;assert sha(path)==h
        d=load(path);assert tuple(d['model_order'])==ARMS
        assert np.isfinite(d['target_share']).all()
        for k in ('nn_share','tree_share','balanced_share','nn_classifier','tree_classifier','balanced_classifier'):
            assert d[k].shape==(len(d['target_share']),5)and np.isfinite(d[k]).all()
            assert (d[k]>=0).all()and(d[k]<=1).all()
        assert np.allclose(d['balanced_share'],.6*d['tree_share']+.4*d['nn_share'],rtol=0,atol=1e-7)
        assert np.allclose(d['balanced_classifier'],.5*d['tree_classifier']+.5*d['nn_classifier'],rtol=0,atol=1e-7)
        predictions[path.stem]=d;pred_verified+=1
    checked=0;errors=[]
    for row in result['rows']:
        name=row['panel'];arm=ARMS.index(row['arm']);pk,ck=METHODS[row['model']]
        if name.endswith('__POOLED'):
            role=row['role'];parts=[predictions[role+'__'+op]for op in OPS]
            y=np.concatenate([x['target_share']for x in parts]);p=np.concatenate([x[pk][:,arm]for x in parts]);c=np.concatenate([x[ck][:,arm]for x in parts]);w=None
        elif name.startswith('legacy_R11__native15_VS_'):
            negative=name.split('legacy_R11__native15_VS_')[1];parts=[predictions['legacy_R11__native_long15'],predictions['legacy_R11__'+negative]]
            y=np.r_[parts[0]['target_share'],parts[1]['target_share']];p=np.r_[parts[0][pk][:,arm],parts[1][pk][:,arm]];c=np.r_[parts[0][ck][:,arm],parts[1][ck][:,arm]]
            w=np.r_[np.full(len(parts[0]['target_share']),.5/len(parts[0]['target_share'])),np.full(len(parts[1]['target_share']),.5/len(parts[1]['target_share']))]
        else:
            d=predictions[name];y=d['target_share'];p=d[pk][:,arm];c=d[ck][:,arm];w=d['sample_weight']
        recompute=metric(y,p,c,w)
        for key,value in recompute.items():
            if value is None:assert row[key]is None
            else:errors.append(abs(value-row[key]));assert errors[-1]<1e-9,(name,row['arm'],row['model'],key,value,row[key])
        if 'unweighted_rmse'in row:
            gap=abs(metric(y,p,c)['rmse']-row['unweighted_rmse']);errors.append(gap);assert gap<1e-9
        checked+=1
    long=[]
    for op in OPS:
        d=load_long_panel(OUT/'data/manifest.json','native12_positive',operator=op,freeze_receipt=freeze_path)
        assert np.all(d['selected_minutes']==12)and len(set(d['reference_index']))==1
        assert set(d['reference_component'])=={'VT_R4_REF_de02fbd6bbb753c153bd'}and np.all(d['share']>0)
        active=(d['added_observed_power']>0).any(2).sum(1);assert active.min()>=11 and active.max()==12
        long.append({'operator':op,'n':len(active),'actual_added_support':{str(int(k)):int(v)for k,v in zip(*np.unique(active,return_counts=True))}})
    conditional=[]
    for negative in ('hf_high_negative','st_francis_dense_negative'):
        d=predictions['native12_VS_'+negative];y=d['target_share'];positive=y>0;site=np.asarray(d['site']).astype(str);ids=d['base_window_index']
        common=set(ids[positive])&set(ids[~positive]);matched=np.isin(ids,list(common))
        assert len(set(d['sample_id']))==len(y)
        assert all(np.sum(ids[positive]==source)==6 and np.sum(ids[~positive]==source)==1 for source in common)
        for scope,ix in [('same_background_ID',matched),('St_Francis_only',site=='7 St_Francis')]:
            if len(np.unique(positive[ix]))<2:continue
            ww=np.where(positive[ix],.5/positive[ix].sum(),.5/(~positive[ix]).sum())
            for ai,arm in enumerate(ARMS):
                for method,(pk,ck)in METHODS.items():
                    conditional.append({'negative':negative,'scope':scope,'arm':arm,'model':method,
                        'unique_backgrounds':len(set(ids[ix])),'unique_positive_backgrounds':len(set(ids[ix&positive])),
                        'unique_negative_backgrounds':len(set(ids[ix&~positive])),'reference_components':1,
                        **metric(y[ix],d[pk][ix,ai],d[ck][ix,ai],ww)})
    for p,h in fr['artifact_sha256'].items():assert sha(ROOT/p)==h
    report={'status':'passed','frozen_artifact_SHA_verified':len(fr['artifact_sha256']),'prediction_SHA_verified':pred_verified,
        'evaluation_metric_rows_independently_recomputed':checked,'maximum_metric_absolute_error':max(errors),
        'independent_native12_actual_arrays':long,'fixed_regression_and_independent_classifier_combination_verified':True,
        'same_background_and_SF_conditional_diagnostics':conditional,'original_gate_unchanged':result['small_gate_passed'],
        'Test_accessed':False,'new_model_training_or_policy_tuning':False,'elapsed_seconds':time.perf_counter()-began,
        'limitations':['Conditional rows reuse fixed samples and policies; no extra tuning or new gate.',
            'Only one heldout12min reference component; neither row count nor CI demonstrates unseen15min transfer.',
            'Primary global gate remains predeclared; conditional site/mask diagnostics explain cohort confounding only.']}
    (OUT/'INDEPENDENT_RESULTS_QA.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k!='same_background_and_SF_conditional_diagnostics'},ensure_ascii=False))

if __name__=='__main__':main()
