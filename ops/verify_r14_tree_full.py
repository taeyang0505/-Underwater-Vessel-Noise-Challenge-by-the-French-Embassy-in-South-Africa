"""Read-only full-N/model/source/saved-metric verification; no inference."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def token(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def auc(y,s):
    y=np.asarray(y)>0;p=np.asarray(s,float)[y];n=np.sort(np.asarray(s,float)[~y])
    if not len(p) or not len(n):return None
    a=np.searchsorted(n,p,'left');b=np.searchsorted(n,p,'right')
    return float(np.sum(a+.5*(b-a))/(len(p)*len(n)))

def run(models,report):
    plan=json.loads((models/'PRE_FIT_FREEZE.json').read_text());f=json.loads((models/'ALL_TREES_FROZEN.json').read_text());r=json.loads((report/'RESULTS.json').read_text())
    for v,key in [(plan,'plan_sha256'),(f,'freeze_sha256')]:
        if token({k:x for k,x in v.items() if k!=key})!=v[key]:raise ValueError('Content hash differs')
    for n,h in plan['source_pins'].items():
        if sha(ROOT/n)!=h:raise ValueError('Source changed:'+n)
    for n,h in r['output_sha256'].items():
        if sha(report/n)!=h:raise ValueError('Saved output changed:'+n)
    models_qa={}
    for arm in ('baselineP1','highshareP1'):
        folder=models/arm;s=json.loads((folder/'model/summary.json').read_text());c=json.loads((folder/'COMPLETED.json').read_text())
        if c['summary_sha256']!=sha(folder/'model/summary.json') or s['train_rows']!=78869 or s['n_features']!=5252 or s['fixture_only']:raise ValueError('Full row/model identity')
        if s['settings']['threads']!=2 or s['settings']['maximum_trees']!=1500 or s['settings']['early_stopping_patience']!=40 or s['inner_rows']!=1000:raise ValueError('Policy differs')
        if s['training_binding']!=c['binding'] or s['feature_matrix_bytes']['train']!=78869*5252*4:raise ValueError('Feature/binding differs')
        for n,h in s['artifact_sha256'].items():
            if sha(folder/'model'/n)!=h:raise ValueError('Model artifact changed')
        if sha(folder/'model/basis.npz')!=plan['fold_basis_sha256']:raise ValueError('Fold PCA mismatch')
        material=json.loads((folder/'epoch1_manifest.json').read_text())
        if material['rows']!=78869 or material['NPZ_sha256']!=sha(folder/'epoch1_train.npz') or not material['full_primary_once']:raise ValueError('Materialization identity differs')
        with np.load(folder/'epoch1_train.npz',allow_pickle=False) as z:
            if len(z['db'])!=78869 or len(np.unique(z['base_window_index']))!=78869 or int((z['share']==0).sum())!=15774:raise ValueError('Full once/counts')
            if not np.array_equal(z['share']>0,z['presence'].astype(bool)):raise ValueError('Presence labels differ')
            metadata={k:z[k].copy() for k in ('sample_id','source_id','background_group','base_window_index','reference_index','site','family','explicit_no_source','highshare')}
            target=z['share'].copy()
        if arm=='baselineP1':base_metadata=metadata;base_target=target
        else:
            for k in ('sample_id','source_id','background_group','base_window_index','reference_index','site','family','explicit_no_source'):
                if not np.array_equal(base_metadata[k],metadata[k]):raise ValueError('Paired source/geometry changed:'+k)
            if int((base_target!=target).sum())!=15774:raise ValueError('Expected quarter-positive highshare override')
        models_qa[arm]={'full_rows':78869,'features':5252,'matrix_bytes':s['feature_matrix_bytes']['train'],
            'selection_rows':1000,'best_round':s['best_iteration'],'classifier_best_round':s['presence_best_iteration'],
            'summary_sha256':sha(folder/'model/summary.json'),'peak_RSS_bytes':c['peak_RSS_bytes']}
    rows=list(csv.DictReader((report/'results.csv').open()));index={(v['panel'],v['model']):v for v in rows}
    count=0;max_error=0.;panels=[];exact=0
    def check(label,actual,expected,tol=2e-10):
        nonlocal count,max_error
        count+=1
        if expected is None:
            if actual not in (None,''):raise ValueError(label+' expected undefined')
            return
        e=abs(float(actual)-float(expected));max_error=max(max_error,e)
        if e>tol:raise ValueError(label+' numerical difference '+str(e))
    for path in sorted((report/'predictions').glob('*.npz')):
        name=path.stem;panels.append(name)
        with np.load(path,allow_pickle=False) as z:
            y=z['target_share'].astype(float);p=z['share'];score=z['score'];names=z['model_order'].astype(str).tolist()
            with np.load(ROOT/'reports/r13_cpu_correction_r1/predictions'/path.name,allow_pickle=False) as old:
                if not np.array_equal(z['sample_id'],old['sample_id']) or not np.array_equal(z['target_share'],old['target_share']):raise ValueError('Panel ID/target differs')
                oldnames=old['model_order'].astype(str).tolist()
                for key in ('R12_matched_share_only','R13_selected','R9_frozen_all5'):
                    if key in names:
                        j=names.index(key);k=oldnames.index(key)
                        if not np.array_equal(p[:,j],old['share'][:,k]) or not np.array_equal(score[:,j],old['score'][:,k]):raise ValueError('Old comparator changed')
                        exact+=2
                for arm in ('baselineP1','highshareP1'):
                    j=names.index('CompactTree_'+arm);k=oldnames.index('Tree_'+arm)
                    if not np.array_equal(p[:,j],old['share'][:,k]) or not np.array_equal(score[:,j],old['score'][:,k]):raise ValueError('Compact tree changed')
                    exact+=2
            presence=y>0;zero=~presence;high=y>.2;enriched_high=np.asarray(y,np.float32)>.2
            for j,key in enumerate(names):
                row=index[(name,key)];pp=p[:,j].astype(float);ss=score[:,j].astype(float);e=pp-y;SSE=float(np.sum(e**2))
                values={'n':len(y),'positive_n':int(presence.sum()),'zero_n':int(zero.sum()),'rmse':float(np.sqrt(np.mean(e**2))),
                    'mae':float(np.mean(abs(e))),'bias':float(e.mean()),'prediction_mean':float(pp.mean()),'prediction_rms':float(np.sqrt(np.mean(pp**2))),
                    'target_mean':float(y.mean()),'target_rms':float(np.sqrt(np.mean(y**2))),'auc_share':auc(y,pp),'auc_score':auc(y,ss),
                    'high_y_n':int(high.sum()),'high_y_SSE':float(np.sum(e[high]**2)),'high_y_SSE_fraction':float(np.sum(e[high]**2)/SSE) if SSE else None,
                    'score_brier':float(np.mean((ss-presence)**2)),'zero_prediction_mean':float(pp[zero].mean()) if zero.any() else None,
                    'high_y_bias':float(e[enriched_high].mean()) if enriched_high.any() else None,
                    'high_y_rmse':float(np.sqrt(np.mean(e[enriched_high]**2))) if enriched_high.any() else None,
                    'share_ge099':float(np.mean(pp>=.99)),'score_ge099':float(np.mean(ss>=.99))}
                if key.startswith('FullTree_'):
                    arm=key.removeprefix('FullTree_');c=z[arm+'_classifier'].astype(float);mix=(.5*pp+.5*c).astype(np.float32)
                    if not np.array_equal(p[:,j],score[:,j]):raise ValueError('Full tree default score changed')
                    values.update(auc_classifier=auc(y,c),classifier_auc=auc(y,c),classifier_brier=float(np.mean((c-presence)**2)),
                        classifier_ge099=float(np.mean(c>=.99)),diagnostic_halfshare_halfclassifier_auc=auc(y,mix),
                        diagnostic_halfshare_halfclassifier_brier=float(np.mean((mix.astype(float)-presence)**2)))
                    exact+=1
                for k,v in values.items():check(name+'/'+key+'/'+k,row[k],v)
    if len(panels)!=17 or len(rows)!=r['metric_rows']:raise ValueError('Expected17panels/rows')
    result={'format':'r14_full_tree_independent_saved_QA_v1','status':'passed','full_models':models_qa,
        'actual_fits_verified':'Two full78869-row learners, each5252features, regressor/classifier separately; no fixture models',
        'full_paired_geometry_metadata_exact':True,'highshare_changed_full_rows':15774,
        'panels':17,'metric_rows':len(rows),'scalar_checks':count,'scalar_max_error':max_error,'exact_comparator_checks':exact,
        'original_sources_unchanged':True,'model_fits_or_inference_Test_GPU':False,
        'scope_limits':['Independent point metrics/IDs/model/source bindings; no repeated inference',
            'Site macro and bootstrap CI endpoints not independently reproduced here','Development panels already exposed; not official loss or blind generalization'],
        'source_RESULTS_sha256':sha(report/'RESULTS.json'),'verification_source_sha256':sha(__file__)}
    path=report/'INDEPENDENT_QA.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','checks':count,'report':str(path),'sha256':sha(path)}));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',default='models/r14_tree_full_r1');p.add_argument('--report',default='reports/r14_tree_full_r1');a=p.parse_args();run(ROOT/a.models,ROOT/a.report)
