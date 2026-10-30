#!/usr/bin/env python3
"""Read-only forensics of frozen R17/R18 predictions; never fit or open raw Test.

Reproduce with .venv_r17_eda/bin/python ops/r18_public_prediction_forensics.py.
Only this new script and reports/r18_public_gap_audit_r1/predictions are written.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, os, sys, zipfile
from collections import Counter
from pathlib import Path
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('lightgbm', 'dinov2_small', 'dasheng_base')
REGIMES = ('D0', 'H0', 'D0H0_50_50')
BLENDS = {'B1': np.array([.60, .25, .15]), 'B2': np.array([.50, .30, .20])}
PANELS = ('fresh_fit_D0', 'fresh_fit_H0', 'local_gate', 'local_report',
          'domain_holdout', 'native12_VS_hf_high_negative',
          'native12_VS_st_francis_dense_negative')
PINNED = {}


def guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = Path(os.fsdecode(args[0]))
        if p.name.lower() in ('test.csv', 'sample_submission.csv'):
            raise PermissionError('Raw Test is forbidden in saved-prediction forensics')


sys.addaudithook(guard)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def pin(p):
    p = Path(p)
    PINNED[str(p.relative_to(ROOT))] = sha(p)
    return p


def read(p):
    return json.loads(pin(p).read_text())


def load(p, keys=None):
    with np.load(pin(p), allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys) if k in z}


def write(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False,
                            default=lambda a: a.item() if isinstance(a, np.generic) else str(a)) + '\n')


def slices(y):
    return {'all': np.ones(len(y), bool), 'y=0': y == 0,
            '0<y<=0.03': (y > 0) & (y <= .03),
            '0.03<y<=0.10': (y > .03) & (y <= .10),
            '0<y<=0.10': (y > 0) & (y <= .10)}


def metrics(d, s, c, meta):
    y = d['share'].astype(float)
    w = np.asarray(d.get('weights', np.ones(len(y))), float)
    w /= w.sum()
    rows = []
    for label, ix in slices(y).items():
        n = int(ix.sum())
        row = dict(meta, target_interval=label, n=n, weighting='original_panel_weights_normalized_within_interval',
                   interval_weight_mass=float(w[ix].sum()))
        if n:
            wi = w[ix] / w[ix].sum()
            error = s[ix] - y[ix]
            z = y[ix] > 0
            row.update(rmse=float(np.sqrt(wi @ (error**2))), bias=float(wi @ error),
                       prediction_mean=float(wi @ s[ix]), target_mean=float(wi @ y[ix]),
                       sse=float(error @ error), panel_weighted_sse=float(w[ix] @ (error**2)),
                       unweighted_rmse=float(np.sqrt(np.mean(error**2))),
                       unweighted_bias=float(error.mean()), unweighted_prediction_mean=float(s[ix].mean()),
                       classifier_auc=float(roc_auc_score(z, c[ix], sample_weight=wi)) if z.any() and (~z).any() else None,
                       share_auc=float(roc_auc_score(z, s[ix], sample_weight=wi)) if z.any() and (~z).any() else None,
                       score_mean=float(wi @ c[ix]), score_ge090=float(wi @ (c[ix] >= .9)))
        rows.append(row)
    return rows


def add_averages(a):
    a = dict(a)
    for target in ('share', 'prob'):
        a[target+'_D0H0_50_50'] = .5*(a[target+'_D0'] + a[target+'_H0'])
    return a


def candidate_arrays(members, regime):
    result = {}
    for generation in ('old', 'new'):
        for family in FAMILIES:
            a = members[generation, family]
            result[generation+'_'+family] = (a['share_'+regime], a['prob_'+regime])
        for blend, weights in BLENDS.items():
            result[generation+'_'+blend] = tuple(
                sum(w*members[generation, family][target+'_'+regime]
                    for family, w in zip(FAMILIES, weights)) for target in ('share', 'prob'))
    return result


def panel_members(final, panel, d):
    members = {}
    for generation in ('old', 'new'):
        for family in FAMILIES:
            p = final/'predictions'/generation/family/(panel+'.npz')
            receipt = read(p.with_suffix('.json'))
            a = load(p)
            assert receipt['prediction_sha256'] == sha(p)
            assert receipt['input_sha256'] == sha(final/'inputs'/(panel+'.npz'))
            assert np.array_equal(a['sample_id'].astype(str), d['sample_id'].astype(str))
            for k in ('share_D0', 'share_H0', 'prob_D0', 'prob_H0'):
                a[k] = a[k].astype(float)
                assert a[k].shape == (len(d['sample_id']),) and np.isfinite(a[k]).all()
                assert np.all((a[k] >= 0) & (a[k] <= 1))
            members[generation, family] = add_averages(a)
    return members


def effects(members, regime, target):
    old = np.stack([members['old', f][target+'_'+regime] for f in FAMILIES])
    new = np.stack([members['new', f][target+'_'+regime] for f in FAMILIES])
    generation = BLENDS['B1'][:, None]*(new-old)
    reweight = (BLENDS['B2']-BLENDS['B1'])[:, None]*new
    return old, new, generation, reweight


def model_provenance(dest):
    rows = []
    selection = []
    source_checks = []
    for regime in ('D0', 'H0'):
        folder = ROOT/'models/r18_production_r1/trees'/regime
        summary = read(folder/'summary.json')
        schema = read(folder/'feature_schema.json')
        assert len(schema['regression_relative']) == 4667
        assert len(schema['regression_auxiliary']) == 4679
        assert len(schema['classification_profile_energy']) == 2408
        assert schema['regression_auxiliary'][:4667] == schema['regression_relative']
        assert len(schema['regression_removed_explicit_mask']) == 493
        assert len(schema['classification_removed_explicit_mask']) == 25
        for branch, fit in summary['fits'].items():
            rows.append(dict(generation='new', family='lightgbm', regime=regime, task=branch,
                             fold='single_source_split', train_rows=summary['train_rows'],
                             selection_rows=summary['clean_ES_rows'], epochs_run=None, best_epoch=None,
                             executed_rounds=fit['executed_rounds'], best_iteration=fit['best_iteration'],
                             input_feature_count=fit['feature_count'],
                             selected=branch == summary['selected_regression'] or branch == 'classification',
                             selection_role='clean_ES', summary=str((folder/'summary.json').relative_to(ROOT))))
        selection.append(dict(regime=regime, selected_regression=summary['selected_regression'],
                              relative_rmse=summary['clean_ES_metrics']['relative']['rmse'],
                              auxiliary_rmse=summary['clean_ES_metrics']['auxiliary']['rmse'],
                              auxiliary_minus_relative_rmse=summary['clean_ES_metrics']['auxiliary']['rmse']-summary['clean_ES_metrics']['relative']['rmse'],
                              actual_input_feature_count=len(schema['regression_'+summary['selected_regression']]),
                              auxiliary_selected=summary['selected_regression']=='auxiliary'))
        for name, digest in summary['code_sha256'].items():
            source_checks.append(dict(binding=str((folder/'summary.json').relative_to(ROOT)), path=name,
                                      expected_sha256=digest, actual_sha256=sha(pin(ROOT/name))))
        for fold in range(5):
            p = ROOT/f'models/r16_multisupport_full_r1/trees/{regime}/fold_{fold:02d}/summary.json'
            old = read(p)
            for task in ('regression', 'classification'):
                rows.append(dict(generation='old', family='lightgbm', regime=regime, task=task,
                                 fold=fold, train_rows=old['train_n'], selection_rows=old['selection_metrics']['n'],
                                 epochs_run=None, best_epoch=None, executed_rounds=None,
                                 best_iteration=old['iterations'][task], input_feature_count=old['feature_counts'][task],
                                 selected=True, selection_role='legacy_selection', summary=str(p.relative_to(ROOT))))
        for family in FAMILIES[1:]:
            for generation, base in [('old', 'models/r16_pretrained_cuda_bf16_r1/both'),
                                     ('new', 'models/r18_production_r1/neural')]:
                for task in ('regression', 'classification'):
                    folder = ROOT/base/family/regime/task
                    s, p = read(folder/'summary.json'), read(folder/'PLAN.json')
                    channels = 17 if generation == 'new' or task == 'regression' else 15
                    assert s['status'] == 'completed'
                    if generation == 'new':
                        assert s['input_channels'] == 17 and p['train_n'] == 60000 and p['clean_ES_n'] == 2000
                    rows.append(dict(generation=generation, family=family, regime=regime, task=task,
                                     fold=p.get('fold', 'single_source_split'), train_rows=p['train_n'],
                                     selection_rows=p.get('clean_ES_n', p.get('CV_n')),
                                     epochs_run=s['epochs_run'], best_epoch=s['best_epoch'],
                                     executed_rounds=None, best_iteration=None, input_feature_count=channels,
                                     selected=True, selection_role='clean_ES' if generation=='new' else p['best_selection_role'],
                                     summary=str((folder/'summary.json').relative_to(ROOT))))
                    for name, digest in s['source_sha256'].items():
                        source_checks.append(dict(binding=str((folder/'summary.json').relative_to(ROOT)), path=name,
                                                  expected_sha256=digest, actual_sha256=sha(pin(ROOT/name))))
    pd.DataFrame(rows).to_csv(dest/'model_training_provenance.csv', index=False)
    pd.DataFrame(selection).to_csv(dest/'lgb_auxiliary_selection.csv', index=False)
    checks = pd.DataFrame(source_checks)
    checks['matches'] = checks.expected_sha256 == checks.actual_sha256
    checks.to_csv(dest/'source_hash_verification.csv', index=False)

    # Parse literal feature names without importing Torch/training modules.
    def constant(relative, name):
        tree = ast.parse(pin(ROOT/relative).read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id==name for t in node.targets):
                if isinstance(node.value, ast.BinOp):
                    return ast.literal_eval(node.value.right)
                return ast.literal_eval(node.value)
        raise KeyError(name)
    names = (constant('pipeline/relative_grid_r6.py', 'FEATURE_NAMES')+
             constant('pipeline/relative_grid_r6_v2.py', 'FEATURE_NAMES')+
             constant('ops/r16_temporal_features.py', 'GRID_NAMES'))
    assert len(names) == 17
    old_clf = constant('pipeline/training_r16_pretrained_cuda_bf16.py', 'CLASSIFIER_CHANNELS')
    pd.DataFrame([dict(index=i, name=n, new_regression=True, new_classification=True,
                       old_regression=True, old_classification_position=old_clf.index(i) if i in old_clf else None,
                       group='relative_grid_v2' if i<8 else 'temporal_grid')
                  for i,n in enumerate(names)]).to_csv(dest/'neural_17channel_dictionary.csv', index=False)
    schema = read(ROOT/'models/r18_production_r1/trees/D0/feature_schema.json')
    feature_rows = []
    for kind in ('regression_relative', 'regression_auxiliary', 'classification_profile_energy',
                 'regression_removed_explicit_mask', 'classification_removed_explicit_mask'):
        for i,n in enumerate(schema[kind]):
            feature_rows.append(dict(schema=kind, index=i, name=n, prefix=n.split('/')[0]))
    pd.DataFrame(feature_rows).to_csv(dest/'tree_feature_dictionary.csv', index=False)
    return rows, selection, int((~checks.matches).sum())


def clean_es(dest):
    rows = []
    availability = []
    for regime in ('D0', 'H0'):
        folder = ROOT/'models/r18_production_r1/trees'/regime
        a = load(folder/'clean_ES_predictions.npz')
        s = read(folder/'summary.json')
        assert s['artifact_sha256']['clean_ES_predictions.npz'] == sha(folder/'clean_ES_predictions.npz')
        d = {'share': a['target_share'], 'sample_id': a['sample_id']}
        members = {'lightgbm': (a['share_'+s['selected_regression']].astype(float), a['classifier_prob'].astype(float))}
        for family in FAMILIES[1:]:
            preds=[]
            for task in ('regression', 'classification'):
                p=ROOT/'models/r18_production_r1/neural'/family/regime/task/'clean_ES_best.npz'
                b=load(p)
                summary=read(p.parent/'summary.json')
                assert summary['artifacts']['clean_ES_best.npz'] == sha(p)
                assert np.array_equal(b['sample_id'],d['sample_id'])
                assert np.array_equal(b['target_share'],d['share'])
                preds.append(b['prediction'].astype(float))
            members[family] = tuple(preds)
        for blend,w in BLENDS.items():
            members[blend]=tuple(sum(wi*members[f][j] for f,wi in zip(FAMILIES,w)) for j in (0,1))
        for label,(share,prob) in members.items():
            rows.extend(metrics(d,share,prob,dict(panel='clean_ES_'+regime,candidate='new_'+label,
                                                regime=regime,scope='own_geometry_only')))
        # Auxiliary is retained solely as a rejected, labelled ablation.
        rows.extend(metrics(d,a['share_auxiliary'].astype(float),a['classifier_prob'].astype(float),
                            dict(panel='clean_ES_'+regime,candidate='new_lightgbm_rejected_auxiliary',regime=regime,
                                 scope='rejected_ablation_not_final_model')))
        availability.append(dict(panel='clean_ES_'+regime,available='new own-regime family and B1/B2',
                                 unavailable='old; cross-regime; D0:H0=50:50',
                                 reason='No stored predictions on a common clean_ES input. D0/H0 IDs are paired but observations and labels differ; cannot average own-geometry predictions. No new inference or training was run.'))
    pd.DataFrame(availability).to_csv(dest/'clean_ES_saved_prediction_availability.csv',index=False)
    return rows


def query(dest, members, ids):
    old_csv = ROOT/'submissions/r17_fixed_blends_20261030/submission_r17_B1_L60_DINO25_Dash15.csv'
    csvd = pd.read_csv(pin(old_csv))
    assert len(csvd)==1500 and csvd.ID.astype(str).tolist()==ids.astype(str).tolist()
    reference = (csvd.share_added_vessel.to_numpy(float),csvd.score_added_vessel.to_numpy(float))
    base = candidate_arrays(members,'D0H0_50_50')['old_B1']
    exact = {target: float(np.max(np.abs(a-b))) for target,a,b in zip(('share','prob'),reference,base)}
    assert max(exact.values())<1e-12
    rows, effects_rows, contributions = [], [], []
    rowdata={'ID':ids}
    for regime in REGIMES:
        for label,(s,c) in candidate_arrays(members,regime).items():
            row=dict(candidate=label,regime=regime,n=len(s),share_mean=float(s.mean()),
                     share_rms=float(np.sqrt(np.mean(s*s))),share_median=float(np.median(s)),
                     share_p90=float(np.quantile(s,.9)),share_std=float(s.std()),
                     score_mean=float(c.mean()),score_ge090=float(np.mean(c>=.9)))
            for target,a,b in zip(('share','score'),(s,c),reference):
                row.update({target+'_oldB1_pearson':float(np.corrcoef(a,b)[0,1]),
                            target+'_oldB1_spearman':float(spearmanr(a,b).statistic),
                            target+'_oldB1_mean_difference':float(np.mean(a-b))})
            rows.append(row)
            for target,a in zip(('share','prob'),(s,c)):
                rowdata[label+'__'+regime+'__'+target] = a
        for target in ('share','prob'):
            old,new,generation,reweight=effects(members,regime,target)
            old_b1=BLENDS['B1']@old
            new_b1=BLENDS['B1']@new
            new_b2=BLENDS['B2']@new
            assert np.allclose(generation.sum(0),new_b1-old_b1,rtol=0,atol=5e-16)
            assert np.allclose(reweight.sum(0),new_b2-new_b1,rtol=0,atol=5e-16)
            for effect,component,source,destination in [('old_to_new_same_B1_weights',generation,old_b1,new_b1),
                                                       ('new_B1_to_new_B2_weights',reweight,new_b1,new_b2)]:
                total=destination-source
                for family,part in zip(FAMILIES,component):
                    rowdata[effect+'__'+family+'__'+regime+'__'+target]=part
                    # Sum of these mean-square terms exactly equals E[dest^2-source^2].
                    ms_contribution=float(np.mean(part*(destination+source)))
                    effects_rows.append(dict(effect=effect,family=family,regime=regime,target=target,
                                             mean_delta=float(part.mean()),rms_delta=float(np.sqrt(np.mean(part**2))),
                                             median_delta=float(np.median(part)),p90_delta=float(np.quantile(part,.9)),
                                             mean_square_prediction_change_contribution=ms_contribution,
                                             mean_delta_fraction=float(part.mean()/total.mean()) if abs(total.mean())>1e-15 else None,
                                             no_query_labels=True))
            for blend,w in BLENDS.items():
                for generation_name,a in [('old',old),('new',new)]:
                    for family,wi,part in zip(FAMILIES,w,a):
                        contributions.append(dict(generation=generation_name,blend=blend,family=family,regime=regime,
                                                  target=target,weight=float(wi),raw_model_mean=float(part.mean()),
                                                  weighted_prediction_mean=float((wi*part).mean()),
                                                  weighted_prediction_rms=float(np.sqrt(np.mean((wi*part)**2)))))
    pd.DataFrame(rows).to_csv(dest/'query_distribution_and_oldB1_correlations.csv',index=False)
    pd.DataFrame(effects_rows).to_csv(dest/'query_exact_generation_and_weight_effects.csv',index=False)
    pd.DataFrame(contributions).to_csv(dest/'query_weighted_model_contributions.csv',index=False)
    pd.DataFrame(rowdata).to_csv(dest/'query_all_1500_rows_predictions_and_effects.csv',index=False)
    write(dest/'query_verification.json',{'rows':1500,'query_labels_available':False,'raw_Test_opened':False,
                                        'original_R17_B1_max_abs_difference':exact,
                                        'decomposition':'newB2-oldB1 = sum_f B1_f*(new_f-old_f) + sum_f (B2_f-B1_f)*new_f; exact per row and per independent target',
                                        'cannot_identify_query_RMSE_bias_or_true_error_without_labels':True})
    return rows, effects_rows


def weak_classifier_and_train_prior(dest, final):
    """Rank diagnostics and supervised TRAIN targets only; no new inference."""
    classifier=[]
    for panel in PANELS:
        d=load(final/'inputs'/(panel+'.npz'),('share','sample_id','weights'))
        y=d['share'];ix=y<=.10;zero=y==0;weak=(y>0)&(y<=.10)
        w=np.asarray(d.get('weights',np.ones(len(y))),float)
        members=panel_members(final,panel,d)
        for regime in REGIMES:
            for candidate in ('old_B1','new_B2'):
                _,prob=candidate_arrays(members,regime)[candidate]
                classifier.append(dict(panel=panel,candidate=candidate,regime=regime,
                                       n=int(ix.sum()),zero_n=int(zero.sum()),weak_positive_n=int(weak.sum()),
                                       classifier_auc_zero_vs_weak=float(roc_auc_score(y[ix]>0,prob[ix],sample_weight=w[ix])),
                                       zero_score_mean=float(np.average(prob[zero],weights=w[zero])),
                                       zero_score_ge090_fpr=float(np.average(prob[zero]>=.9,weights=w[zero])),
                                       weak_positive_score_mean=float(np.average(prob[weak],weights=w[weak])),
                                       weak_positive_score_ge090_tpr=float(np.average(prob[weak]>=.9,weights=w[weak]))))
    pd.DataFrame(classifier).to_csv(dest/'zero_vs_weak_classifier_rank_and_fpr.csv',index=False)
    oldmanifest=read(ROOT/'data/synthetic/r16_multisupport_v1/manifest.json')
    newmanifest=read(ROOT/'data/synthetic/r18_production_v1/manifest.json')
    targets=[];bins=[];missing=[]
    for generation,manifest in [('old_R16MS_fold0',oldmanifest),('new_R18_production',newmanifest)]:
        for regime in ('D0','H0'):
            e=manifest['folds']['0']['roles']['train'][regime] if generation=='old_R16MS_fold0' else manifest['roles']['train'][regime]
            p=ROOT/e['path']
            if not p.exists():
                missing.append(dict(generation=generation,regime=regime,path=e['path'],reason='Manifest TRAIN NPZ absent locally'))
                continue
            actual=sha(pin(p));assert actual==e['sha256']
            # Explicitly open only share.npy. db/cell_share are never decompressed.
            with zipfile.ZipFile(p) as archive:
                with archive.open('share.npy') as stream:
                    y=np.lib.format.read_array(stream,allow_pickle=False).astype(float)
            assert y.shape==(e['n'],) and np.isfinite(y).all()
            total_sse=float(y@y)
            assert abs(float(y.mean())-e['share_mean'])<1e-12
            assert abs(float(np.sqrt(np.mean(y*y)))-e['share_rms'])<1e-12
            targets.append(dict(generation=generation,regime=regime,n=len(y),target_mean=float(y.mean()),
                                target_rms=float(np.sqrt(np.mean(y*y))),target_sse=total_sse,
                                source_path=e['path'],sha256=actual,loaded_npz_members='share.npy only',
                                label='supervised synthetic TRAIN targets; not Test labels'))
            masks={'y=0':y==0,'0<y<=0.03':(y>0)&(y<=.03),'0.03<y<=0.10':(y>.03)&(y<=.10),
                   '0.10<y<=0.20':(y>.10)&(y<=.20),'y>0.20':y>.20}
            assert sum(int(a.sum()) for a in masks.values())==len(y)
            for label,ix in masks.items():
                sse=float(y[ix]@y[ix])
                bins.append(dict(generation=generation,regime=regime,target_interval=label,n=int(ix.sum()),
                                 row_fraction=float(ix.mean()),target_sse=sse,target_sse_fraction=sse/total_sse,
                                 target_mean_in_bin=float(y[ix].mean()) if ix.any() else None))
    pd.DataFrame(targets).to_csv(dest/'train_target_prior_overall.csv',index=False)
    pd.DataFrame(bins).to_csv(dest/'train_target_prior_bins.csv',index=False)
    write(dest/'train_target_prior_loading_receipt.json',{'loaded_array_members':['share.npy'],
          'db_loaded':False,'cell_share_loaded':False,'missing_sources':missing,
          'scope':'Old R16MS fold0 is exact old neural TRAIN; old LightGBM additionally averages folds1-4, so this is a fold0 contrast, not the full old five-fold training mixture.',
          'targetSSE_definition':'sum(y^2), the squared error of the constant-zero prediction; not a learned model error.'})
    return classifier,targets,bins


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=ROOT/'reports/r18_public_gap_audit_r1/predictions')
    args=ap.parse_args();dest=args.output.resolve();dest.mkdir(parents=True,exist_ok=True)
    pin(Path(__file__).resolve())
    final=ROOT/'reports/r18_production_r1/finalization'
    policy=read(final/'selected_policy.json');plan=read(final/'PLAN.json');freeze=read(final/'DECISION_FREEZE.json')
    assert plan['B1']==BLENDS['B1'].tolist() and plan['B2']==BLENDS['B2'].tolist()
    assert policy['D0_H0_weights']==[.5,.5] and policy['score_share_weight']==0
    rows=[];error_effect_rows=[]
    for panel in PANELS:
        d=load(final/'inputs'/(panel+'.npz'),('share','sample_id','weights'))
        members=panel_members(final,panel,d)
        for regime in REGIMES:
            for label,(s,c) in candidate_arrays(members,regime).items():
                rows.extend(metrics(d,s,c,dict(panel=panel,candidate=label,regime=regime,scope='same_common_input')))
            for target in ('share','prob'):
                old,new,generation,reweight=effects(members,regime,target)
                if target=='prob':continue
                y=d['share'];w=np.asarray(d.get('weights',np.ones(len(y))),float);w=w/w.sum()
                for effect,component,source,destination in [('old_to_new_same_B1_weights',generation,BLENDS['B1']@old,BLENDS['B1']@new),
                                                           ('new_B1_to_new_B2_weights',reweight,BLENDS['B1']@new,BLENDS['B2']@new)]:
                    for interval,ix in slices(y).items():
                        if not ix.any():continue
                        wi=w[ix]/w[ix].sum()
                        for family,part in zip(FAMILIES,component):
                            # Identity delta*MSE = sum delta_family*(new+old-2*y).
                            attribution=float(wi@(part[ix]*(destination[ix]+source[ix]-2*y[ix])))
                            error_effect_rows.append(dict(panel=panel,regime=regime,effect=effect,family=family,
                                                         target_interval=interval,n=int(ix.sum()),mean_prediction_delta=float(wi@part[ix]),
                                                         mse_change_attribution=attribution,
                                                         total_mse_change=float(wi@((destination[ix]-y[ix])**2-(source[ix]-y[ix])**2))))
    rows.extend(clean_es(dest))
    metrics_df=pd.DataFrame(rows);metrics_df.to_csv(dest/'all_saved_prediction_metrics.csv',index=False)
    metrics_df[metrics_df.target_interval=='all'].to_csv(dest/'overall_metrics.csv',index=False)
    pd.DataFrame(error_effect_rows).to_csv(dest/'labelled_exact_mse_change_attribution.csv',index=False)
    verification=[]
    for stage in ('development','report'):
        original=pd.read_csv(pin(final/('BLEND_METRICS_'+stage+'.csv')))
        original=original[original.candidate.isin(['old_B1','old_B2','new_B1','new_B2'])]
        common=metrics_df[(metrics_df.target_interval=='all')&(metrics_df.regime=='D0H0_50_50')].merge(original,on=['panel','candidate'],suffixes=('_audit','_original'))
        for name in ('rmse','bias','classifier_auc'):
            difference=float(np.max(np.abs(common[name+'_audit']-common[name+'_original'])))
            assert difference<1e-12
            verification.append(dict(check=stage+'_'+name,n=len(common),max_abs_residual=difference))
    attribution=pd.DataFrame(error_effect_rows)
    residual=max(abs(g.mse_change_attribution.sum()-g.total_mse_change.iloc[0]) for _,g in attribution.groupby(['panel','regime','effect','target_interval']))
    assert residual<1e-12
    verification.append(dict(check='all_labelled_mse_attribution_sum_identity',n=len(attribution),max_abs_residual=residual))
    pd.DataFrame(verification).to_csv(dest/'verification_against_frozen_metrics.csv',index=False)
    # Complementary high-share rows expose what offsets low-share degradation.
    group_rows=[]
    for (panel,candidate),g in metrics_df[(metrics_df.regime=='D0H0_50_50')&metrics_df.candidate.isin(['old_B1','new_B1','new_B2'])].groupby(['panel','candidate']):
        g=g.set_index('target_interval')
        for label in ('all','y=0','0<y<=0.10'):
            group_rows.append(dict(panel=panel,candidate=candidate,group=label,n=int(g.loc[label,'n']),
                                   rmse=g.loc[label,'rmse'],bias=g.loc[label,'bias'],sse=g.loc[label,'sse'],
                                   weighted_mse_mass=g.loc[label,'panel_weighted_sse']))
        n=int(g.loc['all','n']-g.loc['y=0','n']-g.loc['0<y<=0.10','n'])
        sse=float(g.loc['all','sse']-g.loc['y=0','sse']-g.loc['0<y<=0.10','sse'])
        weighted_sse=float(g.loc['all','panel_weighted_sse']-g.loc['y=0','panel_weighted_sse']-g.loc['0<y<=0.10','panel_weighted_sse'])
        mass=float(1-g.loc['y=0','interval_weight_mass']-g.loc['0<y<=0.10','interval_weight_mass'])
        bias_mass=float(g.loc['all','bias']-g.loc['y=0','bias']*g.loc['y=0','interval_weight_mass']-g.loc['0<y<=0.10','bias']*g.loc['0<y<=0.10','interval_weight_mass'])
        group_rows.append(dict(panel=panel,candidate=candidate,group='y>0.10',n=n,
                               rmse=float(np.sqrt(weighted_sse/mass)),bias=bias_mass/mass,
                               sse=sse,weighted_mse_mass=weighted_sse))
    pd.DataFrame(group_rows).to_csv(dest/'zero_low_high_error_tradeoff.csv',index=False)
    d=load(final/'inputs/query.npz',('sample_id',))
    assert len(d['sample_id'])==1500
    query_rows,query_effects=query(dest,panel_members(final,'query',d),d['sample_id'])
    provenance,selection,source_mismatches=model_provenance(dest)
    weak_classifier,train_targets,train_bins=weak_classifier_and_train_prior(dest,final)
    summary={'status':'completed','metrics_rows':len(metrics_df),'labelled_panel_count':len(PANELS)+2,
             'common_input_predictions':len(PANELS)*2*3,'query_rows':1500,'model_provenance_rows':len(provenance),
             'selection_policy_unchanged':True,'training_performed':False,'new_inference_performed':False,
             'raw_Test_opened':False,'query_labels_available':False,'Test_labels_inferred':False,
             'Test_based_calibration':False,'source_hash_mismatches':source_mismatches,
             'lgb_selection':selection,'selected_policy':policy['selected'],
             'clean_ES_limitation':'Only new own-regime best predictions exist. D0/H0 clean_ES are different observations/targets despite paired IDs. Missing old/cross/50:50 metrics were not fabricated.',
             'metric_semantics':'Native12 full metrics use stored positive/negative 50:50 panel weights. Interval metrics normalize those weights within each interval. sse is unweighted sum of squared errors; panel_weighted_sse uses globally normalized original weights. Unweighted companions are supplied.',
             'selection_limitations':'Frozen report/gates have historical repeated development exposure. Old five-fold LGB can share development sources. clean_ES is checkpoint/branch selection data, not an independent new test.',
             'query_limitations':'All query comparisons are label-free prediction descriptions. No query RMSE, bias, prevalence, correct direction, or public-score failure cause can be established from these predictions alone.'}
    write(dest/'summary.json',summary)
    write(dest/'input_sha256.json',PINNED)
    # A concise, reproducible table for the parent audit.
    out=['# R18 저장 예측 전수 진단','',
         '추가 학습·추론 없이 저장 예측만 사용했다. raw Test.csv는 열지 않았다. Query 1,500행에는 라벨이 없으므로 아래 분포 변화는 오류 원인의 증명이 아니다. ensemble 가중치와 선택 정책은 변경하지 않았다.','',
         'B1=LGB 0.60+DINO 0.25+Dasheng 0.15, B2=0.50+0.30+0.20. 각 계열 D0:H0=50:50, 독립 분류 확률에도 같은 비중.','',
         '## 최종 LGB 확인','',pd.DataFrame(selection).to_markdown(index=False,floatfmt='.8f'),'',
         'D0/H0 모두 상대 4,667특징이 최종 선택이다. 공통 레벨/조건부 TRAIN 배경 12개 보조특징을 더한 4,679특징은 clean_ES에서 탈락했다. 선택되지 않은 aux를 최종 실패 원인으로 해석할 수 없다.','',
         '## 공통 패널 고정 50:50 지표','']
    table=metrics_df[(metrics_df.target_interval=='all')&(metrics_df.regime=='D0H0_50_50')&metrics_df.candidate.isin(['old_B1','new_B1','new_B2'])]
    out += [table[['panel','candidate','n','rmse','bias','prediction_mean','classifier_auc']].to_markdown(index=False,floatfmt='.6f'),'',
            '## 전체 개선과 저share 악화의 공존','',
            '전체 RMSE 개선은 주로 y>0.10 오차 감소에서 왔다. local_report에서 old B1→new B2 전체 RMSE는 0.078255→0.073748이지만, 0<y≤0.10의 1,070행은 RMSE 0.026007→0.040986, bias 0.008883→0.019392로 악화했다. y>0.10의 850행은 RMSE 0.123603→0.110186으로 개선했다. 이 저share 약화는 fresh_fit 두 입력·local_gate·native HF/SF에도 보이며 domain_holdout에서는 반대다. 저share 취약성은 검증 라벨로 확인된 사실이지만, 실제 Query에도 같은 정답 구성이 있다고 가정할 수는 없다.','',
            metrics_df[(metrics_df.target_interval=='0<y<=0.10')&(metrics_df.regime=='D0H0_50_50')&metrics_df.candidate.isin(['old_B1','new_B2'])][['panel','candidate','n','rmse','bias','sse']].to_markdown(index=False,floatfmt='.6f'),'',
            '## Query 분포 (라벨 없음)','']
    q=pd.DataFrame(query_rows);q=q[(q.regime=='D0H0_50_50')&q.candidate.isin(['old_B1','new_B1','new_B2','new_lightgbm','new_dinov2_small','new_dasheng_base'])]
    out += [q[['candidate','share_mean','share_rms','share_median','share_p90','share_std','score_mean','score_ge090','share_oldB1_pearson','share_oldB1_spearman','share_oldB1_mean_difference']].to_markdown(index=False,floatfmt='.6f'),'',
            '## Query 변화의 정확한 분해','']
    e=pd.DataFrame(query_effects);e=e[(e.regime=='D0H0_50_50')&(e.target=='share')]
    out += [e[['effect','family','mean_delta','rms_delta','mean_square_prediction_change_contribution']].to_markdown(index=False,floatfmt='.8f'),'',
            '## 저share 독립 분류 순위와 음성 FPR','',
            'y=0과0<y≤.10만 남긴 분류 AUC는 두 군 사이 순위를 직접 평가한다. 음성 score≥.9 비율은 고정 threshold의 FPR이며, score 평균이나 높은 score 비율만으로 AUC 실패를 판단하지 않는다.','',
            pd.DataFrame(weak_classifier).query("regime == 'D0H0_50_50'")[['panel','candidate','n','classifier_auc_zero_vs_weak','zero_score_mean','zero_score_ge090_fpr']].to_markdown(index=False,floatfmt='.6f'),'',
            '## TRAIN 타깃 사전분포 검산','',
            '기존 R16MS fold0 및 R18 Production TRAIN archive의 share.npy만 읽었다. db/cell_share는 로드하지 않았다. targetSSE는 sum(y²), 즉 항상0 예측의 오차 규모이며 모델 오차가 아니다. old neural은 이 fold0 TRAIN, old LGB는 추가 fold1–4 평균이라는 범위 차이가 있다.','',
            pd.DataFrame(train_targets)[['generation','regime','n','target_mean','target_rms','target_sse']].to_markdown(index=False,floatfmt='.6f'),'',
            pd.DataFrame(train_bins)[['generation','regime','target_interval','n','row_fraction','target_sse_fraction']].to_markdown(index=False,floatfmt='.6f'),'',
            '## 재현과 해석 범위','',
            '`.venv_r17_eda/bin/python ops/r18_public_prediction_forensics.py`로 독립 재현한다. 전체 1,500행·각 계열·D0/H0/50:50 예측과 기여도를 CSV에 저장했다. 구간별 n/RMSE/bias/예측평균/SSE와 classifier AUC는 all_saved_prediction_metrics.csv에 있다. Native12에는 원래 양성/음성 50:50 가중치를 유지하고 비가중 동반 지표도 제공한다.','',
            'clean_ES 저장파일에는 각 새 계열의 own-regime 예측만 있다. D0/H0는 ID가 같아도 입력과 정답이 달라 두 예측을 평균하면 다른 문제의 정답을 섞게 된다. 따라서 clean_ES의 old·교차 regime·50:50은 계산 불가로 명시했다.','',
            '고정 report/gate의 반복 개발 노출, old LGB 5-fold의 개발 원천 공유, clean_ES의 checkpoint 선택 사용을 고려해야 한다. Query 평균 상승이 실제 과대예측인지, 낮은 예측이 정답에 가까운지는 라벨 없이 판별할 수 없다.']
    (dest/'PREDICTION_FORENSICS.md').write_text('\n'.join(out)+'\n')
    write(dest/'output_sha256.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='output_sha256.json'})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
