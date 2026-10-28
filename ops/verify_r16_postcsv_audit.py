"""Preflight the additive audit on fixtures and TRAIN/development fit only.

Does not train, read Test, unlock fixed evaluation panels or mutate training.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def guard_fixture():
    from ops.audit_r16_first_ensemble import guard,OPS
    os.environ['MPLCONFIGDIR']=str(OPS/'fixture_mplconfig')
    sys.addaudithook(guard)
    attempts=[(ROOT/'raw/Test.csv','r',0),
              (ROOT/'models/example/query_predictions.npz','rb',0),
              (ROOT/'pipeline/production_must_not_change.py','w',0)]
    for args in attempts:
        try:guard('open',args)
        except PermissionError:pass
        else:raise AssertionError('Forbidden path accepted')
    path=OPS/'guard_fixture.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('{"guard": "passed"}\n')
    assert json.loads(path.read_text())['guard']=='passed'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();ax.plot([0,1],[0,1]);fig.savefig(OPS/'guard_plot_fixture.png');plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--guard-fixture',action='store_true');args=parser.parse_args()
    if args.guard_fixture:guard_fixture();return
    import numpy as np,torch
    from sklearn.metrics import roc_auc_score
    from ops.r16_postaudit_metrics import (score_aggregates,classification_metrics,
          capped_fit,block_bootstrap,residual_audit,CENTER)
    from ops.audit_r16_first_ensemble import (ROOT,OPS,REPORT,static_tree_features,
          donors,verify_parent,sha,write,Trees)
    from pipeline.training_r16_multisupport import codes,features,tree_features,MODELS
    from pipeline.r16_multisupport_data import role,load
    from pipeline.models_r16_multisupport import MultiSupportModel
    pins=codes();began=time.perf_counter();checks=[];rng=np.random.default_rng(2026102815)
    torch.set_num_threads(1)

    # Known rank invariance and independent cohort sensitivity counterexample.
    p=rng.uniform(.001,.999,(80,5));z=np.arange(80)%3>0
    a=score_aggregates(p,fit_probabilities=p)
    assert np.array_equal(np.argsort(a['logit_mean']),np.argsort(a['logit_mean_prior80_to50']))
    assert roc_auc_score(z,a['logit_mean'])==roc_auc_score(z,a['logit_mean_prior80_to50'])
    query=p[:5];fixed_a=score_aggregates(query,fit_probabilities=p)
    extended=score_aggregates(np.r_[query,rng.uniform(.01,.99,(19,5))],fit_probabilities=p)
    assert np.array_equal(fixed_a['rank_mean_fixed_local_fit_CDF'],extended['rank_mean_fixed_local_fit_CDF'][:5])
    assert not np.array_equal(fixed_a['rank_mean_panel_diagnostic_only'],extended['rank_mean_panel_diagnostic_only'][:5])
    m=classification_metrics([0,1,0,1],[.2,.2,.8,.8],conditional_mask=[1,0,1,0])
    assert m['float64_duplicate_count']==m['CSV17g_duplicate_count']==2
    assert m['conditional_same_probmean_top_quartile']['auc'] is None
    assert m['auc']==roc_auc_score([0,1,0,1],[.2,.2,.8,.8])
    checks.append('Rank invariance, fixed-CDF query-cohort independence, exact float64 ties and one-class AUC')

    # Symmetric errors are ties, not >100% wins; strict caps survive block draws.
    assert residual_audit([.5],[.4],[.6],[0,1])['discordant_new_win_rate']==.5
    y=rng.uniform(0,.7,72);x=np.clip(y[:,None]+rng.normal(0,.09,(72,5)),0,1)
    w=capped_fit(x,y);assert abs(w.sum()-1)<1e-8 and w[0]>=.5-1e-8 and max(w[1:])<=.25+1e-8
    groups=np.repeat(np.arange(12),6);shares=np.stack([x,np.clip(x+.01,0,1)])
    b=block_bootstrap(shares,y,groups,reps=20,seed=23)
    assert b==block_bootstrap(shares,y,groups,reps=20,seed=23)
    assert b['groups']==12 and b['valid_OOB_replications']>0
    assert b['bootstrap_SD_not_divided_by_sqrt_replications']
    checks.append('Exclusive discordant ties, paired source-cluster bootstrap determinism and weight caps')

    # Actual unprotected local_fit rows: no report/domain/long panels.
    d=role(0,'D0','selection');db=d['db'][:16];xx,ww=features(db)
    rebuilt=static_tree_features(db,xx);actual=tree_features(db,'regression')
    assert rebuilt.shape==(16,5160) and np.array_equal(rebuilt,actual)
    zero=np.flatnonzero(d['share']==0);unique,idx=np.unique(d['base_window_index'][zero],return_index=True)
    zd=d['db'][zero[np.sort(idx)]];zs=d['site'][zero[np.sort(idx)]]
    pairs=donors(zd,zs)
    for i,j in pairs:
        assert zs[i]!=zs[j] and np.array_equal(np.isfinite(zd[i]),np.isfinite(zd[j]))
    checks.append('Actual 5,160 Tree feature baseline bitwise equality and cross-site exact-mask donor restrictions')

    # Four architectures: baseline isolated mask equals production, forbidden
    # input-only sqrt(w) remains rejected by PRODUCTION forward. Classifier
    # is algebraically independent of both regression-only static channels.
    nn_checks={}
    x=torch.from_numpy(xx[:2]);pool=torch.from_numpy(ww[:2])
    for kind in ('bg_cnn','fdy','itransformer','tfgrid_fdy_conformer'):
        torch.manual_seed(218);model=MultiSupportModel(kind).eval()
        with torch.no_grad():
            pred,_,q=model(x,pool)
            direct=model.regressor._cell_mask(x)
            direct=torch.where(pool>0,direct,torch.zeros_like(direct))
            assert torch.allclose(pred,(direct*pool).sum((1,2)),atol=2e-6,rtol=0)
            changed=x.clone();changed[:,4]*=-1;changed[:,7]*=.8
            assert torch.equal(model.classifier(x),model.classifier(changed))
            try:model(changed,pool)
            except ValueError:pass
            else:raise AssertionError('Production sqrt(w) contract was weakened')
        nn_checks[kind]={'fresh_fixture_only':True,'parameters':sum(v.numel() for v in model.parameters()),
           'mask_pool_forward_max_abs':float(abs(pred-(direct*pool).sum((1,2))).max()),
           'independent_classifier_static_invariant':True,'production_sqrtw_contract_rejects_intervention':True}
        del model
    checks.append('All four CPU architecture fixtures: mask/pooling parity and independent-classifier invariance')

    # Actual ten completed Tree local_fit predictions, no inference/retraining.
    stored=[];n=0
    for r in ('D0','H0'):
        for fold in range(5):
            path=MODELS/'trees'/r/f'fold_{fold:02d}'/'selection.npz';v=load(path)
            assert np.array_equal(v['sample_id'],d['sample_id']) and np.array_equal(v['target_share'],d['share'])
            assert np.isfinite(v['classifier_prob']).all();stored.append(v['classifier_prob']);n+=1
    cp=np.column_stack(stored);cm=classification_metrics(d['share']>0,cp.mean(1))
    checks.append('All ten completed Trees: saved local_fit IDs and targets aligned')
    trees=Trees();newshare,newprob,oldshare=trees.predict(db)
    assert newshare.shape==newprob.shape==(16,10) and oldshare.shape==(16,5)
    assert np.allclose(newprob,cp[:16],atol=2e-8,rtol=1e-7)
    assert np.isfinite(oldshare).all()
    checks.append('Actual 15 Tree checkpoint pairs load and development inference parity')
    parent_phase=json.loads((ROOT/'ops/r16_multisupport_full_r1/state.json').read_text())['phase']
    if parent_phase!='completed':
        try:verify_parent()
        except ValueError as e:assert 'CSV required' in str(e)
        else:raise AssertionError('Audit completion barrier bypassed')
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--guard-fixture'],cwd=ROOT,check=True)
    checks.append('Read/write query exclusion and first-complete-CSV barrier')
    assert codes()==pins
    assert not REPORT.exists(),'Final report must not exist before real audit'
    out={'status':'passed','checks':checks,'neural_fixture_checks':nn_checks,
      'local_fit_only_development_smoke':{'rows':len(d['share']),'unique_backgrounds':len(np.unique(d['base_window_index'])),
       'background_groups':len(np.unique(d['background_group'])),'completed_Tree_pairs':n,
       'stored_Tree10_probability_mean_metrics':cm,'other_site_exact_mask_negative_pairs':len(pairs)},
      'locked_evaluation_executed':False,'Test_read':False,'model_training':False,
      'original_source_sha256_unchanged':pins,'parent_phase':parent_phase,'seconds':time.perf_counter()-began}
    write(OPS/'PREFLIGHT_VERIFICATION.json',out)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
