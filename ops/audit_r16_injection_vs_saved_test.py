"""Describe existing R16 TRAIN against archived, unlabeled Test summaries.

No new Test parsing, fitting, synthesis, inference, tuning or target recovery.
Thresholds and feature definitions are reused unchanged from the R15 audit.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd

from ops.r15_score_duration_audit import regions, temporal

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'data/synthetic/r16_multisupport_v1/manifest.json'
TEST = 'reports/r15_score_duration_r3/Test_derived.npz'
DELIVERY = 'reports/r15_score_duration_r3/DELIVERY_MANIFEST.json'
BANDS = (31.6,39.8,50.1,63.1,79.4,100.,125.9,158.5,199.5,251.2,
         316.2,398.1,501.2,631.,794.3,1000.,1258.9,1584.9,1995.3,
         2511.9,3162.3,3981.1,5011.9,6309.6,7943.3)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def stats(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(n_valid=0, mean=None, rms=None, p10=None, p50=None, p90=None)
    return dict(n_valid=len(v), mean=float(v.mean()),
                rms=float(np.sqrt(np.mean(v*v))),
                **{f'p{int(q*100)}': float(np.quantile(v,q)) for q in (.1,.5,.9)})


def flatten(region_stats):
    return {r+'__'+k: v for r, d in region_stats.items() for k,v in d.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='reports/r16_injection_vs_saved_test_r1')
    a = p.parse_args()
    out = (ROOT/a.output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'reports'):
        raise FileExistsError('Use a fresh reports directory')
    out.mkdir(parents=True)
    started = time.time()
    manifest = read(ROOT/MANIFEST)
    delivery = read(ROOT/DELIVERY)
    if sha(ROOT/TEST) != delivery['files_sha256']['Test_derived.npz']:
        raise ValueError('Archived Test features differ from their delivery SHA')
    band_test = 'reports/r15_band_comparison_r1/Test_derived.npz'
    pins = {s:sha(ROOT/s) for s in (MANIFEST,TEST,DELIVERY,band_test,
            'ops/r15_score_duration_audit.py',
            'reports/r15_score_duration_r3/USE_RESTRICTIONS.json',
            'reports/r15_validation_checks_r1/POLICY_PAUSE_RULES_VERIFICATION.json')}
    plan = dict(scope='Fold0 complete stored TRAIN; archived Test derived observations only',
                training_or_inference=False, new_raw_Test_read=False,
                Test_label_inference=False, leaderboard_target_recovery=False,
                generator_or_prediction_changes=False,
                regions='Unchanged R15 linear band sum -> dB -> max-observed-median',
                thresholds_db=[3,6,10], additional_HF_window_thresholds_db=[5,15],
                strata=['all','Nobs5','Nobs13plus','Nobs15'],
                label_groups='TRAIN all/known_added/known_zero; Test all only',
                input_sha256=pins,
                use_restriction='Descriptive only; no generator acceptance, calibration, weights or submissions selected using these comparisons')
    write(out/'PLAN.json',plan)
    # Independent arithmetic fixtures: common gain invariance and missing break.
    fixture = np.zeros((1,15,25),float)
    # Avoid exact threshold boundaries in this floating-point invariance test.
    fixture[:,6:8,:] = 9
    a0 = regions(fixture); a1 = regions(fixture+37.)
    if not np.isclose(a0['HF3162_7943']['bump'][0],9):
        raise AssertionError('Regional bump fixture')
    for r in a0:
        for k in a0[r]:
            if not np.allclose(a0[r][k],a1[r][k],equal_nan=True,atol=1e-10):
                raise AssertionError('Common gain invariance')
    seq = np.array([[0.]*11+[10.,np.nan,10.,0.]])
    if temporal(seq)['active_gt6_max_contiguous_run'][0] != 1:
        raise AssertionError('Missing minute must break run')
    write(out/'NUMERIC_FIXTURE.json',{'status':'passed','gain_invariance_37dB':True,
                                      'regional_linear_sum_bump':True,'missing_break':True})
    rows=[]; bandrows=[]; truth=[]; populations=[]

    def consume(name, f, presence=None):
        nobs=f['Nobs']; n=len(nobs)
        subsets=[('all',np.ones(n,bool))]
        if presence is not None:
            subsets += [('known_added',presence==1),('known_zero',presence==0)]
        for obs,count in zip(*np.unique(nobs,return_counts=True)):
            populations.append(dict(dataset=name,nobs=int(obs),rows=int(count),total=n))
        for stratum, sm in [('all',np.ones(n,bool)),('Nobs5',nobs==5),
                             ('Nobs13plus',nobs>=13),('Nobs15',nobs==15)]:
            for subset,lm in subsets:
                mask=sm&lm
                for metric,v in f.items():
                    if metric=='Nobs' or metric.endswith('__nobs'):continue
                    rows.append(dict(dataset=name,stratum=stratum,label_subset=subset,
                                     population_rows=int(mask.sum()),metric=metric,
                                     **stats(v[mask])))
                for t in (5,15):
                    v=f['HF3162_7943__bump'][mask]
                    rows.append(dict(dataset=name,stratum=stratum,label_subset=subset,
                                     population_rows=int(mask.sum()),
                                     metric=f'HF3162_7943__window_bump_gt{t}_fraction',
                                     **stats((v>t).astype(float))))
        return n

    with np.load(ROOT/TEST,allow_pickle=False) as z:
        test={k:z[k] for k in z.files if k!='source_id'}
    if consume('Test_saved_unlabeled',test)!=1500:raise ValueError('Expected1500')
    with np.load(ROOT/band_test,allow_pickle=False) as z:
        btest=z['band_max_minus_median']
    for j,hz in enumerate(BANDS):
        bandrows.append(dict(dataset='Test_saved_unlabeled',frequency_hz=hz,
                             **stats(btest[:,j])))
    for regime in ('D0','H0'):
        source=manifest['folds']['0']['roles']['train'][regime]
        path=ROOT/source['path']; pins[source['path']]=sha(path)
        if pins[source['path']]!=source['sha256']:raise ValueError('TRAIN SHA differs')
        with np.load(path,allow_pickle=False) as z:
            db=z['db']; y=z['share']; presence=z['presence']; active=z['active_minutes']
            no_source=z['explicit_no_source']; site=z['site']; refs=z['reference_index']
        n=len(db); nobs=np.isfinite(db).all(2).sum(1); added=y>0
        if n!=source['n'] or not np.array_equal(added,presence.astype(bool)):
            raise ValueError('TRAIN target/presence mismatch')
        if not np.array_equal(~added,no_source.astype(bool)):
            raise ValueError('Unexpected implicit zero')
        if np.any(active>nobs) or np.any(active[~added]!=0):
            raise ValueError('Support exceeds observations')
        record=dict(dataset='R16_Fold0_'+regime,n=n,added_count=int(added.sum()),
                    added_fraction=float(added.mean()),share_all=stats(y),
                    share_added=stats(y[added]),active_added_minutes=stats(active[added]),
                    active_added_fraction=stats(active[added]/nobs[added]),
                    added_active_ge12_count=int((active[added]>=12).sum()),
                    added_active_ge12_fraction=float((active[added]>=12).mean()),
                    reference_count=int(len(np.unique(refs[added]))),
                    site_counts={str(s):int(c) for s,c in zip(*np.unique(site,return_counts=True))})
        truth.append(record)
        f={'Nobs':nobs,**flatten(regions(db,chunk=512))}
        consume(record['dataset'],f,presence)
        np.savez_compressed(out/(regime+'_TRAIN_derived.npz'),**f)
        # Same observed-only per-band max minus median as archived Test.
        for j,hz in enumerate(BANDS):
            v=np.nanmax(db[:,:,j],axis=1)-np.nanmedian(db[:,:,j],axis=1)
            bandrows.append(dict(dataset=record['dataset'],frequency_hz=hz,**stats(v)))
        print(json.dumps({'completed':regime,'n':n,'seconds':time.time()-started}),flush=True)
        del db,f
    pd.DataFrame(rows).to_csv(out/'observed_metrics.csv',index=False)
    pd.DataFrame(bandrows).to_csv(out/'band25_bumps.csv',index=False)
    pd.DataFrame(populations).to_csv(out/'observation_counts.csv',index=False)
    write(out/'known_TRAIN_truth.json',truth)
    for path,h in pins.items():
        if sha(ROOT/path)!=h:raise ValueError('Input changed during audit:'+path)
    result=dict(status='complete_descriptive_only',seconds=time.time()-started,
                input_sha256=pins,script_sha256=sha(__file__),known_TRAIN_truth=truth,
                all_sources_unchanged=True,new_training=False,new_inference=False,
                raw_Test_read=False,Test_labels_available=False,policy_changed=False,
                limitations=[
                    'Observed HF bump/activity is not added vessel strength/frequency/support.',
                    'Test actual added fraction/share is unknown; no label reconstruction.',
                    'Per-window temporal median removes persistent additions; activity counts cannot measure true full-support duration.',
                    'No calibrated pressure equivalence between external WAV and released SA data.',
                    'Site, masks and reused backgrounds confound marginal descriptive comparisons; no causal claim.',
                    'Do not tune generators, thresholds, weights or predictions using this audit.'],
                output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    write(out/'RESULTS.json',result)
    print(json.dumps({'status':result['status'],'seconds':result['seconds'],'output':str(out)}))


if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='All-NaN slice encountered')
        main()
