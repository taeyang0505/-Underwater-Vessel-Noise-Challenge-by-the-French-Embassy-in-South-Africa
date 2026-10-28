"""Check R15 duration tables from saved derived vectors, never raw Test/model."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/r15_score_duration_r3'
OUT=ROOT/'reports/r15_validation_checks_r1/SCORE_DURATION_SAVED_TABLE_QA.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def rank(x):
    _,inv,cnt=np.unique(x,return_inverse=True,return_counts=True)
    return (np.cumsum(cnt)-.5*(cnt-1))[inv]


def main():
    if OUT.exists():raise FileExistsError('Keep previous QA')
    result=json.loads((SOURCE/'RESULTS.json').read_text())
    assert result['status']=='passed_descriptive_only' and result['TRAIN_and_SA_rows_each']==78869
    assert result['original_SA_scope_is_only_Fold0TRAIN_not_all_public_rows']
    paths=[SOURCE/'RESULTS.json']+[SOURCE/p for p in result['output_sha256']]
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for p,h in result['output_sha256'].items():assert sha(SOURCE/p)==h
    qs=rows(SOURCE/'dataset_bump_duration_quantiles.csv');quantile_errors=[];quantile_assertions=0
    test=None
    for dataset in result['datasets']:
        d=load(SOURCE/(dataset+'_derived.npz'));n=len(d['source_id']);assert len(set(d['source_id'].astype(str)))==n
        assert n==(1500 if dataset=='Test' else 78869)
        masks={'all':np.ones(n,bool),'Nobs5':d['Nobs']==5,'Nobs13plus':d['Nobs']>=13,'Nobs15':d['Nobs']==15}
        for r in [v for v in qs if v['dataset']==dataset]:
            mask=masks[r['stratum']];x=d[r['region']+'__'+r['metric']][mask];x=x[np.isfinite(x)]
            assert int(r['population_rows'])==n and int(r['stratum_rows'])==int(mask.sum()) and int(r['n_valid'])==len(x)
            fields=[k for k in r if k.startswith('p') and k[1:].isdigit()]
            if not len(x):assert all(not r[k] for k in fields);continue
            values=np.quantile(x,[int(k[1:])/100 for k in fields])
            for k,value in zip(fields,values):
                e=abs(float(r[k])-value);assert e<2e-12,(dataset,r['metric'],k,e)
                quantile_errors.append(e);quantile_assertions+=1
        for region in ['HF3162_7943','LF31_200','LF_historical31_501']:
            nobs=d[region+'__nobs'];assert ((nobs>=1)&(nobs<=15)).all()
            for threshold in [3,6,10]:
                count=d[f'{region}__active_gt{threshold}_count'];fraction=d[f'{region}__active_gt{threshold}_fraction'];run=d[f'{region}__active_gt{threshold}_max_contiguous_run']
                assert ((0<=run)&(run<=count)&(count<=nobs)).all()
                assert np.max(np.abs(fraction-count/nobs))<1e-14
        if dataset=='Test':test=d
    hf=test['HF3162_7943__bump']>15;groups=rows(SOURCE/'Test_LF_HF_groups.csv')
    for r in groups:
        region=r['LF_definition'];lf=test[region+'__bump'];co=test[region+'__corr_with_HF']
        conditions={'HF_gt15':hf,'HF_gt15_AND_LF_le3':hf&(lf<=3),
            'HF_gt15_AND_LF_gt6_corr_gt05':hf&(lf>6)&(co>.5),'LF_le3_all':lf<=3}
        mask=conditions[r['group']]
        assert int(r['n'])==int(mask.sum()) and int(r['HFgt15_denominator'])==int(hf.sum())==801
        assert abs(float(r['fraction_of_all1500'])-mask.mean())<1e-14
        assert abs(float(r['fraction_of_HFgt15_intersection'])-mask[hf].mean())<1e-14
    joined=rows(SOURCE/'Test_joined_duration1500.csv')
    assert [r['ID'] for r in joined]==test['source_id'].astype(str).tolist()
    sf=np.asarray([r['probe_site']=='07 St Francis' for r in joined]);assert sf.sum()==487
    sfq=rows(SOURCE/'SF_vs_nonSF_prediction_quantiles.csv');prediction_errors=[]
    for r in sfq:
        mask=sf if r['estimated_group_not_GT']=='estimated_SF487' else ~sf
        x=np.asarray([float(v[r['prediction']]) for v in joined])[mask]
        assert int(r['n'])==int(r['n_valid'])==len(x)
        expected={'mean':float(x.mean()),'RMS':float(np.sqrt(np.mean(x*x)))}
        for k in r:
            if k.startswith('p') and k[1:].isdigit():expected[k]=float(np.quantile(x,int(k[1:])/100))
        for k,v in expected.items():
            e=abs(v-float(r[k]));assert e<2e-12;prediction_errors.append(e)
    rankrows=rows(SOURCE/'SF_vs_nonSF_rankprob_KS.csv');rank_errors=[]
    for r in rankrows:
        x=np.asarray([float(v[r['prediction']]) for v in joined]);a=x[sf];b=x[~sf]
        # Independent explicit pairwise oracle, including half credit for ties.
        probability=float(((a[:,None]>b[None,:])+.5*(a[:,None]==b[None,:])).mean())
        grid=np.unique(x);ks=float(np.max(np.abs(np.searchsorted(np.sort(a),grid,'right')/len(a)-np.searchsorted(np.sort(b),grid,'right')/len(b))))
        for k,v in [('P_SF_greater_plus_half_ties',probability),('P_nonSF_greater_plus_half_ties',1-probability),('empirical_KS_D',ks)]:
            e=abs(float(r[k])-v);assert e<2e-14;rank_errors.append(e)
        assert r['label_AUC_claimed']=='False' and int(r['n_SF'])==487 and int(r['n_nonSF'])==1013
    corrrows=rows(SOURCE/'feature_prediction_correlations.csv');corr_errors=[]
    for r in corrrows:
        a=test[r['feature']];b=np.asarray([float(v[r['prediction']]) for v in joined]);valid=np.isfinite(a)&np.isfinite(b);a=a[valid];b=b[valid]
        assert int(r['n'])==len(a)
        for k,v in [('Pearson',np.corrcoef(a,b)[0,1]),('Spearman',np.corrcoef(rank(a),rank(b))[0,1])]:
            e=abs(float(r[k])-v);assert e<2e-12;corr_errors.append(e)
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths}
    report={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Saved derived-feature and saved-prediction table reaggregation only; no raw Test/background bank, model fit/inference, hidden labels or production edits',
        'datasets':result['datasets'],'TRAIN_SA_scope':'Fold0TRAIN78869 only, not all1715779 released rows',
        'quantile_rows':len(qs),'quantile_assertions':quantile_assertions,'quantile_max_error':max(quantile_errors),
        'prediction_distribution_rows':len(sfq),'prediction_distribution_max_error':max(prediction_errors),
        'group_definition_rows':len(groups),'HF_gt15_count':801,'LF31_200_near_zero_within_HF_count':120,
        'LF31_200_joint_rise_within_HF_count':364,'rankprob_KS_rows':len(rankrows),'pairwise_oracle_max_error':max(rank_errors),
        'correlation_rows':len(corrrows),'correlation_max_error':max(corr_errors),
        'duration_count_fraction_run_bounds_verified':True,'estimated_SF_count':487,'estimated_site_not_ground_truth':True,
        'label_AUC_not_claimed':True,'input_and_output_hashes_unchanged':True,'input_sha256':before,
        'coverage_limits':['Does not recompute original acoustics-to-feature transform; consumes frozen saved derived vectors.',
            'No Test-derived acceptance thresholds or generator selection; descriptive reaggregation only.',
            'Activity counts/longest runs are observed level excursions, not true passage duration.',
            'Rank probability/KS between inferred site groups is not diagnostic label AUC.']}
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'PASS','sha256':sha(OUT),'report':str(OUT.relative_to(ROOT)),'quantile_assertions':quantile_assertions}))


if __name__=='__main__':main()
