"""Additional immutable clean-fit variant excluding every old local_fit row.

TRAIN/earlystop stay unchanged while active refits run.  Only this new fit
is authoritative for legacy-model comparisons without old-fit source reuse.
"""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops import r17_clean_data as clean


def main():
    from pipeline.dataset_r9 import R9Sources
    from pipeline.background_cv_r9 import proposed_purge_gap
    from pipeline.r16_production_data import _crossrole
    from pipeline.r16_multisupport_data import guard
    sys.addaudithook(guard)
    oldout = clean.OUT
    clean.OUT = ROOT/'data/synthetic/r17_clean_freshfit_v1'
    clean.REPORT = ROOT/'reports/r17_clean_pipeline_r1/data/fresh_fit'
    if clean.OUT.exists():
        raise FileExistsError('Immutable fresh-fit version exists')
    clean.OUT.mkdir(parents=True);clean.REPORT.mkdir(parents=True,exist_ok=True)
    cv = pd.read_parquet(ROOT/'data/synthetic/r16_production_v1/fold_00/cv_validation_source_pool.parquet')
    cv = cv[cv.qc_observed_minutes.ge(12)].copy().reset_index(drop=True)
    mapping={}
    for site, frame in cv.groupby('site',sort=True):
        groups=sorted(frame.background_group.unique(),key=lambda g:hashlib.sha256(('r17clean'+str(g)).encode()).hexdigest())
        mapping.update({g:'selection' if i%2==0 else'ensemble_fit' for i,g in enumerate(groups)})
    cv['proposed_role']=cv.background_group.map(mapping)
    cv['r17_cross_role_support_gap_hours']=proposed_purge_gap(cv)/3600.
    pool=cv[cv.proposed_role.eq('ensemble_fit')&cv.r17_cross_role_support_gap_hours.ge(24)].copy()
    oldfit=pd.read_parquet(ROOT/'reports/r16_support_temporal_r1/data/local_fit_backgrounds.parquet')
    pool=pool[~pool.base_window_index.isin(oldfit.base_window_index)].copy()
    chosen=clean.balanced_sites(pool,400,'fresh_ensemblefit_source')
    checks={n:pd.read_parquet(oldout/f'{n}_backgrounds.parquet') for n in ['train','earlystop']}
    checks.update({n:pd.read_parquet(ROOT/f'reports/r16_support_temporal_r1/data/{n}_backgrounds.parquet') for n in ['local_gate','local_report','domain_holdout']})
    checks['legacy_local_fit']=oldfit
    proof=_crossrole(chosen,checks)
    chosen.to_parquet(clean.OUT/'ensemble_fit_fresh_backgrounds.parquet',index=False)
    chosen.groupby(['site','qc_observed_minutes']).size().rename('unique_backgrounds').reset_index().to_csv(clean.REPORT/'background_site_density_composition.csv',index=False)
    source=R9Sources(verify_banks=False)
    refs=pd.read_parquet(oldout/'reference_pool_465.parquet')
    protocol={'version':'r17_clean_freshfit_v1','seed':clean.SEED,'changes_only':'ensemble_fit; TRAIN/earlystop R17 clean v1 unchanged',
              'generator_code_sha256':clean.sha(ROOT/'ops/r17_clean_data.py'),'wrapper_code_sha256':clean.sha(Path(__file__)),
              'base_manifest_sha256':clean.sha(oldout/'manifest.json'),'all_old_local_fit_source_overlap':0,
              'all_old_local_fit_group_overlap':0,'source_pool':'ownCV designated ensemble_fit groups only; all old shared local_fit excluded',
              'source_purge_checks':proof,'Test_used':False,'reference_reservations_excluded':44,
              'reference_independence_claimed':False,'nobs_min':12,'unique_backgrounds':400,'rows_per_regime':2000}
    clean.write(clean.OUT/'PROTOCOL_BEFORE_GENERATION.json',protocol)
    outputs={reg:clean.materialize(source,chosen,refs,'ensemble_fit_fresh',reg) for reg in ['D0','H0']}
    receipt={**protocol,'status':'completed','roles':{'ensemble_fit_fresh':outputs}}
    clean.write(clean.OUT/'manifest.json',receipt);clean.write(clean.REPORT/'FRESH_FIT_RECEIPT.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
