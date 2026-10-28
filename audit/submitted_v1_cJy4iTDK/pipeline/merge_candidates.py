"""Merge frozen, aligned validation predictions without reading challenge Test.

The original model folders are never modified. Foundation share predictions are
also retained as uncalibrated controlled-addition evidence for the diagnostic
score field; they are not a separately trained presence probability.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPLITS=('validation','calibration','domain_holdout')
BASE='fullsources_cpu_a6d7f3a5_seed42'
FOUNDATION={
    'tabicl':{'model_run_name':'tabicl_cpu_context16000',
              'synthetic_dataset_name':BASE,'context_rows':16000,'members':4},
    'kumo_large':{'model_run_name':'kumo_large_cpu_context4096',
                  'synthetic_dataset_name':BASE,'context_rows':4096,'members':4},
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(a):
    dest=a.output
    if (dest/'postprocess.json').exists():
        raise RuntimeError('Destination already has a frozen policy; use a new output folder')
    dest.mkdir(parents=True,exist_ok=True)
    base=ROOT/'models'/BASE
    data_root=ROOT/'data/synthetic'/BASE
    base_info=json.loads((base/'run_manifest.json').read_text())
    provenance={'base_run_name':BASE,'foundation_specs':FOUNDATION,'splits':{},
        'source_manifest_sha256':sha(ROOT/'data/processed/external_preparation_manifest.json'),
        'source_hash':base_info['external_manifest']['source_hash'],
        'challenge_test_accessed':False,'sealed_NOAA_accessed':False,
        'score_policy':'Foundation score is its bounded share proxy, not calibrated presence probability; ranking uses share RMSE only.',
        'alignment_limit':'All share targets exactly match the source NPZ in order. Historical foundation manifests do not contain independent example IDs; runner code iterates NPZ rows in order.'}
    for split in SPLITS:
        bp=base/f'{split}_predictions.npz';d=dict(np.load(bp))
        source=dict(np.load(data_root/f'{split}.npz'))
        assert np.array_equal(d['share_true'],source['share'])
        assert np.array_equal(d['presence_true'],source['presence'])
        checks={'base_predictions_sha256':sha(bp),'source_data_npz_sha256':sha(data_root/f'{split}.npz'),
                'n':len(d['share_true']),'target_order_sha256':hashlib.sha256(d['share_true'].tobytes()).hexdigest(),
                'foundation_predictions':{}}
        for kind,spec in FOUNDATION.items():
            fp=ROOT/'models'/spec['model_run_name']/f'{split}_predictions.npz'
            f=dict(np.load(fp));key='tabicl_share' if kind=='tabicl' else 'kumo_share'
            assert np.array_equal(f['share_true'],d['share_true']),f'{kind}/{split}: target/order mismatch'
            pred=np.asarray(f[key],dtype=np.float32)
            assert pred.shape==d['share_true'].shape and np.isfinite(pred).all()
            assert ((pred>=0)&(pred<=1)).all()
            d[kind+'_share']=pred;d[kind+'_score']=pred.copy()
            checks['foundation_predictions'][kind]={'file_sha256':sha(fp),'targets_match_exactly':True}
        np.savez_compressed(dest/f'{split}_predictions.npz',**d)
        provenance['splits'][split]=checks
    (dest/'foundation_specs.json').write_text(json.dumps(FOUNDATION,indent=2))
    (dest/'candidate_merge_manifest.json').write_text(json.dumps(provenance,indent=2))
    print('Merged 9 fixed candidates into',dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'models/ensemble_foundation')
    main(p.parse_args())
