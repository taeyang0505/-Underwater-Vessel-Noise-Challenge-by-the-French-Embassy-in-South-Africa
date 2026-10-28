"""Mean model predictions across seeds sharing identical frozen evaluation examples."""
from pathlib import Path
import argparse,json
import numpy as np
def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    manifests=[json.loads((p/'run_manifest.json').read_text()) for p in a.runs]
    hashes={m['external_manifest']['source_hash'] for m in manifests}
    assert len(hashes)==1,'Mismatching source snapshots'
    for split in ['validation','calibration','domain_holdout']:
        data=[dict(np.load(p/f'{split}_predictions.npz')) for p in a.runs]
        assert all(set(d)==set(data[0]) for d in data)
        for key in ['share_true','presence_true']:
            assert all(np.array_equal(d[key],data[0][key]) for d in data),'Evaluation seeds/examples must be identical'
        out={key:(data[0][key] if key.endswith('_true') else np.mean([d[key] for d in data],0)) for key in data[0]}
        np.savez_compressed(a.output/f'{split}_predictions.npz',**out)
    (a.output/'members.json').write_text(json.dumps({'member_run_names':[p.name for p in a.runs],
        'source_hash':next(iter(hashes)),'same_evaluation_examples_verified':True},indent=2))
    print('Merged',len(a.runs),'seeds with identical evaluation examples')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('runs',type=Path,nargs='+');main(p.parse_args())
