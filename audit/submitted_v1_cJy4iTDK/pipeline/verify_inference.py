from pathlib import Path
import json,tempfile,hashlib
import numpy as np
import pandas as pd
import torch
from pipeline.infer import predict
ROOT=Path(__file__).resolve().parents[1]
def main():
    torch.set_num_threads(2);run=ROOT/'models/fullsources_cpu_a6d7f3a5_seed42'
    policy=json.loads((run/'postprocess.json').read_text())
    d=np.load(ROOT/'data/synthetic/fullsources_cpu_a6d7f3a5_seed42/validation.npz')['db'][:3]
    batch=predict(d,run,policy,torch.device('cpu'))
    singles=[predict(d[i:i+1],run,policy,torch.device('cpu')) for i in range(len(d))]
    for j in [0,1]:assert np.allclose(batch[j],np.concatenate([s[j] for s in singles]),atol=2e-6)
    sub={**policy,'member_run_names':[run.name,run.name]}
    multi=predict(d,run,sub,torch.device('cpu'))
    assert np.allclose(batch[0],multi[0],atol=2e-6)
    out=pd.read_csv(ROOT/'outputs/submission_cpu_candidate.csv')
    sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv')
    assert len(out)==1500 and out.ID.equals(sample.ID) and list(out)==list(sample)
    assert out.iloc[:,1:].apply(lambda c:np.isfinite(c)&c.between(0,1)).all().all()
    manifest=json.loads((ROOT/'outputs/submission_cpu_candidate.manifest.json').read_text())
    assert hashlib.sha256((run/'postprocess.json').read_bytes()).hexdigest()==manifest['frozen_policy_sha256']
    result={'per_window_batch_independence':'passed','identical_member_ensemble_equivalence':'passed',
        'submission_ID_order_columns_bounds':'passed','frozen_policy_unchanged':'passed','submitted':False}
    (ROOT/'verification_inference.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
