"""Read-only replay against stored OWN-validation predictions, never Test.

Writes only new audit artifacts. Does not change model weights, calibrators,
context or ensemble policies and does not use a leaderboard score as a target.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
import torch
from pipeline.infer import predict
from pipeline.models import make_model

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4)
    run=ROOT/'models/ensemble_foundation';base=ROOT/'models/fullsources_cpu_a6d7f3a5_seed42'
    policy=json.loads((run/'postprocess.json').read_text())
    policy_hash=sha(run/'postprocess.json')
    before_files=['pipeline/infer.py','pipeline/predict_foundation.py','pipeline/predict_lgbm.py',
                  'pipeline/models.py','pipeline/mixtures.py','pipeline/train.py',
                  'pipeline/train_tabicl.py','pipeline/train_kumo.py']
    code_hashes={p:sha(ROOT/p) for p in before_files}
    submission_manifest=json.loads((ROOT/'outputs/submission_foundation_candidate.manifest.json').read_text())
    recorded_hashes=json.loads((ROOT/'outputs/submission_foundation_reproducibility_hashes.json').read_text())
    frozen=json.loads((run/'ensemble_policy_freeze.json').read_text())
    assert policy_hash==submission_manifest['frozen_policy_sha256']==frozen['files']['postprocess.json']
    assert sha(ROOT/'outputs/submission_foundation_candidate.csv')=='3b61240ef270412afa0329cce623e55d8e9267a39a2516530469627ac2dfb630'
    assert all(recorded_hashes[p]==h for p,h in code_hashes.items())
    # Diverse fixed OWN-validation positions. No choice is informed by Test values or score.
    ids=np.array([0,1,17,301,640,1200,2099,2999])
    data=dict(np.load(ROOT/'data/synthetic/fullsources_cpu_a6d7f3a5_seed42/validation.npz'))
    stored=dict(np.load(run/'validation_predictions.npz'))
    assert np.array_equal(data['share'],stored['share_true'])
    weights=np.array(policy['weights']);a,b=policy['affine']
    expected_raw=np.stack([stored[k+'_share'][ids] for k in policy['models']],1)@weights
    expected_share=np.clip(a*expected_raw+b,0,1)
    expected_score=np.clip(np.stack([stored[k+'_score'][ids] for k in policy['models']],1)@weights,0,1)
    start=time.perf_counter()
    replay_share,replay_score=predict(data['db'][ids],run,policy,torch.device('cpu'))
    errors={'share_max_abs_error':float(np.max(np.abs(expected_share-replay_share))),
            'score_max_abs_error':float(np.max(np.abs(expected_score-replay_score)))}
    assert np.allclose(expected_share,replay_share,atol=4e-5)
    assert np.allclose(expected_score,replay_score,atol=4e-5)
    training=[]
    for kind in ['tabm','cnn','fdy','itransformer']:
        saved=torch.load(base/(kind+'.pt'),map_location='cpu',weights_only=True)
        torch.manual_seed(42);initial=make_model(kind,width=saved['width'],k=saved['k']).state_dict()
        changed=0;total=0;delta2=0.;initial2=0.
        for key,value in saved['state_dict'].items():
            if value.is_floating_point():
                x=initial[key];total+=1;changed+=not torch.equal(value,x)
                delta2+=float((value-x).square().sum());initial2+=float(x.square().sum())
        hist=json.loads((base/(kind+'_history.json')).read_text())
        training.append({'model':kind,'floating_state_tensors':total,'tensors_different_from_seeded_initialization':changed,
                         'parameter_L2_change_relative_to_initialization':float(np.sqrt(delta2/max(initial2,1e-30))),
                         'epochs_recorded':len(hist),'first_validation_RMSE':hist[0]['validation_rmse'],
                         'best_validation_RMSE':min(h['validation_rmse'] for h in hist),
                         'last_validation_RMSE':hist[-1]['validation_rmse']})
        assert changed>0
    for p,h in code_hashes.items():assert sha(ROOT/p)==h
    assert sha(run/'postprocess.json')==policy_hash
    info={'status':'frozen 9-model inference replay PASSED on own validation',
          'challenge_Test_values_read':False,'sealed_NOAA_read':False,
          'submission_bytes_hashed_only':True,'policy_sha256':policy_hash,
          'submission_csv_sha256':'3b61240ef270412afa0329cce623e55d8e9267a39a2516530469627ac2dfb630',
          'own_validation_indices':ids.tolist(),'elapsed_seconds':time.perf_counter()-start,
          'replay_errors':errors,'small_neural_training_evidence':training,
          'code_matches_submission_reproducibility_ledger':True,'code_sha256':code_hashes,
          'limits':'This verifies execution consistency with stored local predictions, not the validity of synthetic distribution, a matched challenge data-generating process, or training convergence.'}
    (ROOT/'reports/frozen_inference_audit.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2),flush=True)

if __name__=='__main__':main()
