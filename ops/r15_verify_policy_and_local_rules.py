"""R15 independent pause/policy/local-rule checks. No models or query input."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/r15_validation_checks_r1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process-observation', required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT/'POLICY_PAUSE_RULES_VERIFICATION.json'
    if result_path.exists():
        raise FileExistsError('Preserve existing verification')
    names = [
        'ops/r15_training_pause_receipt.json',
        'models/r14_production_outputs_r2/hybrid_fast3/policy.json',
        'pipeline/ensemble_r14_production.py', 'pipeline/models_r14_production.py',
        'ops/run_r14_production_outputs_v2.py',
        'models/r14_production_nn_fold0_r1/tfgrid_fdy_conformer/epochs.jsonl',
        'scratch/discussions/thread_bodies.json', 'TASK_EDA_REPORT.md',
    ]
    before = {n:sha(ROOT/n) for n in names}
    pause = json.loads((ROOT/names[0]).read_text())
    process_path = Path(args.process_observation)
    observation = json.loads(process_path.read_text())
    assert pause['PID'] == observation['PID'] == 80906
    assert 'T' in observation['STAT'] and 'ops.train_r14_production_nn train' in observation['command']
    checkpoints = []
    for entry in pause['checkpoints']:
        p = ROOT/entry['path']
        assert sha(p) == entry['sha256'] and p.stat().st_size == entry['bytes']
        checkpoints.append(dict(entry, matches_pause_receipt=True))
    history = [json.loads(x) for x in (ROOT/names[5]).read_text().splitlines() if x]
    best = min(history, key=lambda x:x['selection']['rmse'])
    assert history[-1]['epoch'] == pause['last_complete_epoch'] == 37
    assert best['epoch'] == pause['best_epoch'] == 37
    assert best['selection']['rmse'] == pause['best_selection_rmse']
    policy = json.loads((ROOT/names[1]).read_text())
    assert policy['score_policy']['lambda'] == .25 and policy['score_policy']['classifier'] == 'highshare'

    # Exact public function, synthetic vectors only. No policy fitting call.
    from pipeline.ensemble_r14_production import apply_policy
    shares = {n:np.asarray([.05+.08*i, .9-.07*i]) for i,n in enumerate(policy['member_order'])}
    classifiers = {'baseline':np.asarray([.1,.9]), 'highshare':np.asarray([.8,.2])}
    classifiers['mean'] = .5*(classifiers['baseline']+classifiers['highshare'])
    result = apply_policy(policy, shares, classifiers)
    manual_share = np.column_stack(list(shares.values())) @ np.asarray(policy['regression_weights'])
    manual_score = .75*manual_share+.25*classifiers['highshare']
    assert np.array_equal(result['share'], manual_share.astype(np.float32))
    assert np.array_equal(result['score'], manual_score.astype(np.float32))
    assert not np.array_equal(result['score'], result['share'])

    # Same observed E admits different latent B,A under the energy equations.
    # This is an identifiability counterexample, not an estimate of Test y.
    time = np.arange(15)[:,None]
    band = np.arange(25)[None,:]
    energy = (1+.2*time)*(1+.05*band)
    decompositions = []
    for fraction in [0., .08, .4, .8]:
        added = fraction*energy
        background = energy-added
        share = float(added.sum()/(background+added).sum())
        assert np.allclose(background+added, energy, atol=1e-14, rtol=0)
        assert abs(share-fraction) < 1e-14
        decompositions.append({'known_toy_share':fraction,'energy_reconstruction_max_error':float(np.max(np.abs(background+added-energy)))})

    archived = json.loads((ROOT/names[6]).read_text())
    strict = next(x for x in archived if x['id']==35368)
    scaling = next(x for x in archived if x['id']==35370)
    assert strict['complete'] and 'Zindi Staff' in strict['body']
    assert 'excludes tuning generators against test distributions' in strict['body']
    assert 'even for diagnostics' in strict['body']
    assert 'calibrate it using your own training/validation data' in scaling['body']
    searches = []
    for p in [Path('/AGENTS.md'),Path('/Users/AGENTS.md'),Path('/Users/hongtaeyang/AGENTS.md'),
              Path('/Users/hongtaeyang/Documents/AGENTS.md'),Path('/Users/hongtaeyang/Documents/ChatGPT/AGENTS.md'),
              ROOT.parent/'AGENTS.md', ROOT/'AGENTS.md']:
        searches.append({'path':str(p),'exists':p.exists()})
    assert before == {n:sha(ROOT/n) for n in names}
    report = {
        'status':'PASS', 'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Pause integrity, source/policy arithmetic using synthetic vectors, archived-rule review, mathematical counterexample; no model training/inference or raw Test read',
        'process_observation':observation, 'process_observation_sha256':sha(process_path),
        'checkpoints':checkpoints,'last_complete_epoch':37,'active_incomplete_epoch_at_pause':38,
        'best_epoch':37,'best_selection_rmse':best['selection']['rmse'],
        'final_score_formula':'clip(0.75*unrounded_convex_regression_share + 0.25*Tree_highshare_classifier,0,1), then float32',
        'main_submission_classifier':'mean of five Tree-highshare classifier vectors; source lines294-301',
        'NN_score_definition':'score=share; no independent NN classifier head',
        'raw_R14_classifier_persistence':'not saved; absence of saved raw vectors is not absence from computation',
        'synthetic_policy_fixture':{'share':result['share'].tolist(),'score':result['score'].tolist(),'exact_manual_equivalence':True,'score_differs_from_share':True},
        'counterexample':{'same_observed_energy_different_latent_share':decompositions,
            'limitation':'Demonstrates non-identifiability under nonnegative energy equations; not proof every decomposition belongs to the official generator/reference corpus.'},
        'archived_rules':[{'id':strict['id'],'url':strict['url'],'title':strict['title'],'staff':'daniil_zindi','timestamp':'20 Oct2026 22:05',
            'finding':'Explicit prohibition of tuning generators against Test distributions and fitting AV classifiers on Test rows, even for diagnostics'},
            {'id':scaling['id'],'url':scaling['url'],'title':scaling['title'],'staff':'daniil_zindi','timestamp':'20 Oct2026 22:06',
             'finding':'Global scaling must be calibrated on own training/validation; Test inference-only; leaderboard hidden-target recovery prohibited'}],
        'descriptive_audit_permission':'No blanket positive authorization for standalone Test descriptive EDA found in these local archived staff replies; do not equate descriptive inspection with permission to fit generator parameters.',
        'applicable_AGENTS_search':searches,'source_sha256':before,'sources_unchanged':True,
        'new_model_fit':False,'new_model_inference':False,'new_Test_input_or_labels_read':False,'production_changed':False,
    }
    result_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','result':str(result_path.relative_to(ROOT)),'sha256':sha(result_path)}))


if __name__ == '__main__':
    main()
