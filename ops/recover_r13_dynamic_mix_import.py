"""Preserve a pre-tree-fit missing-import failure, with no learner changes.

The original paired-CNN process/source/plan remain untouched. A fresh output
receives the exact plan, a newly fitted tree, and (only after100 epochs) an
exact completed CNN copy. The sole correction imports Mapping from stdlib.
This is executor repair, not a second recipe/split/seed/hyperparameter trial.
"""
from pathlib import Path
from collections.abc import Mapping
import argparse,json,shutil,time
from ops import train_r13_dynamic_mix as base

ROOT=base.ROOT
SOURCE=ROOT/'models/r13_dynamic_mix_r1'
DEST=ROOT/'models/r13_dynamic_mix_r2'

def tree():
    base.imports(2)
    plan=base.load_plan(SOURCE)
    log=(SOURCE/'tree.log').read_text()
    if "NameError: name 'Mapping' is not defined" not in log or (SOURCE/'tree').exists():
        raise RuntimeError('Only documented failure before tree folder/booster fit may be repaired')
    if DEST.exists():raise FileExistsError('Fresh repaired output required')
    DEST.mkdir()
    for n in ('PRE_FIT_FREEZE.json','epoch_choices.npz','NUMERIC_FIXTURE.json'):shutil.copyfile(SOURCE/n,DEST/n)
    receipt={'format':'r13_pre_fit_missing_import_executor_correction_v1','created_unix':time.time(),
        'original_run':'models/r13_dynamic_mix_r1','corrected_run':'models/r13_dynamic_mix_r2',
        'original_source_sha256':base.sha(ROOT/'ops/train_r13_dynamic_mix.py'),'repair_source_sha256':base.sha(__file__),
        'original_tree_failure_log_sha256':base.sha(SOURCE/'tree.log'),'original_tree_booster_fit_started':False,
        'missing_import':'collections.abc.Mapping','all_learner_data_seed_hyperparameters_unchanged':True,
        'original_plan_sha256':plan['plan_sha256'],'CNN_continues_original_process_without_restart':True,
        'old_failure_logs_preserved':True,'parameter_or_recipe_search':False}
    base.write(DEST/'EXECUTOR_CORRECTION_BEFORE_TREE_FIT.json',receipt)
    # Supply exactly the omitted standard-library name; no operator substitution.
    base.Mapping=Mapping
    base.fit_tree(DEST)

def finish(evaluate=False):
    base.imports(2)
    base.load_plan(DEST)
    receipt=json.loads((DEST/'EXECUTOR_CORRECTION_BEFORE_TREE_FIT.json').read_text())
    if receipt['repair_source_sha256']!=base.sha(__file__) or receipt['original_source_sha256']!=base.sha(ROOT/'ops/train_r13_dynamic_mix.py'):
        raise ValueError('Repair/source changed')
    c=json.loads((SOURCE/'cnn/summary.json').read_text())
    if c.get('status')!='completed' or c['epochs_run']!=100 or c['early_stopping'] or base.sha(SOURCE/'cnn/best.pt')!=c['checkpoint_sha256']:
        raise ValueError('Only complete unmodified CNN100 may be copied; no partial state reuse')
    if (DEST/'cnn').exists():raise FileExistsError('CNN import is one-shot')
    shutil.copytree(SOURCE/'cnn',DEST/'cnn')
    cnn_hash={p.name:base.sha(p) for p in (SOURCE/'cnn').iterdir() if p.is_file()}
    if any(base.sha(DEST/'cnn'/n)!=h for n,h in cnn_hash.items()):raise ValueError('CNN byte copy differs')
    base.Mapping=Mapping
    frozen=base.freeze(DEST)
    proof={'format':'r13_single_condition_corrected_executor_completion_v1','status':'completed',
        'original_parent_failure':str(SOURCE/'FAILED.json'),
        'original_parent_failure_sha256':base.sha(SOURCE/'FAILED.json') if (SOURCE/'FAILED.json').exists() else None,
        'correction_receipt_sha256':base.sha(DEST/'EXECUTOR_CORRECTION_BEFORE_TREE_FIT.json'),
        'repair_source_sha256':base.sha(__file__),'completed_original_CNN_copied_byte_exact':cnn_hash,
        'CNN_new_fits':1,'tree_actual_booster_fits':1,'recipe_conditions':1,
        'all2_freeze_sha256':frozen['freeze_sha256'],'no_partial_CNN_checkpoint_reused':True}
    base.write(DEST/'CORRECTED_EXECUTOR_COMPLETION.json',proof)
    if evaluate:
        report=ROOT/base.REPORT
        base.evaluate(DEST,report)
        result=json.loads((report/'RESULTS.json').read_text())
        result['executor_correction_provenance']={'completion_path':str((DEST/'CORRECTED_EXECUTOR_COMPLETION.json').relative_to(ROOT)),
            'completion_sha256':base.sha(DEST/'CORRECTED_EXECUTOR_COMPLETION.json'),
            'note':'Original tree failed before any boosting because Mapping import was absent; original CNN100 copied only after completion; one actual tree fit.'}
        base.write(report/'RESULTS.json',result)
    print(json.dumps({'status':'completed','output':str(DEST),'evaluation':evaluate}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['tree','finish','finish-evaluate'])
    a=p.parse_args()
    if a.action=='tree':tree()
    else:finish(a.action=='finish-evaluate')
