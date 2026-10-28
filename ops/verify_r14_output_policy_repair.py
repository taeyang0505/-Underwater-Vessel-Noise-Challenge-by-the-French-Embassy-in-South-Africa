"""Synthetic-only regression fixture for the R2 output policy alias repair."""
from pathlib import Path
import copy,hashlib,json,sys,tempfile,datetime,ast
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    import numpy as np
    from ops import run_r14_production_outputs_v2 as driver
    from pipeline.ensemble_r14_production import fit_policy,apply_policy,token
    rng=np.random.default_rng(202610271402)
    y=np.r_[np.zeros(20),rng.uniform(.01,.8,80)]
    shares={'tree_baseline':np.clip(y+.04,0,1),'nn_bg_cnn':np.clip(y-.03,0,1)}
    clfs={'baseline':np.clip(.1+.7*(y>0)+rng.normal(0,.1,len(y)),0,1),
          'highshare':np.clip(.15+.6*(y>0)+rng.normal(0,.1,len(y)),0,1)}
    clfs['mean']=.5*(clfs['baseline']+clfs['highshare'])
    rejected=[]
    def reject(name,fn):
        try:fn()
        except (ValueError,FileExistsError,PermissionError):rejected.append(name);return
        raise AssertionError('Invalid case accepted:'+name)
    def fit(binding):return fit_policy(y,shares,clfs,y,shares,clfs,binding=binding,fixture=True)
    def apply(p):return apply_policy(p,shares,clfs,allow_fixture=True)
    with tempfile.TemporaryDirectory(prefix='r14-output-policy-repair-') as td:
        base=Path(td);driver_root=driver.ROOT
        try:
            driver.ROOT=base
            source=base/'frozen_source.txt';source.write_text('synthetic source fixture\n')
            pins={'frozen_source.txt':sha(source)}
            old=fit({'source_pins':pins,'role':{'name':'synthetic fit/gate'}})
            original_prediction=apply(old)
            # Exact historical failure: appending the policy file pin changes
            # binding by alias after the policy's canonical hash was frozen.
            oldfile=base/'old_policy.json';driver.write(oldfile,old)
            pins['old_policy.json']=sha(oldfile)
            reject('historical_alias_reproduces_policy_content_changed',lambda:apply(old))
            assert token({k:v for k,v in driver.read(oldfile).items() if k!='policy_sha256'})==driver.read(oldfile)['policy_sha256']
            pins={'frozen_source.txt':sha(source)}
            binding={'source_pins':pins,'role':{'name':'synthetic fit/gate'}}
            detached=driver.detached_binding(binding)
            fixed=fit(detached);fixed_token=fixed['policy_sha256']
            saved,execution=driver.freeze_policy_snapshot(base/'fixed_policy.json',fixed,pins)
            assert fixed==saved and fixed['policy_sha256']==fixed_token
            assert 'fixed_policy.json' not in saved['binding']['source_pins']
            assert execution['fixed_policy.json']==sha(base/'fixed_policy.json')
            driver.verify_pins(execution)
            # Mutations to every caller-owned structure and the new execution
            # map cannot reach the deserialized frozen policy.
            pins['future_checkpoint']='outside';binding['role']['name']='changed'
            detached['role']['name']='changed after freeze';fixed['binding']['role']['name']='changed after freeze'
            execution['future_policy']='outside'
            assert saved['policy_sha256']==fixed_token
            fixed_prediction=apply(saved)
            assert all(np.array_equal(original_prediction[k],fixed_prediction[k]) for k in ('share','score'))
            assert saved==driver.read(base/'fixed_policy.json')
            reject('existing_policy_file',lambda:driver.freeze_policy_snapshot(base/'fixed_policy.json',saved,{'frozen_source.txt':sha(source)}))
            invalid=copy.deepcopy(saved);invalid['regression_weights'][0]+=.01
            reject('changed_policy_before_freeze',lambda:driver.freeze_policy_snapshot(base/'bad.json',invalid,{'frozen_source.txt':sha(source)}))
            reject('mismatched_source_map',lambda:driver.freeze_policy_snapshot(base/'badmap.json',saved,{}))
            file_pins={'frozen_source.txt':sha(source),'fixed_policy.json':sha(base/'fixed_policy.json')}
            source.write_text('changed\n')
            reject('changed_execution_source',lambda:driver.verify_pins(file_pins))
            source.write_text('synthetic source fixture\n')
            with (base/'fixed_policy.json').open('a') as f:f.write(' ')
            reject('changed_frozen_policy_bytes',lambda:driver.verify_pins(file_pins))
            stage=driver.reserve_scope('fast3')
            assert stage==base/driver.OUT/'hybrid_fast3'
            reject('partial_R2_scope_reuse',lambda:driver.reserve_scope('fast3'))
            reject('unknown_scope',lambda:driver.reserve_scope('tree-only'))
            # full4 is a separate fresh namespace; its existing CSV blocks any
            # generation before source arrays or models are queried.
            collision=base/'submissions/r14_main_r2/submission_r14_step1_with_tfgrid.csv'
            collision.parent.mkdir(parents=True);collision.write_text('occupied\n')
            reject('existing_R2_submission',lambda:driver.reserve_scope('full4'))
        finally:driver.ROOT=driver_root
    # Compare all fitted decisions independently of binding/provenance.
    legacy=ROOT/driver.LEGACY_FAST3_POLICY
    assert sha(legacy)==driver.LEGACY_FAST3_POLICY_SHA
    old=driver.read(legacy);same=copy.deepcopy(old);same['binding']={'different':'provenance only'}
    same.pop('policy_sha256');same['policy_sha256']=token(same)
    driver.verify_legacy_decisions(same,'fast3')
    changed=copy.deepcopy(same);changed['score_policy']['lambda']=.25 if changed['score_policy']['lambda']!=.25 else .5
    reject('changed_original_fit_gate_decision',lambda:driver.verify_legacy_decisions(changed,'fast3'))
    protected=driver.read(ROOT/'reports/R14_OUTPUT_V2_PROTECTED_BEFORE.json')['sha256']
    after={p:sha(ROOT/p) for p in protected}
    assert after==protected
    source=Path(driver.__file__).read_text();ast.parse(source);ast.parse(Path(__file__).read_text())
    assert 'def tree_only(' not in source
    assert "choices=['hybrid-fast3','hybrid-full4']" in source
    assert "pp=root/'policy.json';policy,pins=freeze_policy_snapshot(pp,policy,pins)" in source
    report={'status':'passed','scope':'synthetic policy-only fixture; no actual model or query inference',
      'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (Path(driver.__file__),Path(__file__),ROOT/'pipeline/ensemble_r14_production.py')},
      'historical_alias_reproduced':True,'frozen_disk_policy_originally_valid':True,
      'caller_binding_and_execution_pins_detached':True,'freeze_disk_readback_valid':True,
      'numerical_prediction_max_abs_difference':{k:float(np.max(np.abs(original_prediction[k]-fixed_prediction[k]))) for k in ('share','score')},
      'legacy_fast3_decision_guard':True,'rejected_cases':rejected,
      'protected_R1_and_completed_fast3_artifact_count':len(protected),
      'protected_original_artifacts_unchanged':after==protected,
      'protected_before_file_sha256':sha(ROOT/'reports/R14_OUTPUT_V2_PROTECTED_BEFORE.json'),
      'real_data_arrays_opened':False,'Test_or_development_panels_opened':False,
      'model_load_or_fit_performed':False,'real_hybrid_output_executed':False,
      'output_paths':{'run':driver.OUT,'reports':driver.REPORT,'exact_fold0_CSV_dir':'submissions/R14_r2','bagged_Tree_CSV_dir':'submissions/r14_main_r2'},
      'existing_Tree_only_generation_supported':False}
    out=ROOT/'reports/R14_PRODUCTION_OUTPUTS_R2_REPAIR_FIXTURE.json'
    if out.exists():raise FileExistsError(out)
    driver.write(out,report)
    print(json.dumps({'status':'passed','report':str(out.relative_to(ROOT)),'report_sha256':sha(out),'rejected_cases':len(rejected),'prediction_difference':report['numerical_prediction_max_abs_difference']}))

if __name__=='__main__':main()
