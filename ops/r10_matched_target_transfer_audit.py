"""Development follow-up: preserve each baseline physical share while changing R.

Predeclared after the initial factorial run, not a blind confirmatory holdout.
No fitting. A scalar scale-only change is mathematically identical at fixed y,
so test the combined trajectory+EQ change, not a redundant scale-only panel.
"""
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from ops.r10_factorial_transfer_audit import render,write,infer_panel
from ops.infer_r9_fold0_all5_pilot import inputs,local,environment,sha,verify_pins,DEFAULT_TREE,DEFAULT_BUNDLE,DEFAULT_CONFIG
from pipeline.dataset_r9 import R9Sources
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--threads',type=int,default=4);a=parser.parse_args();environment(a.threads)
    old=ROOT/'reports/r10_diagnostics/factorial_transfer_r2';out=ROOT/'reports/r10_diagnostics/matched_target_transfer_r1';out.mkdir(exist_ok=False)
    start=time.perf_counter();plan={'version':'matched_target_transfer_v1','script_sha256':sha(__file__),
        'purpose':'separate changed reference morphology from target-scale distribution change',
        'development_followup_after_initial_factorial':True,'n_per_domain':1000,
        'source':'same paired source rows, schedule and native R9 references as factorial_transfer_r2',
        'baseline_target':'float64 physical share reconstructed from r9_complete; compare stored float32 target',
        'comparison':'combined_stress reference shape, alpha solved to EXACT same per-window share',
        'interpretation':'scale-only alpha variation disappears at fixed y; complete observations/explicit20%zero remain common',
        'no_Test_report_training_GPU_weight_fitting':True}
    write(out/'predeclared_plan.json',plan)
    nn=json.loads((ROOT/'ops/r9_fold0_pilot_nn_map.json').read_text());pins,modelpins,binding,tree=inputs(nn,ROOT/DEFAULT_TREE,ROOT/DEFAULT_BUNDLE,ROOT/DEFAULT_CONFIG)
    policy=ROOT/'submissions/r9_fold0_all5_cpu_r1/frozen_policy.json';pins[str(policy.relative_to(ROOT))]=sha(policy);weights=json.loads(policy.read_text())['weights']
    sources=R9Sources();schedule=json.loads((old/'schedule.json').read_text());results={};audit=[]
    for domain in ['seen','unseen']:
        frame=pd.read_parquet(old/(domain+'_source_manifest.parquet'));records=[];target0=[]
        baseline=np.load(old/(domain+'__r9_complete')/'samples.npz',allow_pickle=False)
        for i,(_,row) in enumerate(frame.iterrows()):
            spec=schedule[i];bg=sources.background(row);ref=sources.reference_minutes[spec['reference_index']]
            original=render(bg,ref,spec,'r9_complete')
            if abs(float(original[2])-float(baseline['share'][i]))>1e-8:raise AssertionError('Baseline target replay')
            if not np.array_equal(original[0],baseline['db'][i]):raise AssertionError('Baseline input replay')
            # The target is stored float32; using it causes <3e-8 physical shift,
            # explicitly bounded and common to all compared members.
            target=float(baseline['share'][i]);z=render(bg,ref,spec,'combined_stress',target=target)
            if float(z[2])!=target:raise AssertionError('Matched physical target')
            records.append(z);target0.append(target)
        db=np.stack([r[0] for r in records]);y=np.asarray(target0,dtype=np.float32)
        results[domain],_=infer_panel(domain,db,y,{'generator':'combined_stress_at_same_physical_share',
                'recipes':[r[3] for r in records],'base_window_index':frame.base_window_index.astype(int).tolist()},
                nn,tree,binding,weights,out,a.threads)
        original=json.loads((old/(domain+'__r9_complete')/'metrics.json').read_text())
        audit.append({'domain':domain,'original':original,'matched':results[domain],
                      'max_target_delta':float(np.max(abs(y-baseline['share'])))})
    sources.assert_unchanged();verify_pins(pins)
    write(out/'RESULTS.json',{'status':'completed','seconds':time.perf_counter()-start,'comparisons':audit,
          'plan_sha256':sha(out/'predeclared_plan.json'),'production_model_policy_source_pins_unchanged':True})
    print(json.dumps({'completed':str(out),'seconds':time.perf_counter()-start}),flush=True)
if __name__=='__main__':main()
