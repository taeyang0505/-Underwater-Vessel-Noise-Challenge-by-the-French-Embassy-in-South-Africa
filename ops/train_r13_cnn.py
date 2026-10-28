"""Three fresh fixed100 CNN fits and one strictly verified completed control.

No Test, gate, heldout or other developmental panel is read by this runner.
The same completed R12 strict control is reusable because its training arrays,
subset, basis, seed, architecture, objective and validation_fit are identical.
"""
from pathlib import Path
import sys,argparse,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/synthetic/r13_cpu_correction_v1')
    p.add_argument('--output',default='models/r13_cpu_cnn_r1')
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    from ops.infer_r9_fold0_all5_pilot import environment
    environment(a.threads)
    import numpy as np
    from pipeline.training_r12 import sha,write,fit
    from pipeline.dataset_r13 import load_r13_arrays,validate_r13_bundle
    from pipeline.dataset_r12 import load_r12_arrays,validate_r12_bundle
    from pipeline.features_r12 import load_spectral_basis
    from ops.build_r12_mask_matched_control_v2 import load_control
    data=ROOT/a.data;manifest=validate_r13_bundle(data)
    original=ROOT/'data/synthetic/r12_cpu_ablation_v3';parent=validate_r12_bundle(original)
    basispath=ROOT/'models/r12_shared_basis_v1/basis.npz'
    basis=load_spectral_basis(basispath,expected_training_binding=parent['basis_training_binding'])
    subset=np.load(original/'cnn_subset_indices.npy',allow_pickle=False)
    val=load_r12_arrays(original,'L0_H0','validation_fit')
    baseline=load_r13_arrays(data,'baseline')
    prior,pm=load_control(ROOT/'data/synthetic/r12_mask_matched_control_v2')
    if set(baseline)!=set(prior) or any(not np.array_equal(baseline[k],prior[k],equal_nan=True) if baseline[k].dtype.kind in 'fc' else not np.array_equal(baseline[k],prior[k]) for k in baseline):
        raise ValueError('Baseline arrays must exactly match completed control')
    control=ROOT/'models/r12_mask_matched_control_r2/cnn'
    cs=json.loads((control/'summary.json').read_text())
    prior_binding=json.loads((control.parent/'PRE_FIT_FREEZE.json').read_text())
    if (prior_binding['control_manifest_sha256']!=sha(ROOT/'data/synthetic/r12_mask_matched_control_v2/manifest.json')
        or prior_binding['original_data_manifest_sha256']!=sha(original/'manifest.json')
        or prior_binding['basis_sha256']!=sha(basispath)):
        raise ValueError('Reused control training source/basis binding changed')
    if (cs['status']!='completed' or cs['epochs_run']!=100 or cs['early_stopping'] or cs['optimizer']!='Adam'
        or cs['parameters']!=26953 or cs['score']!='share' or cs['arm']['shape'] is not True
        or (control.parent/'FAILED.json').exists() or sha(control/'best.pt')!=cs['checkpoint_sha256']):
        raise ValueError('Only fully completed strict control may be reused')
    config=ROOT/'configs/r13_cpu_correction_v1.json'
    source_paths=[Path(__file__),config,ROOT/'pipeline/training_r12.py',ROOT/'pipeline/features_r12.py',ROOT/'pipeline/dataset_r13.py',
        data/'manifest.json',data/'readiness.json',control/'summary.json',control/'best.pt',basispath,original/'cnn_subset_indices.npy']
    pins={str(n.relative_to(ROOT)):sha(n) for n in source_paths}
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False)
    binding={'data_hash':manifest['data_hash'],'source_pins':pins,'control_reuse':'baselineP1 only; no partial run reuse',
        'control_path':str(control.relative_to(ROOT)),'subset_examples':len(subset),'validation_fit_rows':len(val['share']),
        'gate_eval_Test_read':False,'created_unix':time.time()}
    write(out/'PRE_FIT_FREEZE.json',binding)
    summaries={'baselineP1':{'reused_summary_path':str((control/'summary.json').relative_to(ROOT)),
        'reused_summary_sha256':sha(control/'summary.json'),'checkpoint_path':str((control/'best.pt').relative_to(ROOT)),
        'checkpoint_sha256':sha(control/'best.pt'),'training_arrays_exact_same':True,'summary':cs}}
    try:
        for arm,shape in [('baselineP0',False),('highshareP0',False),('highshareP1',True)]:
            arrays=baseline if arm.startswith('baseline') else load_r13_arrays(data,'highshare')
            train={k:v[subset] for k,v in arrays.items()}
            specification={'name':arm,'long':True,'HF':True,'shape':shape,'legacy_control':False,'highshare':arm.startswith('highshare')}
            s=fit(train,val,basis,specification,out/arm,threads=a.threads,batch=128)
            summaries[arm]={'summary_path':str((out/arm/'summary.json').relative_to(ROOT)),'summary_sha256':sha(out/arm/'summary.json'),
                'checkpoint_path':str((out/arm/'best.pt').relative_to(ROOT)),'checkpoint_sha256':sha(out/arm/'best.pt'),'summary':s}
            for n,h in pins.items():
                if sha(ROOT/n)!=h:raise ValueError('Frozen source changed:'+n)
        write(out/'ALL_CNN_FROZEN.json',{'format':'r13_cpu_cnn_frozen_v1','status':'completed','new_fits':3,'completed_control_reused':1,
            'all_models_frozen':True,'models':summaries,'binding':binding,'source_pins':pins,'gate_eval_Test_read':False})
    except BaseException as e:
        write(out/'FAILED.json',{'error':repr(e),'partial_reuse':False});raise

if __name__=='__main__':main()
