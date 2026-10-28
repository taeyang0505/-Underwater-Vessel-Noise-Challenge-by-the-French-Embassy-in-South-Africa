"""Additional pre-evaluation HF control preserving site/sensor/mask counts.

This does not alter the original eight arms or their frozen selection rule.
It fits only TRAIN and validation_fit, never any gate or challenge panel.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='data/synthetic/r12_mask_matched_control_v2')
    p.add_argument('--output',default='models/r12_mask_matched_control_r2');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    from ops.infer_r9_fold0_all5_pilot import environment
    environment(a.threads)
    import numpy as np
    from pipeline.training_r12 import fit,sha,write
    from pipeline.dataset_r12 import load_r12_arrays,validate_r12_bundle
    from pipeline.features_r12 import load_spectral_basis
    from ops.build_r12_mask_matched_control_v2 import load_control
    from ops.train_r12_lightgbm import CompactPhysicsLGBM
    data=ROOT/a.data;train,manifest=load_control(data);source=ROOT/'data/synthetic/r12_cpu_ablation_v3'
    original=validate_r12_bundle(source);basispath=ROOT/'models/r12_shared_basis_v1/basis.npz'
    basis=load_spectral_basis(basispath,expected_training_binding=original['basis_training_binding'])
    val=load_r12_arrays(source,'L0_H0','validation_fit');subset=np.load(source/'cnn_subset_indices.npy',allow_pickle=False)
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False)
    binding={'scope':'additional metadata-matched HF control; not part of original8recipe selector',
             'control_manifest_sha256':sha(data/'manifest.json'),'original_data_hash':original['data_hash'],
             'original_data_manifest_sha256':sha(source/'manifest.json'),'basis_sha256':sha(basispath),'script_sha256':sha(__file__)}
    write(out/'PRE_FIT_FREEZE.json',binding)
    try:
        arm={'name':'L1HM1P1','long':True,'HF':True,'shape':True,'legacy_control':False}
        nn=fit({k:v[subset] for k,v in train.items()},val,basis,arm,out/'cnn',threads=a.threads,batch=128)
        tree=CompactPhysicsLGBM(include_shape=True,basis=basis,basis_path=basispath,basis_sha256=sha(basispath),basis_training_binding=original['basis_training_binding'],settings={'threads':a.threads})
        summary=tree.fit(train,val,out/'tree',binding)
        if sha(data/'manifest.json')!=binding['control_manifest_sha256'] or sha(__file__)!=binding['script_sha256']:raise ValueError('Extra control inputs changed during fit')
        write(out/'ALL_CONTROL_FROZEN.json',{'status':'completed','epochs_run':nn['epochs_run'],'original8policy_changed':False,
            'binding':binding,'model_summary_sha256':{str((out/k/'summary.json').relative_to(ROOT)):sha(out/k/'summary.json') for k in ['cnn','tree']},
            'CNN_checkpoint_sha256':sha(out/'cnn/best.pt'),'tree_booster_sha256':{k:sha(out/'tree'/k) for k in ['share.txt','presence.txt']}})
    except BaseException as e:
        write(out/'FAILED.json',{'error':repr(e),'partial_reuse':False});raise
if __name__=='__main__':main()
