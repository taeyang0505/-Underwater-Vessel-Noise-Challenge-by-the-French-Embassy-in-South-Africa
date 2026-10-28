"""Fresh serial R12 CPU CNN fits; physical8arm+one legacy loss control."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--basis',required=True)
    p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=4);p.add_argument('--batch',type=int,default=128)
    p.add_argument('--train-samples',type=int,default=6000)
    a=p.parse_args()
    from ops.infer_r9_fold0_all5_pilot import environment
    environment(a.threads)
    import numpy as np
    from pipeline.features_r12 import load_spectral_basis
    from pipeline.dataset_r12 import load_r12_arrays,validate_r12_bundle
    from pipeline.training_r12 import fit,write,sha
    data=ROOT/a.data;manifest=validate_r12_bundle(data)
    basis=load_spectral_basis(ROOT/a.basis,expected_training_binding=manifest['basis_training_binding'])
    control=load_r12_arrays(data,'L0_H0','train')
    selected=np.load(data/'cnn_subset_indices.npy',allow_pickle=False)
    if len(selected)!=a.train_samples or len(np.unique(selected))!=len(selected) or int((control['share'][selected]==0).sum())!=a.train_samples//5:raise ValueError('Frozen common balanced CNN subset mismatch')
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'training_subset.npz',row_indices=selected,source_id=control['source_id'][selected],sample_id=control['sample_id'][selected])
    arms=[{'name':f'L{l}H{h}P{s}','long':bool(l),'HF':bool(h),'shape':bool(s),'legacy_control':False} for l in [0,1] for h in [0,1] for s in [0,1]]
    arms=[{'name':'legacy_contrast_control','long':False,'HF':False,'shape':False,'legacy_control':True}]+arms
    freeze={'data_manifest_sha256':sha(data/'manifest.json'),'basis_sha256':sha(ROOT/a.basis),'arms':arms,
            'training_code_sha256':sha(ROOT/'pipeline/training_r12.py'),'script_sha256':sha(__file__),'fit_val_only':True,
            'train_examples_per_arm':len(selected),'same_indexed_subset_all_arms':True,
            'training_subset_sha256':sha(out/'training_subset.npz'),
            'source_code_sha256':{name:sha(ROOT/name) for name in ['pipeline/features_r12.py','pipeline/dataset_r12.py','pipeline/relative_models_r6_v2.py','pipeline/relative_grid_r6_v2.py','pipeline/relative_grid_r6.py']}}
    write(out/'PRE_FIT_FREEZE.json',freeze)
    val=load_r12_arrays(data,'L0_H0','validation_fit');summaries={}
    try:
        for arm in arms:
            name=f"L{int(arm['long'])}_H{int(arm['HF'])}";train=load_r12_arrays(data,name,'train')
            if not np.array_equal(control['share']==0,train['share']==0):raise ValueError('Factorial arms changed explicit zero positions')
            train={k:v[selected] for k,v in train.items()}
            print(json.dumps({'start_arm':arm,'train_rows':len(train['share']),'train_arm':name}),flush=True)
            result=fit(train,val,basis,arm,out/arm['name'],threads=a.threads,batch=a.batch)
            summaries[arm['name']]=result;write(out/'progress.json',{'completed':list(summaries),'last':result,'total_arms':len(arms)})
        for name,h in freeze['source_code_sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Frozen training source changed:'+name)
        if sha(data/'manifest.json')!=freeze['data_manifest_sha256']:raise ValueError('Data changed during training')
        if sha(ROOT/a.basis)!=freeze['basis_sha256'] or sha(ROOT/'pipeline/training_r12.py')!=freeze['training_code_sha256'] or sha(__file__)!=freeze['script_sha256']:raise ValueError('Frozen model/basis/driver changed during training')
        write(out/'ALL_CNN_FROZEN.json',{'all_models_frozen':True,'status':'completed','summary_sha256':{k:sha(out/k/'summary.json') for k in summaries},'data_manifest_sha256':freeze['data_manifest_sha256']})
    except BaseException as e:
        write(out/'FAILED.json',{'type':type(e).__name__,'error':str(e),'resume_or_partial_reuse':False});raise
if __name__=='__main__':main()
