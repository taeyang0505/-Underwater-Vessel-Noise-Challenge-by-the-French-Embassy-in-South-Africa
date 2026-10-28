"""Explicit R4 roles for the R5 runner; legacy validation metadata is OUTER.

Metadata is never a query feature. Site fields are TRAIN weighting or held-out
diagnostic slices, not estimated Test identities or source-matching inputs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pipeline.grid_inputs_r5 import binding_from_fold, file_sha256, canonical_sha256

ROOT=Path(__file__).resolve().parents[1]
ROLE_FILES={'train':('train.npz','train_sample_metadata.parquet'),
            'inner':('validation.npz','inner_validation_metadata.parquet'),
            'outer':('outer_validation.npz','outer_validation_metadata.parquet')}


class FoldDataR5:
    def __init__(self,folder,seed=42,*,project_root=ROOT):
        self.root=Path(project_root).resolve();self.folder=Path(folder)
        if not self.folder.is_absolute():self.folder=self.root/self.folder
        self.folder=self.folder.resolve();self.binding=binding_from_fold(self.folder,seed,project_root=self.root)
        self.gate=json.loads((self.folder/'readiness_manifest.json').read_text())
        for name in ['source_checks_passed','reference_group_and_duplicate_isolation_passed',
                     'background_week_and_purge_isolation_passed','sealed_holdout_locked']:
            if self.gate.get(name) is not True:raise ValueError('Required source-isolation gate failed: '+name)
        for name in ['challenge_test_accessed','sealed_arrays_accessed','calibration_accessed',
                     'model_evaluated_on_sealed','frozen_v1_data_modified']:
            if self.gate.get(name) is not False:raise ValueError('Unsafe or missing source gate: '+name)
        if self.gate.get('validation_role')!='inner_early_stopping' or self.gate.get('outer_validation_role')!='outer_oof_reporting_only':
            raise ValueError('Explicit nested INNER/OUTER role declarations required')
        self.opened_roles=[]

    def load(self,role,*,frozen_policy=None):
        if role not in ROLE_FILES:raise ValueError('Only explicit train/inner/outer roles allowed; no Test')
        if role=='outer':
            if not isinstance(frozen_policy,dict) or frozen_policy.get('fit_split')!='inner' or frozen_policy.get('frozen_before_outer') is not True:
                raise ValueError('Freeze INNER policy before opening OUTER arrays')
            content={k:v for k,v in frozen_policy.items() if k!='policy_sha256'}
            if frozen_policy.get('policy_sha256')!=canonical_sha256(content):raise ValueError('Frozen ensemble policy SHA changed')
            if frozen_policy.get('training_binding')!=self.binding:raise ValueError('Policy belongs to a different fold/TRAIN')
        npz_name,meta_name=ROLE_FILES[role]
        for name in [npz_name,meta_name]:
            path=self.folder/name
            if path.is_symlink() or file_sha256(path)!=self.gate['file_sha256'].get(name):raise ValueError('Split SHA mismatch: '+name)
        with np.load(self.folder/npz_name,allow_pickle=False) as z:
            arrays={k:z[k] for k in ['db','share','presence','cell_share']}
        metadata=pd.read_parquet(self.folder/meta_name).reset_index(drop=True)
        n=len(arrays['db']);x=arrays['db'];q=arrays['cell_share'];y=arrays['share'];z=arrays['presence']
        if x.shape!=(n,15,25) or q.shape!=x.shape or y.shape!=(n,) or z.shape!=(n,) or len(metadata)!=n:raise ValueError('Role shape/count mismatch')
        if np.isinf(x).any() or not np.isfinite(q).all() or not np.isfinite(y).all() or not np.isfinite(z).all():raise ValueError('Invalid role values')
        obs=np.isfinite(x)
        if (q[~obs]!=0).any() or (q<0).any() or (q>1).any() or not (obs.any(2).sum(1)>=3).all():raise ValueError('Invalid observed-cell labels')
        if not np.array_equal(z,(y>0).astype('float32')):raise ValueError('Addition presence mismatch')
        peak=np.nanmax(x,axis=(1,2),keepdims=True)
        e=np.where(obs,10**((x.astype('float64')-peak)/10),0)
        recovered=(q*e).sum((1,2))/e.sum((1,2))
        if np.max(np.abs(y-recovered))>3e-6:raise ValueError('Exact controlled-energy identity failed')
        expected_role={'train':'train','inner':'validation','outer':'outer_validation'}[role]
        prefix=f"passage_r4_f{self.binding['fold']:02d}_{expected_role}_"
        if not metadata.sample_id.astype(str).str.startswith(prefix).all() or metadata.sample_id.duplicated().any():raise ValueError('Metadata role/sample order mismatch')
        if not np.array_equal(metadata.validation_id.to_numpy(),np.arange(n)):raise ValueError('Metadata row index mismatch')
        if not np.array_equal(metadata.is_outer_oof.to_numpy(),np.full(n,role=='outer')):raise ValueError('Metadata OUTER identity mismatch')
        if not np.allclose(metadata.effective_share.to_numpy(),y,atol=1e-7,rtol=0):raise ValueError('Metadata and target row order differ')
        if not metadata.fold.eq(self.binding['fold']).all():raise ValueError('Metadata fold mismatch')
        self.opened_roles.append(role)
        return arrays,metadata


def fit_site_balance(train_sites,*,fit_split='train',maximum_ratio=4.):
    """Optional bounded inverse-TRAIN-frequency loss weights; no resampling.

    All clean backgrounds still occur once per epoch. This changes the training
    risk and is an explicit candidate, never an estimate of Test site mixture.
    """
    if fit_split!='train':raise ValueError('Site-loss weights may only fit TRAIN metadata')
    g=np.asarray(train_sites).astype(str)
    if g.ndim!=1 or not len(g) or not 1<=maximum_ratio<=10:raise ValueError('Invalid TRAIN site balance request')
    names,counts=np.unique(g,return_counts=True)
    raw=np.minimum(len(g)/(len(names)*counts),float(maximum_ratio));normalizer=float(np.dot(raw,counts)/len(g))
    values=raw/normalizer
    return {'version':'train_site_loss_weights_r5_v1','fit_split':'train',
            'maximum_unnormalized_inverse_frequency_weight':float(maximum_ratio),
            'train_counts':dict(zip(names.tolist(),counts.astype(int).tolist())),
            'weights':dict(zip(names.tolist(),values.tolist())),
            'test_site_proportions_used':False,'samples_resampled':False}


def site_loss_weights(train_sites,state):
    if state.get('fit_split')!='train' or state.get('version')!='train_site_loss_weights_r5_v1':raise ValueError('A frozen TRAIN site-weight state is required')
    sites=np.asarray(train_sites).astype(str)
    if sites.ndim!=1 or set(sites)-set(state['weights']):raise ValueError('Unknown TRAIN site in weighting request')
    return np.asarray([state['weights'][s] for s in sites],dtype='float32')
