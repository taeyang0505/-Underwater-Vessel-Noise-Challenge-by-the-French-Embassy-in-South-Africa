"""R12 db-only relative inputs and TRAIN-A0 spectral-shape residuals.

The first eight channels and all physical weights are EXACTLY the frozen
relative_grid_v2(db) representation. Optional channels 8--10 are signed
rank-3 PCA reconstruction residual/10, its positive part, and signed local
spectral curvature (2*db[f]-db[f-1]-db[f+1])/10 at f=10..21. PCA is fit only
on caller-verified TRAIN released-SA A=0 backgrounds. A=0 means no added
component, NOT natural-vessel absence. Residuals are not source truth.

PCA uses rounded db in fixed safe bands 0..22 (31.6--5011.9 Hz), after
removing each minute's common mean level. Fitting excludes incomplete safe
spectra, never fills missing minutes. Query projection is independently
computed per minute on its observed safe bands; no population fitting,
labels, site/recorder IDs or query-source inference is accepted. Curvature
requires all three neighboring measurements and is zero otherwise.

The added three maps NEVER use bands23/24, including for their centering.
This protection does not delete those bands from the unchanged original8
or their 25-band physical weights. It is not a correction for every possible
recorder response. Common dB gain cancels after quantization; arbitrary
gain before 0.1dB rounding has bounded quantization error, not exact
invariance. Static band offsets generally do NOT cancel in shape residuals.

No Torch, sklearn, fitted scaler, raw q/y, WAV, Test or role-file reader.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
from pipeline.relative_grid_r6 import quantized_db
from pipeline.relative_grid_r6_v2 import relative_grid_v2, FEATURE_NAMES as BASE_FEATURE_NAMES

VERSION='r12_safe_spectral_pca_v1'
SAFE_BANDS=tuple(range(23))
CURVATURE_BANDS=tuple(range(10,22))
SHAPE_FEATURE_NAMES=('train_safe23_rank3_signed_residual_div10',
                     'train_safe23_rank3_positive_residual_div10',
                     'midband_spectral_peak_curvature_div10')
FEATURE_NAMES=BASE_FEATURE_NAMES+SHAPE_FEATURE_NAMES
SOURCE_KIND='released_sa_background'


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _digest(value):return hashlib.sha256(_canonical(value)).hexdigest()


def _array_digest(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(_canonical({'shape':list(a.shape),'dtype':str(a.dtype)})+a.tobytes()).hexdigest()


def _file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def _ids(source_ids,n):
    ids=np.asarray(source_ids)
    if ids.ndim!=1 or len(ids)!=n or ids.dtype.kind not in 'iuUS':
        raise ValueError('One explicit integer/string TRAIN background source ID per window required')
    values=ids.astype(str).tolist()
    if any(not x.strip() for x in values) or len(set(values))!=n:
        raise ValueError('Nonempty unique TRAIN source IDs required; do not duplicate sources for PCA fitting')
    return values


def fit_spectral_basis(db_background,source_ids,*,split='train',source_kind=SOURCE_KIND,
                       n_components=3,seed=42,max_fit_spectra=32768,training_binding=None):
    """Fit a JSON-safe rank3 PCA state on pure released-SA TRAIN backgrounds.

    The caller must establish the TRAIN-source/A0 identity before calling;
    numerical values cannot prove absence of a synthetically added source.
    This assertion and raw input/ID SHA are persisted. Validation, report,
    Test and synthetic-mixture fits are forbidden. No target labels enter.
    All complete safe-band minute spectra are eligible; max_fit_spectra is
    a fixed deterministic uniformly sampled cap, not a quality/error filter.
    """
    if split!='train' or source_kind!=SOURCE_KIND:
        raise ValueError('PCA fit requires released_sa_background in split=train only')
    if type(n_components) is not int or n_components!=3:
        raise ValueError('Predeclared R12 PCA has exactly three components')
    if type(seed) is not int or seed<0 or type(max_fit_spectra) is not int or max_fit_spectra<4:
        raise ValueError('Nonnegative integer seed and cap>=4 required')
    raw=np.asarray(db_background)
    x=quantized_db(db_background)
    ids=_ids(source_ids,len(x))
    binding=json.loads(_canonical({} if training_binding is None else training_binding))
    if not isinstance(binding,dict):raise ValueError('TRAIN provenance binding must be a JSON object')
    safe=x[:,:,:23].reshape(-1,23);valid=np.isfinite(safe).all(1);indices=np.flatnonzero(valid)
    eligible=len(indices)
    if eligible<4:raise ValueError('At least four complete observed safe-band TRAIN minute spectra required')
    if eligible>max_fit_spectra:
        indices=np.sort(np.random.default_rng(seed).choice(indices,max_fit_spectra,replace=False))
    values=safe[indices];values=values-values.mean(1,keepdims=True)
    mean=values.mean(0);centered=values-mean
    _,singular,axes=np.linalg.svd(centered,full_matrices=False)
    tolerance=np.finfo(float).eps*max(centered.shape)*singular[0] if singular[0]>0 else 0.
    rank=int(np.sum(singular>tolerance))
    if rank<3:raise ValueError('TRAIN spectral variation has rank<3; a three-component nuisance basis is unidentifiable')
    components=axes[:3].copy()
    # Resolve SVD sign ambiguity without looking at any validation data.
    for row in components:
        if row[np.argmax(np.abs(row))]<0:row*=-1
    variance=singular*singular/(len(values)-1)
    basis={'version':VERSION,'fit_split':'train','source_kind':SOURCE_KIND,
        'A0_semantics':'no controlled addition; natural vessel absence is NOT asserted',
        'caller_asserted_background_provenance':True,'training_binding':binding,
        'input_shape':list(raw.shape),'input_db_sha256':_array_digest(raw),'source_ids_sha256':_digest(ids),
        'unique_background_source_count':len(ids),'safe_band_indices':list(SAFE_BANDS),
        'safe_band_highest_center_hz':float(10**(1.5+.1*22)),
        'n_components':3,'quantization_decimals':1,
        'centering':'per-minute mean over observed safe bands; PCA mean from TRAIN only',
        'fit_incomplete_safe_spectra':'excluded, never imputed',
        'query_incomplete_safe_spectra':'observed-only least-squares projection; missing residuals zero',
        'eligible_complete_spectra':eligible,'fitted_spectra':len(values),'max_fit_spectra':max_fit_spectra,
        'seed':seed,'selected_flat_spectrum_indices_sha256':_array_digest(indices),
        'sampled_source_windows':int(len(np.unique(indices//15))),'effective_spectral_rank':rank,
        'mean_db':mean.tolist(),'components':components.tolist(),
        'explained_variance':variance[:3].tolist(),
        'explained_variance_ratio':(variance[:3]/variance.sum()).tolist(),
        'fit_residual_root_mean_square_db':float(np.sqrt(np.mean((centered-centered@components.T@components)**2))),
        'feature_channel_names':list(FEATURE_NAMES),'labels_accepted_or_redefined':False,
        'not_source_separation_or_ship_truth':True}
    basis['basis_sha256']=_digest(basis)
    validate_spectral_basis(basis)
    return basis


def validate_spectral_basis(basis,*,expected_training_binding=None):
    """Fail closed on state tampering or a foreign TRAIN binding."""
    if not isinstance(basis,dict):raise TypeError('Explicit fitted TRAIN basis dictionary required')
    b={k:v for k,v in basis.items() if k!='basis_sha256'}
    if basis.get('basis_sha256')!=_digest(b):raise ValueError('Spectral basis content SHA mismatch')
    if (b.get('version')!=VERSION or b.get('fit_split')!='train' or b.get('source_kind')!=SOURCE_KIND
            or b.get('n_components')!=3 or b.get('safe_band_indices')!=list(SAFE_BANDS)
            or b.get('quantization_decimals')!=1 or b.get('caller_asserted_background_provenance') is not True):
        raise ValueError('Foreign/non-TRAIN spectral basis is forbidden')
    mean=np.asarray(b['mean_db'],float);components=np.asarray(b['components'],float)
    if (mean.shape!=(23,) or components.shape!=(3,23) or not np.isfinite(mean).all()
            or not np.isfinite(components).all() or not np.allclose(components@components.T,np.eye(3),atol=1e-9,rtol=0)):
        raise ValueError('Invalid finite orthonormal TRAIN PCA state')
    if expected_training_binding is not None and _canonical(b['training_binding'])!=_canonical(expected_training_binding):
        raise ValueError('Spectral basis TRAIN binding mismatch')
    return basis


def _shape_channels(x,basis):
    validate_spectral_basis(basis)
    mean=np.asarray(basis['mean_db'],float);components=np.asarray(basis['components'],float)
    safe=x[:,:,:23].reshape(-1,23);observed=np.isfinite(safe)
    count=observed.sum(1,keepdims=True)
    center=np.divide(np.where(observed,safe,0).sum(1,keepdims=True),count,
                     out=np.zeros_like(count,dtype=float),where=count>0)
    residual=np.zeros_like(safe)
    # Batch every availability pattern, without cross-window fitting.
    patterns,inverse=np.unique(observed,axis=0,return_inverse=True)
    for j,mask in enumerate(patterns):
        if not mask.any():continue
        rows=np.flatnonzero(inverse==j)
        values=(safe[rows][:,mask]-center[rows])-mean[mask]
        if mask.all():coeff=values@components.T
        else:coeff=values@np.linalg.pinv(components[:,mask].T).T
        errors=values-coeff@components[:,mask]
        residual[np.ix_(rows,np.flatnonzero(mask))]=errors/10.
    signed=np.zeros_like(x);signed[:,:,:23]=residual.reshape(x.shape[0],15,23)
    positive=np.maximum(signed,0)
    curvature=np.zeros_like(x)
    for f in CURVATURE_BANDS:
        good=np.isfinite(x[:,:,f-1:f+2]).all(2)
        values=(2*x[:,:,f]-x[:,:,f-1]-x[:,:,f+1])/10.
        curvature[:,:,f]=np.where(good,values,0)
    return np.stack([signed,positive,curvature],1).astype(np.float32)


def transform_channels(db,basis=None,include_shape=False):
    """db-only -> finite float32 x[N,8 or11,15,25], unchanged rounded w.

    No fitting occurs. include_shape=True requires an already fitted TRAIN
    basis. basis is ignored only when include_shape=False, preserving the
    exact old8 path; no extra source/label/ID parameters exist in this API.
    """
    if type(include_shape) is not bool:raise TypeError('Explicit include_shape bool required')
    base,w=relative_grid_v2(db)
    if not include_shape:return base,w
    if basis is None:raise ValueError('Shape features require a frozen TRAIN spectral basis')
    extra=_shape_channels(quantized_db(db),basis)
    result=np.concatenate([base,extra],1)
    if not np.isfinite(result).all() or not np.array_equal(result[:,5],(w>0).astype(np.float32)):
        raise ValueError('Feature mask/finite invariant failed')
    return result,w


def compact_feature_names(include_shape=False):
    """Names for 466 base or673 shape compact relative tree features."""
    if type(include_shape) is not bool:raise TypeError('Explicit include_shape bool required')
    indices=(0,1,2,3,4,6)+(tuple(range(8,11)) if include_shape else ())
    names=[f'{FEATURE_NAMES[c]}__band{f:02d}__{stat}' for c in indices for f in SAFE_BANDS for stat in ('mean','std','max')]
    names += [f'rounded_energy_fraction_band{f:02d}' for f in SAFE_BANDS]
    names += [f'observed_fraction_band{f:02d}' for f in SAFE_BANDS]
    names += ['observed_cell_fraction','complete_minute_fraction','first_minute_complete','last_minute_complete',
              'rounded_energy_entropy_div_log375','maximum_minute_energy_fraction']
    return tuple(names)


def compact_relative_features(db,basis=None,include_shape=False):
    """Return fixed, label-free finite float32 [N,466 or673] tree features.

    No scaler or query-population fit is necessary. Statistics are unweighted
    over observed minutes per safe band; normalized energy features use the
    unchanged full25-band physical w. Base profile/weights retain their old
    upper-band dependence; only new shape channels have strict safe23 scope.
    """
    x,w=transform_channels(db,basis,include_shape)
    observed=w>0;count=observed[:,:,:23].sum(1)
    cols=[]
    for c in (0,1,2,3,4,6)+(tuple(range(8,11)) if include_shape else ()):
        values=x[:,c,:,:23].astype(float)
        m=np.divide(values.sum(1),count,out=np.zeros_like(count,dtype=float),where=count>0)
        var=np.divide(np.where(observed[:,:,:23],(values-m[:,None,:])**2,0).sum(1),count,
                      out=np.zeros_like(m),where=count>0)
        mx=np.max(np.where(observed[:,:,:23],values,-np.inf),1);mx=np.where(count>0,mx,0)
        cols.append(np.stack([m,np.sqrt(np.maximum(var,0)),mx],2).reshape(len(x),-1))
    columns=cols+[w[:,:,:23].sum(1),observed[:,:,:23].mean(1),
        observed.mean((1,2))[:,None],observed.all(2).mean(1)[:,None],
        observed[:,0].all(1)[:,None],observed[:,-1].all(1)[:,None],
        (-np.sum(np.where(w>0,w*np.log(np.maximum(w,np.finfo(np.float32).tiny)),0),axis=(1,2))/np.log(375))[:,None],
        w.sum(2).max(1)[:,None]]
    result=np.concatenate(columns,1).astype(np.float32)
    if result.shape!=(len(x),len(compact_feature_names(include_shape))) or not np.isfinite(result).all():
        raise ValueError('Compact feature shape/finite invariant failed')
    return result


def save_spectral_basis(basis,path):
    """Write a fresh numeric/Unicode-only NPZ; return file and content SHA."""
    validate_spectral_basis(basis);p=Path(path)
    if p.suffix!='.npz':raise ValueError('Use a fresh .npz basis path')
    metadata={k:v for k,v in basis.items() if k not in ('mean_db','components')}
    with p.open('xb') as f:
        np.savez_compressed(f,mean_db=np.asarray(basis['mean_db'],np.float64),
            components=np.asarray(basis['components'],np.float64),metadata_json=np.asarray(_canonical(metadata).decode()))
    return {'path':str(p),'file_sha256':_file_digest(p),'basis_sha256':basis['basis_sha256']}


def load_spectral_basis(path,*,expected_sha256=None,expected_training_binding=None):
    """Restore without pickle and validate state SHA plus optional file/binding."""
    if expected_sha256 is not None and _file_digest(path)!=expected_sha256:
        raise ValueError('Spectral basis NPZ file SHA mismatch')
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'mean_db','components','metadata_json'}:raise ValueError('Unexpected PCA NPZ fields')
        b=json.loads(z['metadata_json'].item());b['mean_db']=z['mean_db'].astype(float).tolist();b['components']=z['components'].astype(float).tolist()
    return validate_spectral_basis(b,expected_training_binding=expected_training_binding)
