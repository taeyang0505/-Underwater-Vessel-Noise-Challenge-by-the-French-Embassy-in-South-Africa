"""One shared TRAIN-fitted prior and db-only eight-channel input for R5.

Synthetic q/y/presence are controlled-addition supervision, not natural ship
probabilities. An offline teacher must produce its own prediction/provenance;
known q or y must NEVER be renamed p_ship. These helpers do not return p_ship
and the query API accepts db only, with a frozen, data-bound aggregate prior.
No query labels, source tags, fitting, or query-population adaptation is used.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from pipeline.background_prior_features import fit_background_prior
from pipeline.mixtures import hybrid_cnn_channels

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='shared_grid8_r5_v1'
SHARED_PRIOR_FILENAME='shared_background_prior.json'
BINDING_KEYS=('train_npz_sha256','data_package_hash','source_hash','seed','fold')
LABEL_POLICY={'known_q_y_are_controlled_addition_labels':True,
    'oracle_q_or_y_as_p_ship_forbidden':True,'query_accepts_labels_or_source_tags':False,
    'teacher_p_ship_requires_model_inference_and_provenance':True}

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _binding(binding):
    if not isinstance(binding,dict) or any(k not in binding for k in BINDING_KEYS):raise ValueError('Explicit TRAIN/package/seed/fold binding required')
    result=dict(binding)
    for k in BINDING_KEYS[:3]:
        value=result[k]
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Invalid SHA binding: '+k)
    for k in ['seed','fold']:
        if isinstance(result[k],(bool,np.bool_)) or not isinstance(result[k],(int,np.integer)):raise ValueError('Integer binding required: '+k)
        result[k]=int(result[k])
    if result['seed']<0:raise ValueError('Nonnegative model seed required')
    if not 0<=result['fold']<5:raise ValueError('Unexpected R5 fold')
    if 'train_examples' in result:
        if int(result['train_examples'])!=result['train_examples'] or result['train_examples']<2:raise ValueError('Invalid training row count binding')
        result['train_examples']=int(result['train_examples'])
    # A JSON round trip also rejects opaque mutable/nonserializable identity.
    return json.loads(json.dumps(result,sort_keys=True,allow_nan=False))

def binding_from_fold(fold_dir,seed,*,project_root=ROOT,verify_train_sha=True):
    """Read TRAIN metadata/hash only; no inner/outer arrays or recipes opened."""
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:raise ValueError('Nonnegative integer seed required')
    folder=Path(fold_dir);folder=folder.resolve() if folder.is_absolute() else (Path(project_root)/folder).resolve()
    gate=json.loads((folder/'readiness_manifest.json').read_text())
    if gate.get('ready_to_train') is not True or gate.get('protocol')!='source_group_nested5_vtuad_passage_r4':raise ValueError('A completed frozen R4 fold is required')
    source=Path(gate['source_version_dir']);source=source if source.is_absolute() else Path(project_root)/source
    if file_sha256(source/'manifest.json')!=gate['source_version_manifest_sha256']:raise ValueError('TRAIN source manifest changed')
    if json.loads((source/'manifest.json').read_text())['source_hash']!=gate['source_hash']:raise ValueError('TRAIN source identity mismatch')
    files=gate['file_sha256']
    if hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()!=gate['data_package_hash']:raise ValueError('Declared package identity changed')
    if verify_train_sha and file_sha256(folder/'train.npz')!=files['train.npz']:raise ValueError('TRAIN NPZ changed')
    return _binding({'train_npz_sha256':files['train.npz'],'data_package_hash':gate['data_package_hash'],
        'source_hash':gate['source_hash'],'seed':int(seed),'fold':gate['fold'],
        'train_examples':gate['partitions']['train']['examples']})

def _array_fingerprint(db,q):
    h=hashlib.sha256()
    for name,a in [('db',db),('cell_share',q)]:
        h.update(json.dumps({'name':name,'shape':a.shape,'dtype':str(a.dtype)},sort_keys=True).encode())
        for start in range(0,len(a),512):h.update(np.ascontiguousarray(a[start:start+512]).tobytes())
    return h.hexdigest()

def fit_shared_prior(train,binding,split='train'):
    """Fit at most 32 aggregate centres once; never fit on held-out labels.

    train must be the verified TRAIN dictionary. With train_examples in the
    binding, a partial array cannot silently masquerade as the full TRAIN.
    The known TRAIN q is used only to estimate B=(1-q)E for this fit.
    """
    if split!='train':raise ValueError('Shared background priors may only fit TRAIN')
    identity=_binding(binding)
    if not isinstance(train,dict) or not {'db','cell_share'}.issubset(train):raise ValueError('Known TRAIN db/cell_share required; no p_ship substitution')
    x=np.asarray(train['db']);q=np.asarray(train['cell_share'])
    if x.ndim!=3 or x.shape[1:]!=(15,25) or q.shape!=x.shape or len(x)<2:raise ValueError('Expected TRAIN [N,15,25] arrays')
    if identity.get('train_examples',len(x))!=len(x):raise ValueError('TRAIN input rows differ from binding')
    if np.isinf(x).any() or not (np.isfinite(x).any(2).sum(1)>=3).all() or not np.isfinite(q).all() or ((q<0)|(q>1)).any() or (q[~np.isfinite(x)]!=0).any():raise ValueError('Invalid known component labels')
    if 'share' in train:
        y=np.asarray(train['share']);peak=np.nanmax(x,(1,2),keepdims=True);e=np.where(np.isfinite(x),10**((x.astype(float)-peak)/10),0)
        recovered=(e*q).sum((1,2))/e.sum((1,2))
        if y.shape!=(len(x),) or not np.isfinite(y).all() or np.max(np.abs(recovered-y))>3e-6:raise ValueError('TRAIN q/y energy identity failed')
        if 'presence' in train and not np.array_equal(np.asarray(train['presence']),(y>0).astype('float32')):raise ValueError('Controlled-addition presence mismatch')
    prior=fit_background_prior(x,q,seed=identity['seed'],clusters=32)
    prior.update({'shared_input_schema':SCHEMA,'fit_split':'train','training_binding':identity,
        'training_arrays_sha256':_array_fingerprint(x,q),'label_policy':dict(LABEL_POLICY),
        'shared_across_all_four_neural_models':True})
    prior['shared_prior_payload_sha256']=canonical_sha256(prior)
    validate_shared_prior(prior,expected_binding=identity)
    return prior

def prior_payload_sha256(prior):
    return canonical_sha256({k:v for k,v in prior.items() if k!='shared_prior_payload_sha256'})

def validate_shared_prior(prior,*,expected_binding,expected_prior_sha256=None):
    if not isinstance(prior,dict) or prior.get('version')!='aggregate_background_prior_v1' or prior.get('shared_input_schema')!=SCHEMA or prior.get('fit_split')!='train':raise ValueError('Unsupported/unbound shared prior')
    identity=_binding(prior.get('training_binding'));expected=_binding(expected_binding)
    for key,value in expected.items():
        if key=='train_examples' and key not in identity:
            if prior.get('training_rows')!=value:raise ValueError('Prior row count differs from expected TRAIN')
        elif identity.get(key)!=value:raise ValueError('Prior belongs to different training data: '+key)
    if 'train_examples' in identity and prior.get('training_rows')!=identity['train_examples']:raise ValueError('Prior fit rows do not match its binding')
    actual=prior_payload_sha256(prior)
    if actual!=prior.get('shared_prior_payload_sha256') or expected_prior_sha256 is not None and actual!=expected_prior_sha256:raise ValueError('Shared prior payload SHA mismatch')
    p=np.asarray(prior.get('centres_power_shape'),dtype=float)
    if p.ndim!=2 or p.shape[1]!=25 or not 1<=len(p)<=32 or not np.isfinite(p).all() or (p<=0).any():raise ValueError('Invalid aggregate prior centres')
    if prior.get('requested_clusters')!=32 or prior.get('individual_training_examples_saved') is not False or prior.get('query_requires_labels_or_source_tags') is not False:raise ValueError('Shared prior policy changed')
    return actual

def shared_grid_inputs(db,prior,*,expected_binding,expected_prior_sha256=None):
    """Query = db only. q/y/presence/p_ship are intentionally absent arguments."""
    if isinstance(db,dict):raise TypeError('Pass query db only, never a labelled training dictionary')
    validate_shared_prior(prior,expected_binding=expected_binding,expected_prior_sha256=expected_prior_sha256)
    x=np.asarray(db)
    if x.ndim!=3 or x.shape[1:]!=(15,25) or len(x)==0 or np.isinf(x).any() or not (np.isfinite(x).any(2).sum(1)>=3).all():raise ValueError('Query must be nonempty observed [N,15,25] db')
    grid,w=hybrid_cnn_channels(x,prior)
    if grid.shape!=(len(x),8,15,25) or w.shape!=x.shape or not np.isfinite(grid).all() or not np.isfinite(w).all():raise ValueError('Invalid common 8-channel transform')
    if (w[~np.isfinite(x)]!=0).any() or (w<0).any() or not np.allclose(w.sum((1,2)),1,atol=1e-6):raise ValueError('Observed energy pooling weights changed')
    return np.ascontiguousarray(grid,dtype='float32'),np.ascontiguousarray(w,dtype='float32')

def save_shared_prior(path,prior,*,expected_binding):
    payload_sha=validate_shared_prior(prior,expected_binding=expected_binding);path=Path(path)
    if path.exists() or path.is_symlink():raise FileExistsError('A fitted shared prior is immutable')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(prior,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return {'filename':path.name,'file_sha256':file_sha256(path),'payload_sha256':payload_sha,
        'training_binding':prior['training_binding'],'shared_input_schema':SCHEMA}

def load_shared_prior(path,*,expected_binding,expected_file_sha256,expected_prior_sha256=None):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or file_sha256(path)!=expected_file_sha256:raise ValueError('Frozen shared prior file SHA mismatch')
    prior=json.loads(path.read_text());validate_shared_prior(prior,expected_binding=expected_binding,expected_prior_sha256=expected_prior_sha256)
    return prior
