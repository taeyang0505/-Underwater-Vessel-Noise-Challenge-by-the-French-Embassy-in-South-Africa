"""Immutable R13 recipe resolution, separate from provisional base config."""
from pathlib import Path
import json,hashlib


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def token(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def freeze_runtime_policy(config,bundle,cpu_evidence,output,*,data_variant,include_shape,code_sha256,production_approved=False):
    from pipeline.full_data_r13 import validate_full_bundle
    config,bundle,cpu_evidence,output=map(Path,(config,bundle,cpu_evidence,output))
    if output.exists():raise FileExistsError('Runtime selection is immutable; fresh path required')
    m=validate_full_bundle(bundle)
    if m['config_sha256']!=sha(config):raise ValueError('Prepared base config differs')
    if data_variant not in ('baseline','highshare') or type(include_shape)is not bool:raise ValueError('Explicit finite recipe choice required')
    if not cpu_evidence.is_file():raise ValueError('Explicit CPU result/policy evidence file required')
    value={'format':'r13_frozen_runtime_policy_v1','config_file_sha256':sha(config),
        'data_bundle_hash':m['data_hash'],'data_variant':data_variant,'include_shape':include_shape,
        'CPU_evidence_file':str(cpu_evidence),'CPU_evidence_sha256':sha(cpu_evidence),
        'production_recipe_approved':bool(production_approved),'code_sha256':dict(code_sha256),
        'production_block_reason':None if production_approved else 'CPU correction failed simultaneous all-panel improvement; candidate preparation is not launch approval',
        'small_CPU_ensemble_weights_or_score_lambda_transferred':False,
        'highshare_positive_fraction':.25 if data_variant=='highshare' else 0.,
        'policy_scope':'operator-selected finite recipe; does not establish GPU availability, CUDA memory or measured performance',
        'Test_or_Public_used_for_selection':False,'CUDA_training_launched_by_this_action':False}
    value['policy_sha256']=token(value)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return value


def load_runtime_policy(path,config_path,bundle_manifest,expected_code,root,*,require_approved=False):
    p=json.loads(Path(path).read_text())
    if (p.get('format')!='r13_frozen_runtime_policy_v1' or type(p.get('production_recipe_approved')) is not bool
        or p.get('config_file_sha256')!=sha(config_path) or p.get('data_bundle_hash')!=bundle_manifest['data_hash']
        or p.get('data_variant') not in ('baseline','highshare') or type(p.get('include_shape'))is not bool
        or p.get('Test_or_Public_used_for_selection') is not False
        or token({k:v for k,v in p.items() if k!='policy_sha256'})!=p.get('policy_sha256')
        or p.get('code_sha256')!=expected_code):raise ValueError('Missing/changed/foreign runtime recipe policy')
    if require_approved and p['production_recipe_approved'] is not True:raise ValueError('Technical candidate prepared, production launch not approved')
    evidence=Path(p['CPU_evidence_file'])
    if not evidence.is_absolute():evidence=Path(root)/evidence
    if not evidence.is_file() or sha(evidence)!=p.get('CPU_evidence_sha256'):
        raise ValueError('CPU recipe-selection evidence missing/changed')
    for name,h in p['code_sha256'].items():
        if sha(Path(root)/name)!=h:raise ValueError('Code changed after runtime freeze:'+name)
    if p.get('highshare_positive_fraction')!=(.25 if p['data_variant']=='highshare' else 0.):raise ValueError('Runtime highshare variant inconsistent')
    return p
