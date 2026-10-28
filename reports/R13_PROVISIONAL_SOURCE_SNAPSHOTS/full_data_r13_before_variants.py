"""R13 full background-CV data, separate from the Fold0 CPU pilot.

All primary TRAIN windows occur once per epoch. HF prioritises no-addition
assignment within fixed site/sensor/recorder/effective-mask zero quotas;
this is NOT the small12k outside-primary replacement operation. R9 source
roles/purge are reused, but R13 fixed numerical observations are regenerated.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.dataset_r9 import R9Sources,sha,digest,array_sha,fixed_background_indices
from pipeline.dataset_r12 import hf_bump,reference_audit,synthesize_r12
from pipeline.features_r12 import fit_spectral_basis,save_spectral_basis,load_spectral_basis,transform_channels
from pipeline.source_split_r6 import apply_minute_qc

ROOT=Path(__file__).resolve().parents[1]
VERSION='r13_full_background_cv_data_v1'
ROLES=('selection','ensemble_fit','ensemble_gate','report')
STRATA=('site','sensor_group','recorder','effective_mask_bits')
DATA_CODE=('pipeline/full_data_r13.py','ops/prepare_r13_training.py','pipeline/dataset_r13.py',
           'pipeline/dataset_r12.py','pipeline/dataset_r9.py','pipeline/sampler_r9.py',
           'pipeline/features_r12.py','pipeline/relative_grid_r6.py','pipeline/relative_grid_r6_v2.py')


def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def read_config(path):
    c=json.loads(Path(path).read_text())
    if c.get('version')!='r13_full_training_v1' or c.get('folds')!=list(range(5)):
        raise ValueError('Explicit full5fold R13 config required')
    g=c['generator'];t=c['training']
    if (g['long_positive_fraction']!=.3 or g['native_long_fraction']!=.25
        or g['highshare_positive_fraction']!=.25 or g['highshare_target_uniform']!=[.25,.85]
        or g['explicit_no_source_fraction']!=.2 or g['smooth_eq_max_db']!=6.
        or g['ordinary_peak_snr_uniform']!=[-10.,10.] or g['parameters_from_Test_statistics'] is not False
        or t['epochs']!=100 or t['early_stopping'] is not False or t['presence_loss_weight']!=0.):
        raise ValueError('R13 provisional physical policy differs; revise/freeze explicitly, never silently')
    return c


def observed_bits(db):
    o=np.isfinite(db)
    if np.isinf(db).any() or (o.any(2)!=o.all(2)).any():raise ValueError('Complete measured/NaN minute masks required')
    return (o.all(2).astype(np.int64)*(1<<np.arange(15))).sum(1)


def stratum_keys(frame):
    return [json.dumps([str(r.site),int(r.sensor_group),str(r.recorder),int(r.effective_mask_bits)],separators=(',',':'))
            for r in frame.itertuples(index=False)]


def describe_backgrounds(source,frame):
    f=frame.copy().reset_index(drop=True);bits=[];bump=[]
    for i in range(0,len(f),512):
        q=f.iloc[i:i+512];x=apply_minute_qc(np.asarray(source.background_bank()[q.base_window_index.to_numpy(int)]),q.boundary_mask_bits.to_numpy(np.int64))
        bits.extend(observed_bits(x).tolist());bump.extend(hf_bump(x).tolist())
    f['effective_mask_bits']=bits;f['HF_bump_18_24_db']=bump;f['stratum_key']=stratum_keys(f)
    if not np.array_equal(np.asarray([int(b).bit_count() for b in bits]),f.qc_observed_minutes):raise AssertionError('Actual QC mask differs')
    return f


def train_HF_thresholds(frame):
    t=frame.groupby('stratum_key',sort=True).HF_bump_18_24_db.quantile(.9)
    return {str(k):float(v) for k,v in t.items()}


def apply_HF_thresholds(frame,thresholds):
    f=frame.copy();f['TRAIN_HF_q90_db']=f.stratum_key.map(thresholds)
    f['HF_threshold_available']=f.TRAIN_HF_q90_db.notna()
    f['HF_candidate']=f.HF_threshold_available & f.HF_bump_18_24_db.gt(f.TRAIN_HF_q90_db)
    return f


def zero_count(n,epoch):
    """Integer law: exact20% over5epochs, without dropping/repeating primaries."""
    base,remainder=divmod(int(n),5)
    return base+int((int(epoch)-1)%5<remainder)


def make_full_schedule(metadata,epoch,fold,seed,*,tag=0):
    n=len(metadata);rng=np.random.default_rng(np.random.SeedSequence([seed,fold,epoch,tag,1313]))
    baseline=np.zeros(n,bool);baseline[rng.permutation(n)[:zero_count(n,epoch)]]=True
    selected=np.zeros(n,bool);priority=np.zeros(n,bool);quota=[]
    groups=metadata.groupby('stratum_key',sort=True).indices
    for key,indices in groups.items():
        ids=np.asarray(indices);z=int(baseline[ids].sum());desired=round(.5*z)
        candidates=ids[metadata.iloc[ids].HF_candidate.to_numpy(bool)]
        chosen=rng.permutation(candidates)[:min(desired,len(candidates))]
        selected[chosen]=True;priority[chosen]=True
        remaining=ids[~np.isin(ids,chosen)];extra=rng.permutation(remaining)[:z-len(chosen)];selected[extra]=True
        if int(selected[ids].sum())!=z:raise AssertionError('Stratum zero quota changed')
        quota.append({'stratum_key':str(key),'source_rows':len(ids),'baseline_zero_quota':z,
            'HF_priority_desired':desired,'HF_priority_actual':len(chosen),'shortfall':desired-len(chosen)})
    order=rng.permutation(n);zero=selected[order];prior=priority[order]
    positives=np.flatnonzero(~zero);refs=np.full(n,-1,np.int64)
    refs[positives]=np.resize(rng.permutation(509),len(positives))
    family=rng.choice(np.asarray(['native_inside','native_short','native_edge']),n,p=[.5,.25,.25]).astype('U32');family[zero]='no_source'
    long=rng.permutation(positives)[:round(.3*len(positives))];native_n=round(.25*len(long))
    family[long[:native_n]]='long_native';family[long[native_n:]]='long_smooth_bell'
    # Reference pools are injected by the caller after its audited native lengths.
    high=np.zeros(n,bool);is_long=np.char.startswith(family,'long_')
    desired_high=round(.25*len(positives));high_long=round(.25*len(long))
    high[rng.permutation(np.flatnonzero(is_long))[:high_long]]=True
    high[rng.permutation(np.flatnonzero((~zero)&(~is_long)))[:desired_high-high_long]]=True
    target=np.full(n,np.nan);target[high]=rng.uniform(.25,.85,int(high.sum()))
    out={'primary_order':order,'zero':zero,'HF_priority_zero':prior,'reference_index':refs,'family':family,
         'highshare':high,'highshare_target':target,'item_seed':rng.integers(0,2**63-1,n,dtype=np.int64)}
    out['audit']={'epoch':epoch,'examples':n,'zero_count':int(zero.sum()),'zero_fraction':float(zero.mean()),
        'exact20percent_this_epoch':int(zero.sum())*5==n,'exact20percent_every5and100epochs':True,
        'primary_once':bool(np.array_equal(np.sort(order),np.arange(n))),
        'zero_count_policy':'floor/ceil predeclared5epoch cycle',
        'stratum_zero_quota_exact':True,'HF_priority_actual':int(prior.sum()),
        'HF_priority_shortfall':sum(q['shortfall'] for q in quota),
        'HF_assignment_is_not_small12k_replacement':True,'quota_rows':quota,
        'highshare_positive_count':int(high.sum()),'long_positive_count':len(long),
        'baseline_mask_class_quotas_preserved_not_positive_background_IDs':True}
    if not out['audit']['primary_once'] or int(zero.sum())!=zero_count(n,epoch):raise AssertionError('Schedule integrity')
    return out


def choose_refs(schedule,reference_minutes,seed,fold,epoch,tag):
    rng=np.random.default_rng(np.random.SeedSequence([seed,fold,epoch,tag,509]))
    sizes=np.asarray([len(x) for x in reference_minutes]);a=np.flatnonzero(sizes>=15);b=np.flatnonzero((sizes>=5)&(sizes<=14))
    if len(a)!=7 or len(b)!=106:raise ValueError('Audited7native/106smooth references changed')
    for name,pool in [('long_native',a),('long_smooth_bell',b)]:
        ix=np.flatnonzero(schedule['family']==name);schedule['reference_index'][ix]=np.resize(rng.permutation(pool),len(ix))
    return schedule


def _basis_metadata(frame,count,seed):
    quota={str(k):0 for k in sorted(frame.site.unique())};groups={str(s):g for s,g in frame.groupby('site')}
    for _ in range(count):
        eligible=[s for s in quota if quota[s]<len(groups[s])];k=min(eligible,key=lambda s:(quota[s],s));quota[k]+=1
    pieces=[]
    for site,g in groups.items():
        keys=[hashlib.sha256(f'R13BASIS|{seed}|{site}|{int(i)}'.encode()).hexdigest() for i in g.base_window_index]
        pieces.append(g.iloc[np.argsort(keys,kind='stable')[:quota[site]]])
    return pd.concat(pieces,ignore_index=True),quota


class R13Sources:
    def __init__(self,config_path='configs/r13_full_training_v1.json',*,bundle=None,verify_banks=True):
        self.root=ROOT;self.config_path=Path(config_path).resolve();self.config=read_config(self.config_path)
        self.original=R9Sources(ROOT/self.config['R9_source_config'],project_root=ROOT,verify_banks=verify_banks)
        self.reference_minutes=self.original.reference_minutes;self.references=self.original.references
        self.bundle=Path(bundle or ROOT/self.config['data_output']).resolve();self._cache={};self._basis={}
    def background(self,row):return self.original.background(row)
    def background_bank(self):return self.original.background_bank()
    def train_metadata(self,fold):
        if fold not in self._cache:
            m=validate_full_bundle(self.bundle,verify_all=False)
            p=f'fold_{fold:02d}/TRAIN_pool.parquet'
            if sha(self.bundle/p)!=m['file_sha256'][p]:raise ValueError('Prepared TRAIN HF cache changed')
            self._cache[fold]=pd.read_parquet(self.bundle/p)
        return self._cache[fold]
    def basis(self,fold):
        if fold not in self._basis:
            m=validate_full_bundle(self.bundle,verify_all=False);f=m['folds'][str(fold)]
            self._basis[fold]=load_spectral_basis(self.bundle/f['basis_path'],expected_sha256=f['basis_sha256'],expected_training_binding=f['basis_training_binding'])
        return self._basis[fold]
    def assert_unchanged(self):
        self.original.assert_unchanged();m=validate_full_bundle(self.bundle,verify_all=True)
        if m['config_sha256']!=sha(self.config_path):raise ValueError('R13 config changed')


def generate_item(sources,row,schedule,index,fold,epoch,role):
    from pipeline.dataset_r13 import reconstruct_reference_from_recipe,solve_observed_mean_mixture
    no=bool(schedule['zero'][index]);ref=int(schedule['reference_index'][index]);family=str(schedule['family'][index])
    bg=sources.background(row);native=None if no else sources.reference_minutes[ref]
    item,recipe=synthesize_r12(bg,native,int(schedule['item_seed'][index]),family,no_source=no)
    if bool(schedule['highshare'][index]):
        placed=reconstruct_reference_from_recipe(native,recipe)
        target=float(schedule['highshare_target'][index]);new,audit=solve_observed_mean_mixture(bg,placed,target,receipt=recipe)
        recipe={'base_peak_SNR_recipe_before_observed_mean_override':recipe,
                'scale_policy':'observed_mean_share','highshare_target':target,**audit}
        item=new
    if bool(item['share']>0)!=(not no):raise AssertionError('R13 positive observed support changed')
    item.update(site=str(row['site']),sample_id=f'r13_f{fold:02d}_{role}_e{epoch:03d}_{index:07d}',
        source_id='SA:'+str(int(row['base_window_index'])),background_group=str(row['background_group']),
        base_window_index=np.int64(row['base_window_index']),reference_index=np.int64(ref),family=family,
        explicit_no_source=np.bool_(no),highshare=np.bool_(schedule['highshare'][index]))
    recipe.update(sample_id=item['sample_id'],source_id=item['source_id'],reference_index=ref,
        family=family,role=role,fold=fold,epoch=epoch,HF_priority_zero=bool(schedule['HF_priority_zero'][index]))
    return item,recipe


def _channels(raw,basis,include_shape):
    x,w=transform_channels(raw[None],basis,include_shape=include_shape)
    if x.shape[1]==8:x=np.pad(x,((0,0),(0,3),(0,0),(0,0)))
    return x[0],w[0]


class DynamicR13Dataset:
    def __init__(self,sources,fold,epoch=1):
        self.sources=sources;self.fold=int(fold);self.metadata=sources.train_metadata(fold).reset_index(drop=True)
        self.rows=self.metadata.to_dict('records');m=validate_full_bundle(sources.bundle,verify_all=False)
        counts=self.metadata.site.value_counts();self.weights=np.asarray([len(self.metadata)/(len(counts)*counts[s]) for s in self.metadata.site],np.float32)
        self.identity={'version':VERSION,'role':'train','fold':fold,'rows':len(self.rows),'data_hash':m['data_hash'],
            'TRAIN_pool_sha256':m['file_sha256'][f'fold_{fold:02d}/TRAIN_pool.parquet'],
            'basis_sha256':m['folds'][str(fold)]['basis_sha256'],'dynamic_epoch_seed':self.sources.config['seed'],
            'all509_shared_references':True,'reference_independent_CV':False,
            'epoch_law':'primaryonce/stratumzeroquotaHF/full-observed-share25pct; exact20pctover5epochs'}
        self.set_epoch(epoch)
    def __len__(self):return len(self.rows)
    def set_epoch(self,epoch):
        if type(epoch)is not int or epoch<1:raise ValueError('Positive integer epoch required')
        self.epoch=epoch;s=self.sources
        self.schedule=choose_refs(make_full_schedule(self.metadata,epoch,self.fold,s.config['seed']),s.reference_minutes,s.config['seed'],self.fold,epoch,0)
    def epoch_audit(self):return dict(self.schedule['audit'])
    def raw_item(self,index):
        row=self.rows[int(self.schedule['primary_order'][index])]
        item,_=generate_item(self.sources,row,self.schedule,index,self.fold,self.epoch,'train')
        item['loss_weight']=self.weights[int(self.schedule['primary_order'][index])]
        return item
    def __getitem__(self,index):
        a=self.raw_item(index);x,w=_channels(a['db'],self.sources.basis(self.fold),self.sources.config['feature_policy']['include_shape'])
        return {k:a[k] for k in ('share','cell_share','presence','site','sample_id','loss_weight')}|{'x':x,'w':w}


def _check_report_gate(folder,gate,fold,data_hash):
    if isinstance(gate,(str,Path)):gate=json.loads(Path(gate).read_text())
    if (not isinstance(gate,dict) or gate.get('fold')!=fold or gate.get('data_bundle_hash')!=data_hash
        or gate.get('all_models_frozen') is not True or gate.get('ensemble_policy_frozen') is not True
        or len(gate.get('model_summary_files',{}))!=5):raise ValueError('R13 actual model/policy freeze required BEFORE report read')
    for name,h in gate['model_summary_files'].items():
        p=ROOT/name
        if sha(p)!=h:raise ValueError('Report model summary changed')
        s=json.loads(p.read_text())
        if s.get('status')!='completed' or s.get('fixture_only') is not False:raise ValueError('Completed production models required')
    if sha(ROOT/gate['ensemble_policy_file'])!=gate['ensemble_policy_sha256']:raise ValueError('Report policy changed')
    if sha(ROOT/gate['global_freeze_file'])!=gate['global_freeze_sha256']:raise ValueError('Global5fold barrier missing/changed')


class FixedR13Dataset:
    def __init__(self,sources,fold,role,*,report_gate=None):
        if role not in ROLES:raise ValueError('Known fixed role required')
        self.sources=sources;self.fold=fold;self.role=role;m=validate_full_bundle(sources.bundle,verify_all=False)
        if role=='report':_check_report_gate(sources.bundle,report_gate,fold,m['data_hash'])
        e=m['folds'][str(fold)]['roles'][role]
        if sha(sources.bundle/e['path'])!=e['sha256']:raise ValueError('R13 fixed numerical values changed')
        with np.load(sources.bundle/e['path'],allow_pickle=False) as z:self.arrays={k:z[k] for k in z.files}
        self.identity={'version':VERSION,'role':role,'fold':fold,'rows':len(self.arrays['share']),
            'data_hash':m['data_hash'],'npz_sha256':e['sha256'],'basis_sha256':m['folds'][str(fold)]['basis_sha256']}
    def __len__(self):return len(self.arrays['share'])
    def raw_arrays(self):return self.arrays.copy()
    def __getitem__(self,index):
        a={k:v[index] for k,v in self.arrays.items()};x,w=_channels(a['db'],self.sources.basis(self.fold),self.sources.config['feature_policy']['include_shape'])
        return {k:a[k] for k in ('share','cell_share','presence','site','sample_id')}|{'x':x,'w':w}


def validate_full_bundle(folder,*,verify_all=True):
    p=Path(folder);m=json.loads((p/'manifest.json').read_text());g=json.loads((p/'readiness.json').read_text())
    if (m.get('version')!=VERSION or g.get('ready_to_train') is not True or sha(p/'manifest.json')!=g['manifest_sha256']
        or digest({k:v for k,v in m.items() if k!='data_hash'})!=m['data_hash'] or g['data_hash']!=m['data_hash']):raise ValueError('Unready R13 full5fold data')
    if verify_all:
        for name,h in m['file_sha256'].items():
            if sha(p/name)!=h:raise ValueError('R13 prepared artifact changed:'+name)
        for name,h in m['code_sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('R13 source changed:'+name)
        if sha(ROOT/m['config_file'])!=m['config_sha256']:raise ValueError('R13 config changed')
    return m
