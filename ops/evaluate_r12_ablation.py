"""R12 all17-frozen CPU evaluation, gate selection then locked reporting.

plan/fixture never load acoustic panels or production weights. Actual
freeze/evaluate requires an explicit operator approval flag. Evaluation
never trains/refits, opens Test, searches weights/thresholds, or changes
legacy artifacts. R11/R10 are exposed development panels, not blind tests.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
ARMS=tuple(f'L{l}H{h}P{s}' for l in (0,1) for h in (0,1) for s in (0,1))
CNN_ARMS=('legacy_contrast_control',)+ARMS
BASE='L0H0P0';REPETITIONS=500;SEED=2026102712
DEFAULT_CONFIG='configs/r12_cpu_ablation_v1.json'
DEFAULT_DATA='data/synthetic/r12_cpu_ablation_v3'
DEFAULT_BASIS='models/r12_shared_basis_v1/basis.npz'
DEFAULT_CNN='models/r12_cpu_cnn_r1';DEFAULT_TREE='models/r12_compact_trees_v1'
R11='reports/r11_diagnostics/full_support_hf_r1'
R10='reports/r10_diagnostics/factorial_transfer_r2'
R9='data/synthetic/group_cv_minute_full_r9_v1/fold_00'
R9_CPU='reports/r10_diagnostics/temporal_permutation_r1/original_CPU_predictions.npz'
R9_PILOT='submissions/r9_fold0_all5_cpu_r1'
DEV_R10=('unseen__r9_complete','seen__mean_scale_only','unseen__combined_stress','seen__combined_stress')
DEV_R11_POS=('full_smooth12','full_smooth13','full_smooth15','native_long15')
DEV_R11_NEG=('st_francis_dense','hf_high')
CODE=('ops/evaluate_r12_ablation.py','pipeline/training_r12.py','ops/train_r12_cnn.py',
      'ops/train_r12_lightgbm.py','pipeline/features_r12.py','pipeline/dataset_r12.py',
      'pipeline/relative_models_r6_v2.py','pipeline/relative_grid_r6_v2.py','pipeline/relative_grid_r6.py')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def token(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def local(name):
    p=Path(name);p=(ROOT/p if not p.is_absolute() else p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('R12 project-contained paths required')
    return p
def rel(p):return local(p).relative_to(ROOT).as_posix()
def pin_paths(paths):return {rel(p):sha(local(p)) for p in paths}
def verify_pins(pins):
    for n,h in pins.items():
        if sha(local(n))!=h:raise ValueError('Frozen input changed: '+n)
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0]))
        if p.name.lower() in ('test.csv','sample_submission.csv','report.npz','outer_validation.npz'):
            raise PermissionError('Test/unused R9report/oldOUTER forbidden')
    if event in ('socket.connect','subprocess.Popen','os.system'):raise PermissionError('No external/process actions')
def environment(threads):
    if os.environ.get('KMP_DUPLICATE_LIB_OK'):raise ValueError('Duplicate OpenMP bypass forbidden')
    if sys.platform=='darwin' and '/opt/homebrew/opt/libomp/lib' not in os.environ.get('DYLD_LIBRARY_PATH','').split(':'):
        raise RuntimeError('Start fresh with DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib')
    for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[n]=str(threads)


def plan(a):
    """Metadata/hashes only, without acoustic NPZ/model loading."""
    cfg=read(local(a.config));m=read(local(a.data)/'manifest.json')
    if local(cfg['fit_sources'])!=local(a.data) or cfg['CNN']['epochs']!=100 or cfg['CNN']['early_stopping'] is not False:
        raise ValueError('Frozen R12 plan differs')
    paths=[*CODE,a.config,local(a.data)/'manifest.json',local(a.data)/'readiness.json',a.basis,
           local(R9_PILOT)/'frozen_policy.json','ops/r9_fold0_pilot_nn_map.json',R9_CPU,
           local(R9_PILOT)/'gate_member_predictions.npz',local(R9_PILOT)/'prequery_plan.json',
           local(R11)/'FROZEN_BEFORE_PREDICTIONS.json',local(R11)/'source_model_pins.json',
           local(R10)/'OUTPUT_MANIFEST.json',local(R10)/'input_pins.json']
    for role in ('selection','ensemble_gate'):
        paths += [local(R9)/(role+'.npz'),local(R9)/(role+'_metadata.parquet')]
    for mode in DEV_R11_POS+DEV_R11_NEG:
        paths += [local(R11)/(mode+'_samples.npz'),local(R11)/(mode+'_predictions.npz')]
        paths.append(local(R11)/('positive_background_manifest.parquet' if mode in DEV_R11_POS else mode+'_background_manifest.parquet'))
    for panel in DEV_R10:
        paths += [local(R10)/panel/'samples.npz',local(R10)/panel/'predictions.npz',local(R10)/(panel.split('__')[0]+'_source_manifest.parquet')]
    # Hash-only closure; no gate/eval numeric members or labels are parsed.
    paths += [local(a.data)/n for n in m['file_sha256']]
    pins=pin_paths(paths)
    baseline=read(local(R9_PILOT)/'frozen_policy.json')
    verify_pins(baseline['source_pins'])
    pins.update(baseline['source_pins'])
    value={'format':'r12_evaluation_predeclared_v1','source_file_sha256':pins,
        'config_path':rel(a.config),'data_dir':rel(a.data),'basis_path':rel(a.basis),'cnn_root':rel(a.cnn),'tree_root':rel(a.tree),
        'data_hash':m['data_hash'],'CNN_models':list(CNN_ARMS),'tree_models':list(ARMS),
        'model_count':17,'output_models_count_with_8blends_and_R9':26,
        'gate_role':'new R12 validation_gate500 only; after all17 frozen',
        'selection':'minimum gate RMSE among8 fixed50:50 CNN/tree shares, strict first canonical tie; baselineL0H0P0 retained if no improvement',
        'score_policy':'R12 pure CNN and8blends score=share; legacy CNN contrast score; individual trees also report classifier and half-share classifier score',
        'unused_roles_after_policy':'R12 eval500 + highSNRlong500, policySHA checked before every load',
        'development_panels':['R9_selection','R9_ensemble_gate',*DEV_R10,*['R11_'+p+'_VS_'+n for p in DEV_R11_POS for n in DEV_R11_NEG]],
        'R9_score_attribution':'Report both original derived/mixed score AUC and share-only AUC. Score-definition changes are not isolated learning gains.',
        'new_R9_inference':'single joined strict inference on new eval/highSNR only; saved old panel predictions reused',
        'CI':{'repetitions':REPETITIONS,'seed':SEED,'method':'paired resampling of known background_group; source_id fallback explicitly marked',
              'comparisons':['selected Blend50 minus baseline Blend50','selected Blend50 minus frozen R9'],
              'metrics':['rmse','auc_share','auc_score'],'interval':[.025,.975],
              'limits':'conditional on fixed panels/models; references shared; multiple comparisons unadjusted; gate-selection optimism remains'},
        'fixed_saturation_threshold':.99,'tail_target_threshold':.2,'Test_fit_inference_or_upload':False,
        'actual_panels_or_production_models_loaded':False}
    value['plan_sha256']=token(value);write(local(a.output),value)
    return value


def check_plan(path):
    p=read(local(path));h=p['plan_sha256']
    if token({k:v for k,v in p.items() if k!='plan_sha256'})!=h:raise ValueError('Evaluation plan content changed')
    verify_pins(p['source_file_sha256']);return p


def freeze(plan_path,output):
    """Build all17 SHA barrier, without opening evaluation arrays."""
    from pipeline.dataset_r12 import validate_r12_bundle
    p=check_plan(plan_path);data=local(p['data_dir']);m=validate_r12_bundle(data,verify_all=True);cnn=local(p['cnn_root']);tree=local(p['tree_root'])
    if m['data_hash']!=p['data_hash']:raise ValueError('Frozen evaluation data differs from predeclared plan')
    cf=read(cnn/'ALL_CNN_FROZEN.json');tf=read(tree/'ALL_TREES_FROZEN.json')
    if any((d/'FAILED.json').exists() for d in (cnn,tree)):raise ValueError('Failed run cannot enter global freeze')
    if (cf.get('all_models_frozen') is not True or tf.get('all_models_frozen') is not True
            or cf['data_manifest_sha256']!=sha(data/'manifest.json') or tf['data_manifest_sha256']!=sha(data/'manifest.json')
            or set(cf['summary_sha256'])!=set(CNN_ARMS) or set(tf['summary_sha256'])!=set(ARMS)):
        raise ValueError('All9CNN+8tree completion barrier required')
    summarymap={};artifactpins={};cnnp=read(cnn/'PRE_FIT_FREEZE.json');tp=read(tree/'PRE_FIT_FREEZE.json')
    if cnnp['basis_sha256']!=sha(local(p['basis_path'])) or tp['basis_sha256']!=cnnp['basis_sha256']:
        raise ValueError('Shared basis differs across model families')
    if cnnp['data_manifest_sha256']!=sha(data/'manifest.json') or tp['data_hash']!=m['data_hash']:raise ValueError('Fit data identity differs')
    verify_pins(cnnp['source_code_sha256']);verify_pins(tp['code_sha256'])
    if sha(local('pipeline/training_r12.py'))!=cnnp['training_code_sha256'] or sha(local('ops/train_r12_cnn.py'))!=cnnp['script_sha256']:
        raise ValueError('CNN engine/driver changed since fit')
    for kind,names,folder,certificate in [('CNN',CNN_ARMS,cnn,cf),('tree',ARMS,tree,tf)]:
        for name in names:
            d=folder/name;s=read(d/'summary.json')
            if s.get('status')!='completed' or (d/'FAILED.json').exists() or sha(d/'summary.json')!=certificate['summary_sha256'][name]:
                raise ValueError('Completed model summary differs: '+name)
            summarymap[rel(d/'summary.json')]=sha(d/'summary.json')
            if kind=='CNN':
                if s['epochs_run']!=100 or s['early_stopping'] is not False or s['parameters']!=26953 or s['arm']['name']!=name:
                    raise ValueError('CNN training policy differs')
                hist=[json.loads(v) for v in (d/'epochs.jsonl').read_text().splitlines()]
                if [v['epoch'] for v in hist]!=list(range(1,101)):raise ValueError('CNN100epoch ledger incomplete')
                best=min(range(100),key=lambda i:hist[i]['validation_fit_rmse'])+1
                if best!=s['best_epoch'] or sha(d/'best.pt')!=s['checkpoint_sha256']:raise ValueError('Strict CNN best checkpoint differs')
                if abs(s['validation_fit_metrics']['rmse']-hist[best-1]['validation_fit_rmse'])>1e-10:raise ValueError('CNN saved best validation metric differs')
                artifactpins.update(pin_paths([d/'best.pt',d/'epochs.jsonl',d/'validation_fit_predictions.npz',d/'STARTED.json']))
            else:
                b=s['training_binding'];arm=f'L{name[1]}_H{name[3]}';shape=name.endswith('1')
                expected={'data_hash':m['data_hash'],'data_manifest_sha256':sha(data/'manifest.json'),'arm':arm,
                    'include_shape':shape,'TRAIN_npz_sha256':m['arms'][arm]['train']['sha256'],
                    'validation_fit_npz_sha256':m['arms'][arm]['validation_fit']['sha256'],
                    'basis_sha256':sha(local(p['basis_path'])) if shape else None,
                    'basis_training_binding':m['basis_training_binding'] if shape else None}
                if s.get('fixture_only') is not False or any(b.get(k)!=v for k,v in expected.items()):raise ValueError('Tree data/source/basis binding differs')
                for n,h in s['artifact_sha256'].items():
                    if Path(n).is_absolute() or '..' in Path(n).parts or sha(d/n)!=h:raise ValueError('Tree artifact changed')
                    artifactpins[rel(d/n)]=h
                verify_pins(s['source_code_sha256']);artifactpins[rel(d/'FROZEN_BEFORE_HELDOUT.json')]=sha(d/'FROZEN_BEFORE_HELDOUT.json')
    artifactpins.update(pin_paths([cnn/'ALL_CNN_FROZEN.json',tree/'ALL_TREES_FROZEN.json',cnn/'PRE_FIT_FREEZE.json',tree/'PRE_FIT_FREEZE.json',cnn/'training_subset.npz']))
    value={'format':'r12_all_models_frozen_v1','all_models_frozen':True,'number_models':17,
        'model_summary_sha256':summarymap,'artifact_sha256':artifactpins,'data_manifest_sha256':sha(data/'manifest.json'),
        'data_hash':m['data_hash'],'basis_sha256':sha(local(p['basis_path'])),'plan_file':rel(plan_path),
        'plan_file_sha256':sha(local(plan_path)),'plan_sha256':p['plan_sha256'],'gate_eval_panels_opened':False}
    value['freeze_sha256']=token(value);write(local(output),value);return value


def check_freeze(path):
    f=read(local(path))
    if (f.get('format')!='r12_all_models_frozen_v1' or f.get('all_models_frozen') is not True
            or f.get('number_models')!=17 or len(f.get('model_summary_sha256',{}))!=17
            or token({k:v for k,v in f.items() if k!='freeze_sha256'})!=f.get('freeze_sha256')):
        raise ValueError('Global17 frozen receipt required')
    p=check_plan(f['plan_file'])
    if sha(local(f['plan_file']))!=f['plan_file_sha256'] or p['plan_sha256']!=f['plan_sha256']:raise ValueError('Foreign plan')
    verify_pins(f['model_summary_sha256']);verify_pins(f['artifact_sha256']);return p,f


def auc(y,score):
    import numpy as np
    z=np.asarray(y)>0;s=np.asarray(score,float);n=int(z.sum());m=len(z)-n
    if n==0 or m==0:return None
    _,inverse,count=np.unique(s,return_inverse=True,return_counts=True)
    ends=np.cumsum(count);ranks=ends-.5*(count-1)
    return float((ranks[inverse][z].sum()-n*(n+1)/2)/(n*m))


def metrics(panel,pred):
    import numpy as np
    y=np.asarray(panel['y'],float);p=np.asarray(pred['share'],float);s=np.asarray(pred['score'],float)
    if y.shape!=p.shape or s.shape!=p.shape or not np.isfinite(np.r_[y,p,s]).all() or ((y<0)|(y>1)|(p<0)|(p>1)|(s<0)|(s>1)).any():raise ValueError('Metric alignment/bounds')
    e=p-y;high=y>.2;sse=float(np.sum(e*e))
    r={'n':len(y),'positive_n':int((y>0).sum()),'zero_n':int((y==0).sum()),'rmse':float(np.sqrt(np.mean(e*e))),
       'mae':float(abs(e).mean()),'bias':float(e.mean()),'prediction_mean':float(p.mean()),'prediction_rms':float(np.sqrt(np.mean(p*p))),
       'target_mean':float(y.mean()),'target_rms':float(np.sqrt(np.mean(y*y))),'auc_share':auc(y,p),'auc_score':auc(y,s),
       'auc_classifier':auc(y,pred['score_classifier']) if 'score_classifier' in pred else None,
       'contrast_ge099_fraction':float(np.mean(np.asarray(pred['contrast'])>=.99)) if 'contrast' in pred else None,
       'component_mixed_ge099_fraction':float(np.mean((2*s-p)>=.99)) if pred.get('mixed_R9_component',False) else None,
       'high_y_n':int(high.sum()),'high_y_SSE':float(np.sum(e[high]**2)),'high_y_SSE_fraction':float(np.sum(e[high]**2)/sse) if sse else None,
       'site_macro_rmse':None}
    if panel.get('site') is not None:
        sites=np.asarray(panel['site']).astype(str);r['site_macro_rmse']=float(np.mean([np.sqrt(np.mean(e[sites==v]**2)) for v in np.unique(sites)]))
    return r


def with_blends(predictions):
    import numpy as np
    result=dict(predictions)
    for arm in ARMS:
        p=(.5*np.asarray(result['CNN_'+arm]['share'],float)+.5*np.asarray(result['Tree_'+arm]['share'],float)).astype(np.float32)
        result['Blend50_'+arm]={'share':p,'score':p.copy()}
    return result


def select_policy(gate,predictions,freeze_sha):
    scores={arm:metrics(gate,predictions['Blend50_'+arm])['rmse'] for arm in ARMS}
    winner=min(ARMS,key=lambda n:scores[n])
    value={'format':'r12_selected_fixed_recipe_v1','selected_arm':winner,'baseline_arm':BASE,
        'weights':{'CNN':.5,'tree':.5},'score_policy':'share','gate_role':'R12_validation_gate',
        'gate_metrics':scores,'strict_canonical_first_tie':True,'selected_improves_gate_baseline':scores[winner]<scores[BASE],
        'global_freeze_sha256':freeze_sha,'weight_threshold_calibration_fitting':False,
        'gate_selection_optimism_and_multiple_comparisons':True}
    value['policy_sha256']=token(value);return value


def check_policy(path,freeze_sha):
    p=read(path)
    if p.get('global_freeze_sha256')!=freeze_sha or token({k:v for k,v in p.items() if k!='policy_sha256'})!=p.get('policy_sha256'):
        raise ValueError('Frozen recipe missing/changed/foreign')
    return p


def paired_ci(panel,selected,other,comparison):
    import numpy as np
    labels=panel.get('groups');key=panel.get('cluster_key')
    if labels is None:return [{'comparison':comparison,'metric':m,'point_difference':metrics(panel,selected)[m]-metrics(panel,other)[m] if metrics(panel,selected)[m] is not None and metrics(panel,other)[m] is not None else None,'lo95':None,'hi95':None,'n_clusters':0,'cluster_key':None,'repetitions':REPETITIONS,'valid_repetitions':0,'reason':'known source IDs unavailable'} for m in ('rmse','auc_share','auc_score')]
    groups=np.asarray(labels).astype(str);unique,inverse=np.unique(groups,return_inverse=True);n=len(unique)
    point1,point2=metrics(panel,selected),metrics(panel,other)
    collected={m:[] for m in ('rmse','auc_share','auc_score')}
    tag=int(hashlib.sha256(panel['name'].encode()).hexdigest()[:8],16);rng=np.random.default_rng(np.random.SeedSequence([SEED,tag]))
    for _ in range(REPETITIONS if n>=2 else 0):
        counts=np.bincount(rng.integers(n,size=n),minlength=n);ix=np.repeat(np.arange(len(groups)),counts[inverse])
        sub={'y':np.asarray(panel['y'])[ix]}
        for m,a,b in [('rmse',selected['share'],other['share']),('auc_share',selected['share'],other['share']),('auc_score',selected['score'],other['score'])]:
            aa=np.asarray(a)[ix];bb=np.asarray(b)[ix]
            av=float(np.sqrt(np.mean((aa-sub['y'])**2))) if m=='rmse' else auc(sub['y'],aa)
            bv=float(np.sqrt(np.mean((bb-sub['y'])**2))) if m=='rmse' else auc(sub['y'],bb)
            if av is not None and bv is not None:collected[m].append(av-bv)
    return [{'comparison':comparison,'metric':m,'point_difference':point1[m]-point2[m] if point1[m] is not None and point2[m] is not None else None,
             'lo95':float(np.quantile(v,.025)) if v else None,'hi95':float(np.quantile(v,.975)) if v else None,
             'n_clusters':n,'cluster_key':key,'repetitions':REPETITIONS,'valid_repetitions':len(v),
             'reason':None if n>=2 else 'fewer than two source clusters'} for m,v in collected.items()]


def csv_rows(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


class ProductionPredictor:
    """Strict completed model loading; predict API receives db ONLY."""
    def __init__(self,p,f,threads=2,batch=128):
        import torch
        from pipeline.training_r12 import R12MaskCNN
        from pipeline.features_r12 import load_spectral_basis
        from ops.train_r12_lightgbm import CompactPhysicsLGBM
        self.batch=batch;self.cnn={};self.tree={};torch.set_num_threads(threads)
        self.basis=load_spectral_basis(local(p['basis_path']),expected_sha256=f['basis_sha256'],expected_training_binding=read(local(p['data_dir'])/'manifest.json')['basis_training_binding'])
        for arm in CNN_ARMS:
            folder=local(p['cnn_root'])/arm;s=read(folder/'summary.json');ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            if ck['epoch']!=s['best_epoch'] or ck['parameters']!=26953 or ck['arm']!=s['arm']:raise ValueError('CNN checkpoint metadata mismatch')
            model=R12MaskCNN();model.load_state_dict(ck['state_dict'],strict=True);model.eval()
            if not all(torch.isfinite(v).all() for v in model.parameters()):raise FloatingPointError('Nonfinite CNN weight')
            self.cnn[arm]=model
        for arm in ARMS:
            folder=local(p['tree_root'])/arm;self.tree[arm]=CompactPhysicsLGBM.load(folder,expected_binding=read(folder/'summary.json')['training_binding'])
    def __call__(self,db):
        import numpy as np,torch
        from pipeline.training_r12 import channels,predict
        result={};cache={}
        for shape in (False,True):
            x,w=channels(db,self.basis,shape);cache[shape]=(torch.from_numpy(x),torch.from_numpy(w))
        for arm,model in self.cnn.items():
            x,w=cache[arm.endswith('1') if arm!='legacy_contrast_control' else False]
            r=predict(model,x,w,self.batch,legacy_score=arm=='legacy_contrast_control')
            if np.max(abs((r['cell_mask']*w.numpy()).sum((1,2))-r['share']))>3e-6 or np.any(r['cell_mask'][w.numpy()==0]!=0):raise ValueError('CNN pool/mask mismatch')
            result['CNN_'+arm]={k:r[k] for k in ('share','score','contrast')}
        for arm,tree in self.tree.items():result['Tree_'+arm]=tree.predict(db,batch_size=self.batch)
        return with_blends(result)


def _panel(name,arrays,meta,stage,baseline=None):
    import numpy as np
    y=np.asarray(arrays['share'],np.float32);db=np.asarray(arrays['db'])
    if db.shape!=(len(y),15,25) or not np.isfinite(db).any((1,2)).all() or np.isinf(db).any():raise ValueError('Invalid panel db')
    if meta is not None and len(meta)!=len(y):raise ValueError('Metadata/array order count mismatch')
    ids=arrays.get('sample_id',np.asarray([name+'_'+str(i) for i in range(len(y))]))
    if len(set(np.asarray(ids).astype(str)))!=len(y):raise ValueError('Panel sample IDs must be unique')
    if meta is not None and 'sample_id' in meta and not np.array_equal(meta.sample_id.astype(str),np.asarray(ids).astype(str)):raise ValueError('Metadata ID order mismatch')
    groups=meta.background_group.astype(str).to_numpy() if meta is not None and 'background_group' in meta else arrays.get('source_id')
    return {'name':name,'db':db,'y':y,'sample_id':np.asarray(ids).astype(str),'groups':groups,
        'cluster_key':'background_group' if meta is not None and 'background_group' in meta else 'source_id_fallback' if groups is not None else None,
        'site':meta.site.astype(str).to_numpy() if meta is not None and 'site' in meta else arrays.get('site'),
        'stage':stage,'development_not_blind':True,'baseline':baseline}


def _npz(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def load_R12(p,role,policy_path=None,freeze_sha=None):
    import pandas as pd
    if role not in ('validation_gate','eval','high_snr_long'):raise ValueError('Explicit approved R12 evaluation role required')
    if role!='validation_gate':check_policy(policy_path,freeze_sha)
    m=read(local(p['data_dir'])/'manifest.json');entry=m['common_validation'][role] if role!='high_snr_long' else m['high_snr_long']
    if role=='high_snr_long' and not entry['created']:return None
    arrays=_npz(local(p['data_dir'])/entry['path']);meta=pd.read_parquet(local(p['data_dir'])/entry['metadata_path'])
    return _panel('R12_'+role,arrays,meta,'gate_selection' if role=='validation_gate' else 'post_policy_locked')


def load_development():
    import numpy as np,pandas as pd
    panels=[];policy=read(local(R9_PILOT)/'frozen_policy.json');weights=np.asarray(policy['weights'],float)
    for role in ('selection','ensemble_gate'):
        arrays=_npz(local(R9)/(role+'.npz'));meta=pd.read_parquet(local(R9)/(role+'_metadata.parquet'))
        if role=='selection':
            saved=_npz(local(R9_CPU));order={v:i for i,v in enumerate(saved['sample_id'].astype(str))};ix=[order[v] for v in arrays['sample_id'].astype(str)]
            if saved['member_order'].astype(str).tolist()[-1]!='frozen_all5':raise ValueError('Saved R9 CPU member order differs')
            baseline={k:saved[k][ix,-1] for k in ('share','score')}
        else:
            saved=_npz(local(R9_PILOT)/'gate_member_predictions.npz')
            if not np.array_equal(saved['sample_id'].astype(str),arrays['sample_id'].astype(str)):raise ValueError('Saved R9 gate ID order differs')
            baseline={k:np.clip(saved[k].astype(float)@weights,0,1).astype(np.float32) for k in ('share','score')}
        baseline['mixed_R9_component']=True;panels.append(_panel('R9_'+role,arrays,meta,'exposed_development',baseline))
    r11={}
    for mode in DEV_R11_POS+DEV_R11_NEG:
        arrays=_npz(local(R11)/(mode+'_samples.npz'));path='positive_background_manifest.parquet' if mode in DEV_R11_POS else mode+'_background_manifest.parquet'
        r11[mode]=_panel('R11_'+mode,arrays,pd.read_parquet(local(R11)/path),'exposed_development')
        saved=_npz(local(R11)/(mode+'_predictions.npz'));r11[mode]['baseline']={k:saved['all5_frozen_'+k] for k in ('share','score')};r11[mode]['baseline']['mixed_R9_component']=True
    for name in DEV_R10:
        arrays=_npz(local(R10)/name/'samples.npz');meta=pd.read_parquet(local(R10)/(name.split('__')[0]+'_source_manifest.parquet'))
        saved=_npz(local(R10)/name/'predictions.npz');b={k:saved['frozen_all5_'+k] for k in ('share','score')};b['mixed_R9_component']=True
        panels.append(_panel('R10_'+name,arrays,meta,'exposed_development',b))
    return panels,r11


def add_R9_new_baselines(panels,batch,threads):
    """One joined strict inference, only after the selected policy is frozen."""
    import numpy as np
    from ops.infer_r9_fold0_all5_pilot import inputs,predict_members,combine,DEFAULT_TREE,DEFAULT_BUNDLE,DEFAULT_CONFIG
    nn=read(local('ops/r9_fold0_pilot_nn_map.json'));pins,_,binding,tree=inputs(nn,local(DEFAULT_TREE),local(DEFAULT_BUNDLE),local(DEFAULT_CONFIG))
    joined=np.concatenate([p['db'] for p in panels]);pred,timing=predict_members(nn,tree,joined,binding,batch_size=batch,threads=threads)
    combined=combine(pred,read(local(R9_PILOT)/'frozen_policy.json')['weights']);start=0
    for p in panels:
        n=len(p['y']);p['baseline']={k:combined[k][start:start+n] for k in ('share','score')};p['baseline']['mixed_R9_component']=True;start+=n
    verify_pins(pins);return timing


def collect(panel,pred,out,rows,cis,policy):
    import numpy as np
    if panel.get('baseline') is not None:pred=dict(pred,**{'R9_frozen_all5':panel['baseline']})
    for model,r in pred.items():rows.append({'panel':panel['name'],'model':model,'stage':panel['stage'],'development_not_blind':True,**metrics(panel,r)})
    selected=pred['Blend50_'+policy['selected_arm']]
    for other,label in [('Blend50_'+BASE,'selected_minus_base_Blend50'),('R9_frozen_all5','selected_minus_R9_frozen_all5')]:
        if other in pred:
            cis.extend({'panel':panel['name'],**r} for r in paired_ci(panel,selected,pred[other],label))
    arrays={'sample_id':panel['sample_id'],'target_share':panel['y'],'model_order':np.asarray(list(pred))}
    for k in ('share','score'):arrays[k]=np.column_stack([r[k] for r in pred.values()])
    np.savez_compressed(out/'predictions'/(panel['name']+'.npz'),**arrays)
    return pred


def evaluate(global_path,output,*,threads=2,batch=128):
    import numpy as np
    p,f=check_freeze(global_path);out=local(output);out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir()
    write(out/'STARTED.json',{'global_freeze_file_sha256':sha(local(global_path)),'all17_frozen_before_any_gate':True,'new_training_Test_used':False})
    try:
        predictor=ProductionPredictor(p,f,threads,batch);gate=load_R12(p,'validation_gate');gp=predictor(gate['db'])
        policy=select_policy(gate,gp,f['freeze_sha256']);policy_path=out/'selected_policy.json';write(policy_path,policy)
        policy_file_sha=sha(policy_path);rows=[];cis=[]
        collect(gate,gp,out,rows,cis,policy)
        # Locked roles are first parsed only after policy JSON is durable.
        locked=[load_R12(p,role,policy_path,f['freeze_sha256']) for role in ('eval','high_snr_long')];locked=[v for v in locked if v is not None]
        timing=add_R9_new_baselines(locked,batch,threads)
        for panel in locked:collect(panel,predictor(panel['db']),out,rows,cis,policy)
        dev,r11=load_development()
        for panel in dev:collect(panel,predictor(panel['db']),out,rows,cis,policy)
        rawpred={mode:predictor(panel['db']) for mode,panel in r11.items()}
        for positive in DEV_R11_POS:
            for negative in DEV_R11_NEG:
                a,b=r11[positive],r11[negative];name=a['name']+'_VS_'+negative
                panel={'name':name,'stage':'exposed_development','y':np.r_[a['y'],b['y']],
                    'sample_id':np.r_[a['sample_id'],b['sample_id']],'groups':np.r_[a['groups'],b['groups']],
                    'cluster_key':a['cluster_key'],'site':np.r_[a['site'],b['site']],
                    'baseline':{k:np.r_[a['baseline'][k],b['baseline'][k]] for k in ('share','score')}}
                panel['baseline']['mixed_R9_component']=True
                pred={m:{k:np.r_[rawpred[positive][m][k],rawpred[negative][m][k]] for k in rawpred[positive][m] if k in ('share','score','contrast','score_classifier')} for m in rawpred[positive]}
                collect(panel,pred,out,rows,cis,policy)
        csv_rows(out/'results.csv',rows);csv_rows(out/'paired_cluster_CI.csv',cis)
        if sha(policy_path)!=policy_file_sha:raise ValueError('Selected policy changed after locked panels')
        check_freeze(global_path)
        value={'format':'r12_full_ablation_evaluation_v1','status':'completed','selected_policy':policy,
            'global_freeze_file_sha256':sha(local(global_path)),'results_csv':'results.csv','paired_cluster_CI_csv':'paired_cluster_CI.csv',
            'all_models_frozen_before_gate':True,'policy_frozen_before_locked_eval':True,'R9_new_baseline_inference_timing':timing,
            'original_sources_models_policy_unchanged':True,'Test_or_training_or_threshold_weight_refit':False,
            'development_not_blind':True,'performance_or_GPU_or_LB_improvement_guaranteed':False,
            'limitations':['Conditional fixed-panel cluster CIs; shared reference recordings remain dependent.',
                'CI does not adjust17models/8recipes/multiple panels or gate-selection optimism.',
                'R11/R10 and R9source groups were already development-exposed.',
                'Compare R9 share AUC separately: replacing derived score with share changes the score definition.']}
        value['output_sha256']={str(v.relative_to(out)):sha(v) for v in out.rglob('*') if v.is_file()}
        write(out/'RESULTS.json',value);return value
    except BaseException as e:
        write(out/'FAILED.json',{'error':repr(e),'partial_reuse_allowed':False,'policy_or_source_not_rewritten':True});raise


def fixture(output):
    """Pure numeric tests; never loads source data or any fitted model."""
    import numpy as np
    out=local(output);out.mkdir(parents=True,exist_ok=False);events=[]
    y=np.asarray([0,0,.05,.2,.3,.5]);panel={'name':'numeric_fixture','y':y,'sample_id':np.asarray(list('abcdef')),
        'groups':np.asarray(['g0','g0','g1','g1','g2','g2']),'cluster_key':'numeric_background_group'}
    members={}
    for i,arm in enumerate(ARMS):
        # All arrays fixed before gate selection. No stochastic search.
        v=np.clip(y+(.04 if i==0 else .02 if i==1 else .06),0,1).astype(np.float32)
        for kind in ('CNN','Tree'):members[kind+'_'+arm]={'share':v,'score':v.copy()}
    predictions=with_blends(members);policy=select_policy(panel,predictions,'numeric_freeze17')
    assert policy['selected_arm']=='L0H0P1'
    path=out/'selected_policy.json';events.append('gate_loaded_after_numeric_global17_freeze');write(path,policy);events.append('policy_written')
    check_policy(path,'numeric_freeze17');events.append('locked_numeric_role_allowed')
    assert events.index('policy_written')<events.index('locked_numeric_role_allowed')
    altered=dict(policy,selected_arm=BASE);write(out/'tampered.json',altered)
    try:check_policy(out/'tampered.json','numeric_freeze17');raise AssertionError('Tampered policy accepted')
    except ValueError:pass
    try:check_policy(path,'foreign_freeze');raise AssertionError('Foreign global freeze accepted')
    except ValueError:pass
    equal=with_blends({k:{'share':y.copy(),'score':y.copy()} for k in members})
    assert select_policy(panel,equal,'numeric_freeze17')['selected_arm']==BASE
    # Independent pairwise AUC with ties, zero denominator and RMSE oracle.
    s=np.array([.1,.1,.1,.5,.6,.6]);z=y>0
    expected=float(np.mean(s[z,None]>s[~z][None,:])+.5*np.mean(s[z,None]==s[~z][None,:]))
    assert abs(auc(y,s)-expected)<1e-12 and auc(np.zeros(3),np.ones(3)) is None
    selected=predictions['Blend50_'+policy['selected_arm']];base=predictions['Blend50_'+BASE]
    ci=paired_ci(panel,selected,base,'numeric_selected_minus_base');assert all(v['valid_repetitions']==500 for v in ci if v['metric']=='rmse')
    assert ci==paired_ci(panel,selected,base,'numeric_selected_minus_base')
    identity=paired_ci(panel,selected,selected,'numeric_identity')
    assert all(v['point_difference']==0 and v['lo95']==0 and v['hi95']==0 for v in identity)
    m=metrics(panel,selected);assert abs(m['rmse']-np.sqrt(np.mean((selected['share']-y)**2)))<1e-12
    assert abs(m['high_y_SSE_fraction']-np.sum((selected['share'][y>.2]-y[y>.2])**2)/np.sum((selected['share']-y)**2))<1e-12
    # Invoke the actual locked-role loader with an invalid policy. It must
    # reject BEFORE consulting even a nonexistent dataset manifest/NPZ.
    from unittest import mock
    with mock.patch(__name__+'._npz',side_effect=AssertionError('locked array opened before policy')) as access:
        try:load_R12({},'eval',out/'tampered.json','numeric_freeze17');raise AssertionError('Locked role accepted a tampered policy')
        except ValueError:pass
        assert access.call_count==0
    incomplete={'format':'r12_all_models_frozen_v1','all_models_frozen':True,'number_models':16,'model_summary_sha256':{}}
    incomplete['freeze_sha256']=token(incomplete);write(out/'incomplete_global.json',incomplete)
    try:check_freeze(out/'incomplete_global.json');raise AssertionError('Incomplete model barrier accepted')
    except ValueError:pass
    result={'status':'passed','fixture_only':True,'new_models_or_acoustic_panels_loaded':False,
        'script_sha256':sha(__file__),'gated_role_event_order':events,'selected_first_arm':'L0H0P1',
        'exact_tie_retains_baseline':True,'tampered_policy_foreign_global_freeze_rejected':True,
        'AUC_pairwise_ties_and_missing_class_oracle':True,'paired_background_cluster500_CI_exact_repeat':True,
        'identical_model_CI_exactzero':True,'metric_high_y_SSE_oracle':True,'CI':ci,
        'locked_loader_tampered_policy_rejected_before_array_access':True,'incomplete_model_global_freeze_rejected':True,
        'performance_claim':False}
    write(out/'VERIFICATION.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    pp=sub.add_parser('plan');pp.add_argument('--config',default=DEFAULT_CONFIG);pp.add_argument('--data',default=DEFAULT_DATA);pp.add_argument('--basis',default=DEFAULT_BASIS);pp.add_argument('--cnn',default=DEFAULT_CNN);pp.add_argument('--tree',default=DEFAULT_TREE);pp.add_argument('--output',required=True)
    ff=sub.add_parser('freeze');ff.add_argument('--plan',required=True);ff.add_argument('--output',required=True);ff.add_argument('--execute-approved',action='store_true')
    ee=sub.add_parser('evaluate');ee.add_argument('--global-freeze',required=True);ee.add_argument('--output',required=True);ee.add_argument('--execute-approved',action='store_true');ee.add_argument('--threads',type=int,default=2);ee.add_argument('--batch',type=int,default=128)
    fx=sub.add_parser('fixture');fx.add_argument('--output',required=True)
    a=parser.parse_args()
    if a.mode in ('freeze','evaluate') and not a.execute_approved:parser.error('Explicit root GO then --execute-approved required; no evaluation is started')
    if a.mode=='evaluate':environment(a.threads)
    sys.addaudithook(guard)
    result=plan(a) if a.mode=='plan' else freeze(a.plan,a.output) if a.mode=='freeze' else evaluate(a.global_freeze,a.output,threads=a.threads,batch=a.batch) if a.mode=='evaluate' else fixture(a.output)
    print(json.dumps({'status':result.get('status','prepared'),'mode':a.mode,'output':a.output}),flush=True)
if __name__=='__main__':main()
