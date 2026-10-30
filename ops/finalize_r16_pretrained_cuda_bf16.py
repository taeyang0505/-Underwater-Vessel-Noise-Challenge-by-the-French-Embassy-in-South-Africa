"""Freeze completed public-PEFT Fold0 + Tree5, evaluate, then infer Test once.

No training, policy retuning on locked panels, distribution calibration or
official upload. Explicit family subsets are disclosed partial candidates.
"""
from pathlib import Path
import argparse, gc, html, json, os, sys, tempfile, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.r16_multisupport_data import OUT, PARENT, REGIMES, read, write, sha, save, load, role
from pipeline.training_r16_pretrained_cuda_bf16 import RUN, OPS, FAMILIES, TASKS, branch, base_fingerprint, source_pins, selection_split, NUMERICAL_POLICY
from pipeline.r16_feature_masking import FeatureMaskingConfig
from pipeline.training_r16_multisupport import MODELS as TREES, codes, verify, features, tree_features, metric

FINAL = RUN/'final'
REPORT = ROOT/'reports/r16_pretrained_cuda_bf16_r1'
RUN_ROOT=RUN
VARIANT = None
SUBMISSION = ROOT/'submissions/r16_pretrained_cuda_bf16_r1/submission_r16_pretrained_cuda_bf16.csv'


def configure_variant(variant):
    global RUN,FINAL,SUBMISSION,VARIANT
    if variant != 'both':raise ValueError('Unknown feature masking variant')
    VARIANT=variant
    RUN=RUN_ROOT/variant;FINAL=RUN/'final'
    SUBMISSION=ROOT/'submissions/r16_pretrained_cuda_bf16_r1'/variant/'submission_r16_pretrained_cuda_bf16.csv'


def check_mask_identity(record, *, variant, config, split_sha256):
    """Reject mixing augmentation variants or checkpoint-selection partitions."""
    if record.get('variant') != variant or record.get('masking_config') != config:
        raise ValueError('Completed checkpoint/model masking identity differs')
    if record.get('local_fit_split_sha256') != split_sha256:
        raise ValueError('Completed checkpoint/model development split differs')


def check_checkpoint(cp, summary, family, regime, task, plan):
    if cp.get('format') != 'r16_pretrained_adapter_checkpoint_v1' or (cp.get('family'),cp.get('regime'),cp.get('task')) != (family,regime,task):
        raise ValueError('Wrong task checkpoint')
    for item in (summary,cp):
        if item.get('numerical_policy') != NUMERICAL_POLICY:raise ValueError('CUDA BF16 checkpoint numerical policy differs')
        check_mask_identity(item,variant=VARIANT,config=plan['masking_variants'][VARIANT],
                            split_sha256=plan['local_fit_split_sha256'])
    if cp['epoch'] != summary['best_epoch'] or cp['best_epoch'] != summary['best_epoch'] or cp['best_cv'] != summary['best_cv']:
        raise ValueError('Selected checkpoint differs from completed best epoch/metric')
    if plan.get('numerical_policy') != NUMERICAL_POLICY:raise ValueError('Frozen plan numerical policy differs')
    if cp['source_sha256'] != plan['source_sha256'] or cp['data_manifest_sha256'] != plan['data_manifest_sha256']:
        raise ValueError('Checkpoint/source/data binding differs')
    if not isinstance(cp['epoch'],int) or cp['epoch']<1 or not np.isfinite(cp['best_cv']):
        raise ValueError('Invalid selected checkpoint epoch/metric')
    if 'epochs_run' in summary and cp['epoch']>summary['epochs_run']:
        raise ValueError('Best checkpoint is later than completed training')


def safe(path):
    path = Path(path); path = (ROOT/path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT): raise ValueError('Project-contained output/artifact required')
    return path


def vector(a, n):
    a = np.asarray(a, float)
    if a.shape != (n,) or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError('Each prediction must be a finite [N] vector in [0,1]')
    return a


def aligned(d, ids, y, key):
    if not np.array_equal(d['sample_id'].astype(str), ids.astype(str)) or not np.array_equal(d['target_share'], y):
        raise ValueError('Selection ID/order/target identity differs')
    return vector(d[key], len(y))


def fit_policy(x, c, y, members):
    from scipy.optimize import minimize
    from sklearn.metrics import roc_auc_score
    x = np.asarray(x, float); c = np.asarray(c, float); y = np.asarray(y, float)
    if x.ndim != 2 or x.shape != c.shape or x.shape != (len(y), len(members)) or members[0] != 'lightgbm5':
        raise ValueError('Aligned prediction matrices and LightGBM-first order required')
    if y.ndim != 1 or not np.isfinite(y).all() or ((y < 0) | (y > 1)).any() or len(np.unique(y > 0)) != 2:
        raise ValueError('Finite physical labels and both classes required')
    for j in range(len(members)): vector(x[:,j], len(y)); vector(c[:,j], len(y))
    neural = [j for j,n in enumerate(members) if n not in ('lightgbm5','catboost5')]
    cats = [j for j,n in enumerate(members) if n == 'catboost5']
    prior = np.zeros(len(members)); prior[neural] = min(.4/len(neural) if cats else .5/len(neural), .25)
    if cats: prior[cats[0]] = .1
    prior[0] = 1-prior.sum(); ridge = 1e-4
    bounds = [(.5,1.)] + [(0.,.5 if n == 'catboost5' else .25) for n in members[1:]]
    opt = minimize(lambda w:np.mean((x@w-y)**2)+ridge*np.sum((w-prior)**2), prior,
        jac=lambda w:2*x.T@(x@w-y)/len(y)+2*ridge*(w-prior), method='SLSQP', bounds=bounds,
        constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(len(w))}],
        options={'ftol':1e-12,'maxiter':1000})
    if not opt.success: raise ValueError('Capped convex policy fit failed: '+opt.message)
    candidates = {}
    tree = np.zeros(len(members)); tree[0] = 1
    candidates['lightgbm_only'] = tree.copy()
    if cats:
        for catmass in (.25,.5):
            v=tree.copy();v[0]-=catmass;v[cats[0]]=catmass;candidates[f'tree_cat_{catmass}']=v
    for tree_mass in (.75,.6,.5):
        v=np.zeros(len(members));v[0]=tree_mass;v[neural]=(1-tree_mass)/len(neural)
        candidates[f'lightgbm{tree_mass}_neural_uniform']=v
    aucs={k:float(roc_auc_score(y>0,c@v)) for k,v in candidates.items()}
    chosen=max(candidates,key=lambda k:(aucs[k],candidates[k][0]));w=opt.x;cw=candidates[chosen]
    if w[0] < .5-1e-7 or (w[neural] > .25+1e-7).any() or (w < -1e-8).any() or abs(w.sum()-1)>1e-7:
        raise ValueError('User regression caps violated')
    return {'regression_weights':w.tolist(),'independent_classifier_weights':cw.tolist(),
        'classifier_candidate':chosen,'classifier_candidates':{k:v.tolist() for k,v in candidates.items()},
        'classifier_fit_AUC_candidates':aucs,'score_share_weight':0.,'ridge_to_prior':ridge,
        'regression_prior':prior.tolist(),'fit_metrics':metric(y,x@w,c@cw)}


def apply(policy, shares, probabilities):
    members=policy['member_order']; n=len(shares['D0']); p=np.zeros(n);c=p.copy()
    if policy['D0_H0_weights'] != [.5,.5] or policy['score_share_weight'] != 0.: raise ValueError('Frozen policy contract differs')
    for r in REGIMES:
        s=np.asarray(shares[r],float);v=np.asarray(probabilities[r],float)
        if s.shape != (n,len(members)) or v.shape != s.shape: raise ValueError('Prediction matrix shape mismatch')
        for j in range(len(members)):vector(s[:,j],n);vector(v[:,j],n)
        p+=.5*(s@policy['experts'][r]['regression_weights'])
        c+=.5*(v@policy['experts'][r]['independent_classifier_weights'])
    return {'share':vector(np.clip(p,0,1),n),'score':vector(np.clip(c,0,1),n)}


def add_completed(folder, pins, *, neural=False, expected=None):
    folder=safe(folder);s=read(folder/'summary.json')
    if s.get('status') != 'completed' or (folder/'FAILED.json').exists(): raise ValueError('Completed unfailed model required: '+str(folder))
    if s.get('data_manifest_sha256') != sha(OUT/'manifest.json'): raise ValueError('Training data identity differs')
    for k,v in (expected or {}).items():
        if s.get(k) != v: raise ValueError('Completed model identity differs: '+k)
    verify(s['source_sha256'] if neural else s['code_sha256'])
    if neural and (s['source_sha256'] != source_pins() or s.get('permanent_base_unchanged') is not True):
        raise ValueError('Current verified frozen-base PEFT run required')
    ext='.cbm' if str(s.get('format','')).startswith('r16_peft_catboost_pair') else '.txt'
    required=({'best.pt','selection.npz','PLAN.json','cv_best.npz','checkpoint_validation_best.npz','epochs.jsonl'} if neural else
              {'regression'+ext,'classification'+ext,'selection.npz'})
    if not required<=set(s['artifact_sha256']):raise ValueError('Required completed artifacts absent from SHA closure')
    for name,h in s['artifact_sha256'].items():
        p=safe(folder/name)
        if p.parent != folder or sha(p) != h: raise ValueError('Completed artifact changed/path escaped')
        pins[str(p.relative_to(ROOT))]=h
    pins[str((folder/'summary.json').relative_to(ROOT))]=sha(folder/'summary.json')
    return s


def prerequisites(families, catboost_root=None):
    if VARIANT is None:raise ValueError('Explicit masking variant must be configured')
    if not families or len(set(families)) != len(families) or not set(families)<=set(FAMILIES): raise ValueError('Explicit unique supported families required')
    plan=read(OPS/'RUN_PLAN.json');pins=source_pins();verify(pins)
    if not set(families)<=set(plan['families']) or plan['source_sha256'] != pins: raise ValueError('Selected families/current source outside immutable run plan')
    if VARIANT not in plan['variants'] or plan['masking_variants'][VARIANT] != FeatureMaskingConfig.for_variant(VARIANT).state_dict():
        raise ValueError('Masking variant/configuration outside immutable run plan')
    split_path=OPS/'LOCAL_FIT_SPLIT.json'
    if not split_path.is_file() or sha(split_path) != plan['local_fit_split_sha256']:
        raise ValueError('Fixed development partition absent or changed')
    selection_split()  # Validate source identities, row order, classes and purge.
    pins[str(split_path.relative_to(ROOT))]=sha(split_path)
    pins[str((OPS/'RUN_PLAN.json').relative_to(ROOT))]=sha(OPS/'RUN_PLAN.json')
    t=read(TREES/'ALL_TREES_COMPLETED.json')
    if t['status'] != 'completed' or t['code_sha256'] != codes() or t['data_manifest_sha256'] != sha(OUT/'manifest.json'):
        raise ValueError('Complete same-data Tree5 registry required')
    pins[str((TREES/'ALL_TREES_COMPLETED.json').relative_to(ROOT))]=sha(TREES/'ALL_TREES_COMPLETED.json')
    for r in REGIMES:
        for f in range(5): add_completed(TREES/'trees'/r/f'fold_{f:02d}',pins)
        for family in families:
            for task in TASKS:
                folder=RUN/family/r/task
                summary=add_completed(folder,pins,neural=True,expected={'family':family,'regime':r,'task':task,'variant':VARIANT})
                check_mask_identity(summary,variant=VARIANT,config=plan['masking_variants'][VARIANT],split_sha256=plan['local_fit_split_sha256'])
                job=read(folder/'PLAN.json')
                if (job.get('family'),job.get('regime'),job.get('task'),job.get('variant')) != (family,r,task,VARIANT):
                    raise ValueError('Job plan identity differs')
                if job.get('augmentation') != plan['masking_variants'][VARIANT] or job.get('local_fit_split_sha256') != plan['local_fit_split_sha256']:
                    raise ValueError('Job plan masking/development partition differs')
                if job.get('run_plan') != plan:
                    raise ValueError('Job and global immutable run plans differ')
                import torch
                cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
                check_checkpoint(cp,summary,family,r,task,plan)
                if cp.get('provenance') != job.get('provenance') or cp.get('permanent_base_fingerprint') != job.get('permanent_base_fingerprint'):
                    raise ValueError('Job and selected public-backbone identities differ')
                del cp
    cat=safe(catboost_root) if catboost_root else RUN/'catboost'
    complete=all((cat/r/f'fold_{f:02d}'/'summary.json').is_file() for r in REGIMES for f in range(5))
    if complete:
        for r in REGIMES:
            for f in range(5):
                s=add_completed(cat/r/f'fold_{f:02d}',pins,expected={'fold':f,'regime':r,'disposable_smoke_only':False})
                from pipeline.catboost_r16_peft import VERSION
                if s['format'] != VERSION: raise ValueError('Wrong CatBoost production format')
    elif catboost_root: raise ValueError('Explicit CatBoost requires all ten complete non-smoke pairs')
    members=['lightgbm5']+(['catboost5'] if complete else [])+list(families)
    pins[str(OUT.relative_to(ROOT)/'manifest.json')]=sha(OUT/'manifest.json')
    return pins,members,cat if complete else None


class Predictor:
    def __init__(self, families, catroot, device, batch):
        import lightgbm as lgb
        self.families,self.catroot,self.device,self.batch=families,catroot,device,batch
        self.trees={r:[(lgb.Booster(model_file=str(TREES/'trees'/r/f'fold_{f:02d}'/'regression.txt')),
                       lgb.Booster(model_file=str(TREES/'trees'/r/f'fold_{f:02d}'/'classification.txt'))) for f in range(5)] for r in REGIMES}
    def neural(self, family, regime, task, x, w):
        import torch
        from pipeline.training_r16_pretrained_cuda_bf16 import predict
        folder=RUN/family/regime/task;cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        summary=read(folder/'summary.json')
        check_checkpoint(cp,summary,family,regime,task,read(OPS/'RUN_PLAN.json'))
        model,prov=branch(family,task,self.device)
        for key in ('model_id','revision','weight_sha256'):
            if prov[key] != cp['provenance'][key]: raise ValueError('Public backbone identity differs')
        if base_fingerprint(model) != cp['permanent_base_fingerprint']: raise ValueError('Pretrained base reconstruction differs')
        model.load_adapter_state(cp['adapter_state']);model.eval()
        p=predict(model,task,{'x':x,'w':w,'share':np.zeros(len(x))},self.device,self.batch)
        del model,cp;gc.collect()
        if self.device=='mps':torch.mps.empty_cache()
        elif self.device=='cuda':torch.cuda.empty_cache()
        return vector(p,len(x))
    def predict(self, db):
        x,w=features(db);tr=tree_features(db,'regression');tc=tree_features(db,'classification');shares={};probs={}
        for r in REGIMES:
            ps=[np.mean([np.clip(a.predict(tr,num_threads=2),0,1) for a,b in self.trees[r]],0)]
            cs=[np.mean([b.predict(tc,num_threads=2) for a,b in self.trees[r]],0)]
            if self.catroot:
                from pipeline.catboost_r16_peft import CatBoostR16Pair
                pairs=[CatBoostR16Pair.load(self.catroot/r/f'fold_{f:02d}').predict_features(tr,tc) for f in range(5)]
                ps.append(np.mean([a for a,b in pairs],0));cs.append(np.mean([b for a,b in pairs],0))
            for family in self.families:
                ps.append(self.neural(family,r,'regression',x,w));cs.append(self.neural(family,r,'classification',x,w))
            shares[r]=np.column_stack(ps);probs[r]=np.column_stack(cs)
        return shares,probs


def finalize(families=FAMILIES, device='cpu', batch=8, output=None, submission=None, catboost_root=None, variant='both'):
    import torch
    from ops.predict_r16_production_nn import guard_state,read_raw_csv
    from ops.finalize_r16_production import write_submission,_stats,descriptive_comparison
    from ops.r16_support_temporal_data import load_panel,load_long_panel
    from ops.r16_forward_data import OPERATORS
    configure_variant(variant)
    output=FINAL if output is None else output;submission=SUBMISSION if submission is None else submission
    if device not in ('cpu','mps','cuda') or not 1<=batch<=32:raise ValueError('Bounded explicit CPU/MPS/CUDA inference required')
    if device=='mps' and (not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0'):raise RuntimeError('Available explicit MPS/fallback0 required')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Available explicit CUDA required')
    state,audit=guard_state();sys.addaudithook(audit);torch.set_num_threads(2)
    families=tuple(families);pins,members,catroot=prerequisites(families,catboost_root)
    final=safe(output);submission=safe(submission)
    if final.exists() or submission.exists():raise FileExistsError('Fresh finalization/CSV only; never reuse partial output')
    final.mkdir(parents=True);began=time.perf_counter();fit=role(0,'D0','selection');y=fit['share'];ids=fit['sample_id'];ps={};cs={};experts={}
    split,split_ix=selection_split();ei=split_ix['ensemble_fit'];ci=split_ix['checkpoint_selection']
    try:
        for r in REGIMES:
            trees=[load(TREES/'trees'/r/f'fold_{f:02d}'/'selection.npz') for f in range(5)]
            a=[np.mean([aligned(d,ids,y,'share') for d in trees],0)];b=[np.mean([aligned(d,ids,y,'classifier_prob') for d in trees],0)]
            if catroot:
                pairs=[load(catroot/r/f'fold_{f:02d}'/'selection.npz') for f in range(5)]
                a.append(np.mean([aligned(d,ids,y,'share') for d in pairs],0));b.append(np.mean([aligned(d,ids,y,'classifier_prob') for d in pairs],0))
            for family in families:
                a.append(aligned(load(RUN/family/r/'regression/selection.npz'),ids,y,'prediction'))
                b.append(aligned(load(RUN/family/r/'classification/selection.npz'),ids,y,'prediction'))
            ps[r]=np.column_stack(a);cs[r]=np.column_stack(b)
            experts[r]=fit_policy(ps[r][ei],cs[r][ei],y[ei],members)
        policy={'format':'r16_pretrained_capped_independent_v1','member_order':members,'experts':experts,
            'D0_H0_weights':[.5,.5],'score_share_weight':0.,'LightGBM_minimum':.5,'each_neural_maximum':.25,
            'fit_role':'local_fit ensemble_fit201bg1206rows; PEFT ES uses separate199bg1194rows',
            'local_fit_split_sha256':sha(OPS/'LOCAL_FIT_SPLIT.json'),'masking_variant':VARIANT,
            'inference_masking':False,'numerical_policy':NUMERICAL_POLICY,'ensemble_fit_rows':len(ei),'ensemble_fit_unique_backgrounds':split['chosen']['ensemble_fit']['unique_backgrounds'],
            'checkpoint_selection_roles':{'new_PEFT':'local_fit checkpoint199bg1194rows; disjoint from ensemble201bg1206rows',
                'preserved_LightGBM5':'legacy early stopping on shared local_fit; reused frozen, no refit',
                'optional_CatBoost5':'own-fold CV, if all ten completed non-smoke pairs available'},
            'local_fit_selection_reuse_disclosure':'Legacy LightGBM rounds and new ensemble weights both used local_fit. Its metrics are development-only, never unbiased validation.',
            'fit_sha256':read(OUT/'manifest.json')['selection']['sha256'],'selected_families':list(families),
            'omitted_families':[f for f in FAMILIES if f not in families],'partial_family_candidate':set(families)!=set(FAMILIES),
            'neural_folds':[0],'Tree_folds':[0,1,2,3,4],'full_hybrid_5fold_OOF':False,
            'postprocessing':'clip [0,1] only','Test_or_locked_panels_used_for_policy':False}
        write(final/'policy.json',policy);pins[str((final/'policy.json').relative_to(ROOT))]=sha(final/'policy.json')
        save(final/'selection_predictions.npz',{'sample_id':ids,'target_share':y,
            'checkpoint_selection_row_indices':ci,'ensemble_fit_row_indices':ei,
            **{'share_'+r:ps[r] for r in REGIMES},**{'classifier_'+r:cs[r] for r in REGIMES}})
        pins[str((final/'selection_predictions.npz').relative_to(ROOT))]=sha(final/'selection_predictions.npz')
        write(final/'MODEL_REGISTRY.json',{'status':'completed','artifact_sha256':pins,'member_order':members})
        freeze=final/'GLOBAL_FREEZE.json';write(freeze,{'status':'completed','all_models_frozen':True,'ensemble_policy_frozen':True,
            'data_manifest_sha256':sha(PARENT),'artifact_sha256':{str((final/'MODEL_REGISTRY.json').relative_to(ROOT)):sha(final/'MODEL_REGISTRY.json'),**pins},'Test_used':False})
        frozen=read(freeze)['artifact_sha256'];verify(frozen);predictor=Predictor(families,catroot,device,batch);rows=[]
        def one(name,d,weights=None):
            verify(frozen);s,c=predictor.predict(d['db']);e=apply(policy,s,c)
            save(final/'evaluation'/f'{name}.npz',{'sample_id':d['sample_id'],'target_share':d['share'],**e,
                **{'share_'+r:s[r] for r in REGIMES},**{'classifier_'+r:c[r] for r in REGIMES}})
            for r in REGIMES:
                for j,k in enumerate(members):rows.append({'panel':name,'model':r+'/'+k,**metric(d['share'],s[r][:,j],c[r][:,j],weights)})
            row={'panel':name,'model':'blend',**metric(d['share'],e['share'],e['score'],weights)}
            if 'site' in d:
                row['site_metrics']={str(site):metric(d['share'][d['site']==site],e['share'][d['site']==site],e['score'][d['site']==site]) for site in np.unique(d['site'])}
                row['site_macro_RMSE']=float(np.mean([v['rmse'] for v in row['site_metrics'].values()]))
            rows.append(row);write(final/'progress.json',{'phase':'frozen_evaluation','latest':row});print(json.dumps(row),flush=True)
        local=apply(policy,ps,cs);rows.append({'panel':'local_fit_ensemble_subset','model':'blend','development_only':True,**metric(y[ei],local['share'][ei],local['score'][ei])})
        for name in ('local_gate','local_report','domain_holdout'):
            parts=[load_panel(PARENT,name,op,freeze_receipt=freeze) for op in OPERATORS];keys=set.intersection(*(set(d) for d in parts))
            one(name,{k:np.concatenate([d[k] for d in parts]) for k in keys})
        pos=[load_long_panel(PARENT,'native12_positive',operator=op,freeze_receipt=freeze) for op in OPERATORS];keys=set.intersection(*(set(d) for d in pos));pos={k:np.concatenate([d[k] for d in pos]) for k in keys}
        for name in ('hf_high_negative','st_francis_dense_negative'):
            neg=load_long_panel(PARENT,name,freeze_receipt=freeze);keys=set(pos)&set(neg)
            d={k:np.concatenate([pos[k],neg[k]]) for k in keys};weights=np.r_[np.full(len(pos['share']),.5/len(pos['share'])),np.full(len(neg['share']),.5/len(neg['share']))]
            one('native12_VS_'+name,d,weights)
        write(final/'EVALUATION.json',{'status':'completed','rows':rows,'full_hybrid_5fold_OOF':False,'policy_reselected_after_locked_evaluation':False})
        verify(frozen);query=(ROOT/'data/raw/Test.csv').resolve();state['allowed_query']=query
        ids,db,qreceipt=read_raw_csv(query);s,c=predictor.predict(db);e=apply(policy,s,c)
        receipt=write_submission(submission,ids,e);save(final/'query_predictions.npz',{'sample_id':ids,**e,**{'share_'+r:s[r] for r in REGIMES},**{'classifier_'+r:c[r] for r in REGIMES}})
        stats={'ensemble':{k:_stats(v) for k,v in e.items()},'historical_comparison':descriptive_comparison(ids,e),
            'query_receipt':qreceipt,'submission':receipt,'model_statistics':{r:{k:{'share':_stats(s[r][:,j]),'independent_score':_stats(c[r][:,j])} for j,k in enumerate(members)} for r in REGIMES},
            'Test_labels_used':False,'policy_changed_after_query':False}
        write(final/'TEST_PREDICTION_AUDIT.json',stats);verify(frozen)
        text=['# R16 public pretrained Fold0 + Tree5 candidate','',f"Masking variant: {VARIANT}; inference masking: off.",f"Families: {', '.join(families)}. Missing: {', '.join(policy['omitted_families']) or 'none'}.",'',
            'New CUDA BF16 PEFT checkpoints use local_fit199backgrounds; numerical losses and Audio2D classifier pooling are FP32; separate201backgrounds choose capped weights. Input masking is training-only; inference is full unmasked context. Preserved LightGBM5 historically used local_fit for early stopping, so local_fit metrics are development-only. D0:H0=50:50, LightGBM≥50%, each neural family≤25%. Score uses independent classifier probabilities only. No Test calibration or official upload.','',
            '| Panel | RMSE | Independent score AUC | Bias |','|---|---:|---:|---:|']
        for r in rows:
            if r['model']=='blend':text.append(f"| {r['panel']} | {r['rmse']:.6f} | {r['classifier_auc']:.6f} | {r['bias']:.6f} |")
        text+=['',f'[Candidate CSV]({os.path.relpath(submission,final)})','', '[Policy](policy.json) · [Evaluation](EVALUATION.json) · [Prediction audit](TEST_PREDICTION_AUDIT.json)', '',
            'Only neural Fold0 has been trained; this is not a five-fold hybrid OOF result. Transfer benefit and Public score remain unmeasured.']
        (final/'RESULTS.md').write_text('\n'.join(text)+'\n');(final/'index.html').write_text('<meta charset="utf-8"><pre>'+html.escape('\n'.join(text))+'</pre>')
        result={'status':'completed','submission':receipt,'elapsed_seconds':time.perf_counter()-began,'official_upload':False,
            'masking_variant':VARIANT,'local_fit_split_sha256':sha(OPS/'LOCAL_FIT_SPLIT.json'),
            'policy_sha256':sha(final/'policy.json'),'partial_family_candidate':policy['partial_family_candidate'],'full_hybrid_5fold_OOF':False}
        write(final/'COMPLETED.json',result);return result
    except BaseException as exc:
        write(final/'FAILED.json',{'status':'failed','error':repr(exc),'automatic_partial_reuse':False});raise


def fixture(output):
    rng=np.random.default_rng(1601);n=120;y=rng.uniform(.01,.8,n);y[::5]=0;members=['lightgbm5',*FAMILIES]
    x=np.column_stack([np.clip(y+rng.normal(0,.12,n),0,1) for _ in members]);x[:,1]=y
    c=np.column_stack([np.clip((y>0)*.7+rng.uniform(0,.2,n),0,1) for _ in members])
    a=fit_policy(x,c,y,members);p={'member_order':members,'experts':{r:a for r in REGIMES},'D0_H0_weights':[.5,.5],'score_share_weight':0.}
    q=apply(p,{r:x for r in REGIMES},{r:c for r in REGIMES});q2=apply(p,{r:1-x for r in REGIMES},{r:c for r in REGIMES})
    assert np.array_equal(q['score'],q2['score']);assert a['regression_weights'][0]>=.5-1e-7;assert max(a['regression_weights'][1:])<=.25+1e-7
    rejected=False
    try:vector(np.zeros((n,1)),n)
    except ValueError:rejected=True
    assert rejected
    for count in (1,2,3):
        mm=['lightgbm5',*FAMILIES[:count]];pp=fit_policy(x[:,:count+1],c[:,:count+1],y,mm)
        assert pp['regression_weights'][0]>=.5-1e-7 and max(pp['regression_weights'][1:])<=.25+1e-7
    mm=['lightgbm5','catboost5',*FAMILIES];xx=np.column_stack([x[:,0],x[:,0],x[:,1:]]);cc=np.column_stack([c[:,0],c[:,0],c[:,1:]])
    pp=fit_policy(xx,cc,y,mm);assert pp['regression_weights'][0]>=.5-1e-7 and max(pp['regression_weights'][2:])<=.25+1e-7
    incomplete=False
    with tempfile.TemporaryDirectory(prefix='.peft-finalizer-fixture-',dir=ROOT/'reports') as temp:
        write(Path(temp)/'summary.json',{'status':'partial'})
        try:add_completed(Path(temp),{})
        except ValueError:incomplete=True
    assert incomplete
    original_variant=VARIANT or 'both'
    paths={}
    for variant in ('both',):
        configure_variant(variant)
        assert RUN == RUN_ROOT/variant and FINAL == RUN_ROOT/variant/'final'
        assert SUBMISSION == ROOT/'submissions/r16_pretrained_cuda_bf16_r1'/variant/'submission_r16_pretrained_cuda_bf16.csv'
        paths[variant]={'models':str(RUN.relative_to(ROOT)),'submission':str(SUBMISSION.relative_to(ROOT))}
    configure_variant(original_variant)
    config=FeatureMaskingConfig.for_variant(VARIANT).state_dict();split_hash='fixture-split-sha'
    identity={'variant':VARIANT,'masking_config':config,'local_fit_split_sha256':split_hash}
    check_mask_identity(identity,variant=VARIANT,config=config,split_sha256=split_hash)
    rejected_metadata=[]
    for field,bad in (('variant','not-the-selected-variant'),('masking_config',{}),('local_fit_split_sha256','wrong-split')):
        other={**identity,field:bad}
        try:check_mask_identity(other,variant=VARIANT,config=config,split_sha256=split_hash)
        except ValueError:rejected_metadata.append(field)
    assert len(rejected_metadata)==3
    fake_plan={'numerical_policy':NUMERICAL_POLICY,'masking_variants':{VARIANT:config},'local_fit_split_sha256':split_hash,
               'source_sha256':{'fixture':'sha'},'data_manifest_sha256':'fixture-data'}
    fake_summary={**identity,'numerical_policy':NUMERICAL_POLICY,'best_epoch':5,'best_cv':.123}
    fake_cp={**identity,'numerical_policy':NUMERICAL_POLICY,'format':'r16_pretrained_adapter_checkpoint_v1','family':FAMILIES[0],
             'regime':'D0','task':'regression','epoch':5,'best_epoch':5,'best_cv':.123,
             'source_sha256':fake_plan['source_sha256'],'data_manifest_sha256':'fixture-data'}
    check_checkpoint(fake_cp,fake_summary,FAMILIES[0],'D0','regression',fake_plan)
    bad_best=False
    try:check_checkpoint({**fake_cp,'epoch':4},fake_summary,FAMILIES[0],'D0','regression',fake_plan)
    except ValueError:bad_best=True
    assert bad_best
    out=safe(output)
    if out.exists():raise FileExistsError('Preserve existing fixture receipt')
    result={'status':'passed','caps_with_perfect_neural':a['regression_weights'],'score_independent_of_share':True,
        'nonvector_rejected':True,'models_loaded':False,'Test_or_locked_inputs_opened':False,'script_sha256':sha(__file__)}
    result.update(incomplete_model_rejected_before_labels_or_query=incomplete,
                  explicit_1_2_3_family_subsets_and_optional_catboost_caps_passed=True,
                  variant_model_and_submission_paths=paths,wrong_variant_mask_split_rejected=rejected_metadata,
                  selected_checkpoint_epoch_mismatch_rejected=bad_best,
                  policy_fit_role='ensemble_fit201bg1206rows only',
                  historical_Tree_local_fit_reuse_disclosed=True)
    write(out,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('fixture','run'))
    parser.add_argument('--variant',choices=('both',),default='both');parser.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES));parser.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    parser.add_argument('--batch',type=int,default=8);parser.add_argument('--output',type=Path);parser.add_argument('--submission',type=Path)
    parser.add_argument('--catboost-root',type=Path);args=parser.parse_args()
    configure_variant(args.variant)
    result=(fixture(args.output or REPORT/'FINALIZER_FIXTURE.json') if args.action=='fixture' else
        finalize(tuple(args.families),args.device,args.batch,args.output,args.submission,args.catboost_root,args.variant))
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
