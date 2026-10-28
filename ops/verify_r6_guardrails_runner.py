"""Separate-exec numeric CNN/tree runner fixtures; never opens source datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/r6_cpu_guardrails_v2.json'
PINNED=['pipeline/cpu_ablation_r6_v2.py','pipeline/relative_grid_r6.py',
        'pipeline/relative_grid_r6_v2.py','pipeline/relative_models_r6_v2.py',
        'pipeline/physics_features_r6_v2.py','pipeline/mixtures.py',
        'pipeline/background_prior_features.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_shas(data):
    result={}
    for key,value in data.items():
        x=np.ascontiguousarray(value)
        h=hashlib.sha256(json.dumps([str(x.dtype),list(x.shape)]).encode());h.update(x.tobytes())
        result[key]=h.hexdigest()
    return result


def generated_data(n,seed):
    r=np.random.default_rng(seed)
    b=np.exp(r.normal(0,.3,(n,15,25)))
    spectral=np.exp(np.linspace(.8,-.4,25))[None,None,:]
    temporal=np.maximum(0,np.sin(np.linspace(-.3,3.5,15)))[None,:,None]
    a=r.uniform(.05,2.,(n,1,1))*spectral*temporal
    a[np.arange(n)%4==0]=0
    observed=np.ones((n,15,25),bool);observed[::3,0]=False;observed[1::5,4:6]=False
    b=np.where(observed,b,0.);a=np.where(observed,a,0.);e=a+b
    q=np.divide(a,e,out=np.zeros_like(e),where=observed)
    raw=np.full_like(e,np.nan);raw[observed]=100.+10*np.log10(e[observed])
    y=a.sum((1,2))/e.sum((1,2))
    data={'db':np.round(raw,1).astype('float32'),'unrounded_db':raw,
          'cell_share':q,'share':y,'presence':(y>0).astype('float32'),
          'site':np.asarray(['numeric_site_'+str(i%3) for i in range(n)]),
          'family':np.asarray(['numeric_fixture']*n)}
    for x in data.values():x.setflags(write=False)
    return data


def deny_source_reads():
    roots=[(ROOT/p).resolve() for p in ['data','data.nosync']]
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if any(p==q or q in p.parents for q in roots):raise PermissionError('Numeric fixture forbids source dataset I/O')
    sys.addaudithook(audit)


def child(kind,mode,folder,output):
    deny_source_reads()
    from pipeline.cpu_ablation_r6_v2 import fit_cnn,fit_tree,evaluate_frozen
    cfg=json.loads(CONFIG.read_text());cfg['neural']['epochs']=2;cfg['tree']['maximum_trees']=10
    # Production hyperparameters otherwise unchanged; these tiny fits are not models.
    arm={'name':'numeric_'+kind+'_'+mode,'generator':'numeric_fixture','channels':mode,'global_context':True}
    train=generated_data(96,101);inner=generated_data(36,202)
    before={'train':array_shas(train),'inner':array_shas(inner)}
    started=time.perf_counter()
    summary=(fit_cnn if kind=='cnn' else fit_tree)(train,inner,arm,cfg,folder)
    measured,pred=evaluate_frozen(folder,inner,kind)
    with np.load(folder/'inner_predictions.npz',allow_pickle=False) as z:
        reload_error={k:float(np.max(abs(pred[k]-z[k]))) for k in z.files}
    assert max(reload_error.values())<1e-7
    e=pred['share'].astype(float)-inner['share'];sites=inner['site']
    macro=float(np.mean([np.sqrt(np.mean(e[sites==s]**2)) for s in sorted(set(sites))]))
    assert abs(macro-summary['best_inner_metrics']['site_macro_rmse'])<1e-12
    assert not summary['heldout_evaluated'] and summary['pilot_only']
    tamper=False
    if mode=='legacy8':
        p=folder/'legacy_train_prior.json';original=p.read_bytes();p.write_bytes(original+b'\n')
        try:evaluate_frozen(folder,inner,kind)
        except ValueError as exc:
            assert 'prior changed' in str(exc);tamper=True
        finally:p.write_bytes(original)
        assert sha(p)==summary['prior_sha256'] and tamper
    else:
        p=folder/'legacy_train_prior.json';p.write_text('{}')
        try:evaluate_frozen(folder,inner,kind)
        except ValueError as exc:
            assert 'Unexpected prior' in str(exc);tamper=True
        finally:p.unlink()
        assert tamper
    score_error=None
    if kind=='cnn':
        from pipeline.cpu_ablation_r6_v2 import grid
        _,weights=grid(inner['db'],mode,
            json.loads((folder/'legacy_train_prior.json').read_text()) if mode=='legacy8' else None)
        oracle=[]
        for q,w in zip(pred['cell_mask'],weights):
            contrasts=[]
            for f in range(10,23):
                v=q[w[:,f]>0,f]
                if len(v):contrasts.append(float(v.max()-v.min()))
            oracle.append(.5*float((q*w).sum())+.5*max(contrasts,default=0.))
        score_error=float(np.max(abs(pred['score']-np.clip(oracle,1e-6,1-1e-6))))
        assert score_error<2e-7
        assert summary['epochs_run']==2
    else:
        assert 'score_classifier' in pred
        assert np.array_equal(pred['score'],.5*pred['share']+.5*pred['score_classifier'])
        assert summary['maximum_trees']==10
    after={'train':array_shas(train),'inner':array_shas(inner)}
    assert before==after
    torch_loaded=any(k=='torch' or k.startswith('torch.') for k in sys.modules)
    if kind=='tree':assert not torch_loaded
    result={'status':'passed','fixture_only':True,'kind':kind,'mode':mode,
            'train_rows':96,'inner_rows':36,'actual_source_data_used':False,
            'numeric_epochs_override':2,'numeric_maximum_trees_override':10,
            'other_production_settings_unchanged':True,'reload_max_abs':reload_error,
            'independent_site_macro_metric_match':True,'prior_tamper_or_unexpected_prior_rejected':tamper,
            'fixed_score_oracle_max_abs':score_error,'input_arrays_unchanged':True,
            'Torch_loaded':torch_loaded,'elapsed_seconds':time.perf_counter()-started,
            'summary_sha256':sha(folder/'summary.json'),
            'summary':{k:summary[k] for k in ['status','selection','pilot_only','heldout_evaluated']},
            'numeric_outputs_production_reuse_forbidden':True}
    output.write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--child-kind',choices=['cnn','tree']);p.add_argument('--mode',choices=['relative8','legacy8'])
    p.add_argument('--folder');p.add_argument('--child-output')
    p.add_argument('--output',default='reports/R6_V2_GUARDRAILS_RUNNER_FIXTURE.json')
    args=p.parse_args()
    if args.child_kind:
        if not all([args.mode,args.folder,args.child_output]):raise ValueError('Complete numeric child arguments required')
        child(args.child_kind,args.mode,Path(args.folder),Path(args.child_output));return
    output=Path(args.output)
    if output.exists():raise FileExistsError('Fresh output report required')
    pinned={n:sha(ROOT/n) for n in PINNED};cfgsha=sha(CONFIG);results=[];started=time.perf_counter()
    temp=Path(tempfile.mkdtemp(prefix='r6v2-runner-numeric-',dir='/private/tmp'))
    for kind in ['cnn','tree']:
        for mode in ['relative8','legacy8']:
            name=kind+'_'+mode;child_out=temp/(name+'.json');log=temp/(name+'.log')
            with log.open('x') as stream:
                proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child-kind',kind,'--mode',mode,
                                     '--folder',str(temp/name),'--child-output',str(child_out)],
                                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=90)
            if proc.returncode:
                raise RuntimeError('Numeric child failed; see '+str(log))
            results.append(json.loads(child_out.read_text()))
    assert pinned=={n:sha(ROOT/n) for n in PINNED} and cfgsha==sha(CONFIG)
    result={'format':'r6_v2_separate_cpu_runner_numeric_fixture_v1','status':'passed','fixture_only':True,
            'formal_model_fit_performed':False,'numeric_cpu_fixture_fit_performed':True,
            'actual_source_arrays_or_Test_OUTER_accessed':False,'GPU_SSH_external_submission_used':False,
            'source_code_sha256':pinned,'config_sha256':cfgsha,'fixture_code_sha256':sha(__file__),
            'core_code_config_unchanged_during_fixture':True,'separate_fresh_exec_per_arm':True,
            'temporary_numeric_artifacts':str(temp),'numeric_artifacts_must_not_be_reused':True,
            'cases':results,'elapsed_seconds':time.perf_counter()-started}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    output.with_suffix('.md').write_text('# R6 v2 runner numeric fixture\n\n'
        '**PASS, numeric fixture only.** Four fresh Python execs independently fit 96 generated TRAIN and 36 generated INNER '
        'numbers: relative8/legacy8 CNN (2 epochs) and LightGBM (10 maximum trees). No competition arrays were opened; '
        'an audit hook rejects source-data I/O. Core source/config files remained unchanged.\n\n'
        'Both tree processes remained Torch-free. Saved prediction replay, independently recomputed site-macro RMSE, '
        'TRAIN-prior tamper rejection, unexpected-prior rejection, immutable inputs and fixed-score formulas all passed. '
        'Temporary numeric weights under /private/tmp are not production models and must never be reused.\n\n'
        'This fixture verifies the CPU utility, not the data split, hard-negative source construction, outer holdout validity '
        'or final source-driver freeze policy. Those require separate source/role audits.\n\n'
        f'[Detailed JSON]({output.name})\n')
    print(json.dumps({'status':'passed','report':str(output),'report_sha256':sha(output),
                      'elapsed_seconds':result['elapsed_seconds'],'cases':len(results)}))


if __name__=='__main__':main()
