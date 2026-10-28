"""Fixed100 control-flow test on numeric arrays and explicitly mocked metrics."""
from __future__ import annotations

import argparse
import copy
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
CONFIG=ROOT/'configs/r6_fixed100_v1.json'
SOURCES=['pipeline/fixed100_r6.py','ops/run_r6_fixed100.py','pipeline/cpu_ablation_r6_v2.py',
         'pipeline/relative_models_r6_v2.py','pipeline/relative_grid_r6_v2.py','pipeline/relative_grid_r6.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(n,seed):
    r=np.random.default_rng(seed);b=r.uniform(.5,1.5,(n,15,25));a=r.uniform(0.,1.,(n,15,25))
    a[::2]=0.;o=np.ones_like(b,dtype=bool);o[::2,0]=False;b[~o]=0.;a[~o]=0.;e=a+b
    raw=np.full_like(e,np.nan);raw[o]=100.+10*np.log10(e[o]);q=np.divide(a,e,out=np.zeros_like(e),where=o)
    return {'db':np.round(raw,1).astype('float32'),'unrounded_db':raw,'cell_share':q,
            'share':a.sum((1,2))/e.sum((1,2)),'presence':(a.sum((1,2))>0).astype('float32'),
            'site':np.array(['numeric_'+str(i%2) for i in range(n)])}


def deny_sources():
    roots=[(ROOT/n).resolve() for n in ['data','data.nosync']]
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if any(p==r or r in p.parents for r in roots):raise PermissionError('No source datasets in numeric fixture')
    sys.addaudithook(audit)


def child(case,folder,result_file):
    deny_sources()
    import torch
    from unittest.mock import patch
    import pipeline.fixed100_r6 as runner
    torch.set_num_threads(1)
    cfg=json.loads(CONFIG.read_text());cfg['neural'].update(width=4,blocks=1,threads=1,batch_size=4)
    assert cfg['neural']['epochs']==100 and cfg['neural']['early_stopping_enabled'] is False
    arm={'name':'numeric_fixed100_'+case,'channels':'relative8','generator':'numeric_fixture','global_context':True}
    train=numeric(4,31);inner=numeric(4,37)
    identity=lambda data:{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in data.items()}
    original={'train':identity(train),'inner':identity(inner)}
    nsteps=0;inner_calls=0;original_step=torch.optim.Adam.step;original_metrics=runner.metrics
    if case=='plateau':expected_window=expected_macro=1
    else:expected_window,expected_macro=27,73
    def mocked_metrics(data,pred):
        nonlocal inner_calls
        if data is not inner:return original_metrics(data,pred)
        inner_calls+=1
        epoch=inner_calls if inner_calls<=100 else [expected_window,expected_macro,100][inner_calls-101]
        out=copy.deepcopy(original_metrics(data,pred))
        if case=='plateau':window=macro=.5
        else:window=.5+1e-6*(epoch-27)**2;macro=.49+1e-6*(epoch-73)**2
        out.update(rmse=window,site_macro_rmse=macro,explicit_mock_metric_fixture=True)
        return out
    def step(self,*args,**kwargs):
        nonlocal nsteps
        nsteps+=1;return original_step(self,*args,**kwargs)
    started=time.perf_counter()
    with patch.object(runner,'metrics',side_effect=mocked_metrics),patch.object(torch.optim.Adam,'step',step):
        summary=runner.fit_fixed100(train,inner,arm,cfg,folder)
    assert nsteps==summary['optimizer_steps']==100 and inner_calls==103
    assert summary['epochs_run']==100 and not summary['early_stopping_enabled']
    assert summary['stop_reason']=='completed_100_epochs' and not summary['heldout_evaluated']
    hist=[json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines()]
    assert len(hist)==100 and [r['epoch'] for r in hist]==list(range(1,101))
    assert [r['optimizer_steps'] for r in hist]==list(range(1,101))
    checkpoints=sorted((folder/'checkpoints').glob('epoch_*.pt'))
    assert len(checkpoints)==100 and len(summary['epoch_checkpoint_sha256'])==100
    # Read every numeric checkpoint, not only the final counter.
    for epoch,p in enumerate(checkpoints,1):
        ck=torch.load(p,map_location='cpu',weights_only=True)
        assert ck['epoch']==epoch and sha(p)==summary['epoch_checkpoint_sha256'][p.name]
        assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
    for label,epoch in [('window',expected_window),('macro',expected_macro),('last',100)]:
        name={'window':'best_window.pt','macro':'best_macro.pt','last':'last.pt'}[label]
        assert sha(folder/name)==sha(folder/'checkpoints'/f'epoch_{epoch:03d}.pt')
        k={'window':'best_inner_window','macro':'best_inner_macro','last':'final_epoch'}[label]
        assert summary[k]['epoch']==epoch
        assert sha(folder/name)==summary['artifact_sha256'][name]
    state=torch.load(folder/'final_training_state.pt',map_location='cpu',weights_only=True)
    assert state['epoch']==100 and state['optimizer_steps']==100
    opt_steps=[float(v['step']) for v in state['optimizer_state_dict']['state'].values()]
    assert opt_steps and all(s==100 for s in opt_steps)
    assert state['scheduler_state_dict']['last_epoch']==100
    if case=='plateau':
        assert summary['shadow_ES7']['stop_epoch']==8 and summary['shadow_ES7']['best_macro_epoch']==1
        assert hist[-1]['next_learning_rate']==1e-6
    before=torch.load(checkpoints[0],weights_only=True,map_location='cpu')['state_dict']
    after=torch.load(checkpoints[-1],weights_only=True,map_location='cpu')['state_dict']
    assert any(not torch.equal(before[k],after[k]) for k in before)
    assert original=={'train':identity(train),'inner':identity(inner)}
    result={'status':'passed','fixture_only':True,'case':case,'validation_metrics_explicitly_mocked':True,
            'mock_metrics_are_not_performance_evidence':True,'train_rows':4,'inner_rows':4,
            'numeric_model_settings':{'width':4,'blocks':1,'threads':1,'batch_size':4},
            'actual_Adam_steps':nsteps,'actual_history_epochs':len(hist),'all_epoch_files_read_and_verified':len(checkpoints),
            'best_window_epoch':expected_window,'best_macro_epoch':expected_macro,'last_epoch':100,
            'shadow_ES7_stop_epoch':summary['shadow_ES7']['stop_epoch'],
            'shadow_ES_never_controls_training':True,'first_exact_tie_retained':case=='plateau',
            'last_learning_rate':hist[-1]['next_learning_rate'],'final_optimizer_state_steps':sorted(set(opt_steps)),
            'parameters_changed_between_epoch1_and100':True,'original_numeric_arrays_unchanged':True,
            'all_numeric_artifacts_production_reuse_forbidden':True,'elapsed_seconds':time.perf_counter()-started}
    result_file.write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['plateau','separate_minima'])
    p.add_argument('--folder');p.add_argument('--result-file');p.add_argument('--output',default='reports/R6_FIXED100_NUMERIC_FIXTURE.json')
    a=p.parse_args()
    if a.case:
        if not a.folder or not a.result_file:raise ValueError('Numeric child paths required')
        child(a.case,Path(a.folder),Path(a.result_file));return
    output=Path(a.output)
    if output.exists() or output.with_suffix('.md').exists():raise FileExistsError('Fresh report paths required')
    pin={n:sha(ROOT/n) for n in SOURCES};cfgsha=sha(CONFIG);start=time.perf_counter()
    tmp=Path(tempfile.mkdtemp(prefix='r6-fixed100-numeric-',dir='/private/tmp'));cases=[]
    for name in ['plateau','separate_minima']:
        dest=tmp/(name+'.json')
        with (tmp/(name+'.log')).open('x') as log:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--case',name,'--folder',str(tmp/name),
                                   '--result-file',str(dest)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=90)
        if result.returncode:raise RuntimeError('Numeric fixture failed; read '+str(tmp/(name+'.log')))
        cases.append(json.loads(dest.read_text()))
    assert pin=={n:sha(ROOT/n) for n in SOURCES} and cfgsha==sha(CONFIG)
    report={'format':'r6_fixed100_no_earlystop_numeric_fixture_v1','status':'passed','fixture_only':True,
            'formal_source_data_training_performed':False,'TRAIN_INNER_OUTER_Test_arrays_opened':False,
            'GPU_SSH_or_external_transmission_used':False,'source_code_sha256':pin,'config_sha256':cfgsha,
            'fixture_code_sha256':sha(__file__),'source_code_config_unchanged':True,
            'temporary_numeric_artifact_directory':str(tmp),'numeric_artifacts_production_reuse_forbidden':True,
            'cases':cases,'elapsed_seconds':time.perf_counter()-start}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    output.with_suffix('.md').write_text('# Fixed100 numeric control-flow fixture\n\n'
        '**PASS, numeric fixture only.** No source TRAIN, INNER, OUTER or Test arrays were opened. '
        'Two independent fresh processes each used four generated TRAIN and four generated INNER windows, '
        'a width4/block1 CNN and 100 actual Adam updates. Validation scalars were explicitly mocked for '
        'control-flow assertions and must never be cited as model performance.\n\n'
        'Constant validation plateau: 100 epochs/100 optimizer updates/100 saved epoch files, first-tie '
        'best checkpoints at epoch1, last checkpoint at100. Shadow ES7 would stop at8 but never stopped training. '
        'The scheduler still reduced learning rate to1e-6. Separate-minima case: best-window epoch27, '
        'best-macro epoch73, last100. Every numeric epoch checkpoint was loaded and its epoch/SHA checked; '
        'final optimizer steps and actual parameter updates were verified.\n\n'
        'This proves runner behavior, not a global minimum or unbiased validation. Repeated INNER selection '
        'remains optimistic, and previously viewed cross panels are not blind. CPU thread changes also prevent '
        'a bitwise prefix comparison with older trajectories. Numeric weights are temporary and forbidden '
        'for production reuse.\n\n'
        f'[Detailed JSON]({output.name})\n')
    print(json.dumps({'status':'passed','report':str(output),'report_sha256':sha(output),'cases':cases,
                      'elapsed_seconds':report['elapsed_seconds']}))


if __name__=='__main__':main()
