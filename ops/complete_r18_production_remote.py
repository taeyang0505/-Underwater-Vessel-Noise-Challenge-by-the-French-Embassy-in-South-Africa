#!/usr/bin/env python3
"""Wait for immutable R18 jobs, collect development outputs, then stop safely.

No data generation/training, network access, raw Test access or Pod actions.
Run detached beside the training supervisor. This wrapper does not modify the
already frozen finalizer, source code, checkpoints or original R17 exports.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key,'2')
from pathlib import Path
import argparse,fcntl,json,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline import r18_production_finalization as f

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-root',type=Path,default=ROOT)
    ap.add_argument('--tree-root',type=Path)
    ap.add_argument('--neural-root',type=Path)
    ap.add_argument('--legacy-root',type=Path)
    ap.add_argument('--legacy-neural-root',type=Path)
    ap.add_argument('--device',choices=['cuda','mps','cpu'],default='cuda')
    ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--poll-seconds',type=int,default=30)
    ap.add_argument('--supervisor-pid',type=int)
    ap.add_argument('--through-select',action='store_true',help='Explicitly run fit/gate selection after all development predictions are complete')
    ap.add_argument('--verify-only',action='store_true',help='Verify development input hashes and isolated Tree/Torch factory imports; do not wait, infer or fit')
    args=ap.parse_args();root=args.input_root.resolve();folder=f.out(root)/'orchestration';folder.mkdir(parents=True,exist_ok=True)
    # Kept alive for the full wait/collect/select lifetime. The wrapper is not a
    # training source; adding this lock does not modify any frozen learner.
    orchestration_lock=(folder/'orchestration.lock').open('a+')
    try:fcntl.flock(orchestration_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('An R18 finalization orchestrator is already active')
    def input_guard(event,values):
        if event=='open'and values and isinstance(values[0],(str,bytes,os.PathLike)):
            name=Path(os.fsdecode(values[0])).name.lower()
            if name in('test.csv','sample_submission.csv','local_report.npz','query.npz','inputs_query.json'):
                raise PermissionError('Report/query inputs are forbidden in development orchestration')
    sys.addaudithook(input_guard)
    tree=args.tree_root or root/'models/r18_production_r1/trees'
    neural=args.neural_root or root/'models/r18_production_r1/neural'
    legacy=args.legacy_root or root
    old_nn=args.legacy_neural_root or legacy/'models/r16_pretrained_cuda_bf16_r1/both'
    finalizer=ROOT/'ops/finalize_r18_production.py'
    base=[sys.executable,str(finalizer),'--input-root',str(root)]
    # argparse requires the positional phase before these shared arguments.
    def command(phase,extra=()):return[sys.executable,str(finalizer),phase,'--input-root',str(root),*map(str,extra)]
    def event(**value):
        value.update(utc=f.utc(),pid=os.getpid());f.write(folder/'progress.json',value)
        with(folder/'events.jsonl').open('a')as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')
        print(json.dumps(value,ensure_ascii=False),flush=True)
    def alive(pid):
        try:
            os.kill(pid,0)
            stat=Path(f'/proc/{pid}/stat')
            return not(stat.exists()and stat.read_text().split(') ',1)[1].startswith('Z'))
        except ProcessLookupError:return False
        except PermissionError:return True
    if args.verify_only:
        f.verify_plan(root)
        inputs={name:{'n':len(f.dataset(root,name)['db']),
                     'sha256':f.sha(f.out(root)/'inputs'/(name+'.npz'))}
                for name in f.names('development')}
        # Import numerical runtimes in separate children, matching production
        # collection. No models are trained or predictions produced here.
        guard_source="""import os,sys\nfrom pathlib import Path\ndef guard(event,args):\n if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)) and Path(os.fsdecode(args[0])).name.lower() in ('test.csv','sample_submission.csv','local_report.npz','query.npz','inputs_query.json'):raise PermissionError('Report/query locked')\nsys.addaudithook(guard)\n"""
        scripts={
            'tree':"from pipeline.trees_r18_production import schema\ns=schema()\nassert len(s['regression_relative'])==4667 and len(s['classification_profile_energy'])==2408\nprint('Tree schema/import passed')\n",
            'neural':"from pipeline import training_r18_production_neural as newapi\nfrom pipeline import training_r16_pretrained_cuda_bf16 as oldapi\nfrom pipeline.pretrained_audio2d_r16 import verify_audio2d_checkpoint\nfor family in newapi.FAMILIES:verify_audio2d_checkpoint(family)\nassert oldapi.CLASSIFIER_CHANNELS==(0,1,2,3,6,5,8,9,10,11,12,13,14,15,16)\nprint('New17 / legacy15 neural factory imports and public SHA passed')\n"}
        imports={}
        for kind,source in scripts.items():
            completed=subprocess.run([sys.executable,'-c',guard_source+source],cwd=ROOT,
                capture_output=True,text=True,check=True,env=os.environ.copy())
            imports[kind]={'status':'passed','stdout':completed.stdout.strip(),
                           'stderr':completed.stderr.strip()[-2000:]}
        receipt={'status':'passed','wrapper_sha256':f.sha(__file__),'PLAN_sha256':f.sha(f.out(root)/'PLAN.json'),
            'development_inputs':inputs,'isolated_runtime_imports':imports,
            'Test_or_report_read':False,'new_training':False,'inference_or_selection':False,'utc':f.utc()}
        f.write(folder/'VERIFY_ONLY.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
        return
    protocol={'PLAN_sha256':f.sha(f.out(root)/'PLAN.json'),'wrapper_sha256':f.sha(__file__),
        'finalizer_sha256':f.sha(finalizer),'model_roots':{'tree':str(tree),'neural':str(neural),'legacy_neural':str(old_nn)},
        'training_or_Pod_restart':False,'Test_or_report_read':False,'through_select':args.through_select}
    existing=folder/'PROTOCOL.json'
    if existing.exists()and f.read(existing)!=protocol:raise ValueError('Orchestration protocol differs; use a fresh namespace')
    f.write(existing,protocol);f.verify_plan(root)
    required=[tree/r/'summary.json'for r in f.REGIMES]+[neural/family/r/task/'summary.json'for family in f.FAMILIES[1:]for r in f.REGIMES for task in f.TASKS]
    while True:
        pending=[]
        for p in required:
            if not p.exists()or f.read(p).get('status')!='completed':pending.append(str(p))
        if not pending:break
        if args.supervisor_pid and not alive(args.supervisor_pid):
            nn_pending=[p for p in pending if str(neural)in p]
            if nn_pending:raise RuntimeError('Training supervisor exited before all neural completion receipts; no automatic restart')
        event(phase='waiting_completed_models',pending=len(pending),pending_paths=pending)
        time.sleep(max(1,min(args.poll_seconds,60)))
    f.completed_models(root,tree,neural)
    event(phase='all_production_models_verified',models=len(required))
    steps=[
        ('collect-tree',['--stage','development','--tree-root',tree],('new',('lightgbm',),f.names('development'))),
        ('collect-neural',['--stage','development','--neural-root',neural,'--device',args.device,'--batch',args.batch],('new',f.FAMILIES[1:],f.names('development'))),
        ('collect-legacy-tree',['--stage','development','--legacy-root',legacy],('old',('lightgbm',),('fresh_fit_D0','fresh_fit_H0'))),
        ('collect-legacy-neural',['--stage','development','--neural-root',old_nn,'--device',args.device,'--batch',args.batch],('old',f.FAMILIES[1:],('fresh_fit_D0','fresh_fit_H0'))),
        ('collect-legacy-saved',['--stage','development','--legacy-root',legacy],('old',f.FAMILIES,f.FIXED)),
    ]
    for phase,extra,identities in steps:
        generation,families,panels=identities
        ready=all(f.pred_path(root,generation,family,name).exists()for family in families for name in panels)
        if ready:
            for family in families:
                for name in panels:f.prediction(root,generation,family,name)
            event(phase=phase,status='verified_existing_outputs');continue
        event(phase=phase,status='running')
        log=folder/(phase+'.log')
        with log.open('a')as stream:
            subprocess.run(command(phase,extra),stdout=stream,stderr=subprocess.STDOUT,check=True,cwd=ROOT,env=os.environ.copy())
        for family in families:
            for name in panels:f.prediction(root,generation,family,name)
        event(phase=phase,status='completed',log_sha256=f.sha(log))
    if args.through_select:
        if not(f.out(root)/'DECISION_FREEZE.json').exists():
            event(phase='select',status='running')
            with(folder/'select.log').open('a')as stream:
                subprocess.run(command('select',['--tree-root',tree,'--neural-root',neural]),stdout=stream,stderr=subprocess.STDOUT,check=True,cwd=ROOT,env=os.environ.copy())
        f.stage_guard(root,'report')
        event(phase='development_and_selection_completed',next_step='Prepare local_report only now, transfer its compact input, collect new report predictions; then report. Raw Test remains absent/locked.')
    else:event(phase='development_completed',next_step='Explicit select command, then separate report and query phases. Raw Test remains absent/locked.')
    f.write(folder/'COMPLETED.json',{'status':'completed','through_select':args.through_select,'PLAN_sha256':protocol['PLAN_sha256'],
        'Test_used':False,'report_read':False,'new_training':False,'created_utc':f.utc()})

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'status':'failed','error':repr(exc),'automatic_restart':False}),flush=True)
        traceback.print_exc();raise
