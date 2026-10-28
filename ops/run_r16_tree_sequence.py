"""One sequential fresh fivefold Tree execution; no neural/Query access."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline.trees_r16_production import sha,write
OUT=ROOT/'models/r16_production_trees_r1'

def main():
    if OUT.exists():raise FileExistsError('Fresh sequence only')
    OUT.mkdir()
    qa=json.loads((ROOT/'reports/R16_PRODUCTION_TREE_INDEPENDENT_QA.json').read_text())
    # Independent QA status is additionally inspected by the human-facing parent.
    pins={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),ROOT/'ops/train_r16_production_trees.py',ROOT/'pipeline/trees_r16_production.py',
          ROOT/'reports/R16_PRODUCTION_DATA_INDEPENDENT_QA.json',ROOT/'reports/R16_PRODUCTION_TREE_INDEPENDENT_QA.json',ROOT/'reports/R16_PRODUCTION_TREE_FIXTURE.json']}
    write(OUT/'SEQUENCE_PLAN.json',{'status':'frozen','folds':list(range(5)),'source_pins':pins,'CPU_threads':2,'regression_features':3827,'classification_features':700,
                                  'maximum_trees':1500,'ES':40,'Test_used':False,'started_unix':time.time()})
    try:
        for fold in range(5):
            for n,h in pins.items():
                if sha(ROOT/n)!=h:raise ValueError('Frozen sequence inputs changed')
            manifest=ROOT/f'data/synthetic/r16_production_v1/fold_{fold:02d}/manifest.json'
            command=[sys.executable,'ops/train_r16_production_trees.py','worker','--manifest',str(manifest),'--fold',str(fold),'--output',str(OUT)]
            write(OUT/'progress.json',{'status':'training','fold':fold,'completed_folds':list(range(fold)),'command':command})
            with (ROOT/f'ops/r16_production_tree_fold{fold}.log').open('x')as log:
                child=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                write(OUT/'active_child.json',{'PID':child.pid,'fold':fold,'command':command})
                code=child.wait()
            if code:raise RuntimeError(f'Fold{fold} failed exit{code}; no partial reuse')
        subprocess.run([sys.executable,'ops/train_r16_production_trees.py','freeze_all','--output',str(OUT)],cwd=ROOT,env=os.environ.copy(),check=True)
        write(OUT/'progress.json',{'status':'completed','completed_folds':list(range(5)),'all_models_frozen':True})
    except BaseException as e:
        write(OUT/'FAILED.json',{'status':'failed','error':repr(e),'no_automatic_restart':True});raise

if __name__=='__main__':main()
