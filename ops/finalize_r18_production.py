#!/usr/bin/env python3
"""Phase-separated R18 inference/finalization; no new training or K-fold."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name,'2')
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline import r18_production_finalization as f

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['plan','prepare-panels','collect-tree','collect-neural','collect-legacy-tree','collect-legacy-neural','collect-legacy-saved','select','report','prepare-query','export'])
    ap.add_argument('--input-root',type=Path,default=ROOT,help='Portable project/data/output root; uploaded compact prediction files retain relative paths')
    ap.add_argument('--stage',choices=['development','report','query'],default='development')
    ap.add_argument('--tree-root',type=Path)
    ap.add_argument('--neural-root',type=Path)
    ap.add_argument('--legacy-root',type=Path)
    ap.add_argument('--device',choices=['cpu','mps','cuda'],default='cuda')
    ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--test-path',type=Path)
    a=ap.parse_args();r=a.input_root.resolve()
    # Read-only audit guard: raw Test only in prepare-query after policy checks.
    def guard(event,args):
        if event=='open'and args and isinstance(args[0],(str,bytes,os.PathLike)):
            n=Path(os.fsdecode(args[0])).name.lower()
            if n in('test.csv','sample_submission.csv')and a.phase!='prepare-query':raise PermissionError('Raw Test is locked outside prepare-query')
            if n=='local_report.npz'and a.stage!='report'and a.phase not in('report','export'):raise PermissionError('Report is locked before frozen decision')
    sys.addaudithook(guard)
    if a.phase=='plan':v=f.plan(r)
    elif a.phase=='prepare-panels':v=f.prepare_panels(r,a.stage)
    elif a.phase=='collect-tree':v=f.collect_tree(r,a.stage,a.tree_root)
    elif a.phase=='collect-neural':v=f.collect_neural(r,a.stage,a.neural_root,a.device,a.batch)
    elif a.phase=='collect-legacy-tree':v=f.collect_legacy_tree(r,a.stage,a.legacy_root)
    elif a.phase=='collect-legacy-neural':v=f.collect_neural(r,a.stage,a.neural_root,a.device,a.batch,old=True)
    elif a.phase=='collect-legacy-saved':v=f.collect_legacy_saved(r,a.stage,a.legacy_root)
    elif a.phase=='select':v=f.select(r,a.tree_root,a.neural_root)
    elif a.phase=='report':v=f.report(r)
    elif a.phase=='prepare-query':v=f.prepare_query(r,a.test_path)
    else:v=f.export(r)
    print(json.dumps(v if v is not None else{'status':'phase_completed','phase':a.phase},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
