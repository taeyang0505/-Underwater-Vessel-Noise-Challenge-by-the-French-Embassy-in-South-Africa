"""Optional CatBoost probe / explicit fresh regime-fold CPU fit (no Test)."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline.catboost_r16_peft import dependency_status,fit_pair

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('probe')
    f=sub.add_parser('fit');f.add_argument('--fold',type=int,required=True,choices=range(5))
    f.add_argument('--regime',required=True,choices=('D0','H0'));f.add_argument('--output',type=Path,required=True)
    f.add_argument('--iterations',type=int,default=1500);f.add_argument('--smoke-limit-rows',type=int)
    a=p.parse_args()
    if a.action=='probe':result=dependency_status(True)
    else:result=fit_pair(a.output,fold=a.fold,regime=a.regime,iterations=a.iterations,limit_rows=a.smoke_limit_rows)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
