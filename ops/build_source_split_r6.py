"""Build a fresh R6 source/QC manifest; no model fitting or legacy eval loading."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.source_split_r6 import build_source_split_r6,source_split_fixture

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'data/versions/source_split_r6_cpu_v1')
    p.add_argument('--seed',type=int,default=20261026)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        print(json.dumps(source_split_fixture(),indent=2));return
    m=build_source_split_r6(output_dir=a.output,seed=a.seed)
    print(json.dumps({'status':'PASS','source_hash':m['source_hash'],'roles':m['audit']['background_role_counts'],
        'references':m['audit']['reference_role_counts'],'output':str(a.output)},indent=2))

if __name__=='__main__':main()
