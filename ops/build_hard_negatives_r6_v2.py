"""Build only R6 TRAIN hard-negative source metadata, never train models."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline.hard_negatives_r6_v2 import candidates,fixture

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-version',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'data/versions/hard_negatives_r6_v2')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:print(json.dumps(fixture(),indent=2));return
    if a.source_version is None:p.error('--source-version is required; no implicit source version')
    m=candidates(a.source_version,a.output)
    print(json.dumps({'status':'PASS','candidate_version_hash':m['candidate_version_hash'],'audit':m['audit'],'output':str(a.output)},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
