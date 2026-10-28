"""Build fresh R12 training/validation data, never fit a model or open Test."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.dataset_r12 import build_r12, numeric_fixture


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='data/synthetic/r12_cpu_ablation_v3')
    p.add_argument('--fixture-only', action='store_true')
    a = p.parse_args()
    fixture = numeric_fixture()
    print(json.dumps({'numeric_fixture': fixture}), flush=True)
    if not a.fixture_only:
        m = build_r12(a.output)
        print(json.dumps({'status': 'completed', 'output': a.output, 'data_hash': m['data_hash'],
            'profiles': {arm: v['train']['profile'] for arm, v in m['arms'].items()}}), flush=True)


if __name__ == '__main__':
    main()
