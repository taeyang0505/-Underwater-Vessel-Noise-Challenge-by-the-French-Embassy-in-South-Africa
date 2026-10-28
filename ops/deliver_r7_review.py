"""Add R7 review artifacts to the approved Desktop copy, preserving IDE edits."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def links(path):
    t = path.read_text()
    values = re.findall(r'(?:href|src)="([^"]+)"', t) if path.suffix == '.html' else re.findall(r'\]\(([^)]+)\)', t)
    for value in values:
        if '://' in value or value.startswith('#'):
            continue
        value = re.sub(r':\d+$', '', value.split('#')[0])
        if value:
            yield (path.parent / value).resolve()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--destination', type=Path, required=True)
    target = p.parse_args().destination.resolve()
    if not target.is_dir() or target == ROOT:
        raise ValueError('Existing independent Desktop copy required')
    names = {'ops/audit_r5_submission_replay_r1.py'}
    for pattern in ['ops/*r7*.py', 'configs/r7*.json', 'reports/R7*']:
        for item in ROOT.glob(pattern):
            if item.is_file(): names.add(str(item.relative_to(ROOT)))
            elif item.is_dir(): names.update(str(v.relative_to(ROOT)) for v in item.rglob('*') if v.is_file())
    for folder in ['models/r7_fixed100_transfer_v1', 'models/r7_small_model_controls_v1', 'audit/r7_uploaded_csv_v1', 'reports/r7_review_figures']:
        names.update(str(v.relative_to(ROOT)) for v in (ROOT/folder).rglob('*') if v.is_file())
    # Original audit dependencies are copied only if absent or identical. No edits are overwritten.
    plan = json.loads((ROOT/'reports/R7_R5_SUBMISSION_PATH_AUDIT_PLAN.json').read_text())
    for relative, digest in plan['original_core17_expected_sha256'].items():
        assert sha(ROOT/relative) == digest
    for absolute in plan['input_file_sha256']:
        path = Path(absolute).resolve()
        if path.is_relative_to(ROOT): names.add(str(path.relative_to(ROOT)))
    for n in list(names):
        path = ROOT/n
        if path.suffix in {'.md', '.html'}:
            for dep in links(path):
                if not dep.is_relative_to(ROOT) or not dep.is_file():
                    raise ValueError('Invalid local report link: '+str(dep))
                if not (target/dep.relative_to(ROOT)).exists(): names.add(str(dep.relative_to(ROOT)))
    known = {}
    base = target/'DESKTOP_PACKAGE_MANIFEST.json'
    if base.exists(): known.update({v['path']:v['sha256'] for v in json.loads(base.read_text())['files']})
    for n, key in [('R6_GUARDRAILS_DESKTOP_DELIVERY.json','file_hashes'), ('R6_FIXED100_DESKTOP_DELIVERY.json','files')]:
        record = target/'reports'/n
        if record.exists(): known.update({name:v['sha256'] for name,v in json.loads(record.read_text())[key].items()})
    for n in sorted(names):
        source, dest = ROOT/n, target/n
        if dest.is_symlink(): raise ValueError('Preserve Desktop symlink: '+n)
        if dest.exists() and sha(dest) != sha(source) and known.get(n) != sha(dest):
            raise ValueError('Preserve IDE edit: '+n)
    records = {}
    copied = 0
    for n in sorted(names):
        source, dest = ROOT/n, target/n
        digest = sha(source)
        if not dest.exists() or sha(dest) != digest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            copied += 1
        assert sha(dest) == digest
        records[n] = {'sha256':digest, 'bytes':source.stat().st_size}
    checked_links = 0
    for n in names:
        path = target/n
        if path.suffix in {'.md','.html'}:
            for dep in links(path):
                assert dep.exists(), (n,str(dep))
                checked_links += 1
    out = {'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),'destination':str(target),
           'files_verified':len(records),'copied_or_updated':copied,'bytes_verified':sum(v['bytes'] for v in records.values()),
           'local_report_links_verified':checked_links,'original_core17_unchanged':True,'IDE_edits_preserved':True,'files':records}
    record = ROOT/'reports/R7_DESKTOP_DELIVERY.json'
    record.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    shutil.copyfile(record,target/'reports'/record.name)
    print(json.dumps({k:v for k,v in out.items() if k != 'files'},ensure_ascii=False))


if __name__ == '__main__':
    main()
