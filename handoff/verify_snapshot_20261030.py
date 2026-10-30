"""Read-only public snapshot checks. No data download, training or model import."""
from pathlib import Path
import ast,csv,hashlib,json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'handoff/PUBLIC_SNAPSHOT_20261030.json').read_text())
bad=[];syntax=[];rows=0
for rel,rec in manifest['files'].items():
 p=root/rel
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=rec['sha256']:bad.append(rel);continue
 if p.suffix=='.py':
  try:ast.parse(p.read_bytes(),filename=rel)
  except SyntaxError:syntax.append(rel)
 if rel.startswith('submissions/') and p.suffix=='.csv':
  with p.open(encoding='utf-8-sig',newline='') as f:a=list(csv.DictReader(f))
  if len(a)!=1500 or len({r['ID'] for r in a})!=1500:bad.append(rel+':rows/IDs')
  for r in a:
   for c in ('share_added_vessel','score_added_vessel'):
    if not 0<=float(r[c])<=1:bad.append(rel+':range')
  rows+=len(a)
print(json.dumps({'files':len(manifest['files']),'hash_or_CSV_errors':bad,'syntax_errors':syntax,'prediction_rows_checked':rows,'training':False},ensure_ascii=False))
raise SystemExit(bool(bad or syntax))
