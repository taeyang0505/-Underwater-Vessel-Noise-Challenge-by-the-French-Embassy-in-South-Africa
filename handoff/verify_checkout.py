"""Verify the published research snapshot with Python standard library only."""
from pathlib import Path
import ast, csv, hashlib, json, sys

def main():
 root=Path(__file__).resolve().parents[1]
 manifest=json.loads((root/'handoff/REPOSITORY_FILE_MANIFEST.json').read_text())
 errors=[]
 for e in manifest['files']:
  p=root/e['path']
  if not p.is_file():errors.append({'path':e['path'],'error':'missing'});continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1<<20),b''):h.update(b)
  if p.stat().st_size!=e['bytes'] or h.hexdigest()!=e['sha256']:
   errors.append({'path':e['path'],'error':'SHA/size differs from published snapshot'})
 py_count=0
 for p in root.rglob('*.py'):
  if '.git' in p.parts or '__pycache__' in p.parts:continue
  try:ast.parse(p.read_text(),filename=str(p));py_count+=1
  except (SyntaxError,UnicodeError) as e:errors.append({'path':str(p.relative_to(root)),'error':str(e)})
 p=root/'submissions/r16_multisupport_tree5_r1/submission_r16_tree5.csv'
 with p.open(newline='') as f:rows=list(csv.reader(f))
 if rows[0]!=['ID','share_added_vessel','score_added_vessel'] or len(rows)!=1501 or len({r[0] for r in rows[1:]})!=1500:
  errors.append({'path':str(p),'error':'Tree CSV schema/ID/rows differs'})
 if hashlib.sha256(p.read_bytes()).hexdigest()!='589bbee56ce9c8432aabd0da174ad040dd261410a84788293992cafd9d029490':
  errors.append({'path':str(p),'error':'Tree CSV SHA differs'})
 print(json.dumps({'status':'passed' if not errors else 'failed','verified_manifest_files':len(manifest['files']),
                   'python_AST_files':py_count,'Tree_CSV_rows':len(rows)-1,'errors':errors,
                   'does_not_verify_private_data_or_training_completion':True},ensure_ascii=False,indent=2))
 return bool(errors)

if __name__=='__main__':sys.exit(main())
