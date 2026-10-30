"""Narrower public export after review rejection of competition-artifact bundle.

Contains own code/configs, the requested README/status narrative and filesystem
size/environment metadata. No new predictions, acoustic matrices, metric CSVs,
report artifacts, waveforms or checkpoints are published by this export.
"""
from pathlib import Path
import ast, datetime, hashlib, json, re, sys
ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'reports/repository_sync_20261030'
FULL=AUDIT/'public_snapshot.nosync'
OUT=AUDIT/'public_code_only.nosync'
FILES={}

def put(rel,b,kind,source=None):
 if isinstance(b,str):b=b.encode()
 if Path(rel).suffix=='.py':ast.parse(b,filename=rel)
 p=OUT/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 FILES[rel]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),
  'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),
  'kind':kind,'source_sha256':source,'source_bytes_preserved':hashlib.sha256(b).hexdigest()==source if source else None}

def prepare():
 m=json.loads((FULL/'handoff/PUBLIC_SNAPSHOT_20261030.json').read_text())
 allow={'source','current_handoff','verification_utility','environment_record'}
 for rel,r in m['files'].items():
  if r['kind'] in allow:
   b=(FULL/rel).read_bytes()
   if hashlib.sha256(b).hexdigest()!=r['sha256']:raise ValueError('Full export changed')
   put(rel,b,r['kind'],r.get('source_sha256'))
 put('ops/prepare_public_code_only_snapshot_20261030.py',Path(__file__).read_bytes(),'source',hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 # Only the explicitly requested current status narrative, not report artifacts.
 readme=(ROOT/'README.md').read_text()
 def links(match):
  label,target=match.group(1),match.group(2)
  if target.startswith('reports/') and not target.startswith('reports/repository_sync_20261030/STORAGE_INVENTORY.md'):
   if target.endswith('/storage_inventory.csv'):return '['+label+'](reports/repository_sync_20261030/storage_inventory.json)'
   return label+' (`'+target+'`, 상세 원문은 로컬 보존)'
  return match.group(0)
 readme=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',links,readme)
 readme=readme.replace('과거 보고서의 가설·예정·학습 중 상태는 당시 기록이다.',
  '상세 R17~R21 보고서·집계 CSV·새 제출 예측 파일은 이번 공개에 포함하지 않았다. 아래 수치와 설명은 사용자가 요청한 현재 상태 요약이다. 과거 보고서의 가설·예정·학습 중 상태는 당시 기록이다.')
 put('README.md',readme,'readme',hashlib.sha256((ROOT/'README.md').read_bytes()).hexdigest())
 for rel in ('CLAUDE.md','handoff/CURRENT_STATE.md'):
  s=(OUT/rel).read_text()
  s=re.sub(r'\[([^\]]+)\]\((?:\.\./)?reports/[^)]+\)',r'\1 (상세 원문은 로컬 보존; 공개 README에 요약)',s)
  put(rel,s,'current_handoff')
 for rel in ('reports/repository_sync_20261030/STORAGE_INVENTORY.md','reports/repository_sync_20261030/storage_inventory.json'):
  b=(ROOT/rel).read_bytes();put(rel,b,'filesystem_size_metadata',hashlib.sha256(b).hexdigest())
 put('.gitignore',(FULL/'.gitignore').read_bytes(),'publication_scope')
 put('handoff/PUBLICATION_SCOPE_20261030.md','''# 이번 공개 범위 — 코드·요청한 상태 요약만

이번 갱신은 자체 코드/설정, 현재 README/인계 문서, 파일시스템 용량 및 환경 버전 정보만 포함한다. 새 제출 예측 CSV, 대회 파생 집계/진단 CSV·보고서, 음향 원자료/합성 행렬/체크포인트는 포함하지 않는다. 기존 저장소 역사 파일은 삭제하지 않았다.

전체 1,201파일·24,648,726bytes 자료 묶음은 로컬에 준비됐으나 자동 승인 검토가 예측값·대회 파생 자료의 외부 공개 범위 승인이 부족하다는 이유로 전송을 거절했다. 이를 작은 조각으로 나눠 전송하지 않고, 해당 자료를 모두 뺀 별도 코드/문서 묶음을 만들었다. 추가 공개가 승인되기 전까지 전체 묶음은 전송하지 않는다.

공개 clone 검증에는 PUBLIC_SNAPSHOT_20261030.json 및 verify_snapshot_20261030.py를 쓴다. 원본 데이터/동결 모델은 비공개 복원이 필요하고, 이전 model Release는 최신 PEFT/R18 전체 백업이 아니다. 보고서·모델·원자료의 재배포 권리는 소유자/원출처 조건을 별도로 따른다. 현재 no-training/CPU diagnostics 계약을 유지한다.
''','publication_scope')
 # No original full-artifact hashes/paths/predictions are needed for this scope.
 snapshot={'timestamp_KST':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
  'scope':'Own code/configs + explicitly requested status narrative + filesystem size/environment only',
  'competition_derived_artifacts_or_predictions_in_new_export':False,
  'files':FILES,'payload_bytes_excluding_manifest':sum(r['bytes'] for r in FILES.values()),
  'files_excluding_manifest':len(FILES),'new_training':False,'official_submission':False}
 b=(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n').encode()
 put('handoff/PUBLIC_SNAPSHOT_20261030.json',b,'public_manifest')
 (AUDIT/'CODE_ONLY_EXPORT_FILES.json').write_text(json.dumps(FILES,ensure_ascii=False,indent=2)+'\n')
 # Private transfer chunks. Files are reassembled before Git tree creation.
 chunks=[]
 for rel,r in sorted(FILES.items()):
  s=(OUT/rel).read_bytes().decode('utf-8')
  for n in range(0,len(s),16000):chunks.append({'path':rel,'offset':n,'last':n+16000>=len(s),'content':s[n:n+16000],'git_blob_sha1':r['git_blob_sha1'],'sha256':r['sha256']})
  if not s:chunks.append({'path':rel,'offset':0,'last':True,'content':'','git_blob_sha1':r['git_blob_sha1'],'sha256':r['sha256']})
 batches=[];a=[];size=0
 for c in chunks:
  z=len(json.dumps(c,ensure_ascii=False).encode())
  if a and size+z>100000:batches.append(a);a=[];size=0
  a.append(c);size+=z
 if a:batches.append(a)
 transport=AUDIT/'code_only_upload_batches';transport.mkdir(exist_ok=True)
 for n,a in enumerate(batches):(transport/f'{n:04d}.json').write_text(json.dumps({'number':n,'total':len(batches),'chunks':a},ensure_ascii=False))
 print(json.dumps({'files':len(FILES),'bytes':sum(r['bytes'] for r in FILES.values()),'batches':len(batches),'scope':snapshot['scope']},ensure_ascii=False))

if __name__=='__main__':prepare()
