"""Retain only public mirror bytes matching the CC0 ScienceDB V1 WAV manifest."""
from pathlib import Path
import re,json,urllib.request,urllib.parse,hashlib,time,os
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent/'external/hearmyship_v1'
RAW=ROOT/'wav';RAW.mkdir(parents=True,exist_ok=True)
html=(ROOT/'deposit_landing.html').read_text()
objects=[json.loads(x) for x in re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',html,re.S)]
deposit=next(x for x in objects if x.get('@id')=='https://doi.org/10.57760/sciencedb.13099')
(ROOT/'deposit_manifest.json').write_text(json.dumps(deposit,ensure_ascii=False,indent=2))
rights=json.loads((ROOT/'datacite.json').read_text())['data']['attributes']['rightsList']
assert any(x.get('rightsIdentifier')=='cc0-1.0' for x in rights)
files=[x for x in deposit['distribution'] if x['name'].lower().endswith('.wav')]
listing=json.loads((ROOT/'public_mirror_listing.json').read_text())
lookup={}
for row in listing:
 if row.get('sound'):
  path='/'+row['sound'].replace('\\','/').lstrip('/')
  lookup.setdefault(path.split('/')[-1],[]).append((path,row))
tasks=[];unmatched=[]
for f in files:
 candidates=lookup.get(f['name'],[])
 if not candidates:unmatched.append(f['name']);continue
 path,row=candidates[0]
 tasks.append({'name':f['name'],'expected_md5':f['md5'],'bytes':int(f['contentSize'].split()[0]),
               'url':'https://hearmyship.fer.hr'+urllib.parse.quote(path,safe='/'),
               'deposit_url':f['contentUrl'],'mirror_id':row['id'],'time':row.get('time'),
               'category':row.get('category'),'length':row.get('length'),'speed':row.get('speed')})
(ROOT/'acquisition_plan.json').write_text(json.dumps({'deposit_wavs':len(files),'matched':len(tasks),
                                                    'unmatched':unmatched,'files':tasks},ensure_ascii=False,indent=2))
def fetch(row):
 target=RAW/(row['expected_md5']+'.wav')
 if target.exists() and hashlib.md5(target.read_bytes()).hexdigest()==row['expected_md5']:
  return {**row,'local_file':str(target.relative_to(ROOT)),'verified':True}
 for attempt in range(3):
  try:
   req=urllib.request.Request(row['url'],headers={'User-Agent':'UnderwaterVesselChallengeResearch/1.0'})
   with urllib.request.urlopen(req,timeout=60) as r:data=r.read(row['bytes']+1)
   if len(data)!=row['bytes'] or hashlib.md5(data).hexdigest()!=row['expected_md5']:
    return {**row,'verified':False,'error':'does not match the original CC0 deposit size/MD5; not used'}
   temp=target.with_suffix('.partial');temp.write_bytes(data);temp.replace(target)
   return {**row,'local_file':str(target.relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest(),'verified':True}
  except Exception as exc:
   if attempt==2:return {**row,'verified':False,'error':f'{type(exc).__name__}: {exc}'}
   time.sleep(attempt+1)
if __name__=='__main__':
 remaining=sum(x['bytes'] for x in tasks if not (RAW/(x['expected_md5']+'.wav')).exists())
 v=os.statvfs(ROOT)
 if v.f_bavail*v.f_frsize<remaining+4*2**30:raise RuntimeError('Keep 4 GiB free; use a smaller explicit acquisition plan.')
 print('CC0 deposit WAVs',len(files),'mirror matches',len(tasks),'planned bytes',sum(x['bytes'] for x in tasks),flush=True)
 records=[]
 with ThreadPoolExecutor(max_workers=6) as pool:
  for i,f in enumerate(as_completed([pool.submit(fetch,r) for r in tasks]),1):
   records.append(f.result())
   if i%25==0:
    (ROOT/'download_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print('Finished',i,'verified',sum(x['verified'] for x in records),flush=True)
 (ROOT/'download_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
 print('Done',len(records),'verified',sum(x['verified'] for x in records),flush=True)
