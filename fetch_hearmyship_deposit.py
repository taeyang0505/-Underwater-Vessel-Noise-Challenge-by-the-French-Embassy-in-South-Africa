"""Complete missing mirror objects from original public CC0 distribution URLs."""
from pathlib import Path
import json,urllib.request,hashlib,re,datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent/'external/hearmyship_v1'
records=json.loads((ROOT/'download_manifest.json').read_text())
seen={r['name'] for r in records};deposit=json.loads((ROOT/'deposit_manifest.json').read_text())
for f in deposit['distribution']:
    if not f['name'].lower().endswith('.wav') or f['name'] in seen:continue
    stamp=re.search(r'(\d{2}\.\d{2}\.\d{2})_(\d{6})',f['name'])
    time=datetime.datetime.strptime('_'.join(stamp.groups()),'%d.%m.%y_%H%M%S').isoformat() if stamp else None
    records.append({'name':f['name'],'expected_md5':f['md5'],'bytes':int(f['contentSize'].split()[0]),
        'deposit_url':f['contentUrl'],'url':f['contentUrl'],'time':time,'category':f['name'].split('_')[0],
        'mirror_id':None,'verified':False})
def fetch(i):
    r=records[i];req=urllib.request.Request(r['deposit_url'],headers={'User-Agent':'UnderwaterVesselChallengeResearch/1.0'})
    try:
        with urllib.request.urlopen(req,timeout=45) as f:data=f.read(r['bytes']+1)
        if len(data)!=r['bytes'] or hashlib.md5(data).hexdigest()!=r['expected_md5']:
            return i,{**r,'original_deposit_error':'size or MD5 mismatch; excluded'}
        p=ROOT/'wav'/(r['expected_md5']+'.wav');tmp=p.with_suffix('.partial');tmp.write_bytes(data);tmp.replace(p)
        return i,{**r,'url':r['deposit_url'],'local_file':str(p.relative_to(ROOT)),
            'verified':True,'sha256':hashlib.sha256(data).hexdigest(),'acquired_from':'original_CC0_deposit'}
    except Exception as e:return i,{**r,'original_deposit_error':str(e)}
pending=[i for i,r in enumerate(records) if not r['verified']]
print('Original deposit pending',len(pending),flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:
    for j,f in enumerate(as_completed([pool.submit(fetch,i) for i in pending]),1):
        i,r=f.result();records[i]=r
        if j%20==0:print('Attempted',j,'verified total',sum(x['verified'] for x in records),flush=True)
tmp=ROOT/'download_manifest.tmp';tmp.write_text(json.dumps(records,ensure_ascii=False,indent=2));tmp.replace(ROOT/'download_manifest.json')
print('Completed',len(records),'verified',sum(x['verified'] for x in records),flush=True)
