"""Resume verified public downloads with bounded HTTP ranges; preserve partial work."""
from pathlib import Path
import os,json,urllib.request,hashlib,time,threading
from concurrent.futures import ThreadPoolExecutor,as_completed
from fetch_external import FILES,DEST,BASE
SIZES={'vtuad_strict_wav_vessel.zip.part01':1237445900,
       'vtuad_strict_wav_vessel.zip.part02':1237445900,
       'vtuad_strict_wav_vessel.zip.part03':1237445899,
       'vtuad_strict_wav_background.zip':926037631}
CHUNK=16*2**20
lock=threading.Lock()
states={};fds={};tasks=[]
for name,size in SIZES.items():
 if (DEST/name).exists():continue
 p=DEST/(name+'.partial');statefile=DEST/(name+'.ranges.json')
 if statefile.exists():state=json.loads(statefile.read_text())
 else:
  state={'contiguous_prefix_bytes':p.stat().st_size if p.exists() else 0,'completed':[],'size':size}
  statefile.write_text(json.dumps(state))
 states[name]=state;fds[name]=os.open(p,os.O_RDWR|os.O_CREAT)
 for start in range(state['contiguous_prefix_bytes'],size,CHUNK):
  end=min(start+CHUNK,size)-1
  if start not in state['completed']:tasks.append((name,start,end))

def get_chunk(task):
 name,start,end=task;expected=f'bytes {start}-{end}/{SIZES[name]}'
 for attempt in range(6):
  try:
   req=urllib.request.Request(BASE+name+'?download=1',headers={'Range':f'bytes={start}-{end}','User-Agent':'UnderwaterVesselChallengeResearch/1.0'})
   with urllib.request.urlopen(req,timeout=90) as src:
    if src.status!=206 or src.headers.get('Content-Range')!=expected:
     raise RuntimeError('Server did not honor the exact requested public byte range')
    pos=start
    while block:=src.read(2**20):
     if pos+len(block)>end+1:raise RuntimeError('Range exceeded')
     os.pwrite(fds[name],block,pos);pos+=len(block)
    if pos!=end+1:raise RuntimeError('Truncated range')
   with lock:
    states[name]['completed'].append(start)
    tmp=DEST/(name+'.ranges.tmp');tmp.write_text(json.dumps(states[name]));tmp.replace(DEST/(name+'.ranges.json'))
   return end-start+1
  except Exception as exc:
   if attempt==5:raise
   print('Retry',name,start,type(exc).__name__,flush=True);time.sleep(min(30,2**attempt))

if __name__=='__main__':
 print('Pending ranges',len(tasks),'bytes',sum(e-s+1 for _,s,e in tasks),flush=True)
 done=0;t0=time.time()
 with ThreadPoolExecutor(max_workers=8) as pool:
  for i,f in enumerate(as_completed([pool.submit(get_chunk,t) for t in tasks]),1):
   done+=f.result()
   if i%8==0:print(f'Ranges {i}/{len(tasks)}; {done/1e6:.0f} MB; mean {done/1e6/(time.time()-t0):.2f} MB/s',flush=True)
 for fd in fds.values():os.close(fd)
 records=[]
 for name,md5 in FILES.items():
  p=DEST/name
  if name in states:p=DEST/(name+'.partial')
  h=hashlib.md5()
  with p.open('rb') as f:
   while block:=f.read(2**20):h.update(block)
  if h.hexdigest()!=md5:raise ValueError('MD5 failed: '+name)
  if p.name.endswith('.partial'):p.replace(DEST/name);p=DEST/name
  print('Verified',name,p.stat().st_size,flush=True)
  records.append({'file':name,'url':BASE+name+'?download=1','bytes':p.stat().st_size,'md5':md5,'verified':True})
 (DEST/'download_manifest.json').write_text(json.dumps(records,indent=2))
