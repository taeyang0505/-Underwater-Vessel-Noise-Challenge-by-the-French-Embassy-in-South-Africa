"""Metadata-selected, SHA-verified clips for early experiments during full download."""
from pathlib import Path
import io,sys,json,urllib.request,hashlib,zipfile,threading,time
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.path.insert(0,'/private/tmp/underwater-eda-deps')
import pandas as pd
from pipeline.prepare_external import split_key
ROOT=Path(__file__).resolve().parent/'external/vtuad_v1.3.2'
CACHE=ROOT/'pilot_ranges';CACHE.mkdir(exist_ok=True)
WAV=ROOT/'pilot_wav';WAV.mkdir(exist_ok=True)
NAMES=[f'vtuad_strict_wav_vessel.zip.part{i:02d}' for i in range(1,4)]
SIZES=[1237445900,1237445900,1237445899]
CHUNK=4*2**20
locks={};master=threading.Lock()
def block(name,start,total):
    key=f'{name}.{start}'
    with master:lock=locks.setdefault(key,threading.Lock())
    with lock:
        p=CACHE/key
        if p.exists():return p.read_bytes()
        end=min(start+CHUNK,total)-1
        for attempt in range(4):
            try:
                req=urllib.request.Request(f'https://zenodo.org/records/23037895/files/{name}?download=1',headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=90) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}':raise ValueError('Range response mismatch')
                    data=r.read()
                if len(data)!=end-start+1:raise ValueError('Truncated block')
                p.write_bytes(data);return data
            except Exception:
                if attempt==3:raise
                time.sleep(attempt+1)
class RemoteReader(io.RawIOBase):
    def __init__(self,names,sizes):
        self.names=names;self.sizes=sizes;self.offset=[0]
        for s in sizes:self.offset.append(self.offset[-1]+s)
        self.pos=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,p,whence=0):
        self.pos=p if whence==0 else self.pos+p if whence==1 else self.offset[-1]+p
        return self.pos
    def read(self,n=-1):
        n=self.offset[-1]-self.pos if n<0 else n;out=[]
        while n and self.pos<self.offset[-1]:
            j=next(k for k in range(len(self.names)) if self.offset[k]<=self.pos<self.offset[k+1])
            local=self.pos-self.offset[j];start=local//CHUNK*CHUNK
            data=block(self.names[j],start,self.sizes[j]);a=local-start
            take=min(n,len(data)-a);out.append(data[a:a+take]);n-=take;self.pos+=take
        return b''.join(out)
def acquire(task):
    row,member,names,sizes=task
    expected=SHA[member];target=WAV/(expected+'.wav')
    if not target.exists():
        with zipfile.ZipFile(RemoteReader(names,sizes)) as z:data=z.read(member)
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Provider WAV SHA-256 failed')
        target.write_bytes(data)
    return {**row,'member':member,'local_file':str(target.relative_to(ROOT)),
            'sha256':expected,'verified':True,'download_mode':'public_metadata_selected_byte_ranges'}
SHA={line.split(maxsplit=1)[1].strip():line.split()[0] for line in (ROOT/'strict_interval_wav_SHA256SUMS.txt').read_text().splitlines()}
if __name__=='__main__':
    reg=pd.read_csv(ROOT/'registry_anonymized.csv');v=reg[reg.path.str.startswith('./vessel/')].copy()
    # Whole hulls are assigned before clips are selected. Use multiple classes/splits.
    v['group_key']='VTUAD_HULL_'+v.hull_id
    v['split']=v.group_key.map(split_key)
    chosen=v.sort_values(['duration_sec','path']).groupby(['label','split'],sort=True).head(1)
    bg=reg[reg.path.str.startswith('./background/')].copy()
    bg['group_key']='VTUAD_BACKGROUND_DAY_'+bg.utc_begin.str[:10]
    bg['split']=bg.group_key.map(split_key)
    bg=bg.sort_values('duration_sec').groupby('split').head(1)
    tasks=[]
    for frame,role,names,sizes in [(chosen,'vessel_passage',NAMES,SIZES),
                                   (bg,'ambient_control',['vtuad_strict_wav_background.zip'],[926037631])]:
        with zipfile.ZipFile(RemoteReader(names,sizes)) as z:members=set(z.namelist())
        for r in frame.to_dict('records'):
            suffix=r['path'].removeprefix('./');member=next(x for x in members if x.endswith('/'+suffix))
            tasks.append(({**r,'archive_role':role},member,names,sizes))
    print('Selected clips',len(tasks),'by public class/split/duration metadata',flush=True)
    records=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(acquire,t) for t in tasks]):
            records.append(f.result())
            (ROOT/'pilot_download_manifest.json').write_text(json.dumps(records,indent=2))
            print('Verified pilot WAV',len(records),'/',len(tasks),flush=True)
