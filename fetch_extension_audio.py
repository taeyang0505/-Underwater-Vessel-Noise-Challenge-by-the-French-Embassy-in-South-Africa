"""Native, quality-qualified 15-min excerpts from pinned public NOAA objects.

Only exact byte ranges are acquired; full-object MD5 is explicitly NOT claimed.
No challenge Test or sealed OC02_02 data is read. SB01 is a new sealed source.
"""
from pathlib import Path
import json,urllib.request,urllib.parse,hashlib,io,threading,time
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
import soundfile as sf
from pipeline.audio import audio_band_power
from fetch_extension_plan import HIGH
ROOT=Path(__file__).resolve().parent/'external/expansion_research'
BASE='https://storage.googleapis.com/noaa-passive-bioacoustic/'
BUDGET=1800*10**6
downloaded=0;lock=threading.Lock()
class PinnedRangeFile(io.RawIOBase):
    def __init__(self,row):
        self.row=row;self.size=int(row['size']);self.pos=0;self.chunk=4*2**20;self.cache={};self.ledger=[]
        self.url=BASE+urllib.parse.quote(row['name'],safe='/')+'?generation='+row['generation']
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        if self.pos<0:raise ValueError('Negative source seek')
        return self.pos
    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)
    def read(self,n=-1):
        global downloaded
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos);out=[]
        while n>0:
            start=self.pos//self.chunk*self.chunk;end=min(start+self.chunk,self.size)-1
            if start not in self.cache:
                count=end-start+1
                with lock:
                    if downloaded+count>BUDGET:raise RuntimeError('Conservative 1.8GB range-transfer budget reached')
                    downloaded+=count
                expected=f'bytes {start}-{end}/{self.size}'
                req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=70) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=expected:raise ValueError('Exact range not honored')
                    data=r.read()
                if len(data)!=count:raise ValueError('Truncated source range')
                self.cache[start]=data;self.ledger.append({'start':start,'end_inclusive':end,'sha256':hashlib.sha256(data).hexdigest()})
            data=self.cache[start];offset=self.pos-start;take=min(n,len(data)-offset)
            if take<=0:raise ValueError('Truncated cache')
            out.append(data[offset:offset+take]);self.pos+=take;n-=take
        return b''.join(out)
def acquire(plan):
    errors=[]
    for source in plan['source_candidates']:
        try:
            reader=PinnedRangeFile(source)
            with sf.SoundFile(reader) as f:
                fs=f.samplerate;channels=f.channels;frames=f.frames;subtype=f.subtype
                if fs/2<HIGH:raise ValueError('Native sample rate does not support all target band edges')
                if channels!=1:raise ValueError('Pilot requires a documented mono source')
                if subtype not in ['PCM_16','PCM_24']:raise ValueError('Unsupported lossless crop subtype '+subtype)
                offset=int(round(source['offset_seconds']*fs));length=900*fs
                if offset<0 or offset+length>frames:raise ValueError('Native file does not cover the complete 15-min crop')
                f.seek(offset);samples=f.read(length,dtype='int16' if subtype=='PCM_16' else 'int32',always_2d=True)
            if len(samples)!=length:raise ValueError('Native frame coverage incomplete')
            maxval=np.iinfo(samples.dtype).max if subtype=='PCM_16' else (2**23-1)<<8
            minval=np.iinfo(samples.dtype).min
            rail=float(((samples==maxval)|(samples==minval)).mean())
            if rail>.001:raise ValueError('More than 0.1% samples at digital rails')
            stamp=pd.Timestamp(source['crop_start_utc']).strftime('%Y%m%dT%H%M%SZ')
            name=plan['deployment']+'_'+stamp
            folder=ROOT/'native_15min'/plan['prospective_split'];folder.mkdir(parents=True,exist_ok=True)
            crop=folder/(name+'.flac');sf.write(crop,samples,fs,format='FLAC',subtype=subtype)
            data=crop.read_bytes();seg,minute,audit=audio_band_power(data)
            if len(minute)!=15:raise ValueError('Expected exactly 15 measured minutes')
            np.savez_compressed(folder/(name+'_band_power.npz'),segments=seg,minutes=minute)
            row={'clip_id':name,'source':'NOAA_SanctSound_extension_v1','licence':'CC0-1.0',
                'deployment':plan['deployment'],'station':plan['station'],'instrument':plan['instrument'],
                'device_serial':Path(source['name']).name.split('_')[3],
                'prospective_split':plan['prospective_split'],'split_is_locked':True,
                'source_object':source['name'],'source_generation':source['generation'],
                'source_size_bytes':int(source['size']),'source_md5_base64':source.get('md5Hash'),
                'full_source_md5_verified':False,'acquisition':'exact generation-pinned HTTPS ranges; decoded native frames',
                'acquired_range_bytes':sum(x['end_inclusive']-x['start']+1 for x in reader.ledger),
                'range_sha256_ledger':reader.ledger,'native_source_frames':frames,
                'native_subtype':subtype,'native_crop_start_frame':offset,'native_crop_frames':length,
                'start_utc':source['crop_start_utc'],
                'end_utc':str(pd.Timestamp(source['crop_start_utc'])+pd.Timedelta(minutes=15)),
                'recording_day':stamp[:8],'quality_label':'Good','quality_start':plan['quality_start'],'quality_end':plan['quality_end'],
                'quality_low_hz':plan['quality_low_hz'],'quality_high_hz':plan['quality_high_hz'],
                'annotation':'ship','annotation_start':plan['annotation_start'],'annotation_end':plan['annotation_end'],
                'crop_relative_path':str(crop.relative_to(ROOT)),'crop_sha256':hashlib.sha256(data).hexdigest(),
                'band_power_relative_path':str((folder/(name+'_band_power.npz')).relative_to(ROOT)),
                'clip_digital_rail_fraction':rail,'zero_sample_fraction':float((samples==0).mean()),
                **audit,'warning':'Ship annotation is not isolated vessel sound or an independently identified hull.',
                'model_evaluated':False,'sealed_if_SB01':plan['prospective_split']=='new_sealed_holdout'}
            return row
        except Exception as e:errors.append({'source':source['name'],'error':str(e)})
    return {'deployment':plan['deployment'],'prospective_split':plan['prospective_split'],'failed':True,'errors':errors}
if __name__=='__main__':
    if (ROOT/'native_15min_manifest.json').exists():
        raise SystemExit('Acquisition already exists. Preserve its locked manifest; use a new versioned destination for another acquisition.')
    plan=json.loads((ROOT/'pilot_acquisition_plan.json').read_text())
    assert all(x['deployment']!='oc02_02' for x in plan['files'])
    records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(acquire,x) for x in plan['files']]):
            row=f.result();records.append(row)
            (ROOT/'native_15min_manifest.json').write_text(json.dumps(records,indent=2))
            print('Native crop',row.get('clip_id',row['deployment']),'failed',row.get('failed',False),'total range MB',downloaded/1e6,flush=True)
    good=[r for r in records if not r.get('failed')]
    seen={}
    for r in good:
        previous=seen.setdefault(r['crop_sha256'],r['prospective_split'])
        assert previous==r['prospective_split'],'Duplicate audio crosses locked prospective partitions'
        a,b=pd.Timestamp(r['start_utc']),pd.Timestamp(r['end_utc'])
        assert pd.Timestamp(r['annotation_start'])<=a and b<=pd.Timestamp(r['annotation_end'])
        assert pd.Timestamp(r['quality_start'])<=a and b<=pd.Timestamp(r['quality_end'])
    summary={'clips':len(good),'native_minutes':15*len(good),'actual_unique_stations':len({r['station'] for r in good}),
        'actual_unique_deployments':len({r['deployment'] for r in good}),
        'actual_unique_days':len({r['recording_day'] for r in good}),
        'downloaded_range_bytes':downloaded,'budget_bytes':BUDGET,'failures':[r for r in records if r.get('failed')],
        'test_accessed':False,'existing_OC02_02_accessed':False,'new_SB01_model_evaluated':False,
        'source_md5_limitation':'Only exact pinned ranges were downloaded; full object MD5 has NOT been verified.',
        'limitations':'Few independent deployments; ship annotations do not certify vessel-only sound or distinct hulls.'}
    (ROOT/'native_15min_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
