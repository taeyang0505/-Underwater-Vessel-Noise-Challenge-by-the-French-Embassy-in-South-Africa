"""Small, generation-pinned public NOAA CC0 holdout; never training data."""
from pathlib import Path
import urllib.request,urllib.parse,json,io,hashlib
import soundfile as sf
from pipeline.audio import audio_band_power
import numpy as np
ROOT=Path(__file__).resolve().parent/'external/noaa_sanctsound_holdout';ROOT.mkdir(parents=True,exist_ok=True)
BASE='https://storage.googleapis.com/noaa-passive-bioacoustic/'
class PublicRangeFile(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.cache={};self.chunk=4*2**20
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def readinto(self,buffer):
        data=self.read(len(buffer));buffer[:len(data)]=data;return len(data)
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos);out=[]
        while n>0:
            start=self.pos//self.chunk*self.chunk;end=min(start+self.chunk,self.size)-1
            if start not in self.cache:
                req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=60) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':raise ValueError('Exact range not honored')
                    self.cache[start]=r.read()
            block=self.cache[start];offset=self.pos-start;take=min(n,len(block)-offset)
            if take<=0:raise ValueError('Truncated public object')
            out.append(block[offset:offset+take]);self.pos+=take;n-=take
        return b''.join(out)

if __name__=='__main__':
    records=[]
    names=[('190712015109',111),('190713015100',960),('190714015052',68)]
    for code,offset in names:
        name=f'SanctSound_OC02_02_671117349_{code}.flac'
        obj='sanctsound/audio/oc02/sanctsound_oc02_02/audio/'+name
        url='https://storage.googleapis.com/storage/v1/b/noaa-passive-bioacoustic/o/'+urllib.parse.quote(obj,safe='')
        with urllib.request.urlopen(url,timeout=30) as r:meta=json.load(r)
        (ROOT/(name+'.metadata.json')).write_text(json.dumps(meta,indent=2))
        src=PublicRangeFile(BASE+obj+'?generation='+meta['generation'],int(meta['size']))
        with sf.SoundFile(src) as f:
            fs=f.samplerate;channels=f.channels;f.seek(offset*fs);samples=f.read(900*fs,dtype='int16',always_2d=True)
        assert len(samples)==900*fs and channels==1
        crop=ROOT/(code+'_15min.flac');sf.write(crop,samples,fs,format='FLAC',subtype='PCM_16')
        data=crop.read_bytes();segments,minutes,audit=audio_band_power(data)
        np.save(ROOT/(code+'_minutes_power.npy'),minutes)
        records.append({'source_object':obj,'generation':meta['generation'],'source_md5_base64':meta.get('md5Hash'),
            'full_source_md5_verified':False,'crop_sha256':hashlib.sha256(data).hexdigest(),
            'crop':crop.name,'start_offset_seconds':offset,'duration_seconds':900,
            'downloaded_range_bytes':sum(map(len,src.cache.values())),**audit,
            'split':'independent_frozen_holdout','licence':'CC0-1.0',
            'licence_evidence':'https://www.fisheries.noaa.gov/inport/item/80342#access-information',
            'warning':'ship-event interval, not isolated ship energy or independent hull identity'})
        (ROOT/'manifest.json').write_text(json.dumps(records,indent=2));print('Acquired real continuous 15-min holdout',code,flush=True)
    ship='sanctsound/products/detections/oc02/sanctsound_oc02_02_ships/data/SanctSound_OC02_02_ships.csv'
    with urllib.request.urlopen(BASE+ship,timeout=30) as r:(ROOT/'ship_events.csv').write_bytes(r.read())
