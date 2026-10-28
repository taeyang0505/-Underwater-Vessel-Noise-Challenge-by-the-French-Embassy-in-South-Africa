"""Reference-equivalent PSD integration in uncalibrated digital energy units."""
import io
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import periodogram

CENTRES=10.0**(1.5+.1*np.arange(25))

def audio_band_power(data):
    """Return complete 2-s segments, minutes, and native WAV audit fields.

    A 20-s recording remains a 20-s reference. No padded/repeated audio is
    claimed to be a measured complete minute.
    """
    with sf.SoundFile(io.BytesIO(data)) as f:
        fs=f.samplerate;channels=f.channels;frames=f.frames
        samples=f.read(dtype='float32',always_2d=True)[:,0]
    nseg=len(samples)//(2*fs)
    if not nseg:raise ValueError('Less than two seconds of observed audio')
    power=[]
    for start in range(0,nseg,30):
        block=samples[start*2*fs:min(nseg,start+30)*2*fs].reshape(-1,2*fs)
        hz,psd=periodogram(block,fs=fs,window='hann',detrend='constant',
                           return_onesided=True,scaling='density',axis=1)
        p=np.full((len(block),25),np.nan,dtype=np.float64)
        for j,c in enumerate(CENTRES):
            low,high=c/10**.05,c*10**.05
            bins=(hz>=low)&(hz<high)
            if high<=fs/2 and bins.any():p[:,j]=psd[:,bins].sum(1)*(hz[1]-hz[0])
        power.append(p)
    segments=np.concatenate(power)
    minutes=segments[:nseg//30*30].reshape(-1,30,25).mean(1)
    if not np.isfinite(segments).all():raise ValueError('Does not support all 25 target bands')
    if not (segments.sum(1)>0).all():raise ValueError('Zero-power audio segment')
    return segments.astype('float32'),minutes.astype('float32'),{
        'sample_rate_hz':fs,'channels':channels,'selected_channel':0,'frames':frames,
        'duration_seconds':frames/fs,'complete_2s_segments':nseg,
        'complete_minutes':len(minutes),'trailing_seconds_dropped':frames/fs-nseg*2,
        'units':'uncalibrated digital mean-square band power'}

class MultipartReader(io.RawIOBase):
    """Seek over byte-split ZIPs without writing a duplicate combined archive."""
    def __init__(self,paths):
        self.files=[Path(p).open('rb') for p in paths]
        self.sizes=[Path(p).stat().st_size for p in paths]
        self.offsets=np.r_[0,np.cumsum(self.sizes)].tolist();self.position=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        p=offset if whence==0 else self.position+offset if whence==1 else self.offsets[-1]+offset
        if p<0:raise ValueError('Negative seek')
        self.position=p;return p
    def read(self,size=-1):
        if size<0:size=self.offsets[-1]-self.position
        result=[]
        while size>0 and self.position<self.offsets[-1]:
            j=int(np.searchsorted(self.offsets,self.position,side='right')-1)
            local=self.position-self.offsets[j]
            amount=min(size,self.sizes[j]-local)
            self.files[j].seek(local);b=self.files[j].read(amount)
            if not b:break
            result.append(b);self.position+=len(b);size-=len(b)
        return b''.join(result)
    def close(self):
        for f in self.files:f.close()
        super().close()
