"""Freeze metadata-selected, native 15-min NOAA candidates before any modelling."""
from pathlib import Path
import json,urllib.request,urllib.parse,re,sys
import pandas as pd
from survey_noaa_sources import listing
ROOT=Path(__file__).resolve().parent/'external/expansion_research'
BASE='https://storage.googleapis.com/noaa-passive-bioacoustic/'
FOLDS={'ci01_01':'train','fk01_01':'train','gr01_01':'train',
       'hi01_01':'validation','oc03_02':'calibration','sb01_01':'new_sealed_holdout'}
LOW=10**1.5/10**.05;HIGH=10**3.9*10**.05
def date(s):return pd.Timestamp(s,tz='UTC') if pd.Timestamp(s).tzinfo is None else pd.Timestamp(s).tz_convert('UTC')
def quality_intervals(meta):
    rows=[]
    if 'DATA_QUALITY' in meta:
        for r in meta['DATA_QUALITY'].values():
            lo,hi=map(float,r['Frequency (Hz)'].split(' to '));a,b=r['Date Range'].split(' to ')
            if r['Quality']=='Good' and lo<=LOW and hi>=HIGH:rows.append((date(a),date(b)))
    else:
        for r in meta['QUALITY_DETAILS']['quality_details']:
            if r['quality']=='Good' and float(r['low_freq'])<=LOW and float(r['high_freq'])>=HIGH:
                rows.append((date(r['start']),date(r['end'])))
    return rows
def first_object(prefix):
    url='https://storage.googleapis.com/storage/v1/b/noaa-passive-bioacoustic/o?'+urllib.parse.urlencode({'prefix':prefix,'maxResults':1})
    with urllib.request.urlopen(url,timeout=40) as r:items=json.load(r).get('items',[])
    # Some public folders contain an explicit zero-byte directory marker.
    files=[x for x in items if x['name'].lower().endswith(('.flac','.wav'))]
    if files:return files[0]
    url='https://storage.googleapis.com/storage/v1/b/noaa-passive-bioacoustic/o?'+urllib.parse.urlencode({'prefix':prefix,'maxResults':25})
    with urllib.request.urlopen(url,timeout=40) as r:items=json.load(r).get('items',[])
    return next(x for x in items if x['name'].lower().endswith(('.flac','.wav')))
def timestamp(name):
    iso=re.search(r'_(\d{8}T\d{6}Z)(?:[^0-9]|$)',Path(name).name)
    if iso:return pd.to_datetime(iso[1],format='%Y%m%dT%H%M%SZ',utc=True)
    m=re.search(r'_(\d{12})(?:[^0-9]|$)',Path(name).name)
    return pd.to_datetime(m[1],format='%y%m%d%H%M%S',utc=True) if m else None
if __name__=='__main__':
    plans=[];rejected=[]
    for dep,fold in FOLDS.items():
        station=dep.split('_')[0];folder=ROOT/dep
        mp=folder/('SanctSound_'+dep.upper()+'.json');meta=json.loads(mp.read_text())
        good=quality_intervals(meta);events=pd.read_csv(folder/'ship_events.csv')
        if dep=='oc03_02':
            # Official Good metadata nevertheless warns of a high-frequency noise
            # floor in the first week of November; exclude that week conservatively.
            good=[(max(a,date('2019-11-08')),b) for a,b in good if b>date('2019-11-08')]
        options=[]
        for i,r in events.iterrows():
            if str(r.Labels).strip().lower()!='ship':continue
            a,b=date(r.ISOStartTime),date(r.ISOEndTime)
            # >30min annotation offers room to choose a complete actual 15-min span.
            for g0,g1 in good:
                start=max(a,g0).ceil('min');end=min(b,g1)
                if end-start>=pd.Timedelta(minutes=30):
                    options.append({'annotation_index':int(i),'annotation_start':str(a),'annotation_end':str(b),
                                    'eligible_start':start,'eligible_end':end,'quality_start':str(g0),'quality_end':str(g1)})
        if not options:rejected.append({'deployment':dep,'reason':'No >=30min ship interval wholly supported by Good full-band quality'});continue
        # Earliest and latest eligible dates, no acoustic/outcome selection.
        options.sort(key=lambda r:r['eligible_start']);chosen=[options[0]]
        last=next((r for r in reversed(options) if (r['eligible_start'].date()-chosen[0]['eligible_start'].date()).days>=14),None)
        if last:chosen.append(last)
        prefix=f'sanctsound/audio/{station}/sanctsound_{dep}/audio/'
        first=first_object(prefix)
        base=Path(first['name']).name
        mark=re.search(r'_(?:\d{8}T\d{6}Z|\d{12})(?:[^0-9]|$)',base)
        instrument_prefix=base[:mark.start()+1]
        for index,event in enumerate(chosen):
            day=event['eligible_start'].strftime('%Y%m%d' if re.search(r'_\d{8}T\d{6}Z',first['name']) else '%y%m%d')
            objects=listing(prefix+instrument_prefix+day)['items']
            # Public conversions can mix legacy and ISO filename timestamps.
            alternate=event['eligible_start'].strftime('%y%m%d' if len(day)==8 else '%Y%m%d')
            objects+=listing(prefix+instrument_prefix+alternate)['items']
            # A midnight event may be inside a multi-hour file beginning yesterday.
            yesterday=event['eligible_start']-pd.Timedelta(days=1)
            objects+=listing(prefix+instrument_prefix+yesterday.strftime('%Y%m%d'))['items']
            objects+=listing(prefix+instrument_prefix+yesterday.strftime('%y%m%d'))['items']
            objects=list({x['name']:x for x in objects}.values())
            (folder/('audio_'+day+'_inventory.json')).write_text(json.dumps(objects,indent=2))
            candidates=[]
            for obj in objects:
                if not obj['name'].lower().endswith(('.flac','.wav')):continue
                t=timestamp(obj['name'])
                if t is None or t>event['eligible_end']-pd.Timedelta(minutes=15):continue
                start=max(t,event['eligible_start'])
                if start+pd.Timedelta(minutes=15)<=event['eligible_end']:
                    candidates.append({**obj,'file_start_utc':str(t),'crop_start_utc':str(start),
                        'offset_seconds':(start-t).total_seconds(),'crop_duration_seconds':900})
            # Favor a source starting inside the event (avoids long seek and unknown file end).
            candidates.sort(key=lambda x:(x['offset_seconds']!=0, x['offset_seconds'],x['name']))
            plans.append({'deployment':dep,'station':station,'prospective_split':fold,
                'selection':'earliest/latest metadata-qualified date, >=14-day separation; no audio/model results considered',
                'instrument':meta.get('INSTRUMENT_NAME',meta.get('INSTRUMENT_TYPE')),
                'quality':'Good over all 25 band edges','quality_low_hz':LOW,'quality_high_hz':HIGH,
                **{k:(str(v) if isinstance(v,pd.Timestamp) else v) for k,v in event.items()},
                'source_candidates':candidates[:8]})
            print(dep,index,'candidates',len(candidates),'day',day,flush=True)
    result={'status':'metadata-selected plan, pending native duration/header verification and checksummed acquisition',
        'sealed_existing_deployment_excluded':'OC02_02','files':plans,'rejected':rejected,
        'licence':'CC0-1.0','licence_evidence':['https://www.fisheries.noaa.gov/inport/item/80342','https://www.fisheries.noaa.gov/inport/item/80343']}
    (ROOT/'pilot_acquisition_plan.json').write_text(json.dumps(result,indent=2))
