"""Metadata-first audit of NOAA deployments distinct from sealed OC02_02."""
from pathlib import Path
import json,urllib.request,urllib.parse,hashlib,base64
from concurrent.futures import ThreadPoolExecutor,as_completed
from survey_noaa_sources import listing
ROOT=Path(__file__).resolve().parent/'external/expansion_research'
ROOT.mkdir(parents=True,exist_ok=True)
BASE='https://storage.googleapis.com/noaa-passive-bioacoustic/'
STATIONS=['ci01','fk01','gr01','hi01','mb01','oc01','pm01','sb01']
def acquire(station,deployment='01'):
    deploy=station+'_'+deployment;folder=ROOT/deploy;folder.mkdir(exist_ok=True)
    records=[]
    for kind in ['metadata','ancillary']:
        obj=listing(f'sanctsound/audio/{station}/sanctsound_{deploy}/{kind}/')
        (folder/(kind+'_object_inventory.json')).write_text(json.dumps(obj,indent=2))
        for row in obj['items']:
            name=row['name'];size=int(row['size'])
            if size>4*2**20:continue
            url=BASE+urllib.parse.quote(name,safe='/')+'?generation='+row['generation']
            with urllib.request.urlopen(url,timeout=40) as r:data=r.read()
            expected=base64.b64decode(row['md5Hash']).hex() if row.get('md5Hash') else None
            assert len(data)==size and (expected is None or hashlib.md5(data).hexdigest()==expected)
            filename=Path(name).name;(folder/filename).write_bytes(data)
            records.append({'object':name,'local_file':str((folder/filename).relative_to(ROOT)),
                'bytes':len(data),'generation':row['generation'],'source_md5_verified':expected is not None,
                'md5_hex':hashlib.md5(data).hexdigest(),'sha256':hashlib.sha256(data).hexdigest()})
    ship=f'sanctsound/products/detections/{station}/sanctsound_{deploy}_ships/data/SanctSound_{deploy.upper()}_ships.csv'
    with urllib.request.urlopen(BASE+ship,timeout=40) as r:data=r.read()
    (folder/'ship_events.csv').write_bytes(data)
    (folder/'metadata_acquisition_manifest.json').write_text(json.dumps(records,indent=2))
    return deploy,[x['object'] for x in records]
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(acquire,s) for s in STATIONS]):print(f.result(),flush=True)
