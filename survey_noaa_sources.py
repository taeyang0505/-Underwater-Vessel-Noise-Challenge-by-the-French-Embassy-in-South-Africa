"""Metadata-only discovery of additional real recordings, excluding sealed OC02_02."""
from pathlib import Path
import json,urllib.request,urllib.parse
ROOT=Path(__file__).resolve().parent/'external/noaa_extension_survey';ROOT.mkdir(parents=True,exist_ok=True)
def listing(prefix,delimiter=None):
    args={'prefix':prefix,'maxResults':1000}
    if delimiter:args['delimiter']=delimiter
    out=[];prefixes=[]
    while True:
        url='https://storage.googleapis.com/storage/v1/b/noaa-passive-bioacoustic/o?'+urllib.parse.urlencode(args)
        with urllib.request.urlopen(url,timeout=40) as r:obj=json.load(r)
        out+=obj.get('items',[]);prefixes+=obj.get('prefixes',[])
        if not obj.get('nextPageToken'):break
        args['pageToken']=obj['nextPageToken']
    return {'items':out,'prefixes':prefixes}
if __name__=='__main__':
    obj=listing('sanctsound/products/detections/')
    (ROOT/'detection_object_inventory.json').write_text(json.dumps(obj,indent=2))
    ships=[x for x in obj['items'] if x['name'].lower().endswith('_ships.csv') and 'oc02_02' not in x['name'].lower()]
    (ROOT/'ship_annotation_candidates.json').write_text(json.dumps(ships,indent=2))
    top=listing('sanctsound/',delimiter='/');(ROOT/'top_prefixes.json').write_text(json.dumps(top,indent=2))
    quality=listing('sanctsound/products/',delimiter='/');(ROOT/'product_prefixes.json').write_text(json.dumps(quality,indent=2))
    print('Independent deployment ship annotation candidates',len(ships),flush=True)
    print('\n'.join(x['name'] for x in ships[:12]),flush=True)
    print('Data product prefixes',quality['prefixes'],flush=True)
