from pathlib import Path
import urllib.request,json
root=Path(__file__).resolve().parent/'external/hearmyship_v1'
root.mkdir(parents=True,exist_ok=True)
sources={
 'datacite.json':'https://api.datacite.org/dois/10.57760/sciencedb.13099',
 'public_mirror_listing.json':'https://hearmyship.fer.hr/api/vessels-download?page=all',
}
for name,url in sources.items():
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'UnderwaterVesselChallengeResearch/1.0'}),timeout=60) as r:
   data=r.read()
  (root/name).write_bytes(data)
  obj=json.loads(data)
  print(name,len(data), 'type',type(obj).__name__,flush=True)
  if name=='datacite.json':print('rights',obj['data']['attributes']['rightsList'],flush=True)
  else:print(str(obj)[:3500],flush=True)
 except Exception as e:print(name,type(e).__name__,str(e),flush=True)
