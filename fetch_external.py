"""Download only the declared, licence-documented strict-interval VTUAD release."""
from pathlib import Path
import urllib.request, hashlib, json, os, time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'external/vtuad_v1.3.2'
DEST.mkdir(parents=True,exist_ok=True)
BASE='https://zenodo.org/records/23037895/files/'
FILES={
 'registry_anonymized.csv':'f58655efd07502583b4b2cf5ca2bc5de',
 'dataset_card.md':'761d13ef032d66dbc2e54ad0bcf5dc53',
 'license_evidence_v131.md':'9bbc4f36cedc06e5aac938135c2a25c1',
 'strict_interval_wav_SHA256SUMS.txt':'3c19f42c80a9c5979c15c4ad1abc9c67',
 'README_vtuad_strict_wav_vessel_parts.txt':'43bab1eba183cdfac74130cdfa135620',
 'vtuad_wraparound_splice_clips_v1.csv':'ca6c17427fb0bd1a3f47ebce667ad9f3',
 'vtuad_strict_wav_vessel.zip.part01':'6e3fe8245218faa4f94868531e272344',
 'vtuad_strict_wav_vessel.zip.part02':'af6408581f6981f45ffdbce811e9f558',
 'vtuad_strict_wav_vessel.zip.part03':'dfa59398483a651765a6265df76febad',
 'vtuad_strict_wav_background.zip':'d28ad0fb6ccc98341ba4f7e953d9c68f',
}
def fetch(item):
 name,expected=item;target=DEST/name
 if target.exists() and hashlib.md5(target.read_bytes()).hexdigest()==expected:
  return {'file':name,'bytes':target.stat().st_size,'md5':expected,'verified':True}
 url=BASE+name+'?download=1'
 for attempt in range(4):
  try:
   req=urllib.request.Request(url,headers={'User-Agent':'UnderwaterVesselChallengeResearch/1.0'})
   h=hashlib.md5();size=0
   with urllib.request.urlopen(req,timeout=90) as src,(DEST/(name+'.partial')).open('wb') as dst:
    while block:=src.read(2**20):
     dst.write(block);h.update(block);size+=len(block)
   if h.hexdigest()!=expected:raise ValueError(f'Checksum mismatch for {name}')
   (DEST/(name+'.partial')).replace(target)
   print(f'Verified {name}: {size:,} bytes',flush=True)
   return {'file':name,'url':url,'bytes':size,'md5':expected,'verified':True}
  except Exception as e:
   print(f'{name}: attempt {attempt+1}: {type(e).__name__}: {e}',flush=True)
   if attempt==3:raise
   time.sleep(2**attempt)
if __name__=='__main__':
 free=os.statvfs(DEST).f_bavail*os.statvfs(DEST).f_frsize
 needed=sum(1237445900 if 'part' in x and '.zip.' in x else 926_000_000 if x.endswith('background.zip') else 200000 for x in FILES if not (DEST/x).exists())
 if free<needed+3*2**30:raise RuntimeError('Insufficient free disk space; preserve at least 3 GiB.')
 with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,FILES.items()))
 (DEST/'download_manifest.json').write_text(json.dumps(records,indent=2))
