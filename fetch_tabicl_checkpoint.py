from pathlib import Path
import urllib.request,json,hashlib
ROOT=Path(__file__).resolve().parent/'models/pretrained';ROOT.mkdir(parents=True,exist_ok=True)
if __name__=='__main__':
    with urllib.request.urlopen('https://huggingface.co/api/models/jingang/TabICL?blobs=true',timeout=30) as r:meta=json.load(r)
    assert meta['cardData']['license']=='bsd-3-clause'
    filename='tabicl-regressor-v2-20260212.ckpt';file=next(x for x in meta['siblings'] if x['rfilename']==filename)
    target=ROOT/filename;sha=file['lfs']['sha256'];commit=meta['sha']
    url=f'https://huggingface.co/jingang/TabICL/resolve/{commit}/{filename}'
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=sha:
        h=hashlib.sha256();temp=target.with_suffix('.partial')
        with urllib.request.urlopen(url,timeout=90) as src,temp.open('wb') as dst:
            while data:=src.read(2**20):dst.write(data);h.update(data)
        assert h.hexdigest()==sha;temp.replace(target)
    info={'model':'TabICLv2 regression','repository':'jingang/TabICL','revision':commit,
          'filename':filename,'sha256':sha,'bytes':target.stat().st_size,'licence':'BSD-3-Clause',
          'model_card':'https://huggingface.co/jingang/TabICL','download_url':url}
    (ROOT/'tabicl_provenance.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
