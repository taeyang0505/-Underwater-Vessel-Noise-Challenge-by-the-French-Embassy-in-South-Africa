from pathlib import Path
import urllib.request,json,hashlib
ROOT=Path(__file__).resolve().parent/'models/pretrained';ROOT.mkdir(parents=True,exist_ok=True)
if __name__=='__main__':
    with urllib.request.urlopen('https://huggingface.co/api/models/nvidia/Kumo-Tabular/revision/v1.0.1?blobs=true',timeout=30) as r:meta=json.load(r)
    assert meta['cardData']['license']=='openmdw-1.1'
    filename='large/regressor.pt';f=next(x for x in meta['siblings'] if x['rfilename']==filename)
    sha=f['lfs']['sha256'];commit=meta['sha'];target=ROOT/'kumo_large_regressor.pt'
    url=f'https://huggingface.co/nvidia/Kumo-Tabular/resolve/{commit}/{filename}'
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=sha:
        temp=target.with_suffix('.partial');h=hashlib.sha256()
        with urllib.request.urlopen(url,timeout=90) as src,temp.open('wb') as dst:
            while data:=src.read(2**20):dst.write(data);h.update(data)
        assert h.hexdigest()==sha;temp.replace(target)
    info={'model':'NVIDIA Kumo Tabular Large regression','repository':'nvidia/Kumo-Tabular',
        'revision':commit,'release_tag':'v1.0.1','filename':filename,'sha256':sha,
        'bytes':target.stat().st_size,'licence':'OpenMDW-1.1','download_url':url,
        'licence_evidence':'https://openmdw.ai/license/1-1/'}
    (ROOT/'kumo_provenance.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
