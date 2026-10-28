from pathlib import Path
import sys,json,hashlib,zipfile,argparse
import numpy as np
import pandas as pd
from pipeline.audio import audio_band_power,MultipartReader
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed';OUT.mkdir(parents=True,exist_ok=True)

def split_key(key):
    v=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%10
    return 'validation' if v==0 else 'calibration' if v==1 else 'domain_holdout' if v==2 else 'train'

def main(include_vtuad=True):
    segments=[];minutes=[];meta=[];failures=[];seg_offset=0;min_offset=0
    cache=ROOT/'data/processed/audio_conversion_cache';cache.mkdir(exist_ok=True)
    def add(data,info):
        nonlocal seg_offset,min_offset
        sha=hashlib.sha256(data).hexdigest()
        try:
            cp=cache/(sha+'.npz')
            if cp.exists():
                with np.load(cp) as z:s=z['segments'];m=z['minutes'];audit=json.loads(str(z['audit']))
            else:
                s,m,audit=audio_band_power(data)
                np.savez_compressed(cp,segments=s,minutes=m,audit=json.dumps(audit))
        except Exception as e:
            failures.append({**info,'error':str(e)});return
        item={**info,**audit,'sha256':sha,'segment_start':seg_offset,'segment_count':len(s),
              'minute_start':min_offset,'minute_count':len(m)}
        segments.append(s);minutes.append(m);meta.append(item)
        seg_offset+=len(s);min_offset+=len(m)
        if len(meta)%50==0:print('Converted clips',len(meta),flush=True)
    hroot=ROOT/'external/hearmyship_v1'
    hf=hroot/'download_manifest.json'
    if hf.exists():
        seen=set()
        for r in json.loads(hf.read_text()):
            if not r['verified'] or r['expected_md5'] in seen:continue
            seen.add(r['expected_md5'])
            p=hroot/r['local_file'];data=p.read_bytes()
            assert hashlib.md5(data).hexdigest()==r['expected_md5']
            day=r['time'][:10]
            add(data,{'source':'HearMyShip_V1','clip_id':r['expected_md5'],'name':r['name'],
                      'archive_role':'vessel_passage','weak_class':r['category'],
                      'group_key':'HMS_DAY_'+day,'recording_day':day,'split':split_key('HMS_DAY_'+day),
                      'hull_isolation_verified':False,'licence':'CC0-1.0',
                      'source_url':r['url'],'deposit_md5_verified':True,
                      'reference_kind':'measured_short_CPA_excerpt_not_a_complete_minute'})
    vr=ROOT/'external/vtuad_v1.3.2'
    parts=[vr/f'vtuad_strict_wav_vessel.zip.part{i:02d}' for i in range(1,4)]
    ready=all(p.exists() for p in parts) and (vr/'vtuad_strict_wav_background.zip').exists()
    if include_vtuad and ready:
        expected={line.split(maxsplit=1)[1].strip():line.split()[0] for line in (vr/'strict_interval_wav_SHA256SUMS.txt').read_text().splitlines()}
        registry=pd.read_csv(vr/'registry_anonymized.csv')
        lookup={str(r['path']).removeprefix('./'):r for r in registry.to_dict('records')}
        # A source identity is retained for split isolation only, never as model input.
        for archive,role in [(MultipartReader(parts),'vessel_passage'),(vr/'vtuad_strict_wav_background.zip','ambient_control')]:
            with zipfile.ZipFile(archive) as z:
                for name in sorted(z.namelist()):
                    if not name.endswith('.wav'):continue
                    data=z.read(name)
                    assert hashlib.sha256(data).hexdigest()==expected[name]
                    key='/'.join(name.split('/')[-2:]);r=lookup[key]
                    day=str(r['utc_begin'])[:10]
                    group='VTUAD_HULL_'+str(r['hull_id']) if role=='vessel_passage' else 'VTUAD_DAY_34_'+day
                    hull_fold=split_key(group)
                    days=pd.date_range(pd.Timestamp(r['utc_begin']).floor('D'),
                        (pd.Timestamp(r['utc_end'])-pd.Timedelta('1ns')).floor('D'),freq='D')
                    day_folds={split_key('VTUAD_DAY_34_'+str(d.date())) for d in days}
                    day_fold=next(iter(day_folds)) if len(day_folds)==1 else 'excluded_cross_day_conflict'
                    # Keep same-hull AND same-day observations out of different folds.
                    # Conflicting assignments remain in the inventory but are not references.
                    fold=day_fold if role=='ambient_control' or hull_fold==day_fold else 'excluded_hull_day_conflict'
                    add(data,{'source':'VTUAD_v1.3.2','clip_id':hashlib.sha256(data).hexdigest()[:16],
                              'name':name,'archive_role':role,'weak_class':r['label'],
                              'group_key':group,'recording_day':str(r['utc_begin'])[:10],
                              'utc_begin':str(r['utc_begin']),'utc_end':str(r['utc_end']),
                              'split':fold,'hull_isolation_verified':role=='vessel_passage',
                              'licence':'CC-BY-4.0','provider_sha256_verified':True,
                              'source_url':'https://doi.org/10.5281/zenodo.23037895',
                              'reference_kind':'measured_strict_AIS_interval_with_ambient_sound'})
            if isinstance(archive,MultipartReader):archive.close()
    if not meta:raise RuntimeError('No verified external audio is ready')
    arr=np.concatenate(segments);mins=np.concatenate(minutes)
    np.save(OUT/'external_segments_power.npy',arr)
    np.save(OUT/'external_minutes_power.npy',mins)
    frame=pd.DataFrame(meta)
    # Byte-identical recordings must not cross partitions, even under different filenames.
    duplicate_folds=frame.groupby('sha256')['split'].nunique()
    conflicts=set(duplicate_folds[duplicate_folds>1].index)
    frame.loc[frame.sha256.isin(conflicts),'split']='excluded_duplicate_split_conflict'
    frame.to_parquet(OUT/'external_clip_manifest.parquet',index=False,compression='zstd')
    info={'clips':len(meta),'segment_shape':list(arr.shape),'complete_minute_shape':list(mins.shape),
          'sources':pd.Series([x['source'] for x in meta]).value_counts().to_dict(),
          'split_counts':frame.groupby(['source','split']).size().to_dict().__str__(),
          'split_policy':'HMS by recording day (unknown hull recurrence); VTUAD hull/day intersection, conflicting clips excluded; byte-duplicate conflicts excluded',
          'vtuad_download_complete':ready,'failures':failures,
          'test_accessed':False,'source_hash':hashlib.sha256(json.dumps(frame.to_dict('records'),sort_keys=True).encode()).hexdigest(),
          'day_split_salt':34,'day_split_selection':'First integer salt with at least 5 AIS-positive clips and 1 ambient recording per fold, selected using registry metadata only, no acoustic features or outcomes',
          'limitations':'AIS classes are weak; short CPA spectra are not measured 15-minute trajectories. No acoustic absence calibration.'}
    (OUT/'external_preparation_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--hms-only',action='store_true');args=parser.parse_args()
    main(not args.hms_only)
