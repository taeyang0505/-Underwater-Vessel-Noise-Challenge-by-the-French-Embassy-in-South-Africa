"""Private, reproducible GPU handoff: transformed data, code and provenance."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parent
out=ROOT/'delivery';out.mkdir(exist_ok=True)
files=[]
for pattern in ['pipeline/*.py','reports/*','figures/pipeline_paper_final.png','eda/*.json','eda/tables/*.csv',
                'eda/figures/*.png','data/processed/*.npy','data/processed/*.parquet','data/processed/*.json',
                'data/synthetic/fullsources_cpu_a6d7f3a5_seed42/*','models/fullsources_cpu_a6d7f3a5_seed42/*',
                'models/weak_teacher/manifest.json','data/raw/maritime_starter_notebook.ipynb',
                'data/raw/Test.csv','data/raw/sample_submission.csv','external/*/manifest.json',
                'external/*/download_manifest.json','external/*/annotation_audit.json',
                'external/vtuad_v1.3.2/*SHA256SUMS.txt','external/vtuad_v1.3.2/dataset_card.md',
                'external/vtuad_v1.3.2/license_evidence_v131.md','external/hearmyship_v1/datacite.json',
                'external/noaa_sanctsound_holdout/*_minutes_power.npy','verification*.json',
                'requirements*','run_gpu.sh','external_data_declaration_DRAFT.txt','model_sizes.json']:
    files.extend(p for p in ROOT.glob(pattern) if p.is_file())
files=sorted(set(files),key=str)
manifest={'purpose':'private GPU training handoff, not public dataset redistribution',
    'raw_external_audio_included':False,'data_provenance_included':True,
    'declaration_posted':False,'official_submission_made':False,
    'GPU_execution':'not run; server not provisioned',
    'warning':'CPU model failed independent NOAA transfer assessment; further data validation required before claiming competitive performance',
    'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,
              'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
mp=out/'handoff_manifest.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(out/'gpu_handoff.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
    for p in files:z.write(p,str(p.relative_to(ROOT)))
    z.write(mp,'handoff_manifest.json')
print('Handoff',len(files),'files', (out/'gpu_handoff.zip').stat().st_size,'bytes',flush=True)
