"""Rebuild the external expansion catalogue from the saved official evidence."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'external/expansion_research'
OTHER = BASE / 'other_sources'

def evidence(name):
    p = OTHER / name
    return {'file': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def zenodo(name):
    x = json.loads((OTHER / name).read_text())
    assert x['metadata']['license']['id'] == 'cc-by-4.0'
    return x

def main():
    portia = zenodo('19204539_latest.json')
    vliz = zenodo('17233667_latest.json')
    vulcano = zenodo('21561423.json')
    portia_counts = json.loads((OTHER / 'portia_metadata/verified_metadata_summary.json').read_text())
    noaa = json.loads((BASE / 'native_reference_final_summary.json').read_text())
    catalogue = {
        'retrieved_date': '2026-10-24',
        'public_declaration_status': 'draft only; not posted',
        'challenge_test_used': False,
        'previous_OC02_02_used': False,
        'sources': [
            {
                'name': 'NOAA SanctSound additional deployments',
                'licence': 'CC0-1.0',
                'source_urls': ['https://www.fisheries.noaa.gov/inport/item/80342', 'https://www.fisheries.noaa.gov/inport/item/80343'],
                'evidence': [evidence('noaa_raw_licence.html'), evidence('noaa_products_licence.html')],
                'status': '12 primary native 15-minute references verified; one NoShip control quarantined',
                'actual_clips': noaa['positive_native_clips'], 'actual_minutes': noaa['positive_native_minutes'],
                'actual_stations': noaa['unique_positive_stations'], 'actual_recording_days': noaa['unique_positive_days'],
                'header_fs_hz': [48000, 96000], 'native_15min_continuity_verified': True,
                'transfer_bytes_including_probes': noaa['network_range_bytes_including_initial_probes'],
                'adapter_manifest': 'external/expansion_research/native_reference_adapter.parquet',
                'limitation': 'Ship annotation is not isolated vessel sound or a verified hull identity. Only pinned source ranges, not full-file provider checksum, verified.',
            },
            {
                'name': 'PORTIA', 'licence': 'CC-BY-4.0',
                'source_urls': ['https://zenodo.org/records/' + str(portia['id']), 'https://www.nature.com/articles/s41597-026-08384-9'],
                'evidence': [evidence('19204539_latest.json')], 'pinned_record': str(portia['id']),
                'status': 'four metadata archives downloaded and full MD5 verified; audio not acquired',
                'public_metadata_counts': portia_counts,
                'audio_archive_bytes': sum(f['size'] for f in portia['files'] if f['key'].startswith('PORTIA_audio.')),
                'native_15min_continuity_verified': False,
                'next_gate': 'Audit original audio headers, sample rate, gaps and interval continuity. Do not concatenate short windows into a claimed measured passage.',
            },
            {
                'name': 'VLIZ AIS-annotated North Sea hydrophone recordings', 'licence': 'CC-BY-4.0',
                'source_urls': ['https://zenodo.org/records/' + str(vliz['id']), 'https://doi.org/10.14284/723'],
                'evidence': [evidence('17233667_latest.json')],
                'status': 'official metadata and licence verified; audio not acquired',
                'advertised_10sec_segments': 27524, 'advertised_recording_days': 116, 'advertised_stations': 2,
                'archive_bytes': sum(f['size'] for f in vliz['files']), 'native_15min_continuity_verified': False,
                'next_gate': 'Confirm native fs, actual frame length, timestamp spacing, vessel provenance and all 25-band support before use.',
            },
            {
                'name': 'MBARI Pacific Sound', 'licence': 'CC-BY-4.0 for data',
                'source_urls': ['https://www.mbari.org/data/passive-acoustic-data/', 'https://docs.mbari.org/pacific-sound/quickstart/'],
                'evidence': [evidence('mbari_licence.html')],
                'status': 'official licence and access documentation verified; audio not acquired',
                'documented_raw_fs_hz': 256000, 'approx_raw_bytes_per_10min': 460800000,
                'native_15min_continuity_verified': False,
                'next_gate': 'Join only header/timestamp-verified adjacent raw files; 16kHz and 2kHz decimated views cannot cover the complete 25 target bands.',
            },
            {
                'name': 'Vulcano Island hydrothermal field', 'licence': 'CC-BY-4.0',
                'source_urls': ['https://zenodo.org/records/' + str(vulcano['id'])],
                'evidence': [evidence('21561423.json')],
                'status': 'official licence and metadata verified; audio not acquired',
                'documented_fs_hz': 32000,
                'example_raw_archive_bytes': next(f['size'] for f in vulcano['files'] if f['key']=='Example_Raw_Audio_Frames.zip'),
                'native_15min_continuity_verified': False,
                'next_gate': 'Environmental diversity only; audit example native duration and gaps, and do not assert ship-positive or verified ship-absent truth.',
            },
        ],
    }
    out = ROOT / 'reports/external_expansion_catalog.json'
    out.write_text(json.dumps(catalogue, indent=2))
    print(str(out))

if __name__ == '__main__':
    main()
