"""Verify an inbound copy of the already-uploaded CSV; never fit or submit."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = 'e32b04a2b876a66da6046e1ef1fdae6fd7e768eb9365bde5aa7227ad235c83fe'


def main():
    downloaded = Path('/Users/hongtaeyang/Downloads/submission.csv')
    original = ROOT / 'submissions/r5_fold0_raw_cnn_cpu_pilot_r1/submission.csv'
    raw = downloaded.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == EXPECTED == hashlib.sha256(original.read_bytes()).hexdigest()
    assert raw == original.read_bytes()
    rows = list(csv.reader(io.StringIO(raw.decode('utf-8'), newline='')))
    assert rows[0] == ['ID', 'share_added_vessel', 'score_added_vessel']
    assert len(rows) == 1501 and len({r[0] for r in rows[1:]}) == 1500
    folder = ROOT / 'audit/r7_uploaded_csv_v1'
    folder.mkdir(exist_ok=False)
    evidence = folder / 'submission_Vz5mjZEr.csv'
    shutil.copyfile(downloaded, evidence)
    result = {
        'status': 'passed', 'verified_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'Downloaded via the file link on the authenticated Zindi My submissions page using native browser UI',
        'submissions_page': 'https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/submissions',
        'submission_ID_observed_in_UI': 'Vz5mjZEr', 'displayed_public_score': '0.65199132',
        'downloaded_bytes': len(raw), 'downloaded_and_local_handoff_sha256': sha,
        'byte_for_byte_identical': True, 'rows': 1500, 'IDs_unique': True,
        'evidence_copy': str(evidence.relative_to(ROOT)),
        'prior_replay_report': 'reports/R7_R5_SUBMISSION_PATH_AUDIT_RESULTS.json',
        'inference_or_fit_executed_here': False, 'new_submission_or_external_message': False,
        'signed_download_URL_saved': False,
        'limitation': 'This closes the uploaded-file identity check, not the hidden-label accuracy or generator-representativeness question.'
    }
    out = ROOT / 'reports/R7_R5_UPLOADED_FILE_VERIFICATION.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
