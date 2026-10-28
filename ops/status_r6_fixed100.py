"""Read-only snapshot of the active local fixed100 run; no source arrays."""
from pathlib import Path
import json,statistics
R=Path(__file__).resolve().parents[1]/'models/r6_fixed100_v1'
report={'phase':json.loads((R/'progress.json').read_text())['phase'],'arms':[]}
for p in sorted(R.glob('*.progress.json')):
 row=json.loads(p.read_text());n=row['arm'];h=[json.loads(s) for s in (R/n/'epochs.jsonl').read_text().splitlines() if s.strip()]
 seconds=[b['elapsed_seconds']-a['elapsed_seconds'] for a,b in zip(h[-9:-1],h[-8:])] if len(h)>=9 else []
 report['arms'].append({'name':n,'epoch':row['epoch'],'latest_train_RMSE':row['train_rmse'],'latest_INNER_RMSE':row['inner_rmse'],
 'best_window_RMSE':row['best_inner_window_rmse'],'best_window_epoch':row['best_window_epoch'],
 'best_macro_RMSE':row['best_inner_site_macro_rmse'],'best_macro_epoch':row['best_macro_epoch'],
 'lr':row['learning_rate'],'ES7_would_have_stopped':row['shadow_ES7_stop_epoch'],
 'peak_RSS_MiB':round(row['process_peak_RSS_bytes']/1048576,1),'recent_seconds_per_epoch':round(statistics.median(seconds),2) if seconds else None})
report['epochs_completed']=sum(v['epoch'] for v in report['arms']);report['epochs_required']=500
if (R/'FAILED.json').exists():report['failure']=json.loads((R/'FAILED.json').read_text())
print(json.dumps(report,ensure_ascii=False,indent=2))
