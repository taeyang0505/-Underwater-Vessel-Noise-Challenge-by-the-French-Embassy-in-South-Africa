"""Post-inference, diagnostic-only paired law contrasts; no policy fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/r19_multislice_law_transfer_r1'


def main():
    source = OUT / 'inference/LAW_TRANSFER_ROW_PREDICTIONS.csv'
    data = pd.read_csv(source)
    vanilla = data[data.law.eq('vanilla_addition')].set_index('sample_id')
    assert len(vanilla) == 2000
    result = []
    max_zero = 0.
    laws = data.loc[~data.law.eq('vanilla_addition'), 'law'].unique()
    for law in [*laws, 'six_custom_mixture']:
        custom = data[~data.law.eq('vanilla_addition') if law == 'six_custom_mixture' else data.law.eq(law)].set_index('sample_id')
        assert not custom.index.duplicated().any()
        original = vanilla.loc[custom.index]
        yc = custom.target_share.to_numpy()
        yv = original.target_share.to_numpy()
        assert np.array_equal(yc > 0, yv > 0)
        groups = {'all': np.ones(len(yv), bool), 'y=0': yv == 0,
                  'vanilla_0<y<=0.05': (yv > 0) & (yv <= .05),
                  'vanilla_0.05<y<=0.15': (yv > .05) & (yv <= .15),
                  'vanilla_y>0.15': yv > .15,
                  'vanilla_0<y<=0.10': (yv > 0) & (yv <= .10)}
        for candidate in ['old_B1', 'new_B2']:
            pc, pv = custom[candidate + '_share'].to_numpy(), original[candidate + '_share'].to_numpy()
            for kind in ['share', 'score']:
                z = yv == 0
                gap = np.max(np.abs(custom[candidate + '_' + kind].to_numpy()[z] - original[candidate + '_' + kind].to_numpy()[z]))
                max_zero = max(max_zero, float(gap))
            for group, ix in groups.items():
                assert ix.any()
                ec, ev = pc[ix] - yc[ix], pv[ix] - yv[ix]
                result.append({'custom_law': law, 'candidate': candidate, 'fixed_source_slice': group,
                               'n': int(ix.sum()), 'custom_RMSE': float(np.sqrt(np.mean(ec**2))),
                               'vanilla_RMSE': float(np.sqrt(np.mean(ev**2))),
                               'custom_minus_vanilla_RMSE': float(np.sqrt(np.mean(ec**2)) - np.sqrt(np.mean(ev**2))),
                               'custom_bias': float(np.mean(ec)), 'vanilla_bias': float(np.mean(ev)),
                               'custom_minus_vanilla_prediction_mean': float(np.mean(pc[ix] - pv[ix])),
                               'custom_minus_vanilla_target_mean': float(np.mean(yc[ix] - yv[ix])),
                               'custom_minus_vanilla_bias': float(np.mean(ec - ev))})
    assert max_zero == 0
    target = OUT / 'PAIRED_FIXED_VANILLA_BIN_EFFECTS.csv'
    pd.DataFrame(result).to_csv(target, index=False)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / 'PAIRED_FIXED_BIN_VERIFICATION.json').write_text(json.dumps({
        'status': 'passed', 'source_sha256': sha(source), 'output_sha256': sha(target),
        'source': str(source.relative_to(ROOT)), 'paired_rows_total': 2000,
        'max_zero_share_or_score_prediction_difference': max_zero,
        'slice_definition': 'Same source IDs selected by vanilla target; custom targets can change under law',
        'selection_or_calibration_performed': False, 'new_training': False,
        'new_model_inference': False, 'Test_row_access': False}, ensure_ascii=False, indent=2) + '\n')
    print(pd.DataFrame(result).query("fixed_source_slice=='all'").to_string(index=False))


if __name__ == '__main__':
    main()
