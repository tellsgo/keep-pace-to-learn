import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, sl
C = sl.C
eta = json.load(open(sl.R / 'results/calibration.json'))['chosen_eta']
assert eta
lam = C['light_speed_mps'] / C['carrier_hz']
fixed = C['speed_mps'] * C['tau_s'] / lam
arms = [('csi', 0.0)] + [('phys', r) for r in C['residual_lambda_grid']] + [('phys', fixed)]
rows = []
curves = {}
for s in C['seeds']:
    for (arm, r) in arms:
        (cur, cos) = sl.run(s, arm, r, eta)
        key = f'{s}_{arm}_{r:.6f}'
        curves[key] = cur
        rows.append({'seed': s, 'arm': arm, 'r': r, 'K85': sl.k_target(cur, C['target_acc']), 'final_acc': float(cur[-1, 1]), 'mean_cos': cos})
        print(rows[-1], flush=True)
(sl.R / 'results/rows.json').write_text(json.dumps(rows, indent=1) + '\n')
curve_dir = sl.R / 'results/curves'
curve_dir.mkdir(exist_ok=True)
for i, (key, arr) in enumerate(curves.items()):
    (curve_dir / f'{i:03}.json').write_text(json.dumps({key: {'data': arr.tolist(), 'dtype': str(arr.dtype)}}, separators=(',', ':')) + '\n')
legacy_curves = sl.R / 'results/curves.json'
if legacy_curves.exists():
    legacy_curves.unlink()
