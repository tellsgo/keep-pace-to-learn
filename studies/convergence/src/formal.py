import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, conv, core as E
C = E.C
R = E.R
eta = json.load(open(R / 'results/calibration.json'))['chosen_eta']
assert eta == 0.005
rows = []
curves = {}
for s in C['seeds']:
    d = conv.world(s)
    for opt in ['sgd', 'adam']:
        for res in C['residual_lambda_grid']:
            (cur, cos) = conv.train(d, res, opt, eta, C['steps'])
            k = conv.k_target(cur)
            rows.append({'seed': s, 'opt': opt, 'd': res, 'K90': k, 'final_ce': float(cur[-1, 1]), 'final_acc': float(cur[-1, 2]), 'acc_at_120': float(cur[120, 2]), 'mean_grad_cos': cos})
            curves[f'{s}_{opt}_{res}'] = cur
            print(rows[-1], flush=True)
(R / 'results/rows.json').write_text(json.dumps(rows, indent=1) + '\n')
curve_dir = R / 'results/curves'
curve_dir.mkdir(exist_ok=True)
for i, (key, arr) in enumerate(curves.items()):
    (curve_dir / f'{i:03}.json').write_text(json.dumps({key: {'data': arr.tolist(), 'dtype': str(arr.dtype)}}, separators=(',', ':')) + '\n')
legacy_curves = R / 'results/curves.json'
if legacy_curves.exists():
    legacy_curves.unlink()
