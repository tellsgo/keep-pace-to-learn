"""Descriptive seed-bootstrap of the SGD P1 slope (resample the 8 seeds jointly across residuals; censored=+inf; any
resample whose median at a residual is censored drops that residual from that resample's fit)."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, core as E
R = E.R
C = E.C
th = {g['d']: g['re'] for g in json.load(open(R / 'results/theory.json'))['grid']}
rows = json.load(open(R / 'results/rows.json'))
ds = C['residual_lambda_grid']
seeds = C['seeds']
K = {(r['seed'], r['d']): r['K90'] if r['K90'] is not None else np.inf for r in rows if r['opt'] == 'sgd'}
g = E.rng(C['bootstrap_seed'], 'bootstrap')
sl = []
for _ in range(20000):
    ix = g.integers(0, 8, 8)
    x = []
    y = []
    for d in ds:
        k = np.sort([K[seeds[i], d] for i in ix])
        m = (k[3] + k[4]) / 2
        if np.isfinite(m):
            x.append(np.log(th[d]))
            y.append(np.log(m))
    if len(x) >= 4:
        sl.append(np.polyfit(x, y, 1)[0])
sl = np.array(sl)
out = {'n_valid': int(len(sl)), 'median': float(np.median(sl)), 'ci95': np.quantile(sl, [0.025, 0.975]).tolist()}
json.dump(out, open(R / 'results/slope_bootstrap.json', 'w'), indent=2)
print(out)
