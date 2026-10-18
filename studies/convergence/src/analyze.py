import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, core as E
R = E.R
C = E.C
th = {g['d']: g for g in json.load(open(R / 'results/theory.json'))['grid']}
rows = json.load(open(R / 'results/rows.json'))
out = {}
for opt in ['sgd', 'adam']:
    tab = []
    for d in C['residual_lambda_grid']:
        ks = [r['K90'] if r['K90'] is not None else 10 ** 9 for r in rows if r['opt'] == opt and r['d'] == d]
        ks = np.array(sorted(ks))
        cens = int(np.sum(ks >= 10 ** 9))
        med = float(np.median(ks)) if cens <= len(ks) // 2 - 1 else 10 ** 9
        acc = [r['final_acc'] for r in rows if r['opt'] == opt and r['d'] == d]
        a120 = [r['acc_at_120'] for r in rows if r['opt'] == opt and r['d'] == d]
        tab.append({'d': d, 're_phi': th[d]['re'], 'abs_phi': th[d]['abs'], 'K90_per_seed': [int(k) if k < 10 ** 9 else None for k in ks], 'K90_median': med if med < 10 ** 9 else None, 'censored_seeds': cens, 'final_acc_mean': float(np.mean(acc)), 'acc120_mean': float(np.mean(a120))})
    use = [t for t in tab if t['K90_median'] is not None]

    def fit(key):
        x = np.log([t[key] for t in use])
        y = np.log([t['K90_median'] for t in use])
        A = np.c_[x, np.ones_like(x)]
        (coef, res, _, _) = np.linalg.lstsq(A, y, rcond=None)
        yhat = A.__matmul__(coef)
        r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
        return {'slope': float(coef[0]), 'intercept': float(coef[1]), 'r2': float(r2), 'n': len(use)}
    f_re = fit('re_phi')
    f_abs = fit('abs_phi')
    verdict = None
    if opt == 'sgd':
        s = f_re['slope']
        if len(use) < 4 or s < -2.0 or s > -0.4:
            verdict = 'FALSIFIED'
        elif -1.4 <= s <= -0.6:
            verdict = 'SUPPORTED'
        else:
            verdict = 'NOT MET, NOT FALSIFIED'
    out[opt] = {'table': tab, 'fit_re': f_re, 'fit_abs': f_abs, 'P1_verdict': verdict}
json.dump(out, open(R / 'results/analysis.json', 'w'), indent=2)
print(json.dumps(out, indent=1))
