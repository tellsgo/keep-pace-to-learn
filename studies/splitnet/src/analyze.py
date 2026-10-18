import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, sl
C = sl.C
R = sl.R
rows = json.load(open(R / 'results/rows.json'))
geos = [sl.geometry(s) for s in C['seeds']]

def phi(d):
    return complex(np.mean([sl.phi_theory(g, d) for g in geos]))
lam = C['light_speed_mps'] / C['carrier_hz']
fixed = C['speed_mps'] * C['tau_s'] / lam

def cell(arm, r):
    v = [x for x in rows if x['arm'] == arm and abs(x['r'] - r) < 1e-09]
    assert len(v) == 8, (arm, r, len(v))
    ks = np.array(sorted([x['K85'] if x['K85'] is not None else 10 ** 9 for x in v]))
    med = float((ks[3] + ks[4]) / 2) if ks[4] < 10 ** 9 else None
    return dict(arm=arm, r=r, re_phi=phi(r).real, abs_phi=abs(phi(r)), K85=[int(k) if k < 10 ** 9 else None for k in ks], K85_median=med, final_acc_mean=float(np.mean([x['final_acc'] for x in v])), final_acc_min=float(np.min([x['final_acc'] for x in v])), mean_cos=float(np.mean([x['mean_cos'] for x in v])))
tab = [cell('csi', 0.0)] + [cell('phys', r) for r in C['residual_lambda_grid']] + [cell('phys', fixed)]
use = [t for t in tab[1:-1] if t['K85_median'] is not None and t['re_phi'] > 0]
x = np.log([t['re_phi'] for t in use])
y = np.log([t['K85_median'] for t in use])
A = np.c_[x, np.ones_like(x)]
co = np.linalg.lstsq(A, y, rcond=None)[0]
r2 = 1 - np.sum((y - A.__matmul__(co)) ** 2) / np.sum((y - y.mean()) ** 2)
s = co[0]
P1 = 'FALSIFIED' if len(use) < 4 or s < -2 or s > -0.4 else 'SUPPORTED' if -1.4 <= s <= -0.6 else 'NOT MET'
d2 = abs(tab[1]['final_acc_mean'] - tab[0]['final_acc_mean']) * 100
P2 = 'SUPPORTED' if d2 <= 1 else 'FALSIFIED' if d2 > 3 else 'NOT MET'
fa = tab[-1]['final_acc_mean']
P3 = 'SUPPORTED' if fa <= 0.3 else 'FALSIFIED' if fa >= 0.6 else 'NOT MET'
out = dict(table=tab, fit=dict(slope=float(s), r2=float(r2), n=len(use)), P1=P1, P2=dict(diff_pp=d2, verdict=P2), P3=dict(final_acc=fa, verdict=P3))
json.dump(out, open(R / 'results/analysis.json', 'w'), indent=2)
for t in tab:
    print(t['arm'], round(t['r'], 3), round(t['re_phi'], 3), t['K85_median'], t['K85'], round(t['final_acc_mean'] * 100, 2), round(t['mean_cos'], 3))
print(out['fit'], P1, out['P2'], out['P3'])
