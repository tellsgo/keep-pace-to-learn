import json, pathlib, numpy as np
R = pathlib.Path(__file__).resolve().parents[1]
C = json.loads((R / 'configs/config.json').read_text())
lam = C['light_speed_mps'] / C['carrier_hz']
fixed = C['speed_mps'] * C['tau_s'] / lam
rows = {}
for f in sorted((R / 'results').glob('*.json')):
    if f.name in ('analysis.json',):
        continue
    d = json.loads(f.read_text())
    key = (d['arm'], round(d['res'], 6))
    rows.setdefault(key, []).append(d)

def ep80(d):
    h = [e['epoch'] for e in d['epochs'] if e['test_acc'] >= C['target_acc']]
    return h[0] if h else None
tab = []
for (arm, res) in [('csi', 0.0), ('phys', 0.0), ('phys', 0.1), ('phys', 0.2), ('phys', 0.25), ('phys', 0.3), ('phys', round(fixed, 6))]:
    v = rows.get((arm, res), [])
    acc = [d['final_acc'] for d in v]
    e80 = [ep80(d) for d in v]
    tab.append(dict(arm=arm, r=res, n=len(v), final_acc=acc, mean=float(np.mean(acc)) if acc else None, epochs_to_80=e80, best_acc_mean=float(np.mean([max((e['test_acc'] for e in d['epochs'])) for d in v])) if v else None))
T = {(t['arm'], t['r']): t for t in tab}
csi = T['csi', 0.0]
z = T['phys', 0.0]
fx = T['phys', round(fixed, 6)]
out = dict(table=tab)
if all((t['n'] == 3 for t in tab)):
    d1 = abs(z['mean'] - csi['mean']) * 100
    out['P1'] = dict(diff_pp=d1, verdict='SUPPORTED' if d1 <= 1 else 'FALSIFIED' if d1 > 3 else 'NOT MET')

    def med(x):
        x = [e if e is not None else 10 ** 9 for e in x]
        return float(np.median(x))
    ratio = med(fx['epochs_to_80']) / med(z['epochs_to_80'])
    gap = (z['mean'] - fx['mean']) * 100
    out['P2'] = dict(gap_pp=gap, epoch_ratio=ratio, verdict='SUPPORTED' if gap >= 5 or ratio >= 3 else 'FALSIFIED' if abs(gap) <= 1 and ratio <= 1.5 else 'NOT MET')
    from scipy.stats import spearmanr
    rs = [t['r'] for t in tab[1:]]
    ms = [t['mean'] for t in tab[1:]]
    rho = float(spearmanr(rs, ms).statistic)
    out['P3'] = dict(spearman=rho, verdict='SUPPORTED' if rho <= -0.8 else 'FALSIFIED' if rho >= 0 else 'NOT MET')
(R / 'results/analysis.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
