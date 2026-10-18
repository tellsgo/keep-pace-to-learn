import json, glob, math, numpy as np, pathlib
R = pathlib.Path(__file__).resolve().parents[1]
D = {}
for f in glob.glob(str(R / 'results_rev1/*.json')):
    if f.endswith('analysis.json'):
        continue
    d = json.load(open(f))
    a = d['args']
    D.setdefault((a['geom'], a['enc'], a['res']), []).append(d)
mu = lambda g, r: min((math.cos(2 * math.pi * c * r) for c in {'sep': [0.1, -0.35, 0.45], 'along': [0.1, -0.35, 0.95]}[g]))

def T(runs):
    m = np.mean([[c['test_acc'] for c in d['curve']] for d in runs], 0)
    st = [c['step'] for c in runs[0]['curve']]
    return next((s for (s, v) in zip(st, m) if v >= 0.9), None)
tab = {}
for k in sorted(D):
    v = D[k]
    tab['%s_%s_r%s' % k] = {'n': len(v), 'mu': mu(k[0], k[2]), 'acc': float(np.mean([d['final_acc'] for d in v]) * 100), 'acc_sd': float(np.std([d['final_acc'] for d in v]) * 100), 'T90': T(v), 'align_mean': float(np.mean([d['align_mean'] for d in v])), 'align_min': float(np.min([d['align_min'] for d in v])), 'spow': float(np.mean([d['spow_last'] for d in v]))}
for (k, v) in tab.items():
    print(k, {a: round(b, 3) if isinstance(b, float) else b for (a, b) in v.items()})
sp = [tab[f'sep_path_r{float(r)}'] for r in [0, 0.1, 0.2, 0.3, 0.4, 0.47]]
q1 = [sp[0]['acc'] - x['acc'] for x in sp[1:]]
T0 = sp[0]['T90']
q2 = [(x['T90'], 1.5 * T0 / x['mu'] + 250, 3 * T0 / x['mu'] + 250) for x in sp]
q3 = [x['align_min'] - (x['mu'] - 0.01) for x in sp]
V = lambda s, f: 'SUPPORTED' if s else 'FALSIFIED' if f else 'NOT MET'
out = {'table': tab, 'Q1': {'gaps_pp': q1, 'verdict': V(all((g <= 2 for g in q1)), any((g > 10 for g in q1)))}, 'Q2': {'T_and_bounds': q2, 'verdict': V(all((t is not None and t <= b for (t, b, _) in q2)), any((t is None or t > c for (t, _, c) in q2)))}, 'Q3': {'margins': q3, 'verdict': V(all((m >= 0 for m in q3)), any((m < 0 for m in q3)))}}
json.dump(out, open(R / 'results_rev1/analysis.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ['Q1', 'Q2', 'Q3']}, indent=1))
