import json, numpy as np
from common import *
R = EV + 'splitnet/'
A = json.load(open(R + 'results/analysis.json'))
cur = load_arrays(R + 'results/curves.json')
cfg = json.load(open(R + 'configs/config.json'))
E = json.load(open(R + 'results/posthoc_eig.json'))
seeds = cfg['seeds']
(fig, ax) = plt.subplots(1, 3, figsize=(7.16, 2.0), gridspec_kw={'wspace': 0.8})
a = ax[0]
sel = [('csi', 0.0, 'Ideal CSI', C['teal'], '-'), ('phys', 0.0, 'MA, $r=0$', C['blue'], '--'), ('phys', 0.2, '$r=0.2\\lambda$', C['purple'], '-'), ('phys', 0.3, '$r=0.3\\lambda$', C['orange'], '-'), ('phys', 0.4669897332774129, 'Fixed, $r=0.47\\lambda$', C['gray'], '-')]
for (arm, r, lab, c, ls) in sel:
    ks = [k for k in cur.files if k.startswith(tuple((f'{s}_{arm}_{r:.6f}' for s in seeds)))]
    m = np.median([cur[k][:, 1] for k in ks], axis=0) * 100
    a.plot(cur[ks[0]][:, 0], m, color=c, ls=ls, label=lab)
a.set_xlabel('Update index $k$')
a.set_ylabel('Median test accuracy (%)')
a.set_ylim(0, 100)
a.legend(fontsize=5.6, loc='lower right')
a.set_title('(a) Learning curves', loc='left')
a = ax[1]
t = A['table'][1:]
x = [u['r'] for u in t]
a.plot(x, [u['final_acc_mean'] * 100 for u in t], 'o-', color=C['blue'], label='Final accuracy (%)')
a.set_xlabel('Residual $r/\\lambda$')
a.set_ylabel('Final accuracy (%)')
a.set_ylim(75, 100)
b = a.twinx()
b.spines['right'].set_visible(True)
b.plot(x, [u['K85_median'] for u in t], 's--', color=C['orange'], mfc='white', label='Median $K_{85}$')
b.set_ylabel('Updates to 85%')
a.set_title('(b) Accuracy and speed', loc='left')
(h1, l1) = a.get_legend_handles_labels()
(h2, l2) = b.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=5.6, loc='center left')
a = ax[2]
rs = sorted(next(iter(E.values())).keys(), key=float)
ev = np.array([[np.mean([E[s][r][i] for s in E]) for i in range(4)] for r in rs])
xr = [float(r) for r in rs]
for (i, c) in enumerate(plt.cm.viridis(np.linspace(0, 0.85, 4))):
    a.plot(xr, ev[:, i], 'o-', color=c, ms=2.5, lw=0.9)
a.axhline(0, color=C['gray'], lw=0.5)
a.axvline(0.25, color=C['red'], ls='-.', lw=0.7)
a.text(0.255, 0.75, '$\\lambda/4$', color=C['red'], fontsize=7)
a.set_xlabel('Residual $r/\\lambda$')
a.set_ylabel('Eigenvalue')
a.set_title('(c) Combiner-output mean operator', loc='left')
fig.savefig(OUT + 'digits.pdf', bbox_inches='tight')
fig.savefig(OUT + 'digits.png', dpi=200, bbox_inches='tight')
