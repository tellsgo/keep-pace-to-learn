import csv, numpy as np
from common import *
S = list(csv.DictReader(open(B1 + 'results/b1_summary.csv')))
Ct = list(csv.DictReader(open(B1 + 'results/b1_contrasts.csv')))
conds = [('A', '3'), ('A', '30'), ('C', '3'), ('C', '30')]
arms = ['S0', 'MA', 'polar_64', 'wait', 'CSI']
names = {'S0': 'Fixed antenna', 'MA': 'MA, exact restoration', 'polar_64': 'Polar correction (64 pilots)', 'wait': 'Waiting', 'CSI': 'Ideal CSI'}
cols = {'S0': C['gray'], 'MA': C['blue'], 'polar_64': C['orange'], 'wait': C['gold'], 'CSI': C['teal']}
(fig, ax) = plt.subplots(1, 2, figsize=(7.16, 1.9), gridspec_kw={'wspace': 0.25, 'width_ratios': [1.3, 1]})
w = 0.16
for (i, arm) in enumerate(arms):
    y = []
    lo = []
    hi = []
    for (p, v) in conds:
        r = [r for r in S if r['profile'] == p and r['speed_kmh'] == v and (r['arm'] == arm)][0]
        y.append(100 * float(r['accuracy_mean']))
        lo.append(100 * float(r['accuracy_ci_low']))
        hi.append(100 * float(r['accuracy_ci_high']))
    x = np.arange(4) + (i - 2) * w
    ax[0].bar(x, y, w, color=cols[arm], label=names[arm], yerr=[np.array(y) - lo, np.array(hi) - y], capsize=1, error_kw={'elinewidth': 0.5})
ax[0].set_xticks(range(4))
ax[0].set_xticklabels([f'CDL-{p}\n{v} km/h' for (p, v) in conds])
ax[0].set_ylim(40, 101)
ax[0].set_ylabel('Test accuracy (%)')
ax[0].legend(ncol=3, fontsize=5.6, loc='lower center', bbox_to_anchor=(0.5, -0.48), columnspacing=0.8, handlelength=1.2)
ax[0].set_title('(a) Accuracy', loc='left')
for (i, arm) in enumerate(['MA', 'polar_64', 'wait']):
    y = []
    lo = []
    hi = []
    for (p, v) in conds:
        r = [r for r in Ct if r['profile'] == p and r['speed_kmh'] == v and (r['arm'] == arm)][0]
        y.append(max(float(r['ce_gap_mean']), 1e-06))
        lo.append(max(float(r['ce_gap_ci_low']), 1e-06))
        hi.append(float(r['ce_gap_ci_high']))
    x = np.arange(4) + (i - 1) * 0.22
    ax[1].bar(x, y, 0.22, color=cols[arm], yerr=[np.array(y) - lo, np.array(hi) - y], capsize=1, error_kw={'elinewidth': 0.5})
ax[1].set_yscale('log')
ax[1].set_ylim(5e-07, 0.5)
ax[1].set_xticks(range(4))
ax[1].set_xticklabels([f'CDL-{p}\n{v} km/h' for (p, v) in conds])
ax[1].set_ylabel('CE excess over ideal CSI')
ax[1].set_title('(b) Paired CE excess (log scale)', loc='left')
ax[1].text(-0.38, 2e-06, 'MA: $|\\Delta|<10^{-5}$', fontsize=5.8, color=C['blue'], rotation=90, va='bottom')
fig.savefig(OUT + 'benchmark.pdf', bbox_inches='tight')
fig.savefig(OUT + 'benchmark.png', dpi=200, bbox_inches='tight')
