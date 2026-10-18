import json, numpy as np
from common import *
from scipy.special import j0, i0e
bias = json.load(open(EV + 'position_bias/results/summary.json'))
rigid = json.load(open(EV + 'position_rigid/results/summary.json'))

def m(s, k):
    a = s[k]['accuracy']
    return (100 * a['mean'], 100 * a['ci95'][0], 100 * a['ci95'][1])
(fig, ax) = plt.subplots(1, 1, figsize=(3.5, 2.0))
ax = [ax]
a = ax[0]
xb = [0, 0.15, 0.383, 0.5]
xj = [0, 0.15, 0.383, 0.45]
for (xs, keys, c, mk, lab) in [(xb, ['MA_bias_0.0', 'MA_bias_0.15', 'MA_bias_0.383', 'MA_bias_0.5'], C['red'], 's', 'Fixed bias $b$'), (xj, ['MA_jitter_0.0', 'MA_jitter_0.15', 'MA_jitter_0.383', 'MA_jitter_0.45'], C['blue'], 'o', 'Independent Gaussian jitter $\\sigma$')]:
    v = np.array([m(bias, k) for k in keys])
    a.errorbar(xs, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk + '-', color=c, capsize=1.2, elinewidth=0.6, label=lab)
a.set_xlabel('Amplitude $b/\\lambda$ or $\\sigma/\\lambda$')
a.set_ylabel('Test accuracy (%)')
a.set_ylim(38, 102)
t = a.twinx()
t.spines['right'].set_visible(True)
xx = np.linspace(0, 0.5, 200)
t.plot(xx, j0(2 * np.pi * xx), color=C['red'], ls=':', lw=0.8)
aa = np.pi ** 2 * xx ** 2
t.plot(xx, i0e(aa), color=C['blue'], ls=':', lw=0.8)
t.axhline(0, color=C['gray'], lw=0.4)
t.set_ylim(-1.1, 1.1)
t.set_ylabel('Theory $\\bar\\rho$ (dotted)')
a.legend(fontsize=5.8, loc='lower left')
fig.savefig(OUT + 'bias_jitter.pdf', bbox_inches='tight')
fig.savefig(OUT + 'bias_jitter.png', dpi=200, bbox_inches='tight')
