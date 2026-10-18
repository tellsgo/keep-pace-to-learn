import json, numpy as np
from common import *
R = EV + 'convergence/'
A = json.load(open(R + 'results/analysis.json'))
cur = load_arrays(R + 'results/curves.json')
cfg = json.load(open(R + 'configs/config.json'))
ds = cfg['residual_lambda_grid']
seeds = cfg['seeds']
th = {g['d']: g for g in json.load(open(R + 'results/theory.json'))['grid']}
cm = plt.cm.viridis(np.linspace(0, 0.9, len(ds)))
(fig, ax) = plt.subplots(1, 3, figsize=(7.16, 2.05), gridspec_kw={'wspace': 0.38})
a = ax[0]
for (c, d) in zip(cm, ds):
    acc = np.median([cur[f'{s}_sgd_{d}'][:, 2] for s in seeds], axis=0) * 100
    a.plot(np.arange(len(acc)), acc, color=c, lw=0.9, label=f"{th[d]['re']:.2f}")
a.set_xlabel('Update index $k$')
a.set_ylabel('Median test accuracy (%)')
a.set_xlim(0, 400)
a.set_ylim(45, 100)
a.legend(title='Re $\\phi(r)$', fontsize=5.8, title_fontsize=6, ncol=2, loc='lower right', handlelength=1.2, columnspacing=0.6)
a.set_title('(a) SGD learning curves', loc='left')
a = ax[1]
for (c, d) in zip(cm, ds):
    acc = np.median([cur[f'{s}_sgd_{d}'][:, 2] for s in seeds], axis=0) * 100
    a.plot(np.arange(len(acc)) * th[d]['re'], acc, color=c, lw=0.9)
a.set_xlabel('Rescaled time $k\\,\\mathrm{Re}\\,\\phi(r)$')
a.set_xlim(0, 160)
a.set_ylim(45, 100)
a.set_title('(b) Rescaled by Re $\\phi$', loc='left')
a = ax[2]
t = A['sgd']['table']
x = np.array([u['re_phi'] for u in t])
y = np.array([u['K90_median'] for u in t])
f = A['sgd']['fit_re']
for u in t:
    ks = [k for k in u['K90_per_seed'] if k is not None]
    a.scatter([u['re_phi']] * len(ks), ks, s=4, color=C['light'], zorder=1)
a.loglog(x, y, 'o', color=C['blue'], label='SGD, median $K_{90}$', zorder=3)
xx = np.linspace(0.18, 1.05, 50)
a.loglog(xx, np.exp(f['intercept']) * xx ** f['slope'], color=C['blue'], lw=0.8, label=f"fit slope {f['slope']:.2f}")
ta = A['adam']['table']
xa = [u['re_phi'] for u in ta if u['K90_median']]
ya = [u['K90_median'] for u in ta if u['K90_median']]
a.loglog(xa, ya, 's', color=C['orange'], mfc='white', label='Adam, median $K_{90}$')
a.set_xlabel('Re $\\phi(r)$')
a.set_ylabel('Updates to 90% accuracy')
a.set_title('(c) Steps to target', loc='left')
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
a.xaxis.set_minor_formatter(NullFormatter())
a.xaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.5, 1]))
a.xaxis.set_major_formatter(FixedFormatter(['0.2', '0.3', '0.5', '1']))
a.legend(fontsize=5.8, loc='upper right')
fig.savefig(OUT + 'convergence.pdf', bbox_inches='tight')
fig.savefig(OUT + 'convergence.png', dpi=200, bbox_inches='tight')
