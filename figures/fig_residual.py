"""Accuracy versus residual displacement on CDL-A (speed study + low-speed refinement) with the training-free spectrum."""
import sys, json, csv, importlib.util, numpy as np
from common import *
from scipy.special import j0
theory = json.load(open(EV + 'residual/results/theory.json'))
points = {row['residual_lambda']: row for row in theory['rows'] if row['residual_lambda'] <= 1.6}
d = np.array(sorted(points))
phi = np.array([points[x]['rho_real'] + 1j * points[x]['rho_imag'] for x in d])
rows = list(csv.DictReader(open(EV + 'residual/results/cells.csv')))
b2 = list(csv.DictReader(open(B1 + 'results/b2_summary.csv')))
(fig, ax) = plt.subplots(2, 1, figsize=(3.5, 3.3), sharex=True, gridspec_kw={'height_ratios': [1, 1.25], 'hspace': 0.08})
a = ax[0]
a.axhline(0, color=C['gray'], lw=0.5)
a.plot(d, phi.real, marker='o', linestyle='none', color=C['blue'], label='Re $\\phi(r)$, original points')
a.plot(d, np.abs(phi), marker='x', linestyle='none', color=C['blue'], lw=0.9, label='$|\\phi(r)|$, CDL-A')
analytic_d = np.linspace(0, 1.6, 1601)
a.plot(analytic_d, j0(2 * np.pi * analytic_d), color=C['gray'], ls='--', lw=0.8, label='$J_0(kr)$, planar isotropic')
z = 0.3517790636126388
for (x, lab, c) in [(0.25, '$\\lambda/4$', C['red']), (z, '$r_0$', C['blue'])]:
    a.axvline(x, color=c, lw=0.7, ls='-' if lab == '$r_0$' else '-.')
    a.text(x + 0.012, 0.86 if lab != 'RMS' else 0.66, lab, color=c, fontsize=7)
a.set_ylabel('Correlation')
a.set_ylim(-0.75, 1.05)
a.legend(loc='lower left', fontsize=6.0, handlelength=1.8, borderaxespad=0.2)
a = ax[1]
seen = set()
mk = {'3': ('o', C['teal']), '10': ('s', C['orange']), '30': ('^', C['purple'])}
for r in rows:
    if r['arm'] == 'CSI':
        continue
    x = float(r['residual_lambda'])
    if x > 1.6:
        continue
    (m, c) = mk[r['speed_kmh']]
    y = 100 * float(r['accuracy'])
    (lo, hi) = (100 * float(r['accuracy_ci_lo']), 100 * float(r['accuracy_ci_hi']))
    lab = f"{r['speed_kmh']} km/h" if r['speed_kmh'] not in seen else None
    seen.add(r['speed_kmh'])
    a.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=m, color=c, mfc='white' if r['arm'] == 'fixed' else c, capsize=1.2, elinewidth=0.6, label=lab, zorder=3)
xs = [float(r['residual_lambda']) for r in b2]
ys = [100 * float(r['accuracy_mean']) for r in b2]
lo = [100 * float(r['accuracy_ci_low']) for r in b2]
hi = [100 * float(r['accuracy_ci_high']) for r in b2]
a.errorbar(xs, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], fmt='D', color=C['red'], ms=2.8, capsize=1.2, elinewidth=0.6, label='3 km/h, refinement', zorder=4)
a.axvline(z, color=C['blue'], lw=0.7)
a.axvline(0.25, color=C['red'], lw=0.7, ls='-.')
for (lo_, hi_) in [(0, z), (0.8944161941957102, 1.4442224793679)]:
    a.axvspan(lo_, hi_, color=C['blue'], alpha=0.07, lw=0)
a.set_ylabel('Test accuracy (%)')
a.set_xlabel('Residual displacement $r/\\lambda$')
a.set_ylim(38, 102)
a.set_xlim(0, 1.6)
a.legend(loc='center right', fontsize=6.3, ncol=1, handletextpad=0.3)
fig.savefig(OUT + 'residual_law.pdf', bbox_inches='tight')
fig.savefig(OUT + 'residual_law.png', dpi=200, bbox_inches='tight')
