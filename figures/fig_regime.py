"""Regime map: which realization compensates aging, by terminal speed and carrier (tau = 5 ms)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
from common import plt, C
OUT = pathlib.Path(__file__).resolve().parent / 'generated'
OUT.mkdir(parents=True, exist_ok=True)
tau = 0.005
c0 = 300000000.0
V = 2.5
Nmax = 16
v = np.logspace(np.log10(0.2), np.log10(40), 400)
f = np.logspace(np.log10(1000000000.0), np.log10(100000000000.0), 400)
(VV, FF) = np.meshgrid(v, f)
lam = c0 / FF
need = VV * tau > lam / 4
sel = VV * tau <= (Nmax - 1) * lam / 2 + lam / 4
mech = VV <= V
(fig, ax) = plt.subplots(figsize=(3.45, 2.35))
ax.contourf(VV, FF / 1000000000.0, np.where(~need, 0, np.where(sel & mech, 3, np.where(sel, 2, np.where(mech, 1, 4)))), levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], colors=['#f2f2f2', '#cfe3f3', '#d4eedf', '#b9d7c9', '#f4d6d6'])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Terminal speed $v$ (m/s)')
ax.set_ylabel('Carrier $f_c$ (GHz)')
ax.contour(VV, FF / 1000000000.0, need.astype(float), levels=[0.5], colors=[C['gray']], linewidths=0.7)
from matplotlib.patches import Patch
for (x, y, t, dx, dy, ha) in [(1, 28, 'pedestrian, 1 m/s', 0, -9, 'center'), (2, 28, 'robot, 2 m/s', 0, 4, 'center'), (8.33, 28, '30 km/h', 0, 4, 'center')]:
    ax.plot(x, y, 'k.', ms=3.5)
    ax.annotate(t, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=6.5, ha=ha)
ax.legend(handles=[Patch(fc='#f2f2f2', ec='0.6', lw=0.4, label='fixed antenna, $v\\tau<\\lambda/4$'), Patch(fc='#b9d7c9', ec='0.6', lw=0.4, label='selection or mechanical'), Patch(fc='#d4eedf', ec='0.6', lw=0.4, label='selection only'), Patch(fc='#f4d6d6', ec='0.6', lw=0.4, label='neither')], loc='lower right', fontsize=6.3, frameon=True, framealpha=0.92, edgecolor='0.7', handlelength=1.2, borderpad=0.4, labelspacing=0.3)
ax.set_xlim(0.2, 40)
ax.set_ylim(1, 100)
ax.set_yticks([1, 3.5, 10, 28, 100])
ax.set_yticklabels(['1', '3.5', '10', '28', '100'])
ax.set_xticks([0.2, 0.5, 1, 2.5, 10, 40])
ax.set_xticklabels(['0.2', '0.5', '1', '2.5', '10', '40'])
fig.tight_layout(pad=0.3)
fig.savefig(OUT / 'regime.pdf')
print('ok')
