import sys, importlib.util, numpy as np
from common import *
spec = importlib.util.spec_from_file_location('sl', EV + 'splitnet/src/sl.py')
sl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sl)
nr = 8
F = np.exp(-2j * np.pi * np.outer(np.arange(nr), np.arange(nr)) / nr) / np.sqrt(nr)
s = sl.C['seeds'][0]
geo = sl.geometry(s)
p = sl.W_RAY ** 2
qx = geo['qx']
m = geo['ar'].__matmul__(F.T)
wb = p[:, None] * np.abs(m) ** 2
Pb = wb.sum(0)
d = np.linspace(0, 2, 401)
omni = np.array([np.sum(p * np.cos(2 * np.pi * qx * x)) for x in d])
beam = np.array([[(wb[:, b] * np.cos(2 * np.pi * qx * x)).sum() / Pb[b] for x in d] for b in range(nr)])
Hm = lambda x: (p * np.cos(2 * np.pi * qx * x))[:, None, None] * (m[:, :, None] * m[:, None, :].conj())
H0 = (p[:, None, None] * (m[:, :, None] * m[:, None, :].conj())).sum(0)
tr = np.trace(H0).real / nr
npos = [int((np.linalg.eigvalsh(Hm(x).sum(0)) > 0).sum()) for x in d]
nbeam = [(beam[:, i] > 0).sum() for i in range(len(d))]
(fig, ax) = plt.subplots(2, 1, figsize=(3.5, 3.0), sharex=True, gridspec_kw={'height_ratios': [1.6, 1], 'hspace': 0.08})
cs = plt.cm.viridis(np.linspace(0, 0.9, nr))
for b in range(nr):
    ax[0].plot(d, beam[b], color=cs[b], lw=0.8)
ax[0].plot(d, omni, color='k', lw=1.6, label='Omnidirectional')
ax[0].plot([], [], color=cs[4], lw=0.8, label='DFT beams (8)')
ax[0].axhline(0, color=C['gray'], lw=0.5)
for x in [0.4667, 0.85, 1.9]:
    ax[0].axvline(x, color=C['red'], ls=':', lw=0.7)
ax[0].set_ylabel('Re $\\phi_w(r)$')
ax[0].legend(fontsize=6, loc='lower left')
ax[0].set_ylim(-0.8, 1.05)
ax[1].step(d, npos, where='mid', color=C['blue'], label='$n_+$ of Herm$(\\bar B_r)$')
ax[1].plot(d, nbeam, '--', color=C['orange'], lw=0.9, label='beams with Re $\\phi_w>0$')
ax[1].set_ylabel('Count')
ax[1].set_xlabel('Residual $r/\\lambda$')
ax[1].legend(fontsize=6, loc='upper right')
ax[1].set_ylim(0, 8.5)
ax[1].set_yticks([0, 2, 4, 6, 8])
fig.savefig(OUT + 'beams.pdf', bbox_inches='tight')
fig.savefig(OUT + 'beams.png', dpi=200, bbox_inches='tight')
print('agree frac', np.mean(np.array(npos) == np.array(nbeam)))
