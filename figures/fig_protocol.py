"""Anchor-and-return protocol: velocity profile and the minimum cycle period / stroke versus acceleration (analytic)."""
import numpy as np
from common import *
lam = 299792458 / 28000000000.0
tau = 0.005
V = 2.5

def tret(S, a):
    return np.where(S <= V ** 2 / a, 2 * np.sqrt(S / a), S / V + V / a)

def cyc(u, a):
    S = u * tau + u ** 2 / a
    return (S, tau + 2 * u / a + tret(S, a))
(fig, ax) = plt.subplots(1, 3, figsize=(7.16, 1.95), gridspec_kw={'wspace': 0.42, 'width_ratios': [1.15, 1, 1]})
a = ax[0]
v = 1.0
acc = 400.0
(S, T) = cyc(v, acc)
t1 = v / acc
t = [0, t1, t1 + tau, 2 * t1 + tau]
u = [0, v, v, 0]
tr = tret(S, acc)
tp = float(tr) / 2
up = min(V, acc * tp)
t2 = [2 * t1 + tau, 2 * t1 + tau + tp, 2 * t1 + tau + 2 * tp]
u2 = [0, -up, 0]
a.plot(np.array(t) * 1000.0, u, color=C['blue'])
a.plot(np.array(t2) * 1000.0, u2, color=C['orange'])
a.axvspan(t1 * 1000.0, (t1 + tau) * 1000.0, color=C['teal'], alpha=0.15, lw=0)
a.axhline(0, color=C['gray'], lw=0.5)
a.text((t1 + tau / 2) * 1000.0, 1.12, 'anchor\n($t_f\\to t_r$)', ha='center', fontsize=6.3, color=C['teal'])
a.text(t2[1] * 1000.0, -up - 0.35, 'return', ha='center', fontsize=6.3, color=C['orange'])
a.set_xlabel('Time (ms)')
a.set_ylabel('Counter-velocity $u(t)$ (m/s)')
a.set_ylim(-2.6, 1.6)
a.set_title('(a) One cycle, $v=1$ m/s, $a=400$ m/s$^2$', loc='left')
A = np.logspace(np.log10(40), np.log10(4000), 200)
cs = plt.cm.viridis(np.linspace(0, 0.85, 4))
for (c, v) in zip(cs, [0.5, 1, 1.5, 2]):
    (S, T) = cyc(v, A)
    ax[1].loglog(A, T * 1000.0, color=c, label=f'$v={v:g}$ m/s')
    ax[2].loglog(A, S * 1000.0, color=c)
    u = max(v - lam / (8 * tau), 0)
    (S2, T2) = cyc(u, A)
    ax[1].loglog(A, T2 * 1000.0, color=c, ls='--', lw=0.8)
    ax[2].loglog(A, S2 * 1000.0, color=c, ls='--', lw=0.8)
ax[1].set_xlabel('Acceleration limit $a$ (m/s$^2$)')
ax[1].set_ylabel('Min. cycle period $T$ (ms)')
ax[1].legend(fontsize=5.8, loc='upper right')
ax[1].set_title('(b) Cycle period', loc='left')
ax[2].set_xlabel('Acceleration limit $a$ (m/s$^2$)')
ax[2].set_ylabel('Rail stroke $S$ (mm)')
ax[2].set_title('(c) Stroke', loc='left')
ax[2].plot([], [], color='k', label='exact anchor')
ax[2].plot([], [], color='k', ls='--', lw=0.8, label='residual $\\lambda/8$')
ax[2].legend(fontsize=5.8, loc='upper right')
fig.savefig(OUT + 'protocol.pdf', bbox_inches='tight')
fig.savefig(OUT + 'protocol.png', dpi=200, bbox_inches='tight')
for v in [0.5, 1, 1.5, 2]:
    (S, T) = cyc(v, 400.0)
    u = max(v - lam / (8 * tau), 0)
    (S2, T2) = cyc(u, 400.0)
    print(v, 'exact S mm', round(float(S) * 1000, 2), 'T ms', round(float(T) * 1000.0, 1), '| lam/8 u', round(u, 3), 'S', round(float(S2) * 1000, 2), 'T', round(float(T2) * 1000.0, 1))
print('lam', lam, 'lam/(4tau)', lam / (4 * tau), 'lam/(8tau)', lam / (8 * tau))
