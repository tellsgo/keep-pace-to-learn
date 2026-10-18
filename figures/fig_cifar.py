import json, glob, numpy as np
from common import *
R = EV + 'cifar/'

def load(pat, nr):
    out = []
    for f in glob.glob(R + pat):
        d = json.load(open(f))
        if 'arm' not in d:
            continue
        d['nr'] = d.get('nr', nr)
        out.append(d)
    return out
runs = load('results/*_*_*.json', 8) + load('results_ext1/*.json', None)

def pick(nr, arm, r):
    return [d for d in runs if d['nr'] == nr and d['arm'] == arm and (abs(d['res'] - r) < 0.002)]
_sp = json.load(open(R + 'results/spectrum_cifar_seeds.json'))['summary']
LMAX = {8: {0: _sp['nr8_nt4_0.0000']['lmax_mean'], 0.25: _sp['nr8_nt4_0.2500']['lmax_mean'], 0.467: _sp['nr8_nt4_0.4667']['lmax_mean'], 0.85: _sp['nr8_nt4_0.8500']['lmax_mean'], 1.9: _sp['nr8_nt4_1.9000']['lmax_mean']}, 2: {0: _sp['nr2_nt4_0.0000']['lmax_mean'], 0.25: _sp['nr2_nt4_0.2500']['lmax_mean'], 0.47: _sp['nr2_nt4_0.4667']['lmax_mean'], 0.6: _sp['nr2_nt4_0.6000']['lmax_mean'], 0.85: _sp['nr2_nt4_0.8500']['lmax_mean']}}
(fig, ax) = plt.subplots(1, 3, figsize=(7.16, 2.05), gridspec_kw={'wspace': 0.42})
a = ax[0]
for (nr, arm, r, lab, c, ls) in [(8, 'csi', 0, '$N_r{=}8$, CSI', C['teal'], '-'), (8, 'phys', 0.467, '$N_r{=}8$, fixed $0.47\\lambda$', C['gray'], '-'), (2, 'phys', 0, '$N_r{=}2$, MA $r{=}0$', C['blue'], '--'), (2, 'phys', 0.6, '$N_r{=}2$, $0.6\\lambda$', C['orange'], '-'), (2, 'phys', 0.85, '$N_r{=}2$, $0.85\\lambda$', C['red'], '-')]:
    v = pick(nr, arm, r)
    if v:
        A = np.array([[e['test_acc'] for e in d['epochs']] for d in v]) * 100
        a.plot(np.arange(1, A.shape[1] + 1), A.mean(0), color=c, ls=ls, label=lab)
a.set_xlabel('Epoch')
a.set_ylabel('Test accuracy (%)')
a.set_ylim(0, 100)
a.legend(fontsize=5.4, loc='center right', bbox_to_anchor=(1, 0.55))
a.set_title('(a) Learning curves', loc='left')
a = ax[1]
for (nr, mk, c) in [(8, 'o', C['blue']), (2, 's', C['orange'])]:
    rs = sorted(set((round(d['res'], 3) for d in runs if d['nr'] == nr and d['arm'] == 'phys')))
    m = []
    lo = []
    hi = []
    for r in rs:
        v = [d['final_acc'] * 100 for d in pick(nr, 'phys', r)]
        m.append(np.mean(v))
        lo.append(min(v))
        hi.append(max(v))
    a.errorbar(rs, m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt=mk + '-', color=c, capsize=1.5, label=f'$N_r={nr}$')
    cs = [d['final_acc'] * 100 for d in pick(nr, 'csi', 0)]
    if cs:
        a.axhline(np.mean(cs), color=c, ls=':', lw=0.8)
import re as _re
ext2 = {}
for f in glob.glob(R + 'results_ext2/*.json'):
    if f.endswith('analysis.json'):
        continue
    t = _re.sub('_\\d+\\.json$', '', f.split('/')[-1]).replace('ext2_', '')
    ext2.setdefault(t, []).append(json.load(open(f))['final_acc'] * 100)
for (t, x, c, lab) in [('v1.82_nr8_fixed', 0.85, C['blue'], 'speed-consistent'), ('v4.07_nr8_fixed', 1.9, C['blue'], None), ('v1.82_nr2_fixed', 0.85, C['orange'], None)]:
    if t in ext2:
        a.plot(x + 0.03, np.mean(ext2[t]), '^', color=c, mfc='white', ms=4, label=lab)
a.axvline(0.25, color=C['red'], ls='-.', lw=0.7)
a.set_xlabel('Residual $r/\\lambda$')
a.set_ylabel('Final test accuracy (%)')
a.legend(fontsize=5.6, loc='lower left')
a.set_title('(b) Final accuracy', loc='left')
a = ax[2]
for (nr, mk, c) in [(8, 'o', C['blue']), (2, 's', C['orange'])]:
    for (r, l) in LMAX[nr].items():
        v = [d['final_acc'] * 100 for d in pick(nr, 'phys', r)]
        if v:
            a.plot(l, np.mean(v), mk, color=c, label=f'$N_r={nr}$' if r == 0 else None)
a.axvline(0, color=C['gray'], lw=0.5)
a.set_xlabel('Largest eigenvalue of Herm$(\\bar B_r)$')
a.set_ylabel('Final test accuracy (%)')
a.legend(fontsize=5.8, loc='lower right')
a.set_title('(c) Accuracy vs. $\\lambda_{\\max}$', loc='left')
fig.savefig(OUT + 'cifar.pdf', bbox_inches='tight')
fig.savefig(OUT + 'cifar.png', dpi=200, bbox_inches='tight')
