import json, glob, re, numpy as np, importlib.util
from common import *
R = EV + 'cifar/'

def load(d, strip):
    out = {}
    for f in glob.glob(R + d + '/*.json'):
        if f.endswith('analysis.json') or 'spectrum' in f:
            continue
        x = json.load(open(f))
        if 'arm' not in x:
            continue
        t = re.sub('_\\d+\\.json$', '', f.split('/')[-1]).replace(strip, '')
        out.setdefault(t, []).append(x['final_acc'] * 100)
    return out
E = load('results_ext4', 'ext4_')
F = load('results', '')
spec = importlib.util.spec_from_file_location('sl', EV + 'splitnet/src/sl.py')
sl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sl)
geos = [sl.geometry(s) for s in sl.C['seeds']]
d = np.linspace(0, 0.5, 201)
ph = np.array([np.mean([np.sum(sl.W_RAY ** 2 * np.cos(2 * np.pi * g['qx'] * x)) for g in geos]) for x in d])
rs = [0, 0.1, 0.2, 0.25, 0.3, 0.467]
ce = [np.mean(E[f'e_nr8_ma{r}']) if r != 0.467 else np.mean(E['e_nr8_fixed']) for r in [0.0, 0.1, 0.2, 0.25, 0.3, 0.467]]

def fk(r):
    for (k, v) in F.items():
        if not k.startswith('phys_'):
            continue
        x = 0.467 if k.split('_')[1] == 'fixed' else float(k.split('_')[1])
        if abs(x - r) < 0.002:
            return np.mean(v)
cm = [fk(r) for r in rs]
(fig, a) = plt.subplots(figsize=(3.5, 2.1))
a.plot(rs, ce, 'o-', color=C['red'], label='CSI-free terminal (energy)')
a.plot(rs, cm, 's-', color=C['blue'], label='MMSE terminal (forward CSI)')
a.axhline(np.mean(E['e_nr8_csi']), color=C['red'], ls=':', lw=0.8)
a.axhline(10, color=C['gray'], lw=0.5)
a.axvline(0.25, color=C['gray'], ls='-.', lw=0.7)
a.text(0.255, 40, '$\\lambda/4$', fontsize=7, color=C['gray'])
a.set_xlabel('Residual $r/\\lambda$')
a.set_ylabel('Final test accuracy (%)')
a.set_ylim(0, 100)
a.legend(fontsize=6, loc='lower left')
t = a.twinx()
t.spines['right'].set_visible(True)
t.plot(d, ph, color=C['gray'], ls='--', lw=0.8)
t.set_ylabel('Re $\\phi_m(r)$ (dashed)')
t.set_ylim(-1, 1.05)
t.axhline(0, color=C['gray'], lw=0.3)
fig.savefig(OUT + 'csifree.pdf', bbox_inches='tight')
fig.savefig(OUT + 'csifree.png', dpi=200, bbox_inches='tight')
print(np.round(ce, 2), np.round(cm, 2))
