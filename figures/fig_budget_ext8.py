"""Period-consistent common-budget comparison (elapsed training time). CIFAR-10, ResNet-18, Nr = 8 MMSE terminal with forward CSI,
three seeds. Trajectories were regenerated at each arm's exchange period; one update per period; computation, pilot and feedback
times are not modeled. 28 GHz equivalents: 5 ms (fixed, selection), 18.66 / 27.30 ms (mechanical exact restoration at 1 / 1.82 m/s)."""
import sys, pathlib, glob, json, re
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
from common import plt, C, EV
OUT = pathlib.Path(__file__).resolve().parent / 'generated'
OUT.mkdir(parents=True, exist_ok=True)
R = EV + 'cifar/results_ext8/'
PER = {'fixed': 5.0, 'sel': 5.0, 'mech': {'v1': 18.66, 'v1.82': 27.3}}
runs = {}
for f in glob.glob(R + 'ext8_*.json'):
    tag = re.sub('_\\d+\\.json$', '', f.split('/')[-1]).replace('ext8_', '')
    runs.setdefault(tag, []).append(json.load(open(f)))
(fig, axs) = plt.subplots(1, 3, figsize=(7.0, 2.15), sharey=True)
sty = {'fixed': dict(color=C['gray'], ls='-.', marker='o', mfc='white', label='Fixed antenna'), 'sel': dict(color=C['blue'], ls='-', marker='s', mfc='white', label='Selection'), 'mech': dict(color='k', ls='-', marker='^', mfc='white', label='Mechanical exact restoration')}
for (ax, (sp, title)) in zip(axs, [('v1', '(a) 1 m/s'), ('v1.82', '(b) 1.82 m/s'), ('v4.07', '(c) 4.07 m/s')]):
    for arm in ['fixed', 'sel', 'mech']:
        k = f'{sp}_{arm}'
        if k not in runs:
            continue
        per = PER[arm][sp] if arm == 'mech' else PER[arm]
        A = np.array([[e['test_acc'] * 100 for e in d['epochs']] for d in runs[k]])
        U = np.array([e['updates'] for e in runs[k][0]['epochs']])
        t = U * per / 1000
        st = sty[arm]
        ax.fill_between(t, A.min(0), A.max(0), color=st['color'], alpha=0.15, lw=0)
        ax.plot(t, A.mean(0), color=st['color'], ls=st['ls'], marker=st['marker'], mfc=st['mfc'], ms=2.6, markevery=4, lw=1.0, label=st['label'])
    if sp == 'v4.07':
        ax.text(0.97, 0.6, 'mechanical: beyond\nactuator speed', transform=ax.transAxes, ha='right', fontsize=6.5, color='k')
    ax.axhline(90, color=C['light'], lw=0.6, zorder=0)
    ax.set_xscale('log')
    ax.set_title(title)
    ax.set_xlabel('Wall-clock time (s)')
    ax.set_xticks([2, 10, 50, 200])
    ax.set_xticklabels(['2', '10', '50', '200'])
    ax.set_xlim(1.5, 320)
axs[0].set_ylabel('Test accuracy (%)')
axs[0].set_ylim(0, 100)
(h, l) = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.9), pad=0.3)
fig.savefig(OUT / 'budget_ext8.pdf')
fig.savefig(OUT / 'budget_ext8.png', dpi=200)
print('ok')
