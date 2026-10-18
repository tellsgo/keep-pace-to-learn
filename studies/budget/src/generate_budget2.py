"""Post-hoc CIFAR timing only. Run with python3; no training or network access."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
OUT = Path(__file__).resolve().parents[1] / 'results'
OUT.mkdir(parents=True, exist_ok=True)
ROOT = OUT.parents[2]
INP = ROOT / 'studies/cifar'
(ROOT / 'figures/generated').mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
(TAU, SWITCH, A, V) = (0.005, 1e-06, 400.0, 2.5)
LAM = 299792458 / 28000000000.0
SPEEDS = [(1.0, '1'), (1.8202, '1.82'), (4.0686, '4.07')]
SEEDS = [710172100, 710172101, 710172102]
METHODS = ['fixed', 'selection', 'mechanical exact restoration']
manifest = {}

def read(path):
    raw = path.read_bytes()
    manifest[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)
cfg = read(INP / 'configs/config.json')
assert cfg['seeds'] == SEEDS and cfg['batch'] == 128 and (cfg['epochs'] == 40)

def timing(v, method):
    if method == 'fixed':
        return (TAU, TAU, 'feasible', None)
    if method == 'selection':
        return (TAU + SWITCH, TAU + SWITCH, 'feasible', None)
    if v > V:
        return (None, TAU, 'ideal reference; actuator speed exceeded', None)
    stroke = v * TAU + v * v / A
    ret = 2 * math.sqrt(stroke / A) if stroke <= V * V / A else stroke / V + V / A
    period = TAU + 2 * v / A + ret
    return (period, period, 'feasible with variable cycle', stroke)

def source_paths(v, display, nr, method):
    if method == 'selection':
        ext = 7 if v == 1 else 6
        return sorted((INP / f'results_ext{ext}').glob(f'ext{ext}_v{display}_nr{nr}_sel_*.json'))
    if v == 1:
        if nr != 8:
            return []
        arm = 'phys_fixed' if method == 'fixed' else 'phys_0.0'
        return sorted((INP / 'results').glob(arm + '_*.json'))
    arm = 'fixed' if method == 'fixed' else 'ma'
    return sorted((INP / 'results_ext2').glob(f'ext2_v{display}_nr{nr}_{arm}_*.json'))
(rows, groups, points) = ([], {}, [])
for nr in [8, 2]:
    for (v, display) in SPEEDS:
        for method in METHODS:
            (actual, mapped, status, stroke) = timing(v, method)
            paths = source_paths(v, display, nr, method)
            row = dict(nr=nr, speed_label_mps=display, speed_mps=v, method=method, data_status='recorded curves' if paths else 'missing curves', seed_count=len(paths), final_accuracy_mean_pct='missing data', final_accuracy_min_pct='missing data', final_accuracy_max_pct='missing data', actual_cycle_ms=actual * 1000 if actual else 'unreachable', time_mapping_cycle_ms=mapped * 1000, physical_status=status, stroke_mm=stroke * 1000 if stroke else '', final_updates='missing data', time_to_80_s='missing data', updates_to_80='missing data', time_to_90_s='missing data', updates_to_90='missing data', source_files=';'.join((str(p.relative_to(ROOT)) for p in paths)), notes='')
            if paths:
                assert len(paths) == 3
                curves = [read(p) for p in paths]
                assert sorted((d['seed'] for d in curves)) == SEEDS
                for d in curves:
                    assert d['arm'] == 'phys' and d.get('project', 'none') == 'none'
                    assert d.get('nr', cfg['nr']) == nr and d.get('nt', cfg['nt']) == 4
                    assert d.get('speed', cfg['speed_mps']) == v
                    assert len(d['epochs']) == 40
                    assert [e['epoch'] for e in d['epochs']] == list(range(1, 41))
                    assert [e['updates'] for e in d['epochs']] == list(range(390, 15601, 390))
                    assert d['final_acc'] == d['epochs'][-1]['test_acc']
                    expected = v * TAU / LAM if method == 'fixed' else 0
                    if method == 'selection':
                        expected = v * TAU / LAM - round(v * TAU / LAM / 0.5) * 0.5
                    assert abs(d['res'] - expected) < 5e-05
                ys = np.array([[e['test_acc'] for e in d['epochs']] for d in curves]) * 100
                assert np.isfinite(ys).all() and ((ys >= 0) & (ys <= 100)).all()
                updates = np.arange(390, 15601, 390)
                (mean, low, high) = (ys.mean(0), ys.min(0), ys.max(0))
                groups[nr, v, method] = dict(updates=updates, mean=mean, low=low, high=high, time=updates * mapped, ideal=actual is None)
                row.update(final_accuracy_mean_pct=float(mean[-1]), final_accuracy_min_pct=float(low[-1]), final_accuracy_max_pct=float(high[-1]), final_updates=15600)
                for target in [80, 90]:
                    hits = np.flatnonzero(mean >= target)
                    k = int(updates[hits[0]]) if len(hits) else None
                    row[f'updates_to_{target}'] = k if k else 'not reached'
                    row[f'time_to_{target}_s'] = k * mapped if k else 'not reached'
                for (i, k) in enumerate(updates):
                    points.append(dict(nr=nr, speed_mps=v, method=method, updates=int(k), time_s=k * mapped, mean_pct=mean[i], min_pct=low[i], max_pct=high[i], **{f"seed_{d['seed']}_pct": ys[j, i] for (j, d) in enumerate(curves)}))
            elif nr == 2 and v == 1:
                row.update(data_status='reported endpoint only; curves missing', final_accuracy_mean_pct=85.5 if method == 'fixed' else 93.77, source_files='studies/cifar/SPEC_ADDENDUM_ext7.md', notes='Rounded reference mean from Ext7 SPEC; seed values and crossing times unavailable.')
            if actual is None:
                row['notes'] += ' Any listed time uses the ideal reference cycle tau, not executable mechanical time.'
            rows.append(row)

def write_csv(name, data):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
write_csv('budget_cifar.csv', rows)
write_csv('curve_points.csv', points)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7.5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.6})
(fig, axes) = plt.subplots(1, 3, figsize=(7, 2.2), sharex=True, sharey=True)
fig.subplots_adjust(left=0.072, right=0.987, top=0.76, bottom=0.255, wspace=0.13)
styles = {'fixed': dict(color='#777777', ls='-.', marker='o'), 'selection': dict(color='#1766a3', ls='-', marker='s'), 'mechanical exact restoration': dict(color='#161616', ls='-', marker='^')}
handles = []
for (panel, (ax, (v, display))) in enumerate(zip(axes, SPEEDS)):
    for method in METHODS:
        g = groups[8, v, method]
        style = dict(styles[method])
        if g['ideal']:
            style['ls'] = '--'
        ax.fill_between(g['time'], g['low'], g['high'], color=style['color'], alpha=0.14, lw=0)
        (line,) = ax.plot(g['time'], g['mean'], **style, lw=1.05, markersize=2.5, markevery=6, markerfacecolor='white', markeredgewidth=0.65, label=method)
        if panel == 0:
            handles.append(line)
    ax.set_xscale('log')
    ax.set_xlim(1.7, 480)
    ax.set_ylim(0, 102)
    ax.set_xticks([2, 10, 50, 200])
    ax.set_xticklabels(['2', '10', '50', '200'])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(which='both', width=0.5, length=2.5, pad=2)
    ax.grid(axis='y', color='.9', lw=0.45)
    ax.set_axisbelow(True)
    ax.set_title(f'({chr(97 + panel)}) {display} m/s', pad=4)
    if v > V:
        ax.text(0.97, 0.67, 'Dashed: ideal reference', transform=ax.transAxes, ha='right', fontsize=7, bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1))
axes[0].set_ylabel('Test accuracy (%)', labelpad=3)
fig.legend(handles=handles, labels=['Fixed', 'Selection', 'Mechanical exact restoration'], ncol=3, loc='upper center', bbox_to_anchor=(0.52, 1.005), frameon=False, handlelength=2.5, columnspacing=2, borderaxespad=0.2)
fig.text(0.53, 0.105, 'Wall-clock time (s)', ha='center', fontsize=8)
fig.text(0.53, 0.018, 'CIFAR-10, Nr = 8; mean and min–max of 3 seeds; post-hoc timing.', ha='center', fontsize=7)
fig.savefig(ROOT / 'figures/generated' / 'budget_cifar.pdf', metadata={'Title': 'CIFAR-10: accuracy versus post-hoc wall-clock time', 'Subject': 'Nr=8, 3 seeds, min-max; no retraining'})
fig.savefig(ROOT / 'figures/generated' / 'budget_cifar.png', dpi=240)
assert len(fig.legends) == 1 and all((ax.get_legend() is None for ax in axes))
assert min((t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text())) >= 7
plt.close(fig)

def num(x):
    return f'{x:.3f}' if isinstance(x, (int, float)) else x

for path in [INP / 'SPEC.md', INP / 'SPEC_ADDENDUM_ext2.md', INP / 'SPEC_ADDENDUM_ext6.md', INP / 'SPEC_ADDENDUM_ext7.md']:
    manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
for (path, digest) in manifest.items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
(OUT / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
assert len(rows) == 18 and len(groups) == 13 and (len(points) == 520)
validation = dict(recorded_groups=13, recorded_seed_curves=39, points_per_curve=40, missing_curve_groups=5, csv_rows=18, figure_inches=[7, 2.2], minimum_font_pt=7, shared_legends=1, input_hashes_unchanged=True, metadata_epoch_update_final_checks=True, no_training=True, threshold_rule='first recorded seed-mean crossing, no interpolation', python_libraries={'numpy': np.__version__, 'matplotlib': matplotlib.__version__})
(OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
print(json.dumps(validation, indent=2))
for r in rows:
    print(r['nr'], r['speed_mps'], r['method'], r['final_accuracy_mean_pct'], r['time_to_90_s'])
