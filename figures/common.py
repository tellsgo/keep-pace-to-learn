import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.1, 'lines.markersize': 3.5, 'legend.frameon': False, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
C = dict(blue='#1f4e79', teal='#2a9d8f', orange='#e07a1f', red='#b23a48', gray='#6c757d', purple='#6a4c93', gold='#c9a227', light='#d9d9d9')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EV = str(ROOT / 'studies') + '/'
B1 = str(ROOT / 'studies/benchmark_refinement') + '/'
OUT = str(ROOT / 'figures/generated') + '/'
Path(OUT).mkdir(parents=True, exist_ok=True)

def load_arrays(path):
    import json
    import numpy as np

    class Arrays(dict):

        @property
        def files(self):
            return list(self)
    path = Path(path)
    if path.is_file():
        raw = json.loads(path.read_text())
    else:
        raw = {}
        for part in sorted(path.with_suffix('').glob('*.json')):
            raw.update(json.loads(part.read_text()))
        if not raw:
            raise FileNotFoundError(path.name)
    return Arrays({k: np.asarray(v['real']) + 1j * np.asarray(v['imag']) if v.get('complex') else np.asarray(v['data'], dtype=v['dtype']) for (k, v) in raw.items()})
