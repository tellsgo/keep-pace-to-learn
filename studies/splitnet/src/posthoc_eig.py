"""POST-HOC: eigenvalues of the Hermitian part of the mean aged operator Bbar_r = E[M H_f H_r^H M^H] (4x4, normalized by
E tr(B_0)/4), per formal geometry; plus K85-vs-Re(rho_eff) slope over first-lobe residuals."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, sl
C = sl.C
g = np.random.default_rng(54321)
lam = C['light_speed_mps'] / C['carrier_hz']
fixed = C['speed_mps'] * C['tau_s'] / lam
grid = C['residual_lambda_grid'] + [fixed]
res = {}
for s in C['seeds']:
    geo = sl.geometry(s)
    B = {r: np.zeros((4, 4), complex) for r in grid}
    den = 0
    for _ in range(400):
        ph = g.uniform(-np.pi, np.pi, 460)
        pos = g.uniform(0, 100)
        Hf = sl.channel(geo, ph, pos)
        M = np.linalg.solve(Hf.conj().T.__matmul__(Hf) + 0.01 * np.eye(4), Hf.conj().T)
        den += np.real(np.trace(M.__matmul__(Hf).__matmul__(Hf.conj().T).__matmul__(M.conj().T))) / 4
        for r in grid:
            B[r] += M.__matmul__(Hf).__matmul__(sl.channel(geo, ph, pos + r).conj().T).__matmul__(M.conj().T)
    res[s] = {f'{r:.4f}': np.linalg.eigvalsh((B[r] + B[r].conj().T) / 2 / den).tolist() for r in grid}
json.dump(res, open(sl.R / 'results/posthoc_eig.json', 'w'), indent=2)
for r in grid:
    e = np.array([res[s][f'{r:.4f}'] for s in C['seeds']])
    print(f'r={r:.3f} max eig mean {e[:, -1].mean():+.3f} min eig mean {e[:, 0].mean():+.3f} trace/4 {e.mean(1).mean():+.3f}')
A = json.load(open(sl.R / 'results/analysis.json'))
rho = json.load(open(sl.R / 'results/posthoc_rho_eff.json'))
t = [x for x in A['table'][1:-1]]
x = np.log([rho[f"{u['r']:.4f}"]['re'] for u in t])
y = np.log([u['K85_median'] for u in t])
print('slope vs Re rho_eff', np.polyfit(x, y, 1)[0])
