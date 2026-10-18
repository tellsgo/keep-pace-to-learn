"""POST-HOC (after formal results): training-free receiver-aware effective correlation
rho_eff(r) = E Re tr(B_r) / E tr(B_0), B_r = M H_f H_r^H M^H (MMSE combiner M of H_f), averaged over random fields/positions
and the 8 formal geometries. Isotropic gradient seeds at the combiner output are assumed (training-free)."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, sl
C = sl.C
g = np.random.default_rng(12345)
lam = C['light_speed_mps'] / C['carrier_hz']
fixed = C['speed_mps'] * C['tau_s'] / lam
grid = C['residual_lambda_grid'] + [fixed]
num = {r: 0j for r in grid}
den = 0.0
for s in C['seeds']:
    geo = sl.geometry(s)
    for _ in range(400):
        ph = g.uniform(-np.pi, np.pi, 460)
        pos = g.uniform(0, 100)
        Hf = sl.channel(geo, ph, pos)
        M = np.linalg.solve(Hf.conj().T.__matmul__(Hf) + 0.01 * np.eye(C['nt']), Hf.conj().T)
        den += np.real(np.trace(M.__matmul__(Hf).__matmul__(Hf.conj().T).__matmul__(M.conj().T)))
        for r in grid:
            Hr = sl.channel(geo, ph, pos + r)
            num[r] += np.trace(M.__matmul__(Hf).__matmul__(Hr.conj().T).__matmul__(M.conj().T))
out = {f'{r:.4f}': {'re': float((num[r] / den).real), 'im': float((num[r] / den).imag)} for r in grid}
json.dump(out, open(sl.R / 'results/posthoc_rho_eff.json', 'w'), indent=2)
print(json.dumps(out, indent=1))
