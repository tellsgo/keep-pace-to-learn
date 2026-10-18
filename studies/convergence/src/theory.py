"""Training-free theory for the convergence-rate study: phi(d) of the projected CDL-A spectrum,
projected RMS direction cosine, and the spectrum-free certificates r0>=lambda/4, r0>=lambda/(sqrt2*pi*sigma_m)."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, core as E
from scipy.optimize import brentq
R = E.R
C = E.C
P = E.WEIGHTS ** 2
rows = []
per = []
for s in C['seeds']:
    g = E.geometry(s)
    qx = g['qr'][:, 0]
    per.append(qx)

def phi(d, qx):
    return np.sum(P * np.exp(2j * np.pi * qx * d))

def phibar(d):
    return np.mean([phi(d, q) for q in per])
grid = np.linspace(0, 1.5, 15001)
re = np.array([phibar(d).real for d in grid])
z = [brentq(lambda d: phibar(d).real, grid[i], grid[i + 1]) for i in range(len(grid) - 1) if re[i] * re[i + 1] < 0]
sig2 = float(np.mean([np.sum(P * q ** 2) for q in per]))
cert_rms = 1 / (np.sqrt(2) * np.pi * np.sqrt(sig2))
out = {'seeds': C['seeds'], 'first_zeros_d': z[:3], 'sigma_m2': sig2, 'sigma_m': float(np.sqrt(sig2)), 'cert_universal_d': 0.25, 'cert_rms_d': float(cert_rms), 'isotropic_planar_d': 2.404825557695773 / (2 * np.pi), 'grid': [{'d': d, 're': float(phibar(d).real), 'im': float(phibar(d).imag), 'abs': float(abs(phibar(d)))} for d in C['residual_lambda_grid']]}
pz = []
for q in per:
    f = lambda d: phi(d, q).real
    r = [f(d) for d in grid]
    i = next((i for i in range(len(grid) - 1) if r[i] * r[i + 1] < 0))
    pz.append(brentq(f, grid[i], grid[i + 1]))
out['per_seed_first_zero_d'] = pz
pathlib.Path(R / 'results/theory.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
