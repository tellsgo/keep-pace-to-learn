import core as E
import numpy as np
from scipy.optimize import brentq
import csv, json
R = E.R
C = E.C

def roots(q):
    d = np.linspace(0, 4, 40001)
    y = np.zeros(len(d))
    for x in np.asarray(q).reshape(-1, 460):
        for j in range(0, len(d), 1000):
            y[j:j + 1000] += np.cos(2 * np.pi * d[j:j + 1000, None] * x).dot(E.WEIGHTS ** 2) / len(q)
    ids = np.where(y[:-1] * y[1:] < 0)[0]
    f = lambda z: float(np.mean([np.sum(E.WEIGHTS ** 2 * np.cos(2 * np.pi * x * z)) for x in q]))
    return [float(brentq(f, d[j], d[j + 1], xtol=1e-14)) for j in ids]

def main():
    E.frozen('protocol_freeze.json')
    q = np.array([E.geometry(s)['qr'][:, 0] for s in C['seeds']])
    zs = roots(q)
    zs_seed = [roots(x[None, :]) for x in q]
    assert len(zs) >= 3
    rows = []
    per = []
    for v in C['speeds_kmh']:
        for k in C['kappas']:
            d = E.distance(v, k)
            ps = np.exp(2j * np.pi * q * d).dot(E.WEIGHTS ** 2)
            p = ps.mean()
            lobe = 1 if d < zs[0] else 2 if zs[1] < d < zs[2] else 0
            row = dict(speed_kmh=v, kappa=k, residual_m=d * E.LAM, residual_lambda=d, rho_real=float(p.real), rho_imag=float(p.imag), rho_abs=float(abs(p)), positive_lobe=lobe, kappa_star=1 - zs[0] / E.distance(v, 0), rho_real_seed_min=float(ps.real.min()), rho_real_seed_max=float(ps.real.max()))
            rows.append(row)
            per.extend([dict(seed=s, speed_kmh=v, kappa=k, rho_real=float(p0.real), rho_imag=float(p0.imag), rho_abs=float(abs(p0))) for (s, p0) in zip(C['seeds'], ps)])
    first = [x for x in rows if x['positive_lobe'] == 1 and x['kappa'] > 0]
    second = [x for x in rows if x['positive_lobe'] == 2 and x['kappa'] > 0]
    pairs = []
    for x in second:
        y = min(first, key=lambda y: (abs(x['rho_abs'] - y['rho_abs']), y['speed_kmh'], y['kappa']))
        pairs.append(dict(second=[x['speed_kmh'], x['kappa']], first=[y['speed_kmh'], y['kappa']], abs_phi_gap=abs(x['rho_abs'] - y['rho_abs'])))
    out = {'created_at': E.now(), 'lambda_m': E.LAM, 'zeroes_residual_lambda': zs, 'zeroes_by_seed': dict(zip(map(str, C['seeds']), zs_seed)), 'kappa_star': {str(v): 1 - zs[0] / E.distance(v, 0) for v in C['speeds_kmh']}, 'rows': rows, 'by_seed': per, 'K3_pairs': pairs}
    E.dump(R / 'results/theory.json', out)
    for (name, data) in [('theory_grid.csv', rows), ('theory_by_seed.csv', per)]:
        with (R / 'results' / name).open('w') as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys())
            w.writeheader()
            w.writerows(data)
    np.savez_compressed(R / 'results/theory_geometry.npz', qx=q, weights=E.WEIGHTS ** 2, seeds=C['seeds'])

if __name__ == '__main__':
    main()
