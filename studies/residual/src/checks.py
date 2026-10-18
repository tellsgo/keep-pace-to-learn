"""Independent scalar reconstructions; no new parameter grid or training arms."""
import os
for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
import sys, json, re, pathlib, hashlib
sys.dont_write_bytecode = True
import numpy as np
import core as E
R = E.R
C = E.C

def direct(phases, geom, res):
    out = np.zeros((len(phases), 4, 2), complex)
    for b in range(len(phases)):
        for u in range(4):
            for s in range(2):
                for l in range(460):
                    phase = phases[b, l] + 2 * np.pi * (geom['qr'][l, 1] * (0.5 * u) + geom['qt'][l, 1] * (0.5 * s) + geom['qr'][l, 0] * res - C['baseband_frequency_hz'] * geom['delays'][l])
                    out[b, u, s] += E.WEIGHTS[l] * complex(np.cos(phase), np.sin(phase))
    return out

def preflight():
    seed = C['diagnostic_seed']
    g = E.geometry(seed)
    rng = E.rng(seed, 'diagnostics')
    ph = rng.uniform(-np.pi, np.pi, (3, 460))
    text = (R / 'sources/TR_38_901_v18.txt').read_text()
    section = text.split('Table 7.7.1-1. CDL-A')[1].split('Per-Cluster Parameters')[0]
    vals = []
    for line in section.splitlines():
        parts = line.split()
        if len(parts) == 7:
            try:
                if 1 <= int(parts[0]) <= 23:
                    vals.append([float(v) for v in parts])
            except ValueError:
                pass
    table = np.array(vals)
    assert table.shape == (23, 7), table.shape
    assert np.array_equal(table, E.T)
    assert len(E.OFFSETS) == 20 and np.isclose(E.P.sum(), 1.0)
    assert np.allclose(np.linalg.norm(g['qr'], axis=1), 1, atol=1e-14)
    maxerr = 0.0
    cancellation = 0.0
    for v in C['speeds_kmh']:
        for k in C['kappas']:
            res = E.distance(v, k)
            a = E.channel(ph, g['steering'], g['qr'], res)
            b = direct(ph, g, res)
            maxerr = max(maxerr, float(np.max(abs(a - b))))
            ue = np.exp(2j * np.pi * g['qr'][:, 0] * (v / 3.6) * C['delay_s'] / E.LAM)
            ma = np.exp(-2j * np.pi * g['qr'][:, 0] * k * (v / 3.6) * C['delay_s'] / E.LAM)
            net = np.exp(2j * np.pi * g['qr'][:, 0] * res)
            cancellation = max(cancellation, float(np.max(abs(ue * ma - net))))
            if k == 1:
                assert np.array_equal(a, E.channel(ph, g['steering'], g['qr']))
    assert maxerr < 1e-12 and cancellation < 1e-13
    x = rng.choice([-1.0, 1.0], (29, 3))
    lab = (x[:, 0] * x[:, 1] > 0).astype(float)
    f = E.cn(rng, (29, 4, 2))
    nf = 0.1 * E.cn(rng, (29, 4))
    W = 0.1 * E.cn(rng, (2, 3))
    w = np.full(4, 0.8)
    b = -0.4

    def obj(W, w, b):
        tx = np.einsum('ni,bi->bn', W, x)
        y = np.einsum('bmn,bn->bm', f, tx) + nf
        z = np.tanh(abs(y) ** 2).dot(w) + b
        return float(np.mean(np.logaddexp(0, z) - lab * z) + 0.5 * C['receiver_l2'] * sum(w * w))
    y = np.einsum('bmn,bn->bm', f, np.einsum('ni,bi->bn', W, x)) + nf
    (_, delta, gw, gb, _) = E.task(y, lab, w, b)
    gx = np.einsum('bmn,bm->bn', f.conj(), delta)
    gW = np.einsum('bn,bi->ni', gx, x) / 29
    errs = []
    eps = 1e-06
    for ix in np.ndindex(W.shape):
        for part in [1, 1j]:
            d = np.zeros_like(W)
            d[ix] = part
            errs.append(abs((obj(W + eps * d, w, b) - obj(W - eps * d, w, b)) / (2 * eps) - np.vdot(gW, d).real))
    for i in range(4):
        d = np.zeros(4)
        d[i] = eps
        errs.append(abs((obj(W, w + d, b) - obj(W, w - d, b)) / (2 * eps) - gw[i]))
    errs.append(abs((obj(W, w, b + eps) - obj(W, w, b - eps)) / (2 * eps) - gb))
    assert max(errs) < 1e-07
    E.dump(R / 'results/preflight.json', {'passed': True, 'time': E.now(), 'source_table_exact': True, 'table_shape': list(table.shape), 'clusters': 23, 'rays_per_cluster': 20, 'direct_ray_sum_max_error': maxerr, 'doppler_reset_net_max_error': cancellation, 'gradient_fd_max_error': max(errs), 'lambda_m': E.LAM, 'displacements': [{'speed': v, 'kappa': k, 'lambda': E.distance(v, k), 'm': E.distance(v, k) * E.LAM} for v in C['speeds_kmh'] for k in C['kappas']]})
