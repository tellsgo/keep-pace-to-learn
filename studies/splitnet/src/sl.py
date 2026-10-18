"""Digits split-learning study over a CDL-A MIMO layer with reciprocal backward transmission (numpy only)."""
import os
for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[k] = '1'
import json, pathlib, warnings, numpy as np
warnings.filterwarnings('ignore', category=RuntimeWarning)
from sklearn.datasets import load_digits
R = pathlib.Path(__file__).resolve().parents[1]
C = json.loads((R / 'configs/config.json').read_text())
T = np.loadtxt(R / 'configs/cdl_a.csv', delimiter=',', skiprows=1)
OFFSETS = np.array([v * s for v in [0.0447, 0.1413, 0.2492, 0.3715, 0.5129, 0.6797, 0.8844, 1.1481, 1.5195, 2.1551] for s in [1, -1]])
P = 10 ** (T[:, 2] / 10)
P /= P.sum()
W_RAY = np.repeat(np.sqrt(P / 20), 20)
ROLES = ['coupling', 'split', 'init', 'fields', 'positions', 'fnoise', 'bnoise', 'batches', 'test']

def rng(seed, role):
    ss = np.random.SeedSequence(C['master_entropy'], spawn_key=(int(seed), ROLES.index(role)))
    return np.random.Generator(np.random.Philox(key=ss.generate_state(2, dtype=np.uint64)))

def unit(az, zen):
    az = np.deg2rad(az)
    zen = np.deg2rad(zen)
    return np.stack([np.sin(zen) * np.cos(az), np.sin(zen) * np.sin(az), np.cos(zen)], axis=-1)

def geometry(seed):
    g = rng(seed, 'coupling')
    ang = np.empty((23, 20, 4))
    for n in range(23):
        p = [g.permutation(20) for _ in range(3)]
        ang[n, :, 0] = T[n, 4] + 11 * OFFSETS
        ang[n, :, 1] = T[n, 6] + 3 * OFFSETS[p[0]]
        ang[n, :, 2] = T[n, 3] + 5 * OFFSETS[p[1]]
        ang[n, :, 3] = T[n, 5] + 3 * OFFSETS[p[2]]
    for j in [1, 3]:
        ang[:, :, j] %= 360
        ang[:, :, j] = np.where(ang[:, :, j] > 180, 360 - ang[:, :, j], ang[:, :, j])
    a = ang.reshape(460, 4)
    qr = unit(a[:, 0], a[:, 1])
    qt = unit(a[:, 2], a[:, 3])
    ar = np.exp(2j * np.pi * qr[:, 1, None] * 0.5 * np.arange(C['nr']))
    at = np.exp(2j * np.pi * qt[:, 1, None] * 0.5 * np.arange(C['nt']))
    return dict(qx=qr[:, 0], ar=ar, at=at)

def channel(geo, phases, pos):
    """H(pos) for ray phases (460,), terminal x-position pos in wavelengths: sum_l w e^{j(phi+2pi qx pos)} a_r a_t^T."""
    c = W_RAY * np.exp(1j * (phases + 2 * np.pi * geo['qx'] * pos))
    return np.einsum('l,lr,lt->rt', c, geo['ar'], geo['at'])

def phi_theory(geo, d):
    return np.sum(W_RAY ** 2 * np.exp(2j * np.pi * geo['qx'] * d))

def data(seed):
    D = load_digits()
    X = D.data.astype(float)
    y = D.target
    ix = rng(seed, 'split').permutation(len(X))
    (tr, te) = (ix[:C['train_n']], ix[C['train_n']:])
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-06
    return ((X[tr] - mu) / sd, y[tr], (X[te] - mu) / sd, y[te])

def init(seed):
    g = rng(seed, 'init')
    h = C['hidden_dev']
    hr = C['hidden_rx']
    return dict(W1=g.normal(0, 1 / np.sqrt(64), (h, 64)), b1=np.zeros(h), W2=g.normal(0, 1 / np.sqrt(h), (2 * C['nt'], h)), b2=np.zeros(2 * C['nt']), V1=g.normal(0, 1 / np.sqrt(2 * C['nt']), (hr, 2 * C['nt'])), c1=np.zeros(hr), V2=g.normal(0, 1 / np.sqrt(hr), (10, hr)), c2=np.zeros(10))

def device(p, X):
    h = np.tanh(X.__matmul__(p['W1'].T) + p['b1'])
    z = h.__matmul__(p['W2'].T) + p['b2']
    nt = C['nt']
    c = z[:, :nt] + 1j * z[:, nt:]
    nrm = np.linalg.norm(c, axis=1, keepdims=True)
    return (h, z, c, nrm, c / nrm)

def device_back(p, X, h, c, nrm, s, Gs):
    """Gs: packed real gradient dL/dRe s + j dL/dIm s per sample. Returns parameter gradients averaged over batch."""
    Gc = (Gs - s * np.real(np.sum(np.conj(s) * Gs, axis=1, keepdims=True))) / nrm
    gz = np.concatenate([Gc.real, Gc.imag], axis=1)
    gh = gz.__matmul__(p['W2']) * (1 - h ** 2)
    n = len(X)
    return dict(W2=gz.T.__matmul__(h) / n, b2=gz.mean(0), W1=gh.T.__matmul__(X) / n, b1=gh.mean(0))

def receiver(p, H, y, lab=None):
    M = np.linalg.solve(H.conj().T.__matmul__(H) + C['forward_noise_std'] ** 2 * np.eye(C['nt']), H.conj().T)
    sh = y.__matmul__(M.T)
    u = np.concatenate([sh.real, sh.imag], axis=1)
    t = np.tanh(u.__matmul__(p['V1'].T) + p['c1'])
    lg = t.__matmul__(p['V2'].T) + p['c2']
    lg -= lg.max(1, keepdims=True)
    pr = np.exp(lg)
    pr /= pr.sum(1, keepdims=True)
    if lab is None:
        return pr
    n = len(y)
    e = pr.copy()
    e[np.arange(n), lab] -= 1
    e /= n
    gV2 = e.T.__matmul__(t)
    gc2 = e.sum(0)
    gt = e.__matmul__(p['V2']) * (1 - t ** 2)
    gV1 = gt.T.__matmul__(u)
    gc1 = gt.sum(0)
    gu = gt.__matmul__(p['V1']) * n
    nt = C['nt']
    Gsh = gu[:, :nt] + 1j * gu[:, nt:]
    delta = Gsh.__matmul__(np.conj(M))
    loss = -np.mean(np.log(pr[np.arange(n), lab] + 1e-30))
    return (dict(V2=gV2, c2=gc2, V1=gV1, c1=gc1), delta, loss)

def cn(g, shape, std):
    return std * (g.normal(size=shape) + 1j * g.normal(size=shape)) / np.sqrt(2)

def evaluate(p, geo, Xte, yte, seed):
    g = rng(seed, 'test')
    acc = []
    for i in range(0, len(Xte), 32):
        X = Xte[i:i + 32]
        H = channel(geo, g.uniform(-np.pi, np.pi, 460), g.uniform(0, 100))
        (_, _, _, _, s) = device(p, X)
        y = s.__matmul__(H.T) + cn(g, (len(X), C['nr']), C['forward_noise_std'])
        acc.append(np.argmax(receiver(p, H, y), 1) == yte[i:i + 32])
    return float(np.mean(np.concatenate(acc)))

def run(seed, arm, res, eta, steps=None, eval_every=None, stop_at=None):
    """arm: 'phys' (residual res in wavelengths) or 'csi'. Returns curve [(step,acc)]."""
    steps = steps or C['steps']
    eval_every = eval_every or C['eval_every']
    (Xtr, ytr, Xte, yte) = data(seed)
    geo = geometry(seed)
    p = init(seed)
    gf = rng(seed, 'fields')
    gp = rng(seed, 'positions')
    gn = rng(seed, 'fnoise')
    gb = rng(seed, 'bnoise')
    gbt = rng(seed, 'batches')
    step_pos = C['speed_mps'] * C['cycle_s'] / (C['light_speed_mps'] / C['carrier_hz'])
    order = np.concatenate([gbt.permutation(len(Xtr)) for _ in range(steps * 32 // len(Xtr) + 2)])
    curve = [(0, evaluate(p, geo, Xte, yte, seed))]
    cos = []
    for k in range(steps):
        if k % C['block_updates'] == 0:
            phases = gf.uniform(-np.pi, np.pi, 460)
            pos = gp.uniform(0, 100)
        else:
            pos += step_pos
        ix = order[k * 32:(k + 1) * 32]
        X = Xtr[ix]
        lab = ytr[ix]
        (h, z, c, nrm, s) = device(p, X)
        Hf = channel(geo, phases, pos)
        nf = cn(gn, (32, C['nr']), C['forward_noise_std'])
        nb = cn(gb, (32, C['nt']), C['backward_noise_std'])
        y = s.__matmul__(Hf.T) + nf
        (grx, delta, loss) = receiver(p, Hf, y, lab)
        exact = delta.__matmul__(np.conj(Hf))
        if arm == 'csi':
            Gs = exact
        else:
            Hr = Hf if res == 0 else channel(geo, phases, pos + res)
            dn = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-30
            Gs = ((delta / dn).__matmul__(np.conj(Hr)) + nb) * dn
        cos.append(float(np.real(np.vdot(exact, Gs)) / (np.linalg.norm(exact) * np.linalg.norm(Gs) + 1e-30)))
        gdev = device_back(p, X, h, c, nrm, s, Gs)
        for (kk, v) in gdev.items():
            p[kk] -= eta * v
        assert all((np.isfinite(v).all() for v in p.values()))
        for (kk, v) in grx.items():
            p[kk] -= eta * C['rx_lr_scale'] * v
        if (k + 1) % eval_every == 0:
            a = evaluate(p, geo, Xte, yte, seed)
            curve.append((k + 1, a))
            if stop_at is not None and a >= stop_at:
                break
    return (np.array(curve), float(np.mean(cos)))

def k_target(curve, t):
    hit = [int(c[0]) for c in curve if c[1] >= t]
    return hit[0] if hit else None
