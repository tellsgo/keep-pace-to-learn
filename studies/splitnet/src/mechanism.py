"""Projection/intervention test (SPEC_ADDENDUM_mechanism.md). Fixed-antenna arm of the digits study."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, sl
C = sl.C
R = sl.R
lam = C['light_speed_mps'] / C['carrier_hz']
FIXED = C['speed_mps'] * C['tau_s'] / lam
ETA = json.load(open(R / 'results/calibration.json'))['chosen_eta']

def projectors(seed):
    g = np.random.default_rng(777)
    geo = sl.geometry(seed)
    B = np.zeros((4, 4), complex)
    for _ in range(400):
        ph = g.uniform(-np.pi, np.pi, 460)
        pos = g.uniform(0, 100)
        Hf = sl.channel(geo, ph, pos)
        M = np.linalg.solve(Hf.conj().T.__matmul__(Hf) + 0.01 * np.eye(4), Hf.conj().T)
        B += M.__matmul__(Hf).__matmul__(sl.channel(geo, ph, pos + FIXED).conj().T).__matmul__(M.conj().T)
    (w, V) = np.linalg.eigh((B + B.conj().T) / 2)
    Vp = V[:, w > 0]
    n = Vp.shape[1]
    gr = np.random.default_rng(778)
    (Q, _) = np.linalg.qr(gr.normal(size=(4, n)) + 1j * gr.normal(size=(4, n)))
    return (Vp.__matmul__(Vp.conj().T), Q.__matmul__(Q.conj().T), w.tolist())

def run(seed, mode, Pp, Pr):
    (Xtr, ytr, Xte, yte) = sl.data(seed)
    geo = sl.geometry(seed)
    p = sl.init(seed)
    gf = sl.rng(seed, 'fields')
    gp = sl.rng(seed, 'positions')
    gn = sl.rng(seed, 'fnoise')
    gb = sl.rng(seed, 'bnoise')
    gbt = sl.rng(seed, 'batches')
    step_pos = C['speed_mps'] * C['cycle_s'] / lam
    steps = C['steps']
    order = np.concatenate([gbt.permutation(len(Xtr)) for _ in range(steps * 32 // len(Xtr) + 2)])
    frac = []
    for k in range(steps):
        if k % C['block_updates'] == 0:
            phases = gf.uniform(-np.pi, np.pi, 460)
            pos = gp.uniform(0, 100)
        else:
            pos += step_pos
        ix = order[k * 32:(k + 1) * 32]
        X = Xtr[ix]
        lab = ytr[ix]
        (h, z, c, nrm, s) = sl.device(p, X)
        Hf = sl.channel(geo, phases, pos)
        nf = sl.cn(gn, (32, C['nr']), C['forward_noise_std'])
        nb = sl.cn(gb, (32, C['nt']), C['backward_noise_std'])
        y = s.__matmul__(Hf.T) + nf
        (grx, delta, loss) = sl.receiver(p, Hf, y, lab)
        M = np.linalg.solve(Hf.conj().T.__matmul__(Hf) + C['forward_noise_std'] ** 2 * np.eye(C['nt']), Hf.conj().T)
        d = np.linalg.lstsq(np.conj(M).T, delta.T, rcond=None)[0].T
        e = np.sum(np.abs(d.__matmul__(Pp.T)) ** 2, axis=1) / np.maximum(np.sum(np.abs(d) ** 2, axis=1), 1e-30)
        frac.append(float(np.mean(e)))
        if mode == 'intervene':
            delta = d.__matmul__(np.conj(np.eye(4) - Pp)).__matmul__(np.conj(M))
        elif mode == 'control':
            delta = d.__matmul__(np.conj(np.eye(4) - Pr)).__matmul__(np.conj(M))
        Hr = sl.channel(geo, phases, pos + FIXED)
        dn = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-30
        Gs = ((delta / dn).__matmul__(np.conj(Hr)) + nb) * dn
        gdev = sl.device_back(p, X, h, c, nrm, s, Gs)
        for (kk, v) in gdev.items():
            p[kk] -= ETA * v
        for (kk, v) in grx.items():
            p[kk] -= ETA * v
    return (sl.evaluate(p, geo, Xte, yte, seed), frac)
out = {}
for s in C['seeds']:
    (Pp, Pr, w) = projectors(s)
    res = {'eigs': w}
    for mode in ['record', 'intervene', 'control']:
        (acc, frac) = run(s, mode, Pp, Pr)
        res[mode] = {'final_acc': acc, 'frac_first200': float(np.mean(frac[:200])), 'frac_last200': float(np.mean(frac[-200:]))}
    out[str(s)] = res
    print(s, json.dumps({m: res[m] for m in ['record', 'intervene', 'control']}), flush=True)
A = np.mean([out[s]['record']['frac_last200'] - out[s]['record']['frac_first200'] for s in out]) * 100
acc = {m: np.mean([out[s][m]['final_acc'] for s in out]) * 100 for m in ['record', 'intervene', 'control']}
lossI = acc['record'] - acc['intervene']
lossC = acc['record'] - acc['control']
verdict = 'SUPPORTED' if A >= 20 and lossI >= 10 and (lossC < lossI / 2) else 'FALSIFIED' if A <= 0 and lossI < 2 else 'PARTIAL/NOT MET'
out['summary'] = {'frac_change_pp': A, 'acc': acc, 'loss_intervene_pp': lossI, 'loss_control_pp': lossC, 'verdict': verdict}
json.dump(out, open(R / 'results/mechanism.json', 'w'), indent=2)
print(json.dumps(out['summary'], indent=1))
