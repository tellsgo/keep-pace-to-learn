import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, sl
C = sl.C
R = sl.R
lam = C['light_speed_mps'] / C['carrier_hz']
tau = C['tau_s']
v = C['speed_mps']
V = 2.5
ETA = json.load(open(R / 'results/calibration.json'))['chosen_eta']

def T(u, a):
    if u <= 0:
        return tau
    S = u * tau + u ** 2 / a
    tr = 2 * np.sqrt(S / a) if S <= V ** 2 / a else S / V + V / a
    return tau + 2 * u / a + tr

def run(seed, res, period):
    (Xtr, ytr, Xte, yte) = sl.data(seed)
    geo = sl.geometry(seed)
    p = sl.init(seed)
    gf = sl.rng(seed, 'fields')
    gp = sl.rng(seed, 'positions')
    gn = sl.rng(seed, 'fnoise')
    gb = sl.rng(seed, 'bnoise')
    gbt = sl.rng(seed, 'batches')
    step_pos = v * period / lam
    steps = C['steps']
    order = np.concatenate([gbt.permutation(len(Xtr)) for _ in range(steps * 32 // len(Xtr) + 2)])
    curve = []
    tnext = 0.0
    for k in range(steps):
        t = k * period
        if t >= tnext:
            phases = gf.uniform(-np.pi, np.pi, 460)
            pos = gp.uniform(0, 100)
            tnext += 15.0
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
        Hr = Hf if res == 0 else sl.channel(geo, phases, pos + res)
        dn = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-30
        Gs = ((delta / dn).__matmul__(np.conj(Hr)) + nb) * dn
        gdev = sl.device_back(p, X, h, c, nrm, s, Gs)
        for (kk, vv) in gdev.items():
            p[kk] -= ETA * vv
        for (kk, vv) in grx.items():
            p[kk] -= ETA * vv
        if (k + 1) % 25 == 0:
            a_ = sl.evaluate(p, geo, Xte, yte, seed)
            curve.append((k + 1, a_))
            if a_ >= C['target_acc']:
                return k + 1
    return None
fixed = v * tau / lam
cfg = [(50, 0.0), (50, 0.15), (50, 0.2), (50, 0.3), (10, 0.0), (10, 0.15), (10, 0.2), (None, fixed)]
out = {}
for (a, r) in cfg:
    per = tau if a is None else T(max(v - r * lam / tau, 0), a)
    ks = [run(s, r, per) for s in C['seeds']]
    kk = sorted([x if x is not None else 10 ** 9 for x in ks])
    med = (kk[3] + kk[4]) / 2 if kk[4] < 10 ** 9 else None
    key = f'a{a}_r{r:.3f}'
    out[key] = {'period_s': per, 'K85': ks, 'K85_median': med, 'wall_s': med * per if med else None}
    print(key, out[key], flush=True)

def best(a):
    return min((out[f'a{a}_r{r:.3f}']['wall_s'] for r in [0.15, 0.2] if out[f'a{a}_r{r:.3f}']['wall_s']))
res = {a: (best(a), out[f'a{a}_r0.000']['wall_s']) for a in [50, 10]}
sup = all((b < 0.9 * e for (b, e) in res.values()))
fal = all((b >= e for (b, e) in res.values()))
out['verdict'] = 'SUPPORTED' if sup else 'FALSIFIED' if fal else 'NOT MET'
out['compare'] = {str(k): v for (k, v) in res.items()}
json.dump(out, open(R / 'results/wallclock.json', 'w'), indent=2)
print(out['verdict'], out['compare'])
