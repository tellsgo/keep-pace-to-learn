"""ray model: scalar-polarized CDL-A ray sum, frozen paired-drop learning experiment."""
import os
for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
import sys, json, pathlib, hashlib, time, traceback, resource, signal
sys.dont_write_bytecode = True
import numpy as np
R = pathlib.Path(__file__).resolve().parents[1]
C = json.loads((R / 'configs/config.json').read_text())
T = np.loadtxt(R / 'configs/cdl_a.csv', delimiter=',', skiprows=1)
np.seterr(all='raise')
ROLES = ['coupling', 'train_phases', 'test_phases', 'train_inputs', 'test_inputs', 'forward_noise', 'reverse_noise', 'test_noise', 'initialization', 'diagnostics', 'bootstrap']
REGISTRY = {}
OFFSETS = np.array([v * s for v in [0.0447, 0.1413, 0.2492, 0.3715, 0.5129, 0.6797, 0.8844, 1.1481, 1.5195, 2.1551] for s in [1, -1]])
LAM = C['light_speed_mps'] / C['carrier_hz']
P = 10 ** (T[:, 2] / 10)
P /= P.sum()
WEIGHTS = np.repeat(np.sqrt(P / 20), 20)

def now():
    from datetime import datetime
    return datetime.now().astimezone().isoformat()

def dump(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def frozen(name):
    doc = json.loads((R / name).read_text())
    bad = [p for (p, h) in doc['hashes'].items() if sha(R / p) != h]
    assert not bad, ('freeze mismatch', bad)
    return True

def rng(seed, role):
    ss = np.random.SeedSequence(C['master_entropy'], spawn_key=(int(seed), ROLES.index(role)))
    key = ss.generate_state(2, dtype=np.uint64)
    REGISTRY[str(seed) + ':' + role] = {'seed': int(seed), 'role': role, 'key': [int(x) for x in key], 'prefix_hash': hashlib.sha256(np.random.Philox(key=key).random_raw(16).tobytes()).hexdigest()}
    return np.random.Generator(np.random.Philox(key=key))

def cn(g, shape):
    return (g.normal(size=shape) + 1j * g.normal(size=shape)) / np.sqrt(2)

def unit(az, zen):
    az = np.deg2rad(az)
    zen = np.deg2rad(zen)
    return np.stack([np.sin(zen) * np.cos(az), np.sin(zen) * np.sin(az), np.cos(zen)], axis=-1)

def geometry(seed):
    g = rng(seed, 'coupling')
    angles = np.empty((23, 20, 4))
    perms = np.empty((23, 3, 20), int)
    for n in range(23):
        perms[n] = np.array([g.permutation(20) for _ in range(3)])
        angles[n, :, 0] = T[n, 4] + 11 * OFFSETS
        angles[n, :, 1] = T[n, 6] + 3 * OFFSETS[perms[n, 0]]
        angles[n, :, 2] = T[n, 3] + 5 * OFFSETS[perms[n, 1]]
        angles[n, :, 3] = T[n, 5] + 3 * OFFSETS[perms[n, 2]]
    for j in [1, 3]:
        angles[:, :, j] %= 360
        angles[:, :, j] = np.where(angles[:, :, j] > 180, 360 - angles[:, :, j], angles[:, :, j])
    a = angles.reshape(460, 4)
    qr = unit(a[:, 0], a[:, 1])
    qt = unit(a[:, 2], a[:, 3])
    arg = qr[:, 1, None, None] * (0.5 * np.arange(4))[None, :, None] + qt[:, 1, None, None] * (0.5 * np.arange(2))[None, None, :]
    delay = np.repeat(T[:, 1] * C['delay_spread_s'], 20)
    steering = np.exp(2j * np.pi * (arg - C['baseband_frequency_hz'] * delay[:, None, None])) * WEIGHTS[:, None, None]
    return dict(angles=angles, permutations=perms, qr=qr, qt=qt, steering=steering, delays=delay)

def channel(phases, steering, qr, residual_lambda=0.0):
    motion = np.exp(2j * np.pi * qr[:, 0] * residual_lambda)
    return np.einsum('bl,lus->bus', np.exp(1j * phases), steering * motion[:, None, None], optimize=False)

def distance(speed, kappa):
    return (1 - kappa) * (speed / 3.6) * C['delay_s'] / LAM

def world(seed, stage):
    path = R / 'results' / stage / 'worlds' / f'{seed}.npz'
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    d = geometry(seed)
    n = C['train_samples']
    nt = C['test_samples']
    d['train_phases'] = rng(seed, 'train_phases').uniform(-np.pi, np.pi, (n, 460))
    d['test_phases'] = rng(seed, 'test_phases').uniform(-np.pi, np.pi, (nt, 460))
    d['x'] = rng(seed, 'train_inputs').choice([-1.0, 1.0], (n, 3))
    d['labels'] = (d['x'][:, 0] * d['x'][:, 1] > 0).astype(float)
    d['test_x'] = rng(seed, 'test_inputs').choice([-1.0, 1.0], (nt, 3))
    d['test_labels'] = (d['test_x'][:, 0] * d['test_x'][:, 1] > 0).astype(float)
    d['nf'] = C['forward_noise_std'] * cn(rng(seed, 'forward_noise'), (n, 4))
    d['nb'] = C['reverse_noise_std'] * cn(rng(seed, 'reverse_noise'), (n, 2))
    d['test_n'] = C['forward_noise_std'] * cn(rng(seed, 'test_noise'), (nt, 4))
    d['f'] = channel(d['train_phases'], d['steering'], d['qr'])
    d['test_f'] = channel(d['test_phases'], d['steering'], d['qr'])
    W = cn(rng(seed, 'initialization'), (2, 3))
    d['initial_W'] = W / np.linalg.norm(W) * C['initial_encoder_norm']
    d['initial_w'] = np.full(4, 0.8)
    d['initial_b'] = np.array(-0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)
    return d

def task(y, lab, w, b):
    u = np.tanh(abs(y) ** 2)
    z = b + np.einsum('bi,i->b', u, w)
    prob = 1 / (1 + np.exp(-z))
    e = prob - lab
    return (np.logaddexp(0, z) - lab * z, 2 * e[:, None] * w[None, :] * (1 - u * u) * y, np.einsum('b,bi->i', e, u) / len(y) + C['receiver_l2'] * w, e.mean(), prob)

def evaluate(d, W, w, b):
    x = np.einsum('ni,bi->bn', W, d['test_x'])
    norm = np.linalg.norm(x, axis=1)
    x = x / np.maximum(1, norm)[:, None]
    y = np.einsum('bmn,bn->bm', d['test_f'], x) + d['test_n']
    (loss, _, _, _, p) = task(y, d['test_labels'], w, b)
    return (loss, p)

class Adam:

    def __init__(self, x):
        self.m = np.zeros_like(x)
        self.v = np.zeros(x.shape)
        self.t = 0

    def step(self, x, g, lr):
        self.t += 1
        self.m = C['beta1'] * self.m + (1 - C['beta1']) * g
        self.v = C['beta2'] * self.v + (1 - C['beta2']) * abs(g) ** 2
        return x - lr * (self.m / (1 - C['beta1'] ** self.t)) / (np.sqrt(self.v / (1 - C['beta2'] ** self.t)) + C['adam_eps'])

def arm_kappa(arm):
    return 1.0 if arm == 'CSI' else 0.0 if arm == 'fixed' else float(arm.split('_')[1])

def train(seed, stage, speed, arm):
    start = time.time()
    d = world(seed, stage)
    k = arm_kappa(arm)
    res = distance(speed, k)
    reverse = d['f'].copy() if k == 1 else channel(d['train_phases'], d['steering'], d['qr'], res)
    W = d['initial_W'].copy()
    w = d['initial_w'].copy()
    b = float(d['initial_b'])
    aw = Adam(W)
    ar = Adam(np.r_[w, b])
    tr = []
    wg = []
    tg = []
    wp = [W.copy()]
    rp = [np.r_[w, b]]
    physical = arm != 'CSI'
    max_energy = 0.0
    for step in range(C['steps']):
        ix = slice(step * 32, (step + 1) * 32)
        x = d['x'][ix]
        lab = d['labels'][ix]
        tx = np.einsum('ni,bi->bn', W, x)
        assert np.max(np.sum(abs(tx) ** 2, axis=1)) <= 1 + 1e-12
        y = np.einsum('bmn,bn->bm', d['f'][ix], tx) + d['nf'][ix]
        (loss, delta, gw, gb, _) = task(y, lab, w, b)
        exact = np.einsum('bmn,bm->bn', d['f'][ix].conj(), delta)
        if physical:
            sd = delta / C['seed_bound']
            energy = float(np.max(np.sum(abs(sd) ** 2, axis=1)))
            max_energy = max(max_energy, energy)
            assert energy <= 1 + 1e-12
            gx = (np.einsum('bmn,bm->bn', reverse[ix].conj(), sd) + d['nb'][ix]) * C['seed_bound']
        else:
            gx = exact
        g = np.einsum('bn,bi->ni', gx, x) / 32
        true = np.einsum('bn,bi->ni', exact, x) / 32
        wg.append(g)
        tg.append(true)
        cosine = float(np.vdot(true, g).real / max(np.linalg.norm(true) * np.linalg.norm(g), 1e-30))
        W = aw.step(W, g, C['encoder_lr'])
        W *= min(1.0, C['encoder_bound'] / max(np.linalg.norm(W), 1e-15))
        par = np.clip(ar.step(np.r_[w, b], np.r_[gw, gb], C['receiver_lr']), -C['receiver_bound'], C['receiver_bound'])
        w = par[:4]
        b = float(par[4])
        wp.append(W.copy())
        rp.append(par.copy())
        tr.append([step, float(loss.mean()), cosine, float(np.linalg.norm(W))])
        assert np.isfinite(W).all() and np.isfinite(par).all()
    (loss, prob) = evaluate(d, W, w, b)
    path = R / 'results' / stage / 'arms' / f'{seed}_v{speed}_{arm}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, reverse=reverse, W=W, w=w, b=b, loss=loss, prob=prob, trace=np.array(tr), estimated_gradients=np.array(wg), true_gradients=np.array(tg), W_history=np.array(wp), receiver_history=np.array(rp))
    rho = np.sum(WEIGHTS ** 2 * np.exp(2j * np.pi * d['qr'][:, 0] * res))
    empirical = np.mean(reverse * d['f'].conj()) / np.mean(abs(d['f']) ** 2)
    row = {'stage': stage, 'seed': seed, 'speed_kmh': speed, 'arm': arm, 'kappa': k, 'residual_lambda': res, 'ce': float(loss.mean()), 'accuracy': float(np.mean((prob >= 0.5) == d['test_labels'])), 'steps': 120, 'training_examples': 3840, 'test_examples': 8192, 'reverse_noise_variance_restored': 0.2304 if physical else 0.0, 'max_reverse_energy': max_energy, 'rho_real': float(rho.real), 'rho_imag': float(rho.imag), 'rho_abs': float(abs(rho)), 'empirical_rho_real': float(empirical.real), 'empirical_rho_imag': float(empirical.imag), 'gradient_cos_mean': float(np.mean(np.array(tr)[:, 2])), 'world_path': str((R / 'results' / stage / 'worlds' / f'{seed}.npz').relative_to(R)), 'raw_path': str(path.relative_to(R)), 'seconds': time.time() - start, 'completed_at': now()}
    with (R / 'results' / stage / 'rows.jsonl').open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row

def stats(vals):
    v = np.asarray(vals)
    assert v.shape == (8,) and np.isfinite(v).all()
    ix = rng(C['bootstrap_seed'], 'bootstrap').integers(0, 8, (C['bootstrap_replicates'], 8))
    return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)), 'min': float(v.min()), 'max': float(v.max()), 'ci95': np.quantile(v[ix].mean(1), [0.025, 0.975]).tolist(), 'per_seed': v.tolist()}

def values(rows, speed, arm, key):
    v = sorted([r for r in rows if r['speed_kmh'] == speed and r['arm'] == arm], key=lambda r: r['seed'])
    assert len(v) == 8
    return np.array([r[key] for r in v])

def batch(rows, stage, speed, arm, seeds):
    for seed in seeds:
        rows.append(train(seed, stage, speed, arm))

def record_registry(stage):
    vals = list(REGISTRY.values())
    assert len(set((tuple(v['key']) for v in vals))) == len(vals)
    dump(R / 'results' / stage / 'rng_registry.json', {'passed': True, 'streams': vals})
