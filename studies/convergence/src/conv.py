"""Convergence-rate study. Inherits residual study physics (core.py, unchanged) and its learning task.
Only the encoder optimizer differs between arms: 'sgd' (fixed step eta) or 'adam' (residual study setting).
Worlds are regenerated deterministically from the seed streams and not cached."""
import sys, json, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, core as E
R = E.R
C = E.C

def world(seed):
    d = E.geometry(seed)
    n = C['train_samples']
    nt = C['test_samples']
    d['train_phases'] = E.rng(seed, 'train_phases').uniform(-np.pi, np.pi, (n, 460))
    d['test_phases'] = E.rng(seed, 'test_phases').uniform(-np.pi, np.pi, (nt, 460))
    d['x'] = E.rng(seed, 'train_inputs').choice([-1.0, 1.0], (n, 3))
    d['labels'] = (d['x'][:, 0] * d['x'][:, 1] > 0).astype(float)
    d['test_x'] = E.rng(seed, 'test_inputs').choice([-1.0, 1.0], (nt, 3))
    d['test_labels'] = (d['test_x'][:, 0] * d['test_x'][:, 1] > 0).astype(float)
    d['nf'] = C['forward_noise_std'] * E.cn(E.rng(seed, 'forward_noise'), (n, 4))
    d['nb'] = C['reverse_noise_std'] * E.cn(E.rng(seed, 'reverse_noise'), (n, 2))
    d['test_n'] = C['forward_noise_std'] * E.cn(E.rng(seed, 'test_noise'), (nt, 4))
    d['f'] = E.channel(d['train_phases'], d['steering'], d['qr'])
    d['test_f'] = E.channel(d['test_phases'], d['steering'], d['qr'])
    W = E.cn(E.rng(seed, 'initialization'), (2, 3))
    d['initial_W'] = W / np.linalg.norm(W) * C['initial_encoder_norm']
    return d

def train(d, res, opt, eta, steps):
    reverse = d['f'] if res == 0 else E.channel(d['train_phases'], d['steering'], d['qr'], res)
    W = d['initial_W'].copy()
    w = np.full(4, 0.8)
    b = -0.4
    aw = E.Adam(W)
    ar = E.Adam(np.r_[w, b])
    curve = []

    def ev():
        (loss, prob) = E.evaluate(d, W, w, b)
        return [float(loss.mean()), float(np.mean((prob >= 0.5) == d['test_labels']))]
    curve.append([0] + ev())
    cos = []
    for step in range(steps):
        ix = slice(step * 32, (step + 1) * 32)
        x = d['x'][ix]
        lab = d['labels'][ix]
        tx = np.einsum('ni,bi->bn', W, x)
        y = np.einsum('bmn,bn->bm', d['f'][ix], tx) + d['nf'][ix]
        (loss, delta, gw, gb, _) = E.task(y, lab, w, b)
        exact = np.einsum('bmn,bm->bn', d['f'][ix].conj(), delta)
        sd = delta / C['seed_bound']
        assert np.max(np.sum(abs(sd) ** 2, axis=1)) <= 1 + 1e-12
        gx = (np.einsum('bmn,bm->bn', reverse[ix].conj(), sd) + d['nb'][ix]) * C['seed_bound']
        g = np.einsum('bn,bi->ni', gx, x) / 32
        true = np.einsum('bn,bi->ni', exact, x) / 32
        cos.append(float(np.vdot(true, g).real / max(np.linalg.norm(true) * np.linalg.norm(g), 1e-30)))
        W = W - eta * g if opt == 'sgd' else aw.step(W, g, C['encoder_lr'])
        W *= min(1.0, C['encoder_bound'] / max(np.linalg.norm(W), 1e-15))
        par = np.clip(ar.step(np.r_[w, b], np.r_[gw, gb], C['receiver_lr']), -C['receiver_bound'], C['receiver_bound'])
        w = par[:4]
        b = float(par[4])
        assert np.isfinite(W).all()
        if (step + 1) % C['eval_every'] == 0:
            curve.append([step + 1] + ev())
    return (np.array(curve), float(np.mean(cos)))

def k_target(curve, acc=0.9):
    hit = [int(c[0]) for c in curve if c[2] >= acc]
    return hit[0] if hit else None
