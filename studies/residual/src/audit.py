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

def verify():
    E.frozen('protocol_freeze.json')
    E.frozen('training_freeze.json')
    rows = []
    for stage in ['precheck', 'formal']:
        p = R / 'results' / stage / 'rows.jsonl'
        if p.exists():
            rows += [json.loads(t) for t in p.read_text().splitlines()]
    errors = {'ce': 0.0, 'accuracy': 0.0, 'channel': 0.0, 'gradient': 0.0, 'first_update': 0.0, 'replay': 0.0}
    details = []
    worlds = {}
    arrays = 0
    rows.sort(key=lambda r: (r['stage'], r['seed'], r['speed_kmh'], r['kappa'], r['arm']))
    last = None
    for row in rows:
        if row['world_path'] != last:
            with np.load(R / row['world_path']) as z:
                d = {k: z[k] for k in z.files}
            last = row['world_path']
        with np.load(R / row['raw_path']) as z:
            a = {k: z[k] for k in z.files}
        for val in list(d.values()) + list(a.values()):
            assert np.isfinite(val).all()
            arrays += 1
        key = (row['stage'], row['seed'])
        if key not in worlds:
            gg = E.geometry(row['seed'])
            for k in ['steering', 'qr', 'qt', 'angles', 'permutations']:
                errors['replay'] = max(errors['replay'], float(np.max(abs(gg[k] - d[k]))))
            for (name, role) in [('train_phases', 'train_phases'), ('test_phases', 'test_phases')]:
                replay = E.rng(row['seed'], role).uniform(-np.pi, np.pi, d[name].shape)
                errors['replay'] = max(errors['replay'], float(np.max(abs(replay - d[name]))))
            worlds[key] = True
        W = a['W']
        w = a['w']
        b = float(a['b'])
        x = d['test_x'].dot(W.T)
        x /= np.maximum(1, np.linalg.norm(x, axis=1))[:, None]
        y = np.sum(d['test_f'] * x[:, None, :], axis=2) + d['test_n']
        features = np.tanh(y.real * y.real + y.imag * y.imag)
        z = features.dot(w) + b
        loss = np.logaddexp(0, z) - d['test_labels'] * z
        prob = 1 / (1 + np.exp(-z))
        acc = float(np.mean((z >= 0) == d['test_labels']))
        errors['ce'] = max(errors['ce'], abs(float(loss.mean()) - row['ce']), float(np.max(abs(loss - a['loss']))))
        errors['accuracy'] = max(errors['accuracy'], abs(acc - row['accuracy']))
        samples = [0, 31, 3839]
        f = direct(d['train_phases'][samples], d, 0)
        rev = direct(d['train_phases'][samples], d, row['residual_lambda'])
        errors['channel'] = max(errors['channel'], float(np.max(abs(f - d['f'][samples]))), float(np.max(abs(rev - a['reverse'][samples]))))
        if row['kappa'] == 1:
            assert np.array_equal(a['reverse'], d['f'])
        tx = d['x'][:32].dot(d['initial_W'].T)
        y = np.sum(d['f'][:32] * tx[:, None, :], axis=2) + d['nf'][:32]
        u = np.tanh(abs(y) ** 2)
        pr = 1 / (1 + np.exp(-(u.dot(d['initial_w']) + float(d['initial_b']))))
        err = pr - d['labels'][:32]
        delta = 2 * err[:, None] * d['initial_w'] * (1 - u * u) * y
        if row['arm'] == 'CSI':
            gx = np.sum(d['f'][:32].conj() * delta[:, :, None], axis=1)
        else:
            gx = np.sum(a['reverse'][:32].conj() * (delta / C['seed_bound'])[:, :, None], axis=1) * C['seed_bound'] + d['nb'][:32] * C['seed_bound']
        grad = np.sum(gx[:, :, None] * d['x'][:32, None, :], axis=0) / 32
        errors['gradient'] = max(errors['gradient'], float(np.max(abs(grad - a['estimated_gradients'][0]))))
        update = d['initial_W'] - C['encoder_lr'] * grad / (abs(grad) + C['adam_eps'])
        update *= min(1.0, C['encoder_bound'] / np.linalg.norm(update))
        errors['first_update'] = max(errors['first_update'], float(np.max(abs(update - a['W_history'][1]))))
        assert row['steps'] == 120 and len(a['trace']) == 120 and (row['training_examples'] == 3840) and (len(loss) == 8192)
        assert np.max(np.linalg.norm(a['W_history'], axis=(1, 2))) <= C['encoder_bound'] + 1e-12
        details.append({'stage': row['stage'], 'seed': row['seed'], 'speed': row['speed_kmh'], 'arm': row['arm'], 'ce_recomputed': float(loss.mean()), 'accuracy_recomputed': acc})
    assert max(errors.values()) < 1e-10, errors
    source = json.loads((R / 'SOURCE_MAP.json').read_text())
    source_ok = all((E.sha(x['source']) == x['sha256'] and E.sha(R / x['copy']) == x['sha256'] for x in source))
    assert source_ok
    pf = json.loads((R / 'protocol_freeze.json').read_text())
    ff = json.loads((R / 'training_freeze.json').read_text())
    pre = json.loads((R / 'results/precheck.json').read_text())
    theory = json.loads((R / 'results/theory.json').read_text())
    assert pf['time'] < theory['created_at'] < ff['time'] < pre['finished_at']
    for stage in ['precheck', 'formal']:
        job = R / 'logs' / f'{stage}_job.json'
        if job.exists():
            assert json.loads(job.read_text())['started_at'] > ff['time']
    formal = [r for r in rows if r['stage'] == 'formal']
    if pre['pass']:
        assert len(formal) == 240
        for v in C['speeds_kmh']:
            for arm in ['CSI', 'fixed'] + ['MA_' + str(k) for k in C['main_kappas']]:
                cell = [r for r in formal if r['speed_kmh'] == v and r['arm'] == arm]
                assert sorted((r['seed'] for r in cell)) == C['seeds']
        assert min((r['completed_at'] for r in formal)) > pre['finished_at']
    else:
        assert not formal
    assert len({(r['stage'], r['seed'], r['speed_kmh'], r['arm']) for r in rows}) == len(rows)
    streams = []
    for stage in ['precheck', 'formal']:
        p = R / 'results' / stage / 'rng_registry.json'
        if p.exists():
            streams += json.loads(p.read_text())['streams']
    assert len(set((tuple(x['key']) for x in streams))) == len(streams)
    theory_error = 0.0
    for x in theory['rows']:
        vals = []
        for seed in C['seeds']:
            geo = E.geometry(seed)
            d = (1 - x['kappa']) * (x['speed_kmh'] / 3.6) * C['delay_s'] / E.LAM
            val = sum((float(E.WEIGHTS[i] ** 2) * complex(np.cos(2 * np.pi * geo['qr'][i, 0] * d), np.sin(2 * np.pi * geo['qr'][i, 0] * d)) for i in range(460)))
            vals.append(val)
        theory_error = max(theory_error, abs(np.mean(vals).real - x['rho_real']), abs(np.mean(vals).imag - x['rho_imag']))
    assert theory_error < 1e-12
    E.dump(R / 'results/verification_rows.json', details)
    E.dump(R / 'results/verification.json', {'passed': True, 'rows': len(rows), 'worlds': len(worlds), 'finite_arrays_checked': arrays, 'max_errors': errors, 'theory_scalar_max_error': float(theory_error), 'read_only_reused_sources_unchanged': source_ok, 'freeze_hashes_and_chronology_passed': True, 'independent_rng_keys': len(streams), 'all_cells_have_exact_8_roots': bool(formal), 'finished_at': E.now()})
if __name__ == '__main__':
    verify()
