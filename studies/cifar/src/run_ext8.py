"""CIFAR-10 split learning over a CDL-A MIMO layer with reciprocal backward transmission (PyTorch).
BS side: ResNet-18 (CIFAR stem) -> 512 -> 2*Nt*L reals -> s in C^{L x Nt}, total power L.
Terminal: y = s H^T + n (same H for all L uses and all samples of a minibatch), MMSE combining with forward CSI,
MLP 64->256->10. BS has no CSI: its gradient arrives through H_r^H (per-sample normalized, noisy, norm fed back).
Usage: python run.py --arm phys --res 0.2 --seed 710172100 --out results/xxx.json"""
import argparse, json, math, os, pathlib, pickle, tarfile, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
R = pathlib.Path(__file__).resolve().parents[1]
C = json.loads((R / 'configs/config.json').read_text())
T = np.loadtxt(R / 'configs/cdl_a.csv', delimiter=',', skiprows=1)
OFFSETS = np.array([v * s for v in [0.0447, 0.1413, 0.2492, 0.3715, 0.5129, 0.6797, 0.8844, 1.1481, 1.5195, 2.1551] for s in [1, -1]])
ROLES = ['coupling', 'init', 'fields', 'positions', 'fnoise', 'bnoise', 'batches', 'test', 'aug']

def gen(seed, role):
    ss = np.random.SeedSequence(C['master_entropy'], spawn_key=(int(seed), ROLES.index(role)))
    return np.random.Generator(np.random.Philox(key=ss.generate_state(2, dtype=np.uint64)))

def tseed(seed, role):
    return int(gen(seed, role).integers(0, 2 ** 62))

def unit(az, zen):
    az = np.deg2rad(az)
    zen = np.deg2rad(zen)
    return np.stack([np.sin(zen) * np.cos(az), np.sin(zen) * np.sin(az), np.cos(zen)], axis=-1)

class Channel:

    def __init__(self, seed, dev):
        g = gen(seed, 'coupling')
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
        P = 10 ** (T[:, 2] / 10)
        P /= P.sum()
        w = np.repeat(np.sqrt(P / 20), 20)
        ar = np.exp(2j * np.pi * qr[:, 1, None] * 0.5 * np.arange(C['nr']))
        at = np.exp(2j * np.pi * qt[:, 1, None] * 0.5 * np.arange(C['nt']))
        self.qx = torch.tensor(qr[:, 0], device=dev, dtype=torch.float64)
        self.K = torch.tensor(w[:, None, None] * ar[:, :, None] * at[:, None, :], device=dev, dtype=torch.complex128)
        self.dev = dev

    def H(self, phases, pos):
        c = torch.exp(1j * (phases + 2 * math.pi * self.qx * pos))
        return torch.einsum('l,lrt->rt', c, self.K).to(torch.complex64)

class Block(nn.Module):

    def __init__(s, i, o, st):
        super().__init__()
        s.c1 = nn.Conv2d(i, o, 3, st, 1, bias=False)
        s.b1 = nn.BatchNorm2d(o)
        s.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False)
        s.b2 = nn.BatchNorm2d(o)
        s.sh = nn.Sequential() if st == 1 and i == o else nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o))

    def forward(s, x):
        return F.relu(s.b2(s.c2(F.relu(s.b1(s.c1(x))))) + s.sh(x))

class Device(nn.Module):

    def __init__(s):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        L = []
        i = 64
        for (o, st) in [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]:
            L.append(Block(i, o, st))
            i = o
        s.layers = nn.Sequential(*L)
        s.fc = nn.Linear(512, 2 * C['nt'] * C['uses'])

    def forward(s, x):
        z = s.fc(F.adaptive_avg_pool2d(s.layers(s.stem(x)), 1).flatten(1)).view(-1, C['uses'], C['nt'], 2)
        c = torch.complex(z[..., 0], z[..., 1])
        n = c.flatten(1).norm(dim=1).clamp_min(1e-08)[:, None, None]
        return c / n * math.sqrt(C['uses'])

class Receiver(nn.Module):

    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(2 * C['nt'] * C['uses'], 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(s, sh):
        return s.net(torch.cat([sh.real, sh.imag], -1).flatten(1))

class EnergyReceiver(nn.Module):
    """CSI-free terminal: per-antenna, per-use received energies only."""

    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(C['nr'] * C['uses'], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(s, y):
        return s.net(torch.log1p(y.abs() ** 2).flatten(1))

class OTA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, s, Hf, Hr, nb, phys, Mc=None, Pc=None):
        ctx.save_for_backward(Hf, Hr, nb)
        ctx.phys = phys
        ctx.Mc = Mc
        ctx.Pc = Pc
        return s.__matmul__(Hf.transpose(0, 1))

    @staticmethod
    def backward(ctx, gy):
        (Hf, Hr, nb) = ctx.saved_tensors
        if not ctx.phys:
            return (gy.__matmul__(Hf.conj()), None, None, None, None, None, None)
        if ctx.Pc is not None:
            Mc = ctx.Mc
            d = gy.__matmul__(torch.linalg.pinv(Mc.conj()))
            I = torch.eye(Mc.shape[0], device=gy.device, dtype=gy.dtype)
            gy = d.__matmul__((I - ctx.Pc).conj()).__matmul__(Mc.conj())
        n = gy.flatten(1).norm(dim=1).clamp_min(1e-30)[:, None, None]
        return (((gy / n).__matmul__(Hr.conj()) + nb) * n, None, None, None, None, None, None)

def mmse(H, y, s2):
    Hh = H.conj().transpose(0, 1)
    M = torch.linalg.solve(Hh.__matmul__(H) + s2 * torch.eye(H.shape[1], device=H.device, dtype=H.dtype), Hh)
    return y.__matmul__(M.transpose(0, 1))

def cifar(root):
    z = np.load(os.path.join(root, 'cifar10.npz'))
    x = np.concatenate([z['train_x'], z['test_x']]).astype(np.float32) / 255
    y = np.concatenate([z['train_y'], z['test_y']])
    m = x[:50000].mean((0, 2, 3), keepdims=True)
    sd = x[:50000].std((0, 2, 3), keepdims=True)
    x = (x - m) / sd
    return (x[:50000], y[:50000], x[50000:], y[50000:])

def augment(xb, g):
    B = xb.shape[0]
    pad = F.pad(xb, (4, 4, 4, 4), mode='reflect')
    i = torch.randint(0, 9, (B,), generator=g, device=xb.device)
    j = torch.randint(0, 9, (B,), generator=g, device=xb.device)
    idx = torch.arange(32, device=xb.device)
    rows = (i[:, None] + idx)[:, :, None]
    cols = (j[:, None] + idx)[:, None, :]
    out = pad[torch.arange(B, device=xb.device)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(B, generator=g, device=xb.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm')
    ap.add_argument('--res')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--out')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--nr', type=int, default=None)
    ap.add_argument('--nt', type=int, default=None)
    ap.add_argument('--rx', default='mmse')
    ap.add_argument('--speed', type=float, default=None)
    ap.add_argument('--project', default='none')
    ap.add_argument('--proj_res', type=float, default=None)
    ap.add_argument('--data', default=str(R.parents[1] / 'data/cifar10'))
    ap.add_argument('--cycle', type=float, default=None)
    a = ap.parse_args()
    dev = 'cuda'
    seed = a.seed
    if a.epochs:
        C['epochs'] = a.epochs
    if a.nr:
        C['nr'] = a.nr
    if a.nt:
        C['nt'] = a.nt
    if a.speed:
        C['speed_mps'] = a.speed
    if a.cycle:
        C['cycle_s'] = a.cycle
    lam = C['light_speed_mps'] / C['carrier_hz']
    res = C['speed_mps'] * C['tau_s'] / lam if a.res == 'fixed' else float(a.res)
    phys = a.arm == 'phys'
    torch.manual_seed(tseed(seed, 'init'))
    (Xtr, Ytr, Xte, Yte) = cifar(a.data)
    Xtr = torch.tensor(Xtr, device=dev)
    Ytr = torch.tensor(Ytr, device=dev)
    Xte = torch.tensor(Xte, device=dev)
    Yte = torch.tensor(Yte, device=dev)
    ch = Channel(seed, dev)
    dv = Device().to(dev).to(memory_format=torch.channels_last)
    rx = (EnergyReceiver() if a.rx == 'energy' else Receiver()).to(dev)
    Pc = None
    if a.project != 'none':
        gq = torch.Generator(device=dev)
        gq.manual_seed(777)
        B = torch.zeros(C['nt'], C['nt'], dtype=torch.complex128, device=dev)
        pr = a.proj_res
        for _ in range(300):
            ph = torch.rand(460, generator=gq, device=dev, dtype=torch.float64) * 2 * math.pi - math.pi
            po = float(torch.rand(1, generator=gq, device=dev)) * 100
            Hf_ = ch.H(ph, po).to(torch.complex128)
            Hr_ = ch.H(ph, po + pr).to(torch.complex128)
            Hh = Hf_.conj().T
            M_ = torch.linalg.solve(Hh.__matmul__(Hf_) + C['forward_noise_std'] ** 2 * torch.eye(C['nt'], device=dev, dtype=torch.complex128), Hh)
            B += M_.__matmul__(Hf_).__matmul__(Hr_.conj().T).__matmul__(M_.conj().T)
        (w, V) = torch.linalg.eigh((B + B.conj().T) / 2)
        Vp = V[:, w > 0]
        npos = Vp.shape[1]
        if a.project == 'rand':
            gr = torch.Generator(device='cpu')
            gr.manual_seed(778)
            (Q, _) = torch.linalg.qr(torch.complex(torch.randn(C['nt'], npos, generator=gr), torch.randn(C['nt'], npos, generator=gr)).to(torch.complex128))
            Vp = Q.to(dev)
        Pc = Vp.__matmul__(Vp.conj().T).to(torch.complex64)
    params = list(dv.parameters()) + list(rx.parameters())
    opt = torch.optim.SGD(params, lr=C['lr'], momentum=C['momentum'], weight_decay=C['weight_decay'])
    nb_per_epoch = 50000 // C['batch']
    total = C['epochs'] * nb_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=C['lr'], total_steps=total, pct_start=0.15)
    gb = torch.Generator(device=dev)
    gb.manual_seed(tseed(seed, 'batches'))
    ga = torch.Generator(device=dev)
    ga.manual_seed(tseed(seed, 'aug'))
    gf = torch.Generator(device=dev)
    gf.manual_seed(tseed(seed, 'fields'))
    gp = torch.Generator(device=dev)
    gp.manual_seed(tseed(seed, 'positions'))
    gn = torch.Generator(device=dev)
    gn.manual_seed(tseed(seed, 'fnoise'))
    gw = torch.Generator(device=dev)
    gw.manual_seed(tseed(seed, 'bnoise'))
    step_pos = C['speed_mps'] * C['cycle_s'] / lam
    s2 = C['forward_noise_std'] ** 2
    sf = C['forward_noise_std'] / math.sqrt(2)
    sb = C['backward_noise_std'] / math.sqrt(2)

    def cn(shape, std, g):
        return torch.complex(torch.randn(shape, generator=g, device=dev), torch.randn(shape, generator=g, device=dev)) * std

    def evaluate():
        dv.eval()
        rx.eval()
        gt = torch.Generator(device=dev)
        gt.manual_seed(tseed(seed, 'test'))
        corr = 0
        with torch.no_grad():
            for i in range(0, 10000, 500):
                H = ch.H(torch.rand(460, generator=gt, device=dev, dtype=torch.float64) * 2 * math.pi - math.pi, float(torch.rand(1, generator=gt, device=dev)) * 100)
                s = dv(Xte[i:i + 500])
                y = s.__matmul__(H.transpose(0, 1)) + cn(s.shape[:2] + (C['nr'],), sf, gt)
                corr += (rx(y if a.rx == 'energy' else mmse(H, y, s2)).argmax(1) == Yte[i:i + 500]).sum().item()
        dv.train()
        rx.train()
        return corr / 10000
    log = {'arm': a.arm, 'rx': a.rx, 'res': res, 'seed': seed, 'nr': C['nr'], 'nt': C['nt'], 'speed': C['speed_mps'], 'project': a.project, 'epochs': []}
    k = 0
    t0 = time.time()
    for ep in range(C['epochs']):
        perm = torch.randperm(50000, generator=gb, device=dev)
        cos_sum = 0.0
        n_cos = 0
        loss_sum = 0.0
        for b in range(nb_per_epoch):
            if k % C['block_updates'] == 0:
                phases = torch.rand(460, generator=gf, device=dev, dtype=torch.float64) * 2 * math.pi - math.pi
                pos = float(torch.rand(1, generator=gp, device=dev)) * 100
            else:
                pos += step_pos
            ix = perm[b * C['batch']:(b + 1) * C['batch']]
            xb = augment(Xtr[ix], ga).contiguous(memory_format=torch.channels_last)
            yb = Ytr[ix]
            Hf = ch.H(phases, pos)
            Hr = Hf if not phys or res == 0 else ch.H(phases, pos + res)
            s = dv(xb)
            nb = cn((xb.shape[0], C['uses'], C['nt']), sb, gw)
            Mc = None
            if Pc is not None:
                Hh = Hf.conj().transpose(0, 1)
                Mc = torch.linalg.solve(Hh.__matmul__(Hf) + s2 * torch.eye(C['nt'], device=dev, dtype=Hf.dtype), Hh)
            y = OTA.apply(s, Hf, Hr, nb, phys, Mc, Pc) + cn((xb.shape[0], C['uses'], C['nr']), sf, gn)
            loss = F.cross_entropy(rx(y if a.rx == 'energy' else mmse(Hf, y, s2)), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            k += 1
            loss_sum += loss.item() if b % 20 == 0 else 0
        acc = evaluate()
        log['epochs'].append({'epoch': ep + 1, 'updates': k, 'test_acc': acc, 'train_loss_sampled': loss_sum / max(1, nb_per_epoch // 20), 'wall_s': time.time() - t0})
        print(json.dumps(log['epochs'][-1]), flush=True)
        assert all((torch.isfinite(p).all() for p in params))
    log['final_acc'] = log['epochs'][-1]['test_acc']
    log['wall_s'] = time.time() - t0
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(log, indent=1))
if __name__ == '__main__':
    main()
