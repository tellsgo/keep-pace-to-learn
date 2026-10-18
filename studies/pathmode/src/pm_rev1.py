"""Path-mode precoding study (Theorem 2 of the JSAC draft). Sparse L-path channel, ULAs, lambda = 1 units.
Transmitter (BS): linear encoder on fixed features. Receiver (moving terminal): MMSE combiner + MLP, exact local gradients.
Backward: reciprocity, H_r = H(p + r) (residual r along the terminal's motion = its array axis)."""
import argparse, json, math, pathlib, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
torch.set_default_dtype(torch.float64)
GEOM = {'sep': {'c': [0.1, -0.35, 0.45], 't': [-0.55, 0.05, 0.62]}, 'close': {'c': [0.1, -0.35, 0.45], 't': [-0.1, 0.0, 0.1]}, 'along': {'c': [0.1, -0.35, 0.95], 't': [-0.55, 0.05, 0.62]}}
(NT, NR, L) = (4, 4, 3)

def steer(cos, n):
    return torch.exp(1j * math.pi * torch.arange(n)[:, None] * torch.tensor(cos)[None, :])

class Chan:

    def __init__(s, g):
        s.c = torch.tensor(GEOM[g]['c'])
        s.Ar = steer(GEOM[g]['c'], NR) / math.sqrt(NR)
        s.At = steer(GEOM[g]['t'], NT) / math.sqrt(NT)
        s.U = s.Ar
        s.V = s.At.conj()

    def H(s, alpha, p):
        return s.U.__matmul__(torch.diag(alpha * torch.exp(2j * math.pi * s.c * p))).__matmul__(s.V.conj().T) * math.sqrt(NR * NT)

class OTA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, s, Hf, Hr, nb):
        ctx.save_for_backward(Hf, Hr, nb)
        return s.__matmul__(Hf.T)

    @staticmethod
    def backward(ctx, gy):
        (Hf, Hr, nb) = ctx.saved_tensors
        n = gy.norm(dim=1, keepdim=True).clamp_min(1e-30)
        return (((gy / n).__matmul__(Hr.conj()) + nb) * n, None, None, None)

def run(a):
    torch.manual_seed(a.seed)
    g = np.random.default_rng(a.seed)
    (X, y) = load_digits(return_X_y=True)
    p = g.permutation(len(X))
    tr = p[:1400]
    X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-09)
    X = np.hstack([X, np.ones((len(X), 1))])
    (Xtr, ytr, Xte, yte) = [torch.tensor(v) for v in (X[p[:1400]], y[p[:1400]], X[p[1400:]], y[p[1400:]])]
    ch = Chan(a.geom)
    D = X.shape[1]
    if a.enc == 'path':
        F_ = ch.V.__matmul__(torch.linalg.inv(ch.V.conj().T.__matmul__(ch.V)))
        P = nn.Parameter(0.01 * torch.randn(L, D, dtype=torch.complex128))
    else:
        F_ = torch.eye(NT, dtype=torch.complex128)
        P = nn.Parameter(0.01 * torch.randn(NT, D, dtype=torch.complex128))
    A = F_.conj().T.__matmul__(ch.V) if a.enc == 'path' else None
    rx = nn.Sequential(nn.Linear(2 * NT, 64), nn.Tanh(), nn.Linear(64, 10))
    opt = torch.optim.SGD([{'params': [P]}, {'params': rx.parameters()}], lr=a.lr)
    s2 = a.sf ** 2
    out = {'args': vars(a), 'curve': [], 'cond_V': float(torch.linalg.cond(ch.V)), 'theta_cos': [float(math.cos(2 * math.pi * c * a.res)) for c in GEOM[a.geom]['c']]}

    def drop():
        return ((torch.randn(L) + 1j * torch.randn(L)) / math.sqrt(2 * L), float(g.uniform(0, 100)))

    def receive(s, H, y_):
        M = torch.linalg.solve(H.conj().T.__matmul__(H) + s2 * torch.eye(NT), H.conj().T)
        sh = y_.__matmul__(M.T)
        return rx(torch.cat([sh.real, sh.imag], 1))

    def test():
        gt = torch.Generator().manual_seed(999)
        corr = 0
        with torch.no_grad():
            for i in range(0, len(Xte), 50):
                al = (torch.randn(L, generator=gt) + 1j * torch.randn(L, generator=gt)) / math.sqrt(2 * L)
                H = ch.H(al, float(torch.rand(1, generator=gt)) * 100)
                s = Xte[i:i + 50].to(torch.complex128).__matmul__(P.T).__matmul__(F_.T)
                y_ = s.__matmul__(H.T) + a.sf / math.sqrt(2) * (torch.randn(s.shape[0], NR, generator=gt) + 1j * torch.randn(s.shape[0], NR, generator=gt))
                corr += (receive(s, H, y_).argmax(1) == yte[i:i + 50]).sum().item()
        return corr / len(Xte)
    ratios = []
    spow = []
    for k in range(a.steps):
        ix = torch.tensor(g.integers(0, len(Xtr), 32))
        (al, pos) = drop()
        Hf = ch.H(al, pos)
        Hr = ch.H(al, pos + a.res)
        s = Xtr[ix].to(torch.complex128).__matmul__(P.T).__matmul__(F_.T)
        spow.append(float((s.detach().abs() ** 2).sum(1).mean()))
        nb = a.sb / math.sqrt(2) * (torch.randn(32, NT) + 1j * torch.randn(32, NT))
        yv = OTA.apply(s, Hf, Hr, nb) + a.sf / math.sqrt(2) * (torch.randn(32, NR) + 1j * torch.randn(32, NR))
        loss = F.cross_entropy(receive(s, Hf, yv), ytr[ix])
        opt.zero_grad()
        loss.backward()
        if k % 10 == 0:
            s_ = s.detach().requires_grad_(True)
            yy = s_.__matmul__(Hf.T) + (yv.detach() - s.detach().__matmul__(Hf.T))
            (gy,) = torch.autograd.grad(F.cross_entropy(receive(s_, Hf, yy), ytr[ix]), yy)
            Xc = Xtr[ix].to(torch.complex128)
            gt_ = gy.__matmul__(Hf.conj()).__matmul__(F_.conj()).T.__matmul__(Xc)
            ga_ = gy.__matmul__(Hr.conj()).__matmul__(F_.conj()).T.__matmul__(Xc)
            ratios.append(float(torch.vdot(gt_.flatten(), ga_.flatten()).real / (gt_.abs() ** 2).sum().clamp_min(1e-30)))
        opt.step()
        if (k + 1) % a.every == 0:
            out['curve'].append({'step': k + 1, 'test_acc': test(), 'loss': float(loss.detach())})
            print(json.dumps(out['curve'][-1]), flush=True)
    out['final_acc'] = out['curve'][-1]['test_acc']
    out['align_mean'] = float(np.mean(ratios))
    out['align_min'] = float(np.min(ratios))
    out['spow_last'] = float(np.mean(spow[-200:]))
    return out
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--geom')
    ap.add_argument('--enc')
    ap.add_argument('--res', type=float)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--every', type=int, default=250)
    ap.add_argument('--sf', type=float, default=0.1)
    ap.add_argument('--sb', type=float, default=0.05)
    ap.add_argument('--out')
    a = ap.parse_args()
    o = run(a)
    pathlib.Path(a.out).write_text(json.dumps(o, indent=1))
