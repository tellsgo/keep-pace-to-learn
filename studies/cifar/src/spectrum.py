"""Training-free spectrum of Herm(Bbar_r) (MMSE combiner output, normalized by tr(Bbar_0)/Nt) for the CIFAR seeds' own geometries.
Per seed: 300 random fields/positions (RNG 4242+seed); eigenvalues computed per seed, then averaged over seeds (also reported per seed)."""
import sys, json, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch, run
C = run.C
dev = 'cuda'
out = {}
for (nr, nt) in [(8, 4), (2, 4), (2, 2)]:
    C['nr'] = nr
    C['nt'] = nt
    for seed in C['seeds']:
        ch = run.Channel(seed, dev)
        g = torch.Generator(device=dev)
        g.manual_seed(4242 + seed % 1000)
        rs = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4667, 0.6, 0.85, 1.9]
        B = {r: torch.zeros(nt, nt, dtype=torch.complex128, device=dev) for r in rs}
        den = 0.0
        for _ in range(300):
            ph = torch.rand(460, generator=g, device=dev, dtype=torch.float64) * 2 * math.pi - math.pi
            po = float(torch.rand(1, generator=g, device=dev)) * 100
            Hf = ch.H(ph, po).to(torch.complex128)
            Hh = Hf.conj().T
            M = torch.linalg.solve(Hh.__matmul__(Hf) + 0.01 * torch.eye(nt, device=dev, dtype=torch.complex128), Hh)
            MH = M.__matmul__(Hf)
            den += float(torch.trace(MH.__matmul__(MH.conj().T)).real) / nt
            for r in rs:
                B[r] += MH.__matmul__(ch.H(ph, po + r).to(torch.complex128).conj().T).__matmul__(M.conj().T)
        out[f'nr{nr}_nt{nt}_{seed}'] = {f'{r:.4f}': torch.linalg.eigvalsh((B[r] + B[r].conj().T) / 2 / den).cpu().tolist() for r in rs}
summ = {}
for (nr, nt) in [(8, 4), (2, 4), (2, 2)]:
    for r in ['0.0000', '0.1000', '0.2000', '0.2500', '0.3000', '0.4667', '0.6000', '0.8500', '1.9000']:
        ev = [out[f'nr{nr}_nt{nt}_{s}'][r] for s in C['seeds']]
        lmax = [max(e) for e in ev]
        npos = [sum((x > 0 for x in e)) for e in ev]
        summ[f'nr{nr}_nt{nt}_{r}'] = {'lmax_mean': sum(lmax) / len(lmax), 'lmax_per_seed': lmax, 'n_pos_per_seed': npos}
json.dump({'per_seed': out, 'summary': summ}, open(run.R / 'results/spectrum_cifar_seeds.json', 'w'), indent=1)
for (k, v) in summ.items():
    print(k, round(v['lmax_mean'], 3), v['n_pos_per_seed'])
