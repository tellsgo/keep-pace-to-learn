"""Training-free check of the beam-resolved certificate: for a fixed DFT combiner M (Nr x Nr), each beam direction w=e_b ages with its
own effective spectrum mu_w (path weights p_l |w^H M a_l|^2). Compare exact Herm(Bbar_r) (analytic per-path form, Prop. array) with
(i) per-beam real correlation Re phi_w(r), (ii) the per-beam certificates lambda/(4 c_w) and lambda/(sqrt2 pi sigma_w)."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, sl
C = sl.C
nr = 8
out = {}
F = np.exp(-2j * np.pi * np.outer(np.arange(nr), np.arange(nr)) / nr) / np.sqrt(nr)
for s in C['seeds']:
    geo = sl.geometry(s)
    p = sl.W_RAY ** 2
    qx = geo['qx']
    m = geo['ar'].__matmul__(F.T)
    rows = {}
    for r in [0.0, 0.25, 0.4667, 0.85, 1.9]:
        Hm = (p * np.cos(2 * np.pi * qx * r))[:, None, None] * (m[:, :, None] * m[:, None, :].conj())
        Hm = Hm.sum(0)
        H0 = (p[:, None, None] * (m[:, :, None] * m[:, None, :].conj())).sum(0)
        ev = np.linalg.eigvalsh(Hm / np.trace(H0).real * nr)
        wb = p[:, None] * np.abs(m) ** 2
        Pb = wb.sum(0)
        rephi = (wb * np.cos(2 * np.pi * qx * r)[:, None]).sum(0) / Pb
        rows[f'{r:.4f}'] = {'lmax': float(ev[-1]), 'n_pos': int((ev > 0).sum()), 'beam_rephi': rephi.tolist(), 'beam_power': (Pb / Pb.sum()).tolist(), 'n_beams_pos': int((rephi > 0).sum()), 'best_beam_rephi': float(rephi.max())}
    wb = p[:, None] * np.abs(m) ** 2
    wn = wb / wb.sum(0)
    sig = np.sqrt((wn * qx[:, None] ** 2).sum(0))
    cmax = [float(np.max(np.abs(qx[wn[:, b] > 0.001 * wn[:, b].max()]))) for b in range(nr)]
    rows['beam_sigma_m'] = sig.tolist()
    rows['beam_rms_radius_lambda'] = (1 / (np.sqrt(2) * np.pi * sig)).tolist()
    rows['beam_cmax'] = cmax
    out[str(s)] = rows
json.dump(out, open(sl.R / 'results/beam_certificate.json', 'w'), indent=1)
for r in ['0.0000', '0.2500', '0.4667', '0.8500', '1.9000']:
    print(r, 'lmax', np.round(np.mean([out[s][r]['lmax'] for s in out]), 3), 'n+', [out[s][r]['n_pos'] for s in out], 'beams with Re phi_w>0', [out[s][r]['n_beams_pos'] for s in out], 'best beam Re phi', np.round(np.mean([out[s][r]['best_beam_rephi'] for s in out]), 3))
s0 = list(out)[0]
print('per-beam sigma_m', np.round(out[s0]['beam_sigma_m'], 3))
print('per-beam RMS radius /lambda', np.round(out[s0]['beam_rms_radius_lambda'], 3))
