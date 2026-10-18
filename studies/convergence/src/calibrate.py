"""Step-size calibration for the SGD arm using ONLY residual 0 (no channel aging) on the 4 calibration seeds.
Rule (amendment 1, frozen before running): eta grid {.0025,.005,.01,.02,.05}; keep those where all 4 seeds reach 90%
within 300 updates; choose the LARGEST eta whose median K90 >= 40 updates (resolution room); if none, the smallest eta. No moving-residual arm is run here."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, conv, core as E
grid = [0.0025, 0.005, 0.01, 0.02, 0.05]
out = []
ws = {s: conv.world(s) for s in E.C['precheck_seeds']}
for eta in grid:
    ks = []
    fin = []
    for s in E.C['precheck_seeds']:
        (cur, _) = conv.train(ws[s], 0.0, 'sgd', eta, 300)
        ks.append(conv.k_target(cur))
        fin.append(cur[-1, 2])
    out.append({'eta': eta, 'K90': ks, 'final_acc': fin})
    print(out[-1], flush=True)
ok = [o for o in out if all((k is not None for k in o['K90']))]
cand = [o for o in ok if np.median(o['K90']) >= 40]
best = max(cand, key=lambda o: o['eta']) if cand else min(ok, key=lambda o: o['eta']) if ok else None
json.dump({'grid': out, 'chosen_eta': best['eta'] if best else None, 'rule': __doc__}, open(E.R / 'results/calibration.json', 'w'), indent=2)
print('CHOSEN', best['eta'] if best else None)
