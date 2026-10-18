import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, sl
C = sl.C
out = []
for eta in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    ks = []
    for s in C['calib_seeds']:
        (cur, _) = sl.run(s, 'phys', 0.0, eta, steps=750)
        ks.append(sl.k_target(cur, C['target_acc']))
    out.append({'eta': eta, 'K85': ks})
    print(out[-1], flush=True)
ok = [o for o in out if all((k is not None for k in o['K85']))]
cand = [o for o in ok if np.median(o['K85']) >= 100]
best = max(cand, key=lambda o: o['eta']) if cand else max(ok, key=lambda o: np.median(o['K85'])) if ok else None
json.dump({'grid': out, 'chosen_eta': best['eta'] if best else None}, open(sl.R / 'results/calibration.json', 'w'), indent=2)
print('CHOSEN', best and best['eta'])
