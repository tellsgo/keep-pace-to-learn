import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, conv, core as E
C = E.C
th = {g['d']: g['re'] for g in json.load(open(E.R / 'results/theory.json'))['grid']}
out = {}
for d in [0.25, 0.2, 0.3]:
    E.C['encoder_lr'] = 0.025 * th[d]
    acc = []
    for s in C['seeds']:
        w = conv.world(s)
        (cur, _) = conv.train(w, d, 'adam', 0, C['steps'])
        acc.append(float(cur[-1, 2]))
    out[str(d)] = {'encoder_lr': E.C['encoder_lr'], 'final_acc': acc, 'mean': float(np.mean(acc))}
    print(d, out[str(d)], flush=True)
out['verdict_d025'] = 'SUPPORTED' if out['0.25']['mean'] >= 0.9 else 'FALSIFIED'
json.dump(out, open(E.R / 'results/adam_scaled.json', 'w'), indent=2)
print(out['verdict_d025'])
