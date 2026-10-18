"""Extension 1 (SPEC_ADDENDUM_ext1.md): Nr=2 grid and Nr=8 weak-eigenvalue residuals; 3 workers per GPU."""
import json, pathlib, subprocess, sys, time, os
R = pathlib.Path(__file__).resolve().parents[1]
(R / 'logs').mkdir(parents=True, exist_ok=True)
C = json.loads((R / 'configs/config.json').read_text())
py = sys.executable
jobs = []
grid = [(2, 'csi', '0.0')] + [(2, 'phys', r) for r in ['0.0', '0.25', '0.47', '0.6', '0.85']] + [(8, 'phys', r) for r in ['0.85', '1.9']]
for seed in C['seeds']:
    for (nr, arm, res) in grid:
        tag = f'ext1_nr{nr}_{arm}_{res}_{seed}'
        out = R / 'results_ext1' / f'{tag}.json'
        if not out.exists():
            jobs.append((tag, ['--arm', arm, '--res', res, '--seed', str(seed), '--nr', str(nr), '--out', str(out)]))
slots = [0, 1, 0, 1, 0, 1]
running = {}
while jobs or running:
    for sl in range(len(slots)):
        if sl not in running and jobs:
            (tag, args) = jobs.pop(0)
            lf = open(R / 'logs' / f'{tag}.log', 'w')
            running[sl] = (tag, subprocess.Popen([py, str(R / 'src/run.py')] + args, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(slots[sl])}), lf)
            print(time.strftime('%H:%M:%S'), 'start', tag, flush=True)
    for (sl, (tag, p, lf)) in list(running.items()):
        if p.poll() is not None:
            lf.close()
            print(time.strftime('%H:%M:%S'), 'done', tag, 'rc', p.returncode, flush=True)
            del running[sl]
    time.sleep(10)
