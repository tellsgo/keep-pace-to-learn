"""Extension 5 (SPEC_ADDENDUM_ext5.md). 3 workers per GPU."""
import json, pathlib, subprocess, sys, time, os
R = pathlib.Path(__file__).resolve().parents[1]
(R / 'logs').mkdir(parents=True, exist_ok=True)
C = json.loads((R / 'configs/config.json').read_text())
py = sys.executable
jobs = []
grid = [('v1_nr8_fixed_scale', ['--arm', 'phys', '--res', 'fixed', '--nr', '8', '--project', 'scale', '--proj_res', '0.46699']), ('v1_nr8_fixed_randmatch', ['--arm', 'phys', '--res', 'fixed', '--nr', '8', '--project', 'randmatch', '--proj_res', '0.46699'])]
(R / 'results_ext5').mkdir(exist_ok=True)
for seed in C['seeds']:
    for (tag, args) in grid:
        t = f'ext5_{tag}_{seed}'
        out = R / 'results_ext5' / f'{t}.json'
        if not out.exists():
            jobs.append((t, args + ['--seed', str(seed), '--out', str(out)]))
slots = [0, 1, 0, 1, 0, 1]
running = {}
while jobs or running:
    for sl in range(len(slots)):
        if sl not in running and jobs:
            (t, args) = jobs.pop(0)
            lf = open(R / 'logs' / f'{t}.log', 'w')
            running[sl] = (t, subprocess.Popen([py, str(R / 'src/run_ext5.py')] + args, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(slots[sl])}), lf)
            print(time.strftime('%H:%M:%S'), 'start', t, flush=True)
    for (sl, (t, p, lf)) in list(running.items()):
        if p.poll() is not None:
            lf.close()
            print(time.strftime('%H:%M:%S'), 'done', t, 'rc', p.returncode, flush=True)
            del running[sl]
    time.sleep(10)
