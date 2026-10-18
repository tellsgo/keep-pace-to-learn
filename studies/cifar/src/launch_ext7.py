"""Extension 7 (SPEC_ADDENDUM_ext7.md). One worker per configuration."""
import json, pathlib, subprocess, sys, time, os
R = pathlib.Path(__file__).resolve().parents[1]
(R / 'logs').mkdir(parents=True, exist_ok=True)
C = json.loads((R / 'configs/config.json').read_text())
py = sys.executable
jobs = []
grid = [('v1_nr8_sel', ['--arm', 'phys', '--res', '-0.033', '--nr', '8']), ('v1_nr2_sel', ['--arm', 'phys', '--res', '-0.033', '--nr', '2'])]
(R / 'results_ext7').mkdir(exist_ok=True)
for seed in C['seeds']:
    for (tag, args) in grid:
        t = f'ext7_{tag}_{seed}'
        out = R / 'results_ext7' / f'{t}.json'
        if not out.exists():
            jobs.append((t, args + ['--seed', str(seed), '--out', str(out)]))
slots = [0, 1, 0, 1, 0, 1, 0, 1, 0]
running = {}
while jobs or running:
    for sl in range(len(slots)):
        if sl not in running and jobs:
            (t, args) = jobs.pop(0)
            lf = open(R / 'logs' / f'{t}.log', 'w')
            running[sl] = (t, subprocess.Popen([py, str(R / 'src/run.py')] + args, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(slots[sl])}), lf)
            print(time.strftime('%H:%M:%S'), 'start', t, flush=True)
    for (sl, (t, p, lf)) in list(running.items()):
        if p.poll() is not None:
            lf.close()
            print(time.strftime('%H:%M:%S'), 'done', t, 'rc', p.returncode, flush=True)
            del running[sl]
    time.sleep(10)
