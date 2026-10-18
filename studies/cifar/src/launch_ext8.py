"""Extension 8 (SPEC_ADDENDUM_ext8.md). 3 workers per GPU."""
import json, pathlib, subprocess, sys, time, os
R = pathlib.Path(__file__).resolve().parents[1]
(R / 'logs').mkdir(parents=True, exist_ok=True)
C = json.loads((R / 'configs/config.json').read_text())
py = sys.executable
jobs = []
(V1, V2, V3) = (1.0, 1.8202, 4.0686)
grid = [('v1_fixed', ['--res', 'fixed', '--speed', str(V1), '--cycle', '0.04']), ('v1_sel', ['--res', '-0.033', '--speed', str(V1), '--cycle', '0.04']), ('v1_mech', ['--res', '0.0', '--speed', str(V1), '--cycle', '0.14928']), ('v1.82_fixed', ['--res', 'fixed', '--speed', str(V2), '--cycle', '0.04']), ('v1.82_sel', ['--res', '-0.15', '--speed', str(V2), '--cycle', '0.04']), ('v1.82_mech', ['--res', '0.0', '--speed', str(V2), '--cycle', '0.2184']), ('v4.07_fixed', ['--res', 'fixed', '--speed', str(V3), '--cycle', '0.04']), ('v4.07_sel', ['--res', '-0.10', '--speed', str(V3), '--cycle', '0.04'])]
(R / 'results_ext8').mkdir(exist_ok=True)
for seed in C['seeds']:
    for (tag, args) in grid:
        t = f'ext8_{tag}_{seed}'
        out = R / 'results_ext8' / f'{t}.json'
        if not out.exists():
            jobs.append((t, ['--arm', 'phys', '--nr', '8'] + args + ['--seed', str(seed), '--out', str(out)]))
slots = [0, 1, 0, 1, 0, 1]
running = {}
while jobs or running:
    for sl in range(len(slots)):
        if sl not in running and jobs:
            (t, args) = jobs.pop(0)
            lf = open(R / 'logs' / f'{t}.log', 'w')
            running[sl] = (t, subprocess.Popen([py, str(R / 'src/run_ext8.py')] + args, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(slots[sl])}), lf)
            print(time.strftime('%H:%M:%S'), 'start', t, flush=True)
    for (sl, (t, p, lf)) in list(running.items()):
        if p.poll() is not None:
            lf.close()
            print(time.strftime('%H:%M:%S'), 'done', t, 'rc', p.returncode, flush=True)
            del running[sl]
    time.sleep(10)
