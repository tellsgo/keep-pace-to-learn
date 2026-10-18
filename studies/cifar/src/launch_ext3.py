"""Extension 3 (SPEC_ADDENDUM_ext3.md)."""
import json, pathlib, subprocess, sys, time, os
R = pathlib.Path(__file__).resolve().parents[1]
(R / 'logs').mkdir(parents=True, exist_ok=True)
C = json.loads((R / 'configs/config.json').read_text())
py = sys.executable
jobs = []
(V1, V2) = (1.8202, 4.0686)
grid = [('v1.82_nr8_csi', ['--arm', 'csi', '--res', '0.0', '--speed', str(V1), '--nr', '8']), ('v4.07_nr8_csi', ['--arm', 'csi', '--res', '0.0', '--speed', str(V2), '--nr', '8']), ('v1.82_nr2_csi', ['--arm', 'csi', '--res', '0.0', '--speed', str(V1), '--nr', '2']), ('nt2nr2_v1_csi', ['--arm', 'csi', '--res', '0.0', '--nr', '2', '--nt', '2']), ('nt2nr2_v1_ma', ['--arm', 'phys', '--res', '0.0', '--nr', '2', '--nt', '2']), ('nt2nr2_v1_fixed', ['--arm', 'phys', '--res', 'fixed', '--nr', '2', '--nt', '2']), ('nt2nr2_v1.82_ma', ['--arm', 'phys', '--res', '0.0', '--speed', str(V1), '--nr', '2', '--nt', '2']), ('nt2nr2_v1.82_fixed', ['--arm', 'phys', '--res', 'fixed', '--speed', str(V1), '--nr', '2', '--nt', '2'])]
for seed in C['seeds']:
    for (tag, args) in grid:
        t = f'ext3_{tag}_{seed}'
        out = R / 'results_ext3' / f'{t}.json'
        if not out.exists():
            jobs.append((t, args + ['--seed', str(seed), '--out', str(out)]))
slots = [0, 1, 0, 1, 0, 1]
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
