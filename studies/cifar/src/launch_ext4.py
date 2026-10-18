import json, pathlib, subprocess, sys, time, os
R = pathlib.Path(__file__).resolve().parents[1]
(R / 'logs').mkdir(parents=True, exist_ok=True)
C = json.loads((R / 'configs/config.json').read_text())
py = sys.executable
jobs = []
grid = [('e_nr8_csi', ['--arm', 'csi', '--res', '0.0'])] + [(f'e_nr8_ma{r}', ['--arm', 'phys', '--res', r]) for r in ['0.0', '0.1', '0.2', '0.25', '0.3']] + [('e_nr8_fixed', ['--arm', 'phys', '--res', 'fixed'])]
grid = [(t, a + ['--nr', '8', '--rx', 'energy']) for (t, a) in grid] + [(t, a + ['--nr', '2', '--nt', '2', '--rx', 'energy']) for (t, a) in [('e_2x2_csi', ['--arm', 'csi', '--res', '0.0']), ('e_2x2_ma0.0', ['--arm', 'phys', '--res', '0.0']), ('e_2x2_fixed', ['--arm', 'phys', '--res', 'fixed'])]]
for seed in C['seeds']:
    for (tag, args) in grid:
        t = f'ext4_{tag}_{seed}'
        out = R / 'results_ext4' / f'{t}.json'
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
