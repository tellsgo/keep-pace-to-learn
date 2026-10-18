"""Run all arms x seeds on 2 GPUs, 2 workers per GPU; skip finished outputs; log each run."""
import json, pathlib, subprocess, sys, time, itertools
R = pathlib.Path(__file__).resolve().parents[1]
(R / 'logs').mkdir(parents=True, exist_ok=True)
C = json.loads((R / 'configs/config.json').read_text())
py = sys.executable
jobs = []
for seed in C['seeds']:
    for (arm, res) in C['arms']:
        tag = f'{arm}_{res}_{seed}'
        out = R / 'results' / f'{tag}.json'
        if not out.exists():
            jobs.append((tag, ['--arm', arm, '--res', str(res), '--seed', str(seed), '--out', str(out)]))
slots = [0, 1, 0, 1]
running = {}
while jobs or running:
    for sl in range(len(slots)):
        if sl not in running and jobs:
            (tag, args) = jobs.pop(0)
            lf = open(R / 'logs' / f'{tag}.log', 'w')
            running[sl] = (tag, subprocess.Popen([py, str(R / 'src/run.py')] + args, stdout=lf, stderr=subprocess.STDOUT, env={**__import__('os').environ, 'CUDA_VISIBLE_DEVICES': str(slots[sl])}), lf)
            print(time.strftime('%H:%M:%S'), 'start', tag, 'gpu', slots[sl], flush=True)
    for (sl, (tag, p, lf)) in list(running.items()):
        if p.poll() is not None:
            lf.close()
            print(time.strftime('%H:%M:%S'), 'done', tag, 'rc', p.returncode, flush=True)
            del running[sl]
    time.sleep(10)
