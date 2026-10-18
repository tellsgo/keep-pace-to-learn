import itertools, subprocess, sys, pathlib, concurrent.futures as cf
R = pathlib.Path(__file__).resolve().parents[1]
jobs = []
for (g, rs) in [('sep', [0, 0.1, 0.2, 0.3, 0.4, 0.47]), ('along', [0, 0.2, 0.47])]:
    for (e, r, s) in itertools.product(['path', 'generic'], rs, range(1, 6)):
        out = R / 'results_rev1' / f'{g}_{e}_r{r}_s{s}.json'
        if not out.exists():
            jobs.append([sys.executable, str(R / 'src/pm_rev1.py'), '--geom', g, '--enc', e, '--res', str(r), '--seed', str(s), '--steps', '12000', '--out', str(out)])

def go(c):
    return subprocess.run(c, capture_output=True).returncode
with cf.ThreadPoolExecutor(8) as ex:
    print(sum((1 for rc in ex.map(go, jobs) if rc != 0)), 'failed of', len(jobs))
