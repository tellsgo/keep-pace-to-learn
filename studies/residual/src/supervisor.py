"""Local bounded experiment runner; holds/reaps one child and preserves logs."""
import os, sys, pathlib, subprocess, json, time, signal
from datetime import datetime
R = pathlib.Path(__file__).resolve().parents[1]

def now():
    return datetime.now().astimezone().isoformat()
stage = sys.argv[1]
assert stage in ['theory', 'precheck', 'formal', 'audit', 'analysis']
script = {'theory': 'theory.py', 'audit': 'audit.py', 'analysis': 'analyze.py'}.get(stage, 'experiment.py')
command = [sys.executable, '-B', str(R / 'src' / script)] + ([stage] if stage in ['precheck', 'formal'] else [])
env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(R / 'tmp'), MPLCONFIGDIR=str(R / 'tmp'), XDG_CACHE_HOME=str(R / 'tmp'))
for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    env[k] = '1'
path = R / 'logs' / f'{stage}.log'
assert not path.exists(), 'refuse overwriting prior attempt'
start = time.time()
with path.open('w') as log:
    p = subprocess.Popen(command, cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    job = {'stage': stage, 'command': command, 'pid': p.pid, 'supervisor_pid': os.getpid(), 'started_at': now(), 'timeout_s': 1800, 'host': 'local CPU', 'threads': 1}
    (R / 'logs' / f'{stage}_job.json').write_text(json.dumps(job, indent=2))
    timed = False
    try:
        code = p.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        timed = True
        os.killpg(p.pid, signal.SIGTERM)
        try:
            code = p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            code = p.wait()
    record = {**job, 'exit_code': code, 'timed_out': timed, 'finished_at': now(), 'wall_s': time.time() - start, 'child_reaped': p.poll() is not None}
    (R / 'logs' / f'{stage}_exit.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
sys.exit(code if code >= 0 else 1)
