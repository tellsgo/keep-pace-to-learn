"""Exercise a short CPU training configuration without altering original results."""
from pathlib import Path
import json
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'checks/release_outputs'
OUT.mkdir(parents=True, exist_ok=True)
command = [sys.executable, '-B', 'studies/pathmode/src/pm_rev1.py', '--geom', 'sep', '--enc', 'path', '--res', '0.2', '--seed', '1', '--steps', '20', '--every', '10', '--out', 'checks/release_outputs/pathmode_smoke.json']
start = time.monotonic()
subprocess.run(command, cwd=ROOT, check=True)
result = json.loads((OUT / 'pathmode_smoke.json').read_text())
assert len(result['curve']) == 2
assert all(0 <= row['test_acc'] <= 1 for row in result['curve'])
assert result['align_min'] >= min(result['theta_cos']) - .01
print(json.dumps({'passed': True, 'seconds': time.monotonic()-start, 'steps': 20, 'minimum_alignment': result['align_min']}))
