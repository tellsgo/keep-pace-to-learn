import pathlib, json, hashlib, sys, datetime
R = pathlib.Path(__file__).resolve().parents[1]
which = sys.argv[1]
assert which in ['protocol', 'training']
paths = [R / 'SPEC.md', *sorted((R / 'configs').glob('*')), *sorted((R / 'src').glob('*.py'))]
if which == 'training':
    paths += [R / 'SPEC.md', R / 'results/theory.json', R / 'results/theory_grid.csv', R / 'results/theory_by_seed.csv', R / 'results/theory_geometry.npz', R / 'protocol_freeze.json']
out = R / f'{which}_freeze.json'
assert not out.exists()
out.write_text(json.dumps({'time': datetime.datetime.now().astimezone().isoformat(), 'hashes': {str(p.relative_to(R)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}, indent=2) + '\n')
print(out)
