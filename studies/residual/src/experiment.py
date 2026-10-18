import core as E
import numpy as np
from scipy.stats import spearmanr
import sys, resource, signal, traceback, json
R = E.R
C = E.C

def precheck():
    E.frozen('training_freeze.json')
    import checks
    checks.preflight()
    rows = []
    for arm in ['CSI', 'fixed']:
        E.batch(rows, 'precheck', 30, arm, C['precheck_seeds'])
    diff = E.stats(E.values(rows, 30, 'fixed', 'ce') - E.values(rows, 30, 'CSI', 'ce'))
    E.dump(R / 'results/precheck.json', {'pass': diff['mean'] >= 0.1, 'threshold': 0.1, 'speed_kmh': 30, 'difference': diff, 'finished_at': E.now()})
    E.record_registry('precheck')

def formal():
    E.frozen('training_freeze.json')
    assert json.loads((R / 'results/precheck.json').read_text())['pass']
    theory = json.loads((R / 'results/theory.json').read_text())
    ts = {(x['speed_kmh'], x['kappa']): x for x in theory['rows']}
    rows = []
    events = []
    closed = False
    k1 = {}
    k2done = False
    for v in C['speeds_kmh']:
        for arm in ['CSI'] + ['MA_' + str(k) for k in sorted(C['main_kappas'], reverse=True)] + ['fixed']:
            purpose = 'K2/K3 and prescribed control; K1 remains closed' if closed else 'K1/K2/K3 prescribed grid'
            print('BATCH_START', E.now(), v, arm, purpose, flush=True)
            E.batch(rows, 'formal', v, arm, C['seeds'])
            if arm != 'CSI' and v not in k1 and (not closed):
                acc = float(E.values(rows, v, arm, 'accuracy').mean())
                if acc < 0.8:
                    k = E.arm_kappa(arm)
                    delta = abs(k - theory['kappa_star'][str(v)])
                    k1[v] = {'observed_kappa': k, 'theory_kappa': theory['kappa_star'][str(v)], 'absolute_difference': delta, 'accuracy': acc}
                    closed = delta > 0.2
                    events.append({'time': E.now(), 'claim': 'K1', 'speed_kmh': v, 'n_seeds': 8, **k1[v], 'triggered': closed, 'action': 'close_line' if closed else 'continue'})
                    E.dump(R / 'results/stop_events.json', events)
            mobile = [r for r in rows if r['arm'] != 'CSI' and r['kappa'] > 0]
            if len(mobile) == 192 and (not k2done):
                keys = [(vv, k) for vv in C['speeds_kmh'] for k in C['main_kappas']]
                av = [float(np.mean([r['accuracy'] for r in mobile if (r['speed_kmh'], r['kappa']) == key])) for key in keys]
                rho = float(spearmanr([ts[key]['rho_real'] for key in keys], av).statistic)
                events.append({'time': E.now(), 'claim': 'K2', 'n_cells': 24, 'n_seeds_per_cell': 8, 'spearman': rho, 'triggered': rho < 0.6, 'action': 'close_line' if rho < 0.6 else 'complete', 'threshold': 0.6})
                E.dump(R / 'results/stop_events.json', events)
                k2done = True
    E.dump(R / 'results/formal_status.json', {'finished_at': E.now(), 'rows': len(rows), 'K1_closed': closed, 'K1_online_crossings': k1, 'K2_checked': k2done, 'purpose_after_stop': 'Only remaining independent K2/K3 grid and prescribed controls; no reopening of K1'})
    E.record_registry('formal')
if __name__ == '__main__':
    stage = sys.argv[1]
    resource.setrlimit(resource.RLIMIT_CPU, (C['cpu_limit_s'], C['cpu_limit_s'] + 5))
    signal.alarm(C['wall_limit_s'])
    try:
        globals()[stage]()
    except BaseException:
        s = traceback.format_exc()
        (R / 'logs' / f'{stage}_failure.txt').write_text(s)
        print(s, flush=True)
        raise
