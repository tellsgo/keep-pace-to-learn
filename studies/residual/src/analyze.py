import core as E
import json, csv
import numpy as np
from scipy.stats import spearmanr, rankdata
R = E.R
C = E.C

def writecsv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

def main():
    E.frozen('training_freeze.json')
    pre = json.loads((R / 'results/precheck.json').read_text())
    theory = json.loads((R / 'results/theory.json').read_text())
    if not pre['pass']:
        E.dump(R / 'results/claims.json', {'K1': {'verdict': 'Precheck failed; not tested'}, 'K2': {'verdict': 'Precheck failed; not tested'}, 'K3': {'verdict': 'Precheck failed; no accuracy comparison'}})
        return
    rows = [json.loads(x) for x in (R / 'results/formal/rows.jsonl').read_text().splitlines()]
    assert len(rows) == 240
    writecsv(R / 'results/by_seed.csv', rows)
    tt = {(r['speed_kmh'], r['kappa']): r for r in theory['rows']}
    cells = []
    lookup = {}
    vec = {}
    for v in C['speeds_kmh']:
        for arm in ['CSI'] + ['MA_' + str(k) for k in C['main_kappas']] + ['fixed']:
            rr = sorted([x for x in rows if x['speed_kmh'] == v and x['arm'] == arm], key=lambda x: x['seed'])
            assert len(rr) == 8
            av = np.array([x['accuracy'] for x in rr])
            ce = np.array([x['ce'] for x in rr])
            k = rr[0]['kappa']
            a = E.stats(av)
            c = E.stats(ce)
            t = tt[v, k]
            row = {'speed_kmh': v, 'arm': arm, 'kappa': k, 'accuracy': a['mean'], 'accuracy_sd': a['sd'], 'accuracy_min': a['min'], 'accuracy_max': a['max'], 'accuracy_ci_lo': a['ci95'][0], 'accuracy_ci_hi': a['ci95'][1], 'ce': c['mean'], 'ce_sd': c['sd'], 'ce_ci_lo': c['ci95'][0], 'ce_ci_hi': c['ci95'][1], 'rho_real': t['rho_real'], 'rho_imag': t['rho_imag'], 'rho_abs': t['rho_abs'], 'positive_lobe': t['positive_lobe'], 'residual_lambda': t['residual_lambda'], 'gradient_cos_mean': float(np.mean([x['gradient_cos_mean'] for x in rr]))}
            cells.append(row)
            lookup[v, arm] = row
            vec[v, arm] = av
    writecsv(R / 'results/cells.csv', cells)
    writecsv(R / 'figures/phase_data.csv', cells)
    k1 = []
    for v in C['speeds_kmh']:
        rr = sorted([r for r in cells if r['speed_kmh'] == v and r['arm'] != 'CSI'], key=lambda r: -r['kappa'])
        low = next((i for (i, r) in enumerate(rr) if r['accuracy'] < 0.8), None)
        star = theory['kappa_star'][str(v)]
        if low is None:
            k1.append({'speed_kmh': v, 'theory_kappa': star, 'observed_kappa': None, 'status': 'No observed crossing; censored'})
            continue
        r = rr[low]
        high = rr[low - 1] if low else None
        seedcritical = []
        for s in C['seeds']:
            ss = sorted([x for x in rows if x['seed'] == s and x['speed_kmh'] == v and (x['arm'] != 'CSI')], key=lambda x: -x['kappa'])
            seedcritical.append(next((x['kappa'] for x in ss if x['accuracy'] < 0.8), None))
        k1.append({'speed_kmh': v, 'theory_kappa': star, 'observed_kappa': r['kappa'], 'absolute_difference': abs(r['kappa'] - star), 'accuracy_below': r['accuracy'], 'bracket': [r['kappa'], high['kappa'] if high else None], 'accuracy_above': high['accuracy'] if high else None, 'main_8_kappa_status': 'Left censored: all main kappa cells at least 80%; only fixed falls below' if r['kappa'] == 0 else 'First crossing observed on main grid', 'observed_kappa_by_seed': seedcritical, 'status': 'Endpoint error at most .1' if abs(r['kappa'] - star) <= 0.1 else 'Endpoint error greater than .2: falsified' if abs(r['kappa'] - star) > 0.2 else 'Endpoint error greater than .1 but at most .2'})
    events = json.loads((R / 'results/stop_events.json').read_text())
    trigger = any((x['claim'] == 'K1' and x['triggered'] for x in events))
    k1v = 'Falsified: preregistered hard threshold triggered' if trigger else 'Finite-grid operational support; continuous low-speed boundary not located' if all((x.get('absolute_difference', 99) <= 0.1 for x in k1)) else 'Prediction not met; not falsified or censored'
    mobile = [x for x in cells if x['arm'] != 'CSI' and x['kappa'] > 0]
    physical = [x for x in cells if x['arm'] != 'CSI']

    def corr(rr, rounding=False):
        return float(spearmanr([round(x['rho_real'], 12) if rounding else x['rho_real'] for x in rr], [x['accuracy'] for x in rr]).statistic)
    sr = corr(mobile)
    ix = E.rng(C['bootstrap_seed'], 'bootstrap').integers(0, 8, (C['bootstrap_replicates'], 8))
    av = np.array([vec[x['speed_kmh'], x['arm']] for x in mobile])
    boot = av[:, ix].mean(axis=2).T
    ranks = rankdata(boot, axis=1)
    ranks -= ranks.mean(axis=1, keepdims=True)
    xr = rankdata([x['rho_real'] for x in mobile])
    xr -= xr.mean()
    br = ranks.__matmul__(xr) / np.sqrt(np.sum(ranks * ranks, axis=1) * np.sum(xr * xr))
    k2 = {'verdict': 'Finite-configuration numerical support' if sr >= 0.85 else 'Falsified: preregistered hard threshold triggered' if sr < 0.6 else 'Prediction not met; not falsified', 'spearman': sr, 'bootstrap_ci95': np.quantile(br, [0.025, 0.975]).tolist(), 'n_cells': 24, 'including_fixed_27_cells': corr(physical), 'including_fixed_CSI_30_cells': corr(cells), 'round12_theory_ties_sensitivity': corr(mobile, True), 'by_speed': {str(v): corr([x for x in mobile if x['speed_kmh'] == v]) for v in C['speeds_kmh']}}
    k3 = []
    for pair in theory['K3_pairs']:
        (vs, ks) = pair['second']
        (vf, kf) = pair['first']
        a = lookup[vs, 'MA_' + str(ks)]
        b = lookup[vf, 'MA_' + str(kf)]
        delta = E.stats(vec[vs, a['arm']] - vec[vf, b['arm']])
        k3.append({'second': a, 'first': b, 'abs_phi_gap': pair['abs_phi_gap'], 'paired_accuracy_difference': delta})
    claims = {'K1': {'verdict': k1v, 'speeds': k1, 'hard_line_triggered': trigger}, 'K2': k2, 'K3': {'verdict': 'Descriptive comparison complete' if k3 else 'Frozen grid does not cover second positive lobe', 'comparisons': k3}}
    E.dump(R / 'results/claims.json', claims)
    E.dump(R / 'results/analysis_metadata.json', {'created_at': E.now(), 'n_rows': len(rows), 'n_cells': len(cells), 'bootstrap_unit': 'independent root, jointly resampled across cells', 'precheck_not_pooled': True, 'no_grid_expansion': True, 'machine_precision_tie_sensitivity_is_descriptive': True})
    print(json.dumps(claims, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
