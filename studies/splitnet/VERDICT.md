
Frozen predictions (SPEC.md, FREEZE_SHA256*.txt), eta=0.05 by calibration rule on residual 0 only.

| Claim | Verdict | Key numbers | Evidence |
|---|---|---|---|
| P1 K85 ~ (Re phi)^-1 with scalar CDL-A Re phi | **NOT MET** (not falsified) | slope -0.45, R^2 0.88, 7 residuals; median K85 150 for r<=0.2, 200 at .25, 300 at .30 | results/analysis.json |
| P2 exact restoration = ideal CSI (<=1 pp) | **SUPPORTED** | 95.54% vs 95.54% (identical runs up to backward noise) | results/analysis.json |
| P3 fixed antenna (r=0.467 lambda, Re phi=-0.39) fails (<=30%) | **FALSIFIED** | final 82.83% (8-seed mean), median K85 1337.5 (1/8 censored) vs 150; mean grad cosine +0.18 | results/analysis.json |

## Post-hoc diagnostics (after formal results; exploratory, not frozen)
- Receiver-aware trace correlation Re rho_eff (MMSE combiner, isotropic seeds): 1, .979, .917, .817, .686, .531, .363 for r=0..0.30 and -0.170 at 0.467. K85 slope vs Re rho_eff: -0.68. (results/posthoc_rho_eff.json)
- Hermitian part of the mean aged operator Bbar_r = E[M H_f H_r^H M^H] (4x4): number of positive eigenvalues (mean over 8 geometries) 4,4,4,4,4,4 for r<=0.25, 3.125 at 0.30, 1.0 at 0.467 (eigs -.714,-.361,-.149,+.557). (results/posthoc_eig.json)
- Interpretation: with a multi-stream encoder and array receiver, a negative scalar/trace correlation does not stop learning; the encoder can signal in the remaining positive eigen-subspace, at reduced speed and accuracy. Aging shrinks the usable transmit subspace; exact restoration restores it fully (B_0 PSD).

Scope: one dataset (sklearn digits), one network, one eta, 8 seeds, CDL-A orientation, receiver with ideal forward CSI, 1 m/s with T=0.15 s along a continuous trajectory, fields redrawn every 100 updates. K85 resolution 25 updates.

## Addendum: projection and intervention test (SPEC_ADDENDUM_mechanism.md, frozen before running)
Verdict by frozen rule: **PARTIAL / NOT MET** (projection threshold missed).
- A: positive-subspace energy fraction of the combiner-output gradient rose by 17.94 pp (first 200 vs last 200 updates; threshold 20 pp).
- B: removing the positive-subspace component from the backward transmission: final accuracy 32.68% vs 82.83% normal (-50.15 pp);
  energy-matched random-subspace control 81.60% (-1.24 pp). Normal arm reproduces the formal fixed-antenna result exactly (82.83%).
- Interpretation: the intervention supports the positive-subspace mechanism causally; the passive drift of the gradient toward it is
  smaller than predicted. results/mechanism.json

## Addendum: wall-clock re-simulation with period-consistent trajectories (SPEC_ADDENDUM_wallclock.md, frozen before running)
**SUPPORTED.** Median K85 x period: a=50: exact 22.39 s (T=149.3 ms, K85 150), r=0.15 18.16 s (T=121.1, 150), r=0.20 19.49 s (T=111.4, 175),
r=0.30 27.31 s (T=91.0, 300); a=10: exact 71.50 s (T=476.6, 150), r=0.15 52.04 s (T=346.9, 150), r=0.20 49.31 s (T=303.4, 162.5).
Savings vs exact: 18.9% (a=50), 31.0% (a=10). No movement (T=40 ms): K85 median 1175 (1/8 censored), 47.0 s - comparable to the slow
actuator's partial anchoring, but its final accuracy plateaus near 83% (formal study) versus 95.5% with compensation.
results/wallclock.json. Also: VERDICT wording "energy-matched" for the mechanism control is corrected to "same dimension" (code matches rank only).
