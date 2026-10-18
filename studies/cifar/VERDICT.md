# CIFAR-10 split-learning benchmark — VERDICT (2026-10-17; , 2x RTX  6000)
SPEC.md frozen before training (FREEZE_SHA256*.txt); amendment 1 (data source) before any formal run. 21/21 runs succeeded.

| Arm (Nr=8) | final acc per seed (%) | mean | epochs to 80% | epochs to 90% | acc at epoch 10 |
|---|---|---|---|---|---|
| ideal CSI | 93.94, 94.26, 93.85 | 94.02 | 8, 6, 8 | 26, 25, 24 | 84.10 |
| MA r=0 | 93.60, 94.10, 93.72 | 93.81 | 8, 8, 6 | 27, 26, 27 | 81.56 |
| r=0.1 | 93.79, 93.80, 94.06 | 93.88 | 9, 7, 5 | 27, 24, 26 | 84.53 |
| r=0.2 | 93.66, 93.60, 93.78 | 93.68 | 9, 8, 9 | 29, 25, 26 | 83.50 |
| r=0.25 | 94.16, 93.69, 93.62 | 93.82 | 8, 8, 8 | 24, 26, 29 | 83.53 |
| r=0.3 | 94.03, 93.30, 93.65 | 93.66 | 9, 9, 9 | 29, 28, 26 | 83.33 |
| fixed r=0.467 | 93.11, 92.56, 93.17 | 92.95 | 13, 12, 9 | 31, 31, 30 | 77.70 |

- P1 (r=0 vs CSI within 1 pp): **SUPPORTED** (0.21 pp).
- P2 (fixed clearly worse: >=5 pp or >=3x epochs to 80%): **FALSIFIED** (0.86 pp, ratio 1.5).
- P3 (Spearman(r, acc) <= -0.8): **NOT MET** (-0.71).
Interpretation: with an 8-antenna MMSE terminal and a ResNet-18 encoder, the uncompensated walking terminal loses little final accuracy
(one positive eigen-direction remains, lmax ~0.58) and is ~1.2x slower to 90%. Extension 1 (SPEC_ADDENDUM_ext1.md) tests 2-antenna
terminals and weak-eigenvalue residuals. results/analysis.json

## Extension 1 (SPEC_ADDENDUM_ext1.md, frozen before running; 24/24 runs succeeded)
| Arm | final acc per seed (%) | mean | acc at ep 10 | epochs to 90% |
|---|---|---|---|---|
| Nr=2 CSI | 93.93, 93.72, 93.51 | 93.72 | 82.39 | 28, 28, 26 |
| Nr=2 r=0 | 93.64, 93.78, 93.88 | 93.77 | 82.62 | 27, 26, 25 |
| Nr=2 r=0.25 | 92.59, 92.65, 92.45 | 92.56 | 77.64 | 32, 32, 34 |
| Nr=2 r=0.47 | 86.63, 87.06, 82.88 | 85.52 | 44.82 | never |
| Nr=2 r=0.60 | 37.11, 33.11, 26.09 | 32.10 | 21.50 | never |
| Nr=2 r=0.85 | 41.94, 39.03, 53.32 | 44.76 | 22.63 | never |
| Nr=8 r=0.85 | 37.51, 56.51, 60.03 | 51.35 | 32.42 | never |
| Nr=8 r=1.90 | 4.11, 16.33, 30.11 | 16.85 | 15.86 | never |
E1 SUPPORTED (0.05 pp); E2 SUPPORTED (drop 49.0 pp); E3 SUPPORTED (Spearman 0.90 with lmax); E4 SUPPORTED (51.35%, 16.85% << 92.95%).
Note: Nr=2 r=0.60 (lmax +0.079) is worse than r=0.85 (lmax -0.025) despite the larger lmax; the ordering is not strict at the low end.
results_ext1/analysis.json

## Extension 2 (SPEC_ADDENDUM_ext2.md, frozen before running; 27/27 runs succeeded)
Speed-consistent trajectories (step v*0.15 s), paired MA at the same speed:
| config | final acc (%) per seed | mean |
|---|---|---|
| v=1.82, Nr=8, fixed (0.85 lambda) | 61.04, 50.69, 64.71 | 58.81 |
| v=1.82, Nr=8, MA r=0 | 93.99, 93.88, 93.65 | 93.84 |
| v=1.82, Nr=2, fixed | 51.30, 50.81, 57.53 | 53.21 |
| v=1.82, Nr=2, MA r=0 | 93.58, 93.85, 93.93 | 93.79 |
| v=4.07, Nr=8, fixed (1.9 lambda) | 17.34, 9.95, 20.91 | 16.07 |
| v=4.07, Nr=8, MA r=0 | 93.75, 93.98, 93.90 | 93.88 |
Intervention at v=1, Nr=8 (projector from Herm(Bbar) at 0.467 lambda):
| fixed, positive eigenspace removed | 18.54, 15.22, 18.91 | 17.56 |
| fixed, random same-dimension subspace removed | 93.16, 92.83, 93.07 | 93.02 |
| MA r=0, same positive eigenspace removed | 93.73, 93.95, 93.81 | 93.83 |
X1 SUPPORTED (gaps 35.0, 40.6 pp); X2 SUPPORTED (77.8 pp); X3 SUPPORTED (removal -75.4 pp vs formal fixed 92.95; random +0.07; MA +0.02).
results_ext2/analysis.json

## Extension 3 (SPEC_ADDENDUM_ext3.md, frozen before running; 24/24 runs succeeded)
| config | final acc (%) per seed | mean |
|---|---|---|
| v=1.82 Nr=8 CSI | 93.75, 93.91, 93.87 | 93.84 (MA 93.84) |
| v=4.07 Nr=8 CSI | 93.82, 93.71, 93.82 | 93.78 (MA 93.88) |
| v=1.82 Nr=2 CSI | 93.83, 94.15, 93.80 | 93.93 (MA 93.79) |
| 2x2 v=1 CSI | 93.33, 93.96, 93.92 | 93.74 |
| 2x2 v=1 MA | 93.78, 93.74, 94.23 | 93.92 |
| 2x2 v=1 fixed (0.467) | 44.07, 51.94, 43.81 | 46.61 |
| 2x2 v=1.82 MA | 93.96, 93.76, 93.76 | 93.83 |
| 2x2 v=1.82 fixed (0.85) | 29.00, 30.86, 33.36 | 31.07 |
Y1 SUPPORTED (|MA-CSI| 0.003, 0.09, 0.14 pp); Y2 SUPPORTED (0.18 pp); Y3 SUPPORTED (62.8 pp). Descriptive: 2x2 walking fixed -47.3 pp.
Training-free spectrum for the CIFAR seeds' geometries: results/spectrum_cifar_seeds.json (2x2: lmax 0.233 at 0.467 lambda, n+=1; negative at 0.6 and 0.85).

## Extension 4 — CSI-free terminal (SPEC_ADDENDUM_ext4.md, frozen before formal runs; 30/30 runs succeeded)
Energy receiver (log(1+|y|^2) per antenna and use), no CSI anywhere. Pilot (seed 1, non-formal): 90.45% ideal backward, 8.72% fixed.
| config | final acc (%) per seed | mean | acc at ep 10 |
|---|---|---|---|
| Nr=8 ideal backward | 93.70, 93.56, 93.89 | 93.72 | 82.58 |
| Nr=8 MA r=0 | 93.81, 93.49, 93.78 | 93.69 | 79.95 |
| Nr=8 r=0.10 | 93.68, 93.75, 93.46 | 93.63 | 82.28 |
| Nr=8 r=0.20 | 93.18, 93.55, 93.60 | 93.44 | 80.58 |
| Nr=8 r=0.25 | 92.63, 92.65, 92.19 | 92.49 | 74.69 |
| Nr=8 r=0.30 | 14.47, 17.31, 16.59 | 16.12 | 14.42 |
| Nr=8 fixed (0.467) | 10.00, 10.00, 10.00 | 10.00 | 10.00 |
| 2x2 ideal backward | 93.76, 93.40, 93.60 | 93.59 | 81.36 |
| 2x2 MA r=0 | 93.88, 93.60, 93.87 | 93.78 | 82.82 |
| 2x2 fixed (0.467) | 10.00, 10.00, 10.00 | 10.00 | 10.00 |
Z1 SUPPORTED (0.02, 0.20 pp); Z2 SUPPORTED (83.7, 83.8 pp); Z3 SUPPORTED (Spearman 1.00 with scalar Re phi).
results_ext4/analysis.json

## Extension 5 (SPEC_ADDENDUM_ext5.md, frozen; 6 runs, none rerun) — energy-matched controls
Positive-eigenspace removal (Ext. 2): 17.56%. Energy-matched rescaling without direction change ("scale"): 93.15 / 92.46 / 93.23,
mean 92.95% (the unmodified fixed antenna: 92.95%). Random same-dimension removal rescaled to the same energy ("randmatch"):
92.68 / 92.44 / 92.66, mean 92.59%.
Y1 SUPPORTED, Y2 SUPPORTED: the collapse under positive-eigenspace removal is caused by the removed direction, not by the removed energy.

## Extension 6 (SPEC_ADDENDUM_ext6.md, frozen; 9 runs, none rerun) — antenna selection where the actuator is out of reach
Selection residual -0.15 lambda at 1.82 m/s and -0.10 lambda at 4.07 m/s (half-wavelength elements along the motion).
Nr=8, 1.82 m/s: 93.48 / 93.84 / 93.29, mean 93.54% (exact restoration 93.84%, fixed 58.81%).
Nr=8, 4.07 m/s: 93.90 / 94.05 / 93.94, mean 93.96% (exact restoration 93.88%, fixed 16.07%).
Nr=2, 1.82 m/s: 93.44 / 93.49 / 94.00, mean 93.64% (exact restoration 93.79%, fixed 53.21%).
Z SUPPORTED: every configuration within 0.3 pp of exact restoration at the same speed.

## Extension 7 (SPEC_ADDENDUM_ext7.md, frozen; 6 runs, none rerun) — antenna selection at 1 m/s
Residual -0.033 lambda. Nr=8: 93.98 / 94.00 / 93.79, mean 93.92% (exact restoration 93.81%, fixed 92.95%).
Nr=2: 93.37 / 93.87 / 93.84, mean 93.69% (exact restoration 93.77%, fixed 85.5%). W SUPPORTED (both within 0.5 pp).

## Extension 8 (SPEC_ADDENDUM_ext8.md, frozen; 24 runs, none rerun) — period-consistent common-budget comparison
Trajectories regenerated at each arm's exchange period (28 GHz equivalents: 5 ms for fixed and selection; 18.66 / 27.30 ms for
mechanical exact restoration at 1 / 1.82 m/s; mechanical unreachable at 4.07 m/s). Nr = 8, seed means. Time to 90%:
1 m/s: selection 54.6 s, fixed 62.4 s, mechanical 181.9 s.  1.82 m/s: selection 48.8 s, fixed never (final 34.7%), mechanical 287.5 s.
4.07 m/s: selection 54.6 s, fixed never (final 20.5%). Final accuracy of selection 93.7-94.0%. Z8 SUPPORTED.
Not used in the frozen paper (v27); held for the revision as the period-consistent version of Appendix J.
