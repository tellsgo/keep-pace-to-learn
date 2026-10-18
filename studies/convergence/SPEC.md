# Convergence-rate study — SPEC (frozen before any training)

Inherits residual study physics and task unchanged (src/core.py is a byte copy of residual study's core.py): CDL-A 23x20 rays,
4Rx/2Tx, 3.5 GHz, single polarization, narrowband, independent drops, forward/return share ray phases,
return residual displacement d = r/lambda along x. Reverse noise and constant-16 seed scaling as residual study.
New: 600 updates (19,200 training examples), test accuracy/CE every 10 updates on 8,192 test examples.

## Question
The convergence corollary (Bottou et al. Thm 4.8 with mu = Re rho, mu_G = |rho|) predicts that, for a fixed-step
SGD encoder update, aging enters through the drift eta*Re(rho): the number of updates to a target scales ~ 1/Re(rho)
in the drift-dominated regime (steeper when noise dominates).

## Grid and seeds
d in {0, .05, .10, .15, .20, .25, .30} (all inside the first positive lobe, first zero d=0.35175 for these seeds;
training-free Re phi in results/theory.json). Two encoder optimizers: SGD (fixed eta, calibrated below) and Adam (residual study lr .025).
Receiver always Adam (local, exact digital gradient), as in residual study. Formal seeds 710170100-107 (8), paired across arms.
Calibration seeds 710170200-203 (4), residual 0 only.

## Calibration (before formal runs)
src/calibrate.py rule as in its docstring; eta grid {.05,.1,.2,.4,.8,1.6}; only residual 0. Result frozen in results/calibration.json.

## Metric and predictions
K90 = first checkpoint (multiple of 10 updates) at which test accuracy >= 90%; censored at 600.
Per residual: median K90 over 8 seeds (censored seeds count as 600+; if the median itself is censored the residual
is excluded from the fit and reported).
P1 (SGD): OLS slope of log(median K90) on log(Re phi) across non-censored residuals.
  Support: slope in [-1.4, -0.6]. Falsified: slope outside [-2.0, -0.4], or fewer than 4 residuals usable.
  In between: not met, not falsified.
P2 (descriptive): accuracy curves versus (update index x Re phi) for SGD; descriptive collapse plot only.
P3 (descriptive): same slope for Adam; no prediction threshold.
P4 (descriptive): compare R^2 of log K90 fits using Re phi versus |phi| as the predictor.
No grid extension, no seed change, no rerun after seeing formal results. All seeds and failures retained.

## Amendment 1 (2026-10-17, after calibration attempt 1, before any formal or moving-residual run)
Attempt 1 (results/calibration_attempt1_floored.json, logs/calibrate_attempt1_floored.log): every stable eta reached 90%
at the first 10-update checkpoint at residual 0, so K90 was floored and the rule could not discriminate. Changes:
eval_every 10 -> 1 (K90 now in single updates); eta grid {.0025,.005,.01,.02,.05}; choose the largest eta whose
median K90 at residual 0 is >= 40 updates (all 4 calibration seeds reaching 90% within 300). Predictions P1-P4 unchanged.

## Analysis revision 1 (after formal runs; analysis code only)
analyze.py rev0 averaged a censored sentinel into an even-n median (Adam, d=.25, 4/8 censored). Rev1 treats the median
as censored whenever the 5th order statistic of 8 is censored, as SPEC already required. Rev0 output kept as
results/analysis_rev0_median_bug.json. SGD P1 inputs unaffected (max 2/8 censored).
