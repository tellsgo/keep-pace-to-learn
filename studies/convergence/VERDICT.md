
| Claim | Verdict | Key numbers | Evidence |
|---|---|---|---|
| P1 SGD: log K90 vs log Re phi slope in [-1.4,-0.6] | **SUPPORTED** (finite configuration) | slope -1.040, R^2 0.993, 7/7 residuals; seed bootstrap 95% [-1.232, -0.681] | results/analysis.json, results/slope_bootstrap.json |
| P4 Re phi vs abs phi as predictor (descriptive) | Re phi fits better | R^2 0.993 (Re) vs 0.970 (abs); abs-slope -1.62 | results/analysis.json |
| P3 Adam (descriptive) | fast at large Re phi; plateaus at small Re phi | median K90 4/4/4.5/5.5/8 for d<=.2; d=.25 and .30 median censored; final acc 85.9/73.3/52.7% at d=.2/.25/.3 vs SGD 97.4/98.1/87.1% | results/analysis.json |

Medians of K90 (SGD, 8 seeds): d=0:33, .05:34, .10:36, .15:41.5, .20:53, .25:73.5, .30:172.5 (Re phi 1, .973, .893, .767, .603, .412, .209).
Censored seeds (SGD): d=.20: 1/8, d=.30: 2/8; all others 0.

Process: SPEC frozen (FREEZE_SHA256.txt); calibration attempt 1 floored (kept); amendment 1 before any moving-residual run
(FREEZE_SHA256_amend1.txt); eta=0.005 chosen by rule on residual 0 only; formal frozen (FREEZE_SHA256_formal.txt).
Analysis rev1 fixes a censored-median bug affecting only Adam d=.25 (rev0 kept). No grid extension, seed change or rerun.

Scope: one CDL-A orientation, 3 km/h worlds (residual specified directly in wavelengths), the residual study task, one SGD step and
one Adam setting, 600 updates. The scaling is an observed finite-configuration law consistent with the drift term
eta*Re(rho) of the corollary; it is not a proof that every optimizer/task obeys 1/Re(rho). The Adam plateau is a single
learning-rate observation; its mechanism (noise floor of the bound with an adaptive step) is an interpretation.

Follow-ups (not run): (1) Adam with lr scaled by Re phi should remove the plateau (falsified if d=.25 final acc stays <90%);
(2) the slope for SGD with larger eta should steepen toward -2 as the noise term dominates (falsified if slope > -0.8 at eta=.02).

## Addendum: Adam with encoder step scaled by Re phi (falsification test proposed by an independent strict review)
SPEC_ADDENDUM_adam_scaled.md frozen before running (FREEZE_SHA256_addendum.txt). **FALSIFIED**: at d=0.25 mean final accuracy
80.95% (< 90%); unscaled Adam 73.25%, SGD 98.10%. Descriptive: d=0.20 95.26% (unscaled 85.85%), d=0.30 66.53% (unscaled 52.74%).
Scaling helps but does not close the gap to fixed-step SGD; the noise-floor interpretation is not established. results/adam_scaled.json
