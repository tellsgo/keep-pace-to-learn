# Digits split-learning study — SPEC (frozen before training)

## Purpose
Test, on a real 10-class dataset with a multi-layer split network, a MIMO OTA layer, and channels that evolve along a
continuous trajectory, (P1) the time-rescaling prediction K_target ~ (Re phi)^-1 for SGD, (P2) that exact mechanical
restoration matches ideal-CSI training, (P3) that an uncompensated terminal at walking speed fails.

## Data and model
sklearn digits (1797 images, 8x8, 10 classes), standardized with training statistics; fixed random split per seed:
1200 train / 597 test. Device (BS side, no CSI): 64 -> 64 tanh -> 8 real -> s in C^4, s = c/||c|| (unit power).
Channel: y = H_f s + n, H_f in C^{8x4}, CN noise std 0.1 per entry. Terminal (receiver) knows H_f (forward pilots):
MMSE combining s_hat = (H^H H + 0.01 I)^-1 H^H y; receiver MLP [Re s_hat, Im s_hat] (8) -> 64 tanh -> 10 softmax, CE loss.
Backward: terminal computes delta = 2 dL/dy* digitally, transmits delta/||delta|| at t_f+tau, BS receives
H_r^H (delta/||delta||) + w (CN std 0.03), rescales by the fed-back norm, backpropagates locally. Receiver updates with exact local gradients.
Ideal-CSI arm: BS gradient = H_f^H delta exactly, no backward noise.

## Channel and motion
CDL-A ray field (23x20 rays, TR 38.901 table as in residual study code), 3.5 GHz; terminal ULA (8, half-wavelength) along y, BS ULA (4) along y,
motion along x. Field (ray phases) static within a block of 100 updates, redrawn between blocks. Terminal position advances by
v*T = 1 m/s * 0.15 s per update (continuous trajectory). Forward at p_k, backward at p_k + r (residual r in wavelengths).
All 32 samples of a minibatch share the exchange's channel. Test: each test minibatch uses a fresh random field and position
(forward only, receiver CSI as in training).

## Arms
Residual r/lambda in {0, .05, .10, .15, .20, .25, .30} (all inside the first positive lobe), fixed antenna r = v*tau/lambda = 0.467,
and ideal CSI. 8 formal seeds 710171100-107 (paired: same data split, init, fields, noise across arms). SGD (no momentum)
for both sides, same step eta; 1500 updates; test accuracy every 25 updates.

## Calibration (residual 0 only, calibration seeds 710171200-203)
eta grid {0.01,0.02,0.05,0.1,0.2,0.5}. Choose the largest eta for which all 4 seeds reach 85% test accuracy within 750 updates
and whose median K85 >= 100 updates; if none has median >=100, choose the eta with largest median K85 among those where all reach.

## Predictions
P1: OLS slope of log(median K85) on log(Re phi) over non-censored residuals in [-1.4,-0.6] supported; outside [-2,-0.4] or fewer than
4 usable residuals: falsified; otherwise not met. (Median censored if the 5th order statistic of 8 is censored at 1500.)
P2: |mean final acc(r=0) - mean final acc(CSI)| <= 1.0 pp supported; > 3 pp falsified.
P3: fixed-antenna mean final accuracy <= 30% supported; >= 60% falsified.
No grid extension, seed change, or rerun after formal results. All seeds and failures retained.
