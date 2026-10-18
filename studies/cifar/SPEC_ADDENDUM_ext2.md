# Extension 2 — frozen before running (2026-10-17); answers an independent strict review (round 7, M1/M2)
Same code/config as the formal study except the options below. Seeds 710172100-102. Trajectory step per update = v * 0.15 s.
(a) Speed-consistent uncompensated terminals: v = 1.8202 m/s (fixed antenna, r = v*tau = 0.85 lambda) and v = 4.0686 m/s (r = 1.9 lambda),
    each paired with exact MA restoration (r = 0) at the SAME speed/trajectory. Nr=8 for both speeds; Nr=2 for v = 1.8202.
(b) Intervention at v = 1 m/s, Nr=8: fixed antenna (r = 0.467) with the positive-eigenspace component of the combiner-output gradient
    removed before the backward transmission (projector from Herm of the mean aged operator at r = 0.467, MMSE combiner, 300 fields,
    RNG 777); control 1: a random subspace of the same dimension removed (RNG 778); control 2: the same positive-eigenspace removal
    under exact restoration (r = 0).
Predictions (mean final accuracy, 3 seeds):
X1: v=1.8202: acc(fixed) <= acc(MA same v) - 20 pp for Nr=8 and for Nr=2 -> supported; within 3 pp for both -> falsified.
X2: v=4.0686, Nr=8: acc(fixed) <= acc(MA same v) - 20 pp -> supported; within 3 pp -> falsified.
X3: removal drops fixed-antenna accuracy by >= 10 pp relative to the formal fixed arm (92.95%), random removal drops < half of that,
    and the same removal at r = 0 drops MA accuracy (93.81%) by < half of that -> supported; removal drop < 2 pp -> falsified.
No reruns or changes after results; all runs retained.
