# Addendum (frozen before running): projection and intervention test of the positive-subspace explanation
Proposed by an independent strict review. Fixed-antenna arm (r = 0.467 lambda), same 8 formal seeds, same training as the formal study.
Per geometry (seed), Bbar = mean over 400 random fields/positions of M H_f H_r^H M^H with the MMSE combiner (as posthoc_eig.py,
fresh RNG 777); V+ = eigenvectors of Herm(Bbar) with positive eigenvalues (dimension n+).
A (projection, descriptive + threshold): during normal training, record the fraction of energy of the combiner-output gradient d
(per sample, then averaged over the minibatch) lying in span(V+). Prediction: mean fraction over updates 1301-1500 exceeds that over
updates 1-200 by >= 20 percentage points.
B (intervention): train with the backward transmission using d' = (I - P+) d (positive-subspace component removed), and a control
using d'' = (I - P_rand) d with P_rand a projector onto a random subspace of the same dimension n+ (fixed per seed, RNG 778).
Prediction: final accuracy (intervention) <= final accuracy (normal fixed arm) - 10 pp, and the control loses < half that amount.
Falsified if fraction change <= 0 AND intervention loss < 2 pp. Other outcomes: partial / not met. No reruns or grid changes.
