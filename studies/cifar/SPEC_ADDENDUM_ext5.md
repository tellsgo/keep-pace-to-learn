# Extension 5 — energy-matched intervention controls; frozen before running (2026-10-17)
Answers an independent blind review (v21): the Extension 2 control removed a random subspace of the same DIMENSION, not the same
gradient ENERGY, so the drop under positive-eigenspace removal could be an effective step-size/SNR loss rather than a loss of direction.
Code: src/run_ext5.py = src/run.py with only the projection branch changed (diff in this addendum's FREEZE file). Same config,
seeds 710172100-102, v = 1 m/s, Nr = 8, fixed antenna (r = 0.467 lambda), projector from Herm of the mean aged operator at
r = 0.46699 (MMSE combiner, 300 fields, RNG 777), random projector RNG 778 (same dimension).
Arms (per example, before the backward transmission; g_pos = backward signal after removing the positive-eigenspace component of d):
  scale     : full backward signal, rescaled to the norm of g_pos (same energy as the removal, direction unchanged)
  randmatch : random same-dimension subspace removed, then rescaled to the norm of g_pos
Reference (Extension 2, not rerun): positive-eigenspace removal 17.6%; formal fixed antenna 92.95%.
Predictions (mean final accuracy over 3 seeds):
Y1: acc(scale) >= 17.6 + 20 pp -> SUPPORTED (direction, not energy, carries the effect); acc(scale) <= 17.6 + 5 pp -> FALSIFIED.
Y2: acc(randmatch) >= 17.6 + 20 pp -> SUPPORTED; acc(randmatch) <= 17.6 + 5 pp -> FALSIFIED.
Otherwise NOT MET. No reruns or changes after results; all runs retained.
