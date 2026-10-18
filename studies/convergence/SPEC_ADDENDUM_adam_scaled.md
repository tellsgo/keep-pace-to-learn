# Addendum (frozen before running): Adam with encoder step scaled by Re phi
Proposed by an independent strict review as a falsification test. Residual d=0.25 (Re phi=0.41244 from results/theory.json),
same 8 formal seeds, 600 updates, same everything as the Adam arm except encoder lr = 0.025 * Re phi.
Prediction: mean final test accuracy >= 90% -> supported; < 90% -> falsified (the scaling alone does not fix this configuration).
Also reported descriptively: same scaling at d=0.20 and d=0.30.
