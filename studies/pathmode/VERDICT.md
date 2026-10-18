# VERDICT — path-mode precoding study (frozen SPEC.md; all 90 runs retained, none rerun)
Q1 SUPPORTED: path-mode precoding at every certified residual (mu = 0.96 ... 0.24, up to r = 0.47 lambda) ends at 93.65-94.46%,
   against 93.55% at r = 0 (no residual more than 2 pp below).
Q2 SUPPORTED: T90 = 500 updates at every residual (T(0) = 500); the bound 1.5 T(0)/mu + 250 ranges 1000-3381. The test is weak:
   the 250-update evaluation grid does not resolve rate differences on this task.
Q3 SUPPORTED: measured parameter-level alignment >= mu - 0.01 at every logged update (smallest margin 0.0000 at r = 0, i.e. exact,
   and positive margins elsewhere); mean alignment 0.98 / 0.92 / 0.82 / 0.69 / 0.59 at mu = 0.96 / 0.84 / 0.66 / 0.43 / 0.24.
Descriptive (no prediction):
 - The unstructured encoder trains equally well on this well-conditioned geometry (93.35-94.26%); its minimum alignment can fall
   below mu (0.218 at mu = 0.24), consistent with Theorem 2 giving a guarantee only for the path-mode structure.
 - Along geometry, r = 0.47 lambda (one path reversed, mu = -0.94, outside Theorem 2): both encoders still reach 88.0%
   (vs 93.9-94.4% at r = 0), with T90 never reached; the certificate is sufficient, not necessary.
 - Mean symbol energy 61-104 (no normalization); the closely-spaced geometry was excluded before the runs (energy 2420 at r = 0).
Interpretation: Theorem 2's identity and drift bound hold exactly in an implemented MMSE + adaptive-receiver system; this small
task does not discriminate between path-mode precoding and an unstructured encoder.
