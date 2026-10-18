# Path-mode precoding study — frozen before any r > 0 run (2026-10-17)
Purpose: instantiate Theorem 2 (path-mode precoding) of the JSAC draft v21 under a channel-dependent MMSE combiner and an adaptive
receiver, which the scalar and fixed-combiner results do not cover. Code: src/pm.py (hash in FREEZE_SHA256.txt).
Setting (lambda = 1 units): sparse L = 3 path channel, BS ULA Nt = 4 (transmitter), terminal ULA Nr = 4 along its motion (receiver),
half-wavelength spacing; i.i.d. CN(0,1/L) path gains and a uniform position per update; residual r along the terminal array axis.
Task: sklearn digits (64 standardized pixels + bias), 1400 train / 397 test; linear encoder on the fixed features;
receiver = MMSE combiner + MLP (8-64-10, tanh), exact local gradients; SGD lr 0.05, batch 32, 12000 updates, no power normalization
(Theorem 2 excludes it; mean symbol energy is reported). Forward noise 0.1, backward noise 0.05 (per-example normalized return).
Encoders: path = F C x with F = V (V^H V)^{-1} (A = I; uses path geometry, not fading); generic = W x (unstructured, Nt outputs).
Geometries (terminal cosines c_l along the motion; BS cosines t_l): sep c = (0.10,-0.35,0.45), t = (-0.55,0.05,0.62), cond(V) = 1.34;
along c = (0.10,-0.35,0.95), same t. Theorem 2 drift mu = min_l cos(2 pi c_l r).
Pilot (r = 0 only, seed 1, 3000 updates): sep path 94.96%, sep generic 94.46%. A third geometry with closely spaced departures
(t = (-0.10,0,0.10), cond(V) = 40.5) was dropped before the formal runs: path-mode precoding inflated the mean symbol energy to 2420
(generic 47), the power cost that Theorem 2 must assume away.
Grid (seeds 1-5): sep x {path, generic} x r in {0, 0.1, 0.2, 0.3, 0.4, 0.47}; along x {path, generic} x r in {0, 0.2, 0.47}.
mu (sep): 1, 0.960, 0.845, 0.661, 0.426, 0.239. mu (along): 1, 0.368, -0.946.
Predictions (seed means):
Q1 (descent where Theorem 2 certifies it): sep path at every r with mu > 0 ends within 2 pp of sep path at r = 0 -> SUPPORTED;
   any such r more than 10 pp below -> FALSIFIED.
Q2 (rate): let T(r) = first evaluation step (every 250 updates) at which the seed-mean test accuracy reaches 90%.
   T(r) <= 1.5 * T(0) / mu(r) + 250 for every sep path residual -> SUPPORTED; T(r) > 3 * T(0) / mu(r) + 250 for any -> FALSIFIED.
Q3 (exactness of the identity): the measured parameter-level alignment Re<g, g_aged>/||g||^2 of path-mode runs is >= mu(r) - 0.01
   at every logged update -> SUPPORTED; otherwise FALSIFIED.
Descriptive (no prediction): generic encoder at all residuals; along geometry at r = 0.47 (mu < 0, outside Theorem 2).
No reruns or changes after results; all runs retained.
