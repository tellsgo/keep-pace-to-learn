# Extension 4 — CSI-free terminal (frozen before formal runs, 2026-10-17)
Motivation: an independent blind review argued that a terminal with forward CSI could compute the exact gradient digitally. Here the
terminal has NO channel knowledge: its receiver-side model sees only log(1+|y|^2) per antenna and channel use (EnergyReceiver,
L*Nr -> 256 -> 256 -> 10). Everything else as the formal study. A non-formal pilot (seed 1, 15 epochs, Nr=8) gave 90.45% (ideal backward)
and 8.72% (fixed antenna at 1 m/s); it is reported as a pilot only.
Arms (seeds 710172100-102, 40 epochs): Nt=4,Nr=8: ideal backward (csi), MA r in {0,0.1,0.2,0.25,0.3}, fixed (0.467 lambda);
Nt=Nr=2: csi, MA r=0, fixed.
Predictions (mean final accuracy):
Z1: |MA(r=0) - csi| <= 1 pp for both links -> supported; > 3 pp for either -> falsified.
Z2: fixed <= MA(r=0) - 30 pp for both links -> supported; within 3 pp for both -> falsified.
Z3: Spearman(Re phi_m(r) of CDL-A, mean acc) over r in {0,0.1,0.2,0.25,0.3,0.467} (Nr=8) >= 0.8 -> supported; <= 0 -> falsified.
    Re phi_m values from the training-free CDL-A spectrum (conv-rate theory: 1, .893, .603, .412, .209; 0.467 -> negative).
No changes after results.
