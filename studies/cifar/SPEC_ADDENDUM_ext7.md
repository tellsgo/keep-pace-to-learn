# Extension 7 — antenna selection at walking speed (1 m/s); frozen before running (2026-10-18)
Fills the 1 m/s selection cell of a common-budget comparison. v*tau/lambda = 0.4670 -> nearest half-wavelength element j* = 1 ->
signed residual -0.033 lambda. Code/config unchanged (src/run.py), seeds 710172100-102, the formal 1 m/s trajectory.
Runs: Nr = 8 and Nr = 2, --res -0.033.
References (formal study and Extension 1, not rerun): exact restoration 93.81% (Nr=8), 93.77% (Nr=2); fixed antenna 92.95% (Nr=8),
85.5% (Nr=2).
Prediction W (mean final accuracy, 3 seeds): selection within 0.5 pp of exact restoration for both Nr -> SUPPORTED; either more than
3 pp below exact restoration -> FALSIFIED; otherwise NOT MET. No reruns or changes after results; all runs retained.
