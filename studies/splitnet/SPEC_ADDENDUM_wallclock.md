# Addendum (frozen before running): wall-clock re-simulation with period-consistent trajectories
Answers an independent strict review (round 7, M2). Same network, data, seeds (8), eta, noise and evaluation as the formal digits study.
Change: the exchange period is the anchor-and-return cycle T(u_h) of Prop. 4 (v=1 m/s, tau=40 ms, V=2.5 m/s, t_g=0), or tau (=40 ms)
for the fixed antenna; the terminal advances v*T per update; the ray field is redrawn every 15 s of real time (not every 100 updates).
Configurations: a=50 m/s^2: r in {0, 0.15, 0.20, 0.30} lambda and fixed; a=10 m/s^2: r in {0, 0.15, 0.20} lambda (fixed shared).
Metric: wall-clock to 85% = K85 (median over 8 seeds, eval every 25 updates, censored at 1500) x T.
Prediction W: for both a=50 and a=10, min over r in {0.15, 0.20} of wall-clock < 0.9 x wall-clock(r=0) -> supported;
partial >= exact for both -> falsified; otherwise not met. No changes after results.
