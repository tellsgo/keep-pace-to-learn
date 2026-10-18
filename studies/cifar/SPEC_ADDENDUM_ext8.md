# Extension 8 — period-consistent common-budget comparison; frozen before running (2026-10-18)
Answers independent reviews of v26: the wall-clock comparison of Appendix J re-timed curves recorded with a fixed advance per update
(18.75 ms at 28 GHz) instead of regenerating trajectories at each realization's exchange period.
Code: src/run_ext8.py = src/run.py plus a --cycle option that sets the trajectory advance per update (diff in FREEZE file).
Source units: 3.5 GHz, tau = 40 ms (28 GHz equivalents in parentheses: times / 8). Periods per update:
  fixed antenna and selection: cycle = 0.04 s (5 ms);  mechanical exact restoration (a = 50 m/s^2 at 3.5 GHz, i.e. 400 at 28 GHz,
  V = 2.5 m/s): cycle = 0.14928 s (18.66 ms) at 1 m/s and 0.2184 s (27.30 ms) at 1.8202 m/s; unreachable at 4.0686 m/s (not run).
Selection residuals as in Extensions 6-7: -0.033 (1 m/s), -0.15 (1.8202 m/s), -0.10 (4.0686 m/s) lambda. Fixed antenna: r = v*tau.
Nr = 8, seeds 710172100-102, 40 epochs, scene redrawn every 100 updates (unchanged; the scene dwell time therefore scales with the
period). Wall-clock time to a target = (first update at which the seed-mean test accuracy, evaluated each epoch, reaches the target)
x period; the 1 us switching time of selection and all computation/pilot/feedback times are not added.
Prediction Z8: at each of the three speeds, selection reaches 90% at a wall-clock time no later than every other arm run at that speed
-> SUPPORTED; at any speed selection's time exceeds 1.2x the fastest other arm -> FALSIFIED; otherwise NOT MET.
No reruns or changes after results; all runs retained.
