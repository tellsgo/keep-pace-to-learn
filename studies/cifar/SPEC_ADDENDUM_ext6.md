# Extension 6 — antenna selection at the speeds where the mechanical actuator is out of reach; frozen before running (2026-10-17)
Answers an independent blind review (v22): at 1.82 and 4.07 m/s the uncompensated terminal fails, but exact mechanical restoration
there exceeds the example actuator (cycle 27.3 ms vs 18.75 ms; 4.07 m/s > V). Discrete-position selection (Proposition 4 of the
draft) has no actuator limit: elements at spacing lambda/2 along the motion, backward element = nearest to the forward world position.
v*tau/lambda = 0.85 -> j* = 2 -> signed residual -0.15 lambda; v*tau/lambda = 1.90 -> j* = 4 -> residual -0.10 lambda.
Code/config unchanged (src/run.py), seeds 710172100-102, speed-consistent trajectories as in Extension 2 (advance v * 0.15 s per update).
Runs: v = 1.8202, Nr = 8, --res -0.15; v = 4.0686, Nr = 8, --res -0.10; v = 1.8202, Nr = 2, --res -0.15.
References (Extension 2, same speeds and trajectories, not rerun): exact restoration 93.8-93.9%; fixed antenna 58.8% / 16.1% (Nr=8),
53.2% (Nr=2, 1.82 m/s).
Prediction Z (mean final accuracy, 3 seeds, each configuration): selection within 2 pp of exact restoration at the same speed for all
three -> SUPPORTED; any configuration more than 10 pp below exact restoration -> FALSIFIED; otherwise NOT MET.
No reruns or changes after results; all runs retained.
