# Extension 3 — frozen before running (2026-10-17); answers independent reviews (round 8 M2; blind review on Nr<Nt)
Same code as before; --nt override added (encoder output and receiver input scale with Nt). Seeds 710172100-102.
(a) Ideal-CSI arms at the high speeds of Extension 2 (trajectory step v*0.15 s): Nr=8 at 1.8202 and 4.0686 m/s; Nr=2 at 1.8202 m/s.
(b) Non-overloaded 2x2 link (Nt=Nr=2): at 1 m/s CSI, MA r=0, fixed (0.467 lambda); at 1.8202 m/s MA r=0 and fixed (0.85 lambda).
Predictions (mean final accuracy, 3 seeds):
Y1: |acc(MA) - acc(CSI)| <= 1 pp at each of the three high-speed configurations (MA from Extension 2) -> supported; > 3 pp at any -> falsified.
Y2: 2x2 at 1 m/s: |acc(MA) - acc(CSI)| <= 1 pp -> supported; > 3 pp falsified.
Y3: 2x2 at 1.82 m/s: acc(fixed) <= acc(MA) - 20 pp -> supported; within 3 pp -> falsified.
Also descriptive: 2x2 fixed at 1 m/s versus MA. Spectrum for the CIFAR seeds' own geometries is computed by src/spectrum.py (training-free).
No changes after results.
