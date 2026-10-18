# Extension 1 — frozen before running (2026-10-17)
Motivation: formal P2 was falsified (8x4 terminal, fixed antenna at 0.47 lambda only 0.86 pp worse). The array analysis predicts that
learning without compensation degrades with the largest eigenvalue of Herm(Bbar_r) (MMSE combiner) and fails when no positive
eigenvalue remains. Training-free table (Monte Carlo, 300 fields x 8 geometries; eigenvalues normalized by tr Bbar_0/4):
Nr=2: r=0.25: n+=4.00, lmax=+0.812 | 0.47: 1.00, +0.318 | 0.60: 1.00, +0.079 | 0.85: 0.25, -0.025
Nr=8: r=0.47: 1.00, +0.578 (0.45) | 0.85: 1.00, +0.131 | 1.90: 1.00, +0.129
Everything as the formal study except --nr. Seeds 710172100-102. Arms: Nr=2: CSI, r in {0,0.25,0.47,0.6,0.85}; Nr=8: r in {0.85,1.9}.
Predictions (mean final accuracy over 3 seeds):
E1: Nr=2, |acc(r=0)-acc(CSI)| <= 1 pp supported; > 3 pp falsified.
E2: Nr=2, acc(0.85) <= acc(0) - 20 pp supported; acc(0.85) >= acc(0) - 3 pp falsified.
E3: Nr=2, Spearman(lmax, mean acc) over r in {0,0.25,0.47,0.6,0.85} (lmax(0)=1) >= 0.8 supported; <= 0 falsified.
E4: Nr=8, acc(0.85) and acc(1.9) both <= acc(0.47 formal fixed, 92.95%) - 1 pp supported; both >= 92.95% falsified.
No changes after results; all runs retained.
