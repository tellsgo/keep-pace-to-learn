# CIFAR-10 split-learning benchmark — SPEC (frozen before training)

## Purpose
Large-model check of the movable-antenna compensation results: (P1) exact restoration matches ideal-CSI training;
(P2) an uncompensated walking terminal is clearly worse; (P3) accuracy degrades with the residual.

## Setup
BS side: ResNet-18 (CIFAR 3x3 stem, no max-pool) -> avgpool 512 -> linear -> L=8 channel uses x Nt=4 complex symbols, total power L.
Channel: CDL-A ray field (23x20 rays), 3.5 GHz, terminal ULA Nr=8 and BS ULA Nt=4 (half-wavelength), motion along x; one channel per
minibatch (all 8 uses, all samples). Terminal moves 1 m/s, one update per 150 ms (1.75 lambda per update); field static for 100 updates, then
redrawn. Forward CN noise std 0.1; terminal MMSE combining with forward CSI; receiver MLP 64->256->10, cross-entropy.
Backward: per-sample normalized gradient sent over H_r^H with CN noise std 0.03, norm fed back. BS has no CSI. Ideal-CSI arm: exact H_f^H.
Training: SGD momentum 0.9, weight decay 5e-4, OneCycle max lr 0.05 (15% warm-up), batch 128, 40 epochs (390 updates/epoch), random crop + flip.
Test: 10,000 CIFAR test images, fresh random channel per 500-image block (fixed test seed), after every epoch.
Arms: ideal CSI; MA residual r/lambda in {0, 0.1, 0.2, 0.25, 0.3}; fixed antenna r = v*tau/lambda = 0.467. Seeds 710172100-102 (3), paired.

## Predictions (mean final test accuracy over 3 seeds)
P1: |acc(r=0) - acc(CSI)| <= 1.0 pp supported; > 3.0 pp falsified.
P2: acc(fixed) <= acc(r=0) - 5 pp, or epochs to 80% for fixed >= 3x those for r=0 (censored at 40): supported;
    |acc(fixed) - acc(r=0)| <= 1 pp and epochs ratio <= 1.5: falsified.
P3: Spearman(r, mean final acc) over the six residual arms (incl. fixed) <= -0.8 supported; >= 0 falsified.
No hyperparameter change, grid extension or seed change after any formal result. All runs retained.

## Amendment 1 (before any formal run)
Data source: the official Toronto tarball was replaced by a mirror in the historical study; CIFAR-10 was obtained from the HuggingFace mirror
(uoft-cs/cifar10 plain_text parquet; sha256 train 8428b53a..., test 841389e6...), decoded to data/cifar10
(50,000/10,000 images, 5,000/1,000 per class). Loader changed accordingly; a --epochs override is added for timing tests only
(formal runs use the frozen 40 epochs). No scientific setting changed.
