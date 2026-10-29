# Keep Pace to Learn: Movable-Antenna Compensation of Channel Aging in Over-the-Air Split Learning

**Xianzhong Li, Yuzhi Yang, Tianhao Guo, Zhaoyang Zhang**

- Xianzhong Li and Tianhao Guo: College of Physics and Electronic Engineering, Shanxi University.
- Yuzhi Yang: Japan Advanced Institute of Science and Technology.
- Zhaoyang Zhang: College of Information Science and Electronic Engineering, Zhejiang University.

Paper: [arXiv:2610.23118](https://arxiv.org/abs/2610.23118), extended version (v2, Appendices C–K) of a manuscript submitted to IEEE JSAC.

Research code and compact results for movable-antenna compensation of channel aging in over-the-air split learning. The studies examine residual displacement, convergence, positioning errors, path-mode precoding, and learning on digits and CIFAR-10. Paper: [arXiv:2610.23118](https://arxiv.org/abs/2610.23118). Numerical support is configuration-specific; negative and inconclusive results are retained.

## Study map

Section names refer to the numerical evaluation of the paper.

| Directory | Paper section / result | Available material |
|---|---|---|
| studies/cifar | CIFAR-10 experiments: residual displacement and receiver dependence, array tolerance and eigenspace intervention, elapsed training time (time-to-90% table); fig_cifar.py, fig_csifree.py, fig_budget_ext8.py | CUDA source, configuration, compact results, protocols and verdict |
| studies/residual | Accuracy follows the real correlation (residual-law figure); coverage map via fig_regime.py; fig_residual.py | Partial source, compact results, protocol and verdict; original continuous-spectrum configuration missing |
| studies/benchmark_refinement | Comparison with digital alternatives on CDL-A/C at 3 and 30 km/h; fig_bench.py | Results, protocol and verdict; training source absent |
| studies/position_bias | Systematic offset versus random jitter; fig_bias.py | Compact results and verdict |
| studies/position_rigid, studies/position_rep | Rigid common displacement and replication for the same section | Compact results and verdict |
| studies/pathmode | Path-mode precoding study (arXiv v2, Appendix H) | Source, original and revised results, protocol and revision notes |
| studies/convergence | Time to target versus the real part of the correlation (arXiv v2, Appendix E); fig_conv.py | Source, calibration, curves, protocol and verdict |
| studies/splitnet | Split network on a digit dataset and partial-compensation timing (arXiv v2, Appendix F), beam-resolved aging (Appendix G); fig_digits.py, fig_beams.py | Source, configuration, compact results, protocol and verdict |
| studies/budget | Earlier post-hoc timing, superseded by the period-consistent CIFAR-10 comparison | Generator and CSV |

## Environment and quick reproduction

Fresh release checks used Python 3.9.6 with the exact versions in requirements.txt on a CPU host. Python 3.10+ is recommended for a fresh environment; this newer interpreter combination has not been tested here. Install a suitable CUDA-enabled PyTorch build for CIFAR training. Figures and NumPy/SciPy studies do not need a GPU; pathmode uses PyTorch on CPU. CIFAR training hard-codes CUDA and received syntax checks only.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -B figures/fig_conv.py
python3 -B figures/fig_digits.py
python3 -B figures/fig_bias.py
python3 -B figures/fig_budget_ext8.py
python3 -B checks/smoke_cpu.py
```

Run commands from the repository root. Figure inputs resolve from the script location, not the shell's working directory; images go to figures/generated. All 12 figure scripts were run; their outputs are summarized in checks/release_outputs/.md. The residual figure uses original discrete theoretical points because its original continuous-spectrum configuration is missing. It is not an exact redraw of the original continuous curve or RMS marker.

For a standalone short path-mode run:

```sh
mkdir -p checks/release_outputs
python3 -B studies/pathmode/src/pm_rev1.py --geom sep --enc path --res 0.2 --seed 1 --steps 20 --every 10 --out checks/release_outputs/pathmode_smoke.json
```

For the full path-mode grid, launch_rev1.py uses 12000 updates and five seeds (90 jobs); existing result files are skipped. To rerun the entire grid, use a separate copy of the repository and move its results_rev1 directory aside before launching. The original pm.py uses full-dataset preprocessing; the preferred pm_rev1.py splits first and fits training statistics only. Original and revision results are kept distinct.

Convergence/splitnet formal.py use the original calibrated step and full frozen grid. These drivers overwrite their study results; run them in a disposable copy when retraining. Their analysis scripts consume rows.json; curve arrays are JSON with dtype metadata (large curve collections split into small files). The original array precision and all seed rows are retained.

CIFAR single configuration after preparing data:

```sh
mkdir -p studies/cifar/results/reproduced
python3 studies/cifar/src/run.py --arm phys --res 0.0 --seed 710172100 --data data/cifar10 --out studies/cifar/results/reproduced/phys_0.json
```

Use each extension's SPEC and launcher to reproduce its exact arm/grid. Launchers assume two CUDA devices (0 and 1), skip existing files and use per-study logs. Change device scheduling to match available GPUs without changing scientific parameters. Complete training was not rerun for this release. Residual, position-error and benchmark-refinement studies lack necessary original inputs; their original experiments cannot be fully reproduced from this package.

Budget postprocessing: `python3 studies/budget/src/generate_budget2.py` regenerates small tables in studies/budget/results and images in figures/generated. Its historical timing assumes transfer of frozen curves to rescaled cycles. Extension 8 instead regenerates trajectories consistently with the cycle; the two analyses should not be conflated.

## Figure → command

Run these commands from the repository root after installing the dependencies. Outputs are written to `figures/generated/`.

| Figure output | Command |
|---|---|
| `beams.pdf`, `beams.png` | `python3 -B figures/fig_beams.py` |
| `benchmark.pdf`, `benchmark.png` | `python3 -B figures/fig_bench.py` |
| `bias_jitter.pdf`, `bias_jitter.png` | `python3 -B figures/fig_bias.py` |
| `budget_ext8.pdf`, `budget_ext8.png` | `python3 -B figures/fig_budget_ext8.py` |
| `cifar.pdf`, `cifar.png` | `python3 -B figures/fig_cifar.py` |
| `convergence.pdf`, `convergence.png` | `python3 -B figures/fig_conv.py` |
| `csifree.pdf`, `csifree.png` | `python3 -B figures/fig_csifree.py` |
| `digits.pdf`, `digits.png` | `python3 -B figures/fig_digits.py` |
| `protocol.pdf`, `protocol.png` | `python3 -B figures/fig_protocol.py` |
| `regime.pdf` | `python3 -B figures/fig_regime.py` |
| `residual_law.pdf`, `residual_law.png` | `python3 -B figures/fig_residual.py` |
| `fig1.pdf`, `fig1.png` | `python3 -B figures/fig_system.py` |

## Runtime planning

Theory checks and individual figures usually take seconds to a few minutes on CPU; the 20-step path-mode smoke takes seconds. Full path-mode 90-run and digits/convergence grids may take tens of minutes to several hours on CPU. CIFAR has 15600 updates per 40-epoch arm; budget tens of minutes to several hours per arm on a capable CUDA GPU and substantially longer for all extensions. These full-training ranges are planning estimates, not measured release benchmarks; hardware, threading and downloads affect them. Fresh smoke results are recorded separately from original historical outputs.

## Data

Digits is bundled with scikit-learn: `from sklearn.datasets import load_digits; data = load_digits()`. No dataset download is needed after installing scikit-learn. Preserve the documented train/test split and train-only normalization in the revised path-mode implementation.

CIFAR-10 is not bundled. The optional `python3 checks/prepare_cifar.py` downloads the official Toronto Python archive, checks its published archive digest, and creates data/cifar10/cifar10.npz with train_x/test_x uint8 NCHW arrays and train_y/test_y labels. No dataset download was performed for this release. The original historical study used a mirror of CIFAR-10; the helper's output has the expected schema but byte-for-byte equivalence to that mirror was not checked. Dataset terms remain with the dataset provider. NumPy's archive is an external dataset/cache, excluded from version control.

The small CDL-A numerical tables in configs are retained from the original material and attributed to 3GPP TR 38.901. They are public-standard parameters, not measured personal data; the full standard is not bundled. Do not infer full hardware/standard conformance from this simplified scalar-polarized implementation. Code is MIT licensed; third-party standards and datasets retain their own terms.

## Evidence

provenance/sources.json maps original files to their release destinations and original SHA256 digests; provenance/excluded.json lists omissions. Historical FREEZE_SHA256 files refer to original sources, not rewritten public files. Scientific protocol text was translated/condensed where necessary, with hypotheses, thresholds, failures and limits preserved. No claims about missing artifacts are promoted to fresh verification.

## Companion papers

- [Wait or Rotate: Statistical Repair of the Backward Pass in Over-the-Air Split Learning Under Channel Aging](https://arxiv.org/abs/2610.23137)
- [Wait to Learn: Timing the Gradient Return in Over-the-Air Split Learning over an Aging Single-Antenna Channel](https://arxiv.org/abs/2610.23135)

## Citation

```bibtex
@misc{li2026keeppace,
  title={Keep Pace to Learn: Movable-Antenna Compensation of Channel Aging in Over-the-Air Split Learning},
  author={Xianzhong Li and Yuzhi Yang and Tianhao Guo and Zhaoyang Zhang},
  year={2026},
  eprint={2610.23118},
  archivePrefix={arXiv},
  primaryClass={eess.SP},
  url={https://arxiv.org/abs/2610.23118}
}
@misc{li2026waitorrotate,
  title={Wait or Rotate: Statistical Repair of the Backward Pass in Over-the-Air Split Learning Under Channel Aging},
  author={Xianzhong Li and Yuzhi Yang and Tianhao Guo},
  year={2026},
  eprint={2610.23137},
  archivePrefix={arXiv},
  primaryClass={eess.SP},
  url={https://arxiv.org/abs/2610.23137}
}
@misc{li2026waittolearn,
  title={Wait to Learn: Timing the Gradient Return in Over-the-Air Split Learning over an Aging Single-Antenna Channel},
  author={Xianzhong Li and Yuzhi Yang and Tianhao Guo},
  year={2026},
  eprint={2610.23135},
  archivePrefix={arXiv},
  primaryClass={eess.SP},
  url={https://arxiv.org/abs/2610.23135}
}
```
