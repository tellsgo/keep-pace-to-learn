# Revision rev1 (2026-10-17) — training-only preprocessing
The frozen study (SPEC.md, results/) standardized pixels with statistics of all 1797 images before the train/test split, so test
inputs (not labels) influenced preprocessing. rev1 reruns the full grid with the split first and training-set statistics only
(src/pm_rev1.py; diff = the two preprocessing lines). Same seeds, grid, and predictions Q1-Q3 as SPEC.md; this is a revised
re-check, not the pre-registered run. Original results are kept unchanged in results/.

## Result (90 runs, results_rev1/analysis.json)
Q1 SUPPORTED (path-mode, geometry S: 93.50% at r = 0; 93.60-94.31% at every certified residual up to 0.47 lambda).
Q2 SUPPORTED (T90 = 500 at every residual). Q3 SUPPORTED (alignment >= mu - 0.01 at every logged update).
Unstructured encoder 93.0-94.3% (geometry S); geometry P at 0.47 lambda: 87.8% for both encoders.
All accuracies within 0.4 pp of the frozen run; conclusions unchanged. The paper reports rev1.
