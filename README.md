# Noto earthquake logistics: V13 data, code and results

This repository contains the inputs, simulation code, per-realization outputs, figures and run manifests used for the V13 analysis. **The manuscript and its supplementary document are not published here.** The final V13 code is under `data/` and `review_work/`.

| Location | Contents |
|---|---|
| `data/noto_data/` | Original illustrative network and demand CSV inputs |
| `data/v13_structure_20261107/` | Ordinary experiments: 20,224 evaluations, paired results, trajectories, calibration and 49,152 Sobol evaluations |
| `data/v13_joint_20261107/` | Five parameter combinations and six cells, 256 paired evaluations per cell |
| `data/v13_20261107/` | Separate resource-accounted delayed task with 12 controls and 256 paired evaluations each |
| `data/v13_lp_20261107/` | Per-epoch maximum-departure linear-programming benchmark with 256 paired realizations for each of two tasks |
| `figs/v13_structure/` | Plots generated from archived records |
| `review_work/` | Generating scripts and independent results verifier |

Each data directory includes a SHA-256 manifest. Paths in the manifests are relative to the repository root. `requirements.txt` lists Python dependencies; specific versions used for the archive are in `data/v13_structure_20261107/metadata.json`.

## Verify the saved results

From the repository root, with Python 3.10+ and the listed dependencies:

```sh
python -m pip install -r requirements.txt
python review_work/verify_v13_results.py
```

The verifier checks every archived source/input/result hash, reconstructed passing probabilities, event indicators, matching reference records across experiments, and the LP comparison against its heuristic controls. No manuscript files are needed for this check.

## Reproduce the simulations and figures

These commands replace generated outputs in the checkout and may take several minutes. Run in this order:

```sh
python review_work/start_v13_structure.py
python review_work/v13_experiments.py
python review_work/v13_joint/joint_check.py
python review_work/v13_joint/service_check.py
python review_work/v13_lp_benchmark.py
python review_work/v13_original_closure_trace.py
python review_work/verify_v13_results.py
```

The V13 ordinary driver is `data/run_v13_structure.py`. The last stage of `start_v13_structure.py` generates the figures from saved records using `v13_finish_figures.py`, clipping only negative error-bar lengths attributable to floating-point roundoff below `1e-12` while leaving confidence interval endpoints unchanged.

## Result scope

The reference immediate-service delivery and currently outstanding deprivation passing probabilities are 0.7734 and 0.9570. Changing the conditional loss driver from requests to executed throughput raises delivery passing probability by 0.2227, with conservative paired 95% interval [0.1499, 0.2867]. In the separate delayed task, the corresponding delivery probabilities are 0.8555 and 0.9844. The LP benchmark changes full-horizon delivery probability by 0.0039 [-0.0341, 0.0416] in the immediate task and 0.0117 [-0.0212, 0.0434] in the delayed task; neither interval excludes zero.

These estimates are conditional on declared scenarios and service thresholds. The loss law is not independently calibrated with same-window field capacity/workload observations. The Sobol pilot does not establish a stable factor ranking. The LP is optimal for the current interval's departures over its declared candidate paths, not for the full 72-hour mission. Initial destination stock and direct accessibility in the delayed task are explicit assumptions. No manuscript, supplementary document or DOI is claimed to be hosted in this repository.
