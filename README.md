# Operational-Rule Sensitivity of Reliability in an Earthquake-Inspired Emergency Logistics Model

This repository contains the **final V13** input data, Python code, numerical results, and figures for the manuscript's *Numerical experiments* section. The manuscript and supporting material are not included. All paths below are relative to the repository root.

| Directory | Contents |
|---|---|
| `data/` | Original illustrative network and demand CSVs (`noto_data/`), plus the prepared input snapshots for the ordinary, joint, and delayed-mission experiments |
| `code/` | V13 simulators, experiment drivers, figure scripts, and result verification |
| `result/` | Four experiment-specific archives with per-realization outputs, summaries, and SHA-256 manifests |
| `figs/` | Final PDF and PNG figures and alignment reports |

## Python files and the experiments they support

| Python file | Role in the manuscript experiments | Result or figure location |
|---|---|---|
| `code/v13_core.py` | Core Noto network simulator used by the ordinary experiment and joint perturbation design. This is the final-path version of the model previously named `noto_endogenous_soft_capacity_caseV1.0.py`. | `result/v13_structure_20261107/`, `result/v13_joint_20261107/` |
| `code/run_v13_structure.py` | Main experiment driver: mechanism comparisons; operational and input controls; thresholds, event times, and trajectories; local sweeps and path pools; routing-by-failure representation; resolution and scheduling; conditional-reliability Sobol diagnostics. Its `main`, `sweeps`, `robust`, and `gsa` stages generate the corresponding records. | `result/v13_structure_20261107/` |
| `code/start_v13_structure.py` | Runs the ordinary stages and final figure generation in sequence with the archived V13 settings. | `result/v13_structure_20261107/`, `figs/v13_structure/` |
| `code/v13_joint/joint_check.py` | Five joint perturbation bundles, six paired comparison cells per bundle, and separately matched M3 calibration. | `result/v13_joint_20261107/` |
| `code/v13_joint/service_check.py` | Supplementary service-threshold, horizon, delay, and deprivation-curvature robustness checks. | `result/v13_joint_20261107/` |
| `code/v13_experiments.py` | Resource-accounted delayed-delivery mission, including initial destination stock, policy controls, retry-clock and routing-interval comparisons. It generates the delayed-task model from `v13_core.py`. | `result/v13_20261107/` |
| `code/v13_model.py` | Generated and archived delayed-task simulator used by `v13_experiments.py` and the LP benchmark. | `result/v13_20261107/`, `result/v13_lp_20261107/` |
| `code/v13_lp_benchmark.py` | Per-epoch maximum-departure linear-programming comparison against the same-task heuristic for immediate and delayed service. | `result/v13_lp_20261107/` |
| `code/v13_original_closure_trace.py` | Original-input WJ-SZ road-closure realization and component/request diagnostic ledger. | `result/v13_structure_20261107/` |
| `code/v13_finish_figures.py` | Regenerates publication figures from saved ordinary-experiment records and writes the ordinary-experiment manifest. | `figs/v13_structure/`, `result/v13_structure_20261107/` |
| `code/audit_panel_alignment.py` | Checks panel alignment in generated figures; it is a figure-quality utility, not a separate experiment. | `figs/v13_structure/` |
| `code/verify_v13_results.py` | Verifies archived hashes, probabilities, threshold events, cross-experiment reference records, and LP comparisons; it is a checker, not a simulation run. | All four `result/` subdirectories |

## The four `result/` subdirectories

| Subdirectory | Corresponding experiment | Main archived files |
|---|---|---|
| `result/v13_structure_20261107/` | Main 256-replicate ordinary experiments, one-factor and scheduling sweeps, robustness controls, 49,152 Sobol trajectories, and closure diagnostics. It covers the manuscript's mechanism, operational, threshold, sensitivity, routing/failure, and resolution subsections. There are 20,224 ordinary evaluations. | `main_replicates.csv`, `sweeps_replicates.csv`, `robust_replicates.csv`, `gsa_raw.npz`, `gsa_indices.csv`, `run_manifest.json` |
| `result/v13_joint_20261107/` | Direct paired comparisons at five joint parameter combinations and six cells (7,680 evaluations), plus 3,072 supplementary service-robustness evaluations. | `joint_replicates.csv`, `joint_contrasts.csv`, `M3_evaluation.csv`, `service_robustness_replicates.csv`, `run_manifest.json` |
| `result/v13_20261107/` | Separate delayed-delivery mission with 12 controls, 256 paired realizations per control, initial-stock resource accounting, and scheduling/retry-clock variants (3,072 evaluations). | `mission_replicates.csv`, `mission_contrasts.csv`, `mission_summary.csv`, trace `.npz` files, `manifest.json` |
| `result/v13_lp_20261107/` | Per-epoch LP versus heuristic comparison for the immediate and delayed tasks, with 256 paired realizations for each task/policy combination (1,024 evaluations). | `replicates.csv`, `epoch_gaps.csv`, `summary.csv`, `manifest.json` |

The `data/` scenario snapshots preserve the exact inputs used for each archive. Every result manifest records repository-relative SHA-256 hashes of its relevant code, inputs, and outputs. Software versions are recorded in `result/v13_structure_20261107/metadata.json`.

## Verify the saved package

From the repository root, with Python 3.10+ and the dependencies in `requirements.txt`:

```sh
python -m pip install -r requirements.txt
python code/verify_v13_results.py
```

## Reproduce the simulations and figures

The commands below replace generated files in the checkout and can take several minutes. Run them in this order:

```sh
python code/start_v13_structure.py
python code/v13_experiments.py
python code/v13_joint/joint_check.py
python code/v13_joint/service_check.py
python code/v13_lp_benchmark.py
python code/v13_original_closure_trace.py
python code/verify_v13_results.py
```

The ordinary driver uses `code/v13_core.py`; the delayed mission uses `code/v13_model.py`. Figure generation clips only negative error-bar lengths attributable to floating-point roundoff below `1e-12`; confidence-interval endpoints are unchanged.

## Interpretation of the archived results

The reference immediate-service delivery and outstanding-deprivation passing probabilities are 0.7734 and 0.9570. Switching the conditional loss driver from requests to executed throughput raises delivery passing probability by 0.2227, with conservative paired 95% interval [0.1499, 0.2867]. In the delayed task, the corresponding delivery probabilities are 0.8555 and 0.9844. The LP-minus-heuristic full-horizon delivery differences are 0.0039 [-0.0341, 0.0416] for immediate service and 0.0117 [-0.0212, 0.0434] for the delayed task; neither interval excludes zero.

These results are conditional on the declared scenarios and service thresholds. The loss law is not independently calibrated with same-window field capacity/workload observations. The Sobol pilot does not establish a stable factor ranking. The LP optimizes current-interval departures over its declared candidate paths, not the full 72-hour mission. Initial destination stock and direct accessibility in the delayed task are explicit assumptions.
