# Noto earthquake logistics: final V13 package

This repository contains only the final V13 code, input data, numerical results and figures. The manuscript and supporting material are not hosted here.

| Directory | Contents |
|---|---|
| `data/` | Original illustrative CSVs and derived scenario input snapshots |
| `code/` | Simulator, experiment drivers, figure generator and independent verifier |
| `result/` | Four V13 experiment archives, including per-realization outputs, summaries and SHA-256 manifests |
| `figs/` | Final PDF and PNG figures and their alignment reports |

The four result archives are `v13_structure_20261107` (20,224 ordinary evaluations and 49,152 Sobol evaluations), `v13_joint_20261107` (five parameter bundles, six comparison cells), `v13_20261107` (3,072 delayed-mission evaluations) and `v13_lp_20261107` (1,024 LP benchmark evaluations). Each archive has a manifest of repository-relative source, input and result hashes. Package versions are recorded in `result/v13_structure_20261107/metadata.json`.

## Verify the archive

From the repository root, using Python 3.10+:

```sh
python -m pip install -r requirements.txt
python code/verify_v13_results.py
```

The verifier checks archived file hashes, passing probabilities, threshold events, paired reference records across experiments and LP feasibility comparisons.

## Reproduce the outputs

These commands regenerate outputs in the checkout and can take several minutes. Run them in this order:

```sh
python code/start_v13_structure.py
python code/v13_experiments.py
python code/v13_joint/joint_check.py
python code/v13_joint/service_check.py
python code/v13_lp_benchmark.py
python code/v13_original_closure_trace.py
python code/verify_v13_results.py
```

The ordinary experiment driver is `code/run_v13_structure.py`. The figure stage uses saved records and clips negative error-bar lengths caused solely by floating-point roundoff below `1e-12`; confidence interval endpoints are unchanged.

## Result scope

The reference immediate-service delivery and outstanding-deprivation passing probabilities are 0.7734 and 0.9570. Changing the conditional loss driver from requests to executed throughput raises delivery passing probability by 0.2227, with conservative paired 95% interval [0.1499, 0.2867]. In the delayed task, the corresponding delivery probabilities are 0.8555 and 0.9844. The LP benchmark changes full-horizon delivery probability by 0.0039 [-0.0341, 0.0416] in the immediate task and 0.0117 [-0.0212, 0.0434] in the delayed task; neither interval excludes zero.

These estimates are conditional on the declared scenarios and service thresholds. The loss law is not independently calibrated with same-window field capacity and workload observations. The Sobol pilot does not establish a stable factor ranking. The LP is optimal for current-interval departures over its declared candidate paths, not for the full 72-hour mission. Initial destination stock and direct accessibility in the delayed task are explicit assumptions.
