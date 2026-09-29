# src/eval

Aggregate jet anomaly scores into event scores, compute detection metrics, and fuse model outputs.

| File | Purpose |
| --- | --- |
| [scoring.py](scoring.py) | Compute reconstruction or latent-cycle scores and aggregate by sum, mean, max, min, or pT weighting |
| [metrics.py](metrics.py) | Compute AUC, SIC, signal efficiency at fixed background efficiency, and classification metrics |
| [gated_scoring.py](gated_scoring.py) | Calibrate structure-based gates and fuse percentile scores from strong-attraction and baseline models |

The labeled-dataset and cycle evaluation scripts read the global-cycle representation
from the checkpoint (legacy checkpoints default to pooled latents).
`scripts.eval_cycle_suite` summarizes only results computed in the current invocation.
`--only_bb1` therefore produces a BB1-only summary; other datasets' saved files are untouched.
