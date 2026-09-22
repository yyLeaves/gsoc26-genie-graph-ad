# src/eval

Aggregate jet anomaly scores into event scores, compute detection metrics, and fuse model outputs.

| File | Purpose |
| --- | --- |
| [scoring.py](scoring.py) | Compute reconstruction or latent-cycle scores and aggregate by sum, mean, max, min, or pT weighting |
| [metrics.py](metrics.py) | Compute AUC, SIC, signal efficiency at fixed background efficiency, and classification metrics |
| [gated_scoring.py](gated_scoring.py) | Calibrate structure-based gates and fuse percentile scores from strong-attraction and baseline models |
