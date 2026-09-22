# src

Data processing, modeling, training, and evaluation for jet-graph anomaly detection.

| Module | Purpose |
| --- | --- |
| [data](data/README.md) | Build jet graphs from LHCO events and store and load datasets |
| [models](models/README.md) | Construct graph autoencoders and compute reconstructions and anomaly scores |
| [models/glad](models/glad/README.md) | Provide graph-level one-class detectors and teacher-student models |
| [objectives](objectives/README.md) | Compute topology regularization, latent-cycle, and contrastive objectives |
| [training](training/README.md) | Split data, run training, evaluate models, and save experiment results |
| [eval](eval/README.md) | Aggregate event scores, compute metrics, and fuse model scores |
| [checkpoint.py](checkpoint.py) | Read and write model specifications, parameters, and resumable training state |
