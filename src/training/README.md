# src/training

Coordinate graph-autoencoder training, validation, checkpoint resumption, and result storage.

| File | Purpose |
| --- | --- |
| [__main__.py](__main__.py) | Run the training CLI with `python -m src.training` |
| [trainer.py](trainer.py) | Build training components and run optimization, learning-rate scheduling, early stopping, and resumption |
| [options.py](options.py) | Define training CLI arguments and validate objective settings |
| [splits.py](splits.py) | Build event-level training, validation, monitoring, and test splits and read fixed split manifests |
| [evaluation.py](evaluation.py) | Compute epoch validation losses and metrics, evaluate final checkpoints, and save scores |
| [artifacts.py](artifacts.py) | Save run configurations, split fingerprints, and training state, and restore checkpoints |
