# GSoC 2026 GENIE Graph Anomaly Detection

Graph autoencoder pipeline for LHCO-style anomaly detection.

The code converts collider events into subjet graphs, trains a graph autoencoder on background-dominated data, and reports event-level anomaly metrics.

## Repository layout

```text
src/       data processing, models, training, evaluation
scripts/   runnable pipeline / training / inference entrypoints
dataset/   local data placeholder; data are not tracked
runs/      local output placeholder; checkpoints are not tracked
```

## Setup

```bash
conda create -n genie python=3.12
conda activate genie
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

Expected LHCO input:

```text
dataset/lhco/events_anomalydetection.h5
```

Optional Kitchen Sink zero-shot inputs:

```text
dataset/kitchensink/<MODEL>/events.h5
```

## Default LHCO pipeline

Default setting:

```text
anti-kT R=1.0 jets, pT > 1.2 TeV
30 exclusive-kT subjets per jet
unique-6 graph construction
log edge features: ln ΔR, ln kT, ln z
EdgeGraphAE
AdamW(lr=0.003, weight_decay=0.01) + linear OneCycleLR
event anomaly score = sum of selected-jet reconstruction errors
```

Run:

```bash
PYTHON=$CONDA_PREFIX/bin/python bash scripts/pipeline_relae.sh
```

Common variants:

```bash
K=8 PYTHON=$CONDA_PREFIX/bin/python bash scripts/pipeline_relae.sh
EDGE_PT_SCALE=raw PYTHON=$CONDA_PREFIX/bin/python bash scripts/pipeline_relae.sh
EVENT_SCORE_AGG=max SKIP_DATA=1 PYTHON=$CONDA_PREFIX/bin/python bash scripts/pipeline_relae.sh
```

## Zero-shot inference

```bash
PYTHON=$CONDA_PREFIX/bin/python \
  bash scripts/zeroshot_kitchensink.sh runs/relae/<timestamp>/last.pt
```

Example with matched graph settings:

```bash
K=8 EDGE_PT_SCALE=raw EVENT_SCORE_AGG=sum \
  PYTHON=$CONDA_PREFIX/bin/python \
  bash scripts/zeroshot_kitchensink.sh runs/relae_unique8_rawpt/<timestamp>/last.pt
```

## Notes

- Metrics are computed at event level through `event_id`.
- Supported event score aggregations: `sum`, `mean`, `max`, `min`, `pt_weighted`.
- Data files, processed shards, checkpoints, and run outputs are intentionally not tracked.
