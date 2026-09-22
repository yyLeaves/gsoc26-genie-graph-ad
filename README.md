# GSoC 2026 GENIE Graph Anomaly Detection

Graph autoencoder pipeline for LHCO-style anomaly detection.

The code converts collider events into subjet graphs, trains a graph autoencoder on background-dominated data, and reports event-level anomaly metrics.

## Repository layout

```text
src/       data processing, models, objectives, training, evaluation
scripts/   runnable pipeline / training / inference entrypoints
dataset/   local data placeholder; data are not tracked
runs/      local output placeholder; checkpoints are not tracked
```

See [Module overview](src/README.md) for package responsibilities.

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

Optional Kitchen Sink inputs (for transfer / zero-shot style eval scripts):

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
EdgeGraphAE (edge_graph + edgeconv)
AdamW(lr=0.003, weight_decay=0.01) + linear OneCycleLR
50 epochs, primary checkpoint = last.pt (unsupervised; --no_early_stop)
event anomaly score = sum of selected-jet reconstruction errors
```

Stages (each can be run alone):

```text
raw H5
  → 1. preprocess   scripts/preprocess.sh
  → 2. build graph  scripts/build_graph.sh
  → 3. event split  scripts/make_event_split.py
  → 4. train        python -m src.training
  → 5. eval         python -m scripts.eval_test_idx
                    (or scripts.eval_labeled_dataset for BB1 / labeled sets)
```

### 1. Preprocess (jets)

Raw HDF5 → selected-jet constituent shards (graph-agnostic).

Important env vars (`scripts/preprocess.sh`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `H5_PATH` | `dataset/lhco/events_anomalydetection.h5` | Input events |
| `LABELS_PATH` | empty | Optional truth/masterkey for unlabeled BB files |
| `OUT_DIR` | `…/lhco_canonical_leadingpt_jets` | Output jet shards |
| `MIN_JET_PT` | `1200` | Leading-jet pT cut (GeV) |
| `JET_SELECTION` | `leading_pt` | Which jets are kept |
| `REQUIRE_TWO_JETS` | `1` | Require two selected jets per event |
| `FEATURES` | `log_phys` | Constituent feature mode |
| `FORCE` | `0` | `1` = delete and rebuild `OUT_DIR` |

```bash
PYTHON=$CONDA_PREFIX/bin/python bash scripts/preprocess.sh
# OUT_DIR=dataset/processed/lhco_canonical_leadingpt_jets
```

### 2. Build graph

Jets → subjet graphs (**unique-6** + log edge features by default).

Important env vars (`scripts/build_graph.sh`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `CANONICAL_DIR` / `INPUT_DIR` | jet preprocess dir | Input |
| `OUT_DIR` | auto `…_sj30_unique6_logef` | Output graph shards |
| `N_SUBJETS` | `30` | Exclusive-kT subjets; empty = graph on constituents |
| `STRATEGY` | `unique` | Graph type (see below) |
| `K` | `6` | Used by `knn` / `sym_knn` / `radius_knn` / `unique` |
| `EDGE_FEATURES` | `log` | `none` / `linear` / `log` |
| `EDGE_PT_SCALE` | `normalized` | `normalized` or `raw` |
| `FORCE` | `0` | `1` = rebuild |

Supported `STRATEGY` values:

| Strategy | Needs | Meaning |
|----------|-------|---------|
| `unique` | `K` | Unique-k rigid graph in pT order (default / mainline) |
| `knn` | `K` | Directed kNN in (η, φ) |
| `sym_knn` | `K` | Symmetrized kNN |
| `radius` | `--radius` | All pairs with ΔR < r |
| `radius_knn` | `K` + `--radius` | Radius graph, capped at k nearest |
| `mst` | — | Angular MST |
| `delaunay` | — | Delaunay triangulation in (η, φ) |
| `laman` | — | Henneberg Laman graph (2N−3 edges) |
| `fully_connected` | — | All ordered pairs |

`scripts/build_graph.sh` currently forwards `STRATEGY` / `K` / edge feature
flags. For `radius` / `radius_knn`, call the Python entry directly and pass
`--radius`:

```bash
PYTHONPATH=. $CONDA_PREFIX/bin/python -m src.data.build_graph \
  --input_dir dataset/processed/lhco_canonical_leadingpt_sj30_jets \
  --output_dir dataset/processed/lhco_canonical_leadingpt_sj30_radius_logef \
  --strategy radius --radius 0.3 \
  --edge_features log --edge_pt_scale normalized
```

```bash
PYTHON=$CONDA_PREFIX/bin/python bash scripts/build_graph.sh
# OUT_DIR=dataset/processed/lhco_canonical_leadingpt_sj30_unique6_logef
```

### 3. Event split

Event-id manifest for train / val / monitor / test (reusable across graph
variants that share the same events).

Important CLI flags (`scripts/make_event_split.py`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--data_dir` | required | Jets (or graph) dir with `event_id` |
| `--output` | required | Manifest `.npz` path |
| `--seed` | `42` | Shuffle seed |
| `--train_bkg_events` | `80000` | Pure (or base) train background |
| `--train_s_over_b` / `--train_sig_events` | `0` / unset | Train signal contamination |
| `--val_bkg_events` | `20000` | Val background |
| `--monitor_sig_events` | `0` | Disjoint signal for epoch AUC log only |
| `--test_bkg_events` / `--test_sig_events` | `340000` / `20000` | Held-out test |

```bash
PYTHONPATH=. $CONDA_PREFIX/bin/python -u scripts/make_event_split.py \
  --data_dir dataset/processed/lhco_canonical_leadingpt_sj30_jets \
  --output dataset/processed/splits/lhco_canonical_leadingpt_sj30_train80000b_val20000b_test340000b_20000s_monsig20000_seed42.npz \
  --seed 42 \
  --train_bkg_events 80000 --val_bkg_events 20000 \
  --monitor_sig_events 20000 \
  --test_bkg_events 340000 --test_sig_events 20000
```

`--data_dir` can be the subjet jet dir or any graph dir that shares the same
`event_id`s (needs current `metadata.pt` with `schema_version`). The raw
constituent jet dir from an older preprocess may need a rebuild (`FORCE=1`)
before `JetDataset` will load it.

Or run steps 1–3 together. `build_ds.sh` defaults to `MONITOR_SIG_EVENTS=0`;
pass `20000` to match the manifest above. Useful toggles:
`RUN_PREPROCESS` / `RUN_GRAPH` / `RUN_SPLIT`, plus the same size vars
(`TRAIN_BKG_EVENTS`, `TRAIN_S_OVER_B`, …).

If an existing `CANONICAL_JETS` dir has legacy `metadata.pt` (no
`schema_version`), either rebuild with `FORCE=1` or point split at
`…_sj30_jets` / the graph dir:

```bash
MONITOR_SIG_EVENTS=20000 \
  PYTHON=$CONDA_PREFIX/bin/python bash scripts/build_ds.sh

# split-only against modern subjet jets:
CANONICAL_JETS=dataset/processed/lhco_canonical_leadingpt_sj30_jets \
  RUN_PREPROCESS=0 RUN_GRAPH=0 RUN_SPLIT=1 MONITOR_SIG_EVENTS=20000 \
  PYTHON=$CONDA_PREFIX/bin/python bash scripts/build_ds.sh
```

### 4. Train

CLI defaults match the main recipe (`--model edge_graph`, `--epochs 50`,
OneCycle, `pt` nodes). Pass `--no_early_stop` so the primary result is
`last.pt` after a full schedule.

Important flags (`python -m src.training`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--data_dir` / `--split_manifest` | required | Graph dir + event split |
| `--output` | required | Run root (writes `<timestamp>/`) |
| `--model` | `edge_graph` | Model family (see below) |
| `--backbone` | `edgeconv` | Message-passing backbone (see below) |
| `--epochs` | `50` | Training length |
| `--lr` / `--weight_decay` | `3e-3` / `0.01` | AdamW |
| `--scheduler` | `onecycle` | `onecycle` / `cosine` / `linear` |
| `--hidden_dim` / `--latent_dim` | `64` / `2` | Width / bottleneck |
| `--node_features` | `pt` | `pt` = column 0; `all` = full node vector |
| `--edge_weight` | `1.0` | Edge MSE weight (joint recon models) |
| `--aggr` | `mean` | `mean` / `add` / `max` (edge models) |
| `--dyn_k` | `16` | Dynamic-graph kNN size |
| `--no_bn` | off | Disable BatchNorm (`edge_graph` never uses BN) |
| `--event_score_agg` | `sum` | Event-level score |
| `--no_early_stop` | off | Train full epochs; primary = `last.pt` |
| `--seed` | `42` | Init + shuffle |
| `--contrast_weight` | `0.0` | GLADC graph-contrast loss weight |
| `--cycle_weight` | `0.0` | Latent cycle-consistency loss weight |
| `--topo_reg` / `--lambda_topo` | `none` / `1.0` | Optional train-time topology reg |

Supported `--model` values:

| `--model` | Class | Notes |
|-----------|-------|-------|
| `edge_graph` | `EdgeGraphAE` | **Default.** Joint node+edge recon; fixed EdgeConvEF (Araz) |
| `node_graph` | `NodeGraphAE` | Node recon only; chooses `--backbone` |
| `edge_feature_node_graph` | `EdgeFeatureNodeGraphAE` | Edge features in MP; node recon only |
| `edge_feature_graph` | `EdgeFeatureGraphAE` | Standard GNN + joint recon (ablation vs `edge_graph`) |
| `dynamic_graph` | `DynamicGraphAE` | Dynamic kNN; node recon only |
| `dynamic_edge_graph` | `DynamicEdgeGraphAE` | Dynamic kNN MP; offline edges as recon target |

Supported `--backbone` values (used by the GNN-style models above; ignored by
fixed `edge_graph` blocks): `edgeconv`, `gcn`, `sage`, `gatv2`, `gin`,
`transformer`.

#### Baseline training (reconstruction only)

```bash
PYTHONPATH=. $CONDA_PREFIX/bin/python -m src.training \
  --data_dir dataset/processed/lhco_canonical_leadingpt_sj30_unique6_logef \
  --split_manifest dataset/processed/splits/lhco_canonical_leadingpt_sj30_train80000b_val20000b_test340000b_20000s_monsig20000_seed42.npz \
  --output runs/leadingpt_ksfixed \
  --no_early_stop --event_score_agg sum
```

mJJ-window training is also supported (same model; train on background
outside a dijet-mass window). Helper script:

```bash
bash scripts/run_mjj_window_training.sh
# defaults: exclude 3600–4000 GeV trainpack, seed 123
```

#### Contrastive learning (GLADC)

Graph contrast aligns max-pooled, projected representations from clean and
weight-perturbed encoders, using other graphs in the minibatch as negatives.
The following examples use the same data, split, and 50-epoch recipe as above:

```bash
DATA_DIR=dataset/processed/lhco_canonical_leadingpt_sj30_unique6_logef
SPLIT_MANIFEST=dataset/processed/splits/lhco_canonical_leadingpt_sj30_train80000b_val20000b_test340000b_20000s_monsig20000_seed42.npz

PYTHONPATH=. $CONDA_PREFIX/bin/python -m src.training \
  --data_dir "$DATA_DIR" --split_manifest "$SPLIT_MANIFEST" \
  --output runs/gladc_contrast --model edge_graph --no_early_stop \
  --contrast_weight 1.0 --contrast_source latent \
  --contrast_projection_dim 128 --contrast_temperature 0.2 \
  --perturb_scale 1.0
```

| Setting | Flag |
|---------|------|
| Contrast on EB3 (2D bottleneck by default) | `--contrast_source latent` |
| Contrast on EB2 (64D hidden layer by default) | `--contrast_source eb2` |
| Weaker encoder-weight perturbation | `--perturb_scale 0.1` |
| Add latent cycle consistency | `--cycle_weight 0.05 --cycle_global_mode pooled` |

`--contrast_weight` scales the contrastive term added to reconstruction loss;
`--cycle_weight` independently scales the cycle term.

#### Our adjacent-subjet regularizer

The regularizer pulls neighboring subjet representations together along the
input graph edges. It adds `lambda_topo * R` to reconstruction loss, where `R`
is the graph-average sum of undirected edge lengths divided by node count.
Using the paths set above:

```bash
PYTHONPATH=. $CONDA_PREFIX/bin/python -m src.training \
  --data_dir "$DATA_DIR" --split_manifest "$SPLIT_MANIFEST" \
  --output runs/adjacent_reg --model edge_graph --no_early_stop \
  --topo_reg graph_eb3_attraction --lambda_topo 1.0
```

| Representation | `--topo_reg` | Example `--lambda_topo` |
|----------------|--------------|-------------------------|
| EB3 bottleneck | `graph_eb3_attraction` | `1.0` |
| EB2 hidden layer | `graph_eb2_attraction` | `1.0` |
| Both EB2 and EB3 | `graph_eb2_eb3_attraction` | `0.5,1.0` (EB2, EB3) |

A single weight applies to both terms in the joint variant. Add `--topo_norm`
to L2-normalize the representations used by the regularizer.

To combine contrastive learning and regularization, add
`--topo_reg graph_eb3_attraction --lambda_topo 1.0` to the contrastive command
and use a separate `--output`, such as `runs/gladc_contrast_reg`.
The weighted losses add together. Both examples keep reconstruction as the
anomaly score; evaluate their `last.pt` checkpoints with the commands below.

### 5. Eval

Held-out test from the training split (`split.npz` written next to the run).

Important flags:

| Flag | Meaning |
|------|---------|
| `--checkpoint` | Usually `…/last.pt` |
| `--data_dir` | Graph dataset to score |
| `--split` | Run `split.npz` (`eval_test_idx`) |
| `--output_dir` | Metrics / scores |
| `--event_score_agg` | Must match training (`sum` by default) |
| `--batch_size` / `--cache_shards` | Throughput (defaults 2048 / 32) |

For BB1 / other labeled dirs use `eval_labeled_dataset` (no `--split`, or
optional `--split_manifest` + `--split` role).

```bash
PYTHONPATH=. $CONDA_PREFIX/bin/python -m scripts.eval_test_idx \
  --checkpoint runs/<run>/<timestamp>/last.pt \
  --data_dir dataset/processed/lhco_canonical_leadingpt_sj30_unique6_logef \
  --split runs/<run>/<timestamp>/split.npz \
  --output_dir runs/<run>/<timestamp>/full_test
```

Labeled external set (e.g. BB1):

```bash
PYTHONPATH=. $CONDA_PREFIX/bin/python -m scripts.eval_labeled_dataset \
  --checkpoint runs/<run>/<timestamp>/last.pt \
  --data_dir dataset/processed/bb1_canonical_leadingpt_sj30_unique6_logef \
  --output_dir runs/<run>/<timestamp>/bb1_eval
```

Optional graph ablations (rebuild graph only; reuse preprocess + split when
event ids match), e.g. `K=8` or `EDGE_PT_SCALE=raw`:

```bash
K=8 PYTHON=$CONDA_PREFIX/bin/python bash scripts/build_graph.sh
EDGE_PT_SCALE=raw PYTHON=$CONDA_PREFIX/bin/python bash scripts/build_graph.sh
```

## Kitchen Sink / transfer eval

Build KS graphs with the same preprocess → graph settings, then score a
trained LHCO checkpoint with the labeled-dataset evaluator (or the formal KS
helper scripts under `scripts/run_*_ks_eval.sh`).

```bash
# example: score a checkpoint on a labeled graph directory
PYTHONPATH=. $CONDA_PREFIX/bin/python -m scripts.eval_labeled_dataset \
  --checkpoint runs/<run>/<timestamp>/last.pt \
  --data_dir dataset/processed/<ks_graph_dir> \
  --output_dir runs/<run>/<timestamp>/ks_eval
```

## Notes

- Metrics are computed at event level through `event_id`.
- Supported event score aggregations: `sum`, `mean`, `max`, `min`, `pt_weighted`.
- Do not use monitor AUC (`--save_monitor_best`) to pick the reported model;
  primary metrics use `last.pt` under `--no_early_stop`.
- Data files, processed shards, checkpoints, and run outputs are intentionally not tracked.
