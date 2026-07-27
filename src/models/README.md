# src/models

Graph autoencoders for LHCO jet anomaly detection. From repo root:

```bash
PYTHONPATH=. python -c "from src.models import ModelSpec, create_model"
```

## Default path

sj30 + unique-6 + log edges → **`edge_graph`** (`EdgeGraphAE`, Araz reference):

```text
JetDataset
  │ ModelSpec(type="edge_graph", ...)
  │ ensure_dataset_matches
  │ create_model
  ▼
EdgeGraphAE
  │ loss / anomaly_score
  ▼
LossTerms / per-graph scores
```

Train via `scripts/train_graph_ae.py` (uses this package). Eval scripts load with
`load_model_and_spec`.

## Types

`ModelSpec.type` / `--model` = class name without `AE`, snake_case:

| type | Class | Notes |
|------|-------|-------|
| `edge_graph` | `EdgeGraphAE` | Default. Joint node+edge recon, fixed EdgeConvEF blocks |
| `node_graph` | `NodeGraphAE` | Node recon only; choose `--backbone` |
| `edge_feature_node_graph` | `EdgeFeatureNodeGraphAE` | Edge features in MP; node recon only |
| `edge_feature_graph` | `EdgeFeatureGraphAE` | Standard GNN + joint recon (ablation vs `edge_graph`) |
| `dynamic_graph` | `DynamicGraphAE` | Dynamic kNN; node recon only |
| `dynamic_edge_graph` | `DynamicEdgeGraphAE` | Dynamic kNN MP; offline edges as recon target |

Old CLI names (`edgeae`, `ae`, …) are listed in `../MODEL_TYPE_NAMES.md` (outside
`develop/`). No in-code aliases.

## Modules

| File | Role |
|------|------|
| `factory.py` | `Entry`: `ModelSpec`, capability table, `create_model`, `load_model*` |
| `ensure_dataset_matches.py` | `Entry`: `ensure_dataset_matches(ds, spec)` — raise if this dataset cannot feed the model |
| `edge_graph_ae.py` | `Entry`: Araz reference `EdgeGraphAE` |
| `node_graph_ae.py` | `Entry`: `NodeGraphAE` |
| `edge_feature_node_graph_ae.py` | `Entry`: edge-aware ablations (`EdgeFeature*`) |
| `dynamic_graph_ae.py` | `Entry`: dynamic-kNN ablations (`Dynamic*`) |
| `reconstruction.py` | `Helper`: `Reconstruction`, `LossTerms`, scores, `EdgeAttrPredictor`, mixin |
| `inputs.py` | `Helper`: `feature_cols` / `edge_attr` width checks |
| `blocks.py` | `Helper`: shared convs, `mlp`, `BACKBONES` |

`from src.models import ...` for the public surface in `__init__.py`. Rest:
`from src.models.<mod> import ...`.

## Construct / load

```python
from src.data import JetDataset
from src.models import (ModelSpec, create_model, load_model_and_spec,
                        ensure_dataset_matches)

ds = JetDataset("graphs")
spec = ModelSpec(
    type="edge_graph", in_dim=1, edge_dim=3, use_bn=False,
    feature_cols=(0,),  # pT column
)
ensure_dataset_matches(ds, spec)
model = create_model(spec)

# or from a checkpoint (one disk read):
model, spec = load_model_and_spec("runs/.../best.pt", device="cpu")
```

`model.loss(batch)` → `LossTerms` (train). `model.anomaly_score(batch)` → per-graph
scores (eval). Joint models use `total = node + edge_weight * edge`.
