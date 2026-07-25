# src/data

Build and load LHCO jet shards. From repo root:

```bash
PYTHONPATH=. python -m src.data.<module> ...
```

## Pipeline

Data dirs and the entry points that write / read them:

```text
HDF5
  │ preprocess
  ▼
constituents/
  │ build_subjets
  ▼
subjets/
  │ build_graph
  ▼
graphs/
  │
  ▼
JetDataset
```

Other `.py` files are helpers under these entry points (see Modules).

## Modules

| File | Role |
|------|------|
| `preprocess.py` | `Entry`: read LHCO HDF5, cluster anti-kT jets, write constituent point-cloud shards |
| `build_subjets.py` | `Entry`: recluster each jet with exclusive-kT into ≤30 subjet nodes |
| `build_graph.py` | `Entry`: add `edge_index` / optional `edge_attr` on a point-cloud shard dir |
| `dataset.py` | `Entry`: `JetDataset` — LRU-cached reader over `shard_*.pt` + `metadata.pt` |
| `iterate.py` | `Entry`: shard-aware batching (`shard_iter`) and GPU prefetch |
| `events.py` | `Helper`: stream HDF5 rows as padded particles + binary labels (`EventReader`) |
| `extractor.py` | `Helper`: jet selection (`JetConfig` / `JetExtractor`) and `jet_to_data` → PyG `Data` |
| `features.py` | `Helper`: build per-node feature vectors: `raw`, `normalized`, or `log_phys` |
| `kinematics.py` | `Helper`: jet 4-vectors, `dijet_mass`, relative `(Δη, Δφ)` |
| `graph.py` | `Helper`: topology builders (`unique`, `knn`, …) and `(θ, k_T, z)` edge features |
| `shards.py` | `Helper`: write/load shards, validate metadata, `ShardWriter` |
| `fix_lhco_h5.py` | `Util`: in-place fix for old Pandas HDF5 byte-string attributes |

`from src.data import JetDataset, shards`. Rest: `from src.data.<mod> import ...`.

## Commands

```bash
python -m src.data.preprocess \
  --h5_path events.h5 --output_dir constituents \
  --features log_phys --min_jet_pt 1200 --min_particles 3 \
  --jet_selection leading_pt --require_two_jets
# --allow_single_jet  |  --labels_path masterkey

python -m src.data.build_subjets \
  --input_dir constituents --output_dir subjets --n_subjets 30

python -m src.data.build_graph \
  --input_dir subjets --output_dir graphs \
  --strategy unique --k 6 \
  --edge_features log --edge_pt_scale normalized
```

`jet_selection`: `leading_pt` (leading jet ≥ pT cut, keep top-2) or `min_pt_all` (every kept jet ≥ cut).

`build_graph`: `--k` for `knn` / `sym_knn` / `radius_knn` / `unique`; `--radius` for `radius` / `radius_knn`.

```bash
python -m src.data.fix_lhco_h5 events.h5
```

## Load

```python
from src.data import JetDataset
from src.data.iterate import shard_iter, prefetch
import numpy as np

ds = JetDataset("graphs", max_cache=4)
for batch in prefetch(shard_iter(
    ds, ds.background_indices(), batch_size=256,
    shuffle_shards=True, shuffle_within=True,
    rng=np.random.default_rng(0), device="cuda",
)):
    ...
```
