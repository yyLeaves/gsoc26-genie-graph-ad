# src/data

Convert LHCO events into jet graphs, with sharded storage and batch loading.

| File | Purpose |
| --- | --- |
| [preprocess.py](preprocess.py) | Read HDF5 events, extract jets, and write constituent point-cloud shards |
| [build_subjets.py](build_subjets.py) | Recluster jet constituents into capped exclusive-kT subjets |
| [build_graph.py](build_graph.py) | Add graph connectivity and edge features to point-cloud shards |
| [dataset.py](dataset.py) | Provide `JetDataset` with an LRU shard cache |
| [iterate.py](iterate.py) | Assemble shard-aware batches and prefetch them in a background thread |
| [events.py](events.py) | Stream particles and labels from HDF5 events |
| [extractor.py](extractor.py) | Cluster and select jets, then convert them to PyG `Data` |
| [features.py](features.py) | Compute `raw`, `normalized`, and `log_phys` node features |
| [kinematics.py](kinematics.py) | Compute four-momenta, dijet masses, and relative angular coordinates |
| [graph.py](graph.py) | Build graph topologies and angular-distance, kT, and momentum-sharing edge features |
| [physics_views.py](physics_views.py) | Generate augmented graphs using soft-collinear splitting and detector smearing |
| [grouping.py](grouping.py) | Group events stably and check within-event label consistency |
| [shards.py](shards.py) | Read, write, and validate shards and metadata |
| [fix_lhco_h5.py](fix_lhco_h5.py) | Convert legacy HDF5 byte-valued attributes to strings |
