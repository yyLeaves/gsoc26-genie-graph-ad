# src/models

Build jet-graph autoencoders and compute reconstruction losses and per-graph anomaly scores.

| File / Module | Purpose |
| --- | --- |
| [factory.py](factory.py) | Define models with `ModelSpec`, create instances, and load checkpoints |
| [ensure_dataset_matches.py](ensure_dataset_matches.py) | Check dataset features and graph structure against model specifications |
| [edge_graph_ae.py](edge_graph_ae.py) | Implement `EdgeGraphAE` for joint node and edge reconstruction |
| [reference_backbone_edge_graph_ae.py](reference_backbone_edge_graph_ae.py) | Replace the `EdgeGraphAE` encoder backbone while sharing its decoder and objective |
| [node_graph_ae.py](node_graph_ae.py) | Implement `NodeGraphAE` for node reconstruction with selectable static GNN backbones |
| [edge_feature_node_graph_ae.py](edge_feature_node_graph_ae.py) | Use edge-aware message passing for node-only or joint node-edge reconstruction |
| [dynamic_graph_ae.py](dynamic_graph_ae.py) | Rebuild kNN neighborhoods at each layer for node-only or joint reconstruction |
| [blocks.py](blocks.py) | Construct MLPs, GNN convolutions, and residual blocks |
| [inputs.py](inputs.py) | Select node-feature columns and validate node and edge feature dimensions |
| [reconstruction.py](reconstruction.py) | Define reconstruction outputs and losses, edge prediction, and per-graph scoring |
| [glad](glad/README.md) | Provide graph-level one-class and distillation baselines |
