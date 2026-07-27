"""Ensure a JetDataset can feed a ModelSpec."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .inputs import require_edge_features, select_node_features

if TYPE_CHECKING:
    from src.data.dataset import JetDataset

    from .factory import ModelSpec


def _require_matrix(value, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"dataset graph requires a rank-2 tensor {name}")
    return value


def ensure_dataset_matches(
    dataset: "JetDataset",
    spec: "ModelSpec",
) -> None:
    """Raise if ``dataset`` cannot feed ``spec`` (call before train/eval).

    Compares the model’s needs (node columns, edges, log edge features, …)
    to dataset metadata and one sample graph. Shard self-consistency stays
    in ``src.data``.
    """
    if len(dataset) == 0:
        raise ValueError("model input dataset must contain at least one graph")

    metadata = dataset.meta
    stored_node_dim = int(metadata["nodes"]["feature_dim"])
    if spec.feature_cols is None:
        if spec.in_dim != stored_node_dim:
            raise ValueError(
                f"model in_dim={spec.in_dim} requires all stored node "
                f"features, but dataset metadata declares {stored_node_dim}"
            )
    elif max(spec.feature_cols) >= stored_node_dim:
        raise ValueError(
            f"model feature_cols requests column {max(spec.feature_cols)}, "
            f"but dataset metadata declares {stored_node_dim} node features"
        )

    graph = dataset[0]
    _require_matrix(getattr(graph, "x", None), "x")
    select_node_features(graph, spec.in_dim, spec.feature_cols)

    if not spec.requires_precomputed_edges:
        return

    edge_metadata = metadata["edges"]
    if edge_metadata is None:
        raise ValueError(
            f"model type {spec.type!r} requires precomputed graph edges; "
            "run src.data.build_graph first"
        )
    edge_index = _require_matrix(
        getattr(graph, "edge_index", None), "edge_index"
    )
    if edge_index.size(0) != 2 or edge_index.dtype != torch.long:
        raise ValueError(
            "dataset graph edge_index must have shape (2, E) and dtype long"
        )
    n_edges = np.asarray(metadata["n_edges"], dtype=np.int64)
    if np.any(n_edges == 0):
        count = int(np.sum(n_edges == 0))
        raise ValueError(
            f"model type {spec.type!r} requires non-empty graph edges for "
            f"message passing, but {count:,} graph(s) have zero edges"
        )

    strategy = edge_metadata.get("strategy")
    if strategy == "knn" and spec.backbone in {"gcn", "gin"}:
        raise ValueError(
            f"backbone={spec.backbone!r} assumes symmetrized / undirected "
            "edges, but dataset strategy='knn' is directed. Use sym_knn, "
            "unique, or an EdgeConv-style backbone."
        )

    if not spec.requires_edge_features:
        return

    edge_features = edge_metadata.get("features")
    if edge_features not in {"linear", "log"}:
        raise ValueError(
            f"model type {spec.type!r} requires edges.features in "
            f"('linear', 'log'), got {edge_features!r}"
        )
    if spec.reconstructs_edges:
        if edge_features != "log":
            raise ValueError(
                f"model type {spec.type!r} reconstructs log edge features "
                f"(ln ΔR, ln k_T, ln z), but dataset metadata declares "
                f"edges.features={edge_features!r}"
            )
        pt_scale = edge_metadata.get("pt_scale")
        if pt_scale != "normalized":
            raise ValueError(
                f"model type {spec.type!r} expects edges.pt_scale="
                f"'normalized', but dataset metadata declares {pt_scale!r}"
            )

    stored_edge_dim = int(edge_metadata["feature_dim"])
    if stored_edge_dim != spec.edge_dim:
        raise ValueError(
            f"model edge_dim={spec.edge_dim}, but dataset metadata declares "
            f"edges.feature_dim={stored_edge_dim}"
        )
    edge_attr = require_edge_features(graph, spec.edge_dim)
    if edge_attr.size(0) != edge_index.size(1):
        raise ValueError(
            "dataset graph edge_attr rows must match edge_index columns"
        )
