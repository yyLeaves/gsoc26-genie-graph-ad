"""FGSD graph fingerprints and shallow one-class anomaly detectors.

FGSD represents a graph by a histogram of pairwise harmonic distances from
the Moore--Penrose inverse of its normalized graph Laplacian.  This module is
a batched 200-bin, [0, 20] variant, not the original MATLAB demo settings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from torch_geometric.utils import to_dense_adj


FGSD_DETECTORS = ("iforest", "lof", "ocsvm")


@dataclass(frozen=True)
class FGSDConfig:
    bins: int = 200
    range_max: float = 20.0

    def __post_init__(self) -> None:
        if self.bins <= 0:
            raise ValueError("FGSD bins must be positive")
        if self.range_max <= 0:
            raise ValueError("FGSD range_max must be positive")


def _spectral_histograms(
    adjacency: torch.Tensor,
    config: FGSDConfig,
) -> torch.Tensor:
    """Return FGSD histograms for equally sized, undirected graphs."""
    adjacency = ((adjacency + adjacency.transpose(1, 2)) > 0).to(torch.float32)
    adjacency.diagonal(dim1=1, dim2=2).zero_()
    degree = adjacency.sum(dim=2)
    inverse_sqrt = degree.clamp_min(1).rsqrt()
    normalized = inverse_sqrt.unsqueeze(2) * adjacency * inverse_sqrt.unsqueeze(1)
    identity = torch.eye(
        adjacency.size(1), device=adjacency.device, dtype=adjacency.dtype
    ).expand_as(adjacency)
    laplacian = identity - normalized
    isolated = degree == 0
    laplacian[isolated] = 0

    inverse = torch.linalg.pinv(laplacian, hermitian=True, rtol=1e-5)
    diagonal = inverse.diagonal(dim1=1, dim2=2)
    distances = (
        diagonal.unsqueeze(2) + diagonal.unsqueeze(1) - 2 * inverse
    ).clamp_min_(0)
    # Degenerate eigenspaces can produce permutation-dependent round-off at
    # exact histogram boundaries even in float64.  Quantizing far below the
    # 0.1-wide default bins removes that numerical artefact without changing
    # the represented distance scale.
    distances = torch.round(distances * 1e5) / 1e5

    bin_index = torch.floor(
        distances * (config.bins / config.range_max)
    ).to(torch.long)
    inside = distances <= config.range_max
    bin_index.clamp_(0, config.bins - 1)
    histograms = torch.zeros(
        adjacency.size(0), config.bins,
        device=adjacency.device, dtype=torch.float32,
    )
    histograms.scatter_add_(
        1,
        bin_index.flatten(1),
        inside.flatten(1).to(torch.float32),
    )
    return histograms


@torch.no_grad()
def fgsd_batch(batch, config: FGSDConfig = FGSDConfig()) -> torch.Tensor:
    """Compute one FGSD fingerprint per graph in a PyG batch."""
    node_counts = torch.diff(batch.ptr)
    max_nodes = int(node_counts.max())
    dense = to_dense_adj(
        batch.edge_index, batch=batch.batch, max_num_nodes=max_nodes
    )
    output = torch.empty(
        batch.num_graphs, config.bins, device=dense.device, dtype=torch.float32
    )
    for count in torch.unique(node_counts).tolist():
        selected = torch.where(node_counts == count)[0]
        adjacency = dense[selected, :count, :count]
        output[selected] = _spectral_histograms(adjacency, config)
    return output


def create_fgsd_detector(name: str, seed: int = 123):
    """Create one of the shallow detectors used with FGSD in GLADC."""
    if name == "iforest":
        return IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "lof":
        return LocalOutlierFactor(
            n_neighbors=20,
            contamination="auto",
            novelty=True,
            n_jobs=-1,
        )
    if name == "ocsvm":
        return OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)
    raise ValueError(f"unknown FGSD detector {name!r}; choose from {FGSD_DETECTORS}")


def fgsd_anomaly_score(detector, fingerprints) -> np.ndarray:
    """Return anomaly-oriented scores (larger means less normal)."""
    fingerprints = np.asarray(fingerprints, dtype=np.float32)
    return -np.asarray(detector.decision_function(fingerprints)).reshape(-1)
