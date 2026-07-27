from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

Scalar = TypeVar("Scalar")


@dataclass(frozen=True, slots=True)
class LossTerms(Generic[Scalar]):
    """Named node/edge reconstruction losses.

    ``total`` is the optimization objective. Node-only models set ``edge`` to
    zero; joint models use ``total = node + edge_weight * edge``.
    """

    total: Scalar
    node: Scalar
    edge: Scalar


@dataclass(frozen=True, slots=True)
class Reconstruction:
    """Named output shared by every graph autoencoder."""

    node: torch.Tensor
    latent: torch.Tensor
    edge: torch.Tensor | None = None


class EdgeAttrPredictor(nn.Module):
    """Reconstruct edge attributes from symmetric endpoint features."""

    def __init__(self, latent_dim: int, edge_dim: int, hidden: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self.fc_direct = nn.Linear(2 * latent_dim, edge_dim)
        self.fc1 = nn.Linear(2 * latent_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, edge_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        src, tgt = edge_index
        endpoints = torch.cat([
            torch.minimum(x[src], x[tgt]),
            torch.maximum(x[src], x[tgt]),
        ], dim=-1)
        hidden = self.dropout(self.relu(self.fc1(endpoints)))
        hidden = self.dropout(self.relu(self.fc2(hidden) + hidden))
        return self.fc3(hidden) + self.fc_direct(endpoints)


def mse_per_graph(pred: torch.Tensor, target: torch.Tensor,
                  index: torch.Tensor,
                  n_graphs: int | None = None) -> torch.Tensor:
    per_row = ((pred - target) ** 2).mean(dim=-1)
    return scatter(per_row, index, dim=0, reduce="mean", dim_size=n_graphs)


def reconstruction_scores(
    output: Reconstruction,
    node_target: torch.Tensor,
    batch,
    *,
    edge_target: torch.Tensor | None = None,
    edge_weight: float = 1.0,
) -> LossTerms[torch.Tensor]:
    """Return per-graph total/node/edge reconstruction errors."""
    node = mse_per_graph(
        output.node, node_target, batch.batch, n_graphs=batch.num_graphs)
    if output.edge is None:
        if edge_target is not None:
            raise ValueError("edge target provided for a node-only reconstruction")
        edge = node.new_zeros(node.shape)
    else:
        if edge_target is None:
            raise ValueError("joint reconstruction requires an edge target")
        edge = mse_per_graph(
            output.edge,
            edge_target,
            batch.batch[batch.edge_index[0]],
            n_graphs=batch.num_graphs,
        )
    return LossTerms(total=node + edge_weight * edge, node=node, edge=edge)


def mean_loss(scores: LossTerms[torch.Tensor]) -> LossTerms[torch.Tensor]:
    """Reduce per-graph reconstruction terms to one optimization loss."""
    return LossTerms(
        total=scores.total.mean(),
        node=scores.node.mean(),
        edge=scores.edge.mean(),
    )


class ReconstructionMixin:

    def _reconstruct(self, batch):
        raise NotImplementedError

    def loss(self, batch) -> LossTerms[torch.Tensor]:
        output, node_target, edge_target = self._reconstruct(batch)
        return mean_loss(reconstruction_scores(
            output, node_target, batch,
            edge_target=edge_target,
            edge_weight=getattr(self, "edge_weight", 1.0),
        ))

    @torch.no_grad()
    def anomaly_score(self, batch) -> torch.Tensor:
        output, node_target, edge_target = self._reconstruct(batch)
        return reconstruction_scores(
            output, node_target, batch,
            edge_target=edge_target,
            edge_weight=getattr(self, "edge_weight", 1.0),
        ).total
