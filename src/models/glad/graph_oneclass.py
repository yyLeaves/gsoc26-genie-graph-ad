"""Graph-level one-class baselines for jet-graph anomaly detection.

These models intentionally do not reconstruct the input graph.  They provide
independent graph-level baselines against the Node+Edge autoencoder:

* ``OneClassGIN`` learns a compact hypersphere of background graph embeddings.
* ``InfoGraphOneClass`` learns local/global mutual information on background
  graphs and scores distance from the resulting background representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.utils import scatter

from ..inputs import select_node_features


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    # Bias-free layers follow the usual Deep-SVDD restriction and avoid the
    # trivial constant solution caused by trainable offsets.
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias=False),
        nn.LeakyReLU(0.1),
        nn.Linear(out_dim, out_dim, bias=False),
    )


class GINGraphEncoder(nn.Module):
    """Three-layer GIN with mean/max graph pooling and no BatchNorm."""

    def __init__(self, in_dim: int = 1, hidden_dim: int = 64,
                 embedding_dim: int = 64):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        dims = (in_dim, hidden_dim, hidden_dim, embedding_dim)
        self.layers = nn.ModuleList(
            GINConv(_mlp(input_dim, output_dim), train_eps=True)
            for input_dim, output_dim in pairwise(dims))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch_index: torch.Tensor, num_graphs: int
                ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = x
        for layer in self.layers:
            nodes = F.leaky_relu(layer(nodes, edge_index), 0.1)
        mean = scatter(nodes, batch_index, dim=0, dim_size=num_graphs,
                       reduce="mean")
        maximum = scatter(nodes, batch_index, dim=0, dim_size=num_graphs,
                          reduce="max")
        return nodes, torch.cat([mean, maximum], dim=-1)


class _GraphOneClassBase(nn.Module):
    method: str
    _uncalibrated_message: str

    def __init__(self, hidden_dim: int = 64, embedding_dim: int = 64):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.encoder = GINGraphEncoder(1, hidden_dim, embedding_dim)
        graph_dim = 2 * embedding_dim
        self.register_buffer("center", torch.zeros(graph_dim))
        self.register_buffer("scale", torch.ones(graph_dim))
        self.register_buffer("calibrated", torch.tensor(False))

    def _inputs(self, batch) -> torch.Tensor:
        return select_node_features(batch, 1, (0,))

    def graph_embedding(self, batch) -> torch.Tensor:
        _, graph = self.encoder(
            self._inputs(batch), batch.edge_index, batch.batch,
            batch.num_graphs)
        return graph

    @torch.no_grad()
    def anomaly_score(self, batch) -> torch.Tensor:
        graph = self.graph_embedding(batch)
        return self.embedding_score(graph)

    def set_background_statistics(self, center: torch.Tensor,
                                  scale: torch.Tensor) -> None:
        self.center.copy_(center)
        self.scale.copy_(scale.clamp_min(1e-3))
        self.calibrated.fill_(True)

    def embedding_score(self, graph: torch.Tensor) -> torch.Tensor:
        if not bool(self.calibrated):
            raise RuntimeError(self._uncalibrated_message)
        return ((graph - self.center) / self.scale).square().mean(dim=-1)

    def config(self) -> dict:
        return {
            "method": self.method,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
        }


class OneClassGIN(_GraphOneClassBase):
    """GIN encoder optimized with a fixed-center Deep-SVDD objective."""

    method = "one_class_gin"
    _uncalibrated_message = "OneClassGIN background center is not set"

    def loss(self, batch) -> torch.Tensor:
        return self.embedding_score(self.graph_embedding(batch)).mean()


@dataclass(frozen=True, slots=True)
class InfoGraphTerms:
    loss: torch.Tensor
    positive_accuracy: torch.Tensor
    negative_accuracy: torch.Tensor


class InfoGraphOneClass(_GraphOneClassBase):
    """InfoGraph-style local/global MI encoder with one-class scoring."""

    method = "infograph_oneclass"
    _uncalibrated_message = "InfoGraph background statistics are not set"

    def __init__(self, hidden_dim: int = 64, embedding_dim: int = 64):
        super().__init__(hidden_dim, embedding_dim)
        self.node_projection = nn.Linear(embedding_dim, embedding_dim,
                                         bias=False)
        self.graph_projection = nn.Linear(2 * embedding_dim, embedding_dim,
                                          bias=False)

    def terms(self, batch) -> InfoGraphTerms:
        nodes, graph = self.encoder(
            self._inputs(batch), batch.edge_index, batch.batch,
            batch.num_graphs)
        node_view = F.normalize(self.node_projection(nodes), dim=-1)
        graph_view = F.normalize(self.graph_projection(graph), dim=-1)
        positive = (node_view * graph_view[batch.batch]).sum(dim=-1)
        if batch.num_graphs > 1:
            shift = int(torch.randint(
                1, batch.num_graphs, (), device=nodes.device).item())
            negative_graph = graph_view.roll(shift, dims=0)
        else:
            negative_graph = -graph_view
        negative = (node_view * negative_graph[batch.batch]).sum(dim=-1)
        loss = (
            F.binary_cross_entropy_with_logits(
                positive, torch.ones_like(positive))
            + F.binary_cross_entropy_with_logits(
                negative, torch.zeros_like(negative))
        )
        return InfoGraphTerms(
            loss=loss,
            positive_accuracy=(positive > 0).float().mean().detach(),
            negative_accuracy=(negative < 0).float().mean().detach(),
        )

    def loss(self, batch) -> torch.Tensor:
        return self.terms(batch).loss


GRAPH_ONECLASS_METHODS = (OneClassGIN.method, InfoGraphOneClass.method)


def create_graph_oneclass(method: str, hidden_dim: int = 64,
                          embedding_dim: int = 64) -> _GraphOneClassBase:
    if method == OneClassGIN.method:
        return OneClassGIN(hidden_dim, embedding_dim)
    if method == InfoGraphOneClass.method:
        return InfoGraphOneClass(hidden_dim, embedding_dim)
    raise ValueError(f"unknown graph one-class method: {method!r}")


def load_graph_oneclass_checkpoint(path, device="cpu") -> _GraphOneClassBase:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["model_config"]
    model = create_graph_oneclass(
        config["method"], config["hidden_dim"], config["embedding_dim"])
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device)


__all__ = [
    "GRAPH_ONECLASS_METHODS",
    "GINGraphEncoder",
    "InfoGraphOneClass",
    "InfoGraphTerms",
    "OneClassGIN",
    "create_graph_oneclass",
    "load_graph_oneclass_checkpoint",
]
