"""EdgeConv adaptation of GLocalKD random representation distillation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from ..edge_graph_ae import EdgeBlock
from ..inputs import require_edge_features, select_node_features
from ..reconstruction import mse_per_graph

GLOCAL_KD_COMPONENTS = ("node", "graph", "both")


class GLocalKDEncoder(nn.Module):
    """Three-block EdgeConv encoder returning node and graph embeddings."""

    def __init__(self, in_dim: int = 1, edge_dim: int = 3,
                 hidden_dim: int = 64, embedding_dim: int = 64,
                 aggr: str = "mean", dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.aggr = aggr
        self.dropout = dropout
        dims = (in_dim, hidden_dim, hidden_dim, embedding_dim)
        self.blocks = nn.ModuleList(
            EdgeBlock(input_dim, output_dim, edge_dim, hidden=hidden_dim,
                      aggr=aggr, dropout=dropout)
            for input_dim, output_dim in pairwise(dims))

    def forward(self, x, edge_index, edge_attr, batch_index, n_graphs):
        nodes = x
        for block in self.blocks:
            nodes = F.normalize(
                block(nodes, edge_index, edge_attr), p=2, dim=-1)
        graph = scatter(
            nodes, batch_index, dim=0, reduce="max", dim_size=n_graphs)
        return nodes, graph


@dataclass(frozen=True, slots=True)
class GLocalKDTerms:
    node: torch.Tensor
    graph: torch.Tensor
    both: torch.Tensor


def select_glocal_kd_component(
        terms: GLocalKDTerms, component: str) -> torch.Tensor:
    if component not in GLOCAL_KD_COMPONENTS:
        raise ValueError(
            f"component must be one of {GLOCAL_KD_COMPONENTS}, got {component!r}")
    return getattr(terms, component)


class GLocalKD(nn.Module):
    """Frozen random teacher and trainable student with matched encoders."""

    def __init__(self, *, component: str = "both", in_dim: int = 1,
                 edge_dim: int = 3, hidden_dim: int = 64,
                 embedding_dim: int = 64, aggr: str = "mean",
                 student_dropout: float = 0.1, teacher_seed: int = 123,
                 student_seed: int = 124):
        super().__init__()
        if component not in GLOCAL_KD_COMPONENTS:
            raise ValueError(
                f"component must be one of {GLOCAL_KD_COMPONENTS}, "
                f"got {component!r}")
        self.component = component
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.aggr = aggr
        self.student_dropout = student_dropout
        self.teacher_seed = teacher_seed
        self.student_seed = student_seed

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(teacher_seed)
            self.teacher = GLocalKDEncoder(
                in_dim, edge_dim, hidden_dim, embedding_dim, aggr, 0.0)
            torch.manual_seed(student_seed)
            self.student = GLocalKDEncoder(
                in_dim, edge_dim, hidden_dim, embedding_dim, aggr,
                student_dropout)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def _inputs(self, batch):
        nodes = select_node_features(batch, self.in_dim, (0,))
        edges = require_edge_features(batch, self.edge_dim)
        return nodes, edges

    def terms(self, batch) -> GLocalKDTerms:
        nodes, edges = self._inputs(batch)
        with torch.no_grad():
            target_nodes, target_graph = self.teacher(
                nodes, batch.edge_index, edges, batch.batch,
                batch.num_graphs)
        student_nodes, student_graph = self.student(
            nodes, batch.edge_index, edges, batch.batch, batch.num_graphs)
        node = mse_per_graph(
            student_nodes, target_nodes, batch.batch, batch.num_graphs)
        graph = ((student_graph - target_graph) ** 2).mean(dim=-1)
        return GLocalKDTerms(node=node, graph=graph, both=node + graph)

    def loss(self, batch) -> torch.Tensor:
        return select_glocal_kd_component(
            self.terms(batch), self.component).mean()

    @torch.no_grad()
    def anomaly_score(self, batch) -> torch.Tensor:
        return select_glocal_kd_component(
            self.terms(batch), self.component)

    def config(self) -> dict:
        return {
            "component": self.component,
            "in_dim": self.in_dim,
            "edge_dim": self.edge_dim,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "aggr": self.aggr,
            "student_dropout": self.student_dropout,
            "teacher_seed": self.teacher_seed,
            "student_seed": self.student_seed,
        }


def load_glocal_kd_checkpoint(path, device="cpu") -> GLocalKD:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = GLocalKD(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device)
