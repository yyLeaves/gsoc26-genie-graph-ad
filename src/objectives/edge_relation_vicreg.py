"""Negative-free consistency for physical edge relations.

Each positive pair is the same physical edge observed once with clean inputs
and once after a mild, topology-preserving corruption. No other edge, jet, or
event is declared a negative. The representation is built directly from the
production node latents and edge reconstruction, so no auxiliary projection
head can absorb the training signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, to_dense_batch

from src.models.reconstruction import Reconstruction

from ._masking import sample_graph_mask
from .edge_relation_contrast import UndirectedEdgePairs, undirected_edge_pairs


@dataclass(frozen=True, slots=True)
class CorruptedRelationView:
    """Topology-preserving corrupted inputs and affected physical edges."""

    node: torch.Tensor
    edge: torch.Tensor
    pairs: UndirectedEdgePairs
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    affected_pairs: torch.Tensor


@dataclass(frozen=True, slots=True)
class RelationVICRegTerms:
    """Negative-free relation consistency and its diagnostics."""

    loss: torch.Tensor
    invariance: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    affected_fraction: torch.Tensor


def symmetric_relation_representation(
    output: Reconstruction,
    pairs: UndirectedEdgePairs,
) -> torch.Tensor:
    """Combine symmetric endpoint latents with production edge predictions."""
    if output.edge is None:
        raise ValueError("edge-relation consistency requires edge reconstruction")
    source, target = pairs.endpoints
    endpoints = torch.cat([
        torch.minimum(output.latent[source], output.latent[target]),
        torch.maximum(output.latent[source], output.latent[target]),
    ], dim=-1)
    pair_edge = scatter(
        output.edge,
        pairs.inverse,
        dim=0,
        dim_size=pairs.target.size(0),
        reduce="mean",
    )
    return torch.cat([endpoints, pair_edge], dim=-1)


def _relation_moments(
    relation: torch.Tensor,
    pair_graph: torch.Tensor,
    num_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-graph mean, standard deviation, and correlation matrix."""
    mean = scatter(
        relation, pair_graph, dim=0, dim_size=num_graphs, reduce="mean")
    centered = relation - mean[pair_graph]
    variance = scatter(
        centered.square(), pair_graph, dim=0,
        dim_size=num_graphs, reduce="mean")
    std = variance.add(1e-6).sqrt()
    normalized = centered / std[pair_graph]
    outer = normalized.unsqueeze(-1) * normalized.unsqueeze(-2)
    correlation = scatter(
        outer, pair_graph, dim=0, dim_size=num_graphs, reduce="mean")
    return mean, std, correlation


class EdgeRelationVICReg(nn.Module):
    """Preserve each edge relation across a mild detector-like corruption.

    The strongest half of each jet's nodes is never masked. Variance and
    covariance are matched to the clean stop-gradient branch instead of being
    forced toward a universal shape, preserving both compact and multi-prong
    background topology.
    """

    def __init__(
        self,
        *,
        node_mask_fraction: float = 0.15,
        edge_mask_fraction: float = 0.10,
        moment_weight: float = 0.10,
    ) -> None:
        super().__init__()
        for name, value in (
            ("node_mask_fraction", node_mask_fraction),
            ("edge_mask_fraction", edge_mask_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if node_mask_fraction == 0.0 and edge_mask_fraction == 0.0:
            raise ValueError("at least one corruption fraction must be positive")
        if moment_weight < 0.0:
            raise ValueError("moment_weight must be non-negative")
        self.node_mask_fraction = float(node_mask_fraction)
        self.edge_mask_fraction = float(edge_mask_fraction)
        self.moment_weight = float(moment_weight)

    @staticmethod
    def _sample_soft_node_mask(
        node_input: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
        fraction: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if fraction == 0.0:
            return torch.zeros(
                node_input.size(0), dtype=torch.bool, device=node_input.device)
        padded, valid_node = to_dense_batch(
            node_input[:, 0], batch_index,
            fill_value=float("-inf"), batch_size=num_graphs)
        node_count = valid_node.sum(dim=1)
        order = torch.argsort(padded, dim=1, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(
            1, order,
            torch.arange(padded.size(1), device=node_input.device)
            .unsqueeze(0).expand(num_graphs, -1),
        )
        node_rank = rank[valid_node]
        soft = (
            (node_count[batch_index] >= 2)
            & (node_rank >= (node_count[batch_index] + 1) // 2)
        )
        return sample_graph_mask(
            batch_index, num_graphs, fraction, eligible=soft, generator=generator)

    def corrupt(
        self,
        node_input: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
        *,
        generator: torch.Generator | None = None,
    ) -> CorruptedRelationView:
        """Mask only soft nodes and edge attributes; retain graph topology."""
        pairs = undirected_edge_pairs(edge_index, edge_attr, batch_index)
        node_mask = self._sample_soft_node_mask(
            node_input, batch_index, num_graphs,
            self.node_mask_fraction, generator)
        edge_mask = sample_graph_mask(
            pairs.graph, num_graphs, self.edge_mask_fraction, generator=generator)

        node_mean = scatter(
            node_input, batch_index, dim=0,
            dim_size=num_graphs, reduce="mean")
        corrupted_node = torch.where(
            node_mask.unsqueeze(-1), node_mean[batch_index], node_input)

        directed_graph = batch_index[edge_index[0]]
        edge_mean = scatter(
            edge_attr, directed_graph, dim=0,
            dim_size=num_graphs, reduce="mean")
        directed_mask = edge_mask[pairs.inverse]
        corrupted_edge = torch.where(
            directed_mask.unsqueeze(-1),
            edge_mean[directed_graph],
            edge_attr,
        )
        affected = (
            edge_mask
            | node_mask[pairs.endpoints[0]]
            | node_mask[pairs.endpoints[1]]
        )
        return CorruptedRelationView(
            node=corrupted_node,
            edge=corrupted_edge,
            pairs=pairs,
            node_mask=node_mask,
            edge_mask=edge_mask,
            affected_pairs=affected,
        )

    def forward(
        self,
        model: nn.Module,
        clean_output: Reconstruction,
        node_input: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
        *,
        generator: torch.Generator | None = None,
    ) -> RelationVICRegTerms:
        view = self.corrupt(
            node_input, edge_index, edge_attr, batch_index, num_graphs,
            generator=generator)
        student_output = model.forward(view.node, edge_index, view.edge)
        student = symmetric_relation_representation(student_output, view.pairs)
        teacher = symmetric_relation_representation(
            clean_output, view.pairs).detach()

        teacher_mean, teacher_std, teacher_corr = _relation_moments(
            teacher, view.pairs.graph, num_graphs)
        student_mean, student_std, student_corr = _relation_moments(
            student, view.pairs.graph, num_graphs)
        student_normalized = (
            student - teacher_mean[view.pairs.graph]
        ) / teacher_std[view.pairs.graph]
        teacher_normalized = (
            teacher - teacher_mean[view.pairs.graph]
        ) / teacher_std[view.pairs.graph]

        affected = view.affected_pairs
        invariance = F.smooth_l1_loss(
            student_normalized[affected], teacher_normalized[affected])
        variance = F.mse_loss(
            student_std.log(), teacher_std.detach().log())
        covariance = F.mse_loss(student_corr, teacher_corr.detach())
        loss = invariance + self.moment_weight * (variance + covariance)
        return RelationVICRegTerms(
            loss=loss,
            invariance=invariance.detach(),
            variance=variance.detach(),
            covariance=covariance.detach(),
            affected_fraction=affected.float().mean().detach(),
        )


__all__ = [
    "CorruptedRelationView",
    "EdgeRelationVICReg",
    "RelationVICRegTerms",
    "symmetric_relation_representation",
]
