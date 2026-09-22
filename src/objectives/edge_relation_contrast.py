"""Training-only contrast between masked edges from the same jet.

The objective never compares different jets. It masks both directions of a
physical edge and asks the model's production edge decoder to identify that
edge's true relation among the other real edges in the same graph. The mask
token is training-only; inference remains ordinary node+edge reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, to_dense_batch

from ._masking import sample_graph_mask


@dataclass(frozen=True, slots=True)
class UndirectedEdgePairs:
    """One row per physical edge, plus directed-to-undirected membership."""

    endpoints: torch.Tensor
    target: torch.Tensor
    graph: torch.Tensor
    inverse: torch.Tensor


@dataclass(frozen=True, slots=True)
class MaskedEdgeView:
    """Masked edge attributes and the physical pairs selected as anchors."""

    edge_attr: torch.Tensor
    pairs: UndirectedEdgePairs
    selected: torch.Tensor


@dataclass(frozen=True, slots=True)
class RelationContrastTerms:
    """Scalar objective and diagnostics for one mini-batch."""

    loss: torch.Tensor
    accuracy: torch.Tensor
    n_anchors: torch.Tensor
    n_graphs: torch.Tensor


def undirected_edge_pairs(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    batch_index: torch.Tensor,
) -> UndirectedEdgePairs:
    """Collapse reciprocal directed edges into physical undirected pairs."""
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if edge_attr.ndim != 2 or edge_attr.size(0) != edge_index.size(1):
        raise ValueError("edge_attr must have shape (E, edge_dim)")
    if batch_index.ndim != 1:
        raise ValueError("batch_index must be one-dimensional")
    if edge_index.size(1) == 0:
        raise ValueError("edge-relation contrast requires at least one edge")

    source, target = edge_index
    if bool((source == target).any()):
        raise ValueError("edge-relation contrast does not support self-loops")
    if bool((batch_index[source] != batch_index[target]).any()):
        raise ValueError("edge_index contains an edge between different graphs")

    low = torch.minimum(source, target)
    high = torch.maximum(source, target)
    stride = max(1, int(batch_index.numel()))
    pair_key, inverse = torch.unique(
        low * stride + high, sorted=True, return_inverse=True)
    endpoints = torch.stack(
        [pair_key.div(stride, rounding_mode="floor"), pair_key.remainder(stride)],
        dim=0,
    )
    pair_target = scatter(
        edge_attr, inverse, dim=0, dim_size=pair_key.numel(), reduce="mean")
    pair_graph = batch_index[endpoints[0]]
    return UndirectedEdgePairs(
        endpoints=endpoints,
        target=pair_target,
        graph=pair_graph,
        inverse=inverse,
    )


class EdgeRelationContrast(nn.Module):
    """Contrast the production edge decoder against same-jet relations."""

    def __init__(
        self,
        edge_dim: int,
        *,
        mask_fraction: float = 0.15,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()
        if edge_dim <= 0:
            raise ValueError("edge_dim must be positive")
        if not 0.0 < mask_fraction <= 1.0:
            raise ValueError("mask_fraction must lie in (0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.mask_fraction = float(mask_fraction)
        self.temperature = float(temperature)
        # A fixed zero token keeps this auxiliary objective training-only: the
        # production Graph-AE checkpoint remains loadable without an extra
        # relation-module state or inference dependency.
        self.register_buffer("mask_token", torch.zeros(edge_dim))

    def masked_view(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
        *,
        generator: torch.Generator | None = None,
    ) -> MaskedEdgeView:
        """Mask reciprocal directions together and retain one anchor per jet."""
        pairs = undirected_edge_pairs(edge_index, edge_attr, batch_index)
        pair_count = torch.bincount(pairs.graph, minlength=num_graphs)
        selected = sample_graph_mask(
            pairs.graph, num_graphs, self.mask_fraction,
            eligible=pair_count[pairs.graph] >= 2, generator=generator)

        directed_mask = selected[pairs.inverse]
        token = self.mask_token.to(dtype=edge_attr.dtype, device=edge_attr.device)
        masked_attr = torch.where(
            directed_mask.unsqueeze(-1), token.unsqueeze(0), edge_attr)
        return MaskedEdgeView(
            edge_attr=masked_attr,
            pairs=pairs,
            selected=selected,
        )

    @staticmethod
    def _relation_vectors(
        values: torch.Tensor,
        pair_graph: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Convert edge values to within-jet standardized unit vectors."""
        center = scatter(
            values, pair_graph, dim=0, dim_size=num_graphs, reduce="mean")
        centered = values - center[pair_graph]
        variance = scatter(
            centered.square(), pair_graph, dim=0, dim_size=num_graphs,
            reduce="mean")
        standardized = centered / variance[pair_graph].add(1e-6).sqrt()
        return F.normalize(standardized, dim=-1)

    def forward(
        self,
        model: nn.Module,
        node_input: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
        *,
        generator: torch.Generator | None = None,
    ) -> RelationContrastTerms:
        view = self.masked_view(
            edge_index, edge_attr, batch_index, num_graphs,
            generator=generator)
        output = model.forward(node_input, edge_index, view.edge_attr)
        if output.edge is None:
            raise ValueError("edge-relation contrast requires edge reconstruction")
        pair_prediction = scatter(
            output.edge, view.pairs.inverse, dim=0,
            dim_size=view.pairs.target.size(0), reduce="mean")
        query = self._relation_vectors(
            pair_prediction, view.pairs.graph, num_graphs)
        key = self._relation_vectors(
            view.pairs.target, view.pairs.graph, num_graphs).detach()

        selected_index = torch.nonzero(
            view.selected, as_tuple=False).flatten()
        if selected_index.numel() == 0:
            zero = output.latent.sum() * 0.0
            return RelationContrastTerms(
                loss=zero,
                accuracy=zero.detach(),
                n_anchors=zero.detach(),
                n_graphs=zero.detach(),
            )

        anchor_graph = view.pairs.graph[selected_index]
        padded_key, key_mask = to_dense_batch(
            key, view.pairs.graph, batch_size=num_graphs)
        padded_query, valid_anchor = to_dense_batch(
            query[selected_index], anchor_graph, batch_size=num_graphs)
        pair_local = (key_mask.cumsum(dim=1) - 1)[key_mask]
        logits = torch.bmm(
            padded_query, padded_key.transpose(1, 2)) / self.temperature
        logits = logits.masked_fill(~key_mask.unsqueeze(1), float("-inf"))

        flat_logits = logits[valid_anchor]
        targets = pair_local[selected_index]
        anchor_loss = F.cross_entropy(flat_logits, targets, reduction="none")
        per_graph_loss = scatter(
            anchor_loss, anchor_graph, dim=0, dim_size=num_graphs,
            reduce="mean")
        contributing = valid_anchor.any(dim=1)
        loss = per_graph_loss[contributing].mean()
        correct = (flat_logits.argmax(dim=-1) == targets).sum()
        n_anchors = selected_index.new_tensor(
            selected_index.numel(), dtype=query.dtype)
        return RelationContrastTerms(
            loss=loss,
            accuracy=(correct / n_anchors.clamp_min(1.0)).detach(),
            n_anchors=n_anchors.detach(),
            n_graphs=contributing.sum().to(dtype=loss.dtype).detach(),
        )


__all__ = [
    "EdgeRelationContrast",
    "MaskedEdgeView",
    "RelationContrastTerms",
    "UndirectedEdgePairs",
    "undirected_edge_pairs",
]
