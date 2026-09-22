"""Latent re-encoding discrepancy for graph autoencoders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter

from src.models.reconstruction import Reconstruction, mse_per_graph

__all__ = ["LatentCycleTerms", "latent_cycle_scores", "select_cycle_component"]


@dataclass(frozen=True, slots=True)
class LatentCycleTerms:
    """Per-graph local, global, and summed latent-cycle discrepancies."""

    total: torch.Tensor
    local: torch.Tensor
    global_: torch.Tensor
    reencoded: torch.Tensor


def select_cycle_component(
        terms: LatentCycleTerms, component: str) -> torch.Tensor:
    """Return one named per-graph cycle term."""
    if component == "total":
        return terms.total
    if component == "local":
        return terms.local
    if component == "global":
        return terms.global_
    raise ValueError(
        f"unknown cycle component {component!r}; expected total, local, or global")


def latent_cycle_scores(model, output: Reconstruction, batch, *,
                        global_mode: str = "pooled") -> LatentCycleTerms:
    """Re-encode the reconstructed graph and compare its latent to the input.

    The graph connectivity is held fixed.  Both reconstructed node and edge
    attributes are re-encoded, matching the complete ``G -> G_hat -> Z_hat``
    path rather than a node-only shortcut.
    """
    if output.edge is None:
        raise ValueError("latent cycle requires reconstructed edge attributes")
    reencoded = model(
        output.node, batch.edge_index, output.edge).latent
    local = mse_per_graph(
        reencoded, output.latent, batch.batch, n_graphs=batch.num_graphs)
    pooled_original = scatter(
        output.latent, batch.batch, dim=0, reduce="max",
        dim_size=batch.num_graphs)
    pooled_reencoded = scatter(
        reencoded, batch.batch, dim=0, reduce="max",
        dim_size=batch.num_graphs)
    if global_mode != "pooled":
        if global_mode not in {"projected", "unit_projected"}:
            raise ValueError(f"unknown global cycle representation: {global_mode}")
        projection = model.contrastive_projection
        if projection is None:
            raise ValueError("projected cycle requires a contrastive projection head")
        pooled_original = projection(pooled_original)
        pooled_reencoded = projection(pooled_reencoded)
        if global_mode == "unit_projected":
            pooled_original = F.normalize(pooled_original, dim=-1)
            pooled_reencoded = F.normalize(pooled_reencoded, dim=-1)
    global_ = ((pooled_reencoded - pooled_original) ** 2).mean(dim=-1)
    return LatentCycleTerms(
        total=local + global_,
        local=local,
        global_=global_,
        reencoded=reencoded,
    )
