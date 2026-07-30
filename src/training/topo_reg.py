"""Cheap topology regularizers for latent / graph geometry.

Used by from-scratch training (``train_graph_ae --topo_reg …``) and by the
short fine-tune probe.  Edge *selection* is stop-grad; lengths use live
embedding distances so gradients shape the latent geometry.
"""

from __future__ import annotations

import numpy as np
import torch

from src.data.graph import unique_k_edges

TOPO_REG_CHOICES = (
    "none",
    "unique_sum",
    "unique_max",
    "unique_ang_sum",
    "unique_ang_max",
    "graph_sum",
    "graph_max",
    "hidden_unique_sum",
    "hidden_unique_max",
    "hidden_graph_sum",
    "hidden_graph_max",
)


def encoder_hidden(model, node_target, edge_index, edge_attr) -> torch.Tensor:
    """Pre-bottleneck encoder hidden (all but last encoder block)."""
    if not hasattr(model, "encoder_blocks"):
        raise TypeError(
            f"hidden_* topo regs require a model with encoder_blocks "
            f"(got {type(model).__name__})")
    h = node_target
    blocks = model.encoder_blocks
    for block in blocks[: max(len(blocks) - 1, 1)]:
        h = block(h, edge_index, edge_attr)
    return h


def _undirected_pairs_from_edge_index(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    pairs = {(int(min(a, b)), int(max(a, b))) for a, b in zip(src, dst) if a != b}
    return sorted(pairs)


def _unique_k_pairs(pos_np: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Undirected unique-k pairs; ``pos_np`` must already be pT-sorted."""
    return _undirected_pairs_from_edge_index(unique_k_edges(pos_np, k))


def pairs_stat_loss(
    emb: torch.Tensor, batch_index: torch.Tensor, num_graphs: int,
    pairs_per_graph: list[list[tuple[int, int]]], stat: str,
) -> torch.Tensor:
    """Mean over graphs of sum/max embedding lengths on given undirected pairs."""
    terms = []
    for g in range(num_graphs):
        z = emb[batch_index == g]
        n = z.size(0)
        pairs = pairs_per_graph[g]
        if n < 2 or not pairs:
            continue
        dist = torch.cdist(z, z)
        ii = torch.tensor([i for i, _ in pairs], device=emb.device, dtype=torch.long)
        jj = torch.tensor([j for _, j in pairs], device=emb.device, dtype=torch.long)
        lengths = dist[ii, jj]
        if stat == "sum":
            terms.append(lengths.sum() / n)
        elif stat == "max":
            terms.append(lengths.max())
        else:
            raise ValueError(stat)
    if not terms:
        return emb.new_zeros(())
    return torch.stack(terms).mean()


def unique_k_stat_loss(
    emb: torch.Tensor, batch_index: torch.Tensor, num_graphs: int,
    pos: torch.Tensor, k: int, stat: str, *, pos_from_emb: bool,
) -> torch.Tensor:
    """Unique-k (Laman-family) edge lengths in embedding space."""
    pairs_per_graph: list[list[tuple[int, int]]] = []
    for g in range(num_graphs):
        mask = batch_index == g
        z = emb[mask]
        n = z.size(0)
        if n < 2:
            pairs_per_graph.append([])
            continue
        if pos_from_emb:
            pos_np = z.detach().cpu().numpy().astype(np.float64)
        else:
            pos_np = pos[mask].detach().cpu().numpy().astype(np.float64)
        pairs_per_graph.append(_unique_k_pairs(pos_np, k))
    return pairs_stat_loss(emb, batch_index, num_graphs, pairs_per_graph, stat)


def graph_edge_stat_loss(
    emb: torch.Tensor, batch_index: torch.Tensor, num_graphs: int,
    edge_index: torch.Tensor, stat: str,
) -> torch.Tensor:
    """Sum/max embedding distance along existing undirected jet-graph edges."""
    src, dst = edge_index[0], edge_index[1]
    keep = src < dst
    src, dst = src[keep], dst[keep]
    if src.numel() == 0:
        return emb.new_zeros(())
    lengths = (emb[src] - emb[dst]).pow(2).sum(dim=-1).sqrt()
    edge_graph = batch_index[src]
    terms = []
    for g in range(num_graphs):
        m = edge_graph == g
        if not m.any():
            continue
        lg = lengths[m]
        n = int((batch_index == g).sum().item())
        if stat == "sum":
            terms.append(lg.sum() / max(n, 1))
        elif stat == "max":
            terms.append(lg.max())
        else:
            raise ValueError(stat)
    if not terms:
        return emb.new_zeros(())
    return torch.stack(terms).mean()


def compute_topo_reg(
    name: str,
    latent: torch.Tensor,
    batch,
    *,
    unique_k: int = 6,
    model=None,
    node_target: torch.Tensor | None = None,
    edge_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch a single named regularizer on latent or pre-bottleneck hidden."""
    if name in (None, "", "none"):
        return latent.new_zeros(())

    needs_hidden = name.startswith("hidden_")
    if needs_hidden:
        if model is None or node_target is None or edge_target is None:
            raise ValueError(f"{name} requires model + node/edge targets")
        emb = encoder_hidden(
            model, node_target, batch.edge_index, edge_target)
        base = name[len("hidden_"):]
    else:
        emb = latent
        base = name

    if base in ("unique_sum", "unique_max"):
        return unique_k_stat_loss(
            emb, batch.batch, batch.num_graphs, batch.pos, unique_k,
            "sum" if base.endswith("sum") else "max",
            pos_from_emb=True)
    if base in ("unique_ang_sum", "unique_ang_max"):
        return unique_k_stat_loss(
            emb, batch.batch, batch.num_graphs, batch.pos, unique_k,
            "sum" if base.endswith("sum") else "max",
            pos_from_emb=False)
    if base in ("graph_sum", "graph_max"):
        return graph_edge_stat_loss(
            emb, batch.batch, batch.num_graphs, batch.edge_index,
            "sum" if base.endswith("sum") else "max")
    raise ValueError(f"unknown topo_reg={name!r}")
