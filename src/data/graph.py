"""Jet graph construction strategies and edge feature transforms.

Graph strategies:
  knn     — directed k nearest neighbors in (η, φ)
  sym_knn — symmetrized k nearest neighbors in (η, φ)
  radius  — all pairs within ΔR < r
  radius_knn — pairs within ΔR < r, capped at k nearest per node
  mst     — minimum-spanning tree in angular distance
  delaunay — planar Delaunay triangulation in (η, φ)
  laman   — Henneberg IRC-safe Laman graph, exactly 2N-3 edges
  unique  — unique-k rigid graph in pT order
  fully_connected — every ordered pair of distinct nodes

kNN/radius use scipy.spatial.cKDTree (no torch-cluster dependency).

"""

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay, QhullError, cKDTree

__all__ = [
    "knn_edges", "sym_knn_edges", "radius_edges", "radius_knn_edges",
    "mst_edges", "delaunay_edges", "laman_edges", "unique_k_edges",
    "fully_connected_edges", "edge_features", "build_edges",
    "EDGE_FEATURE_DIM", "EDGE_FEATURE_MODES", "EDGE_PT_SCALES",
    "GRAPH_STRATEGIES",
    "STRATEGIES_WITH_K", "STRATEGIES_WITH_RADIUS",
]

GRAPH_STRATEGIES = (
    "knn",
    "sym_knn",
    "radius",
    "radius_knn",
    "mst",
    "delaunay",
    "laman",
    "unique",
    "fully_connected",
)
STRATEGIES_WITH_K = frozenset({"knn", "sym_knn", "radius_knn", "unique"})
STRATEGIES_WITH_RADIUS = frozenset({"radius", "radius_knn"})


# ─── Graph edge construction ──────────────────────────────────────────────────

def knn_edges(pos: np.ndarray, k: int) -> torch.Tensor:
    """k nearest neighbors, directed (k incoming edges per node).

    PyG propagates messages from ``edge_index[0]`` to ``edge_index[1]``.  Each
    node is therefore the target of edges from its own nearest neighbours.
    """
    N = len(pos)
    if N <= 1:
        return torch.zeros(2, 0, dtype=torch.long)
    k_use = min(k, N - 1)
    _, idx = cKDTree(pos).query(pos, k=k_use + 1)   # (N, k+1), col 0 = self
    src = idx[:, 1:].ravel()
    dst = np.repeat(np.arange(N), k_use)
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def sym_knn_edges(pos: np.ndarray, k: int) -> torch.Tensor:
    """Symmetrized kNN graph.

    Start from directed kNN in angular space, take the union with the reversed
    edges, and deduplicate. This keeps the local kNN topology while making
    message passing comparable to the bidirectional Laman/unique graphs.
    """
    directed = knn_edges(pos, k)
    if directed.numel() == 0:
        return directed
    pairs = directed.t().numpy()
    pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    pairs = np.unique(pairs, axis=0)
    return torch.tensor(pairs.T, dtype=torch.long)


def radius_edges(pos: np.ndarray, r: float) -> torch.Tensor:
    pairs = np.array(list(cKDTree(pos).query_pairs(r)))
    if len(pairs) == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def radius_knn_edges(pos: np.ndarray, r: float, k: int) -> torch.Tensor:
    """Capped-radius graph.

    For each node, keep up to the k nearest neighbours among particles within
    ΔR < r, then symmetrize and deduplicate. This is a useful control for plain
    radius graphs: it preserves the physical angular scale while limiting how
    much raw local density/degree can dominate.
    """
    N = len(pos)
    if N <= 1:
        return torch.zeros(2, 0, dtype=torch.long)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    src, dst = [], []
    k_use = min(k, N - 1)
    for i in range(N):
        cand = np.flatnonzero(d[i] < r)
        if cand.size == 0:
            continue
        cand = cand[np.argsort(d[i, cand], kind="stable")[:k_use]]
        src.extend([i] * len(cand))
        dst.extend(cand.tolist())
    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    pairs = np.stack([np.asarray(src), np.asarray(dst)], axis=1)
    pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    pairs = np.unique(pairs, axis=0)
    return torch.tensor(pairs.T, dtype=torch.long)


def mst_edges(pos: np.ndarray) -> torch.Tensor:
    """Minimum-spanning tree in angular distance, returned bidirectionally.

    This is a lightweight tree-graph proxy: it keeps the graph connected with
    only N-1 undirected edges, without imposing pT ordering or fixed k.
    """
    N = len(pos)
    if N <= 1:
        return torch.zeros(2, 0, dtype=torch.long)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    zero_offdiag = (d == 0.0) & (~np.eye(N, dtype=bool))
    d[zero_offdiag] = 1e-12
    np.fill_diagonal(d, 0.0)
    tree = minimum_spanning_tree(d).tocoo()
    if tree.row.size == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    src = np.concatenate([tree.row, tree.col]).astype(np.int64)
    dst = np.concatenate([tree.col, tree.row]).astype(np.int64)
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def delaunay_edges(pos: np.ndarray) -> torch.Tensor:
    """Planar Delaunay triangulation in angular space, bidirectional.

    Delaunay is a geometry-driven local graph with variable degree. Degenerate
    point sets fall back to the MST so the graph remains valid.
    """
    N = len(pos)
    if N <= 1:
        return torch.zeros(2, 0, dtype=torch.long)
    if N == 2:
        return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    try:
        tri = Delaunay(pos)
    except (QhullError, ValueError):
        return mst_edges(pos)

    undirected = set()
    for simplex in tri.simplices:
        for a in range(len(simplex)):
            for b in range(a + 1, len(simplex)):
                i, j = sorted((int(simplex[a]), int(simplex[b])))
                if i != j:
                    undirected.add((i, j))
    if not undirected:
        return mst_edges(pos)
    pairs = np.array(sorted(undirected), dtype=np.int64)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def laman_edges(pos: np.ndarray) -> torch.Tensor:
    """Henneberg Laman construction: exactly 2N-3 undirected edges.

    Adds particles one at a time in pT order; each new particle connects to its
    2 nearest among the prior ones. IRC-safe: soft/collinear particles connect
    locally without disturbing existing edges.

    pos: (N, 2) (η, φ), pT-sorted descending. Returns (2, 2*(2N-3)) long.
    """
    N = len(pos)
    if N <= 1:
        return torch.zeros(2, 0, dtype=torch.long)
    if N == 2:
        return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    src, dst = [0], [1]   # seed: two hardest particles

    for i in range(2, N):
        prev = pos[:i]
        dists = ((prev - pos[i]) ** 2).sum(1)      # squared ΔR to prior nodes
        k = min(2, i)
        # stable sort: ties broken by index (= pT order), deterministic
        nn = np.argsort(dists, kind="stable")[:k]
        for j in nn.tolist():
            src.append(i)
            dst.append(j)

    return torch.tensor([src + dst, dst + src], dtype=torch.long)


def unique_k_edges(pos: np.ndarray, k: int) -> torch.Tensor:
    """'unique-k' graph of Araz et al. (2506.19920), §2.3: exactly
    kN - k(k+1)/2 undirected edges.

    One particle at a time in pT order: seed = clique on the k+1 hardest; each
    later particle adds k edges to its k nearest (angular) among prior nodes.

    k=2 = Laman (locally rigid); k>=3 globally rigid (Hendrickson) so geometry
    fixes the embedding uniquely. Paper's best AD result is unique-6.

    pos: (N, 2) (Δη, Δφ), pT-sorted descending.
    """
    N = len(pos)
    if N <= 1:
        return torch.zeros(2, 0, dtype=torch.long)
    if N <= k + 1:
        return fully_connected_edges(pos)  # not enough nodes to seed: clique

    src, dst = [], []
    for i in range(k + 1):                  # seed: clique on the k+1 hardest
        for j in range(i + 1, k + 1):
            src.append(i)
            dst.append(j)
    for i in range(k + 1, N):               # each new node → k nearest prior
        dists = ((pos[:i] - pos[i]) ** 2).sum(1)
        # stable sort: ties broken by pT order, deterministic
        nn = np.argsort(dists, kind="stable")[:k]
        for j in nn.tolist():
            src.append(i)
            dst.append(j)

    return torch.tensor([src + dst, dst + src], dtype=torch.long)


def fully_connected_edges(pos: np.ndarray) -> torch.Tensor:
    """Fully connected directed graph — every ordered pair (i != j).

    EdgeConv sees the whole jet in one layer; edges grow as N(N-1) (vs k·N for
    kNN). Used by Tsan et al. (2111.12849).
    """
    N = len(pos)
    if N <= 1:
        return torch.zeros(2, 0, dtype=torch.long)
    idx = np.arange(N)
    src = np.repeat(idx, N)
    dst = np.tile(idx, N)
    mask = src != dst
    return torch.tensor(np.stack([src[mask], dst[mask]]), dtype=torch.long)


def edge_features(pos: np.ndarray, pt: np.ndarray, edge_index: torch.Tensor,
                  log: bool = False, normalize_pt: bool = True) -> torch.Tensor:
    """Per-edge physics features (E, 3): (θ, k_T, z), the three pairwise
    quantities of Araz et al. (2506.19920), Eq. 3.1:

        θ_ij  = ||pos_i - pos_j||             angular distance in (Δη,Δφ)
        k_T   = min(pt_i, pt_j) · θ_ij        transverse-momentum scale
        z     = min(pt_i, pt_j) / (pt_i+pt_j) momentum sharing

    Carry the pairwise geometry node features lack; input + reconstruction
    target for the edge-aware autoencoder (EdgeGraphAE).

    Args:
        log:          if True return (lnθ, ln k_T, ln z) — what the Araz
                      reference code reconstructs and what gives the best AD
                      here. The paper *text* uses linear (log=False), weaker.
        normalize_pt: divide pt by Σpt first so k_T is O(1) per jet. z is a
                      ratio, unaffected. Set False to match the reference repo
                      function when its upstream input pt is raw.
    """
    pos = np.asarray(pos, dtype=np.float64)
    pt = np.asarray(pt, dtype=np.float64)
    if normalize_pt:
        pt = pt / (pt.sum() + 1e-12)
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    dR = np.linalg.norm(pos[src] - pos[dst], axis=1)
    pt_min = np.minimum(pt[src], pt[dst])
    kt = pt_min * dR
    z = pt_min / np.maximum(pt[src] + pt[dst], 1e-12)
    if log:
        feats = np.stack([
            np.log(np.maximum(dR, 1e-6)),
            np.log(np.maximum(kt, 1e-6)),
            np.log(np.maximum(z, 1e-12)),
        ], axis=1)
    else:
        feats = np.stack([dR, kt, z], axis=1)
    return torch.tensor(feats, dtype=torch.float)


EDGE_FEATURE_DIM = 3
EDGE_FEATURE_MODES = ("none", "linear", "log")
EDGE_PT_SCALES = ("normalized", "raw")


def build_edges(
    pos: np.ndarray,
    strategy: str,
    k: int = 0,
    *,
    radius: float = 0.0,
) -> torch.Tensor:
    """Dispatch to the edge builder named by `strategy`."""
    if strategy in STRATEGIES_WITH_K and k < 1:
        raise ValueError(f"strategy={strategy!r} requires k >= 1, got {k}")
    if strategy in STRATEGIES_WITH_RADIUS and radius <= 0:
        raise ValueError(
            f"strategy={strategy!r} requires radius > 0, got {radius}"
        )
    if strategy == "knn":
        return knn_edges(pos, k)
    if strategy == "sym_knn":
        return sym_knn_edges(pos, k)
    if strategy == "radius":
        return radius_edges(pos, radius)
    if strategy == "radius_knn":
        return radius_knn_edges(pos, radius, k)
    if strategy == "mst":
        return mst_edges(pos)
    if strategy == "delaunay":
        return delaunay_edges(pos)
    if strategy == "laman":
        return laman_edges(pos)
    if strategy == "unique":
        return unique_k_edges(pos, k)
    if strategy == "fully_connected":
        return fully_connected_edges(pos)
    raise ValueError(f"Unknown strategy {strategy!r}. Use 'knn', 'sym_knn', "
                     "'radius', 'radius_knn', 'mst', 'delaunay', 'laman', "
                     "'unique', or 'fully_connected'.")
