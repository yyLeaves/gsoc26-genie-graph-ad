"""
Jet graph construction strategies and node feature transforms.

Graph strategies:
  knn    — k nearest neighbors in (η, φ)
  radius — all pairs within ΔR < r
  laman  — Henneberg IRC-safe Laman graph, exactly 2N-3 edges

Node feature modes (Δη, Δφ relative to the pT-weighted jet axis):
  raw        — [pT, η, φ]                                        (N, 3)
  normalized — [pT/ΣpT, Δη, Δφ, ΔR]                            (N, 4)
  log_phys   — [log(pT), log(pT/ΣpT), Δη, Δφ, log(ΔR+10⁻³)]    (N, 5)

kNN/radius use scipy.spatial.cKDTree (no torch-cluster dependency).

    g = build_graph(pt, eta, phi, label=0, strategy="laman", features="log_phys")
"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from .kinematics import relative_coords

__all__ = [
    "knn_edges", "radius_edges", "laman_edges", "unique_k_edges",
    "fully_connected_edges", "edge_features", "compute_node_features",
    "build_graph", "FEATURE_DIM", "EDGE_FEATURE_DIM",
]

FEATURE_DIM = {"raw": 3, "normalized": 4, "log_phys": 5}


# ─── Graph edge construction ──────────────────────────────────────────────────

def knn_edges(pos: np.ndarray, k: int) -> torch.Tensor:
    """k nearest neighbors, directed (k outgoing edges per node)."""
    N = len(pos)
    if N <= 1:
        return torch.zeros(2, 0, dtype=torch.long)
    k_use = min(k, N - 1)
    _, idx = cKDTree(pos).query(pos, k=k_use + 1)   # (N, k+1), col 0 = self
    src = np.repeat(np.arange(N), k_use)
    dst = idx[:, 1:].ravel()
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def radius_edges(pos: np.ndarray, r: float) -> torch.Tensor:
    """Radius graph, undirected (both directions for all pairs within ΔR < r)."""
    pairs = np.array(list(cKDTree(pos).query_pairs(r)))
    if len(pairs) == 0:
        return torch.zeros(2, 0, dtype=torch.long)
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


def unique_k_edges(pos: np.ndarray, k: int = 6) -> torch.Tensor:
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


def build_edges(pos: np.ndarray, strategy: str, k: int, r: float) -> torch.Tensor:
    """Dispatch to the edge builder named by `strategy`."""
    if strategy == "knn":
        return knn_edges(pos, k)
    if strategy == "radius":
        return radius_edges(pos, r)
    if strategy == "laman":
        return laman_edges(pos)
    if strategy == "unique":
        return unique_k_edges(pos, k)
    if strategy == "fully_connected":
        return fully_connected_edges(pos)
    raise ValueError(f"Unknown strategy {strategy!r}. Use 'knn', 'radius', "
                     "'laman', 'unique', or 'fully_connected'.")


# ─── Node features ────────────────────────────────────────────────────────────

def compute_node_features(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray,
                          mode: str) -> np.ndarray:
    """Build (N, d) feature matrix; arrays must be nonzero, pT-sorted descending."""
    pt_sum = pt.sum() + 1e-10
    d_eta, d_phi = relative_coords(pt, eta, phi)
    dR = np.sqrt(d_eta ** 2 + d_phi ** 2)

    if mode == "raw":
        return np.stack([pt, eta, phi], axis=1)                          # (N, 3)
    if mode == "normalized":
        return np.stack([pt / pt_sum, d_eta, d_phi, dR], axis=1)         # (N, 4)
    if mode == "log_phys":
        return np.stack([
            np.log(pt + 1e-10),
            np.log(pt / pt_sum + 1e-10),
            d_eta, d_phi,
            np.log(dR + 1e-3),
        ], axis=1)                                                        # (N, 5)
    raise ValueError(
        f"Unknown feature mode {mode!r}. Use 'raw', 'normalized', or 'log_phys'.")


# ─── Full graph builder ───────────────────────────────────────────────────────

def build_graph(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    label: int = 0,
    strategy: str = "knn",
    k: int = 16,
    r: float = 0.4,
    features: str = "log_phys",
    max_nodes: int = None,
) -> Data:
    """Convert (pT, η, φ) particle arrays to a PyG graph.

    Zeros filtered; survivors sorted by pT descending (required for Laman,
    harmless otherwise). Returns None if none survive pT > 0, else Data with:
      x          (N, d)     node features
      pos        (N, 2)     (Δη, Δφ) vs pT-weighted axis, Δφ wrapped to (-π, π]
                            — same convention as extractor.jet_to_data
      edge_index (2, E)     connectivity
      y          (1,)       label
    """
    keep = pt > 0
    pt, eta, phi = pt[keep], eta[keep], phi[keep]

    if len(pt) == 0:
        return None

    # argsort(-pt) matches the pipeline; argsort(pt)[::-1] would REVERSE the tie
    # order and build a different graph for equal-pT nodes
    order = np.argsort(-pt)[:max_nodes]
    pt, eta, phi = pt[order], eta[order], phi[order]
    d_eta, d_phi = relative_coords(pt, eta, phi)
    pos = np.stack([d_eta, d_phi], axis=1)

    edge_index = (build_edges(pos, strategy, k, r) if len(pt) > 1
                  else torch.zeros(2, 0, dtype=torch.long))

    return Data(
        x=torch.tensor(compute_node_features(pt, eta, phi, features),
                       dtype=torch.float),
        pos=torch.tensor(pos, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
    )
