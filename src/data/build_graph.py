import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from . import shards
from .graph import build_edges, edge_features

__all__ = ["with_edges", "with_knn_edges", "build_graph_shards", "add_knn_edges"]


def _pt_of(g: Data) -> np.ndarray:
    """Raw pT from the `pt` attribute stored by preprocess.

    Can't recover pT from node features (mode-dependent: log pT / pT / pT/ΣpT),
    so pre-`pt` shards must be regenerated, not decoded.
    """
    if getattr(g, "pt", None) is None:
        raise ValueError(
            "point-cloud Data has no `pt` attribute — regenerate the shards "
            "with the current preprocess.py (edge features need raw pT).")
    return g.pt.numpy()


def with_edges(g: Data, strategy: str, k: int, edge_feats: str = "none",
               normalize_edge_pt: bool = True) -> Data:
    """Shallow copy of g with edge_index from `strategy`, optionally an
    edge_attr of (θ, k_T, z) features.

    strategy:   "knn" | "laman" | "unique" | "fully_connected"
                ("unique" = Araz unique-k; k=2 ≡ laman).
    edge_feats: "none" | "linear" | "log".
    """
    if edge_feats not in {"none", "linear", "log"}:
        raise ValueError(
            f"edge_feats must be 'none', 'linear', or 'log', got {edge_feats!r}")
    g2 = copy.copy(g)
    pos = g.pos.numpy()
    g2.edge_index = build_edges(pos, strategy, k, r=0.4)
    if edge_feats != "none":
        g2.edge_attr = edge_features(pos, _pt_of(g), g2.edge_index,
                                     log=(edge_feats == "log"),
                                     normalize_pt=normalize_edge_pt)
    return g2


def with_knn_edges(g: Data, k: int) -> Data:
    """Backward-compatible kNN wrapper."""
    return with_edges(g, "knn", k)


def build_graph_shards(input_dir: str, output_dir: str, strategy: str = "knn",
                       k: int = 16, max_shards: int = None,
                       edge_feats: str = "none",
                       normalize_edge_pt: bool = True) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = shards.load_metadata(input_dir)
    total_shards = meta["num_shards"]
    num_shards = min(max_shards, total_shards) if max_shards else total_shards

    print(f"Input     : {input_dir}  ({meta['n']:,} jets, {total_shards} shards)")
    print(f"Output    : {output_dir}")
    print(f"Strategy  : {strategy}"
          + (f"  k={k}" if strategy in ("knn", "unique") else ""))
    if num_shards < total_shards:
        print(f"Subset    : first {num_shards}/{total_shards} shards")
    print(f"Features  : {meta['features']}  dim={meta['feature_dim']}")
    edge_pt_scale = "normalized" if normalize_edge_pt else "raw"
    print(f"Edge feat : {edge_feats}  pt_scale={edge_pt_scale}")

    # stats from jets actually written → metadata self-consistent for partial
    # (max_shards) builds
    labels, n_nodes, n_edges, event_ids, jet_indices = [], [], [], [], []
    have_event_ids = "event_ids" in meta
    have_jet_indices = "jet_idx" in meta
    in_deg_max = 0
    t0 = time.time()

    for si in range(num_shards):
        jets = [with_edges(g, strategy, k, edge_feats=edge_feats,
                           normalize_edge_pt=normalize_edge_pt)
                for g in shards.load_shard(input_dir, si)]
        shards.save_shard(jets, output_dir, si)
        for g in jets:
            labels.append(int(g.y))
            n_nodes.append(g.x.shape[0])
            n_edges.append(g.edge_index.shape[1])
            if have_event_ids:
                event_ids.append(int(g.event_id))
            if have_jet_indices:
                jet_indices.append(int(g.jet_idx))
            if g.edge_index.shape[1]:
                in_deg_max = max(in_deg_max,
                                 int(torch.bincount(g.edge_index[1]).max()))

        elapsed = time.time() - t0
        eta_min = elapsed / (si + 1) * (num_shards - si - 1) / 60
        print(f"  Shard {si+1:>3}/{num_shards}  jets={len(labels):,}"
              f"  {elapsed:.0f}s  eta={eta_min:.1f}min", flush=True)

    labels = np.array(labels, dtype=np.int64)
    n_nodes = np.array(n_nodes, dtype=np.int32)
    n_edges = np.array(n_edges, dtype=np.int32)
    out_meta = {
        **meta,
        "n": len(labels),
        "labels": labels,
        "n_nodes": n_nodes,
        "num_shards": num_shards,
        "strategy": strategy,
        "n_edges": n_edges,
        "has_edges": True,
        "edge_feats": edge_feats,
        "edge_pt_scale": edge_pt_scale,
        "edge_dim": (0 if edge_feats == "none" else 3),
    }
    if strategy in ("knn", "unique"):
        out_meta["k"] = k
    if have_event_ids:
        out_meta["event_ids"] = np.array(event_ids, dtype=np.int64)
    if have_jet_indices:
        out_meta["jet_idx"] = np.array(jet_indices, dtype=np.int8)
    shards.save_metadata(out_meta, output_dir)

    elapsed = time.time() - t0
    size_gb = sum(f.stat().st_size for f in output_dir.glob("*.pt")) / 1e9
    print(f"\nDone in {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  Shards     : {num_shards}  x  shard_size={meta['shard_size']}")
    print(f"  Jets       : {len(labels):,}  (signal={labels.sum():,}  "
          f"bkg={(labels == 0).sum():,})")
    print(f"  Nodes/jet  : mean={n_nodes.mean():.1f}  max={n_nodes.max()}")
    print(f"  Edges/jet  : mean={n_edges.mean():.1f}  max={n_edges.max()}  "
          f"({strategy})")
    print(f"  Edges/node : mean="
          f"{n_edges.sum(dtype=np.int64) / n_nodes.sum(dtype=np.int64):.2f}  "
          f"(= mean in-degree)")
    print(f"  In-degree  : max={in_deg_max}")
    print(f"  Disk       : {size_gb:.1f} GB")
    print(f"  Saved to   : {output_dir}/")


def add_knn_edges(input_dir: str, output_dir: str, k: int = 16) -> None:
    """Backward-compatible entry point (kNN only)."""
    build_graph_shards(input_dir, output_dir, strategy="knn", k=k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Add edges to point-cloud shards (Step 2 of 2)")
    ap.add_argument("--input_dir", required=True,
                    help="Point-cloud shard directory (output of preprocess.py)")
    ap.add_argument("--output_dir", required=True,
                    help="Destination directory for graph shards")
    ap.add_argument("--strategy", default="knn",
                    choices=["knn", "laman", "unique", "fully_connected"])
    ap.add_argument("--k", type=int, default=16,
                    help="kNN neighbours, or unique-k k (Araz best: unique-6). "
                         "(Δη,Δφ) space (default: 16)")
    ap.add_argument("--max_shards", type=int, default=None,
                    help="Build only the first N shards (pilot subsets)")
    ap.add_argument("--edge_features", dest="edge_feats", default="none",
                    choices=["none", "linear", "log"],
                    help="Attach (θ,k_T,z) edge features for EdgeGraphAE "
                         "(log = Araz reference form, best AD; linear weaker)")
    ap.add_argument("--raw_edge_pt", action="store_false",
                    dest="normalize_edge_pt",
                    help="Compute k_T with raw input pt instead of pt/Σpt_jet "
                         "(closer to the reference repo pairwise function when "
                         "upstream pt is raw).")
    build_graph_shards(**vars(ap.parse_args()))
