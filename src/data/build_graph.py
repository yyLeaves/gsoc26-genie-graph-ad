import argparse
import copy
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from . import shards
from .graph import (
    EDGE_FEATURE_DIM,
    EDGE_FEATURE_MODES,
    EDGE_PT_SCALES,
    GRAPH_STRATEGIES,
    STRATEGIES_WITH_K,
    STRATEGIES_WITH_RADIUS,
    build_edges,
    edge_features,
)

__all__ = ["GraphConfig", "with_edges", "build_graph_shards"]


@dataclass(frozen=True, kw_only=True)
class GraphConfig:
    """Topology and edge-feature specification.

    ``k`` / ``radius`` are set only when the chosen strategy uses them.
    """

    strategy: str
    edge_features: str
    edge_pt_scale: str
    k: int | None = None
    radius: float | None = None

    def __post_init__(self) -> None:
        for name, choices in (
            ("strategy", GRAPH_STRATEGIES),
            ("edge_features", EDGE_FEATURE_MODES),
            ("edge_pt_scale", EDGE_PT_SCALES),
        ):
            value = getattr(self, name)
            if value not in choices:
                raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")

        needs_k = self.strategy in STRATEGIES_WITH_K
        if needs_k and (self.k is None or self.k < 1):
            raise ValueError(
                f"strategy={self.strategy!r} requires k >= 1, got {self.k}"
            )
        if not needs_k and self.k is not None:
            raise ValueError(
                f"strategy={self.strategy!r} does not use k; got k={self.k}"
            )

        needs_radius = self.strategy in STRATEGIES_WITH_RADIUS
        if needs_radius and (self.radius is None or self.radius <= 0):
            raise ValueError(
                f"strategy={self.strategy!r} requires radius > 0, "
                f"got {self.radius}"
            )
        if not needs_radius and self.radius is not None:
            raise ValueError(
                f"strategy={self.strategy!r} does not use radius; "
                f"got radius={self.radius}"
            )


def _build_output_metadata(
    input_meta: dict,
    shard_stats: dict,
    config: GraphConfig,
) -> dict:
    """Merge preserved input metadata with the written graph topology."""
    edge_metadata = {
        "strategy": config.strategy,
        "features": config.edge_features,
        "pt_scale": config.edge_pt_scale,
        "feature_dim": 0 if config.edge_features == "none" else EDGE_FEATURE_DIM,
    }
    if config.k is not None:
        edge_metadata["k"] = config.k
    if config.radius is not None:
        edge_metadata["radius"] = float(config.radius)

    return {
        **input_meta,
        **shard_stats,
        "edges": edge_metadata,
    }


def _print_summary(
    meta: dict,
    elapsed: float,
    output_dir: Path,
    max_in_degree: int,
) -> None:
    labels = meta["labels"]
    node_counts = meta["n_nodes"]
    edge_counts = meta["n_edges"]
    size_gb = sum(path.stat().st_size for path in output_dir.glob("*.pt")) / 1e9

    print(f"\nDone in {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  Shards     : {meta['n_shards']}  "
          f"x  shard_size={meta['shard_size']}")
    print(f"  Jets       : {meta['n_jets']:,}  (signal={labels.sum():,}  "
          f"bkg={(labels == 0).sum():,})")
    print(f"  Nodes/jet  : mean={node_counts.mean():.1f}  "
          f"max={node_counts.max()}")
    print(f"  Edges/jet  : mean={edge_counts.mean():.1f}  "
          f"max={edge_counts.max()}  ({meta['edges']['strategy']})")
    print(f"  Edges/node : mean="
          f"{edge_counts.sum(dtype=np.int64) / node_counts.sum(dtype=np.int64):.2f}  "
          f"(= mean in-degree)")
    print(f"  In-degree  : max={max_in_degree}")
    print(f"  Disk       : {size_gb:.1f} GB")
    print(f"  Saved to   : {output_dir}/")


def with_edges(source: Data, config: GraphConfig) -> Data:
    """Replace edges on a shallow copy; leave nodes and event fields unchanged."""
    if not isinstance(source.pos, torch.Tensor):
        raise ValueError("graph requires a pos tensor to build edges")
    positions = source.pos.numpy()
    graph = copy.copy(source)
    graph.edge_index = build_edges(
        positions,
        config.strategy,
        config.k if config.k is not None else 0,
        radius=config.radius if config.radius is not None else 0.0,
    )
    graph.edge_attr = None
    if config.edge_features != "none":
        if getattr(source, "pt", None) is None:
            raise ValueError("graph requires raw pt to compute edge features")
        graph.edge_attr = edge_features(
            positions, source.pt.numpy(), graph.edge_index,
            log=config.edge_features == "log",
            normalize_pt=config.edge_pt_scale == "normalized",
        )
    return graph


def build_graph_shards(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    config: GraphConfig,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    meta = shards.load_metadata(input_dir)
    n_shards = meta["n_shards"]

    print(f"Input     : {input_dir}  ({meta['n_jets']:,} jets, "
          f"{n_shards} shards)")
    print(f"Output    : {output_dir}")
    print(f"Strategy  : {config.strategy}"
          + (f"  k={config.k}" if config.k is not None else "")
          + (f"  r={config.radius:g}" if config.radius is not None else ""))
    print(f"Features  : {meta['nodes']['features']}  "
          f"dim={meta['nodes']['feature_dim']}")
    print(f"Edge feat : {config.edge_features}  "
          f"pt_scale={config.edge_pt_scale}")

    writer = shards.ShardWriter(output_dir, meta["shard_size"])
    max_in_degree = 0
    t0 = time.time()

    for si in range(n_shards):
        for source in shards.load_shard(input_dir, si):
            graph = with_edges(source, config)
            writer.add(graph)
            if graph.edge_index.shape[1]:
                graph_max = int(torch.bincount(graph.edge_index[1]).max())
                max_in_degree = max(max_in_degree, graph_max)

        elapsed = time.time() - t0
        eta_min = elapsed / (si + 1) * (n_shards - si - 1) / 60
        print(f"  Shard {si+1:>3}/{n_shards}  jets={writer.n_jets:,}"
              f"  {elapsed:.0f}s  eta={eta_min:.1f}min", flush=True)

    shard_stats = writer.finish()
    output_meta = _build_output_metadata(meta, shard_stats, config)
    shards.save_metadata(output_meta, output_dir)
    _print_summary(output_meta, time.time() - t0, output_dir,
                   max_in_degree)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Add edges to point-cloud shards "
            "(after preprocess.py, optionally build_subjets.py)"
        ))
    ap.add_argument("--input_dir", required=True,
                    help="Point-cloud shard directory")
    ap.add_argument("--output_dir", required=True,
                    help="Destination directory for graph shards")
    ap.add_argument("--strategy", required=True,
                    choices=GRAPH_STRATEGIES)
    ap.add_argument("--k", type=int,
                    help="kNN neighbours, or unique-k k (Araz best: unique-6). "
                         "(Δη,Δφ) space")
    ap.add_argument("--radius", type=float,
                    help="Radius threshold in ΔR for radius strategies")
    ap.add_argument("--edge_features", required=True,
                    choices=EDGE_FEATURE_MODES,
                    help="Attach (θ,k_T,z) edge features for EdgeGraphAE "
                         "(log = Araz reference form, best AD; linear weaker)")
    ap.add_argument(
        "--edge_pt_scale", required=True, choices=EDGE_PT_SCALES,
        help="pT used for k_T: normalized = pt/Σpt_jet, raw = stored pt",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_graph_shards(
        args.input_dir,
        args.output_dir,
        config=GraphConfig(
            strategy=args.strategy,
            k=args.k,
            radius=args.radius,
            edge_features=args.edge_features,
            edge_pt_scale=args.edge_pt_scale,
        ),
    )
