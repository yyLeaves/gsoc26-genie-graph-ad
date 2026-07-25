from __future__ import annotations

import argparse
import time
from pathlib import Path

import fastjet
import numpy as np

from . import shards
from .extractor import JET_RADIUS, jet_to_data
from .features import FEATURE_DIM
from .kinematics import Jet, p4_components

__all__ = ["build_subjet_shards"]

_SUBJET_DEFINITION = fastjet.JetDefinition(fastjet.kt_algorithm, JET_RADIUS)


def _recluster_subjets(jet: Jet, n_subjets: int) -> Jet:
    """Exclusive-kT recluster into at most ``n_subjets`` E-scheme nodes."""
    pt, eta, phi = jet
    if len(pt) <= n_subjets:
        return jet

    px, py, pz, energy = p4_components(pt, eta, phi)
    particles = [
        fastjet.PseudoJet(float(x), float(y), float(z), float(e))
        for x, y, z, e in zip(px, py, pz, energy)
    ]
    subjets = fastjet.ClusterSequence(
        particles,
        _SUBJET_DEFINITION,
    ).exclusive_jets(n_subjets)
    subjet_pt = np.asarray([subjet.pt() for subjet in subjets])
    subjet_eta = np.asarray([subjet.eta() for subjet in subjets])
    subjet_phi = np.asarray([subjet.phi_std() for subjet in subjets])
    descending_pt = np.argsort(-subjet_pt)
    return (
        subjet_pt[descending_pt],
        subjet_eta[descending_pt],
        subjet_phi[descending_pt],
    )


def _to_subjet_graph(graph, n_subjets: int, features: str):
    """Recluster one selected jet and preserve its event-level fields."""
    missing = [
        name
        for name in (
            "pt", "eta", "phi", "y", "event_id", "jet_idx", "mjj"
        )
        if getattr(graph, name, None) is None
    ]
    if missing:
        raise ValueError(
            f"Input graph is missing required fields {missing}. Rebuild "
            "selected-jet preprocessing with the current src.data.extractor."
        )

    constituents = (
        graph.pt.detach().cpu().numpy(),
        graph.eta.detach().cpu().numpy(),
        graph.phi.detach().cpu().numpy(),
    )
    subjets = _recluster_subjets(constituents, n_subjets)
    return jet_to_data(
        subjets,
        label=int(graph.y),
        jet_idx=int(graph.jet_idx),
        mjj=float(graph.mjj),
        features=features,
        event_id=int(graph.event_id),
    )


def _build_output_metadata(
    input_metadata: dict,
    shard_stats: dict,
    n_subjets: int,
    features: str,
) -> dict:
    """Merge preserved preprocessing settings with written subjet statistics."""
    return {
        **{key: value for key, value in input_metadata.items()
           if key != "n_edges"},
        **shard_stats,
        "nodes": {
            "representation": "exclusive_kt_subjets",
            "features": features,
            "feature_dim": FEATURE_DIM[features],
            "max_nodes": n_subjets,
            "padding": False,
        },
        "edges": None,
    }


def _print_summary(metadata: dict, elapsed: float, output_dir: Path) -> None:
    labels = metadata["labels"]
    node_counts = metadata["n_nodes"]
    n_subjets = metadata["nodes"]["max_nodes"]
    size_gb = sum(path.stat().st_size for path in output_dir.glob("*.pt")) / 1e9

    print(f"\nDone in {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  Jets       : {metadata['n_jets']:,}  (signal={labels.sum():,}  "
          f"bkg={(labels == 0).sum():,})")
    print(f"  Nodes/jet  : mean={node_counts.mean():.1f}  "
          f"max={node_counts.max()}")
    print(f"  At cap     : {(node_counts == n_subjets).sum():,}/{len(node_counts):,} "
          f"({(node_counts == n_subjets).mean():.1%})")
    print(f"  Disk       : {size_gb:.1f} GB")
    print(f"  Saved to   : {output_dir}/")


def build_subjet_shards(
    input_dir: str | Path,
    output_dir: str | Path,
    n_subjets: int = 30,
    features: str | None = None,
) -> None:
    """Convert selected-jet shards to capped exclusive-kT subjet shards."""
    if n_subjets <= 0:
        raise ValueError(f"n_subjets must be positive, got {n_subjets}")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_metadata = shards.load_metadata(input_dir)
    if (
        input_metadata.get("n_jets", 0) <= 0
        or input_metadata.get("n_shards", 0) <= 0
    ):
        raise ValueError(f"Input dataset contains no jets: {input_dir}")
    representation = input_metadata["nodes"].get("representation")
    if representation != "constituents":
        raise ValueError(
            "build_subjets expects selected-jet constituent shards "
            f"(nodes.representation='constituents'), got {representation!r}. "
            "Do not recluster exclusive-kT subjet datasets."
        )
    if features is None:
        features = input_metadata["nodes"]["features"]
    if features not in FEATURE_DIM:
        raise ValueError(
            f"features must be one of {sorted(FEATURE_DIM)}, got {features!r}")

    print(f"Input     : {input_dir}  ({input_metadata['n_jets']:,} jets, "
          f"{input_metadata['n_shards']} shards)")
    print(f"Output    : {output_dir}")
    print(f"Subjets   : max={n_subjets}  features={features}  padding=none")
    writer = shards.ShardWriter(output_dir, input_metadata["shard_size"])
    t0 = time.time()
    for shard_index in range(input_metadata["n_shards"]):
        subjet_graphs = [
            _to_subjet_graph(graph, n_subjets=n_subjets, features=features)
            for graph in shards.load_shard(input_dir, shard_index)
        ]
        writer.extend(subjet_graphs)

        elapsed = time.time() - t0
        shards_done = shard_index + 1
        eta_min = (
            elapsed / shards_done
            * (input_metadata["n_shards"] - shards_done)
            / 60
        )
        print(f"  Shard {shards_done:>3}/{input_metadata['n_shards']}  "
              f"jets={writer.n_jets:,}"
              f"  {elapsed:.0f}s  eta={eta_min:.1f}min", flush=True)

    shard_stats = writer.finish()
    output_metadata = _build_output_metadata(
        input_metadata,
        shard_stats,
        n_subjets,
        features,
    )
    shards.save_metadata(output_metadata, output_dir)
    _print_summary(output_metadata, time.time() - t0, output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selected-jet shards → capped-subjet point-cloud shards."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_subjets", type=int, default=30,
                        help="Maximum exclusive-kT subjets per jet; jets with "
                             "fewer constituents are not padded.")
    parser.add_argument("--features", default=None, choices=sorted(FEATURE_DIM))
    return parser.parse_args()


if __name__ == "__main__":
    build_subjet_shards(**vars(_parse_args()))
