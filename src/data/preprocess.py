"""
HDF5 → sharded point-cloud .pt files (no edges).

Composes EventReader + JetExtractor:

    reader.chunks() → extractor.from_chunks(...) → shards.write_shards(...)

Works on any LHCO-layout file (key auto-detected).

Output per Data object:
    x          (N, d)   node features  (d=5 for log_phys)
    pos        (N, 2)   (Δη, Δφ) relative to pT-weighted jet axis
    y          (1,)     label  0=background  1=signal
    event_id   (1,)     original event ordinal inside this preprocessing stream
    jet_idx    (1,)     0=leading jet  1=subleading jet
    mjj        (1,)     dijet invariant mass in GeV  (0 if <2 jets)

Usage:
    python -m src.data.preprocess \\
        --h5_path    develop/dataset/lhco/events_anomalydetection.h5 \\
        --output_dir develop/dataset/processed/lhco_jets
"""

import argparse
import time
from typing import Iterator

from . import shards
from .events import EventReader
from .extractor import JET_RADIUS, JetConfig, JetExtractor
from .graph import FEATURE_DIM

__all__ = ["preprocess_file", "make_metadata", "with_progress"]


def with_progress(chunks: Iterator, n_events: int) -> Iterator:
    """Pass-through stage: report throughput per chunk."""
    t0 = time.time()
    done = 0
    for labels, data in chunks:
        yield labels, data
        done += len(labels)
        elapsed = time.time() - t0
        print(f"  {done:>9,}/{n_events:,}  elapsed={elapsed:.0f}s  "
              f"eta={elapsed / done * (n_events - done) / 60:.1f}min",
              flush=True)


def make_metadata(stats: dict, cfg: JetConfig, n_events: int) -> dict:
    """Merge shard-writer stats + run config into metadata."""
    return {
        **stats,
        "n_events": n_events,
        "features": cfg.features,
        "feature_dim": FEATURE_DIM[cfg.features],
        "min_particles": cfg.min_particles,
        "min_jet_pt": cfg.min_jet_pt,
        "jet_radius": JET_RADIUS,
        "n_subjets": cfg.n_subjets,
        "has_edges": False,
    }


def print_summary(meta: dict, seconds: float) -> None:
    labels, n_nodes = meta["labels"], meta["n_nodes"]
    print(f"\nDone in {seconds:.0f}s  ({seconds/60:.1f} min)")
    print(f"  Shards    : {meta['num_shards']}  x  shard_size={meta['shard_size']}")
    print(f"  Jets      : {meta['n']:,}  (signal={labels.sum():,}  "
          f"bkg={(labels==0).sum():,})")
    print(f"  Nodes/jet : mean={n_nodes.mean():.1f}  max={n_nodes.max()}")


def preprocess_file(
    h5_path: str,
    output_dir: str,
    features: str = "log_phys",
    min_jet_pt: float = 200.0,
    min_particles: int = 3,
    max_nodes: int = None,
    shard_size: int = 8192,
    n_subjets: int = None,
) -> None:
    reader = EventReader(h5_path)
    extractor = JetExtractor(JetConfig(features, min_jet_pt, min_particles,
                                       max_nodes, n_subjets))

    print(f"Input    : {h5_path}  ({reader.n_events:,} events, "
          f"key={reader.key!r})")
    print(f"Output   : {output_dir}")
    print(f"Features : {features} (dim={FEATURE_DIM[features]})  "
          f"min_jet_pt={min_jet_pt} GeV  min_particles={min_particles}"
          + (f"  n_subjets={n_subjets}" if n_subjets else ""))

    t0 = time.time()
    chunks = with_progress(reader.chunks(), reader.n_events)
    stats = shards.write_shards(extractor.from_chunks(chunks), output_dir,
                                shard_size)
    meta = make_metadata(stats, extractor.cfg, reader.n_events)
    shards.save_metadata(meta, output_dir)
    print_summary(meta, time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="LHCO-layout HDF5 → point-cloud shards")
    ap.add_argument("--h5_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--features",
                    default="log_phys",
                    choices=["log_phys", "raw", "normalized"])
    ap.add_argument("--min_jet_pt", type=float, default=200.0)
    ap.add_argument("--min_particles", type=int, default=3)
    ap.add_argument("--max_nodes", type=int, default=None)
    ap.add_argument("--shard_size", type=int, default=8192)
    ap.add_argument("--n_subjets", type=int, default=None,
                    help="Recluster each jet's constituents to this many "
                         "exclusive-kT subjets (Araz et al.; best ~30). "
                         "Fixes node count → removes the multiplicity confound.")
    preprocess_file(**vars(ap.parse_args()))
