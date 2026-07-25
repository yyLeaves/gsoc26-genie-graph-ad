import argparse
import time
from pathlib import Path
from typing import Iterator

from . import shards
from .events import EventReader
from .extractor import JET_RADIUS, JET_SELECTIONS, JetConfig, JetExtractor
from .features import FEATURE_DIM

__all__ = ["preprocess_file"]


def _with_progress(chunks: Iterator, n_events: int) -> Iterator:
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


def _build_metadata(
    shard_stats: dict,
    reader: EventReader,
    config: JetConfig,
) -> dict:
    """Describe the source events, jet selection, and written shards."""
    h5_path = Path(reader.h5_path)
    h5_stat = h5_path.stat()
    labels_path = Path(reader.labels_path) if reader.labels_path else None
    return {
        **shard_stats,
        "schema_version": shards.DATASET_SCHEMA_VERSION,
        "source": {
            "h5_path": str(h5_path.resolve()),
            "h5_key": reader.key,
            "h5_size_bytes": h5_stat.st_size,
            "h5_mtime_ns": h5_stat.st_mtime_ns,
            "n_events": reader.n_events,
            "labels_path": reader.labels_path,
            "labels_resolved_path": (
                str(labels_path.resolve()) if labels_path is not None else None
            ),
            "labels_size_bytes": (
                labels_path.stat().st_size if labels_path is not None else None
            ),
            "label_mode": (
                "external_binary_gt0"
                if reader.labels_path is not None
                else "inline_binary_gt0"
            ),
        },
        "selection": {
            "algorithm": "anti_kt",
            "radius": JET_RADIUS,
            "jet_selection": config.jet_selection,
            "min_jet_pt": config.min_jet_pt,
            "min_particles": config.min_particles,
            "require_two_jets": config.require_two_jets,
        },
        "nodes": {
            "representation": "constituents",
            "features": config.features,
            "feature_dim": FEATURE_DIM[config.features],
        },
        "edges": None,
    }


def _print_summary(metadata: dict, elapsed_seconds: float) -> None:
    labels = metadata["labels"]
    node_counts = metadata["n_nodes"]
    print(
        f"\nDone in {elapsed_seconds:.0f}s  "
        f"({elapsed_seconds / 60:.1f} min)"
    )
    print(
        f"  Shards    : {metadata['n_shards']}  x  "
        f"shard_size={metadata['shard_size']}"
    )
    print(f"  Events    : selected={metadata['n_events']:,}  "
          f"input={metadata['source']['n_events']:,}")
    print(f"  Jets      : {metadata['n_jets']:,}  (signal={labels.sum():,}  "
          f"bkg={(labels==0).sum():,})")
    print(
        f"  Nodes/jet : mean={node_counts.mean():.1f}  max={node_counts.max()}"
    )


def _validate_shard_size(shard_size: int) -> None:
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")


def preprocess_file(
    h5_path: str | Path,
    output_dir: str | Path,
    *,
    config: JetConfig,
    labels_path: str | Path | None = None,
    shard_size: int = 8192,
) -> None:
    _validate_shard_size(shard_size)

    h5_path = str(h5_path)
    output_dir = str(output_dir)
    labels_path = str(labels_path) if labels_path else None
    reader = EventReader(h5_path, labels_path=labels_path)
    extractor = JetExtractor(config)

    print(f"Input    : {h5_path}  ({reader.n_events:,} events, "
          f"key={reader.key!r})")
    if labels_path:
        print(f"Labels   : {labels_path}  (binary: label = raw > 0)")
    print(f"Output   : {output_dir}")
    print(f"Features : {config.features} "
          f"(dim={FEATURE_DIM[config.features]})  "
          f"min_jet_pt={config.min_jet_pt} GeV  "
          f"min_particles={config.min_particles}"
          + f"  jet_selection={config.jet_selection}"
          + ("  require_two_jets" if config.require_two_jets else ""))

    t0 = time.time()
    chunks = _with_progress(reader.chunks(), reader.n_events)
    shard_stats = shards.write_shards(
        extractor.from_chunks(chunks), output_dir, shard_size
    )
    if shard_stats["n_jets"] == 0:
        raise ValueError("No jets passed preprocessing selection")
    metadata = _build_metadata(shard_stats, reader, config)
    shards.save_metadata(metadata, output_dir)
    _print_summary(metadata, time.time() - t0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build selected-jet constituent shards from LHCO HDF5"
    )
    parser.add_argument("--h5_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--labels_path",
        default=None,
        help=(
            "Optional one-label-per-line masterkey/truth file. Positive "
            "labels are mapped to binary signal=1."
        ),
    )
    parser.add_argument(
        "--features", required=True, choices=sorted(FEATURE_DIM)
    )
    parser.add_argument("--min_jet_pt", type=float, required=True)
    parser.add_argument("--min_particles", type=int, required=True)
    parser.add_argument("--shard_size", type=int, default=8192)
    parser.add_argument(
        "--jet_selection",
        required=True,
        choices=JET_SELECTIONS,
        help=(
            "min_pt_all: all returned jets must pass min_jet_pt "
            "(reference-paper-like). leading_pt: require only the leading "
            "jet to pass min_jet_pt, then keep top-2 jets (Kitchen Sink style)."
        ),
    )
    event_selection = parser.add_mutually_exclusive_group(required=True)
    event_selection.add_argument(
        "--require_two_jets",
        dest="require_two_jets",
        action="store_true",
        help=(
            "Drop events unless two selected jets survive min_particles. "
            "Use for strict event-level dijet evaluation."
        ),
    )
    event_selection.add_argument(
        "--allow_single_jet",
        dest="require_two_jets",
        action="store_false",
        help=(
            "Keep events with one or two usable selected jets. "
            "When two jets were selected, mjj is still their dijet mass "
            "even if only one survives min_particles."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess_file(
        h5_path=args.h5_path,
        output_dir=args.output_dir,
        config=JetConfig(
            features=args.features,
            min_jet_pt=args.min_jet_pt,
            min_particles=args.min_particles,
            jet_selection=args.jet_selection,
            require_two_jets=args.require_two_jets,
        ),
        labels_path=args.labels_path,
        shard_size=args.shard_size,
    )
