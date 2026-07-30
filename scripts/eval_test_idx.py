"""Score one checkpoint on split.npz ``test_idx`` with ``score_events``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import JetDataset
from src.eval.metrics import (
    best_f1_metrics,
    classification_metrics,
    summarize_scores,
)
from src.eval.scoring import EVENT_SCORE_AGGREGATIONS, score_events
from src.models import ensure_dataset_matches, load_model_and_spec
from scripts.latent_tda_probe import load_model_legacy


def load_any(checkpoint: Path, device: torch.device):
    try:
        return load_model_and_spec(checkpoint, device)
    except (ValueError, RuntimeError, KeyError):
        model, spec, _ = load_model_legacy(checkpoint, device)
        return model, spec


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, spec = load_any(Path(args.checkpoint), device)
    ds = JetDataset(args.data_dir, max_cache=args.cache_shards)
    ensure_dataset_matches(ds, spec)

    with np.load(args.split) as split:
        indices = np.asarray(split["test_idx"], dtype=np.int64)

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Data       : {args.data_dir}")
    print(f"Split      : {args.split}  test_idx={len(indices):,}")
    print(f"Device     : {device}")
    print(f"Aggregation: {args.event_score_agg}")

    t0 = time.time()
    scored = score_events(
        model, ds, device, batch_size=args.batch_size, indices=indices,
        aggregation=args.event_score_agg)
    elapsed = time.time() - t0
    metrics = summarize_scores(scored.scores, scored.labels)
    metrics["classification_at_max_sic_threshold"] = classification_metrics(
        scored.scores, scored.labels, metrics["best_sic_threshold"])
    metrics["classification_at_best_f1_threshold"] = best_f1_metrics(
        scored.scores, scored.labels)
    metrics.update({
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "split": str(args.split),
        "split_role": "test_idx",
        "n_scored_jets": int(len(indices)),
        "n_scored_events": int(len(scored.labels)),
        "event_score_aggregation": args.event_score_agg,
        "elapsed_s": elapsed,
        "model_type": spec.type,
    })

    np.save(out / "scores.npy", scored.scores)
    np.save(out / "labels.npy", scored.labels)
    np.save(out / "event_ids.npy", scored.event_ids)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"AUC={metrics['auc']:.4f}  MaxSIC={metrics['max_sic']:.3f}  "
        f"eS@1e-2={metrics['eS_at_eB1e-2']:.4f}  "
        f"eS@1e-3={metrics['eS_at_eB1e-3']:.4f}  "
        f"n_events={metrics['n_scored_events']:,}  ({elapsed:.1f}s)"
    )
    print(f"Wrote {out / 'metrics.json'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", required=True, help="Path to split.npz")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--cache_shards", type=int, default=32)
    p.add_argument("--event_score_agg", default="sum",
                   choices=EVENT_SCORE_AGGREGATIONS)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
