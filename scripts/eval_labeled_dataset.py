"""
Evaluate a checkpoint on one labeled graph dataset.

For LHCO R&D / labeled black-box files. Background and signal live in the same
shard directory. Optionally restrict scoring to a split-manifest role
(``--split_manifest`` + ``--split``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.checkpoint import checkpoint_cycle_mode
from src.data.dataset import JetDataset
from src.eval.metrics import (
    best_f1_metrics,
    classification_metrics,
    summarize_scores,
)
from src.eval.scoring import (ANOMALY_SCORE_MODES, EVENT_SCORE_AGGREGATIONS,
                              score_events)
from src.models import ensure_dataset_matches, load_model_and_spec
from scripts.analyze_adj_latent_geometry import load_edge_graph_ae


def load_any(checkpoint: Path, device: torch.device):
    try:
        return load_model_and_spec(checkpoint, device)
    except (ValueError, RuntimeError, KeyError, TypeError):
        model = load_edge_graph_ae(checkpoint, device)
        # Minimal spec for ensure_dataset_matches / logging.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw = dict(payload["model"]["spec"])
        if raw.get("type") in {"edgeae", "edge_ae"}:
            raw["type"] = "edge_graph"
        from src.models.factory import ModelSpec
        return model, ModelSpec(
            type=raw["type"],
            in_dim=int(raw["in_dim"]),
            backbone=str(raw.get("backbone", "edgeconv")),
            hidden_dim=int(raw.get("hidden_dim", 64)),
            latent_dim=int(raw.get("latent_dim", 2)),
            use_bn=bool(raw.get("use_bn", False)),
            edge_dim=int(raw.get("edge_dim", 3)),
            edge_weight=float(raw.get("edge_weight", 1.0)),
            aggr=str(raw.get("aggr", "mean")),
            dropout=float(raw.get("dropout", 0.0)),
            dyn_k=int(raw.get("dyn_k", 16)),
            feature_cols=tuple(raw["feature_cols"])
            if raw.get("feature_cols") is not None else None,
        )

_SPLIT_ROLES = {
    "train": ("train_bkg_events", "train_sig_events"),
    "val": ("val_bkg_events",),
    "monitor": ("val_bkg_events", "monitor_sig_events"),
    "test": ("test_bkg_events", "test_sig_events"),
}


def indices_from_manifest(ds: JetDataset, manifest_path: str,
                          split: str) -> np.ndarray:
    if split not in _SPLIT_ROLES:
        raise ValueError(
            f"--split must be one of {sorted(_SPLIT_ROLES)}, got {split!r}")
    with np.load(manifest_path) as manifest:
        missing = [k for k in _SPLIT_ROLES[split] if k not in manifest.files]
        if missing:
            raise ValueError(
                f"{manifest_path} is missing required arrays {missing}")
        event_ids = np.unique(np.concatenate([
            np.asarray(manifest[k], dtype=np.int64)
            for k in _SPLIT_ROLES[split]
        ]))
    ds_event_ids = np.asarray(ds.event_ids, dtype=np.int64)
    mask = np.isin(ds_event_ids, event_ids)
    found = np.unique(ds_event_ids[mask])
    missing_ids = np.setdiff1d(event_ids, found)
    if missing_ids.size:
        raise ValueError(
            f"{ds.shard_dir} is missing {missing_ids.size:,} events from "
            f"manifest {manifest_path}; examples={missing_ids[:5].tolist()}")
    return np.where(mask)[0].astype(np.int64)


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, model_spec = load_any(Path(args.checkpoint), device)
    global_mode = checkpoint_cycle_mode(args.checkpoint)
    ds = JetDataset(args.data_dir, max_cache=args.cache_shards)
    ensure_dataset_matches(ds, model_spec)

    if args.split_manifest:
        indices = indices_from_manifest(ds, args.split_manifest, args.split)
    else:
        if args.split != "all":
            raise ValueError(
                "--split without --split_manifest is only valid as 'all'")
        indices = np.arange(len(ds), dtype=np.int64)

    print(f"Dataset    : {args.data_dir}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print(f"Jets       : {len(indices):,} / {len(ds):,}")
    if args.split_manifest:
        print(f"Manifest   : {args.split_manifest}  split={args.split}")
    print(f"Aggregation: {args.event_score_agg}")
    print(f"Jet score  : {args.anomaly_score}")
    if args.anomaly_score != "reconstruction":
        print(f"Global cycle: {global_mode}")

    t0 = time.time()
    scored = score_events(
        model, ds, device, batch_size=args.batch_size, indices=indices,
        aggregation=args.event_score_agg, score_mode=args.anomaly_score,
        cycle_global_mode=global_mode)
    metrics = summarize_scores(scored.scores, scored.labels)
    metrics["classification_at_max_sic_threshold"] = classification_metrics(
        scored.scores, scored.labels, metrics["best_sic_threshold"])
    metrics["classification_at_best_f1_threshold"] = best_f1_metrics(
        scored.scores, scored.labels)
    metrics.update({
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "split_manifest": args.split_manifest,
        "split": args.split,
        "event_score_aggregation": args.event_score_agg,
        "anomaly_score": args.anomaly_score,
        "cycle_global_mode": global_mode,
        "n_jets_scored": int(len(indices)),
        "n_events_scored": int(len(scored.scores)),
        "dataset_meta": {
            "n_input_events": int(ds.meta.get(
                "n_input_events", ds.meta.get("n_events", -1))),
            "n_selected_events": int(ds.meta.get("n_selected_events", -1)),
            "jet_selection": ds.meta.get("jet_selection"),
            "require_two_jets": bool(ds.meta.get("require_two_jets", False)),
            "min_jet_pt": float(ds.meta.get("min_jet_pt", float("nan"))),
            "n_subjets": ds.meta.get("n_subjets"),
            "strategy": ds.meta.get("strategy"),
            "k": ds.meta.get("k"),
            "edge_feats": ds.meta.get("edge_feats"),
            "edge_pt_scale": ds.meta.get("edge_pt_scale"),
        },
        "elapsed_s": time.time() - t0,
    })

    np.save(out / "scores.npy", scored.scores)
    np.save(out / "labels.npy", scored.labels)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"AUC={metrics['auc']:.6f}  "
          f"MaxSIC={metrics['max_sic']:.6f}  "
          f"thr={metrics['best_sic_threshold']:.6g}  "
          f"n_bkg={metrics['n_background']:,}  "
          f"n_sig={metrics['n_signal']:,}")
    print(f"Saved → {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate one labeled graph dataset at event level.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split_manifest", default=None,
                   help="Optional event-level .npz; use with --split.")
    p.add_argument("--split", default="all",
                   choices=["all", "train", "val", "monitor", "test"],
                   help="Manifest role to score (default: all jets).")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--cache_shards", type=int, default=4)
    p.add_argument("--event_score_agg", default="sum",
                   choices=EVENT_SCORE_AGGREGATIONS)
    p.add_argument("--anomaly_score", default="reconstruction",
                   choices=ANOMALY_SCORE_MODES)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
