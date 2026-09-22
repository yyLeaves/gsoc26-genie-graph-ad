"""
Evaluate a checkpoint with held-out background at the Kitchen Sink test ratio.

Default protocol:
    signal events = first 20,000 label-1 events in each signal directory
    background    = 17 x signal events from label-0 events in --bkg_dir
"""

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
from scripts.analyze_adj_latent_geometry import load_edge_graph_ae


def load_any(checkpoint: Path, device: torch.device):
    """Load genie-format or legacy edge_graph / raw-state checkpoints."""
    try:
        return load_model_and_spec(checkpoint, device)
    except (ValueError, RuntimeError, KeyError, TypeError):
        pass

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    from src.models.factory import ModelSpec
    from src.models.edge_graph_ae import EdgeGraphAE

    # Wrapped legacy bundle: {"model": {"spec", "state"}, ...} without format.
    if isinstance(payload, dict) and "model" in payload:
        model = load_edge_graph_ae(checkpoint, device)
        raw = dict(payload["model"]["spec"])
        if raw.get("type") in {"edgeae", "edge_ae"}:
            raw["type"] = "edge_graph"
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

    # Raw state_dict next to config.json (old EdgeAE runs).
    if isinstance(payload, dict) and any(
            k.startswith("encoder_blocks.") for k in payload):
        cfg_path = Path(checkpoint).parent / "config.json"
        if not cfg_path.exists():
            raise ValueError(
                f"raw state_dict checkpoint {checkpoint} needs sibling "
                f"config.json"
            )
        import json
        cfg = json.loads(cfg_path.read_text())
        m = cfg["model"]
        # Old EdgeAE configs omit feature_cols; datasets store 5 node
        # features and the model reads column 0 (log pT) only.
        feature_cols = (
            tuple(m["feature_cols"])
            if m.get("feature_cols") is not None
            else (0,)
        )
        model = EdgeGraphAE(
            in_dim=int(m["in_dim"]),
            edge_dim=int(m.get("edge_dim", 3)),
            hidden_dim=int(m.get("hidden_dim", 64)),
            latent_dim=int(m.get("latent_dim", 2)),
            edge_weight=float(m.get("edge_weight", 1.0)),
            aggr=str(m.get("aggr", "mean")),
            dropout=float(m.get("dropout", 0.0)),
            feature_cols=feature_cols,
        )
        model.load_state_dict(payload)
        model.to(device).eval()
        return model, ModelSpec(
            type="edge_graph",
            in_dim=int(m["in_dim"]),
            backbone=str(m.get("backbone", "edgeconv")),
            hidden_dim=int(m.get("hidden_dim", 64)),
            latent_dim=int(m.get("latent_dim", 2)),
            use_bn=False,
            edge_dim=int(m.get("edge_dim", 3)),
            edge_weight=float(m.get("edge_weight", 1.0)),
            aggr=str(m.get("aggr", "mean")),
            dropout=float(m.get("dropout", 0.0)),
            feature_cols=feature_cols,
        )

    raise ValueError(f"unrecognised checkpoint format: {checkpoint}")


def parse_signal_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name, path
    path = Path(spec)
    return path.name, str(path)


def event_indices(ds: JetDataset, label: int, max_events: int | None,
                  skip_events: int = 0):
    """Jet indices for the first ``max_events`` distinct events of ``label``."""
    labels = np.asarray(ds.labels, dtype=np.int64)
    event_ids = np.asarray(ds.event_ids, dtype=np.int64)
    selected_events = []
    seen = set()

    for event_id, y in zip(event_ids, labels):
        event_id = int(event_id)
        if int(y) != label or event_id in seen:
            continue
        seen.add(event_id)
        if skip_events > 0:
            skip_events -= 1
            continue
        selected_events.append(event_id)
        if max_events is not None and len(selected_events) >= max_events:
            break

    selected = np.array(selected_events, dtype=np.int64)
    if len(selected) == 0:
        return np.array([], dtype=np.int64), selected
    mask = (labels == label) & np.isin(event_ids, selected)
    return np.where(mask)[0].astype(np.int64), selected


def indices_for_event_ids(ds: JetDataset, label: int, event_ids):
    labels = np.asarray(ds.labels, dtype=np.int64)
    ds_event_ids = np.asarray(ds.event_ids, dtype=np.int64)
    requested = np.asarray(event_ids, dtype=np.int64)
    if requested.size == 0:
        return np.array([], dtype=np.int64), requested
    mask = np.isin(ds_event_ids, requested)
    found = np.unique(ds_event_ids[mask])
    missing = np.setdiff1d(requested, found)
    if missing.size:
        raise ValueError(
            f"{ds.shard_dir} is missing {missing.size:,} requested event ids; "
            f"examples={missing[:5].tolist()}"
        )
    wrong = np.unique(ds_event_ids[mask & (labels != label)])
    if wrong.size:
        raise ValueError(
            f"{ds.shard_dir} has {wrong.size:,} requested event ids with "
            f"label != {label}; examples={wrong[:5].tolist()}"
        )
    return np.where(mask & (labels == label))[0].astype(np.int64), requested


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, model_spec = load_any(Path(args.checkpoint), device)
    bkg = JetDataset(args.bkg_dir)
    ensure_dataset_matches(bkg, model_spec)

    max_bkg_events = args.n_signal * args.bkg_to_signal
    if args.split_manifest:
        if args.bkg_skip_events:
            raise ValueError(
                "Use either --split_manifest or --bkg_skip_events, not both.")
        with np.load(args.split_manifest) as manifest:
            if "test_bkg_events" not in manifest.files:
                raise ValueError(
                    f"{args.split_manifest} is missing test_bkg_events")
            bkg_idx, bkg_event_ids = indices_for_event_ids(
                bkg, label=0, event_ids=manifest["test_bkg_events"])
    else:
        bkg_idx, bkg_event_ids = event_indices(
            bkg, label=0, max_events=max_bkg_events,
            skip_events=args.bkg_skip_events)
    if len(bkg_event_ids) == 0:
        raise ValueError(f"No label-0 events found in {args.bkg_dir}")

    print(f"Model      : {model.__class__.__name__}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print(f"Background : {args.bkg_dir}")
    if args.split_manifest:
        print(f"Manifest   : {args.split_manifest}")
    else:
        print(f"Bkg skip   : {args.bkg_skip_events:,} label-0 events")
    print(f"Protocol   : n_signal={args.n_signal:,}, "
          f"bkg:signal={args.bkg_to_signal}:1")

    t0 = time.time()
    bkg_scored = score_events(
        model, bkg, device, batch_size=args.batch_size, indices=bkg_idx,
        aggregation=args.event_score_agg)
    if not np.all(bkg_scored.labels == 0):
        raise ValueError("Background selection produced non-background labels.")
    print(f"Scored background: {len(bkg_scored.scores):,} events "
          f"({time.time() - t0:.0f}s)")

    summary = {
        "checkpoint": str(args.checkpoint),
        "bkg_dir": str(args.bkg_dir),
        "event_score_aggregation": args.event_score_agg,
        "requested_n_signal": args.n_signal,
        "bkg_to_signal": args.bkg_to_signal,
        "split_manifest": args.split_manifest,
        "bkg_skip_events": args.bkg_skip_events,
        "allow_short_bkg": args.allow_short_bkg,
        "available_scored_background": int(len(bkg_scored.scores)),
        "signals": {},
    }

    for spec in args.signal:
        name, sig_path = parse_signal_spec(spec)
        sig = JetDataset(sig_path)
        ensure_dataset_matches(sig, model_spec)
        sig_idx, sig_event_ids = event_indices(
            sig, label=1, max_events=args.n_signal)
        if len(sig_event_ids) == 0:
            raise ValueError(f"No label-1 events found in {sig_path}")
        sig_scored = score_events(
            model, sig, device, batch_size=args.batch_size, indices=sig_idx,
            aggregation=args.event_score_agg)
        if not np.all(sig_scored.labels == 1):
            raise ValueError(
                f"Signal selection for {name} produced non-signal labels.")

        required_bkg = int(args.bkg_to_signal * len(sig_scored.scores))
        if len(bkg_scored.scores) < required_bkg and not args.allow_short_bkg:
            raise ValueError(
                f"{name}: need {required_bkg:,} scored background events for "
                f"{len(sig_scored.scores):,} signal events at "
                f"{args.bkg_to_signal}:1, but only "
                f"{len(bkg_scored.scores):,} are available. Pass "
                "--allow_short_bkg to evaluate with a lower effective ratio."
            )
        n_bkg = min(len(bkg_scored.scores), required_bkg)
        scores = np.concatenate([
            bkg_scored.scores[:n_bkg], sig_scored.scores])
        labels = np.concatenate([
            np.zeros(n_bkg, dtype=np.int64),
            np.ones(len(sig_scored.scores), dtype=np.int64),
        ])

        metrics = summarize_scores(scores, labels)
        metrics["classification_at_max_sic_threshold"] = classification_metrics(
            scores, labels, metrics["best_sic_threshold"])
        metrics["classification_at_best_f1_threshold"] = best_f1_metrics(
            scores, labels)
        metrics.update({
            "signal_dir": str(sig_path),
            "n_signal_requested": args.n_signal,
            "n_signal_scored": int(len(sig_scored.scores)),
            "n_background_scored": int(n_bkg),
            "effective_bkg_to_signal": float(n_bkg / len(sig_scored.scores)),
            "event_score_aggregation": args.event_score_agg,
            "background_event_id_min": int(bkg_scored.event_ids[:n_bkg].min()),
            "background_event_id_max": int(bkg_scored.event_ids[:n_bkg].max()),
            "signal_event_id_min": int(sig_scored.event_ids.min()),
            "signal_event_id_max": int(sig_scored.event_ids.max()),
        })

        sig_out = out / name
        sig_out.mkdir(parents=True, exist_ok=True)
        np.save(sig_out / "scores.npy", scores)
        np.save(sig_out / "labels.npy", labels)
        (sig_out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        summary["signals"][name] = metrics
        print(f"{name:16s} AUC={metrics['auc']:.4f}  "
              f"MaxSIC={metrics['max_sic']:.3f}  "
              f"thr={metrics['best_sic_threshold']:.6g}  "
              f"n_bkg={n_bkg:,}  n_sig={len(sig_scored.scores):,}  "
              f"F1@SIC={metrics['classification_at_max_sic_threshold']['f1']:.4f}")

    summary["elapsed_s"] = time.time() - t0
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved → {out}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Held-out-background eval at Kitchen Sink ratio.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bkg_dir", required=True)
    p.add_argument("--signal", action="append", required=True,
                   help="Signal graph dir, optionally name=dir. Repeatable.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_signal", type=int, default=20_000)
    p.add_argument("--bkg_to_signal", type=int, default=17)
    p.add_argument("--bkg_skip_events", type=int, default=0)
    p.add_argument("--split_manifest", default=None,
                   help="Use manifest test_bkg_events for background.")
    p.add_argument("--event_score_agg", default="sum",
                   choices=EVENT_SCORE_AGGREGATIONS)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--allow_short_bkg", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
