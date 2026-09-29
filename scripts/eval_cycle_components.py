"""Evaluate reconstruction and latent-cycle scores in one dataset pass."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from scripts.eval_labeled_dataset import indices_from_manifest, load_any
from src.checkpoint import checkpoint_cycle_mode
from src.data.dataset import JetDataset
from src.data.iterate import prefetch, shard_iter
from src.eval.metrics import summarize_scores
from src.eval.scoring import aggregate_event_scores
from src.models import EdgeGraphAE, ModelSpec, ensure_dataset_matches
from src.models.reconstruction import reconstruction_scores
from src.objectives.latent_cycle import latent_cycle_scores


@torch.no_grad()
def score_components(model, dataset, indices, device, batch_size, name, *, global_mode='pooled'):
    values = {name: [] for name in (
        "reconstruction", "node_mse", "edge_mse", "cycle", "cycle_local", "cycle_global")}
    event_ids, labels = [], []
    seen = 0
    next_report = 100_000
    model.eval()
    for batch in prefetch(shard_iter(
            dataset, indices, batch_size, shuffle_shards=False,
            shuffle_within=False, device=device), depth=2):
        output, node_target, edge_target = model._reconstruct(batch)
        reconstruction = reconstruction_scores(
            output, node_target, batch, edge_target=edge_target,
            edge_weight=getattr(model, "edge_weight", 1.0))
        cycle = latent_cycle_scores(model, output, batch, global_mode=global_mode)
        values["reconstruction"].append(reconstruction.total.cpu())
        values["node_mse"].append(reconstruction.node.cpu())
        values["edge_mse"].append(reconstruction.edge.cpu())
        values["cycle"].append(cycle.total.cpu())
        values["cycle_local"].append(cycle.local.cpu())
        values["cycle_global"].append(cycle.global_.cpu())
        event_ids.append(batch.event_id.view(-1).cpu())
        labels.append(batch.y.view(-1).cpu())
        seen += batch.num_graphs
        if seen >= next_report or seen == len(indices):
            print(f"  {name}: {seen:,}/{len(indices):,} jets", flush=True)
            next_report += 100_000

    ids = torch.cat(event_ids).numpy()
    truth = torch.cat(labels).numpy()
    events = {}
    for name, chunks in values.items():
        events[name] = aggregate_event_scores(
            torch.cat(chunks).numpy(), ids, truth, aggregation="sum")
    reference = events["reconstruction"]
    for scored in events.values():
        if not (np.array_equal(scored.event_ids, reference.event_ids)
                and np.array_equal(scored.labels, reference.labels)):
            raise RuntimeError("cycle component event aggregation misaligned")
    return events


def percentile(calibration: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(calibration, dtype=np.float64))
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def load_cycle_model(checkpoint: Path, device: torch.device):
    """Load current checkpoints and early raw-state EdgeGraphAE checkpoints."""
    try:
        return load_any(checkpoint, device)
    except ValueError:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if not isinstance(state, dict) or not state or not all(
                isinstance(key, str) for key in state):
            raise
        config_path = checkpoint.parent / "config.json"
        if not config_path.exists():
            raise ValueError(
                f"raw state_dict checkpoint requires sibling {config_path.name}")
        config = json.loads(config_path.read_text())
        raw = config["model"]
        spec = ModelSpec(
            type="edge_graph",
            in_dim=int(raw["in_dim"]),
            backbone="edgeconv",
            hidden_dim=int(raw.get("hidden_dim", 64)),
            latent_dim=int(raw.get("latent_dim", 2)),
            use_bn=False,
            edge_dim=int(raw.get("edge_dim", 3)),
            edge_weight=float(raw.get("edge_weight", 1.0)),
            aggr=str(raw.get("aggr", "mean")),
            dropout=float(raw.get("dropout", 0.0)),
            feature_cols=tuple(raw["feature_cols"])
            if raw.get("feature_cols") is not None else None,
        )
        model = EdgeGraphAE(
            in_dim=spec.in_dim, edge_dim=spec.edge_dim,
            hidden_dim=spec.hidden_dim, latent_dim=spec.latent_dim,
            edge_weight=spec.edge_weight, aggr=spec.aggr,
            dropout=spec.dropout, feature_cols=spec.feature_cols)
        model.load_state_dict(state)
        return model.to(device), spec


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model, spec = load_cycle_model(Path(args.checkpoint), device)
    global_mode = checkpoint_cycle_mode(Path(args.checkpoint))
    if spec.type != "edge_graph":
        raise ValueError("cycle-component evaluation requires EdgeGraphAE")
    dataset = JetDataset(args.data_dir, max_cache=args.cache_shards)
    ensure_dataset_matches(dataset, spec)
    calibration_idx = indices_from_manifest(
        dataset, args.split_manifest, "val")
    test_idx = indices_from_manifest(dataset, args.split_manifest, "test")

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Dataset    : {args.data_dir}")
    print(f"Device     : {device}")
    print(f"Calibration: {len(calibration_idx):,} jets")
    print(f"Test       : {len(test_idx):,} jets")
    started = time.time()
    calibration = score_components(
        model, dataset, calibration_idx, device, args.batch_size,
        "calibration", global_mode=global_mode)
    test = score_components(
        model, dataset, test_idx, device, args.batch_size, "test", global_mode=global_mode)

    metrics = {}
    arrays = {}
    for name, scored in test.items():
        metrics[name] = summarize_scores(scored.scores, scored.labels)
        arrays[name] = scored.scores

    q_reconstruction = percentile(
        calibration["reconstruction"].scores,
        test["reconstruction"].scores)
    q_cycle = percentile(
        calibration["cycle"].scores, test["cycle"].scores)
    fusion = 0.5 * (q_reconstruction + q_cycle)
    metrics["percentile_fusion_50_50"] = summarize_scores(
        fusion, test["reconstruction"].labels)
    arrays["percentile_fusion_50_50"] = fusion

    background = test["reconstruction"].labels == 0
    correlation = float(spearmanr(
        test["reconstruction"].scores[background],
        test["cycle"].scores[background]).statistic)
    result = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "split_manifest": str(args.split_manifest),
        "calibration": "val_bkg_events",
        "event_aggregation": "sum",
        "cycle_global_mode": global_mode,
        "n_calibration_events": int(
            len(calibration["reconstruction"].scores)),
        "n_test_events": int(len(test["reconstruction"].scores)),
        "background_spearman_reconstruction_vs_cycle": correlation,
        "metrics": metrics,
        "elapsed_s": time.time() - started,
    }
    np.savez_compressed(
        output / "scores.npz",
        labels=test["reconstruction"].labels,
        event_ids=test["reconstruction"].event_ids,
        **arrays,
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--cache_shards", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
