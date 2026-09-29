"""Evaluate frozen latent-cycle components on LHCO, Kitchen Sink, and BB1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.eval_cycle_components import (
    load_cycle_model,
    percentile,
    score_components,
)
from scripts.eval_ks_ratio import (
    event_indices,
    indices_for_event_ids,
    parse_signal_spec,
)
from src.checkpoint import checkpoint_cycle_mode
from src.data.dataset import JetDataset
from src.eval.metrics import summarize_scores
from src.models import ensure_dataset_matches


COMPONENTS = ("reconstruction", "node_mse", "edge_mse", "cycle", "cycle_local", "cycle_global")


def add_score_sums(metrics, arrays, labels):
    combined = arrays["reconstruction"] + arrays["cycle"]
    arrays["reconstruction_plus_cycle"] = combined
    metrics["reconstruction_plus_cycle"] = summarize_scores(combined, labels)


def evaluate_parts(calibration, background, signal):
    """Return component metrics for separate background and signal samples."""
    metrics, arrays = {}, {}
    labels = np.concatenate([
        np.zeros(len(background["reconstruction"].scores), dtype=np.int64),
        np.ones(len(signal["reconstruction"].scores), dtype=np.int64),
    ])
    for name in COMPONENTS:
        values = np.concatenate([background[name].scores, signal[name].scores])
        metrics[name] = summarize_scores(values, labels)
        arrays[name] = values

    q_reconstruction = percentile(
        calibration["reconstruction"].scores, arrays["reconstruction"])
    q_cycle = percentile(calibration["cycle"].scores, arrays["cycle"])
    fusion = 0.5 * (q_reconstruction + q_cycle)
    metrics["percentile_fusion_50_50"] = summarize_scores(fusion, labels)
    arrays["percentile_fusion_50_50"] = fusion
    add_score_sums(metrics, arrays, labels)
    return metrics, labels, arrays


def evaluate_labeled(calibration, scored):
    """Return component metrics for one dataset carrying both truth labels."""
    labels = scored["reconstruction"].labels
    metrics = {
        name: summarize_scores(scored[name].scores, labels)
        for name in COMPONENTS
    }
    arrays = {name: scored[name].scores for name in COMPONENTS}
    q_reconstruction = percentile(
        calibration["reconstruction"].scores, arrays["reconstruction"])
    q_cycle = percentile(calibration["cycle"].scores, arrays["cycle"])
    fusion = 0.5 * (q_reconstruction + q_cycle)
    metrics["percentile_fusion_50_50"] = summarize_scores(fusion, labels)
    arrays["percentile_fusion_50_50"] = fusion
    add_score_sums(metrics, arrays, labels)
    return metrics, labels, arrays


def save_dataset(output, name, metrics, labels, arrays):
    target = output / name
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "scores.npz", labels=labels, **arrays)
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"{name:16s} recon={metrics['reconstruction']['auc']:.4f} "
        f"node={metrics['node_mse']['auc']:.4f} edge={metrics['edge_mse']['auc']:.4f} "
        f"cycle={metrics['cycle']['auc']:.4f} "
        f"fusion={metrics['percentile_fusion_50_50']['auc']:.4f}",
        flush=True,
    )


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    global_mode = args.cycle_global_mode or checkpoint_cycle_mode(Path(args.checkpoint))
    model, spec = load_cycle_model(Path(args.checkpoint), device)

    background_ds = JetDataset(args.bkg_dir, max_cache=args.cache_shards)
    ensure_dataset_matches(background_ds, spec)
    with np.load(args.split_manifest) as manifest:
        calibration_idx, _ = indices_for_event_ids(
            background_ds, 0, manifest["val_bkg_events"])
        background_idx, _ = indices_for_event_ids(
            background_ds, 0, manifest["test_bkg_events"])
        lhco_signal_idx, _ = indices_for_event_ids(
            background_ds, 1, manifest["test_sig_events"])

    started = time.time()
    calibration = score_components(
        model, background_ds, calibration_idx, device, args.batch_size,
        "calibration", global_mode=global_mode)
    summary = {
        "checkpoint": args.checkpoint,
        "bkg_dir": args.bkg_dir,
        "split_manifest": args.split_manifest,
        "calibration": "val_bkg_events",
        "event_aggregation": "sum",
        "cycle_global_mode": global_mode,
        "reconstruction_scores": {
            "node_mse": "node feature MSE, summed over the two jets",
            "edge_mse": "physical edge feature MSE, summed over the two jets",
            "reconstruction": "node_mse + model.edge_weight * edge_mse",
            "reconstruction_plus_cycle": "reconstruction + total latent-cycle score",
        },
        "n_calibration_events": int(
            len(calibration["reconstruction"].scores)),
        "datasets": {},
    }

    if not args.only_bb1:
        background = score_components(
            model, background_ds, background_idx, device, args.batch_size,
            "background", global_mode=global_mode)
        lhco_signal = score_components(
            model, background_ds, lhco_signal_idx, device, args.batch_size,
            "LHCO signal", global_mode=global_mode)

        metrics, labels, arrays = evaluate_parts(
            calibration, background, lhco_signal)
        save_dataset(output, "LHCO", metrics, labels, arrays)
        summary["datasets"]["LHCO"] = metrics

        for signal_spec in args.signal:
            name, path = parse_signal_spec(signal_spec)
            signal_ds = JetDataset(path, max_cache=args.cache_shards)
            ensure_dataset_matches(signal_ds, spec)
            signal_idx, signal_ids = event_indices(
                signal_ds, 1, args.n_signal)
            if len(signal_ids) != args.n_signal:
                raise ValueError(
                    f"{name} provides {len(signal_ids):,} signal events; "
                    f"expected {args.n_signal:,}")
            signal = score_components(
                model, signal_ds, signal_idx, device, args.batch_size, name,
                global_mode=global_mode)
            metrics, labels, arrays = evaluate_parts(
                calibration, background, signal)
            save_dataset(output, name, metrics, labels, arrays)
            summary["datasets"][name] = metrics

    if not args.skip_bb1:
        if args.bb1_dir is None:
            raise ValueError("--bb1_dir is required unless --skip_bb1 is set")
        bb1_ds = JetDataset(args.bb1_dir, max_cache=args.cache_shards)
        ensure_dataset_matches(bb1_ds, spec)
        bb1 = score_components(
            model, bb1_ds, np.arange(len(bb1_ds), dtype=np.int64), device,
            args.batch_size, "BB1", global_mode=global_mode)
        metrics, labels, arrays = evaluate_labeled(calibration, bb1)
        save_dataset(output, "BB1", metrics, labels, arrays)
        summary["datasets"]["BB1"] = metrics

    summary["elapsed_s"] = time.time() - started
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved suite -> {output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bkg_dir", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--bb1_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--signal", action="append", required=True)
    parser.add_argument("--n_signal", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--cache_shards", type=int, default=32)
    parser.add_argument('--cycle_global_mode', choices=('pooled', 'projected', 'unit_projected'),
                        help='Default: use the representation saved during training.')
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--skip_bb1", action="store_true",
        help="Evaluate LHCO and requested signals without loading BB1.")
    parser.add_argument(
        "--only_bb1", action="store_true",
        help="Score BB1 only; the summary contains only this invocation's results.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
