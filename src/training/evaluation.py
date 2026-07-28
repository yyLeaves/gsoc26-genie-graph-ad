import json
import time

import numpy as np
import torch

from src.data.iterate import prefetch, shard_iter
from src.eval.metrics import summarize_scores
from src.eval.scoring import score_events
from src.models import LossTerms, load_model


def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _save_scores(output_dir, stem, scores, labels):
    np.save(output_dir / f"scores_{stem}.npy", scores)
    np.save(output_dir / f"labels_{stem}.npy", labels)


@torch.no_grad()
def eval_loss_components(model, ds, indices, batch_size, device):
    """Return total/node/edge losses averaged over the selected graphs."""
    if len(indices) == 0:
        raise ValueError("Loss evaluation requires at least one jet.")
    model.eval()
    sum_total = sum_node = sum_edge = n = 0.0
    for batch in prefetch(shard_iter(
            ds, indices, batch_size,
            shuffle_shards=False, shuffle_within=False, device=device)):
        losses = model.loss(batch)
        b = batch.num_graphs
        sum_total += losses.total.item() * b
        sum_node += losses.node.item() * b
        sum_edge += losses.edge.item() * b
        n += b
    return LossTerms(total=sum_total / n, node=sum_node / n, edge=sum_edge / n)


def evaluate_epoch(model, ds, splits, args, device, compute_metrics=True):
    """Val loss every call; AUC/SIC only when ``compute_metrics``."""
    val = eval_loss_components(
        model, ds, splits.val_idx, args.batch_size * 2, device)
    ev = {
        "val_loss": val.total,
        "val_node_loss": val.node,
        "val_edge_loss": val.edge,
    }
    if not compute_metrics:
        return ev
    scored = score_events(
        model, ds, device, batch_size=args.batch_size * 2,
        indices=splits.eval_idx,
        aggregation=getattr(args, "event_score_agg", "sum"))
    scores, labels = scored.scores, scored.labels
    m = summarize_scores(scores, labels)
    bkg, sig = scores[labels == 0], scores[labels == 1]
    ev.update({
        "mean_score_bkg": float(bkg.mean()) if bkg.size else float("nan"),
        "mean_score_sig": float(sig.mean()) if sig.size else float("nan"),
        "auc": m["auc"], "max_sic": m["max_sic"],
        "eS_at_eB1e-2": m["eS_at_eB1e-2"], "eS_at_eB1e-3": m["eS_at_eB1e-3"],
    })
    return ev


def _evaluate_checkpoint(ds, final_idx, checkpoint_path, name, epoch,
                         val_loss, output_dir, args, device, run_name):
    """Score one checkpoint; write ``scores_<name>.npy`` and ``metrics_<name>.json``."""
    model = load_model(checkpoint_path, device)
    scored = score_events(
        model, ds, device, batch_size=args.batch_size * 2,
        indices=final_idx, aggregation=args.event_score_agg)
    scores, labels = scored.scores, scored.labels
    _save_scores(output_dir, name, scores, labels)
    m = summarize_scores(scores, labels)
    metrics = {
        "run_name": run_name,
        "checkpoint": name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(epoch),
        "checkpoint_val_loss": float(val_loss),
        "auc": m["auc"],
        "max_sic": m["max_sic"],
        "best_sic_threshold": m["best_sic_threshold"],
        "eS_at_eB1e-2": m["eS_at_eB1e-2"],
        "eS_at_eB1e-3": m["eS_at_eB1e-3"],
        "n_scored_events": int(len(labels)),
        "n_background_events": m["n_background"],
        "n_signal_events": m["n_signal"],
        "event_score_aggregation": args.event_score_agg,
    }
    _write_json(output_dir / f"metrics_{name}.json", metrics)
    return metrics, scores, labels


def final_evaluation(ds, splits, output_dir, args, device, log,
                     best_val, best_epoch, last_val, n_epochs, t0, run_name,
                     monitor_best=None):
    """Score best/last (and optional monitor_best); write comparison artifacts.

    Primary metrics: ``last.pt`` under ``--no_early_stop``, else ``best.pt``.
    """
    final_idx = splits.test_idx
    log.info("\nFinal evaluation checkpoints: best.pt and last.pt")

    best_m, best_s, best_y = _evaluate_checkpoint(
        ds, final_idx, output_dir / "best.pt", "best", best_epoch,
        best_val, output_dir, args, device, run_name)
    if best_epoch == n_epochs:
        last_m = {**best_m, "checkpoint": "last",
                  "checkpoint_path": str(output_dir / "last.pt")}
        last_s, last_y = best_s, best_y
        _write_json(output_dir / "metrics_last.json", last_m)
        _save_scores(output_dir, "last", last_s, last_y)
    else:
        last_m, last_s, last_y = _evaluate_checkpoint(
            ds, final_idx, output_dir / "last.pt", "last", n_epochs,
            last_val, output_dir, args, device, run_name)

    primary = "last" if args.no_early_stop else "best"
    by_name = {
        "best": (best_m, best_s, best_y),
        "last": (last_m, last_s, last_y),
    }
    primary_m, primary_s, primary_y = by_name[primary]
    np.save(output_dir / "scores.npy", primary_s)
    np.save(output_dir / "labels.npy", primary_y)
    metrics = {
        **primary_m,
        "primary_checkpoint": primary,
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "last_val_loss": float(last_val),
        "n_epochs": int(n_epochs),
        "total_time_s": time.time() - t0,
    }
    _write_json(output_dir / "metrics.json", metrics)

    deltas = {
        "auc": last_m["auc"] - best_m["auc"],
        "max_sic": last_m["max_sic"] - best_m["max_sic"],
        "eS_at_eB1e-2": last_m["eS_at_eB1e-2"] - best_m["eS_at_eB1e-2"],
        "eS_at_eB1e-3": last_m["eS_at_eB1e-3"] - best_m["eS_at_eB1e-3"],
        "val_loss": float(last_val - best_val),
    }
    comparison = {
        "primary_checkpoint": primary,
        "best": best_m,
        "last": last_m,
        "delta_last_minus_best": deltas,
    }

    monitor_m = None
    monitor_path = output_dir / "monitor_best.pt"
    if monitor_best is not None and monitor_path.exists():
        monitor_m, _, _ = _evaluate_checkpoint(
            ds, final_idx, monitor_path, "monitor_best",
            monitor_best["epoch"], monitor_best["val_loss"],
            output_dir, args, device, run_name)
        monitor_m.update({
            "selection_metric": "monitor_auc",
            "selection_value": monitor_best["auc"],
            "selection_warning": (
                "Oracle research diagnostic: monitor signal labels were used "
                "for checkpoint selection; not a purely unsupervised result."),
        })
        _write_json(output_dir / "metrics_monitor_best.json", monitor_m)
        comparison["monitor_best_oracle"] = monitor_m
    _write_json(output_dir / "metrics_comparison.json", comparison)

    log.info(f"\n{'='*72}")
    log.info("Final held-out test comparison")
    log.info("ckpt   epoch  val_loss      AUC   MaxSIC   eS@1e2   eS@1e3")
    rows = [("best", best_m), ("last", last_m)]
    if monitor_m is not None:
        rows.append(("mon*", monitor_m))
    for name, m in rows:
        log.info(f"{name:<5}  {m['checkpoint_epoch']:>5}  "
                 f"{m['checkpoint_val_loss']:8.4f}  "
                 f"{m['auc']:7.4f}  {m['max_sic']:7.3f}  "
                 f"{m['eS_at_eB1e-2']:7.4f}  {m['eS_at_eB1e-3']:7.4f}")
    log.info(f"Δ last-best    val={deltas['val_loss']:+.4f}  "
             f"AUC={deltas['auc']:+.4f}  MaxSIC={deltas['max_sic']:+.3f}")
    log.info(f"Primary metrics.json checkpoint: {primary}.pt")
    if monitor_m is not None:
        log.info("mon* is an oracle research diagnostic (monitor labels); "
                 "not the primary result")
    log.info(f"{'='*72}\nAll results saved to {output_dir}/")
    return metrics
