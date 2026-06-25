import json
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import JetDataset
from src.data.iterate import prefetch, shard_iter
from src.eval.evaluate import eval_loss_components, eval_scores
from src.eval.scoring import report
from src.models import NodeGraphAE, EdgeGraphAE


EDGE_AE_MODEL_NAMES = {"edgeae", "relae"}


@dataclass
class Splits:
    """Flat jet index sets for one run."""
    train_idx: np.ndarray   # background, 80%
    val_idx: np.ndarray     # background, 20%
    sig_idx: np.ndarray     # all signal
    eval_idx: np.ndarray    # val_bkg + all signal (epoch AUC/SIC set)
    all_idx: np.ndarray     # everything (final eval)


def setup_logger(output_dir: Path, name: str) -> logging.Logger:
    log_path = output_dir / f"{name}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # logger is a process-wide singleton; drop stale handlers or a sweep
    # duplicates every line and leaks FileHandlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(message)s")
    for h in [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, "w")]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger


def build_dataset(args, log):
    """Load the JetDataset and return (ds, stats, in_dim)."""
    log.info(f"\nLoading dataset from {args.data_dir} ...")
    # EdgeGraphAE defaults to node = pT only (col 0): geometry lives on edges, so edge
    # recon dominates (Araz et al.). --node_features all keeps the full vector.
    feature_cols = ([0] if args.model in EDGE_AE_MODEL_NAMES and args.node_features == "pt"
                    else None)
    ds = JetDataset(args.data_dir, feature_cols=feature_cols,
                    max_cache=args.cache_shards)
    log.info(str(ds))
    stats = ds.stats()
    log.info(f"  n={stats['n']:,}  signal={stats['n_signal']:,}  "
             f"background={stats['n_background']:,}  "
             f"mean_nodes={stats['mean_nodes']:.1f}  "
             f"mean_edges={stats['mean_edges']:.1f}")
    return ds, stats, ds[0].x.shape[-1]


def build_splits(ds, args, log, rng) -> Splits:
    """Background-only 80/20 train/val split + signal/eval/all index sets.
    Shared `rng` is advanced here, then reused for per-epoch shuffles."""
    # Pilot: first --fraction of shards (contiguous → shard-sequential I/O;
    # a random 5% of jets would still touch every shard each epoch)
    if args.fraction < 1.0:
        n_use = max(1, round(ds.num_shards * args.fraction))
        # clamp to len(ds): last shard is partial, so n_use*shard_size can overshoot
        limit = min(n_use * ds.shard_size, len(ds))
        log.info(f"\nPilot run: fraction={args.fraction}  "
                 f"→ first {n_use}/{ds.num_shards} shards ({limit:,} jets)")
    else:
        limit = len(ds)

    if ds.has_event_ids():
        all_event_ids = np.asarray(ds.event_ids, dtype=np.int64)
        selected_ids, selected_counts = np.unique(all_event_ids[:limit],
                                                  return_counts=True)
        full_ids, full_counts = np.unique(all_event_ids, return_counts=True)
        full_count = dict(zip(full_ids.tolist(), full_counts.tolist()))
        complete_events = {
            int(e) for e, c in zip(selected_ids, selected_counts)
            if int(c) == full_count[int(e)]
        }
        all_idx = np.array([i for i, e in enumerate(all_event_ids[:limit])
                            if int(e) in complete_events], dtype=np.int64)
        if len(all_idx) < limit:
            log.info(f"Pilot event clamp: dropped {limit - len(all_idx):,} "
                     "boundary jet(s) from incomplete event(s)")

        event_ids = np.asarray(ds.event_ids[all_idx], dtype=np.int64)
        labels = np.asarray(ds.labels[all_idx], dtype=np.int64)
        event_label = {}
        for event_id, label in zip(event_ids, labels):
            event_id, label = int(event_id), int(label)
            if event_id in event_label and event_label[event_id] != label:
                raise ValueError(
                    f"inconsistent labels inside event_id={event_id}: "
                    f"{event_label[event_id]} vs {label}")
            event_label[event_id] = label
        bg_events = np.array([e for e, y in event_label.items() if y == 0],
                             dtype=np.int64)
        sig_events = np.array([e for e, y in event_label.items() if y == 1],
                              dtype=np.int64)
        bg_events = bg_events[rng.permutation(len(bg_events))]
        split = int(len(bg_events) * 0.8)
        train_events = set(bg_events[:split].tolist())
        val_events = set(bg_events[split:].tolist())
        sig_events_set = set(sig_events.tolist())
        train_idx = np.array([i for i, e in zip(all_idx, event_ids)
                              if int(e) in train_events], dtype=np.int64)
        val_idx = np.array([i for i, e in zip(all_idx, event_ids)
                            if int(e) in val_events], dtype=np.int64)
        sig_idx = np.array([i for i, e in zip(all_idx, event_ids)
                            if int(e) in sig_events_set], dtype=np.int64)
        log.info("Split mode : event-level (jets from one event stay together)")
    else:
        log.warning("Split mode : jet-level fallback; event-level scoring will "
                    "require rebuilt shards with event_ids")
        bg_idx = ds.background_indices()
        bg_idx = bg_idx[bg_idx < limit]
        bg_idx = bg_idx[rng.permutation(len(bg_idx))]
        split = int(len(bg_idx) * 0.8)
        train_idx, val_idx = bg_idx[:split], bg_idx[split:]
        sig_idx = np.where(ds.labels[:limit] == 1)[0]
        all_idx = np.arange(limit)
    eval_idx = np.concatenate([val_idx, sig_idx])

    log.info(f"\nSplit  train={len(train_idx):,} bkg | val={len(val_idx):,} bkg"
             f" | eval_set={len(eval_idx):,} (val_bkg + all_sig)")
    return Splits(train_idx, val_idx, sig_idx, eval_idx, all_idx)


def build_model(args, in_dim, ds, device, log):
    """Construct the model named by args.model."""
    use_edge_ae = args.model in EDGE_AE_MODEL_NAMES
    use_bn = not args.no_bn
    if use_edge_ae:
        edge_dim = ds.meta.get("edge_dim", 0)
        if edge_dim == 0 or ds[0].edge_attr is None:
            raise ValueError(
                "model=edgeae needs edge features. Rebuild shards with: "
                "build_graph.py --edge_features log  (Araz reference form, "
                "best AD; linear is weaker)")
        model = EdgeGraphAE(in_dim=in_dim, edge_dim=edge_dim,
                           hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
                           edge_weight=args.edge_weight, aggr=args.aggr).to(device)
    else:
        model = NodeGraphAE(in_dim=in_dim, backbone=args.backbone,
                        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
                        use_bn=use_bn).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"\nModel      : {model.__class__.__name__}  "
             + (f"edge_dim={ds.meta.get('edge_dim')}  aggr={args.aggr}  "
                f"edge_w={args.edge_weight}  "
                if use_edge_ae else f"backbone={args.backbone}  ")
             + f"hidden={args.hidden_dim}  latent={args.latent_dim}"
             + ("" if use_edge_ae else f"  bn={use_bn}"))
    log.info(f"Parameters : {n_params:,}")
    return model


def build_optimizer_scheduler(args, model, ds, train_idx, log):
    """AdamW + requested LR scheduler.
    Returns (optimizer, scheduler, sched_per_batch, total_steps, sched_desc)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # OneCycleLR steps per batch + needs exact step count; shard_iter makes
    # ceil(n/bs) batches/shard, so total = Σ_shard ceil(n_shard/bs)
    sched_per_batch = (args.scheduler == "onecycle")
    shard_counts = Counter(int(i) // ds.shard_size for i in train_idx)
    steps_per_epoch = sum((c + args.batch_size - 1) // args.batch_size
                          for c in shard_counts.values())
    total_steps = None
    if sched_per_batch:
        total_steps = steps_per_epoch * args.epochs
        num_warmup_steps = max(1, int(0.02 * total_steps))
        pct_start = min(max(num_warmup_steps / total_steps, 0.01), 0.9)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps,
            pct_start=pct_start, anneal_strategy="linear",
            div_factor=5.0, final_div_factor=3.0)
        sched_desc = (f"OneCycleLR  max_lr={args.lr}  total_steps={total_steps}"
                      f"  warmup={pct_start:.3f}  ({steps_per_epoch}/epoch, "
                      "linear, div=5, final_div=3)")
    elif args.scheduler == "linear":
        # Araz-style: linear anneal lr → lr_end per epoch
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0,
            end_factor=max(args.lr_end / args.lr, 1e-6),
            total_iters=max(args.epochs - 1, 1))
        sched_desc = f"LinearLR  {args.lr}→{args.lr_end}  over {args.epochs} ep"
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        sched_desc = f"CosineAnnealing  T_max={args.epochs}"

    log.info(f"Optimizer  : AdamW  lr={args.lr}  wd={args.weight_decay}")
    log.info(f"Scheduler  : {sched_desc}")
    log.info(f"Early stop : {'off (fixed epochs)' if args.no_early_stop else f'patience={args.patience}'}")
    log.info(f"Eval every : {args.eval_interval} epochs  (AUC + SIC on eval_set)")
    log.info(f"Batch size : {args.batch_size}")
    log.info(f"Score agg  : {args.event_score_agg}")
    return optimizer, scheduler, sched_per_batch, total_steps, sched_desc


def write_run_config(output_dir, run_name, ts, args, model, ds, stats, splits,
                     in_dim, n_params, sched_desc, device, log):
    """Dump config.json + split.npz at run start."""
    exp_config = {
        "run_name": run_name, "timestamp": ts, "output_dir": str(output_dir),
        "model": {
            "type": args.model, "backbone": args.backbone, "in_dim": in_dim,
            "hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim,
            # Persist every constructor knob that changes module structure or
            # score semantics, so load_model rebuilds the checkpoint exactly.
            "use_bn": not args.no_bn,
            "edge_dim": ds.meta.get("edge_dim", 0),
            "edge_weight": args.edge_weight,
            "aggr": args.aggr,
            "n_params": n_params,
            "architecture": str(model),
        },
        "dataset": {
            "data_dir": str(args.data_dir),
            "strategy": ds.meta.get("strategy", "?"),
            "features": ds.meta.get("features", "?"),
            "n_total": stats["n"], "n_signal": stats["n_signal"],
            "n_background": stats["n_background"], "mean_nodes": stats["mean_nodes"],
            "mean_edges": stats["mean_edges"], "shard_size": ds.shard_size,
            "num_shards": ds.num_shards,
        },
        "split": {
            "n_train": len(splits.train_idx), "n_val": len(splits.val_idx),
            "n_eval_set": len(splits.eval_idx), "fraction": args.fraction,
            "seed": args.seed,
        },
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "patience": args.patience,
            "eval_interval": args.eval_interval, "cache_shards": args.cache_shards,
            "event_score_aggregation": args.event_score_agg,
        },
        "optimizer": "AdamW", "scheduler": sched_desc,
        "no_early_stop": args.no_early_stop, "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(exp_config, f, indent=2)
    log.info(f"\nConfig saved → {output_dir}/config.json")
    np.savez(output_dir / "split.npz", train_idx=splits.train_idx,
             val_idx=splits.val_idx, sig_idx=splits.sig_idx,
             eval_idx=splits.eval_idx)
    log.info(f"Split  saved → {output_dir}/split.npz")


def train_one_epoch(model, ds, splits, optimizer, scheduler, sched_per_batch,
                    total_steps, args, device, rng):
    """One training pass via model.loss. Returns (total, c1, c2) means; (c1, c2)
    are the two components (recon/kl for AE/VAE, node/edge for EdgeGraphAE —
    they only sum to total when --edge_weight == 1)."""
    model.train()
    sum_total = sum_c1 = sum_c2 = n_seen = 0.0
    for batch in prefetch(shard_iter(ds, splits.train_idx, args.batch_size,
                                     shuffle_shards=True, shuffle_within=True,
                                     rng=rng, device=device)):
        optimizer.zero_grad()
        total, c1, c2 = model.loss(batch)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if sched_per_batch and scheduler.last_epoch + 1 < total_steps:
            scheduler.step()                 # OneCycleLR: per batch, guard overshoot
        B = batch.num_graphs
        sum_total += total.item() * B
        sum_c1    += c1.item()    * B
        sum_c2    += c2.item()    * B
        n_seen    += B
    if not sched_per_batch:
        scheduler.step()                     # cosine/linear: per epoch
    return sum_total / n_seen, sum_c1 / n_seen, sum_c2 / n_seen


def evaluate_epoch(model, ds, splits, args, device, compute_metrics=True):
    """Val loss every call (cheap, drives best.pt / early stop); AUC/SIC/ε_S over
    the eval set only when compute_metrics (gated by --eval_interval).
    Returns val_loss/recon/kl always, plus metric keys when scored."""
    v_total, v_c1, v_c2 = eval_loss_components(
        model, ds, splits.val_idx, args.batch_size * 2, device)
    ev = {"val_loss": v_total, "val_recon": v_c1, "val_kl": v_c2}
    if not compute_metrics:
        return ev
    score_agg = getattr(args, "event_score_agg", "sum")
    ev_scores, ev_labels = eval_scores(
        model, ds, splits.eval_idx, args.batch_size * 2, device,
        aggregation=score_agg)
    m = report(ev_scores, ev_labels)
    # --fraction pilot with no signal → single-class eval; guard per-class
    # means to yield NaN, not a warning
    bkg, sig = ev_scores[ev_labels == 0], ev_scores[ev_labels == 1]
    ev.update({
        "mean_score_bkg": float(bkg.mean()) if bkg.size else float("nan"),
        "mean_score_sig": float(sig.mean()) if sig.size else float("nan"),
        "auc": m["auc"], "max_sic": m["max_sic"],
        "eS_at_eB1e-2": m["eS_at_eB1e-2"], "eS_at_eB1e-3": m["eS_at_eB1e-3"],
    })
    return ev


def final_evaluation(model, ds, splits, output_dir, args, device, log,
                     best_val, n_epochs, t0, run_name):
    """Score the full dataset with the chosen checkpoint; write metrics.json."""
    # fixed-schedule runs eval the final model; early-stop runs eval best-val
    ckpt = "last.pt" if args.no_early_stop else "best.pt"
    log.info(f"\nLoading {ckpt} for final evaluation ...")
    model.load_state_dict(torch.load(output_dir / ckpt, map_location=device,
                                     weights_only=True))
    scores, labels = eval_scores(model, ds, splits.all_idx, args.batch_size * 2,
                                 device, aggregation=args.event_score_agg)
    np.save(output_dir / "scores.npy", scores)
    np.save(output_dir / "labels.npy", labels)

    m = report(scores, labels)
    log.info(f"\n{'='*50}")
    log.info(f"Final AUC     : {m['auc']:.4f}")
    log.info(f"Final MaxSIC  : {m['max_sic']:.4f}  (threshold={m['best_threshold']:.4f})")
    log.info(f"ε_S @ 100x rej: {m['eS_at_eB1e-2']:.4f}   "
             f"ε_S @ 1000x rej: {m['eS_at_eB1e-3']:.4f}")
    log.info(f"{'='*50}")

    metrics = {
        "run_name": run_name, "auc": m["auc"],
        "max_sic": m["max_sic"], "best_threshold": m["best_threshold"],
        "eS_at_eB1e-2": m["eS_at_eB1e-2"], "eS_at_eB1e-3": m["eS_at_eB1e-3"],
        "best_val_loss": best_val, "n_epochs": n_epochs,
        "total_time_s": time.time() - t0,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"\nAll results saved to {output_dir}/")
    return metrics


def train(args):
    """Run one training job end to end; returns the final metrics dict."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{args.backbone}_{ts}"
    output_dir = Path(args.output) / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(output_dir, run_name)
    log.info(f"Run      : {run_name}")
    log.info(f"Output   : {output_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")
    log.info(f"Device   : {device}")

    # one shared rng: split permutation then per-epoch shuffles (single-rng order)
    rng = np.random.default_rng(args.seed)
    ds, stats, in_dim = build_dataset(args, log)
    splits = build_splits(ds, args, log, rng)
    model = build_model(args, in_dim, ds, device, log)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer, scheduler, sched_per_batch, total_steps, sched_desc = \
        build_optimizer_scheduler(args, model, ds, splits.train_idx, log)
    write_run_config(output_dir, run_name, ts, args, model, ds, stats, splits,
                     in_dim, n_params, sched_desc, device, log)

    c_recon, c_kl = model.COMPONENT_NAMES
    line = "─" * 92
    log.info(f"\n{line}")
    log.info(f"{'Epoch':>6}  {'train':>8}  {'val':>8}  {c_recon:>8}  {c_kl:>7}  "
             f"{'AUC':>6}  {'SIC':>6}  {'eS@1e2':>7}  {'eS@1e3':>7}  {'time':>6}")
    log.info(line)

    best_val, patience, history = float("inf"), 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        tr_total, tr_recon, tr_kl = train_one_epoch(
            model, ds, splits, optimizer, scheduler, sched_per_batch,
            total_steps, args, device, rng)
        # --eval_interval gates the expensive AUC/SIC scoring; val_loss runs every
        # epoch and the final epoch is always scored
        do_metrics = (args.eval_interval > 0
                      and (epoch % args.eval_interval == 0 or epoch == args.epochs))
        ev = evaluate_epoch(model, ds, splits, args, device,
                            compute_metrics=do_metrics)

        torch.save(model.state_dict(), output_dir / "last.pt")
        improved = ev["val_loss"] < best_val
        if improved:
            best_val, patience = ev["val_loss"], 0
            torch.save(model.state_dict(), output_dir / "best.pt")
        else:
            patience += 1

        entry = {
            "epoch": epoch, "lr": scheduler.get_last_lr()[0],
            "train_loss": tr_total, "train_recon": tr_recon, "train_kl": tr_kl,
            **ev, "improved": improved, "epoch_time_s": time.time() - ep_t0,
        }
        if do_metrics:
            entry["score_sep"] = ev["mean_score_sig"] - ev["mean_score_bkg"]
        history.append(entry)
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        mark = " *" if improved else "  "
        row = (f"{epoch:>6}{mark}  {tr_total:8.4f}  {ev['val_loss']:8.4f}  "
               f"{ev['val_recon']:8.4f}  {ev['val_kl']:7.4f}  ")
        if do_metrics:
            row += (f"{ev['auc']:6.4f}  {ev['max_sic']:6.3f}  "
                    f"{ev['eS_at_eB1e-2']:7.4f}  {ev['eS_at_eB1e-3']:7.4f}")
        else:
            row += f"{'--':>6}  {'--':>6}  {'--':>7}  {'--':>7}"
        log.info(row + f"  {entry['epoch_time_s']:5.0f}s")

        if not args.no_early_stop and patience >= args.patience:
            log.info(f"\nEarly stop at epoch {epoch}  (patience={args.patience})")
            break

    log.info(f"{'─'*80}")
    return final_evaluation(model, ds, splits, output_dir, args, device,
                            log, best_val, len(history), t0, run_name)
