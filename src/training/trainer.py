from dataclasses import dataclass
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.checkpoint import load_checkpoint
from src.data.dataset import JetDataset
from src.data.iterate import prefetch, shard_iter
from src.models import (EDGE_FEATURE_MODEL_TYPES, PT_NODE_MODEL_TYPES,
                        ModelSpec, create_model,
                        ensure_dataset_matches)
from src.models.reconstruction import mean_loss, reconstruction_scores
from src.training.evaluation import evaluate_epoch, final_evaluation
from src.training.artifacts import (capture_training_state,
                                    restore_training_checkpoint,
                                    resume_signature,
                                    save_training_checkpoint,
                                    write_run_config)
from src.training.splits import build_splits
from src.training.topo_reg import compute_topo_reg


@dataclass(frozen=True, slots=True)
class TrainEpochStats:
    """Per-epoch training means. ``total/node/edge`` are reconstruction only."""

    total: float
    node: float
    edge: float
    reg: float = 0.0

FIRST_NODE_FEATURE_SEMANTICS = {
    "raw": "raw_pt",
    "normalized": "pt_fraction",
    "log_phys": "log_pt",
}


def setup_logger(
    output_dir: Path,
    name: str,
    *,
    append: bool = False,
) -> logging.Logger:
    log_path = output_dir / f"{name}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(message)s")
    mode = "a" if append else "w"
    for h in [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode),
    ]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger


def build_dataset(args, log):
    """Load one faithful, full-feature graph dataset."""
    log.info(f"\nLoading dataset from {args.data_dir} ...")
    ds = JetDataset(args.data_dir, max_cache=args.cache_shards)
    log.info(str(ds))
    stats = ds.stats()
    log.info(f"  n={stats['n_jets']:,}  signal={stats['n_signal']:,}  "
             f"background={stats['n_background']:,}  "
             f"mean_nodes={stats['mean_nodes']:.1f}  "
             f"mean_edges={stats.get('mean_edges', 0.0):.1f}")
    return ds, stats


def _resolve_node_input(args, ds) -> tuple[int, tuple[int, ...] | None, str]:
    """Resolve the model-owned feature view against dataset metadata."""
    full_dim = int(ds.meta["nodes"]["feature_dim"])
    feature_mode = ds.meta["nodes"]["features"]
    if args.model in PT_NODE_MODEL_TYPES and args.node_features == "pt":
        semantics = FIRST_NODE_FEATURE_SEMANTICS.get(
            feature_mode, f"column_0_of_{feature_mode}")
        return 1, (0,), semantics
    return full_dim, None, f"all_{feature_mode}"


def build_model(args, ds, device, log):
    """Resolve dataset-dependent dimensions and construct one model spec."""
    in_dim, feature_cols, node_input_semantics = _resolve_node_input(args, ds)
    edges = ds.meta.get("edges") or {}
    edge_dim = (int(edges.get("feature_dim", 0))
                if args.model in EDGE_FEATURE_MODEL_TYPES else 0)
    spec = ModelSpec(
        type=args.model,
        in_dim=in_dim,
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        use_bn=False if args.model == "edge_graph" else not args.no_bn,
        edge_dim=edge_dim,
        edge_weight=args.edge_weight,
        aggr=args.aggr,
        dyn_k=args.dyn_k,
        feature_cols=feature_cols,
    )
    ensure_dataset_matches(ds, spec)
    model = create_model(spec).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"\nModel      : {model.__class__.__name__}  type={spec.type}  "
             f"backbone={spec.backbone}  hidden={spec.hidden_dim}  "
             f"latent={spec.latent_dim}  edge_dim={spec.edge_dim}  "
             f"bn={spec.use_bn}")
    log.info(f"Parameters : {n_params:,}")
    log.info(f"Node input : {node_input_semantics}  "
             f"feature_cols={feature_cols if feature_cols is not None else 'all'}")
    return model, spec, node_input_semantics, n_params


def build_optimizer_scheduler(args, model, ds, train_idx, log):
    """AdamW + requested LR scheduler.

    Returns ``(optimizer, scheduler, sched_per_batch, total_steps, sched_desc)``.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # OneCycleLR steps per batch; shard_iter → ceil(n/bs) batches/shard.
    sched_per_batch = args.scheduler == "onecycle"
    shard_counts = Counter(int(i) // ds.shard_size for i in train_idx)
    steps_per_epoch = sum((c + args.batch_size - 1) // args.batch_size
                          for c in shard_counts.values())
    total_steps = None
    if sched_per_batch:
        total_steps = steps_per_epoch * args.epochs
        pct_start = min(max(max(1, int(0.02 * total_steps)) / total_steps,
                            0.01), 0.9)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps,
            pct_start=pct_start, anneal_strategy="linear",
            div_factor=5.0, final_div_factor=3.0)
        sched_desc = (f"OneCycleLR  max_lr={args.lr}  total_steps={total_steps}"
                      f"  warmup={pct_start:.3f}  ({steps_per_epoch}/epoch, "
                      "linear, div=5, final_div=3)")
    elif args.scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0,
            end_factor=max(args.lr_end / args.lr, 1e-6),
            total_iters=max(args.epochs - 1, 1))
        sched_desc = f"LinearLR  {args.lr}→{args.lr_end}  over {args.epochs} ep"
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        sched_desc = f"CosineAnnealing  T_max={args.epochs}"

    early = ("off (fixed epochs)" if args.no_early_stop
             else f"patience={args.patience}")
    log.info(f"Optimizer  : AdamW  lr={args.lr}  wd={args.weight_decay}")
    log.info(f"Scheduler  : {sched_desc}")
    log.info(f"Early stop : {early}")
    log.info(f"Eval every : {args.eval_interval} epochs  (AUC + SIC on eval_set)")
    log.info(f"Batch size : {args.batch_size}")
    log.info(f"Score agg  : {args.event_score_agg}")
    topo_reg = getattr(args, "topo_reg", "none") or "none"
    if topo_reg != "none":
        log.info(f"Topo reg  : {topo_reg}  λ={args.lambda_topo}  "
                 f"unique_k={args.unique_k}  (val/anomaly still recon-only)")
    return optimizer, scheduler, sched_per_batch, total_steps, sched_desc


def train_one_epoch(model, ds, splits, optimizer, scheduler, sched_per_batch,
                    total_steps, args, device, rng):
    """Run one training pass and return mean recon + optional topo-reg stats.

    Returned ``total/node/edge`` are *reconstruction* means (same units as the
    reference runs).  When ``--topo_reg`` is set the optimized objective is
    ``recon + λ * reg``.
    """
    model.train()
    sum_total = sum_node = sum_edge = sum_reg = n_seen = 0.0
    topo_reg = getattr(args, "topo_reg", "none") or "none"
    lambda_topo = float(getattr(args, "lambda_topo", 0.0) or 0.0)
    unique_k = int(getattr(args, "unique_k", 6))
    for batch in prefetch(shard_iter(
            ds, splits.train_idx, args.batch_size,
            shuffle_shards=True, shuffle_within=True, rng=rng, device=device)):
        optimizer.zero_grad()
        output, node_target, edge_target = model._reconstruct(batch)
        losses = mean_loss(reconstruction_scores(
            output, node_target, batch,
            edge_target=edge_target,
            edge_weight=getattr(model, "edge_weight", 1.0),
        ))
        if topo_reg != "none" and lambda_topo != 0.0:
            reg = compute_topo_reg(
                topo_reg, output.latent, batch, unique_k=unique_k,
                model=model, node_target=node_target,
                edge_target=edge_target)
            opt_loss = losses.total + lambda_topo * reg
        else:
            reg = losses.total.new_zeros(())
            opt_loss = losses.total
        opt_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if sched_per_batch and scheduler.last_epoch + 1 < total_steps:
            scheduler.step()
        b = batch.num_graphs
        sum_total += losses.total.item() * b
        sum_node += losses.node.item() * b
        sum_edge += losses.edge.item() * b
        sum_reg += float(reg.detach()) * b
        n_seen += b
    if not sched_per_batch:
        scheduler.step()
    return TrainEpochStats(
        total=sum_total / n_seen, node=sum_node / n_seen,
        edge=sum_edge / n_seen, reg=sum_reg / n_seen)


def _open_run(args):
    """Create a fresh run dir, or reopen an interrupted one for resume."""
    resume_value = getattr(args, "resume", None)
    if not resume_value:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.model}_{args.backbone}_{ts}"
        output_dir = Path(args.output) / ts
        output_dir.mkdir(parents=True, exist_ok=True)
        log = setup_logger(output_dir, run_name)
        return output_dir, run_name, ts, log, None

    resume_path = Path(resume_value)
    payload = load_checkpoint(resume_path, map_location="cpu")
    run_config = payload["run_config"]
    output_dir = resume_path.resolve().parent
    requested = Path(args.output).resolve()
    if output_dir.parent != requested:
        raise ValueError(
            f"resume checkpoint belongs to output base "
            f"{output_dir.parent}, not {requested}")
    run_name = run_config["run_name"]
    ts = run_config["timestamp"]
    log = setup_logger(output_dir, run_name, append=True)
    log.info("\nResuming interrupted run")
    log.info(f"Checkpoint: {resume_path}")
    return output_dir, run_name, ts, log, payload


def _seed_run(args, log):
    """Seed NumPy (via caller rng) and PyTorch for reproducible init."""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    log.info(f"Seed     : {args.seed} (NumPy split/shuffle + PyTorch init)")
    return np.random.default_rng(args.seed)


def _fresh_training_state(output_dir, run_name, ts, args, model, model_spec,
                          ds, stats, splits, n_params, sched_desc, device, log,
                          node_input_semantics):
    run_config = write_run_config(
        output_dir, run_name, ts, args, model, model_spec, ds, stats, splits,
        n_params, sched_desc, device, log, node_input_semantics)
    return {
        "run_config": run_config,
        "start_epoch": 1,
        "best_val": float("inf"),
        "best_epoch": 0,
        "patience": 0,
        "history": [],
        "monitor_best": None,
        "t0": time.time(),
    }


def _restore_training_state(payload, args, model, model_spec, ds, splits,
                            optimizer, scheduler, rng, log):
    state = restore_training_checkpoint(
        payload, model=model, optimizer=optimizer, scheduler=scheduler,
        rng=rng,
        expected_signature=resume_signature(args, model_spec, ds, splits))
    if len(state["history"]) != state["epoch"] or (
            state["history"]
            and state["history"][-1].get("epoch") != state["epoch"]):
        raise ValueError(
            "resume checkpoint history is inconsistent with its epoch")
    start_epoch = int(state["epoch"]) + 1
    log.info(
        f"Restored epoch={state['epoch']}  next={start_epoch}  "
        f"best_epoch={state['best_epoch']}  "
        f"best_val={float(state['best_val_loss']):.6g}")
    return {
        "run_config": payload["run_config"],
        "start_epoch": start_epoch,
        "best_val": float(state["best_val_loss"]),
        "best_epoch": int(state["best_epoch"]),
        "patience": int(state["patience"]),
        "history": list(state["history"]),
        "monitor_best": state["monitor_best"],
        "t0": time.time() - float(state["elapsed_time_s"]),
    }


def _maybe_update_monitor(args, do_metrics, ev, epoch, monitor_best):
    if not (getattr(args, "save_monitor_best", False) and do_metrics
            and np.isfinite(ev["auc"])
            and (monitor_best is None or ev["auc"] > monitor_best["auc"])):
        return monitor_best, False
    return {
        "epoch": epoch,
        "auc": float(ev["auc"]),
        "max_sic": float(ev["max_sic"]),
        "val_loss": float(ev["val_loss"]),
    }, True


def _write_checkpoints(output_dir, model, model_spec, run_config, state,
                       improved, monitor_improved):
    kwargs = dict(model=model, model_spec=model_spec, run_config=run_config,
                  training_state=state)
    save_training_checkpoint(output_dir / "last.pt", **kwargs)
    if improved:
        save_training_checkpoint(output_dir / "best.pt", **kwargs)
    if monitor_improved:
        save_training_checkpoint(output_dir / "monitor_best.pt", **kwargs)


def _log_epoch(log, epoch, train_losses, ev, do_metrics, improved, epoch_s):
    mark = " *" if improved else "  "
    row = (f"{epoch:>6}{mark}  {train_losses.total:8.4f}  "
           f"{ev['val_loss']:8.4f}  {ev['val_node_loss']:8.4f}  "
           f"{ev['val_edge_loss']:8.4f}  ")
    if do_metrics:
        row += (f"{ev['auc']:6.4f}  {ev['max_sic']:6.3f}  "
                f"{ev['eS_at_eB1e-2']:7.4f}  {ev['eS_at_eB1e-3']:7.4f}")
    else:
        row += f"{'--':>6}  {'--':>6}  {'--':>7}  {'--':>7}"
    log.info(row + f"  {epoch_s:5.0f}s")


def train(args):
    """Run one training job end to end; returns the final metrics dict."""
    output_dir, run_name, ts, log, resume_payload = _open_run(args)
    log.info(f"Run      : {run_name}")
    log.info(f"Output   : {output_dir}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    log.info(f"Device   : {device}")

    rng = _seed_run(args, log)
    ds, stats = build_dataset(args, log)
    splits = build_splits(ds, args, log, rng)
    model, model_spec, node_input_semantics, n_params = build_model(
        args, ds, device, log)
    optimizer, scheduler, sched_per_batch, total_steps, sched_desc = (
        build_optimizer_scheduler(args, model, ds, splits.train_idx, log))

    if resume_payload is None:
        ctx = _fresh_training_state(
            output_dir, run_name, ts, args, model, model_spec, ds, stats,
            splits, n_params, sched_desc, device, log, node_input_semantics)
    else:
        ctx = _restore_training_state(
            resume_payload, args, model, model_spec, ds, splits,
            optimizer, scheduler, rng, log)
    run_config = ctx["run_config"]
    start_epoch = ctx["start_epoch"]
    best_val = ctx["best_val"]
    best_epoch = ctx["best_epoch"]
    patience = ctx["patience"]
    history = ctx["history"]
    monitor_best = ctx["monitor_best"]
    t0 = ctx["t0"]

    line = "─" * 92
    log.info(f"\n{line}")
    log.info(f"{'Epoch':>6}  {'train':>8}  {'val':>8}  {'val_node':>8}  "
             f"{'val_edge':>8}  "
             f"{'AUC':>6}  {'SIC':>6}  {'eS@1e2':>7}  {'eS@1e3':>7}  {'time':>6}")
    log.info(line)

    for epoch in range(start_epoch, args.epochs + 1):
        ep_t0 = time.time()
        train_losses = train_one_epoch(
            model, ds, splits, optimizer, scheduler, sched_per_batch,
            total_steps, args, device, rng)
        do_metrics = (len(splits.sig_idx) > 0
                      and args.eval_interval > 0
                      and (epoch % args.eval_interval == 0
                           or epoch == args.epochs))
        ev = evaluate_epoch(model, ds, splits, args, device,
                            compute_metrics=do_metrics)

        improved = ev["val_loss"] < best_val
        if improved:
            best_val, best_epoch, patience = ev["val_loss"], epoch, 0
        else:
            patience += 1
        monitor_best, monitor_improved = _maybe_update_monitor(
            args, do_metrics, ev, epoch, monitor_best)

        entry = {
            "epoch": epoch, "lr": scheduler.get_last_lr()[0],
            "train_loss": train_losses.total,
            "train_node_loss": train_losses.node,
            "train_edge_loss": train_losses.edge,
            **ev, "improved": improved,
            "monitor_auc_improved": monitor_improved,
            "epoch_time_s": time.time() - ep_t0,
        }
        if (getattr(args, "topo_reg", "none") or "none") != "none":
            entry["train_reg"] = float(train_losses.reg)
            entry["lambda_topo"] = float(args.lambda_topo)
        if do_metrics:
            entry["score_sep"] = ev["mean_score_sig"] - ev["mean_score_bkg"]
        history.append(entry)
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        checkpoint_state = capture_training_state(
            epoch=epoch, optimizer=optimizer, scheduler=scheduler,
            best_val=best_val, best_epoch=best_epoch, patience=patience,
            history=history, monitor_best=monitor_best, rng=rng,
            elapsed_time_s=time.time() - t0)
        _write_checkpoints(
            output_dir, model, model_spec, run_config, checkpoint_state,
            improved, monitor_improved)
        _log_epoch(log, epoch, train_losses, ev, do_metrics, improved,
                   entry["epoch_time_s"])

        if not args.no_early_stop and patience >= args.patience:
            log.info(f"\nEarly stop at epoch {epoch}  "
                     f"(patience={args.patience})")
            break

    log.info(f"{'─'*80}")
    last_val = history[-1]["val_loss"] if history else float("nan")
    n_epochs = int(history[-1]["epoch"]) if history else 0
    return final_evaluation(
        ds, splits, output_dir, args, device, log, best_val, best_epoch,
        last_val, n_epochs, t0, run_name, monitor_best)
