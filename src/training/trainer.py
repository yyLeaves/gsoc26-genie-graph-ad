from dataclasses import dataclass, fields
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
from src.training.options import validate_objectives
from src.objectives.latent_cycle import latent_cycle_scores, select_cycle_component
from src.objectives.perturbed_contrast import perturbed_graph_contrast
from src.objectives.edge_relation_contrast import EdgeRelationContrast
from src.objectives.edge_relation_vicreg import EdgeRelationVICReg
from src.objectives.topo_reg import (
    compute_topo_regs,
    parse_lambda_list,
    parse_topo_reg_spec,
)


@dataclass(frozen=True, slots=True)
class TrainEpochStats:
    """Per-epoch training means. ``total/node/edge`` are reconstruction only."""

    total: float
    node: float
    edge: float
    reg: float = 0.0
    cycle: float = 0.0
    contrast: float = 0.0
    relation: float = 0.0
    relation_diagnostic: float = 0.0

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
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(message)s")
    mode = "a" if append else "w"
    for handler in [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode),
    ]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
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
    contrast_weight = args.contrast_weight
    projection_dim = args.contrast_projection_dim
    contrast_source = args.contrast_source
    needs_projection = contrast_weight > 0 or args.cycle_global_mode != "pooled"
    spec = ModelSpec(
        type=args.model,
        in_dim=in_dim,
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        use_bn=(False if args.model in {
            "edge_graph", "reference_backbone_edge_graph",
        } else not args.no_bn),
        edge_dim=edge_dim,
        edge_features=edges["features"] if edge_dim else "log",
        edge_pt_scale=edges["pt_scale"] if edge_dim else "normalized",
        edge_weight=args.edge_weight,
        aggr=args.aggr,
        dyn_k=args.dyn_k,
        contrastive_projection_dim=(projection_dim
                                    if needs_projection else 0),
        contrastive_source=(contrast_source
                            if needs_projection else "latent"),
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
        pct_start = min(max(max(2, int(0.02 * total_steps)) / total_steps,
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
    log.info(f"Jet score  : {args.anomaly_score}")
    topo_reg = args.topo_reg
    if topo_reg != "none":
        regs = parse_topo_reg_spec(topo_reg)
        lams = parse_lambda_list(args.lambda_topo, len(regs))
        log.info(f"Topo reg  : {regs}  λ={lams}  "
                 f"normalize={args.topo_norm}  "
                 f"unique_k={args.unique_k}  (val/anomaly still recon-only)")
    cycle_weight = args.cycle_weight
    if cycle_weight:
        cycle_component = args.cycle_component
        log.info(f"Cycle     : EB3 {cycle_component}  β={cycle_weight}  "
                 f"global={args.cycle_global_mode}")
    contrast_weight = args.contrast_weight
    if contrast_weight:
        log.info(
            "Contrast  : clean/weight-perturbed graph embeddings  "
            f"α={contrast_weight}  η={args.perturb_scale}  "
            f"T={args.contrast_temperature}  "
            f"projection={args.contrast_projection_dim}  "
            f"source={args.contrast_source}"
        )
    relation_objective = args.relation_objective
    if relation_objective != "none":
        log.info(
            f"Relation  : {relation_objective}  "
            f"weight={args.relation_weight}  "
            f"node_mask={args.relation_node_mask}  "
            f"edge_mask={args.relation_edge_mask}"
        )
    return optimizer, scheduler, sched_per_batch, total_steps, sched_desc


def train_one_epoch(model, ds, splits, optimizer, scheduler, sched_per_batch,
                    total_steps, args, device, rng):
    """Run one training pass and return mean recon + optional topo-reg stats.

    Returned ``total/node/edge`` are *reconstruction* means using the selected
    node loss. When ``--topo_reg`` is set the optimized objective is
    ``recon + sum_i λ_i R_i``.
    """
    model.train()
    totals = {field.name: 0.0 for field in fields(TrainEpochStats)}
    n_seen = 0.0
    reg_names = parse_topo_reg_spec(args.topo_reg)
    reg_lams = parse_lambda_list(
        args.lambda_topo, len(reg_names))
    if args.relation_objective == "nce":
        relation_objective = EdgeRelationContrast(
            edge_dim=int(model.edge_dim),
            mask_fraction=args.relation_edge_mask,
            temperature=args.relation_temperature,
        )
    elif args.relation_objective == "vicreg":
        relation_objective = EdgeRelationVICReg(
            node_mask_fraction=args.relation_node_mask,
            edge_mask_fraction=args.relation_edge_mask,
            moment_weight=args.relation_moment_weight,
        )
    else:
        relation_objective = None
    for batch in prefetch(shard_iter(
            ds, splits.train_idx, args.batch_size,
            shuffle_shards=True, shuffle_within=True, rng=rng, device=device),
            depth=args.prefetch_batches):
        optimizer.zero_grad()
        output, node_target, edge_target = model._reconstruct(batch)
        losses = mean_loss(reconstruction_scores(
            output, node_target, batch,
            edge_target=edge_target,
            edge_weight=getattr(model, "edge_weight", 1.0),
            node_reduction=args.node_reconstruction,
        ))
        if reg_names and any(lam != 0.0 for lam in reg_lams):
            weighted, unweighted = compute_topo_regs(
                reg_names, output.latent, batch, lambdas=reg_lams,
                unique_k=args.unique_k, normalize=args.topo_norm,
                model=model, node_target=node_target,
                edge_target=edge_target)
            opt_loss = losses.total + weighted
            reg = unweighted
        else:
            reg = losses.total.new_zeros(())
            opt_loss = losses.total
        if args.cycle_weight:
            cycle = select_cycle_component(
                latent_cycle_scores(model, output, batch,
                                    global_mode=args.cycle_global_mode),
                args.cycle_component).mean()
            opt_loss = opt_loss + args.cycle_weight * cycle
        else:
            cycle = losses.total.new_zeros(())
        if args.contrast_weight and batch.num_graphs > 1:
            contrast = perturbed_graph_contrast(
                model, output.latent, node_target, batch, edge_target,
                scale=args.perturb_scale,
                temperature=args.contrast_temperature,
                source=args.contrast_source,
                center=args.contrast_center,
            ).loss
            opt_loss = opt_loss + args.contrast_weight * contrast
        else:
            contrast = losses.total.new_zeros(())
        if relation_objective is not None:
            if args.relation_objective == "nce":
                relation_terms = relation_objective(
                    model, node_target, batch.edge_index, edge_target,
                    batch.batch, batch.num_graphs)
                relation = relation_terms.loss
                relation_diagnostic = relation_terms.accuracy
            else:
                relation_terms = relation_objective(
                    model, output, node_target, batch.edge_index, edge_target,
                    batch.batch, batch.num_graphs)
                relation = relation_terms.loss
                relation_diagnostic = relation_terms.affected_fraction
            opt_loss = opt_loss + args.relation_weight * relation
        else:
            relation = losses.total.new_zeros(())
            relation_diagnostic = losses.total.new_zeros(())
        opt_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if sched_per_batch and scheduler.last_epoch + 1 < total_steps:
            scheduler.step()
        batch_stats = dict(
            total=losses.total, node=losses.node, edge=losses.edge,
            reg=reg, cycle=cycle, contrast=contrast, relation=relation,
            relation_diagnostic=relation_diagnostic)
        for name, value in batch_stats.items():
            totals[name] += value.item() * batch.num_graphs
        n_seen += batch.num_graphs
    if not sched_per_batch:
        scheduler.step()
    return TrainEpochStats(**{name: total / n_seen for name, total in totals.items()})


def _open_run(args):
    """Create a fresh run dir, or reopen an interrupted one for resume."""
    resume_value = args.resume
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
    if not (args.save_monitor_best and do_metrics
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
    validate_objectives(args)
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
        if (args.topo_reg) != "none":
            entry["train_reg"] = float(train_losses.reg)
            entry["lambda_topo"] = args.lambda_topo
        if args.cycle_weight:
            entry["train_cycle_loss"] = float(train_losses.cycle)
            entry["cycle_weight"] = args.cycle_weight
            entry["cycle_component"] = args.cycle_component
        if args.contrast_weight:
            entry["train_contrast_loss"] = float(train_losses.contrast)
            entry["contrast_weight"] = args.contrast_weight
            entry["perturb_scale"] = args.perturb_scale
            entry["contrast_temperature"] = args.contrast_temperature
            entry["contrast_source"] = args.contrast_source
        if args.relation_objective != "none":
            entry["train_relation_loss"] = float(train_losses.relation)
            entry["train_relation_diagnostic"] = float(
                train_losses.relation_diagnostic)
            entry["relation_objective"] = args.relation_objective
            entry["relation_weight"] = args.relation_weight
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
