import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from src.checkpoint import require_training_state, save_checkpoint
from src.data.shards import dataset_spec
from src.models import ModelSpec
from src.training.splits import Splits

_SPLIT_ROLES = ("train_idx", "val_idx", "sig_idx", "eval_idx", "test_idx")
_SPLIT_SET_NAMES = (
    ("train", "train_idx"),
    ("validation", "val_idx"),
    ("monitor_signal", "sig_idx"),
    ("epoch_evaluation", "eval_idx"),
    ("test", "test_idx"),
)


def _split_summary(ds, indices) -> dict:
    """Count jets and source events, keeping the two units explicit."""
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(ds.labels, dtype=np.int64)[indices]
    event_ids = np.asarray(ds.event_ids, dtype=np.int64)[indices]
    if indices.size == 0:
        event_labels = np.array([], dtype=np.int64)
    else:
        order = np.argsort(event_ids, kind="stable")
        sorted_ids, sorted_labels = event_ids[order], labels[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
        lo = np.minimum.reduceat(sorted_labels, starts)
        hi = np.maximum.reduceat(sorted_labels, starts)
        if np.any(lo != hi):
            bad = sorted_ids[starts[lo != hi]][:5].tolist()
            raise ValueError(
                f"split contains inconsistent labels inside events {bad}")
        event_labels = lo
    return {
        "n_jets": int(indices.size),
        "n_background_jets": int(np.sum(labels == 0)),
        "n_signal_jets": int(np.sum(labels == 1)),
        "n_events": int(event_labels.size),
        "n_background_events": int(np.sum(event_labels == 0)),
        "n_signal_events": int(np.sum(event_labels == 1)),
    }


def split_fingerprint(splits: Splits) -> str:
    """Hash the exact jet indices used by every split role."""
    digest = hashlib.sha256()
    for name in _SPLIT_ROLES:
        values = np.asarray(getattr(splits, name), dtype=np.int64)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(values.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _backfill_topo_defaults(sig: dict) -> dict:
    """Add topo_reg fields with defaults to a pre-topo resume_signature."""
    sig = {**sig}
    if "training" in sig:
        t = {**sig["training"]}
        t.setdefault("topo_reg", "none")
        t.setdefault("lambda_topo", 0.0)
        t.setdefault("unique_k", 6)
        sig["training"] = t
    return sig


def resume_signature(args, model_spec: ModelSpec, ds, splits: Splits) -> dict:
    """Fields that must match for an exact epoch-boundary resume."""
    return {
        "model": model_spec.to_dict(),
        "dataset_fingerprint_sha256": dataset_spec(ds.meta)["fingerprint_sha256"],
        "split_fingerprint_sha256": split_fingerprint(splits),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "lr_end": args.lr_end,
            "no_early_stop": args.no_early_stop,
            "patience": args.patience,
            "eval_interval": args.eval_interval,
            "event_score_aggregation": args.event_score_agg,
            "save_monitor_best": getattr(args, "save_monitor_best", False),
            "topo_reg": getattr(args, "topo_reg", "none") or "none",
            "lambda_topo": float(getattr(args, "lambda_topo", 0.0) or 0.0),
            "unique_k": int(getattr(args, "unique_k", 6)),
        },
    }


def _code_provenance() -> dict:
    """Git HEAD/dirty flag plus a content hash of the Python source tree."""
    repo_root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    paths = sorted((repo_root / "src").rglob("*.py"))
    paths.append(repo_root / "scripts" / "train_graph_ae.py")
    for path in paths:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    def git(*git_args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *git_args], cwd=repo_root, check=True,
                capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=normal")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "source_sha256": digest.hexdigest(),
    }


def _split_config(ds, args, splits: Splits) -> dict:
    return {
        "fraction": args.fraction,
        "seed": args.seed,
        "protocol": getattr(args, "split_protocol", "manifest"),
        "manifest": getattr(args, "split_manifest", None),
        "sets": {
            name: _split_summary(ds, getattr(splits, attr))
            for name, attr in _SPLIT_SET_NAMES
        },
        "requested_event_counts": {
            "train_background": getattr(args, "train_bkg_events", None),
            "validation_background": getattr(args, "val_bkg_events", None),
            "test_background": getattr(args, "test_bkg_events", None),
            "test_signal": getattr(args, "test_sig_events", None),
        },
        "fingerprint_sha256": split_fingerprint(splits),
    }


def write_run_config(output_dir, run_name, ts, args, model, model_spec, ds,
                     stats, splits, n_params, sched_desc, device, log,
                     node_input_semantics):
    """Dump config.json + split.npz at run start."""
    edges = ds.meta.get("edges") or {}
    config = {
        "run_name": run_name,
        "timestamp": ts,
        "output_dir": str(output_dir),
        "code": _code_provenance(),
        "model": model_spec.to_dict(),
        "model_summary": {
            "n_params": n_params,
            "architecture": str(model),
            "node_input_semantics": node_input_semantics,
        },
        "dataset": {
            "data_dir": str(args.data_dir),
            "strategy": edges.get("strategy", "none"),
            "features": ds.meta["nodes"]["features"],
            "n_total": stats["n_jets"],
            "n_signal": stats["n_signal"],
            "n_background": stats["n_background"],
            "mean_nodes": stats["mean_nodes"],
            "mean_edges": stats.get("mean_edges", 0.0),
            "shard_size": ds.shard_size,
            "n_shards": ds.n_shards,
            "spec": dataset_spec(ds.meta),
        },
        "split": _split_config(ds, args, splits),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "eval_interval": args.eval_interval,
            "cache_shards": args.cache_shards,
            "event_score_aggregation": args.event_score_agg,
            "save_monitor_best": getattr(args, "save_monitor_best", False),
            "topo_reg": getattr(args, "topo_reg", "none") or "none",
            "lambda_topo": float(getattr(args, "lambda_topo", 0.0) or 0.0),
            "unique_k": int(getattr(args, "unique_k", 6)),
        },
        "optimizer": {
            "type": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
        },
        "scheduler": {
            "type": args.scheduler, "lr_end": args.lr_end,
            "description": sched_desc,
        },
        "no_early_stop": args.no_early_stop,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_version": (
            str(torch.version.cuda) if torch.cuda.is_available() else None),
        "resume_signature": resume_signature(args, model_spec, ds, splits),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"\nConfig saved → {output_dir}/config.json")
    np.savez(output_dir / "split.npz",
             **{name: getattr(splits, name) for name in _SPLIT_ROLES})
    log.info(f"Split  saved → {output_dir}/split.npz")
    return config


def capture_training_state(
    *,
    epoch: int,
    optimizer,
    scheduler,
    best_val: float,
    best_epoch: int,
    patience: int,
    history: list[dict],
    monitor_best: dict | None,
    rng: np.random.Generator,
    elapsed_time_s: float,
) -> dict:
    """Capture all mutable state needed to continue at the next epoch."""
    return {
        "epoch": int(epoch),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "patience": int(patience),
        "history": history,
        "monitor_best": monitor_best,
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "elapsed_time_s": float(elapsed_time_s),
    }


def save_training_checkpoint(
    path: Path,
    *,
    model,
    model_spec: ModelSpec,
    run_config: dict,
    training_state: dict,
) -> None:
    save_checkpoint(
        path,
        model_spec=model_spec.to_dict(),
        model_state=model.state_dict(),
        run_config=run_config,
        training_state=training_state,
    )


def restore_training_checkpoint(
    payload: dict,
    *,
    model,
    optimizer,
    scheduler,
    rng: np.random.Generator,
    expected_signature: dict,
) -> dict:
    """Validate a run context and restore model/training/RNG state."""
    run_config = payload.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("resume checkpoint does not contain run_config")
    saved_sig = run_config.get("resume_signature")
    if saved_sig != expected_signature:
        # Allow resuming pre-topo checkpoints when topo_reg is "none"
        if saved_sig is not None:
            patched = _backfill_topo_defaults(saved_sig)
            if patched == expected_signature:
                saved_sig = patched
        if saved_sig != expected_signature:
            raise ValueError(
                "resume checkpoint does not match the requested model, "
                "dataset, split, or training schedule")

    state = require_training_state(payload)
    model.load_state_dict(payload["model"]["state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    rng.bit_generator.state = state["numpy_rng_state"]
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if state["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return state
