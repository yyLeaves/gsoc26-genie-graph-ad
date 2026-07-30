"""Versioned persistence format shared by training and inference.

A checkpoint owns the information required to reconstruct its model. Training
checkpoints additionally carry optimizer, scheduler, history, and RNG state so
an interrupted run can continue exactly from an epoch boundary.
"""

from pathlib import Path
from typing import Any, Mapping

import torch

CHECKPOINT_FORMAT = "genie.graph_ae"
CHECKPOINT_VERSION = 1

_TRAINING_KEYS = {
    "epoch",
    "optimizer_state",
    "scheduler_state",
    "best_val_loss",
    "best_epoch",
    "patience",
    "history",
    "monitor_best",
    "numpy_rng_state",
    "torch_rng_state",
    "cuda_rng_state",
    "elapsed_time_s",
}


def save_checkpoint(
    path: str | Path,
    *,
    model_spec: Mapping[str, Any],
    model_state: Mapping[str, Any],
    run_config: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
) -> None:
    """Atomically write one self-describing checkpoint bundle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model": {
            "spec": dict(model_spec),
            "state": dict(model_state),
        },
        "run_config": dict(run_config) if run_config is not None else None,
        "training": (
            dict(training_state) if training_state is not None else None
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load and validate the common model portion of a checkpoint bundle."""
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} is not a checkpoint bundle")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"checkpoint {path} has unsupported format "
            f"{payload.get('format')!r}; expected {CHECKPOINT_FORMAT!r}"
        )
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint {path} has unsupported version "
            f"{payload.get('version')!r}; expected {CHECKPOINT_VERSION}"
        )
    model = payload.get("model")
    if not isinstance(model, dict) or set(model) != {"spec", "state"}:
        raise ValueError(
            f"checkpoint {path} must contain model.spec and model.state"
        )
    if not isinstance(model["spec"], dict) or not isinstance(
        model["state"], dict
    ):
        raise ValueError(
            f"checkpoint {path} has invalid model.spec or model.state"
        )
    return payload


def require_training_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated resumable training state from a loaded bundle."""
    state = payload.get("training")
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain resumable training state")
    missing = sorted(_TRAINING_KEYS - set(state))
    if missing:
        raise ValueError(
            f"checkpoint training state is missing required fields {missing}"
        )
    return state


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "load_checkpoint",
    "require_training_state",
    "save_checkpoint",
]
