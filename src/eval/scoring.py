from dataclasses import dataclass

import numpy as np
import torch

from src.data.iterate import prefetch, shard_iter
from src.data.grouping import group_events
from src.objectives.latent_cycle import latent_cycle_scores

EVENT_SCORE_AGGREGATIONS = ("sum", "mean", "max", "min", "pt_weighted")
CYCLE_SCORE_MODES = ("cycle", "cycle_local", "cycle_global")
ANOMALY_SCORE_MODES = ("reconstruction",) + CYCLE_SCORE_MODES


@dataclass(frozen=True)
class EventScores:
    scores: np.ndarray
    labels: np.ndarray
    event_ids: np.ndarray


def _scoring_indices(ds, indices, batch_size: int) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    idx = (
        np.arange(len(ds), dtype=np.int64)
        if indices is None
        else np.array(sorted(int(i) for i in indices), dtype=np.int64)
    )
    if idx.size == 0:
        raise ValueError("scoring requires at least one jet")
    if np.any(np.diff(idx) == 0):
        raise ValueError("scoring indices must be unique")
    if idx[0] < 0 or idx[-1] >= len(ds):
        raise IndexError(f"scoring indices must lie in [0, {len(ds)})")
    return idx


def _require_two_jets(ds, indices: np.ndarray) -> None:
    if not ds.meta["selection"]["require_two_jets"]:
        return
    event_ids = np.asarray(ds.event_ids, dtype=np.int64)[indices]
    jet_idx = np.asarray(ds.jet_idx, dtype=np.int64)[indices]
    unique_events, counts = np.unique(event_ids, return_counts=True)
    if np.any(counts != 2):
        bad = unique_events[counts != 2]
        raise ValueError(
            "strict two-jet scoring requires exactly two jets per event; "
            f"{bad.size:,} event(s) violate this, examples={bad[:5].tolist()}"
        )
    order = np.lexsort((jet_idx, event_ids))
    pairs = jet_idx[order].reshape(-1, 2)
    valid = (pairs[:, 0] == 0) & (pairs[:, 1] == 1)
    if not valid.all():
        bad = unique_events[~valid]
        raise ValueError(
            "strict two-jet scoring requires jet_idx={{0,1}} once per event; "
            f"{bad.size:,} event(s) violate this, examples={bad[:5].tolist()}"
        )


def graph_pt_sums(batch) -> torch.Tensor:
    sums = torch.zeros(
        batch.num_graphs, dtype=batch.pt.dtype, device=batch.pt.device)
    return sums.scatter_add_(0, batch.batch, batch.pt).cpu()


def aggregate_event_scores(
    jet_scores,
    event_ids,
    labels,
    jet_strengths=None,
    aggregation: str = "sum",
) -> EventScores:
    if aggregation not in EVENT_SCORE_AGGREGATIONS:
        raise ValueError(
            f"aggregation must be one of {EVENT_SCORE_AGGREGATIONS}, "
            f"got {aggregation!r}"
        )
    if aggregation == "pt_weighted" and jet_strengths is None:
        raise ValueError("pt_weighted event scoring requires jet_strengths")

    jet_scores = np.asarray(jet_scores, dtype=np.float64).reshape(-1)
    event_ids = np.asarray(event_ids, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if jet_strengths is not None:
        jet_strengths = np.asarray(jet_strengths, dtype=np.float64).reshape(-1)
    if not (len(jet_scores) and len(jet_scores) == len(event_ids) == len(labels)
            and (jet_strengths is None or len(jet_strengths) == len(jet_scores))):
        raise ValueError(
            "jet_scores, event_ids, labels, and any jet_strengths must be "
            "non-empty and the same length"
        )

    groups = group_events(
        event_ids, labels, label_error="inconsistent labels inside event ids")
    sorted_scores = jet_scores[groups.order]
    starts = groups.starts

    if aggregation == "sum":
        scores = np.add.reduceat(sorted_scores, starts)
    elif aggregation == "mean":
        scores = np.add.reduceat(sorted_scores, starts) / groups.counts
    elif aggregation == "max":
        scores = np.maximum.reduceat(sorted_scores, starts)
    elif aggregation == "min":
        scores = np.minimum.reduceat(sorted_scores, starts)
    else:
        sorted_strengths = jet_strengths[groups.order]
        weighted = np.add.reduceat(sorted_scores * sorted_strengths, starts)
        total = np.add.reduceat(sorted_strengths, starts)
        scores = weighted / np.maximum(total, 1e-12)
    return EventScores(
        scores=np.asarray(scores, dtype=np.float64),
        labels=groups.labels,
        event_ids=groups.event_ids,
    )


@torch.no_grad()
def score_events(model, ds, device, batch_size=2048, indices=None,
                 aggregation: str = "sum",
                 score_mode: str = "reconstruction",
                 cycle_global_mode: str = "pooled") -> EventScores:
    idx = _scoring_indices(ds, indices, batch_size)
    _require_two_jets(ds, idx)
    if score_mode not in ANOMALY_SCORE_MODES:
        raise ValueError(
            f"score_mode must be one of {ANOMALY_SCORE_MODES}, "
            f"got {score_mode!r}")
    jet_scores, event_ids, labels, strengths = [], [], [], []
    model.eval()
    for batch in prefetch(shard_iter(
            ds, idx, batch_size,
            shuffle_shards=False, shuffle_within=False, device=device),
            depth=2):
        if score_mode == "reconstruction":
            scores = model.anomaly_score(batch)
        else:
            output, _, _ = model._reconstruct(batch)
            cycle = latent_cycle_scores(model, output, batch,
                                        global_mode=cycle_global_mode)
            scores = {
                "cycle": cycle.total,
                "cycle_local": cycle.local,
                "cycle_global": cycle.global_,
            }[score_mode]
        jet_scores.append(scores.view(-1).cpu())
        event_ids.append(batch.event_id.view(-1).cpu())
        labels.append(batch.y.view(-1).cpu())
        if aggregation == "pt_weighted":
            strengths.append(graph_pt_sums(batch))
    return aggregate_event_scores(
        torch.cat(jet_scores).numpy(),
        torch.cat(event_ids).numpy(),
        torch.cat(labels).numpy(),
        torch.cat(strengths).numpy() if strengths else None,
        aggregation,
    )
