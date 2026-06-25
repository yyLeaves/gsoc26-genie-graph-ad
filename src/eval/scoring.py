import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from src.eval.metrics import auc_score, compute_sic, signal_eff_at_bkg_eff
from src.models import NodeGraphAE, EdgeGraphAE


def load_model(checkpoint, edge_dim, device):
    """Rebuild model from the sibling config.json + load weights.

    `edge_dim` is kept for old configs; new runs persist it in config.json so a
    checkpoint is self-contained.
    """
    checkpoint = Path(checkpoint)
    cfg = json.loads((checkpoint.parent / "config.json").read_text())["model"]
    mtype, in_dim = cfg["type"], cfg["in_dim"]
    if mtype in {"edgeae", "relae"}:
        model = EdgeGraphAE(in_dim=in_dim, edge_dim=cfg.get("edge_dim", edge_dim),
                           hidden_dim=cfg["hidden_dim"],
                           latent_dim=cfg["latent_dim"],
                           edge_weight=cfg.get("edge_weight", 1.0),
                           aggr=cfg.get("aggr", "mean"))
    else:
        model = NodeGraphAE(in_dim=in_dim, backbone=cfg["backbone"],
                        hidden_dim=cfg["hidden_dim"],
                        latent_dim=cfg["latent_dim"],
                        use_bn=cfg.get("use_bn", True))
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        details = {
            "type": mtype,
            "in_dim": in_dim,
            "edge_dim": cfg.get("edge_dim", edge_dim),
            "hidden_dim": cfg.get("hidden_dim"),
            "latent_dim": cfg.get("latent_dim"),
            "use_bn": cfg.get("use_bn", True),
            "edge_weight": cfg.get("edge_weight", 1.0),
            "aggr": cfg.get("aggr", "mean"),
        }
        raise RuntimeError(
            f"Failed to load checkpoint {checkpoint} with model config "
            f"{details}. The checkpoint and config.json likely describe "
            "different model architectures."
        ) from exc
    return model.to(device).eval()


@torch.no_grad()
def score_jets(model, ds, device, batch_size=2048, indices=None):
    """Return one anomaly score per jet graph. Kept for debugging/tests; AD
    metrics use event-level score_dataset below."""
    idx = list(range(len(ds))) if indices is None else sorted(int(i) for i in indices)
    scores = []
    for start in range(0, len(idx), batch_size):
        items = [ds[i] for i in idx[start:start + batch_size]]
        batch = Batch.from_data_list(items).to(device)
        scores.append(model.anomaly_score(batch).cpu().numpy())
    return np.concatenate(scores)


EVENT_SCORE_AGGREGATIONS = ("sum", "mean", "max", "min", "pt_weighted")


def _event_score_arrays(jet_scores, event_ids, labels, jet_strengths=None,
                        aggregation: str = "sum"):
    """Aggregate jet scores to event-level arrays.

    aggregation:
      - sum: score(event)=Σ score(jet), matching the reference repo.
      - mean: score(event)=mean_j score(jet), sum normalized by selected jets.
      - max: score(event)=max_j score(jet), only the most anomalous jet counts.
      - min: score(event)=min_j score(jet), all selected jets must be anomalous.
      - pt_weighted: score(event)=Σ w_j score(jet), w_j=jet_pt/Σ_event jet_pt.
    """
    if aggregation not in EVENT_SCORE_AGGREGATIONS:
        raise ValueError(
            f"aggregation must be one of {EVENT_SCORE_AGGREGATIONS}, "
            f"got {aggregation!r}")
    if aggregation == "pt_weighted" and jet_strengths is None:
        raise ValueError("pt_weighted event scoring requires jet_strengths")

    event_scores, event_labels, event_weights, event_counts = {}, {}, {}, {}
    if jet_strengths is None:
        jet_strengths = np.ones_like(jet_scores, dtype=np.float64)
    for score, event_id, label, strength in zip(
            jet_scores, event_ids, labels, jet_strengths):
        event_id = int(event_id)
        label = int(label)
        strength = float(strength)
        if aggregation == "sum":
            event_scores[event_id] = event_scores.get(event_id, 0.0) + float(score)
        elif aggregation == "mean":
            event_scores[event_id] = event_scores.get(event_id, 0.0) + float(score)
            event_counts[event_id] = event_counts.get(event_id, 0) + 1
        elif aggregation == "max":
            event_scores[event_id] = max(event_scores.get(event_id, -np.inf),
                                         float(score))
        elif aggregation == "min":
            event_scores[event_id] = min(event_scores.get(event_id, np.inf),
                                         float(score))
        else:
            event_scores[event_id] = (
                event_scores.get(event_id, 0.0) + float(score) * strength)
            event_weights[event_id] = event_weights.get(event_id, 0.0) + strength
        if event_id in event_labels and event_labels[event_id] != label:
            raise ValueError(
                f"inconsistent labels inside event_id={event_id}: "
                f"{event_labels[event_id]} vs {label}")
        event_labels[event_id] = label
    ordered = sorted(event_scores)
    if aggregation == "pt_weighted":
        scores = [event_scores[e] / max(event_weights[e], 1e-12)
                  for e in ordered]
    elif aggregation == "mean":
        scores = [event_scores[e] / event_counts[e] for e in ordered]
    else:
        scores = [event_scores[e] for e in ordered]
    return (np.array(scores, dtype=np.float64),
            np.array([event_labels[e] for e in ordered], dtype=np.int64),
            np.array(ordered, dtype=np.int64))


@torch.no_grad()
def score_events(model, ds, device, batch_size=2048, indices=None,
                 aggregation: str = "sum"):
    """Return (scores, labels, event_ids) at event level.

    Each Data item is still one jet graph; this sums all selected jets with the
    same event_id, matching the reference repo's event score = jet0 + jet1.
    """
    if not getattr(ds, "has_event_ids", lambda: False)():
        raise ValueError(
            "Event-level scoring requires event_ids in metadata. Rebuild point "
            "cloud and graph shards with the current preprocess/build_subset.")

    idx = list(range(len(ds))) if indices is None else sorted(int(i) for i in indices)
    jet_scores = score_jets(model, ds, device, batch_size=batch_size, indices=idx)
    event_ids = np.asarray(ds.event_ids[idx], dtype=np.int64)
    labels = np.asarray(ds.labels[idx], dtype=np.int64)
    jet_strengths = None
    if aggregation == "pt_weighted":
        jet_strengths = np.array([float(ds[i].pt.sum()) for i in idx],
                                 dtype=np.float64)
    return _event_score_arrays(jet_scores, event_ids, labels, jet_strengths,
                               aggregation)


@torch.no_grad()
def score_dataset(model, ds, device, batch_size=2048, indices=None,
                  aggregation: str = "sum"):
    """Return event-level anomaly scores only."""
    scores, _, _ = score_events(model, ds, device, batch_size, indices,
                                aggregation)
    return scores


def report(scores, labels):
    """AD metrics (high score = anomalous): AUC, max-SIC, ε_S at 100x/1000x rej."""
    max_sic, thr, _, _ = compute_sic(scores, labels)
    return {
        "auc": auc_score(scores, labels),
        "max_sic": max_sic,
        "best_threshold": thr,
        "eS_at_eB1e-2": signal_eff_at_bkg_eff(scores, labels, 0.01),
        "eS_at_eB1e-3": signal_eff_at_bkg_eff(scores, labels, 0.001),
        "n_signal": int((labels == 1).sum()),
        "n_background": int((labels == 0).sum()),
    }
