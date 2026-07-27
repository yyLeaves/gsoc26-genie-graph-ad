from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class SICResult:
    maximum: float
    threshold: float
    signal_efficiency: np.ndarray
    background_efficiency: np.ndarray


def _binary_scores(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(scores) != len(labels):
        raise ValueError(
            f"scores and labels must have the same length, got "
            f"{len(scores)} and {len(labels)}"
        )
    return scores, labels


def _sic(scores: np.ndarray, labels: np.ndarray,
         min_background_efficiency: float) -> SICResult:
    if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
        empty = np.array([], dtype=np.float64)
        return SICResult(float("nan"), float("nan"), empty, empty)

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    is_sig = sorted_labels == 1
    is_bkg = sorted_labels == 0
    group_end = np.r_[np.diff(sorted_scores) != 0, True]
    group_idx = np.flatnonzero(group_end)
    thresholds = sorted_scores[group_idx]
    eps_S = np.cumsum(is_sig)[group_idx] / is_sig.sum()
    eps_B = np.cumsum(is_bkg)[group_idx] / is_bkg.sum()
    valid = (eps_B >= min_background_efficiency) & (eps_B > 0)
    if not valid.any():
        return SICResult(0.0, float(thresholds[0]), eps_S, eps_B)
    with np.errstate(divide="ignore", invalid="ignore"):
        sic = np.where(valid, eps_S / np.sqrt(eps_B), 0.0)
    best = int(np.argmax(sic))
    return SICResult(
        maximum=float(sic[best]),
        threshold=float(thresholds[best]),
        signal_efficiency=eps_S,
        background_efficiency=eps_B,
    )


def _signal_eff(scores: np.ndarray, labels: np.ndarray,
                background_efficiency: float) -> float:
    bkg, sig = scores[labels == 0], scores[labels == 1]
    if len(bkg) == 0 or len(sig) == 0:
        return float("nan")
    values, counts = np.unique(bkg, return_counts=True)
    values, counts = values[::-1], counts[::-1]
    cumulative = np.cumsum(counts)
    valid = cumulative / len(bkg) <= background_efficiency + 1e-15
    if not valid.any():
        return 0.0
    thr = values[np.flatnonzero(valid)[-1]]
    return float((sig >= thr).mean())


def _classification(scores: np.ndarray, labels: np.ndarray,
                    threshold: float) -> dict:
    predicted = scores >= threshold
    signal = labels == 1
    tp = int(np.sum(predicted & signal))
    fp = int(np.sum(predicted & ~signal))
    tn = int(np.sum(~predicted & ~signal))
    fn = int(np.sum(~predicted & signal))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    accuracy = (tp + tn) / len(labels) if len(labels) else float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_sic(scores: np.ndarray, labels: np.ndarray,
                min_background_efficiency: float = 0.001) -> SICResult:
    scores, labels = _binary_scores(scores, labels)
    if not 0 < min_background_efficiency <= 1:
        raise ValueError(
            f"background_efficiency must be in (0, 1], got "
            f"{min_background_efficiency}"
        )
    return _sic(scores, labels, min_background_efficiency)


def signal_eff_at_bkg_eff(scores: np.ndarray, labels: np.ndarray,
                          background_efficiency: float) -> float:
    scores, labels = _binary_scores(scores, labels)
    if not 0 < background_efficiency <= 1:
        raise ValueError(
            f"background_efficiency must be in (0, 1], got "
            f"{background_efficiency}"
        )
    return _signal_eff(scores, labels, background_efficiency)


def classification_metrics(scores: np.ndarray, labels: np.ndarray,
                           threshold: float) -> dict:
    scores, labels = _binary_scores(scores, labels)
    return _classification(scores, labels, threshold)


def best_f1_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    scores, labels = _binary_scores(scores, labels)
    if len(scores) == 0:
        return _classification(scores, labels, float("nan"))
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    is_signal = labels[order] == 1
    total_signal = int(is_signal.sum())
    if total_signal == 0:
        return _classification(scores, labels, float(sorted_scores[0]))
    group_end = np.r_[np.diff(sorted_scores) != 0, True]
    group_idx = np.flatnonzero(group_end)
    tp = np.cumsum(is_signal)[group_idx]
    selected = group_idx + 1
    fp = selected - tp
    precision = tp / selected
    recall = tp / total_signal
    denominator = precision + recall
    f1 = np.zeros_like(denominator, dtype=np.float64)
    np.divide(2 * precision * recall, denominator, out=f1,
              where=denominator > 0)
    threshold = float(sorted_scores[group_idx[int(np.argmax(f1))]])
    return _classification(scores, labels, threshold)


def summarize_scores(scores: np.ndarray, labels: np.ndarray) -> dict:
    scores, labels = _binary_scores(scores, labels)
    sic = _sic(scores, labels, 0.001)
    auc = (
        float("nan") if len(np.unique(labels)) < 2
        else float(roc_auc_score(labels, scores))
    )
    return {
        "auc": auc,
        "max_sic": sic.maximum,
        "best_sic_threshold": sic.threshold,
        "eS_at_eB1e-2": _signal_eff(scores, labels, 0.01),
        "eS_at_eB1e-3": _signal_eff(scores, labels, 0.001),
        "n_signal": int((labels == 1).sum()),
        "n_background": int((labels == 0).sum()),
    }
