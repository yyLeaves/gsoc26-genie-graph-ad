import numpy as np
from sklearn.metrics import roc_auc_score


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC; high score = anomalous. NaN if either class absent."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_sic(scores: np.ndarray, labels: np.ndarray,
                n_points: int = 1000, min_eps_B: float = 0.001):
    """max(ε_S / √ε_B). NaN if a class is absent.
    https://arxiv.org/html/2511.14832."""
    if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
        nan_curve = np.full(n_points, np.nan)
        return float("nan"), float("nan"), nan_curve, nan_curve
    thresholds = np.linspace(scores.min(), scores.max(), n_points)
    eps_S = np.array([(scores[labels == 1] >= t).mean() for t in thresholds])
    eps_B = np.array([(scores[labels == 0] >= t).mean() for t in thresholds])
    valid = eps_B >= min_eps_B
    if not valid.any():
        return 0.0, float(thresholds[0]), eps_S, eps_B
    with np.errstate(divide="ignore", invalid="ignore"):
        sic = np.where(valid, eps_S / np.sqrt(eps_B), 0.0)
    best = int(np.argmax(sic))
    return float(sic[best]), float(thresholds[best]), eps_S, eps_B


def signal_eff_at_bkg_eff(scores: np.ndarray, labels: np.ndarray,
                          eps_b: float) -> float:
    """ε_S at a fixed background efficiency. eps_b = ε_B = fraction of background
    let through the cut (0.01 → 100x rejection, 0.001 → 1000x). E10/E100,
    https://arxiv.org/pdf/1808.08992"""
    bkg, sig = scores[labels == 0], scores[labels == 1]
    if len(bkg) == 0 or len(sig) == 0:
        return float("nan")
    thr = np.quantile(bkg, 1.0 - eps_b)
    return float((sig >= thr).mean())
