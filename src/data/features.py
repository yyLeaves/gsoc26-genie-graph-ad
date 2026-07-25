"""Node feature representations for jet constituents and subjets."""

import numpy as np

from .kinematics import relative_coords

__all__ = ["FEATURE_DIM", "compute_node_features"]

FEATURE_DIM = {"raw": 3, "normalized": 4, "log_phys": 5}


def compute_node_features(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Build one node-feature row per nonzero, pT-sorted constituent."""
    pt_sum = pt.sum() + 1e-10
    d_eta, d_phi = relative_coords(pt, eta, phi)
    delta_r = np.sqrt(d_eta**2 + d_phi**2)

    if mode == "raw":
        return np.stack([pt, eta, phi], axis=1)
    if mode == "normalized":
        return np.stack([pt / pt_sum, d_eta, d_phi, delta_r], axis=1)
    if mode == "log_phys":
        return np.stack([
            np.log(pt + 1e-10),
            np.log(pt / pt_sum + 1e-10),
            d_eta,
            d_phi,
            np.log(delta_r + 1e-3),
        ], axis=1)
    raise ValueError(
        f"Unknown feature mode {mode!r}. "
        "Use 'raw', 'normalized', or 'log_phys'."
    )
