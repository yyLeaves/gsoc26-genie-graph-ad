"""Physics-motivated paired views for selected jet constituents.

The transformations in this module deliberately operate on constituent
kinematics, before exclusive-kT reclustering and graph construction.  This
keeps node features, Unique-k connectivity, and edge features mutually
consistent.  Labels are never used to construct a view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch_geometric.data import Data

from .build_graph import GraphConfig, with_edges
from .build_subjets import _recluster_subjets
from .extractor import jet_to_data
from .kinematics import Jet, relative_coords

__all__ = [
    "PhysicsViewConfig",
    "augment_jet",
    "detector_smear",
    "make_graph_view",
    "soft_collinear_split",
]

_VIEW_MODES = frozenset({"soft_collinear", "detector", "combined"})


@dataclass(frozen=True, slots=True, kw_only=True)
class PhysicsViewConfig:
    """Parameters for one deterministic physics-view family.

    Momentum values are assumed to use GeV, matching the LHCO datasets.
    The defaults are intentionally conservative and must pass the kinematic
    gate before being used in training.
    """

    mode: str
    seed: int = 123
    split_fraction_min: float = 0.01
    split_fraction_max: float = 0.05
    split_dr_min: float = 0.002
    split_dr_max: float = 0.02
    min_parent_pt_fraction: float = 0.01
    pt_log_sigma: float = 0.02
    angular_scale_gev: float = 0.1
    angular_sigma_max: float = 0.05

    def __post_init__(self) -> None:
        if self.mode not in _VIEW_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VIEW_MODES)}, got {self.mode!r}")
        if not 0.0 < self.split_fraction_min <= self.split_fraction_max < 1.0:
            raise ValueError("split fractions must satisfy 0 < min <= max < 1")
        if not 0.0 <= self.split_dr_min <= self.split_dr_max:
            raise ValueError("split angular scales must satisfy 0 <= min <= max")
        if not 0.0 <= self.min_parent_pt_fraction < 1.0:
            raise ValueError("min_parent_pt_fraction must lie in [0, 1)")
        if self.pt_log_sigma < 0.0:
            raise ValueError("pt_log_sigma must be non-negative")
        if self.angular_scale_gev < 0.0:
            raise ValueError("angular_scale_gev must be non-negative")
        if self.angular_sigma_max < 0.0:
            raise ValueError("angular_sigma_max must be non-negative")


def _copy_jet(jet: Jet) -> Jet:
    return tuple(np.asarray(values, dtype=np.float64).copy()
                 for values in jet)  # type: ignore[return-value]


def _wrap_phi(phi: np.ndarray) -> np.ndarray:
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def _circular_axis(pt: np.ndarray, phi: np.ndarray) -> float:
    return float(np.arctan2(
        (pt * np.sin(phi)).sum(), (pt * np.cos(phi)).sum()))


def _align_phi_axis(pt: np.ndarray, phi: np.ndarray,
                    target_axis: float) -> np.ndarray:
    shift = float(_wrap_phi(np.asarray(target_axis - _circular_axis(pt, phi))))
    return _wrap_phi(phi + shift)


def _sorted_jet(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray) -> Jet:
    order = np.argsort(-pt, kind="stable")
    return pt[order], eta[order], _wrap_phi(phi[order])


def soft_collinear_split(jet: Jet, rng: np.random.Generator, *,
                         fraction_min: float = 0.01,
                         fraction_max: float = 0.05,
                         dr_min: float = 0.002,
                         dr_max: float = 0.02,
                         min_parent_pt_fraction: float = 0.01) -> Jet:
    """Split one constituent into a hard and nearby soft daughter.

    Scalar pT and the local pT-weighted angular centroid are conserved exactly.
    A finite opening angle makes this a quasi-collinear, rather than exact
    four-vector-preserving, transformation; its jet-mass drift is therefore a
    required downstream diagnostic.
    """
    pt, eta, phi = _copy_jet(jet)
    if pt.size == 0:
        return pt, eta, phi
    original_phi_axis = _circular_axis(pt, phi)

    pt_fraction = pt / pt.sum()
    candidates = np.flatnonzero(pt_fraction >= min_parent_pt_fraction)
    if candidates.size == 0:
        candidates = np.arange(pt.size)
    weights = pt[candidates] / pt[candidates].sum()
    parent = int(rng.choice(candidates, p=weights))

    epsilon = float(rng.uniform(fraction_min, fraction_max))
    if dr_min == dr_max:
        separation = float(dr_min)
    elif dr_min == 0.0:
        separation = float(rng.uniform(0.0, dr_max))
    else:
        separation = float(np.exp(rng.uniform(np.log(dr_min), np.log(dr_max))))
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    direction = np.array([np.cos(angle), np.sin(angle)])

    # Place the daughters on opposite sides of the parent so their pT-weighted
    # local centroid remains at the original coordinate.
    hard_offset = -epsilon * separation * direction
    soft_offset = (1.0 - epsilon) * separation * direction
    parent_coord = np.array([eta[parent], phi[parent]])

    parent_pt = pt[parent]
    pt[parent] = (1.0 - epsilon) * parent_pt
    eta[parent], phi[parent] = parent_coord + hard_offset
    pt = np.append(pt, epsilon * parent_pt)
    soft_coord = parent_coord + soft_offset
    eta = np.append(eta, soft_coord[0])
    phi = np.append(phi, soft_coord[1])
    phi = _align_phi_axis(pt, phi, original_phi_axis)
    return _sorted_jet(pt, eta, phi)


def detector_smear(jet: Jet, rng: np.random.Generator, *,
                   pt_log_sigma: float = 0.02,
                   angular_scale_gev: float = 0.1,
                   angular_sigma_max: float = 0.05) -> Jet:
    """Apply a small shape-only detector-response view.

    Multiplicative pT noise is renormalized to preserve scalar jet pT.  Angular
    noise scales approximately as 1/pT and is capped for very soft particles.
    The result is recentered on the original pT-weighted jet axis.
    """
    pt, eta, phi = _copy_jet(jet)
    if pt.size == 0:
        return pt, eta, phi

    original_pt_sum = float(pt.sum())
    eta_axis = float(np.average(eta, weights=pt))
    phi_axis = _circular_axis(pt, phi)
    d_eta, d_phi = relative_coords(pt, eta, phi)

    if pt_log_sigma:
        pt *= np.exp(rng.normal(0.0, pt_log_sigma, size=pt.size))
        pt *= original_pt_sum / pt.sum()

    if angular_scale_gev:
        sigma = np.minimum(
            angular_scale_gev / np.maximum(pt, 1e-12), angular_sigma_max)
        d_eta += rng.normal(0.0, sigma)
        d_phi += rng.normal(0.0, sigma)

    # Recenter after both pT and angular perturbations.  This makes the view
    # shape-only in the local jet frame and avoids an artificial axis shift.
    d_eta -= np.average(d_eta, weights=pt)
    d_phi -= np.average(d_phi, weights=pt)
    eta = eta_axis + d_eta
    phi = _align_phi_axis(pt, _wrap_phi(phi_axis + d_phi), phi_axis)
    return _sorted_jet(pt, eta, phi)


def augment_jet(jet: Jet, config: PhysicsViewConfig,
                rng: np.random.Generator) -> Jet:
    """Apply one configured view without consulting labels."""
    transformed = _copy_jet(jet)
    if config.mode in {"soft_collinear", "combined"}:
        transformed = soft_collinear_split(
            transformed,
            rng,
            fraction_min=config.split_fraction_min,
            fraction_max=config.split_fraction_max,
            dr_min=config.split_dr_min,
            dr_max=config.split_dr_max,
            min_parent_pt_fraction=config.min_parent_pt_fraction,
        )
    if config.mode in {"detector", "combined"}:
        transformed = detector_smear(
            transformed,
            rng,
            pt_log_sigma=config.pt_log_sigma,
            angular_scale_gev=config.angular_scale_gev,
            angular_sigma_max=config.angular_sigma_max,
        )
    return transformed


def _rng_for_graph(data: Data, config: PhysicsViewConfig,
                   view_index: int) -> np.random.Generator:
    event_id = int(data.event_id.item())
    jet_idx = int(data.jet_idx.item())
    seed = np.random.SeedSequence(
        [int(config.seed), event_id, jet_idx, int(view_index)])
    return np.random.default_rng(seed)


def make_graph_view(
    constituent_data: Data,
    config: PhysicsViewConfig,
    *,
    graph_config: GraphConfig,
    n_subjets: int = 30,
    features: str = "log_phys",
    view_index: int = 0,
) -> Data:
    """Transform one selected constituent jet, then recluster and rebuild it."""
    missing = [name for name in ("pt", "eta", "phi", "event_id", "jet_idx",
                                  "mjj", "y")
               if getattr(constituent_data, name, None) is None]
    if missing:
        raise ValueError(f"constituent Data is missing required fields {missing}")
    jet = tuple(
        getattr(constituent_data, name).detach().cpu().numpy()
        for name in ("pt", "eta", "phi")
    )
    rng = _rng_for_graph(constituent_data, config, view_index)
    transformed = augment_jet(jet, config, rng)
    subjets = _recluster_subjets(transformed, n_subjets)
    point_cloud = jet_to_data(
        subjets,
        label=int(constituent_data.y.item()),
        jet_idx=int(constituent_data.jet_idx.item()),
        mjj=float(constituent_data.mjj.item()),
        features=features,
        event_id=int(constituent_data.event_id.item()),
    )
    return with_edges(point_cloud, graph_config)
