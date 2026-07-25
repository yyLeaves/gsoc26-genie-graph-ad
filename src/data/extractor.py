from dataclasses import dataclass
from typing import Iterator

import awkward as ak
import fastjet
import numpy as np
import torch
from torch_geometric.data import Data

from .features import FEATURE_DIM, compute_node_features
from .kinematics import Jet, dijet_mass, relative_coords, to_pseudojets

__all__ = ["JetConfig", "JetExtractor", "jet_to_data",
           "JET_RADIUS", "JET_SELECTIONS"]

JET_RADIUS = 1.0
JET_SELECTIONS = ("leading_pt", "min_pt_all")
_MAX_SELECTED_JETS = 2
_JET_DEFINITION = fastjet.JetDefinition(fastjet.antikt_algorithm, JET_RADIUS)


@dataclass(frozen=True, kw_only=True)
class JetConfig:
    """Jet-selection and node-feature specification. All fields are required."""

    features: str
    min_jet_pt: float
    min_particles: int
    jet_selection: str
    require_two_jets: bool

    def __post_init__(self) -> None:
        if self.features not in FEATURE_DIM:
            raise ValueError(
                f"features must be one of {sorted(FEATURE_DIM)}, "
                f"got {self.features!r}"
            )
        if self.min_jet_pt < 0:
            raise ValueError(
                f"min_jet_pt must be non-negative, got {self.min_jet_pt}"
            )
        if self.min_particles <= 0:
            raise ValueError(
                f"min_particles must be positive, got {self.min_particles}"
            )
        if self.jet_selection not in JET_SELECTIONS:
            raise ValueError(
                f"jet_selection must be one of {list(JET_SELECTIONS)}, "
                f"got {self.jet_selection!r}"
            )


def _pseudojets_by_event(data: np.ndarray) -> ak.Array:
    """Convert padded ``(pT, eta, phi)`` slots to event-wise pseudojets."""
    pt, eta, phi = data[..., 0], data[..., 1], data[..., 2]
    active = pt > 0
    particles_per_event = active.sum(axis=1)
    return ak.unflatten(
        to_pseudojets(pt[active], eta[active], phi[active]),
        particles_per_event,
    )


def _select_constituents(
    cluster_sequence,
    cfg: JetConfig,
) -> tuple[ak.Array, np.ndarray | None]:
    """Select up to two jets and return their constituents.

    ``min_pt_all`` applies the threshold to every returned jet. ``leading_pt``
    applies it only to the leading jet and returns a mask for restoring events
    rejected by that requirement.
    """
    if cfg.jet_selection == "min_pt_all":
        jets = cluster_sequence.inclusive_jets(min_pt=cfg.min_jet_pt)
        constituents = cluster_sequence.constituents(min_pt=cfg.min_jet_pt)
        jet_pt = np.sqrt(jets["px"]**2 + jets["py"]**2)
        descending_order = ak.argsort(jet_pt, axis=1, ascending=False)
        constituents = constituents[descending_order][:, :_MAX_SELECTED_JETS]
        return constituents, None

    jets = cluster_sequence.inclusive_jets()
    constituents = cluster_sequence.constituents()
    jet_pt = np.sqrt(jets["px"]**2 + jets["py"]**2)
    descending_order = ak.argsort(jet_pt, axis=1, ascending=False)
    constituents = constituents[descending_order][:, :_MAX_SELECTED_JETS]
    ordered_jet_pt = jet_pt[descending_order]
    leading_jet_pt = ak.fill_none(ak.firsts(ordered_jet_pt, axis=1), -np.inf)
    selected_events = ak.to_numpy(leading_jet_pt >= cfg.min_jet_pt)
    return constituents[selected_events], selected_events


def _constituents_to_event_jets(constituents: ak.Array) -> list[list[Jet]]:
    """Decode selected FastJet constituents with batched NumPy conversion."""
    jets_per_event = ak.to_numpy(ak.num(constituents, axis=1))
    flattened_jets = ak.flatten(constituents, axis=1)
    constituents_per_jet = ak.to_numpy(ak.num(flattened_jets, axis=1))
    if len(constituents_per_jet) == 0:
        return [[] for _ in jets_per_event]

    flattened_constituents = ak.flatten(flattened_jets, axis=1)
    px, py, pz = (
        ak.to_numpy(flattened_constituents[field])
        for field in ("px", "py", "pz")
    )
    constituent_pt = np.sqrt(px**2 + py**2)
    constituent_eta = np.arcsinh(pz / (constituent_pt + 1e-10))
    constituent_phi = np.arctan2(py, px)

    jet_boundaries = np.cumsum(constituents_per_jet)[:-1]
    jets = []
    for pt, eta, phi in zip(
        np.split(constituent_pt, jet_boundaries),
        np.split(constituent_eta, jet_boundaries),
        np.split(constituent_phi, jet_boundaries),
    ):
        descending_order = np.argsort(-pt)
        jets.append((
            pt[descending_order],
            eta[descending_order],
            phi[descending_order],
        ))

    event_boundaries = np.concatenate(([0], np.cumsum(jets_per_event)))
    return [
        jets[start:stop]
        for start, stop in zip(event_boundaries[:-1], event_boundaries[1:])
    ]


def jet_to_data(jet: Jet, label: int, jet_idx: int, mjj: float,
                features: str, event_id: int) -> Data:
    """Convert one jet's constituent arrays to a PyG Data object."""
    pt, eta, phi = jet
    d_eta, d_phi = relative_coords(pt, eta, phi)
    return Data(
        x=torch.tensor(compute_node_features(pt, eta, phi, mode=features),
                       dtype=torch.float),
        pos=torch.tensor(np.stack([d_eta, d_phi], axis=1), dtype=torch.float),
        # Absolute kinematics for edge features / subjet reclustering.
        pt=torch.tensor(pt, dtype=torch.float),
        eta=torch.tensor(eta, dtype=torch.float),
        phi=torch.tensor(phi, dtype=torch.float),
        y=torch.tensor([label], dtype=torch.long),
        jet_idx=torch.tensor([jet_idx], dtype=torch.long),
        mjj=torch.tensor([mjj], dtype=torch.float),
        event_id=torch.tensor([event_id], dtype=torch.long),
    )


class JetExtractor:
    """Events → per-jet PyG Data objects; all settings read from one JetConfig."""

    def __init__(self, cfg: JetConfig):
        self.cfg = cfg

    def cluster_event(self, particles: np.ndarray) -> list[Jet]:
        """Cluster one (N, 3) array of active particles (batch of one)."""
        return next(self.cluster_chunk(particles[None, :, :]))

    def cluster_chunk(self, data: np.ndarray) -> Iterator[list[Jet]]:
        """Cluster a (n_events, slots, 3) chunk with ONE fastjet call.

        Yields each event's up-to-2 leading jets, pT-descending, constituents
        pT-sorted (truncation and Laman rely on hardest-first).
        """
        pseudojets = _pseudojets_by_event(data)
        cluster_sequence = fastjet.ClusterSequence(
            pseudojets, _JET_DEFINITION)
        constituents, selected_event_mask = _select_constituents(
            cluster_sequence, self.cfg)
        selected_events = _constituents_to_event_jets(constituents)

        if selected_event_mask is None:
            yield from selected_events
            return

        selected_event_iter = iter(selected_events)
        for is_selected in selected_event_mask:
            yield next(selected_event_iter) if is_selected else []

    def event_jets(self, jets: list[Jet], label: int,
                   event_id: int) -> Iterator[Data]:
        """Apply min_particles and convert one event's selected jets.

        ``mjj`` is the dijet mass of the two selected jets when both exist,
        including cases where only one later survives ``min_particles``.
        Single-jet events get ``mjj=0``.
        """
        cfg = self.cfg
        usable = [(ji, jet) for ji, jet in enumerate(jets)
                  if len(jet[0]) >= cfg.min_particles]
        if cfg.require_two_jets and len(usable) != 2:
            return
        if len(jets) >= 2:
            mjj = dijet_mass(jets[0], jets[1])
        else:
            mjj = 0.0
        for ji, jet in usable:
            yield jet_to_data(jet, label, ji, mjj, cfg.features, event_id)

    def from_chunks(self, chunks: Iterator[tuple[np.ndarray, np.ndarray]],
                    ) -> Iterator[Data]:
        event_id = 0
        for labels, data in chunks:
            if len(labels) != len(data):
                raise ValueError(
                    "chunk labels and events must have the same length, got "
                    f"{len(labels)} labels and {len(data)} events"
                )
            for label, jets in zip(
                labels,
                self.cluster_chunk(data),
                strict=True,
            ):
                yield from self.event_jets(jets, int(label), event_id)
                event_id += 1

    def from_events(self, events: Iterator[tuple[int, np.ndarray]],
                    ) -> Iterator[Data]:
        for event_id, (label, particles) in enumerate(events):
            yield from self.event_jets(self.cluster_event(particles), label,
                                       event_id)

    def __repr__(self) -> str:
        return f"JetExtractor({self.cfg})"
