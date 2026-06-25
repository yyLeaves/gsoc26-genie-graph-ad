from dataclasses import dataclass
from typing import Iterator

import awkward as ak
import fastjet
import numpy as np
import torch
from torch_geometric.data import Data

from .graph import compute_node_features
from .kinematics import (Jet, dijet_mass, p4_components, relative_coords,
                         to_pseudojets)

__all__ = ["JetConfig", "JetExtractor", "cluster_jets", "truncated",
           "jet_to_data", "recluster_subjets", "JET_RADIUS"]

JET_RADIUS = 1.0
JET_DEF = fastjet.JetDefinition(fastjet.antikt_algorithm, JET_RADIUS)
# exclusive-kT reclustering into a fixed # of subjets (Araz et al. 2506.19920);
# R large enough that exclusive clustering completes without beam merging.
KT_DEF = fastjet.JetDefinition(fastjet.kt_algorithm, JET_RADIUS)


@dataclass(frozen=True)
class JetConfig:
    """Jet selection / feature settings."""
    features: str = "log_phys"
    min_jet_pt: float = 200.0
    min_particles: int = 3
    max_nodes: int | None = None
    n_subjets: int | None = None   # exclusive-kT recluster to this many subjets


def recluster_subjets(jet: Jet, n_subjets: int) -> Jet:
    """Exclusive-kT recluster (E-scheme) into <= n_subjets subjets; return
    (pT, y, φ) pT-descending. Jets with <= n_subjets constituents unchanged.

    Araz et al. (2506.19920) "fixed number of subjets": fixes node count so a
    graph AE can't separate signal by multiplicity instead of substructure.
    Subjets are massive (E-scheme) → angular coord is rapidity y, not η.
    """
    pt, eta, phi = jet
    if len(pt) <= n_subjets:
        return jet
    px, py, pz, E = p4_components(pt, eta, phi)
    pjs = [fastjet.PseudoJet(float(a), float(b), float(c), float(d))
           for a, b, c, d in zip(px, py, pz, E)]
    sub = fastjet.ClusterSequence(pjs, KT_DEF).exclusive_jets(n_subjets)
    s_pt = np.array([s.pt() for s in sub])
    s_y = np.array([s.rap() for s in sub])
    s_phi = np.array([s.phi_std() for s in sub])   # φ in (-π, π]
    order = np.argsort(-s_pt)                       # hardest-first
    return s_pt[order], s_y[order], s_phi[order]


def truncated(jet: Jet, max_nodes: int | None) -> Jet:
    """Keep the max_nodes hardest constituents (arrays are pT-descending)."""
    return tuple(a[:max_nodes] for a in jet)


def jet_to_data(jet: Jet, label: int, jet_idx: int, mjj: float,
                features: str, event_id: int | None = None) -> Data:
    """Convert one jet's constituent arrays to a PyG Data object."""
    pt, eta, phi = jet
    d_eta, d_phi = relative_coords(pt, eta, phi)
    g = Data(
        x=torch.tensor(compute_node_features(pt, eta, phi, mode=features),
                       dtype=torch.float),
        pos=torch.tensor(np.stack([d_eta, d_phi], axis=1), dtype=torch.float),
        # raw pT (pT-sorted), kept so step 2 builds (θ, k_T, z) edge features
        # regardless of node-feature mode
        pt=torch.tensor(pt, dtype=torch.float),
        y=torch.tensor([label], dtype=torch.long),
        jet_idx=torch.tensor([jet_idx], dtype=torch.long),
        mjj=torch.tensor([mjj], dtype=torch.float),
    )
    if event_id is not None:
        g.event_id = torch.tensor([event_id], dtype=torch.long)
    return g


class JetExtractor:
    """Events → per-jet PyG Data objects; all settings read from one JetConfig."""

    def __init__(self, cfg: JetConfig = JetConfig()):
        self.cfg = cfg

    def cluster_event(self, particles: np.ndarray) -> list[Jet]:
        """Cluster one (N, 3) array of active particles (batch of one)."""
        return next(self.cluster_chunk(particles[None, :, :]))

    def cluster_chunk(self, data: np.ndarray) -> Iterator[list[Jet]]:
        """Cluster a (n_events, slots, 3) chunk with ONE fastjet call.

        Yields each event's up-to-2 leading jets, pT-descending, constituents
        pT-sorted (truncation and Laman rely on hardest-first).
        """
        pt, eta, phi = data[..., 0], data[..., 1], data[..., 2]
        mask = pt > 0
        counts = mask.sum(axis=1)
        p4 = ak.unflatten(to_pseudojets(pt[mask], eta[mask], phi[mask]),
                          counts)

        cs = fastjet.ClusterSequence(p4, JET_DEF)
        # inclusive_jets/constituents align only at the SAME min_pt; come back
        # pT-ascending → sort to pick leading jets
        jets = cs.inclusive_jets(min_pt=self.cfg.min_jet_pt)
        consts = cs.constituents(min_pt=self.cfg.min_jet_pt)
        jet_pt = np.sqrt(jets["px"]**2 + jets["py"]**2)
        leading = ak.argsort(jet_pt, axis=1, ascending=False)
        consts = consts[leading][:, :2]

        # flatten chunk to numpy ONCE (per-jet awkward→numpy would dominate
        # runtime), then slice back into jets
        njets = ak.to_numpy(ak.num(consts, axis=1))
        flat_jets = ak.flatten(consts, axis=1)
        ncons = ak.to_numpy(ak.num(flat_jets, axis=1))
        flat = ak.flatten(flat_jets, axis=1)
        px, py, pz = (ak.to_numpy(flat[f]) for f in ("px", "py", "pz"))
        c_pt = np.sqrt(px**2 + py**2)
        c_eta = np.arcsinh(pz / (c_pt + 1e-10))
        c_phi = np.arctan2(py, px)

        bounds = np.cumsum(ncons)[:-1]
        jet_list = []
        for p, e, f in zip(np.split(c_pt, bounds), np.split(c_eta, bounds),
                           np.split(c_phi, bounds)):
            order = np.argsort(-p)           # hardest-first
            jet_list.append((p[order], e[order], f[order]))

        pos = 0
        for nj in njets:
            yield jet_list[pos:pos + nj]
            pos += nj

    def event_jets(self, jets: list[Jet], label: int,
                   event_id: int | None = None) -> Iterator[Data]:
        """Apply min_particles / max_nodes and convert one event's jets.

        mjj uses full constituents (before subjet reclustering); per-node repr
        uses subjets when cfg.n_subjets is set.
        """
        cfg = self.cfg
        mjj = dijet_mass(*jets) if len(jets) == 2 else 0.0
        for ji, jet in enumerate(jets):
            if len(jet[0]) >= cfg.min_particles:
                # subjets already fix node count; max_nodes too would break the
                # fixed-count guarantee
                if cfg.n_subjets is not None:
                    jet = recluster_subjets(jet, cfg.n_subjets)
                else:
                    jet = truncated(jet, cfg.max_nodes)
                yield jet_to_data(jet, label, ji, mjj, cfg.features, event_id)

    def from_chunks(self, chunks: Iterator[tuple[np.ndarray, np.ndarray]],
                    ) -> Iterator[Data]:
        """Batched path: one clustering call per chunk."""
        event_id = 0
        for labels, data in chunks:
            for label, jets in zip(labels, self.cluster_chunk(data)):
                yield from self.event_jets(jets, int(label), event_id)
                event_id += 1

    def from_events(self, events: Iterator[tuple[int, np.ndarray]],
                    ) -> Iterator[Data]:
        """Per-event reference path: same output as from_chunks."""
        for event_id, (label, particles) in enumerate(events):
            yield from self.event_jets(self.cluster_event(particles), label,
                                       event_id)

    def __repr__(self) -> str:
        return f"JetExtractor({self.cfg})"


def cluster_jets(pt, eta, phi, ptmin: float) -> list[Jet]:
    """Cluster ONE event (anti-kT); up to 2 leading jets, pT-descending.
    Convenience for tests/notebooks."""
    particles = np.stack([pt, eta, phi], axis=1)
    return JetExtractor(JetConfig(min_jet_pt=ptmin)).cluster_event(particles)
