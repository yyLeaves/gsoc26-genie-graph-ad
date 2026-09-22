"""Distill a frozen, background-trained Graph-AE encoder into a student."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from ..edge_graph_ae import EdgeBlock, EdgeGraphAE
from ..factory import ModelSpec, create_model, load_model_and_spec
from ..inputs import require_edge_features, select_node_features
from ..reconstruction import mse_per_graph


TEACHER_KD_COMPONENTS = ("eb2", "eb3", "graph", "joint")
TEACHER_KD_CORRUPTIONS = ("none", "node", "edge", "both")


@dataclass(frozen=True, slots=True)
class TeacherKDTerms:
    eb2: torch.Tensor
    eb3: torch.Tensor
    graph: torch.Tensor
    joint: torch.Tensor


def select_teacher_kd_component(
    terms: TeacherKDTerms,
    component: str,
) -> torch.Tensor:
    if component not in TEACHER_KD_COMPONENTS:
        raise ValueError(
            f"component must be one of {TEACHER_KD_COMPONENTS}, "
            f"got {component!r}"
        )
    return getattr(terms, component)


def _load_teacher_checkpoint(
    checkpoint: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[EdgeGraphAE, ModelSpec]:
    """Load a self-contained Graph-AE teacher checkpoint."""
    model, spec = load_model_and_spec(checkpoint, device)
    if not isinstance(model, EdgeGraphAE):
        raise ValueError(
            "baseline teacher must be an EdgeGraphAE, "
            f"got {type(model).__name__}"
        )
    return model, spec


class StudentEncoder(nn.Module):
    """Three-block edge-aware student with projections to teacher widths."""

    def __init__(
        self,
        *,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        latent_dim: int,
        teacher_hidden_dim: int,
        teacher_latent_dim: int,
        aggr: str,
        dropout: float,
    ):
        super().__init__()
        dims = (in_dim, hidden_dim, hidden_dim, latent_dim)
        self.blocks = nn.ModuleList(
            EdgeBlock(input_dim, output_dim, edge_dim, hidden=hidden_dim,
                      aggr=aggr, dropout=dropout)
            for input_dim, output_dim in pairwise(dims))
        self.eb2_projection = (
            nn.Identity() if hidden_dim == teacher_hidden_dim
            else nn.Linear(hidden_dim, teacher_hidden_dim)
        )
        self.eb3_projection = (
            nn.Identity() if latent_dim == teacher_latent_dim
            else nn.Linear(latent_dim, teacher_latent_dim)
        )

    def forward(self, nodes, edge_index, edges):
        hidden = nodes
        outputs = []
        for block in self.blocks:
            hidden = block(hidden, edge_index, edges)
            outputs.append(hidden)
        return self.eb2_projection(outputs[1]), self.eb3_projection(outputs[2])


def _encode_teacher(model: EdgeGraphAE, nodes, edge_index, edges):
    hidden = nodes
    outputs = []
    for block in model.encoder_blocks:
        hidden = block(hidden, edge_index, edges)
        outputs.append(hidden)
    return outputs[1], outputs[2]


def _graph_summary(eb2, eb3, batch_index, n_graphs):
    """Retain both typical and extreme responses at both encoder depths."""
    pooled = []
    for representation in (eb2, eb3):
        pooled.extend([
            scatter(
                representation, batch_index, dim=0, reduce="mean",
                dim_size=n_graphs,
            ),
            scatter(
                representation, batch_index, dim=0, reduce="max",
                dim_size=n_graphs,
            ),
        ])
    return torch.cat(pooled, dim=-1)


class BaselineTeacherKD(nn.Module):
    """Frozen Graph-AE teacher with clean or denoising student training."""

    def __init__(
        self,
        teacher: EdgeGraphAE,
        teacher_spec: ModelSpec,
        *,
        component: str = "joint",
        corruption: str = "none",
        mask_fraction: float = 0.15,
        student_hidden_dim: int = 64,
        student_latent_dim: int = 2,
        student_dropout: float = 0.0,
        student_seed: int = 124,
    ):
        super().__init__()
        if component not in TEACHER_KD_COMPONENTS:
            raise ValueError(
                f"component must be one of {TEACHER_KD_COMPONENTS}"
            )
        if corruption not in TEACHER_KD_CORRUPTIONS:
            raise ValueError(
                f"corruption must be one of {TEACHER_KD_CORRUPTIONS}"
            )
        if not 0.0 <= mask_fraction < 1.0:
            raise ValueError("mask_fraction must lie in [0, 1)")
        if student_hidden_dim <= 0 or student_latent_dim <= 0:
            raise ValueError("student dimensions must be positive")
        if not 0.0 <= student_dropout < 1.0:
            raise ValueError("student_dropout must lie in [0, 1)")

        self.teacher = teacher
        self.teacher_spec = teacher_spec
        self.component = component
        self.corruption = corruption
        self.mask_fraction = mask_fraction
        self.student_hidden_dim = student_hidden_dim
        self.student_latent_dim = student_latent_dim
        self.student_dropout = student_dropout
        self.student_seed = student_seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(student_seed)
            self.student = StudentEncoder(
                in_dim=teacher_spec.in_dim,
                edge_dim=teacher_spec.edge_dim,
                hidden_dim=student_hidden_dim,
                latent_dim=student_latent_dim,
                teacher_hidden_dim=teacher_spec.hidden_dim,
                teacher_latent_dim=teacher_spec.latent_dim,
                aggr=teacher_spec.aggr,
                dropout=student_dropout,
            )
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    @classmethod
    def from_teacher_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: torch.device | str = "cpu",
        **kwargs,
    ) -> "BaselineTeacherKD":
        teacher, spec = _load_teacher_checkpoint(checkpoint, device)
        return cls(teacher, spec, **kwargs).to(device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def trainable_parameters(self):
        return self.student.parameters()

    def _inputs(self, batch):
        nodes = select_node_features(
            batch, self.teacher_spec.in_dim, self.teacher_spec.feature_cols)
        edges = require_edge_features(batch, self.teacher_spec.edge_dim)
        return nodes, edges

    def _student_inputs(self, nodes, edges):
        if not self.training or self.corruption == "none" or not self.mask_fraction:
            return nodes, edges
        if self.corruption in {"node", "both"}:
            keep = torch.rand(
                (nodes.size(0), 1), device=nodes.device,
                dtype=nodes.dtype,
            ) >= self.mask_fraction
            nodes = nodes * keep
        if self.corruption in {"edge", "both"}:
            keep = torch.rand(
                (edges.size(0), 1), device=edges.device,
                dtype=edges.dtype,
            ) >= self.mask_fraction
            edges = edges * keep
        return nodes, edges

    def terms(self, batch) -> TeacherKDTerms:
        clean_nodes, clean_edges = self._inputs(batch)
        with torch.no_grad():
            teacher_eb2, teacher_eb3 = _encode_teacher(
                self.teacher, clean_nodes, batch.edge_index, clean_edges)
        student_nodes, student_edges = self._student_inputs(
            clean_nodes, clean_edges)
        student_eb2, student_eb3 = self.student(
            student_nodes, batch.edge_index, student_edges)

        eb2 = mse_per_graph(
            student_eb2, teacher_eb2, batch.batch, batch.num_graphs)
        eb3 = mse_per_graph(
            student_eb3, teacher_eb3, batch.batch, batch.num_graphs)
        teacher_graph = _graph_summary(
            teacher_eb2, teacher_eb3, batch.batch, batch.num_graphs)
        student_graph = _graph_summary(
            student_eb2, student_eb3, batch.batch, batch.num_graphs)
        graph = ((student_graph - teacher_graph) ** 2).mean(dim=-1)
        return TeacherKDTerms(
            eb2=eb2,
            eb3=eb3,
            graph=graph,
            joint=eb2 + eb3 + graph,
        )

    def loss(self, batch) -> torch.Tensor:
        return select_teacher_kd_component(
            self.terms(batch), self.component).mean()

    @torch.no_grad()
    def anomaly_score(self, batch) -> torch.Tensor:
        return select_teacher_kd_component(
            self.terms(batch), self.component)

    def config(self) -> dict:
        return {
            "teacher_spec": self.teacher_spec.to_dict(),
            "component": self.component,
            "corruption": self.corruption,
            "mask_fraction": self.mask_fraction,
            "student_hidden_dim": self.student_hidden_dim,
            "student_latent_dim": self.student_latent_dim,
            "student_dropout": self.student_dropout,
            "student_seed": self.student_seed,
        }


def load_baseline_teacher_kd_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> BaselineTeacherKD:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = dict(payload["model_config"])
    teacher_spec = ModelSpec.from_dict(config.pop("teacher_spec"))
    teacher = create_model(teacher_spec)
    if not isinstance(teacher, EdgeGraphAE):
        raise ValueError("stored teacher spec does not describe EdgeGraphAE")
    model = BaselineTeacherKD(teacher, teacher_spec, **config)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()
