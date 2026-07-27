from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn

from src.checkpoint import load_checkpoint

from .blocks import BACKBONES
from .dynamic_graph_ae import DynamicEdgeGraphAE, DynamicGraphAE
from .edge_feature_node_graph_ae import (EdgeFeatureGraphAE,
                                         EdgeFeatureNodeGraphAE)
from .edge_graph_ae import EdgeGraphAE
from .inputs import normalize_feature_cols
from .node_graph_ae import NodeGraphAE


@dataclass(frozen=True, slots=True)
class _ModelCaps:
    """Per-type capabilities; frozensets below are views over this table."""

    edge_features: bool = False
    reconstructs_edges: bool = False
    dynamic: bool = False
    default_pt_node: bool = False
    precomputed_edges: bool = True


# One row per --model / ModelSpec.type (name = class without AE, snake_case).
_MODEL_CAPS: dict[str, _ModelCaps] = {
    "node_graph": _ModelCaps(),
    "edge_feature_node_graph": _ModelCaps(edge_features=True),
    "edge_feature_graph": _ModelCaps(
        edge_features=True, reconstructs_edges=True, default_pt_node=True),
    "edge_graph": _ModelCaps(
        edge_features=True, reconstructs_edges=True, default_pt_node=True),
    "dynamic_graph": _ModelCaps(dynamic=True, precomputed_edges=False),
    "dynamic_edge_graph": _ModelCaps(
        edge_features=True, reconstructs_edges=True, dynamic=True,
        default_pt_node=True),
}

MODEL_TYPES = tuple(_MODEL_CAPS)
EDGE_FEATURE_MODEL_TYPES = frozenset(
    name for name, caps in _MODEL_CAPS.items() if caps.edge_features)
DYNAMIC_MODEL_TYPES = frozenset(
    name for name, caps in _MODEL_CAPS.items() if caps.dynamic)
PT_NODE_MODEL_TYPES = frozenset(
    name for name, caps in _MODEL_CAPS.items() if caps.default_pt_node)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Complete, serializable description of a model and its input view."""

    type: str
    in_dim: int
    backbone: str = "edgeconv"
    hidden_dim: int = 64
    latent_dim: int = 2
    use_bn: bool = True
    edge_dim: int = 0
    edge_weight: float = 1.0
    aggr: str = "mean"
    dropout: float = 0.0
    dyn_k: int = 16
    feature_cols: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.type not in MODEL_TYPES:
            raise ValueError(
                f"model type must be one of {MODEL_TYPES}, got {self.type!r}")
        if self.in_dim <= 0 or self.hidden_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("in_dim, hidden_dim, and latent_dim must be positive")
        if self.backbone not in BACKBONES:
            raise ValueError(
                f"backbone must be one of {BACKBONES}, got {self.backbone!r}")
        if self.requires_edge_features and self.edge_dim <= 0:
            raise ValueError(f"model type {self.type!r} requires edge_dim > 0")
        if not self.requires_edge_features and self.edge_dim != 0:
            raise ValueError(
                f"model type {self.type!r} does not use edge features; "
                "edge_dim must be 0"
            )
        if self.edge_weight < 0.0:
            raise ValueError("edge_weight must be non-negative")
        if self.aggr not in {"mean", "add", "max"}:
            raise ValueError("aggr must be 'mean', 'add', or 'max'")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        if self.dyn_k <= 0:
            raise ValueError("dyn_k must be positive")
        if self.type == "edge_graph":
            if self.hidden_dim < 2:
                raise ValueError(
                    "edge_graph decoder uses hidden_dim // 2; hidden_dim "
                    "must be >= 2"
                )
            if self.use_bn:
                raise ValueError("edge_graph has no BatchNorm; use_bn must be False")
            if self.backbone != "edgeconv":
                raise ValueError("edge_graph has a fixed backbone='edgeconv'")
        elif self.aggr != "mean" or self.dropout != 0.0:
            raise ValueError(
                "aggr and dropout are configurable only for edge_graph"
            )
        if self.type in DYNAMIC_MODEL_TYPES:
            if self.backbone != "edgeconv":
                raise ValueError(
                    "dynamic models use a fixed dynamic EdgeConv backbone"
                )
        elif self.dyn_k != 16:
            raise ValueError("dyn_k is configurable only for dynamic models")
        if not self.reconstructs_edges and self.edge_weight != 1.0:
            raise ValueError(
                "edge_weight is configurable only for edge-reconstruction "
                "models"
            )
        object.__setattr__(
            self,
            "feature_cols",
            normalize_feature_cols(self.feature_cols, self.in_dim),
        )

    @property
    def requires_edge_features(self) -> bool:
        return _MODEL_CAPS[self.type].edge_features

    @property
    def requires_precomputed_edges(self) -> bool:
        """Whether message passing or reconstruction consumes stored edges."""
        return _MODEL_CAPS[self.type].precomputed_edges

    @property
    def reconstructs_edges(self) -> bool:
        return _MODEL_CAPS[self.type].reconstructs_edges

    def to_dict(self) -> dict:
        values = asdict(self)
        if self.feature_cols is not None:
            values["feature_cols"] = list(self.feature_cols)
        return values

    @classmethod
    def from_dict(cls, values: dict) -> "ModelSpec":
        known = {field.name for field in fields(cls)}
        # Drop retired experiment knobs so older checkpoints still load.
        data = {
            key: value for key, value in values.items()
            if key not in {
                "mask_fraction", "noise_std",
                "dyn_first_knn", "dyn_input_norm",
            }
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown ModelSpec fields: {sorted(unknown)}")
        if data.get("feature_cols") is not None:
            data["feature_cols"] = tuple(data["feature_cols"])
        return cls(**data)


def create_model(spec: ModelSpec) -> nn.Module:
    """Construct exactly one model from a validated :class:`ModelSpec`."""
    builder = _MODEL_BUILDERS.get(spec.type)
    if builder is None:
        raise AssertionError(f"unhandled validated model type: {spec.type!r}")
    return builder(spec)


def _build_node_graph(spec: ModelSpec) -> nn.Module:
    return NodeGraphAE(
        in_dim=spec.in_dim, backbone=spec.backbone,
        hidden_dim=spec.hidden_dim, latent_dim=spec.latent_dim,
        use_bn=spec.use_bn, feature_cols=spec.feature_cols)


def _build_edge_feature_node_graph(spec: ModelSpec) -> nn.Module:
    return EdgeFeatureNodeGraphAE(
        in_dim=spec.in_dim, edge_dim=spec.edge_dim,
        backbone=spec.backbone, hidden_dim=spec.hidden_dim,
        latent_dim=spec.latent_dim, use_bn=spec.use_bn,
        feature_cols=spec.feature_cols)


def _build_edge_feature_graph(spec: ModelSpec) -> nn.Module:
    return EdgeFeatureGraphAE(
        in_dim=spec.in_dim, edge_dim=spec.edge_dim,
        backbone=spec.backbone, hidden_dim=spec.hidden_dim,
        latent_dim=spec.latent_dim, use_bn=spec.use_bn,
        edge_weight=spec.edge_weight, feature_cols=spec.feature_cols)


def _build_edge_graph(spec: ModelSpec) -> nn.Module:
    return EdgeGraphAE(
        in_dim=spec.in_dim, edge_dim=spec.edge_dim,
        hidden_dim=spec.hidden_dim, latent_dim=spec.latent_dim,
        edge_weight=spec.edge_weight, aggr=spec.aggr,
        dropout=spec.dropout, feature_cols=spec.feature_cols)


def _build_dynamic_graph(spec: ModelSpec) -> nn.Module:
    return DynamicGraphAE(
        in_dim=spec.in_dim, hidden_dim=spec.hidden_dim,
        latent_dim=spec.latent_dim, k=spec.dyn_k,
        use_bn=spec.use_bn, feature_cols=spec.feature_cols)


def _build_dynamic_edge_graph(spec: ModelSpec) -> nn.Module:
    return DynamicEdgeGraphAE(
        in_dim=spec.in_dim, edge_dim=spec.edge_dim,
        hidden_dim=spec.hidden_dim, latent_dim=spec.latent_dim,
        k=spec.dyn_k, use_bn=spec.use_bn,
        edge_weight=spec.edge_weight, feature_cols=spec.feature_cols)


_MODEL_BUILDERS = {
    "node_graph": _build_node_graph,
    "edge_feature_node_graph": _build_edge_feature_node_graph,
    "edge_feature_graph": _build_edge_feature_graph,
    "edge_graph": _build_edge_graph,
    "dynamic_graph": _build_dynamic_graph,
    "dynamic_edge_graph": _build_dynamic_edge_graph,
}


def load_model_and_spec(
    checkpoint: str | Path,
    device: torch.device | str,
) -> tuple[nn.Module, ModelSpec]:
    """Load model weights and the embedded :class:`ModelSpec` in one read."""
    checkpoint = Path(checkpoint)
    payload = load_checkpoint(checkpoint, map_location=device)
    spec = ModelSpec.from_dict(payload["model"]["spec"])
    model = create_model(spec)
    try:
        model.load_state_dict(payload["model"]["state"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint {checkpoint} with ModelSpec "
            f"{spec.to_dict()}. The checkpoint and ModelSpec describe "
            "different model architectures."
        ) from exc
    return model.to(device).eval(), spec


def load_model(checkpoint: str | Path,
               device: torch.device | str) -> nn.Module:
    """Reconstruct a model from one self-contained checkpoint."""
    model, _ = load_model_and_spec(checkpoint, device)
    return model
