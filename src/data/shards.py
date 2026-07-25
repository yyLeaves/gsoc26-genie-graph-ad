import copy
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from .graph import (
    EDGE_FEATURE_MODES,
    EDGE_PT_SCALES,
    GRAPH_STRATEGIES,
    STRATEGIES_WITH_K,
    STRATEGIES_WITH_RADIUS,
)
from .features import FEATURE_DIM

# Keep in sync with extractor.JET_SELECTIONS (avoid importing fastjet here).
_JET_SELECTIONS = ("leading_pt", "min_pt_all")

__all__ = [
    "shard_file",
    "save_shard",
    "load_shard",
    "save_metadata",
    "load_metadata",
    "validate_metadata",
    "dataset_spec",
    "ShardWriter",
    "write_shards",
]

DATASET_SCHEMA_VERSION = 1
_SPEC_FIELDS = (
    "schema_version",
    "source",
    "selection",
    "nodes",
    "edges",
    "materialization",
)
_LAYOUT_FIELDS = ("n_jets", "n_events", "n_shards", "shard_size")
_PER_JET_FIELDS = ("labels", "n_nodes", "n_edges", "event_ids", "jet_idx")


def shard_file(directory: Path, index: int) -> Path:
    return Path(directory) / f"shard_{index:04d}.pt"


def save_shard(jets: list, directory: Path, index: int) -> None:
    destination = shard_file(directory, index)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(jets, temporary)
    temporary.replace(destination)


def load_shard(directory: Path, index: int) -> list:
    return torch.load(shard_file(directory, index), weights_only=False)


def save_metadata(meta: dict, directory: Path) -> None:
    directory = Path(directory)
    validate_metadata(meta, directory)
    destination = directory / "metadata.pt"
    temporary = directory / "metadata.pt.tmp"
    torch.save(meta, temporary)
    temporary.replace(destination)


def load_metadata(directory: Path) -> dict:
    directory = Path(directory)
    meta = torch.load(directory / "metadata.pt", weights_only=False)
    validate_metadata(meta, directory)
    return meta


def validate_metadata(meta: dict, directory: Path | None = None) -> None:
    """Validate the common shard metadata contract and on-disk shard set."""
    if not isinstance(meta, dict):
        raise ValueError(f"metadata must be a dict, got {type(meta).__name__}")
    selection = meta.get("selection")
    if isinstance(selection, dict):
        # Legacy alias from older preprocess metadata.
        if "jet_selection" not in selection and "pt_cut_scope" in selection:
            selection["jet_selection"] = selection.pop("pt_cut_scope")
        elif "pt_cut_scope" in selection:
            if selection["pt_cut_scope"] != selection.get("jet_selection"):
                raise ValueError(
                    "metadata selection has conflicting jet_selection and "
                    "pt_cut_scope values"
                )
            selection.pop("pt_cut_scope")
    required = (
        "schema_version",
        "n_jets",
        "n_events",
        "labels",
        "n_nodes",
        "event_ids",
        "jet_idx",
        "n_shards",
        "shard_size",
        "source",
        "selection",
        "nodes",
        "edges",
    )
    missing = [name for name in required if name not in meta]
    if missing:
        raise ValueError(f"metadata is missing required fields {missing}")
    if int(meta["schema_version"]) != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported metadata schema_version={meta['schema_version']!r}; "
            f"expected {DATASET_SCHEMA_VERSION}"
        )

    n_jets = int(meta["n_jets"])
    n_events = int(meta["n_events"])
    n_shards = int(meta["n_shards"])
    shard_size = int(meta["shard_size"])
    if n_jets <= 0 or n_events <= 0 or n_shards <= 0 or shard_size <= 0:
        raise ValueError(
            "n_jets, n_events, n_shards, and shard_size must be positive"
        )
    expected_n_shards = (n_jets + shard_size - 1) // shard_size
    if n_shards != expected_n_shards:
        raise ValueError(
            f"n_shards={n_shards} is inconsistent with n_jets={n_jets} "
            f"and shard_size={shard_size}; expected {expected_n_shards}"
        )

    per_jet_fields = ("labels", "n_nodes", "n_edges", "event_ids", "jet_idx")
    for name in per_jet_fields:
        if name in meta and len(meta[name]) != n_jets:
            raise ValueError(
                f"metadata {name} has length {len(meta[name])}, "
                f"but n_jets={n_jets}"
            )

    labels = np.asarray(meta["labels"], dtype=np.int64)
    unexpected_labels = np.setdiff1d(np.unique(labels), np.array([0, 1]))
    if unexpected_labels.size:
        raise ValueError(
            "metadata labels must be binary {0, 1}, got "
            f"{unexpected_labels.tolist()}"
        )

    node_counts = np.asarray(meta["n_nodes"], dtype=np.int64)
    if np.any(node_counts <= 0):
        raise ValueError("metadata n_nodes must be positive for every jet")

    source = meta["source"]
    if not isinstance(source, dict) or "n_events" not in source:
        raise ValueError("metadata source must be a dict with n_events")
    source_events = int(source["n_events"])
    if source_events < n_events:
        raise ValueError(
            f"metadata source.n_events={source_events} is smaller than "
            f"selected n_events={n_events}"
        )
    selection = meta["selection"]
    required_selection = {
        "algorithm",
        "radius",
        "jet_selection",
        "min_jet_pt",
        "min_particles",
        "require_two_jets",
    }
    if not isinstance(selection, dict):
        raise ValueError("metadata selection must be a dict")
    missing_selection = sorted(required_selection - set(selection))
    if missing_selection:
        raise ValueError(
            f"metadata selection is missing fields {missing_selection}"
        )
    if not isinstance(selection["require_two_jets"], (bool, np.bool_)):
        raise ValueError("metadata selection.require_two_jets must be boolean")
    if float(selection["radius"]) <= 0:
        raise ValueError("metadata selection.radius must be positive")
    if float(selection["min_jet_pt"]) < 0:
        raise ValueError("metadata selection.min_jet_pt must be non-negative")
    if int(selection["min_particles"]) <= 0:
        raise ValueError("metadata selection.min_particles must be positive")
    if selection["jet_selection"] not in _JET_SELECTIONS:
        raise ValueError(
            f"metadata selection.jet_selection must be one of "
            f"{list(_JET_SELECTIONS)}, got {selection['jet_selection']!r}"
        )

    nodes = meta["nodes"]
    required_nodes = {"representation", "features", "feature_dim"}
    if not isinstance(nodes, dict):
        raise ValueError("metadata nodes must be a dict")
    missing_nodes = sorted(required_nodes - set(nodes))
    if missing_nodes:
        raise ValueError(f"metadata nodes is missing fields {missing_nodes}")
    if nodes["features"] not in FEATURE_DIM:
        raise ValueError(
            f"metadata nodes.features must be one of "
            f"{sorted(FEATURE_DIM)}, got {nodes['features']!r}"
        )
    expected_dim = FEATURE_DIM[nodes["features"]]
    if int(nodes["feature_dim"]) != expected_dim:
        raise ValueError(
            f"metadata nodes.feature_dim={nodes['feature_dim']} does not "
            f"match features={nodes['features']!r} (expected {expected_dim})"
        )

    edges = meta["edges"]
    if edges is None:
        if "n_edges" in meta:
            raise ValueError(
                "metadata with edges=None must not contain n_edges"
            )
    else:
        required_edges = {"strategy", "features", "pt_scale", "feature_dim"}
        if not isinstance(edges, dict):
            raise ValueError("metadata edges must be a dict or None")
        missing_edges = sorted(required_edges - set(edges))
        if missing_edges:
            raise ValueError(
                f"metadata edges is missing fields {missing_edges}"
            )
        strategy = edges["strategy"]
        if strategy not in GRAPH_STRATEGIES:
            raise ValueError(
                f"metadata edges.strategy must be one of "
                f"{list(GRAPH_STRATEGIES)}, got {strategy!r}"
            )
        if edges["features"] not in EDGE_FEATURE_MODES:
            raise ValueError(
                f"metadata edges.features must be one of "
                f"{list(EDGE_FEATURE_MODES)}, got {edges['features']!r}"
            )
        if edges["pt_scale"] not in EDGE_PT_SCALES:
            raise ValueError(
                f"metadata edges.pt_scale must be one of "
                f"{list(EDGE_PT_SCALES)}, got {edges['pt_scale']!r}"
            )
        needs_k = strategy in STRATEGIES_WITH_K
        if needs_k:
            if "k" not in edges or int(edges["k"]) < 1:
                raise ValueError(
                    f"metadata edges.strategy={strategy!r} requires k >= 1"
                )
        elif "k" in edges:
            raise ValueError(
                f"metadata edges.strategy={strategy!r} must not set k"
            )
        needs_radius = strategy in STRATEGIES_WITH_RADIUS
        if needs_radius:
            if "radius" not in edges or float(edges["radius"]) <= 0:
                raise ValueError(
                    f"metadata edges.strategy={strategy!r} requires "
                    "radius > 0"
                )
        elif "radius" in edges:
            raise ValueError(
                f"metadata edges.strategy={strategy!r} must not set radius"
            )
        if "n_edges" not in meta:
            raise ValueError("graph metadata must contain n_edges")
        edge_counts = np.asarray(meta["n_edges"], dtype=np.int64)
        if np.any(edge_counts < 0):
            raise ValueError("metadata n_edges must be non-negative")
        feature_dim = int(edges["feature_dim"])
        if feature_dim < 0:
            raise ValueError("metadata edges.feature_dim must be non-negative")
        if (edges["features"] == "none") != (feature_dim == 0):
            raise ValueError(
                "metadata edges.features='none' iff edge feature_dim is zero"
            )

    event_ids = np.asarray(meta["event_ids"], dtype=np.int64)
    jet_indices = np.asarray(meta["jet_idx"], dtype=np.int64)
    if np.any(event_ids < 0) or np.any(jet_indices < 0):
        raise ValueError("metadata event_ids and jet_idx must be non-negative")
    actual_events = int(len(np.unique(event_ids)))
    if n_events != actual_events:
        raise ValueError(
            f"n_events={n_events} does not match "
            f"{actual_events} unique event_ids"
        )

    order = np.lexsort((jet_indices, event_ids))
    sorted_events = event_ids[order]
    sorted_jets = jet_indices[order]
    sorted_labels = labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_events)) + 1]
    counts = np.diff(np.r_[starts, len(sorted_events)])
    label_min = np.minimum.reduceat(sorted_labels, starts)
    label_max = np.maximum.reduceat(sorted_labels, starts)
    if np.any(label_min != label_max):
        bad = sorted_events[starts[label_min != label_max]][:5].tolist()
        raise ValueError(
            f"metadata has inconsistent labels inside events {bad}"
        )

    bad_two_jet_events = []
    for start, count in zip(starts, counts, strict=True):
        event_jets = sorted_jets[start:start + count]
        event_id = int(sorted_events[start])
        if len(np.unique(event_jets)) != count:
            raise ValueError(
                f"metadata event_id={event_id} has duplicate jet_idx values"
            )
        if np.any(event_jets > 1):
            raise ValueError(
                f"metadata event_id={event_id} has jet_idx outside {{0, 1}}"
            )
        if selection["require_two_jets"] and not np.array_equal(
            event_jets, np.array([0, 1])
        ):
            bad_two_jet_events.append(event_id)
    if bad_two_jet_events:
        raise ValueError(
            "metadata require_two_jets=True requires jet_idx={0,1} "
            f"for every event; bad events={bad_two_jet_events[:5]}"
        )

    if directory is not None:
        directory = Path(directory)
        expected = {shard_file(directory, index) for index in range(n_shards)}
        actual = set(directory.glob("shard_*.pt"))
        if actual != expected:
            missing_files = sorted(path.name for path in expected - actual)
            extra_files = sorted(path.name for path in actual - expected)
            raise ValueError(
                "metadata shard set does not match disk; "
                f"missing={missing_files[:5]}, extra={extra_files[:5]}"
            )


def dataset_spec(meta: dict) -> dict:
    """Return the compact, hash-addressed part of dataset metadata.

    Large per-jet arrays are represented by shape/dtype/content hashes. This
    distinguishes subsets and physical reorderings without copying those arrays
    into every run config.
    """
    spec = {
        key: copy.deepcopy(meta.get(key))
        for key in _SPEC_FIELDS
    }
    spec["layout"] = {
        key: int(meta[key])
        for key in _LAYOUT_FIELDS
    }
    spec["per_jet_arrays"] = {}
    for key in _PER_JET_FIELDS:
        if key not in meta:
            continue
        values = np.ascontiguousarray(np.asarray(meta[key]))
        spec["per_jet_arrays"][key] = {
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }
    payload = json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **spec,
        "fingerprint_sha256": hashlib.sha256(payload).hexdigest(),
    }


class ShardWriter:
    """Write fixed-size ``Data`` shards and collect their common statistics."""

    def __init__(self, directory: str | Path, shard_size: int):
        if shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {shard_size}")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.directory.glob("shard_*.pt"))
        metadata_exists = (self.directory / "metadata.pt").exists()
        if existing or metadata_exists:
            raise FileExistsError(
                f"Refusing to overwrite an existing dataset in {self.directory} "
                f"({len(existing)} shard file(s), metadata={metadata_exists})"
            )
        self.shard_size = shard_size
        self.n_shards = 0
        self._buffer: list[Data] = []
        self._labels: list[int] = []
        self._node_counts: list[int] = []
        self._edge_counts: list[int] = []
        self._event_ids: list[int] = []
        self._jet_indices: list[int] = []
        self._has_edges: bool | None = None
        self._has_edge_attrs: bool | None = None
        self._node_feature_dim: int | None = None
        self._edge_feature_dim: int | None = None
        self._finished = False

    @property
    def n_jets(self) -> int:
        return len(self._labels)

    def _check_optional(
        self,
        state_name: str,
        field_name: str,
        present: bool,
    ) -> None:
        expected = getattr(self, state_name)
        if expected is None:
            setattr(self, state_name, present)
        elif expected != present:
            raise ValueError(
                f"{field_name} must be present on either every jet or none"
            )

    def add(self, graph: Data) -> None:
        if self._finished:
            raise RuntimeError("cannot add jets after finish()")

        if not isinstance(graph, Data):
            raise TypeError(
                f"ShardWriter expects PyG Data, got {type(graph).__name__}"
            )
        x = getattr(graph, "x", None)
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.size(0) == 0:
            raise ValueError("graph.x must be a non-empty rank-2 tensor")
        if self._node_feature_dim is None:
            self._node_feature_dim = int(x.size(1))
        elif x.size(1) != self._node_feature_dim:
            raise ValueError(
                f"graph.x feature width changed from "
                f"{self._node_feature_dim} to {x.size(1)}"
            )

        label = getattr(graph, "y", None)
        event_id = getattr(graph, "event_id", None)
        jet_idx = getattr(graph, "jet_idx", None)
        if label is None or torch.as_tensor(label).numel() != 1:
            raise ValueError("graph.y must contain one scalar binary label")
        label = int(label)
        if label not in {0, 1}:
            raise ValueError(f"graph.y must be binary, got {label}")
        if event_id is None or torch.as_tensor(event_id).numel() != 1:
            raise ValueError("event_id must be present as one scalar per graph")
        if jet_idx is None or torch.as_tensor(jet_idx).numel() != 1:
            raise ValueError("jet_idx must be present as one scalar per graph")
        event_id = int(event_id)
        jet_idx = int(jet_idx)
        if event_id < 0 or jet_idx < 0:
            raise ValueError("event_id and jet_idx must be non-negative")

        edge_index = getattr(graph, "edge_index", None)
        edge_attr = getattr(graph, "edge_attr", None)
        if edge_index is None:
            edge_count = None
            if edge_attr is not None:
                raise ValueError("graph.edge_attr requires graph.edge_index")
        else:
            if (
                not isinstance(edge_index, torch.Tensor)
                or edge_index.ndim != 2
                or edge_index.size(0) != 2
                or edge_index.dtype != torch.long
            ):
                raise ValueError(
                    "graph.edge_index must have shape (2, E) and dtype long"
                )
            if edge_index.numel() and (
                int(edge_index.min()) < 0 or int(edge_index.max()) >= x.size(0)
            ):
                raise ValueError("graph.edge_index contains invalid node indices")
            edge_count = int(edge_index.size(1))
            self._check_optional(
                "_has_edge_attrs", "edge_attr", edge_attr is not None
            )
            if edge_attr is not None:
                if (
                    not isinstance(edge_attr, torch.Tensor)
                    or edge_attr.ndim != 2
                    or edge_attr.size(0) != edge_count
                ):
                    raise ValueError(
                        "graph.edge_attr must have shape (E, edge_dim)"
                    )
                if self._edge_feature_dim is None:
                    self._edge_feature_dim = int(edge_attr.size(1))
                elif edge_attr.size(1) != self._edge_feature_dim:
                    raise ValueError(
                        f"graph.edge_attr feature width changed from "
                        f"{self._edge_feature_dim} to {edge_attr.size(1)}"
                    )
        self._check_optional("_has_edges", "edge_index", edge_count is not None)

        self._buffer.append(graph)
        self._labels.append(label)
        self._node_counts.append(int(graph.num_nodes))
        if edge_count is not None:
            self._edge_counts.append(int(edge_count))
        self._event_ids.append(event_id)
        self._jet_indices.append(jet_idx)
        if len(self._buffer) == self.shard_size:
            self._flush()

    def extend(self, graphs: Iterable[Data]) -> None:
        for graph in graphs:
            self.add(graph)

    def _flush(self) -> None:
        if not self._buffer:
            return
        save_shard(self._buffer, self.directory, self.n_shards)
        self.n_shards += 1
        self._buffer = []

    def finish(self) -> dict:
        if self._finished:
            raise RuntimeError("finish() may only be called once")
        self._flush()
        self._finished = True

        stats = {
            "n_jets": self.n_jets,
            "n_events": None,
            "labels": np.asarray(self._labels, dtype=np.int64),
            "n_nodes": np.asarray(self._node_counts, dtype=np.int32),
            "n_shards": self.n_shards,
            "shard_size": self.shard_size,
        }
        if self._has_edges:
            stats["n_edges"] = np.asarray(self._edge_counts, dtype=np.int32)
        event_ids = np.asarray(self._event_ids, dtype=np.int64)
        stats["event_ids"] = event_ids
        stats["n_events"] = int(len(np.unique(event_ids)))
        stats["jet_idx"] = np.asarray(self._jet_indices, dtype=np.int8)
        return stats


def write_shards(
    jets: Iterable[Data], directory: str | Path, shard_size: int
) -> dict:
    """Consume a jet stream; write fixed-size shards; return per-jet stats."""
    writer = ShardWriter(directory, shard_size)
    writer.extend(jets)
    return writer.finish()
