from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch_geometric.data import Data

__all__ = [
    "shard_file",
    "save_shard",
    "load_shard",
    "save_metadata",
    "load_metadata",
    "write_shards",
]


def shard_file(directory: Path, index: int) -> Path:
    return Path(directory) / f"shard_{index:04d}.pt"


def save_shard(jets: list, directory: Path, index: int) -> None:
    torch.save(jets, shard_file(directory, index))


def load_shard(directory: Path, index: int) -> list:
    return torch.load(shard_file(directory, index), weights_only=False)


def save_metadata(meta: dict, directory: Path) -> None:
    torch.save(meta, Path(directory) / "metadata.pt")


def load_metadata(directory: Path) -> dict:
    return torch.load(Path(directory) / "metadata.pt", weights_only=False)


def write_shards(jets: Iterator[Data], directory: Path,
                 shard_size: int) -> dict:
    """Consume a jet stream; write fixed-size shards; return per-jet stats."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    buffer: list[Data] = []
    labels, n_nodes, event_ids, jet_indices, num_shards = [], [], [], [], 0
    have_event_ids = have_jet_indices = True
    for g in jets:
        buffer.append(g)
        labels.append(int(g.y))
        n_nodes.append(g.x.shape[0])
        if getattr(g, "event_id", None) is None:
            have_event_ids = False
        else:
            event_ids.append(int(g.event_id))
        if getattr(g, "jet_idx", None) is None:
            have_jet_indices = False
        else:
            jet_indices.append(int(g.jet_idx))
        if len(buffer) == shard_size:
            save_shard(buffer, directory, num_shards)
            num_shards += 1
            buffer = []
    if buffer:
        save_shard(buffer, directory, num_shards)
        num_shards += 1

    meta = {
        "n": len(labels),
        "labels": np.array(labels, dtype=np.int64),
        "n_nodes": np.array(n_nodes, dtype=np.int32),
        "num_shards": num_shards,
        "shard_size": shard_size,
    }
    if have_event_ids:
        meta["event_ids"] = np.array(event_ids, dtype=np.int64)
    if have_jet_indices:
        meta["jet_idx"] = np.array(jet_indices, dtype=np.int8)
    return meta
