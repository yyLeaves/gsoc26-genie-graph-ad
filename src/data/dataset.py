import copy
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from . import shards

__all__ = ["JetDataset"]


class JetDataset(torch.utils.data.Dataset):
    """Jet graph dataset over sharded .pt files; each item is one jet's PyG Data.

    Shards load on demand into an LRU cache (max_cache in memory).

    Args:
        shard_dir:    dir from build_graph.py (shard_NNNN.pt + metadata.pt)
        feature_cols: column indices to select from x; None = all (default)
        max_cache:    shards held in the LRU cache (~136 MB each, 32 GB RAM)
    """

    def __init__(self, shard_dir: str, feature_cols=None, max_cache: int = 4):
        self.shard_dir = Path(shard_dir)
        self.meta = shards.load_metadata(self.shard_dir)
        self.labels = self.meta["labels"]  # (n,) int64
        self.event_ids = self.meta.get("event_ids")
        self.jet_idx = self.meta.get("jet_idx")
        self.shard_size = self.meta["shard_size"]
        self.num_shards = self.meta["num_shards"]
        self.feature_cols = feature_cols
        self.max_cache = max_cache
        self._cache: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return self.meta["n"]

    def __getitem__(self, idx: int) -> Data:
        shard_idx, local_idx = divmod(idx, self.shard_size)
        g = self.load_shard(shard_idx)[local_idx]
        if self.feature_cols is not None:
            g = copy.copy(g)
            g.x = g.x[:, self.feature_cols]
        return g

    def select_features(self, batch):
        """Apply feature_cols to Batch.x in place; lets the shard-sequential
        loop (which bypasses __getitem__) reuse the same column selection.
        """
        if self.feature_cols is not None:
            batch.x = batch.x[:, self.feature_cols]
        return batch

    def load_shard(self, shard_idx: int) -> list:
        if shard_idx not in self._cache:
            self._cache[shard_idx] = shards.load_shard(self.shard_dir,
                                                       shard_idx)
            if len(self._cache) > self.max_cache:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(shard_idx)
        return self._cache[shard_idx]


    def background_indices(self) -> np.ndarray:
        """Indices of background jets (label == 0)."""
        return np.where(self.labels == 0)[0]

    def has_event_ids(self) -> bool:
        return self.event_ids is not None and len(self.event_ids) == len(self)

    def stats(self) -> dict:
        out = {
            "n": self.meta["n"],
            "n_events": self.meta.get("n_events", "?"),
            "n_signal": int(self.labels.sum()),
            "n_background": int((self.labels == 0).sum()),
            "mean_nodes": float(self.meta["n_nodes"].mean()),
            "has_event_ids": self.has_event_ids(),
        }
        if "n_edges" in self.meta:  # absent for point-cloud (step-1)
            out["mean_edges"] = float(self.meta["n_edges"].mean())
        return out

    def __repr__(self) -> str:
        fcols = (f", feature_cols={self.feature_cols}"
                 if self.feature_cols is not None else "")
        return (f"JetDataset(n={len(self):,}, shards={self.num_shards}, "
                f"strategy={self.meta.get('strategy', '?')!r}, "
                f"features={self.meta.get('features', '?')!r}{fcols}, "
                f"dir={self.shard_dir.name})")
