import queue
import threading
from collections import defaultdict

import numpy as np
from torch_geometric.data import Batch

from .dataset import JetDataset


def shard_iter(ds: JetDataset, indices: np.ndarray, batch_size: int,
               shuffle_shards: bool = False, shuffle_within: bool = False,
               rng=None, device=None):
    """Yield Batches, loading one shard at a time (one sequential read each,
    not per batch). Shuffling is explicit and requires an rng.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"indices must be one-dimensional, got {indices.shape}")
    if indices.size and ((indices < 0).any() or (indices >= len(ds)).any()):
        raise IndexError(f"indices must lie in [0, {len(ds)})")
    if (shuffle_shards or shuffle_within) and rng is None:
        raise ValueError("shuffling requires an explicit rng")

    # flat indices → shard_id → local positions
    shard_to_local: dict = defaultdict(list)
    for idx in indices:
        shard_id, local_index = divmod(int(idx), ds.shard_size)
        shard_to_local[shard_id].append(local_index)

    shard_ids = list(shard_to_local.keys())
    if shuffle_shards:
        rng.shuffle(shard_ids)

    for shard_id in shard_ids:
        shard = ds.load_shard(shard_id)
        local_indices = shard_to_local[shard_id]
        if shuffle_within:
            rng.shuffle(local_indices)
        for start in range(0, len(local_indices), batch_size):
            items = [shard[j] for j in local_indices[start : start + batch_size]]
            batch = Batch.from_data_list(items)
            if device is not None:
                batch = batch.to(device)
            yield batch


def prefetch(batches, depth: int = 4):
    """Prefetch in a background thread, or iterate synchronously at depth 0."""
    if depth == 0:
        yield from batches
        return
    if depth < 0:
        raise ValueError(f"prefetch depth must be nonnegative, got {depth}")
    buffer: queue.Queue = queue.Queue(maxsize=depth)
    end = object()
    stop = threading.Event()

    def enqueue(item):
        while not stop.is_set():
            try:
                buffer.put(item, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def producer():
        try:
            for batch in batches:
                if not enqueue(batch):
                    return
            enqueue(end)
        except BaseException as error:
            enqueue(error)

    threading.Thread(target=producer, daemon=True).start()
    try:
        while True:
            item = buffer.get()
            if item is end:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
