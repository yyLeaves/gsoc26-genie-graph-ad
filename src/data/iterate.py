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
        si = int(idx) // ds.shard_size
        li = int(idx) %  ds.shard_size
        shard_to_local[si].append(li)

    shard_ids = list(shard_to_local.keys())
    if shuffle_shards and rng is not None:
        rng.shuffle(shard_ids)

    for si in shard_ids:
        shard  = ds.load_shard(si)           # list[Data], LRU-cached
        locs   = shard_to_local[si]
        if shuffle_within and rng is not None:
            rng.shuffle(locs)
        for i in range(0, len(locs), batch_size):
            items = [shard[j] for j in locs[i : i + batch_size]]
            batch = Batch.from_data_list(items)
            if device is not None:
                batch = batch.to(device)
            yield batch


def prefetch(batches, depth: int = 4):
    """Run a batch iterator in a background thread, `depth` ahead.

    Collation + shard loading are CPU-bound and the model is tiny, so without
    this the GPU idles waiting for batches.
    """
    if depth <= 0:
        raise ValueError(f"prefetch depth must be positive, got {depth}")
    q: queue.Queue = queue.Queue(maxsize=depth)
    END = object()
    stop = threading.Event()

    def producer():
        try:
            for b in batches:
                if stop.is_set():
                    return
                q.put(b)
            q.put(END)
        except BaseException as e:        # propagate to consumer
            q.put(e)

    threading.Thread(target=producer, daemon=True).start()
    try:
        while True:
            item = q.get()
            if item is END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # consumer left early: signal + drain so a blocked q.put() unblocks and
        # the daemon thread exits — else it leaks, pinning GPU-resident batches
        stop.set()
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
