from typing import Iterator

import numpy as np
import pandas as pd

__all__ = [
    "EventReader", "hdf_key", "count_events", "read_chunk", "active_particles"
]

N_SLOTS = 700  # particle slots per event
CHUNK_SIZE = 50_000  # events read from HDF5 at a time


def hdf_key(h5_path: str) -> str:
    """Name of the single dataframe in the file."""
    with pd.HDFStore(h5_path, mode="r") as store:
        return store.keys()[0].lstrip("/")


def count_events(h5_path: str, key: str = None) -> int:
    with pd.HDFStore(h5_path, mode="r") as store:
        storer = store.get_storer(key or hdf_key(h5_path))
        # row count varies by pandas version: nrows, scalar shape, or shape[0]
        if storer.nrows is not None:
            return int(storer.nrows)
        shape = storer.shape
        return int(shape if np.isscalar(shape) else shape[0])


def read_chunk(h5_path: str, start: int, stop: int, key: str = "df"):
    """Read one chunk; return (labels, particle_array)."""
    df = pd.read_hdf(h5_path, key=key, start=start, stop=stop)
    labels = df.iloc[:, -1].values.astype(np.int64)
    data = df.iloc[:, :-1].values.reshape(-1, N_SLOTS, 3).astype("float32")
    return labels, data


def active_particles(row: np.ndarray) -> np.ndarray:
    """Remove zero-padded slots; return (N, 3) array of (pT, η, φ)."""
    return row[row[:, 0] > 0]


class EventReader:
    """Streams an LHCO-layout HDF5 file; holds path/key/count state."""

    def __init__(self, h5_path: str, chunk_size: int = CHUNK_SIZE):
        self.h5_path = h5_path
        self.chunk_size = chunk_size
        self.key = hdf_key(h5_path)
        self.n_events = count_events(h5_path, self.key)

    def read(self, start: int, stop: int):
        return read_chunk(self.h5_path, start, stop, self.key)

    def chunks(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """(labels, particle_array) one chunk_size block at a time."""
        for start in range(0, self.n_events, self.chunk_size):
            yield self.read(start, start + self.chunk_size)

    def events(self) -> Iterator[tuple[int, np.ndarray]]:
        """(label, active_particles) per event — reference path."""
        for labels, data in self.chunks():
            for label, row in zip(labels, data):
                yield int(label), active_particles(row)

    def __repr__(self) -> str:
        return (f"EventReader({self.h5_path!r}, key={self.key!r}, "
                f"n_events={self.n_events:,})")
