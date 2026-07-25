"""Stream labeled or black-box LHCO events from Pandas HDF5 files."""

from typing import Iterator

import numpy as np
import pandas as pd
import tables

__all__ = [
    "EventReader", "hdf_key", "count_events", "read_chunk", "active_particles",
    "read_external_labels"
]

N_SLOTS = 700  # particle slots per event
N_PARTICLE_COLUMNS = N_SLOTS * 3  # (pT, eta, phi) per slot
CHUNK_SIZE = 50_000  # events read from HDF5 at a time


def hdf_key(h5_path: str) -> str:
    """Name of the single dataframe in the file."""
    with pd.HDFStore(h5_path, mode="r") as store:
        keys = [key.lstrip("/") for key in store.keys()]
    if len(keys) != 1:
        raise ValueError(
            f"Expected exactly one HDF key in {h5_path}, got {keys}"
        )
    return keys[0]


def _fixed_values_path(key: str) -> str:
    return f"/{key.strip('/')}/block0_values"


def count_events(h5_path: str, key: str | None = None) -> int:
    """Return the number of rows stored under an HDF5 dataframe key."""
    key = key or hdf_key(h5_path)
    try:
        with tables.open_file(h5_path, mode="r") as h5:
            return int(h5.get_node(_fixed_values_path(key)).shape[0])
    except tables.NoSuchNodeError:
        pass
    with pd.HDFStore(h5_path, mode="r") as store:
        storer = store.get_storer(key)
        # row count varies by pandas version: nrows, scalar shape, or shape[0]
        if storer.nrows is not None:
            return int(storer.nrows)
        shape = storer.shape
        return int(shape if np.isscalar(shape) else shape[0])


def read_external_labels(labels_path: str) -> np.ndarray:
    """Load one label per line and map every positive label to binary signal."""
    labels = np.loadtxt(labels_path, dtype=np.float64)
    labels = np.atleast_1d(labels)
    if labels.ndim != 1:
        raise ValueError(
            f"External labels must be one-dimensional, got {labels.shape}"
        )
    if not np.isfinite(labels).all():
        raise ValueError("External labels must contain only finite values")
    return (labels > 0).astype(np.int64)


def _read_values_chunk(h5_path: str, key: str, start: int,
                       stop: int) -> np.ndarray:
    """Read the fixed-format Pandas HDF5 payload directly with PyTables.

    The official LHCO black-box files are old fixed-format Pandas HDF5 files
    that recent pandas can list but not select reliably.  The actual payload is
    still a simple `/key/block0_values` CArray, so read it directly.
    """
    try:
        with tables.open_file(h5_path, mode="r") as h5:
            return h5.get_node(_fixed_values_path(key))[start:stop]
    except tables.NoSuchNodeError:
        df = pd.read_hdf(h5_path, key=key, start=start, stop=stop)
        return df.values


def read_chunk(
    h5_path: str,
    start: int,
    stop: int,
    key: str = "df",
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read rows ``[start:stop)`` as binary labels and padded particles."""
    values = np.asarray(_read_values_chunk(h5_path, key, start, stop))
    if values.ndim != 2:
        raise ValueError(
            f"Expected a 2D LHCO event table, got shape {values.shape}.")

    n_cols = values.shape[1]
    if n_cols not in {N_PARTICLE_COLUMNS, N_PARTICLE_COLUMNS + 1}:
        raise ValueError(
            f"Expected {N_PARTICLE_COLUMNS} particle columns, optionally plus "
            f"one label column, got {n_cols} columns.")

    particle_values = values[:, :N_PARTICLE_COLUMNS]
    inline_labels = (
        values[:, N_PARTICLE_COLUMNS]
        if n_cols == N_PARTICLE_COLUMNS + 1
        else None
    )
    if labels is None:
        if inline_labels is None:
            raise ValueError(
                f"{h5_path} has no inline label column; pass labels_path.")
        chunk_labels = (inline_labels > 0).astype(np.int64)
    else:
        actual_stop = start + values.shape[0]
        chunk_labels = labels[start:actual_stop].astype(np.int64)
        if len(chunk_labels) != values.shape[0]:
            raise ValueError(
                f"External labels length mismatch for rows "
                f"[{start}:{actual_stop}).")
        if inline_labels is not None:
            inline_binary = (inline_labels > 0).astype(np.int64)
            if not np.array_equal(chunk_labels, inline_binary):
                disagree = int(np.count_nonzero(chunk_labels != inline_binary))
                raise ValueError(
                    f"{h5_path} has inline labels that disagree with "
                    f"external labels on {disagree} rows in "
                    f"[{start}:{actual_stop}); pass unlabeled HDF5 or "
                    "matching labels."
                )

    particle_slots = particle_values.reshape(-1, N_SLOTS, 3).astype("float32")
    return chunk_labels, particle_slots


def active_particles(row: np.ndarray) -> np.ndarray:
    """Remove zero-padded slots; return (N, 3) array of (pT, η, φ)."""
    return row[row[:, 0] > 0]


class EventReader:
    """Read one LHCO HDF5 source as chunks or active-particle events."""

    def __init__(self, h5_path: str, chunk_size: int = CHUNK_SIZE,
                 labels_path: str | None = None, key: str | None = None):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.h5_path = h5_path
        self.chunk_size = chunk_size
        self.key = key.strip("/") if key is not None else hdf_key(h5_path)
        self.n_events = count_events(h5_path, self.key)
        self.labels_path = labels_path
        self.labels = (read_external_labels(labels_path)
                       if labels_path is not None else None)
        if self.labels is not None and len(self.labels) != self.n_events:
            raise ValueError(
                f"labels_path has {len(self.labels):,} rows but HDF5 has "
                f"{self.n_events:,} events.")

    def read(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """Return labels and padded ``(N, 700, 3)`` particle slots."""
        if not 0 <= start <= stop <= self.n_events:
            raise IndexError(
                f"row range [{start}:{stop}) is outside [0:{self.n_events})"
            )
        return read_chunk(self.h5_path, start, stop, self.key, self.labels)

    def chunks(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield labels and padded particles in contiguous input chunks."""
        for start in range(0, self.n_events, self.chunk_size):
            yield self.read(start, min(start + self.chunk_size, self.n_events))

    def events(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield one ``(label, active particles)`` pair per event."""
        for chunk_labels, particle_slots in self.chunks():
            for label, slots in zip(chunk_labels, particle_slots):
                yield int(label), active_particles(slots)

    def __repr__(self) -> str:
        return (f"EventReader({self.h5_path!r}, key={self.key!r}, "
                f"n_events={self.n_events:,}, "
                f"labels_path={self.labels_path!r})")
