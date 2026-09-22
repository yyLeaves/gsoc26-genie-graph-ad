"""Array grouping shared by dataset validation, scoring, and run summaries."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventGroups:
    order: np.ndarray
    event_ids: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    labels: np.ndarray


def group_events(event_ids, labels, *, order=None,
                 label_error="inconsistent labels inside events") -> EventGroups:
    """Group by event ID, preserving the relative jet order within each event.

    Empty arrays are valid. A supplied permutation must sort by event ID;
    metadata validation uses it to additionally sort jets within each event.
    """
    event_ids = np.asarray(event_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if order is None:
        order = np.argsort(event_ids, kind="stable")
    sorted_ids = event_ids[order]
    if sorted_ids.size:
        starts = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1]
    else:
        starts = np.empty(0, dtype=np.int64)
    unique_ids = sorted_ids[starts]
    counts = np.diff(np.append(starts, sorted_ids.size))
    sorted_labels = labels[order]
    label_min = np.minimum.reduceat(sorted_labels, starts)
    label_max = np.maximum.reduceat(sorted_labels, starts)
    if np.any(label_min != label_max):
        bad = unique_ids[label_min != label_max][:5].tolist()
        raise ValueError(f"{label_error} {bad}")
    return EventGroups(order, unique_ids, starts, counts, label_min)
