from dataclasses import dataclass, field

import numpy as np

# (manifest field name, required event label)
_MANIFEST_ROLES = (
    ("train_bkg_events", 0),
    ("train_sig_events", 1),
    ("val_bkg_events", 0),
    ("monitor_sig_events", 1),
    ("test_bkg_events", 0),
    ("test_sig_events", 1),
)
_MANIFEST_KEYS = {name for name, _ in _MANIFEST_ROLES}


def validate_split_args(args) -> None:
    """Validate split-related CLI args; shared by CLI and ``build_splits``."""
    protocol = getattr(args, "split_protocol", "manifest")
    manifest = getattr(args, "split_manifest", None)
    if protocol not in {"manifest", "ks_fixed"}:
        raise ValueError(
            f"split_protocol must be 'manifest' or 'ks_fixed', got {protocol!r}")
    if protocol == "manifest" and not manifest:
        raise ValueError("split_protocol='manifest' requires --split_manifest.")
    if protocol == "ks_fixed" and manifest:
        raise ValueError(
            "--split_manifest cannot be combined with split_protocol='ks_fixed'.")
    if not 0 < args.fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {args.fraction}")
    if manifest and args.fraction < 1.0:
        raise ValueError(
            "--split_manifest cannot be combined with --fraction < 1; "
            "use a smaller manifest for pilot runs.")


@dataclass
class Splits:
    train_idx: np.ndarray
    val_idx: np.ndarray
    sig_idx: np.ndarray
    eval_idx: np.ndarray  # val_bkg + monitor signal (epoch AUC/SIC)
    all_idx: np.ndarray
    test_idx: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )


def _as_event_set(manifest, key: str) -> set[int]:
    values = manifest[key].astype(np.int64)
    if len(np.unique(values)) != len(values):
        raise ValueError(f"split manifest field {key} contains duplicate ids")
    return set(values.tolist())


def _read_manifest(path: str, event_label: dict[int, int]) -> dict[str, set[int]]:
    """Load the fixed 6-field schema; enforce disjointness and label roles."""
    with np.load(path) as manifest:
        files = set(manifest.files)
        missing = sorted(_MANIFEST_KEYS - files)
        if missing:
            raise ValueError(
                f"split manifest {path} is missing required event arrays "
                f"{missing}")
        # Ignore legacy companion arrays (train_events / val_events, …).
        split_sets = {k: _as_event_set(manifest, k) for k, _ in _MANIFEST_ROLES}

    names = list(split_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(
                    f"split manifest {path} has {len(overlap):,} overlapping "
                    f"event ids between {left} and {right}; "
                    f"examples={sorted(overlap)[:5]}")
    absent = set().union(*split_sets.values()) - set(event_label)
    if absent:
        raise ValueError(
            f"split manifest {path} references {len(absent):,} event ids "
            f"not present in dataset; examples={sorted(absent)[:5]}")
    for name, expected in _MANIFEST_ROLES:
        bad = [e for e in split_sets[name] if event_label[int(e)] != expected]
        if bad:
            raise ValueError(
                f"split manifest {path} has {len(bad):,} wrong-label events "
                f"in {name}; examples={bad[:5]}")
    return split_sets


def _pilot_all_idx(ds, fraction, log) -> np.ndarray:
    """First ``fraction`` of shards, dropping jets from incomplete events."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return np.arange(len(ds), dtype=np.int64)

    n_use = max(1, round(ds.n_shards * fraction))
    limit = min(n_use * ds.shard_size, len(ds))
    log.info(f"\nPilot run: fraction={fraction}  "
             f"→ first {n_use}/{ds.n_shards} shards ({limit:,} jets)")

    all_eids = np.asarray(ds.event_ids, dtype=np.int64)
    sel_ids, sel_counts = np.unique(all_eids[:limit], return_counts=True)
    full_ids, full_counts = np.unique(all_eids, return_counts=True)
    complete = sel_ids[sel_counts == full_counts[np.searchsorted(full_ids, sel_ids)]]
    all_idx = np.flatnonzero(np.isin(all_eids[:limit], complete))
    if len(all_idx) < limit:
        log.info(f"Pilot event clamp: dropped {limit - len(all_idx):,} "
                 "boundary jet(s) from incomplete event(s)")
    return all_idx


def _event_maps(all_idx, event_ids, labels):
    event_label: dict[int, int] = {}
    by_event: dict[int, list[int]] = {}
    for index, event_id, label in zip(all_idx, event_ids, labels):
        event_id, label = int(event_id), int(label)
        previous = event_label.get(event_id)
        if previous is not None and previous != label:
            raise ValueError(
                f"inconsistent labels inside event_id={event_id}: {previous} vs {label}")
        event_label[event_id] = label
        by_event.setdefault(event_id, []).append(int(index))
    background = np.array(
        sorted(event_id for event_id, label in event_label.items() if label == 0),
        dtype=np.int64)
    signal = np.array(
        sorted(event_id for event_id, label in event_label.items() if label == 1),
        dtype=np.int64)
    return event_label, by_event, background, signal


def _jets_in(all_idx, event_ids, events) -> np.ndarray:
    return np.asarray(all_idx, dtype=np.int64)[np.isin(event_ids, list(events))]


def _jets_by_events(by_event, events) -> np.ndarray:
    return np.array([index for event_id in sorted(events) for index in by_event[event_id]],
                    dtype=np.int64)


def _roles_manifest(path, event_label, log):
    roles = _read_manifest(path, event_label)
    train_background, train_signal = roles["train_bkg_events"], roles["train_sig_events"]
    validation, monitor_signal = roles["val_bkg_events"], roles["monitor_sig_events"]
    test_background, test_signal = roles["test_bkg_events"], roles["test_sig_events"]
    log.info(f"Split mode : manifest  {path}")
    log.info(f"  train_bkg={len(train_background):,} train_sig={len(train_signal):,} "
             f"val_bkg={len(validation):,} monitor_sig={len(monitor_signal):,} "
             f"test_bkg={len(test_background):,} test_sig={len(test_signal):,} "
             f"ratio={len(test_background) / max(len(test_signal), 1):.1f}:1")
    if train_signal:
        log.info(f"  train contamination S/B="
                 f"{len(train_signal) / max(len(train_background), 1):.4f} "
                 "(labels hidden from AE loss)")
    log.info("  epoch AUC/SIC "
             + ("uses disjoint monitor_sig" if monitor_signal else "disabled (no monitor_sig)"))
    return train_background | train_signal, validation, monitor_signal, test_background | test_signal


def _roles_ks_fixed(args, bg, sig, rng, log):
    n_train, n_val, n_test, n_sig = (
        args.train_bkg_events, args.val_bkg_events,
        args.test_bkg_events, args.test_sig_events)
    if min(n_train, n_val, n_test, n_sig) <= 0:
        raise ValueError(
            "ks_fixed train/val/test background and test signal event "
            "counts must all be positive")
    need = n_train + n_val + n_test
    if len(bg) < need:
        raise ValueError(
            f"ks_fixed needs {need:,} scored background events "
            f"({n_train:,}+{n_val:,}+{n_test:,}), got {len(bg):,}.")
    if len(sig) < n_sig:
        raise ValueError(
            f"ks_fixed needs {n_sig:,} scored signal events, got {len(sig):,}.")
    bg = bg[rng.permutation(len(bg))]
    sig = sig[rng.permutation(len(sig))]
    train = set(bg[:n_train].tolist())
    val = set(bg[n_train:n_train + n_val].tolist())
    test = set(bg[n_train + n_val:need].tolist()) | set(sig[:n_sig].tolist())
    log.info("Split mode : ks_fixed")
    log.info(f"  train_bkg={n_train:,} val_bkg={n_val:,} "
             f"test_bkg={n_test:,} test_sig={n_sig:,} "
             f"ratio={n_test / max(n_sig, 1):.1f}:1")
    log.info("  epoch AUC/SIC disabled (no monitor_sig under ks_fixed)")
    return train, val, set(), test


def build_splits(ds, args, log, rng) -> Splits:
    """Build event-safe train/val/eval/test jet index sets.

    Shared ``rng`` is advanced here, then reused for per-epoch shuffles.
    """
    validate_split_args(args)
    manifest = getattr(args, "split_manifest", None)

    all_idx = _pilot_all_idx(ds, args.fraction, log)
    event_ids = np.asarray(ds.event_ids[all_idx], dtype=np.int64)
    labels = np.asarray(ds.labels[all_idx], dtype=np.int64)
    event_label, by_event, bg, sig = _event_maps(all_idx, event_ids, labels)

    if manifest:
        train_e, val_e, mon_e, test_e = _roles_manifest(
            manifest, event_label, log)
    else:
        train_e, val_e, mon_e, test_e = _roles_ks_fixed(
            args, bg, sig, rng, log)

    train_idx = _jets_in(all_idx, event_ids, train_e)
    val_idx = _jets_in(all_idx, event_ids, val_e)
    sig_idx = _jets_in(all_idx, event_ids, mon_e)
    test_idx = _jets_by_events(by_event, test_e)
    eval_idx = np.concatenate([val_idx, sig_idx])

    for name, idx in (("training", train_idx), ("validation", val_idx),
                      ("held-out test", test_idx)):
        if len(idx) == 0:
            raise ValueError(f"{name} split contains no jets")
    if not np.array_equal(
            np.unique(np.asarray(ds.labels[test_idx], dtype=np.int64)),
            np.array([0, 1])):
        raise ValueError(
            "held-out test split must contain both background and signal")
    train_y = np.asarray(ds.labels[train_idx], dtype=np.int64)
    if not np.any(train_y == 0):
        raise ValueError("training split contains no background events")
    val_y = np.asarray(ds.labels[val_idx], dtype=np.int64)
    eval_desc = ("val_bkg only; AUC/SIC skipped" if len(sig_idx) == 0
                 else "val_bkg + monitor_sig")
    log.info(f"\nSplit  train={len(train_idx):,} "
             f"(bkg={int(np.sum(train_y == 0)):,}, "
             f"sig={int(np.sum(train_y == 1)):,})"
             f" | val={len(val_idx):,} "
             f"(bkg={int(np.sum(val_y == 0)):,}, "
             f"sig={int(np.sum(val_y == 1)):,})"
             f" | eval_set={len(eval_idx):,} ({eval_desc})")
    log.info(f"       heldout_test={len(test_idx):,} jets")
    return Splits(train_idx, val_idx, sig_idx, eval_idx, all_idx, test_idx)
