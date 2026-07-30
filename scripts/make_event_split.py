"""Create a reusable event-id split manifest for training / eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.dataset import JetDataset

# Must match src.training.splits._MANIFEST_ROLES
_MANIFEST_KEYS = (
    "train_bkg_events",
    "train_sig_events",
    "val_bkg_events",
    "monitor_sig_events",
    "test_bkg_events",
    "test_sig_events",
)


def _to_event_array(events) -> np.ndarray:
    return np.asarray(list(events), dtype=np.int64)


def event_labels(ds: JetDataset) -> dict[int, int]:
    labels = np.asarray(ds.labels, dtype=np.int64)
    event_ids = np.asarray(ds.event_ids, dtype=np.int64)
    out: dict[int, int] = {}
    for event_id, label in zip(event_ids, labels):
        event_id = int(event_id)
        label = int(label)
        if event_id in out and out[event_id] != label:
            raise ValueError(
                f"inconsistent labels inside event_id={event_id}: "
                f"{out[event_id]} vs {label}"
            )
        out[event_id] = label
    return out


def shuffled_events(events: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    events = np.asarray(sorted(events.tolist()), dtype=np.int64)
    return events[rng.permutation(len(events))]


def split_range(events: np.ndarray, start: int, count: int) -> np.ndarray:
    return events[start:start + count].astype(np.int64)


def disjoint_or_raise(named_sets: dict[str, np.ndarray]) -> None:
    seen: dict[int, str] = {}
    for name, values in named_sets.items():
        for event_id in np.asarray(values, dtype=np.int64):
            event_id = int(event_id)
            previous = seen.get(event_id)
            if previous is not None:
                raise ValueError(
                    f"event_id={event_id} appears in both {previous} and {name}"
                )
            seen[event_id] = name


def event_id_summary(events: np.ndarray) -> dict:
    events = np.asarray(events, dtype=np.int64)
    if events.size == 0:
        return {"n": 0, "min": None, "max": None, "preview": []}
    return {
        "n": int(events.size),
        "min": int(events.min()),
        "max": int(events.max()),
        "preview": [int(x) for x in events[:5]],
    }


def main(args: argparse.Namespace) -> None:
    ds = JetDataset(args.data_dir)
    labels = event_labels(ds)
    rng = np.random.default_rng(args.seed)
    bg_events = shuffled_events(
        _to_event_array(e for e, y in labels.items() if y == 0), rng)
    sig_events = shuffled_events(
        _to_event_array(e for e, y in labels.items() if y == 1), rng)

    if args.train_sig_events is not None and args.train_sig_events < 0:
        raise ValueError("--train_sig_events must be non-negative.")
    if args.train_s_over_b < 0:
        raise ValueError("--train_s_over_b must be non-negative.")
    if args.monitor_sig_events < 0:
        raise ValueError("--monitor_sig_events must be non-negative.")

    train_sig_count = (
        int(args.train_sig_events)
        if args.train_sig_events is not None
        else int(round(args.train_s_over_b * args.train_bkg_events))
    )

    need_bkg = args.train_bkg_events + args.val_bkg_events + args.test_bkg_events
    if len(bg_events) < need_bkg:
        raise ValueError(
            f"Need {need_bkg:,} background events "
            f"({args.train_bkg_events:,} train + {args.val_bkg_events:,} val + "
            f"{args.test_bkg_events:,} test), but dataset has {len(bg_events):,}."
        )
    need_sig = train_sig_count + args.monitor_sig_events + args.test_sig_events
    if len(sig_events) < need_sig:
        raise ValueError(
            f"Need {need_sig:,} signal events "
            f"({train_sig_count:,} train contamination + "
            f"{args.monitor_sig_events:,} monitor + "
            f"{args.test_sig_events:,} test), "
            f"but dataset has {len(sig_events):,}."
        )

    train_bkg_events = split_range(bg_events, 0, args.train_bkg_events)
    val_bkg_events = split_range(bg_events, args.train_bkg_events,
                                 args.val_bkg_events)
    test_bkg_events = split_range(
        bg_events, args.train_bkg_events + args.val_bkg_events,
        args.test_bkg_events)

    train_sig_events = split_range(sig_events, 0, train_sig_count)
    monitor_sig_events = split_range(sig_events, train_sig_count,
                                     args.monitor_sig_events)
    test_sig_events = split_range(
        sig_events, train_sig_count + args.monitor_sig_events,
        args.test_sig_events)

    arrays = {
        "train_bkg_events": train_bkg_events,
        "train_sig_events": train_sig_events,
        "val_bkg_events": val_bkg_events,
        "monitor_sig_events": monitor_sig_events,
        "test_bkg_events": test_bkg_events,
        "test_sig_events": test_sig_events,
    }
    disjoint_or_raise(arrays)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **{k: arrays[k] for k in _MANIFEST_KEYS})

    summary = {
        "data_dir": str(args.data_dir),
        "output": str(output),
        "seed": int(args.seed),
        "train_bkg_events": int(len(train_bkg_events)),
        "train_sig_events": int(len(train_sig_events)),
        "train_total_events": int(len(train_bkg_events) + len(train_sig_events)),
        "train_s_over_b_requested": float(args.train_s_over_b),
        "train_s_over_b_effective": (
            float(len(train_sig_events) / len(train_bkg_events))
            if len(train_bkg_events) else 0.0
        ),
        "val_bkg_events": int(len(val_bkg_events)),
        "monitor_sig_events": int(len(monitor_sig_events)),
        "test_bkg_events": int(len(test_bkg_events)),
        "test_sig_events": int(len(test_sig_events)),
        "test_bkg_to_signal": (
            float(len(test_bkg_events) / len(test_sig_events))
            if len(test_sig_events) else None
        ),
        "available_bkg_events": int(len(bg_events)),
        "available_sig_events": int(len(sig_events)),
        "leakage_check": "passed",
        "event_id_summary": {
            "train_bkg": event_id_summary(train_bkg_events),
            "train_sig": event_id_summary(train_sig_events),
            "val_bkg": event_id_summary(val_bkg_events),
            "monitor_sig": event_id_summary(monitor_sig_events),
            "test_bkg": event_id_summary(test_bkg_events),
            "test_sig": event_id_summary(test_sig_events),
        },
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"Saved split manifest → {output}")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reusable event-id split manifest."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Selected-jet or graph shard directory.")
    parser.add_argument("--output", required=True,
                        help="Destination .npz manifest path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_bkg_events", type=int, default=80_000)
    parser.add_argument("--train_sig_events", type=int, default=None,
                        help="Explicit train signal contamination count. "
                             "Overrides --train_s_over_b when set.")
    parser.add_argument("--train_s_over_b", type=float, default=0.0,
                        help="Train signal/background contamination ratio.")
    parser.add_argument("--val_bkg_events", type=int, default=20_000)
    parser.add_argument("--monitor_sig_events", type=int, default=0,
                        help="Disjoint signal events for epoch AUC/SIC.")
    parser.add_argument("--test_bkg_events", type=int, default=340_000)
    parser.add_argument("--test_sig_events", type=int, default=20_000)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
