import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import JetDataset
from src.eval.scoring import EVENT_SCORE_AGGREGATIONS, load_model, report, score_events


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")
    data = JetDataset(args.data_dir)
    model = load_model(args.checkpoint, data.meta.get("edge_dim", 0), device)
    print(f"Model     : {model.__class__.__name__}  (from {args.checkpoint})")
    print(f"Device    : {device}")

    t0 = time.time()
    if args.bkg_dir:                         # zero-shot: data_dir = signal
        bkg = JetDataset(args.bkg_dir)
        # background reference = label-0 jets only (bkg_dir may be a mixed
        # train set; including its signal would contaminate the reference)
        bkg_idx = np.where(bkg.labels == 0)[0]
        bkg_scores, _, _ = score_events(model, bkg, device, indices=bkg_idx,
                                        aggregation=args.event_score_agg)
        sig_scores, _, _ = score_events(model, data, device,
                                        aggregation=args.event_score_agg)
        scores = np.concatenate([bkg_scores, sig_scores])
        labels = np.concatenate([np.zeros(len(bkg_scores)),
                                 np.ones(len(sig_scores))]).astype(np.int64)
        print(f"Zero-shot : {len(bkg_scores):,} bkg (label0 of {args.bkg_dir}) "
              f"vs {len(sig_scores):,} sig ({args.data_dir})")
    else:                                    # labelled eval
        scores, labels, _ = score_events(model, data, device,
                                         aggregation=args.event_score_agg)

    metrics = report(scores, labels)
    metrics.update({
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "bkg_dir": None if args.bkg_dir is None else str(args.bkg_dir),
        "event_score_aggregation": args.event_score_agg,
        "device": str(device),
    })
    print(f"Scored {len(scores):,} events in {time.time()-t0:.0f}s")
    print(f"  AUC          : {metrics['auc']:.4f}")
    print(f"  max SIC      : {metrics['max_sic']:.3f}  "
          f"(threshold={metrics['best_threshold']:.6g})")
    print(f"  eS @ 100x rej: {metrics['eS_at_eB1e-2']:.4f}   "
          f"@ 1000x: {metrics['eS_at_eB1e-3']:.4f}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "scores.npy", scores)
        np.save(out / "labels.npy", labels)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  saved → {out}/")
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Score graph shards with a trained AE")
    p.add_argument("--checkpoint", required=True, help="best.pt / last.pt")
    p.add_argument("--data_dir", required=True,
                   help="graph-shard dir to score (signal-only in zero-shot mode)")
    p.add_argument("--bkg_dir", default=None,
                   help="held-out background shard dir → zero-shot AUC vs data_dir")
    p.add_argument("--output_dir", default=None,
                   help="write scores.npy / labels.npy / metrics.json here")
    p.add_argument("--event_score_agg", default="sum",
                   choices=EVENT_SCORE_AGGREGATIONS,
                   help="event-level anomaly score aggregation")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
