"""
CLI for training a graph autoencoder on LHCO jet graphs.
The training logic lives in src.training.trainer; this file is just argument
parsing.

    python -m scripts.train_graph_ae \\
        --data_dir dataset/processed/lhco_canonical_leadingpt_sj30_unique6_logef \\
        --split_manifest dataset/processed/splits/...npz \\
        --output runs/leadingpt \\
        --no_early_stop

Per-run outputs land in <output>/<timestamp>/: config.json, split.npz, best.pt,
last.pt, history.json, metrics_best.json, metrics_last.json,
metrics_comparison.json, primary scores.npy/labels.npy/metrics.json, and the log.

Resume an interrupted run with ``--resume <output>/<timestamp>/last.pt``.
"""

import argparse

from src.eval.scoring import EVENT_SCORE_AGGREGATIONS
from src.models import BACKBONES, MODEL_TYPES
from src.training.splits import validate_split_args
from src.training.trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="Train a graph autoencoder on LHCO jets")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="edge_graph", choices=MODEL_TYPES,
                   help="edge_graph = EdgeGraphAE (default); node_graph = "
                        "node-feature AE; edge_feature_graph = edge-recon "
                        "variant")
    p.add_argument("--backbone", default="edgeconv", choices=BACKBONES)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--latent_dim", type=int, default=2,
                   help="Per-node bottleneck; must stay below in_dim or the "
                        "AE can learn the identity map")
    p.add_argument("--no_bn", action="store_true",
                   help="Disable BatchNorm (recommended for anomaly detection)")
    p.add_argument("--scheduler", default="onecycle",
                   choices=["cosine", "onecycle", "linear"],
                   help="onecycle steps per batch; linear = per-epoch anneal "
                        "lr→lr_end. Pair onecycle/linear with --no_early_stop "
                        "so the schedule can finish.")
    p.add_argument("--lr_end", type=float, default=2e-4,
                   help="Final lr for the linear scheduler")
    p.add_argument("--edge_weight", type=float, default=1.0,
                   help="Weight on the edge-reconstruction MSE term")
    p.add_argument("--node_features", default="pt", choices=["pt", "all"],
                   help="For pt-default models: column 0 only (Araz form) or "
                        "full node-feature vector")
    p.add_argument("--aggr", default="mean", choices=["mean", "add", "max"],
                   help="Message aggregation for edge models")
    p.add_argument("--dyn_k", type=int, default=16,
                   help="k for dynamic-graph models")
    p.add_argument("--event_score_agg", default="sum",
                   choices=EVENT_SCORE_AGGREGATIONS,
                   help="Event-level anomaly score aggregation")
    p.add_argument("--topo_reg", default="none",
                   choices=["none", "unique_sum", "unique_max",
                            "unique_ang_sum", "unique_ang_max",
                            "graph_sum", "graph_max",
                            "hidden_unique_sum", "hidden_unique_max",
                            "hidden_graph_sum", "hidden_graph_max"],
                   help="Optional latent/hidden topology regularizer added to "
                        "the train objective (val/anomaly scores stay recon-only)")
    p.add_argument("--lambda_topo", type=float, default=1.0,
                   help="Weight on --topo_reg (ignored when topo_reg=none)")
    p.add_argument("--unique_k", type=int, default=6,
                   help="k for unique_* topo regularizers")
    p.add_argument("--no_early_stop", action="store_true",
                   help="Train the full --epochs; primary final metrics use "
                        "last.pt and best.pt is still reported for comparison")
    p.add_argument("--save_monitor_best", action="store_true",
                   help="Also save monitor_best.pt when epoch AUC improves "
                        "(oracle diagnostic; not the primary result)")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--eval_interval", type=int, default=10,
                   help="AUC/SIC every N epochs (0=disable)")
    p.add_argument("--cache_shards", type=int, default=4,
                   help="Max shards in LRU cache")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Use only the first fraction of shards (pilot runs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume", default=None,
                   help="Resume from <output>/<timestamp>/last.pt only")
    p.add_argument("--split_protocol", default="manifest",
                   choices=["manifest", "ks_fixed"],
                   help="manifest: event split from --split_manifest. "
                        "ks_fixed: deterministic Kitchen-Sink-style counts.")
    p.add_argument("--split_manifest", default=None,
                   help="Event-level .npz with train_bkg_events, "
                        "train_sig_events, val_bkg_events, monitor_sig_events, "
                        "test_bkg_events, test_sig_events")
    p.add_argument("--train_bkg_events", type=int, default=80_000,
                   help="ks_fixed: background training events")
    p.add_argument("--val_bkg_events", type=int, default=20_000,
                   help="ks_fixed: background validation events")
    p.add_argument("--test_bkg_events", type=int, default=340_000,
                   help="ks_fixed: held-out background test events")
    p.add_argument("--test_sig_events", type=int, default=20_000,
                   help="ks_fixed: held-out signal test events")
    args = p.parse_args()
    validate_split_args(args)
    return args


if __name__ == "__main__":
    train(parse_args())
