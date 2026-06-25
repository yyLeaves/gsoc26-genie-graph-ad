"""
CLI for training a graph autoencoder (NodeGraphAE / EdgeGraphAE) on LHCO jet graphs.
The training logic lives in src.training.trainer; this file is just argument
parsing.

    python -m develop.scripts.train_graph_ae \\
        --data_dir develop/dataset/processed/lhco_relae_unique6_logef \\
        --output   develop/runs/relae \\
        --model edgeae --node_features pt --scheduler onecycle \\
        --epochs 50 --no_early_stop

Per-run outputs land in <output>/<timestamp>/: config.json, split.npz, best.pt,
last.pt, history.json, scores.npy, labels.npy, metrics.json, and the log.
"""

import argparse

from src.eval.scoring import EVENT_SCORE_AGGREGATIONS
from src.models import BACKBONES
from src.training.trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="Train a graph autoencoder on LHCO jets")
    p.add_argument("--data_dir",      required=True)
    p.add_argument("--output",        required=True)
    p.add_argument("--model",         default="ae",
                   choices=["ae", "edgeae", "relae"],
                   help="ae = node-feature autoencoder (baseline); "
                        "edgeae = edge-feature reconstructing AE "
                        "(Araz et al.); relae is a deprecated alias. "
                        "Needs graph shards built with --edge_features log")
    p.add_argument("--backbone",      default="edgeconv",  choices=BACKBONES)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=512)
    p.add_argument("--hidden_dim",    type=int,   default=64)
    p.add_argument("--latent_dim",    type=int,   default=2,
                   help="Per-node bottleneck; must stay below in_dim (5) "
                        "or the AE learns the identity map")
    p.add_argument("--no_bn",         action="store_true",
                   help="Remove all BatchNorm (per Araz et al.: BN hurts AD)")
    p.add_argument("--scheduler",     default="onecycle",
                   choices=["cosine", "onecycle", "linear"],
                   help="onecycle steps per batch; linear = per-epoch linear "
                        "anneal lr→lr_end. Default onecycle matches the "
                        "reference repo: AdamW lr=3e-3, linear OneCycleLR, "
                        "~2% warmup, div_factor=5, final_div_factor=3. "
                        "NOTE: linear/"
                        "onecycle only reach lr_end on the final epoch, so "
                        "pair with --no_early_stop or the anneal is cut short")
    p.add_argument("--lr_end",        type=float, default=2e-4,
                   help="Final lr for the linear scheduler (Araz: 2e-4)")
    p.add_argument("--edge_weight",   type=float, default=1.0,
                   help="relae: weight λ on the edge-reconstruction MSE term")
    p.add_argument("--node_features", default="pt", choices=["pt", "all"],
                   help="relae node input: 'pt' (column 0 only, Araz form) or "
                        "'all' (full node-feature vector)")
    p.add_argument("--aggr",          default="mean",
                   choices=["mean", "add", "max"],
                   help="relae message aggregation (mean for dense kNN; "
                        "add matches Araz on sparse graphs)")
    p.add_argument("--event_score_agg", default="sum",
                   choices=EVENT_SCORE_AGGREGATIONS,
                   help="event-level anomaly score aggregation: sum matches "
                        "the reference repo; mean averages selected jets; "
                        "max uses the most anomalous jet; min requires all "
                        "selected jets to look anomalous; pt_weighted uses "
                        "jet ΣpT weights")
    p.add_argument("--no_early_stop", action="store_true",
                   help="Train the full --epochs; final eval uses last.pt "
                        "(required for onecycle to anneal fully)")
    p.add_argument("--lr",            type=float, default=3e-3)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--eval_interval", type=int,   default=10,
                   help="AUC/SIC every N epochs (0=disable)")
    p.add_argument("--cache_shards",  type=int,   default=4,
                   help="Max shards in LRU cache (default=4; shard-sequential "
                        "needs only 1, but 4 gives breathing room for val/eval)")
    p.add_argument("--fraction",      type=float, default=1.0,
                   help="Use only the first fraction of shards (pilot runs)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--cpu",           action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
