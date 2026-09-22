"""Training CLI options, grouped by purpose; defaults live only here."""

import argparse

from src.eval.scoring import EVENT_SCORE_AGGREGATIONS
from src.models import BACKBONES, MODEL_TYPES
from .splits import validate_split_args


def parse_args():
    p = argparse.ArgumentParser(description="Train a graph autoencoder on LHCO jets")
    group = p.add_argument_group("Data and output")
    group.add_argument("--data_dir", required=True)
    group.add_argument("--output", required=True)
    group.add_argument("--split_protocol", default="manifest",
                   choices=["manifest", "ks_fixed"],
                   help="manifest: event split from --split_manifest. "
                        "ks_fixed: deterministic Kitchen-Sink-style counts.")
    group.add_argument("--split_manifest", default=None,
                   help="Event-level .npz with train_bkg_events, "
                        "train_sig_events, val_bkg_events, monitor_sig_events, "
                        "test_bkg_events, test_sig_events")
    group.add_argument("--train_bkg_events", type=int, default=80_000,
                   help="ks_fixed: background training events")
    group.add_argument("--val_bkg_events", type=int, default=20_000,
                   help="ks_fixed: background validation events")
    group.add_argument("--test_bkg_events", type=int, default=340_000,
                   help="ks_fixed: held-out background test events")
    group.add_argument("--test_sig_events", type=int, default=20_000,
                   help="ks_fixed: held-out signal test events")
    group.add_argument("--fraction", type=float, default=1.0,
                   help="Use only the first fraction of shards (pilot runs)")
    group.add_argument("--cache_shards", type=int, default=4,
                   help="Max shards in LRU cache")
    group.add_argument("--prefetch_batches", type=int, default=4,
                       help="Background prefetch depth; 0 uses synchronous loading")

    group = p.add_argument_group("Model")
    group.add_argument("--model", default="edge_graph", choices=MODEL_TYPES,
                   help="edge_graph = EdgeGraphAE (default); node_graph = "
                        "node-feature AE; edge_feature_graph = edge-recon "
                        "variant; reference_backbone_edge_graph = reference "
                        "decoder with a replaced encoder backbone")
    group.add_argument("--backbone", default="edgeconv", choices=BACKBONES)
    group.add_argument("--hidden_dim", type=int, default=64)
    group.add_argument("--latent_dim", type=int, default=2,
                   help="Dimension of each node's latent representation")
    group.add_argument("--no_bn", action="store_true",
                   help="Disable BatchNorm (recommended for anomaly detection)")
    group.add_argument("--edge_weight", type=float, default=1.0,
                   help="Weight on the edge-reconstruction MSE term")
    group.add_argument("--node_features", default="pt", choices=["pt", "all"],
                   help="For pt-default models: column 0 only (Araz form) or "
                        "full node-feature vector")
    group.add_argument("--aggr", default="mean", choices=["mean", "add", "max"],
                   help="Message aggregation for edge models")
    group.add_argument("--dyn_k", type=int, default=16,
                   help="k for dynamic-graph models")

    group = p.add_argument_group("Training")
    group.add_argument("--node_reconstruction", choices=("mse", "official_l2"),
                       default="mse", help="Node training/validation loss; evaluation scores stay fixed")
    group.add_argument("--epochs", type=int, default=50)
    group.add_argument("--batch_size", type=int, default=512)
    group.add_argument("--scheduler", default="onecycle",
                   choices=["cosine", "onecycle", "linear"],
                   help="onecycle steps per batch; linear = per-epoch anneal "
                        "lr→lr_end. Pair onecycle/linear with --no_early_stop "
                        "so the schedule can finish.")
    group.add_argument("--lr", type=float, default=3e-3)
    group.add_argument("--lr_end", type=float, default=2e-4,
                   help="Final lr for the linear scheduler")
    group.add_argument("--weight_decay", type=float, default=0.01)
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--cpu", action="store_true")
    group.add_argument("--resume", default=None,
                   help="Resume from <output>/<timestamp>/last.pt only")
    group.add_argument("--no_early_stop", action="store_true",
                   help="Train the full --epochs; primary final metrics use "
                        "last.pt and best.pt is still reported for comparison")
    group.add_argument("--patience", type=int, default=15)

    group = p.add_argument_group("Evaluation")
    group.add_argument("--eval_interval", type=int, default=10,
                   help="AUC/SIC every N epochs (0=disable)")
    group.add_argument("--save_monitor_best", action="store_true",
                   help="Also save monitor_best.pt when epoch AUC improves "
                        "(oracle diagnostic; not the primary result)")
    group.add_argument("--event_score_agg", default="sum",
                   choices=EVENT_SCORE_AGGREGATIONS,
                   help="Event-level anomaly score aggregation")
    group.add_argument("--anomaly_score", default="reconstruction",
                   choices=["reconstruction", "cycle", "cycle_local",
                            "cycle_global"],
                   help="Jet anomaly score. Cycle modes use the reconstructed "
                        "graph re-encoding discrepancy.")

    group = p.add_argument_group("Adjacent-subjet regularization")
    group.add_argument("--topo_reg", default="none",
                   help="Topology regularizer: atomic name, alias "
                        "(graph_eb3_attraction), or joint alias "
                        "(graph_eb2_eb3_attraction). Val/anomaly stay "
                        "reconstruction-only.")
    group.add_argument("--lambda_topo", default="1.0",
                   help="Weight(s) on --topo_reg: scalar or comma list "
                        "matching each '+' term (ignored when topo_reg=none)")
    group.add_argument("--topo_norm", action="store_true",
                   help="L2-normalize node representations before computing "
                        "the topology regularizer")
    group.add_argument("--unique_k", type=int, default=6,
                   help="k for unique_* topo regularizers")

    group = p.add_argument_group("Latent cycle")
    group.add_argument("--cycle_weight", type=float, default=0.0,
                   help="Weight beta on latent re-encoding consistency. "
                        "Zero keeps the exact reconstruction baseline path.")
    group.add_argument("--cycle_component", default="total",
                   choices=["total", "local", "global"],
                   help="Latent-cycle term optimized when --cycle_weight is "
                        "nonzero. total = local + global.")
    group.add_argument("--cycle_global_mode", default="pooled",
                   choices=["pooled", "projected", "unit_projected"],
                   help="Global cycle compares max-pooled latents, projection "
                        "outputs, or unit-normalized projection outputs.")

    group = p.add_argument_group("Graph contrast")
    group.add_argument("--contrast_weight", type=float, default=0.0,
                   help="Weight on GLADC clean/perturbed graph contrast. "
                        "Zero keeps the exact reconstruction baseline path.")
    group.add_argument("--perturb_scale", type=float, default=1.0,
                   help="Encoder-weight perturbation scale eta")
    group.add_argument("--contrast_temperature", type=float, default=0.2,
                   help="Cosine contrastive temperature")
    group.add_argument("--contrast_projection_dim", type=int, default=64,
                   help="Graph-level projection dimension when contrast is on")
    group.add_argument("--contrast_center", action="store_true",
                       help="Subtract each branch's batch mean inside contrastive loss")
    group.add_argument("--contrast_source", default="latent",
                   choices=["latent", "eb2"],
                   help="Encoder representation used by GLADC contrast: "
                        "2D bottleneck or 64D second encoder block")

    group = p.add_argument_group("Edge relation")
    group.add_argument("--relation_objective", default="none",
                   choices=["none", "nce", "vicreg"],
                   help="Training-only physical-edge relation objective")
    group.add_argument("--relation_weight", type=float, default=0.0,
                   help="Weight on the relation objective")
    group.add_argument("--relation_node_mask", type=float, default=0.15,
                   help="VICReg fraction of soft nodes replaced per graph")
    group.add_argument("--relation_edge_mask", type=float, default=0.10,
                   help="Fraction of physical edge attributes masked")
    group.add_argument("--relation_temperature", type=float, default=0.2,
                   help="Within-jet NCE temperature")
    group.add_argument("--relation_moment_weight", type=float, default=0.1,
                   help="VICReg clean-moment matching weight")

    args = p.parse_args()
    validate_split_args(args)
    return args


def validate_objectives(args):
    """Check auxiliary losses before opening a run or loading data."""
    if args.cycle_weight < 0.0:
        raise ValueError("cycle_weight must be non-negative")
    if args.cycle_component not in {"total", "local", "global"}:
        raise ValueError("cycle_component must be one of: total, local, global")
    if args.contrast_weight < 0.0:
        raise ValueError("contrast_weight must be non-negative")
    if args.contrast_projection_dim <= 0:
        raise ValueError("contrast_projection_dim must be positive")
    if args.perturb_scale < 0.0:
        raise ValueError("perturb_scale must be non-negative")
    if args.contrast_temperature <= 0.0:
        raise ValueError("contrast_temperature must be positive")
    if args.relation_weight < 0.0:
        raise ValueError("relation_weight must be non-negative")
    if args.relation_objective == "none" and args.relation_weight:
        raise ValueError("--relation_weight requires --relation_objective nce or vicreg")
    if args.relation_objective != "none" and args.relation_weight <= 0.0:
        raise ValueError("--relation_objective requires a positive --relation_weight")

    uses_latent_objective = (
        args.cycle_weight > 0
        or args.contrast_weight > 0
        or args.relation_objective != "none"
        or args.anomaly_score != "reconstruction"
    )
    if uses_latent_objective and args.model != "edge_graph":
        raise ValueError("cycle, contrast and relation objectives require --model edge_graph")
