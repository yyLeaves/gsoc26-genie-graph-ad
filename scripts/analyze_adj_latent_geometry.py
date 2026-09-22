"""Mechanism analysis: latent geometry before/after adjacent-subjet reg.

For each checkpoint, encode the same balanced probe jets and measure:

  * background intra-jet pairwise latent distance (median, q90)
  * signal–background separation of jet-pooled latents
  * participation ratio of background latent covariance
  * EB3 2-d scatter (directly plottable)

Outputs:
  runs/adj_mechanism/latent_geometry.json
  figs/adj_mechanism_latent_geometry.png
  appends a section to runs/adj_mechanism/summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.utils import scatter

from src.data.dataset import JetDataset
from src.data.iterate import prefetch, shard_iter
from src.models.edge_graph_ae import EdgeGraphAE
from src.models.inputs import require_edge_features, select_node_features
from src.objectives.topo_reg import encoder_hidden

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS = {
    "baseline": ROOT / "runs/mjj_window_exclude3600_4000/"
    "reference_pure_seed123/20260721_231710/last.pt",
    "unique_EB3_sum": ROOT / "runs/topology_cheap_signals/"
    "mjj_window_exclude3600_4000/laman6/unique_sum/finetuned.pt",
    "unique_EB2_sum": ROOT / "runs/topology_cheap_signals/"
    "mjj_window_exclude3600_4000/laman6/hidden_unique_sum/finetuned.pt",
    "graph_EB3_sum": ROOT / "runs/topology_cheap_signals/"
    "mjj_window_exclude3600_4000/laman6/graph_sum/finetuned.pt",
}

CHECKPOINTS_FROMSCRATCH = {
    "baseline": ROOT / "runs/mjj_window_exclude3600_4000/"
    "reference_pure_seed123/20260721_231710/last.pt",
    "unique_EB3_sum": ROOT / "runs/mjj_window_exclude3600_4000/"
    "fromscratch_topo/unique_sum_lam1.0_seed123/20260729_223522/last.pt",
    "unique_EB2_sum": ROOT / "runs/mjj_window_exclude3600_4000/"
    "fromscratch_topo/hidden_unique_sum_lam1.0_seed123/20260729_223834/last.pt",
    "graph_EB3_sum": ROOT / "runs/mjj_window_exclude3600_4000/"
    "fromscratch_topo/graph_sum_lam1.0_seed123/20260729_223522/last.pt",
}


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def load_edge_graph_ae(path: Path, device: torch.device) -> EdgeGraphAE:
    """Load legacy ``edgeae`` / current ``edge_graph`` checkpoints."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"unrecognised checkpoint format: {path}")
    spec = dict(payload["model"]["spec"])
    state = payload["model"]["state"]
    # Historical alias.
    if spec.get("type") in {"edgeae", "edge_ae"}:
        spec["type"] = "edge_graph"
    model = EdgeGraphAE(
        in_dim=int(spec["in_dim"]),
        edge_dim=int(spec.get("edge_dim", 3)),
        hidden_dim=int(spec.get("hidden_dim", 64)),
        latent_dim=int(spec.get("latent_dim", 2)),
        edge_weight=float(spec.get("edge_weight", 1.0)),
        aggr=str(spec.get("aggr", "mean")),
        dropout=float(spec.get("dropout", 0.0)),
        feature_cols=tuple(spec["feature_cols"])
        if spec.get("feature_cols") is not None else None,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def sample_event_jet_indices(ds: JetDataset, label: int, n_events: int) -> np.ndarray:
    labels = np.asarray(ds.labels, dtype=np.int64)
    event_ids = np.asarray(ds.event_ids, dtype=np.int64)
    first_pos: dict[int, int] = {}
    for i, (y, eid) in enumerate(zip(labels, event_ids)):
        if y != label:
            continue
        first_pos.setdefault(int(eid), i)
    ordered = sorted(first_pos, key=first_pos.get)
    chosen = np.asarray(ordered[:n_events], dtype=np.int64)
    return np.sort(np.flatnonzero(
        np.isin(event_ids, chosen) & (labels == label)).astype(np.int64))


def participation_ratio(cov: np.ndarray) -> float:
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    if eig.sum() <= 0:
        return float("nan")
    return float((eig.sum() ** 2) / (eig ** 2).sum())


def pairwise_intra_distances(points: np.ndarray, max_graphs: int = 400) -> np.ndarray:
    """Median pairwise distance within each point cloud."""
    out = []
    for i, cloud in enumerate(points):
        if i >= max_graphs:
            break
        if cloud.shape[0] < 2:
            continue
        # subsample nodes if huge
        if cloud.shape[0] > 40:
            rng = np.random.default_rng(i)
            idx = rng.choice(cloud.shape[0], 40, replace=False)
            cloud = cloud[idx]
        d = np.linalg.norm(cloud[:, None, :] - cloud[None, :, :], axis=-1)
        iu = np.triu_indices(d.shape[0], k=1)
        out.append(float(np.median(d[iu])))
    return np.asarray(out, dtype=np.float64)


@torch.no_grad()
def collect_embeddings(
    model: EdgeGraphAE,
    ds: JetDataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    eb2_mean, eb3_mean = [], []
    eb2_clouds, eb3_clouds = [], []
    labels, event_ids, n_nodes = [], [], []
    for batch in prefetch(shard_iter(
            ds, indices, batch_size,
            shuffle_shards=False, shuffle_within=False, device=device),
            depth=2):
        node_target = select_node_features(
            batch, model.in_dim, model.feature_cols)
        edge_attr = require_edge_features(batch, model.edge_dim)
        eb2 = encoder_hidden(
            model, node_target, batch.edge_index, edge_attr)
        output = model.forward(node_target, batch.edge_index, edge_attr)
        eb3 = output.latent
        # mean-pool per jet
        eb2_m = scatter(
            eb2, batch.batch, dim=0, dim_size=batch.num_graphs, reduce="mean")
        eb3_m = scatter(
            eb3, batch.batch, dim=0, dim_size=batch.num_graphs, reduce="mean")
        eb2_mean.append(eb2_m.cpu().numpy())
        eb3_mean.append(eb3_m.cpu().numpy())
        # per-graph clouds (CPU lists)
        for g in range(batch.num_graphs):
            mask = batch.batch == g
            eb2_clouds.append(eb2[mask].cpu().numpy())
            eb3_clouds.append(eb3[mask].cpu().numpy())
            n_nodes.append(int(mask.sum().item()))
        labels.append(batch.y.view(-1).cpu().numpy())
        event_ids.append(batch.event_id.view(-1).cpu().numpy())
    return {
        "eb2_mean": np.concatenate(eb2_mean),
        "eb3_mean": np.concatenate(eb3_mean),
        "eb2_clouds": eb2_clouds,
        "eb3_clouds": eb3_clouds,
        "label": np.concatenate(labels).astype(np.int64),
        "event_id": np.concatenate(event_ids).astype(np.int64),
        "n_nodes": np.asarray(n_nodes, dtype=np.int64),
    }


def geometry_stats(emb: dict[str, np.ndarray]) -> dict:
    labels = emb["label"]
    bkg = labels == 0
    sig = labels == 1
    out = {}
    for level in ("eb2", "eb3"):
        mean = emb[f"{level}_mean"]
        clouds = emb[f"{level}_clouds"]
        bkg_clouds = [c for c, y in zip(clouds, labels) if y == 0]
        sig_clouds = [c for c, y in zip(clouds, labels) if y == 1]
        bkg_intra = pairwise_intra_distances(bkg_clouds)
        sig_intra = pairwise_intra_distances(sig_clouds)
        # jet-pooled separation
        mu_b = mean[bkg].mean(axis=0)
        mu_s = mean[sig].mean(axis=0)
        sep = float(np.linalg.norm(mu_s - mu_b))
        bkg_spread = float(np.median(np.linalg.norm(mean[bkg] - mu_b, axis=1)))
        cov = np.cov(mean[bkg], rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        out[level] = {
            "bkg_intra_median": float(np.median(bkg_intra)) if len(bkg_intra) else float("nan"),
            "bkg_intra_q90": float(np.quantile(bkg_intra, 0.9)) if len(bkg_intra) else float("nan"),
            "sig_intra_median": float(np.median(sig_intra)) if len(sig_intra) else float("nan"),
            "sig_intra_q90": float(np.quantile(sig_intra, 0.9)) if len(sig_intra) else float("nan"),
            "pooled_centroid_separation": sep,
            "pooled_bkg_spread": bkg_spread,
            "separation_over_spread": sep / max(bkg_spread, 1e-12),
            "participation_ratio": participation_ratio(np.atleast_2d(cov)),
            "pooled_dim": int(mean.shape[1]),
        }
    return out


def plot_geometry(all_emb: dict[str, dict], out: Path) -> None:
    names = list(all_emb)
    fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6.2))
    for col, name in enumerate(names):
        emb = all_emb[name]
        labels = emb["label"]
        # EB3 scatter
        ax = axes[0, col]
        z = emb["eb3_mean"]
        ax.scatter(
            z[labels == 0, 0], z[labels == 0, 1],
            s=4, alpha=0.25, c="#4c4c4c", label="bkg", rasterized=True)
        ax.scatter(
            z[labels == 1, 0], z[labels == 1, 1],
            s=4, alpha=0.35, c="#d62728", label="sig", rasterized=True)
        ax.set_title(f"{name}\nEB3 pooled", fontsize=9)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("EB3 dim 1")
            ax.legend(fontsize=7, frameon=False, markerscale=2)
        ax.set_xlabel("EB3 dim 0")
        # EB2 PCA 2-d for visualisation
        ax = axes[1, col]
        h = emb["eb2_mean"]
        h0 = h - h.mean(axis=0, keepdims=True)
        # thin SVD for top-2
        _, _, vt = np.linalg.svd(h0, full_matrices=False)
        proj = h0 @ vt[:2].T
        ax.scatter(
            proj[labels == 0, 0], proj[labels == 0, 1],
            s=4, alpha=0.25, c="#4c4c4c", rasterized=True)
        ax.scatter(
            proj[labels == 1, 0], proj[labels == 1, 1],
            s=4, alpha=0.35, c="#d62728", rasterized=True)
        ax.set_title("EB2 pooled (PCA-2)", fontsize=9)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("PC2")
        ax.set_xlabel("PC1")
    fig.suptitle(
        "Adjacent-subjet latent geometry (same 2k+2k probe jets)",
        y=1.02, fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def append_summary(stats: dict, path: Path) -> None:
    lines = [
        "",
        "# Latent geometry",
        "",
        "Same 2k+2k probe jets as residual decomp. Intra = median pairwise "
        "distance inside each jet's node embedding cloud.",
        "",
        "## EB3 (2-d bottleneck)",
        "",
        "| Method | bkg intra med | bkg intra q90 | sep/spread | PR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in stats.items():
        e = s["eb3"]
        lines.append(
            f"| `{name}` | {e['bkg_intra_median']:.3f} | "
            f"{e['bkg_intra_q90']:.3f} | "
            f"{e['separation_over_spread']:.3f} | "
            f"{e['participation_ratio']:.2f} |"
        )
    lines += [
        "",
        "## EB2 (64-d hidden)",
        "",
        "| Method | bkg intra med | bkg intra q90 | sep/spread | PR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in stats.items():
        e = s["eb2"]
        lines.append(
            f"| `{name}` | {e['bkg_intra_median']:.3f} | "
            f"{e['bkg_intra_q90']:.3f} | "
            f"{e['separation_over_spread']:.3f} | "
            f"{e['participation_ratio']:.2f} |"
        )
    base = stats["baseline"]
    lines += ["", "## Auto-read", ""]
    for name, s in stats.items():
        if name == "baseline":
            continue
        shrink = (
            s["eb3"]["bkg_intra_median"]
            / max(base["eb3"]["bkg_intra_median"], 1e-12) - 1.0
        )
        sep = (
            s["eb3"]["separation_over_spread"]
            / max(base["eb3"]["separation_over_spread"], 1e-12) - 1.0
        )
        if shrink < -0.1 and sep > 0.05:
            verdict = (
                "background latent contracts and signal–bkg separation "
                "improves — supports selective QCD-manifold compression."
            )
        elif shrink < -0.1 and sep <= 0.05:
            verdict = (
                "background contracts but separation does not improve — "
                "likely isotropic collapse."
            )
        elif shrink >= -0.1 and sep > 0.05:
            verdict = (
                "separation improves without strong contraction — "
                "geometry rearranges rather than shrinks."
            )
        else:
            verdict = "little geometric change at EB3 vs baseline."
        lines.append(f"- `{name}`: {verdict} (Δintra={shrink:+.0%}, Δsep={sep:+.0%}).")
    lines.append("")
    existing = path.read_text() if path.exists() else ""
    # replace previous latent section if re-run
    marker = "\n# Latent geometry\n"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"
    path.write_text(existing + "\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data_dir",
        default=str(
            ROOT / "dataset/processed/"
            "lhco_leadingpt_sj30_unique6_mjj_exclude3600_4000_trainpack_seed42"),
    )
    p.add_argument("--n_events", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--cache_shards", type=int, default=8)
    p.add_argument("--output_dir", default=str(ROOT / "runs/adj_mechanism"))
    p.add_argument("--fig_dir", default=str(ROOT / "figs"))
    p.add_argument("--preset", choices=("finetune", "fromscratch"),
                   default="finetune")
    p.add_argument("--fig_name", default="adj_mechanism_latent_geometry.png")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    ckpts = (
        CHECKPOINTS_FROMSCRATCH if args.preset == "fromscratch" else CHECKPOINTS
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = JetDataset(args.data_dir, max_cache=args.cache_shards)
    probe_idx = np.sort(np.concatenate([
        sample_event_jet_indices(ds, 0, args.n_events),
        sample_event_jet_indices(ds, 1, args.n_events),
    ]))
    print(f"Device={device}  probe jets={len(probe_idx):,}", flush=True)

    all_emb = {}
    stats = {}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ckpt in ckpts.items():
        print(f"=== {name} ===", flush=True)
        model = load_edge_graph_ae(Path(ckpt), device)
        # soft check: in_dim / edge_dim
        if model.in_dim != 1 or model.edge_dim != 3:
            raise SystemExit(f"{name}: unexpected dims {model.in_dim}/{model.edge_dim}")
        emb = collect_embeddings(
            model, ds, probe_idx, device, args.batch_size)
        # cache means for replot
        np.savez_compressed(
            out_dir / f"{name}_latent_means.npz",
            eb2_mean=emb["eb2_mean"],
            eb3_mean=emb["eb3_mean"],
            label=emb["label"],
            event_id=emb["event_id"],
            n_nodes=emb["n_nodes"],
        )
        all_emb[name] = emb
        stats[name] = geometry_stats(emb)
        print(
            f"  EB3 bkg_intra={stats[name]['eb3']['bkg_intra_median']:.3f}  "
            f"sep/spread={stats[name]['eb3']['separation_over_spread']:.3f}  "
            f"PR={stats[name]['eb3']['participation_ratio']:.2f}",
            flush=True,
        )

    (out_dir / "latent_geometry.json").write_text(
        json.dumps(_jsonable(stats), indent=2))
    plot_geometry(all_emb, Path(args.fig_dir) / args.fig_name)
    append_summary(stats, out_dir / "summary.md")
    print(f"Wrote {out_dir / 'latent_geometry.json'}")
    print(f"Wrote {Path(args.fig_dir) / args.fig_name}")


if __name__ == "__main__":
    main()
