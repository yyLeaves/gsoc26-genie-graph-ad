import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from .blocks import mse_per_graph


class EdgeConvEF(MessagePassing):
    """EdgeConv that uses edge features (Araz et al.'s EdgeConvWithEdgeFeatures).

    Message = nn([x_i ‖ edge_attr]); decoder mode replaces edge_attr with
    (x_j - x_i), the standard EdgeConv difference.
    """

    def __init__(self, nn_module: nn.Module, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.nn = nn_module

    def forward(self, x, edge_index, edge_attr, dec):
        if edge_attr is None:
            # Only the decoder passes None (message() then uses x_j - x_i); the
            # 0-column placeholder avoids implying a wrong edge-feature width.
            # Encoder blocks need real edge_attr or the concat dim is wrong.
            edge_attr = torch.zeros((edge_index.size(1), 0), device=x.device)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, dec=dec)

    def message(self, x_i, x_j, edge_attr, dec):
        if dec:
            edge_attr = x_j - x_i
        return self.nn(torch.cat([x_i, edge_attr], dim=-1))


class EdgeBlock(nn.Module):
    """EdgeConvEF + linear residual skip (Araz et al.'s 'Block')."""

    def __init__(self, node_in, node_out, edge_dim, hidden=64, aggr="mean",
                 dropout=0.0, dec=False, final=False):
        super().__init__()
        self.residual = (nn.Linear(node_in, node_out)
                         if node_in != node_out else nn.Identity())
        self.dec = dec
        enc_in = 2 * node_in if dec else node_in + edge_dim
        mlp = nn.Sequential(
            nn.Linear(enc_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, node_out),
            nn.Identity() if final else nn.ReLU(),
        )
        self.edgeconv = EdgeConvEF(mlp, aggr=aggr)

    def forward(self, x, edge_index, edge_attr):
        return self.edgeconv(x, edge_index, edge_attr, dec=self.dec) \
            + self.residual(x)


class EdgeAttrPredictor(nn.Module):
    """Edge-feature predictor (Araz et al.'s EdgeAttrPredictor): symmetric
    min/max pair feature → deep path (fc1, residual fc2, fc3) + direct linear,
    summed."""

    def __init__(self, latent_dim, edge_dim, hidden=64, dropout=0.0):
        super().__init__()
        self.fc_direct = nn.Linear(2 * latent_dim, edge_dim)
        self.fc1 = nn.Linear(2 * latent_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, edge_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        src, tgt = edge_index
        x_cat = torch.cat([torch.min(x[src], x[tgt]),
                           torch.max(x[src], x[tgt])], dim=-1)
        h = self.dropout(self.relu(self.fc1(x_cat)))
        h = self.dropout(self.relu(self.fc2(h) + h))   # residual block
        return self.fc3(h) + self.fc_direct(x_cat)     # deep + direct paths


class EdgeGraphAE(nn.Module):
    """Edge-feature graph autoencoder (faithful Araz reference).

    Args:
        in_dim:      node feature dim (1 = pT only)
        edge_dim:    edge feature dim (3 = lnΔR, ln k_T, ln z)
        latent_dim:  per-node bottleneck (2)
        edge_weight: λ on edge MSE term
    """

    COMPONENT_NAMES = ("node", "edge")

    def __init__(self, in_dim, edge_dim=3, hidden_dim=64, latent_dim=2,
                 edge_weight=1.0, aggr="mean", dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.edge_weight = edge_weight
        self.aggr = aggr
        self.dropout = dropout

        # encoder: blocks [in→64], [64→64], [64→latent]
        enc_dims = [hidden_dim, hidden_dim, latent_dim]
        self.encoder_blocks = nn.ModuleList()
        cur = in_dim
        for out in enc_dims:
            self.encoder_blocks.append(
                EdgeBlock(cur, out, edge_dim, hidden=hidden_dim, aggr=aggr,
                         dropout=dropout, dec=False))
            cur = out

        # decoder: blocks [latent→32], [32→in_dim] (edge_dim=0, dec=True)
        dec_dims = [hidden_dim // 2, in_dim]
        self.decoder_blocks = nn.ModuleList()
        cur = latent_dim
        for i, out in enumerate(dec_dims):
            self.decoder_blocks.append(
                EdgeBlock(cur, out, edge_dim=0, hidden=hidden_dim, aggr=aggr,
                         dropout=dropout, dec=True, final=(i == len(dec_dims) - 1)))
            cur = out

        self.edge_predictor = EdgeAttrPredictor(latent_dim, edge_dim,
                                                 hidden=hidden_dim,
                                                 dropout=dropout)

    def _node_input(self, batch):
        if batch.x.size(-1) < self.in_dim:
            raise ValueError(
                f"batch.x has {batch.x.size(-1)} features, but "
                f"EdgeGraphAE(in_dim={self.in_dim}) requires at least "
                f"{self.in_dim}")
        return batch.x[:, :self.in_dim]

    def _edge_attr(self, batch):
        edge_attr = getattr(batch, "edge_attr", None)
        if edge_attr is None:
            raise ValueError("EdgeGraphAE requires batch.edge_attr, got None")
        if edge_attr.size(-1) != self.edge_dim:
            raise ValueError(
                f"batch.edge_attr has {edge_attr.size(-1)} features, "
                f"but EdgeGraphAE(edge_dim={self.edge_dim}) requires "
                f"{self.edge_dim}")
        return edge_attr

    def forward(self, x, edge_index, edge_attr):
        """Returns (x_recon, e_pred, z)."""
        h = x
        for block in self.encoder_blocks:
            h = block(h, edge_index, edge_attr)
        z = h  # latent (per node)
        r = z
        for block in self.decoder_blocks:
            r = block(r, edge_index, None)
        e_pred = self.edge_predictor(z, edge_index)
        return r, e_pred, z

    def loss(self, batch):
        """Training loss → (total, node, edge).  total = node + edge_weight·edge."""
        x = self._node_input(batch)
        edge_attr = self._edge_attr(batch)
        x_hat, e_hat, _ = self.forward(x, batch.edge_index, edge_attr)
        node_loss = F.mse_loss(x_hat, x)
        edge_loss = F.mse_loss(e_hat, edge_attr)
        return node_loss + self.edge_weight * edge_loss, node_loss, edge_loss

    @torch.no_grad()
    def anomaly_score(self, batch):
        """Per-graph score = mean node MSE + λ·mean edge MSE → (B,)."""
        x = self._node_input(batch)
        edge_attr = self._edge_attr(batch)
        x_hat, e_hat, _ = self.forward(x, batch.edge_index, edge_attr)
        node_score = mse_per_graph(x_hat, x, batch.batch)
        edge_score = mse_per_graph(e_hat, edge_attr,
                                   batch.batch[batch.edge_index[0]],
                                   n_graphs=node_score.shape[0])
        return node_score + self.edge_weight * edge_score

    def extra_repr(self) -> str:
        return (f"in_dim={self.in_dim}, edge_dim={self.edge_dim}, "
                f"hidden_dim={self.hidden_dim}, latent_dim={self.latent_dim}, "
                f"edge_weight={self.edge_weight}, aggr={self.aggr!r}, "
                f"dropout={self.dropout}")
