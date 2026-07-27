import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from .inputs import (normalize_feature_cols, require_edge_features,
                     select_node_features)
from .reconstruction import (EdgeAttrPredictor, Reconstruction,
                             ReconstructionMixin)


class EdgeFeatureConv(MessagePassing):
    """Encoder EdgeConv: message = nn([x_i ‖ edge_attr])."""

    def __init__(self, nn_module: nn.Module, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.nn = nn_module

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, edge_attr):
        return self.nn(torch.cat([x_i, edge_attr], dim=-1))


class EdgeDiffConv(MessagePassing):
    """Decoder EdgeConv: message = nn([x_i ‖ x_j - x_i])."""

    def __init__(self, nn_module: nn.Module, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.nn = nn_module

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))


class EdgeBlock(nn.Module):
    """EdgeConv + linear residual skip (Araz et al.'s 'Block')."""

    def __init__(self, node_in, node_out, edge_dim, hidden=64, aggr="mean",
                 dropout=0.0, dec=False, final=False):
        super().__init__()
        self.residual = (nn.Linear(node_in, node_out)
                         if node_in != node_out else nn.Identity())
        self.decode = dec
        msg_in = 2 * node_in if dec else node_in + edge_dim
        layers: list[nn.Module] = [
            nn.Linear(msg_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, node_out),
        ]
        if not final:
            layers.append(nn.ReLU())
        mlp = nn.Sequential(*layers)
        self.edgeconv = (
            EdgeDiffConv(mlp, aggr=aggr) if dec
            else EdgeFeatureConv(mlp, aggr=aggr)
        )

    def forward(self, x, edge_index, edge_attr=None):
        if self.decode:
            return self.edgeconv(x, edge_index) + self.residual(x)
        return self.edgeconv(x, edge_index, edge_attr) + self.residual(x)


class EdgeGraphAE(ReconstructionMixin, nn.Module):
    """Edge-feature graph autoencoder (faithful Araz reference).

    Args:
        in_dim:      node feature dim (1 = pT only)
        edge_dim:    edge feature dim (3 = lnΔR, ln k_T, ln z)
        latent_dim:  per-node bottleneck (2)
        edge_weight: λ on edge MSE term
    """

    def __init__(self, in_dim, edge_dim=3, hidden_dim=64, latent_dim=2,
                 edge_weight=1.0, aggr="mean", dropout=0.0,
                 feature_cols: tuple[int, ...] | None = None):
        super().__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.edge_weight = edge_weight
        self.aggr = aggr
        self.dropout = dropout
        self.feature_cols = normalize_feature_cols(feature_cols, in_dim)

        enc_dims = [hidden_dim, hidden_dim, latent_dim]
        self.encoder_blocks = nn.ModuleList()
        cur = in_dim
        for out in enc_dims:
            self.encoder_blocks.append(
                EdgeBlock(cur, out, edge_dim, hidden=hidden_dim, aggr=aggr,
                         dropout=dropout))
            cur = out

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

    def forward(self, x, edge_index, edge_attr):
        h = x
        for block in self.encoder_blocks:
            h = block(h, edge_index, edge_attr)
        z = h
        r = z
        for block in self.decoder_blocks:
            r = block(r, edge_index)
        e_pred = self.edge_predictor(z, edge_index)
        return Reconstruction(node=r, latent=z, edge=e_pred)

    def _reconstruct(self, batch):
        target = select_node_features(batch, self.in_dim, self.feature_cols)
        edge_attr = require_edge_features(batch, self.edge_dim)
        output = self.forward(target, batch.edge_index, edge_attr)
        return output, target, edge_attr

    def extra_repr(self) -> str:
        return (f"in_dim={self.in_dim}, edge_dim={self.edge_dim}, "
                f"hidden_dim={self.hidden_dim}, latent_dim={self.latent_dim}, "
                f"edge_weight={self.edge_weight}, aggr={self.aggr!r}, "
                f"dropout={self.dropout}, feature_cols={self.feature_cols}")
