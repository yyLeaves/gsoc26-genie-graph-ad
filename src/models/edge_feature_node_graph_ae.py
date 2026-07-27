"""Node autoencoders whose message passing explicitly consumes edge features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GATv2Conv, GCNConv, GINEConv, MessagePassing,
                                TransformerConv)

from .blocks import ATTENTION_HEADS, BACKBONES, mlp
from .inputs import (normalize_feature_cols, require_edge_features,
                     select_node_features)
from .reconstruction import (EdgeAttrPredictor, Reconstruction,
                             ReconstructionMixin)


class EdgeFeatureEdgeConv(MessagePassing):
    """DGCNN message ``MLP[x_i, x_j-x_i, e_ij]`` with max aggregation."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
        super().__init__(aggr="max")
        self.nn = mlp([2 * in_dim + edge_dim, out_dim, out_dim])

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.nn(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))


class EdgeFeatureSAGEConv(MessagePassing):
    """GraphSAGE-style mean aggregation with edge-conditioned messages."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
        super().__init__(aggr="mean")
        self.message_nn = mlp([in_dim + edge_dim, out_dim, out_dim])

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.message_nn(torch.cat([x_j, edge_attr], dim=-1))


class EdgeWeightedGCNConv(nn.Module):
    """GCN with a learned positive scalar weight for every physical edge."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
        super().__init__()
        self.edge_gate = mlp([edge_dim, out_dim, 1], act_last=False)
        self.conv = GCNConv(in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_weight = F.softplus(self.edge_gate(edge_attr).squeeze(-1)) + 1e-6
        return self.conv(x, edge_index, edge_weight=edge_weight)


def make_edge_feature_conv(backbone: str, in_dim: int, out_dim: int,
                           edge_dim: int) -> nn.Module:
    """Build an edge-aware counterpart of each static NodeGraphAE backbone."""
    if backbone == "edgeconv":
        return EdgeFeatureEdgeConv(in_dim, out_dim, edge_dim)
    if backbone == "gcn":
        return EdgeWeightedGCNConv(in_dim, out_dim, edge_dim)
    if backbone == "sage":
        return EdgeFeatureSAGEConv(in_dim, out_dim, edge_dim)
    if backbone == "gatv2":
        return GATv2Conv(in_dim, out_dim, heads=ATTENTION_HEADS, concat=False,
                         dropout=0.0, edge_dim=edge_dim)
    if backbone == "gin":
        return GINEConv(mlp([in_dim, out_dim, out_dim]), train_eps=True,
                        edge_dim=edge_dim)
    if backbone == "transformer":
        return TransformerConv(
            in_dim, out_dim, heads=ATTENTION_HEADS, concat=False,
            beta=True, dropout=0.0, edge_dim=edge_dim)

    raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")


class EdgeFeatureConvBlock(nn.Module):
    """Edge-aware convolution plus residual projection and optional BN."""

    def __init__(self, backbone: str, in_dim: int, out_dim: int,
                 edge_dim: int, use_bn: bool = True):
        super().__init__()
        self.conv = make_edge_feature_conv(
            backbone, in_dim, out_dim, edge_dim)
        self.skip = (nn.Linear(in_dim, out_dim)
                     if in_dim != out_dim else nn.Identity())
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        return F.relu(
            self.bn(self.conv(x, edge_index, edge_attr) + self.skip(x)))


class EdgeFeatureNodeGraphAE(ReconstructionMixin, nn.Module):
    """Full-node AE conditioned on physical edge features.

    Unlike :class:`EdgeGraphAE`, this model does *not* reconstruct edge
    attributes. It isolates the benefit of feeding ``(ln dR, ln kT, ln z)``
    into message passing from the benefit of an explicit edge-loss objective.
    """

    def __init__(self, in_dim: int, edge_dim: int = 3,
                 backbone: str = "edgeconv", hidden_dim: int = 64,
                 latent_dim: int = 2, use_bn: bool = True,
                 feature_cols: tuple[int, ...] | None = None):
        super().__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_bn = use_bn
        self.feature_cols = normalize_feature_cols(feature_cols, in_dim)
        self.input_bn = nn.BatchNorm1d(in_dim) if use_bn else nn.Identity()
        self.encoder = nn.ModuleList([
            EdgeFeatureConvBlock(backbone, in_dim, hidden_dim, edge_dim,
                                 use_bn=use_bn),
            EdgeFeatureConvBlock(backbone, hidden_dim, hidden_dim, edge_dim,
                                 use_bn=use_bn),
            EdgeFeatureConvBlock(backbone, hidden_dim, hidden_dim, edge_dim,
                                 use_bn=use_bn),
        ])
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = EdgeFeatureConvBlock(
            backbone, latent_dim, hidden_dim, edge_dim, use_bn=use_bn)
        self.head = nn.Linear(hidden_dim, in_dim)

    def forward(self, x, edge_index, edge_attr):
        h = self.input_bn(x)
        for block in self.encoder:
            h = block(h, edge_index, edge_attr)
        z = self.to_latent(h)
        recon = self.head(self.decoder(z, edge_index, edge_attr))
        return Reconstruction(node=recon, latent=z)

    def _reconstruct(self, batch):
        target = select_node_features(batch, self.in_dim, self.feature_cols)
        edge_attr = require_edge_features(batch, self.edge_dim)
        return self.forward(target, batch.edge_index, edge_attr), target, None

    def extra_repr(self) -> str:
        return (f"backbone={self.backbone!r}, in_dim={self.in_dim}, "
                f"edge_dim={self.edge_dim}, hidden_dim={self.hidden_dim}, "
                f"latent_dim={self.latent_dim}, use_bn={self.use_bn}, "
                f"feature_cols={self.feature_cols}")


class EdgeFeatureGraphAE(EdgeFeatureNodeGraphAE):
    """Edge-aware backbone with joint node and edge reconstruction.

    This keeps the same objective as :class:`EdgeGraphAE` while replacing its
    custom reference block with one of the standard GNN backbone families.
    It is therefore the fair architecture ablation for EdgeConv/GCN/SAGE/
    GATv2/GINE/TransformerConv.
    """

    def __init__(self, *args, edge_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_weight = edge_weight
        self.edge_predictor = EdgeAttrPredictor(
            self.latent_dim, self.edge_dim, hidden=self.hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        output = super().forward(x, edge_index, edge_attr)
        edge_recon = self.edge_predictor(output.latent, edge_index)
        return Reconstruction(node=output.node, latent=output.latent,
                              edge=edge_recon)

    def _reconstruct(self, batch):
        target = select_node_features(batch, self.in_dim, self.feature_cols)
        edge_attr = require_edge_features(batch, self.edge_dim)
        output = self.forward(target, batch.edge_index, edge_attr)
        return output, target, edge_attr

    def extra_repr(self) -> str:
        return super().extra_repr() + f", edge_weight={self.edge_weight}"
