"""Controlled encoder-backbone variants of the reference EdgeGraphAE."""

from itertools import pairwise

import torch.nn as nn
import torch.nn.functional as F

from .edge_feature_node_graph_ae import make_edge_feature_conv
from .edge_graph_ae import EdgeGraphAE


class ReferenceBackboneEncoderBlock(nn.Module):
    """Native edge-aware message passing with the reference residual layout."""

    def __init__(self, backbone: str, node_in: int, node_out: int,
                 edge_dim: int):
        super().__init__()
        self.conv = make_edge_feature_conv(
            backbone, node_in, node_out, edge_dim)
        self.residual = (nn.Linear(node_in, node_out)
                         if node_in != node_out else nn.Identity())

    def forward(self, x, edge_index, edge_attr):
        return F.relu(self.conv(x, edge_index, edge_attr)) + self.residual(x)


class ReferenceBackboneEdgeGraphAE(EdgeGraphAE):
    """Reference EdgeGraphAE with only the encoder backbone replaced.

    The decoder, latent edge predictor, reconstruction loss, dimensions, and
    mean-aggregation reference layout are inherited unchanged from
    :class:`EdgeGraphAE`.
    """

    def __init__(self, in_dim: int, edge_dim: int = 3,
                 backbone: str = "gcn", hidden_dim: int = 64,
                 latent_dim: int = 2, edge_weight: float = 1.0,
                 feature_cols: tuple[int, ...] | None = None):
        super().__init__(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            edge_weight=edge_weight,
            aggr="mean",
            dropout=0.0,
            feature_cols=feature_cols,
        )
        self.backbone = backbone
        encoder_dims = (in_dim, hidden_dim, hidden_dim, latent_dim)
        self.encoder_blocks = nn.ModuleList(
            ReferenceBackboneEncoderBlock(backbone, input_dim, output_dim, edge_dim)
            for input_dim, output_dim in pairwise(encoder_dims))

    def extra_repr(self) -> str:
        return f"backbone={self.backbone!r}, " + super().extra_repr()
