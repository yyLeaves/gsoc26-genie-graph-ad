import torch
import torch.nn as nn

from .blocks import ConvBlock
from .inputs import normalize_feature_cols, select_node_features
from .reconstruction import Reconstruction, ReconstructionMixin


class GraphEncoder(nn.Module):
    """Input BatchNorm → 3 backbone layers → linear projection → per-node z."""

    def __init__(self, backbone: str, in_dim: int, hidden_dim: int,
                 latent_dim: int, use_bn: bool = True):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_dim) if use_bn else nn.Identity()
        self.layers = nn.ModuleList([
            ConvBlock(backbone, in_dim, hidden_dim, use_bn=use_bn),
            ConvBlock(backbone, hidden_dim, hidden_dim, use_bn=use_bn),
            ConvBlock(backbone, hidden_dim, hidden_dim, use_bn=use_bn),
        ])
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_bn(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.proj(h)


class GraphDecoder(nn.Module):
    """Graph-structured decoder: 1 backbone layer on the jet graph → x̂.

    Using the graph (vs a plain MLP) forces each node to be reconstructed from
    its neighbourhood's latents, so graph structure matters to the task.
    """

    def __init__(self, backbone: str, latent_dim: int, hidden_dim: int,
                 out_dim: int, use_bn: bool = True):
        super().__init__()
        self.block = ConvBlock(backbone, latent_dim, hidden_dim, use_bn=use_bn)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, z: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(z, edge_index))


class NodeGraphAE(ReconstructionMixin, nn.Module):
    """Graph Autoencoder for jet anomaly detection.

    Args:
        latent_dim: per-node bottleneck (default 2 — usually keep below in_dim
                    so the AE cannot learn the identity map too easily)
        use_bn:     BatchNorm in every block + input. Araz et al. (2506.19920)
                    report BN costs ~20% background rejection for AD.
    """

    def __init__(
        self,
        in_dim: int,
        backbone: str = "edgeconv",
        hidden_dim: int = 64,
        latent_dim: int = 2,
        use_bn: bool = True,
        feature_cols: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_bn = use_bn
        self.feature_cols = normalize_feature_cols(feature_cols, in_dim)
        self.encoder = GraphEncoder(backbone, in_dim, hidden_dim, latent_dim,
                                    use_bn=use_bn)
        self.decoder = GraphDecoder(backbone, latent_dim, hidden_dim, in_dim,
                                    use_bn=use_bn)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> Reconstruction:
        z = self.encoder(x, edge_index)
        recon = self.decoder(z, edge_index)
        return Reconstruction(node=recon, latent=z)

    def _reconstruct(self, batch):
        target = select_node_features(batch, self.in_dim, self.feature_cols)
        return self.forward(target, batch.edge_index), target, None

    def extra_repr(self) -> str:
        return (f"backbone={self.backbone!r}, in_dim={self.in_dim}, "
                f"hidden_dim={self.hidden_dim}, latent_dim={self.latent_dim}, "
                f"use_bn={self.use_bn}, feature_cols={self.feature_cols}")
