import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (BACKBONES, ConvBlock, encoder_layers, mse_per_graph,
                     run_encoder)


class GraphEncoder(nn.Module):
    """Input BatchNorm → 3 backbone layers → linear projection → per-node z."""

    def __init__(self, backbone: str, in_dim: int, hidden_dim: int,
                 latent_dim: int, use_bn: bool = True):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_dim) if use_bn else nn.Identity()
        self.layers = encoder_layers(backbone, in_dim, hidden_dim, use_bn)
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        h = run_encoder(self.input_bn, self.layers, x, edge_index)
        return self.proj(h)  # (N, latent_dim)


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
        return self.head(self.block(z, edge_index))  # (N, out_dim)


class NodeGraphAE(nn.Module):
    """Graph Autoencoder for jet anomaly detection.

    Args:
        latent_dim: per-node bottleneck (default 2 — usually keep below in_dim
                    so the AE cannot learn the identity map too easily)
        use_bn:     BatchNorm in every block + input. Araz et al. (2506.19920)
                    report BN costs ~20% background rejection for AD.
    """

    COMPONENT_NAMES = ("recon", "kl")   # kl is always 0

    def __init__(
        self,
        in_dim: int,
        backbone: str = "edgeconv",
        hidden_dim: int = 64,
        latent_dim: int = 2,
        use_bn: bool = True,
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(
                f"backbone must be one of {BACKBONES}, got {backbone!r}")
        self.in_dim = in_dim
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_bn = use_bn
        self.encoder = GraphEncoder(backbone, in_dim, hidden_dim, latent_dim,
                                    use_bn=use_bn)
        self.decoder = GraphDecoder(backbone, latent_dim, hidden_dim, in_dim,
                                    use_bn=use_bn)

    def _node_input(self, batch) -> torch.Tensor:
        if batch.x.size(-1) < self.in_dim:
            raise ValueError(
                f"batch.x has {batch.x.size(-1)} features, but "
                f"NodeGraphAE(in_dim={self.in_dim}) requires at least "
                f"{self.in_dim}")
        return batch.x[:, :self.in_dim]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Returns (recon, z): reconstructed node features and latent vectors."""
        z = self.encoder(x, edge_index)
        recon = self.decoder(z, edge_index)
        return recon, z

    def loss(self, batch):
        """Training loss → (total, recon, kl); kl is 0 (plain AE has no KL).

        Reads the first `in_dim` node columns so a full-feature inference batch
        still gets the trained-on input.
        """
        x = self._node_input(batch)
        recon, _ = self.forward(x, batch.edge_index)
        recon_l = F.mse_loss(recon, x)
        return recon_l, recon_l, recon_l.new_zeros(())

    @torch.no_grad()
    def anomaly_score(self, batch) -> torch.Tensor:
        """Per-graph anomaly score = mean per-node reconstruction MSE → (B,)."""
        x = self._node_input(batch)
        recon, _ = self.forward(x, batch.edge_index)
        return mse_per_graph(recon, x, batch.batch)

    def extra_repr(self) -> str:
        return (f"backbone={self.backbone!r}, in_dim={self.in_dim}, "
                f"hidden_dim={self.hidden_dim}, latent_dim={self.latent_dim}, "
                f"use_bn={self.use_bn}")
