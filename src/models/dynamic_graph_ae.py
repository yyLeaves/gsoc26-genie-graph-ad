import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from .blocks import mlp
from .inputs import (normalize_feature_cols, require_edge_features,
                     select_node_features)
from .reconstruction import (EdgeAttrPredictor, Reconstruction,
                             ReconstructionMixin)


class DenseDynamicEdgeBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, k: int = 16,
                 hidden_dim: int = 64, use_bn: bool = True):
        super().__init__()
        self.k = k
        self.out_dim = out_dim
        self.msg = mlp([2 * in_dim, hidden_dim, out_dim], act_last=False)
        self.skip = (nn.Linear(in_dim, out_dim)
                     if in_dim != out_dim else nn.Identity())
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        dense, mask = to_dense_batch(x, batch)  # (B, Nmax, F), (B, Nmax)
        B, N, Fdim = dense.shape
        if N <= 1:
            pooled = x.new_zeros((x.size(0), self.out_dim))
        else:
            k = min(self.k, N - 1)
            dist = torch.cdist(dense, dense)
            eye = torch.eye(N, dtype=torch.bool, device=x.device).view(1, N, N)
            valid = mask.unsqueeze(1) & mask.unsqueeze(2) & ~eye
            dist = dist.masked_fill(~valid, float("inf"))
            nn_idx = dist.topk(k, dim=-1, largest=False).indices  # (B, N, k)

            gather_idx = nn_idx.unsqueeze(-1).expand(B, N, k, Fdim)
            dense_expanded = dense.unsqueeze(1).expand(B, N, N, Fdim)
            neigh = torch.gather(dense_expanded, 2, gather_idx)
            center = dense.unsqueeze(2).expand(B, N, k, Fdim)
            msg_in = torch.cat([center, neigh - center], dim=-1)
            msg = self.msg(msg_in.reshape(B * N * k, 2 * Fdim))
            msg = msg.view(B, N, k, -1)

            neigh_valid = torch.gather(
                mask.unsqueeze(1).expand(B, N, N), 2, nn_idx)
            msg = msg.masked_fill(~neigh_valid.unsqueeze(-1), -torch.inf)
            pooled = msg.max(dim=2).values
            pooled = torch.where(torch.isfinite(pooled), pooled,
                                 torch.zeros_like(pooled))
            pooled = pooled[mask]

        out = pooled + self.skip(x)
        return F.relu(self.bn(out))


class DynamicGraphAE(ReconstructionMixin, nn.Module):
    """Node-feature autoencoder with dynamic kNN graph construction.

    This is the minimal ParticleNet/DGCNN-style control: each block recomputes
    neighbors in the current embedding space, so graph construction is learned
    implicitly through the node embeddings rather than fixed offline.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, latent_dim: int = 2,
                 k: int = 16, use_bn: bool = True,
                 feature_cols: tuple[int, ...] | None = None):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.k = k
        self.use_bn = use_bn
        self.feature_cols = normalize_feature_cols(feature_cols, in_dim)
        self.input_bn = nn.BatchNorm1d(in_dim) if use_bn else nn.Identity()
        self.encoder = nn.ModuleList([
            DenseDynamicEdgeBlock(in_dim, hidden_dim, k=k,
                                  hidden_dim=hidden_dim, use_bn=use_bn),
            DenseDynamicEdgeBlock(hidden_dim, hidden_dim, k=k,
                                  hidden_dim=hidden_dim, use_bn=use_bn),
            DenseDynamicEdgeBlock(hidden_dim, hidden_dim, k=k,
                                  hidden_dim=hidden_dim, use_bn=use_bn),
        ])
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = DenseDynamicEdgeBlock(latent_dim, hidden_dim, k=k,
                                             hidden_dim=hidden_dim,
                                             use_bn=use_bn)
        self.head = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> Reconstruction:
        h = self.input_bn(x)
        for block in self.encoder:
            h = block(h, batch)
        z = self.to_latent(h)
        recon = self.head(self.decoder(z, batch))
        return Reconstruction(node=recon, latent=z)

    def _reconstruct(self, batch):
        target = select_node_features(batch, self.in_dim, self.feature_cols)
        return self.forward(target, batch.batch), target, None

    def extra_repr(self) -> str:
        return (f"in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, "
                f"latent_dim={self.latent_dim}, k={self.k}, "
                f"use_bn={self.use_bn}, feature_cols={self.feature_cols}")


class DynamicEdgeGraphAE(DynamicGraphAE):
    """Dynamic EdgeConv encoder with joint node and physical-edge loss.

    Dynamic kNN controls message passing; the fixed offline edges are used only
    as reconstruction targets for ``(ln dR, ln kT, ln z)``. This isolates the
    neighborhood mechanism while matching the successful joint AD objective.
    """

    def __init__(self, *args, edge_dim: int = 3, edge_weight: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_dim = edge_dim
        self.edge_weight = edge_weight
        self.edge_predictor = EdgeAttrPredictor(
            self.latent_dim, edge_dim, hidden=self.hidden_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor,
                edge_index: torch.Tensor) -> Reconstruction:
        output = super().forward(x, batch)
        edge_recon = self.edge_predictor(output.latent, edge_index)
        return Reconstruction(node=output.node, latent=output.latent,
                              edge=edge_recon)

    def _reconstruct(self, batch):
        target = select_node_features(batch, self.in_dim, self.feature_cols)
        edge_attr = require_edge_features(batch, self.edge_dim)
        output = self.forward(target, batch.batch, batch.edge_index)
        return output, target, edge_attr

    def extra_repr(self) -> str:
        return (super().extra_repr() + f", edge_dim={self.edge_dim}, "
                f"edge_weight={self.edge_weight}")
