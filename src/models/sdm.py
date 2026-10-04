"""SDM-NAT from the ICML 2025 paper, not an author-code port.

See SDM.md for the explicit reduction, masking and architecture choices.
Only node features and adjacency are used; physical edge features are ignored.
"""

import torch
from torch import nn
from torch_geometric.nn import DenseGINConv
from torch_geometric.utils import to_dense_adj, to_dense_batch


def _pair_mask(mask):
    """Real, distinct node pairs; GIN handles its own self contribution."""
    diagonal = torch.eye(mask.shape[1], device=mask.device, dtype=torch.bool)
    return mask[:, :, None] & mask[:, None, :] & ~diagonal


def _dense_inputs(batch):
    x, mask = to_dense_batch(batch.x, batch.batch)
    adjacency = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=x.shape[1])
    adjacency = adjacency.gt(0).to(x) * _pair_mask(mask)
    return x, adjacency, mask


class _GIN(nn.Module):
    """Three 16-channel GIN layers, concatenated at each real node (Eq. 8)."""

    def __init__(self, layer_norm=True):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseGINConv(nn.Sequential(
                nn.Linear(size, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.LayerNorm(16) if layer_norm else nn.Identity(),
                nn.ReLU(),
            ))
            for size in (1, 16, 16)
        ])

    def forward(self, x, adjacency, mask):
        x = x * mask[..., None]
        adjacency = adjacency * _pair_mask(mask)
        layers = []
        for layer in self.layers:
            x = layer(x, adjacency, mask)
            layers.append(x)
        return torch.cat(layers, dim=-1)


class _VariationalEncoder(nn.Module):
    """Node-wise Gaussian latents; separate copies generate attributes and adjacency."""

    def __init__(self, layer_norm=True):
        super().__init__()
        self.backbone = _GIN(layer_norm)
        self.mean = DenseGINConv(nn.Linear(48, 10))
        self.log_std = DenseGINConv(nn.Linear(48, 10))
        # Start at N(0, I); the paper does not specify posterior initialization.
        for head in (self.mean, self.log_std):
            nn.init.zeros_(head.nn.weight)
            nn.init.zeros_(head.nn.bias)

    def forward(self, x, adjacency, mask):
        hidden = self.backbone(x, adjacency, mask)
        mean = self.mean(hidden, adjacency, mask)
        # Same upper log-std bound as PyG VGAE; KL uses this same distribution.
        log_std = self.log_std(hidden, adjacency, mask).clamp(max=10)
        latent = (mean + torch.randn_like(mean) * log_std.exp()) * mask[..., None]
        return latent, mean, log_std


class _Generator(nn.Module):
    def __init__(self, layer_norm=True):
        super().__init__()
        self.node_encoder = _VariationalEncoder(layer_norm)
        self.adjacency_encoder = _VariationalEncoder(layer_norm)
        self.node_decoder = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, adjacency, mask):
        node_z, node_mean, node_log_std = self.node_encoder(x, adjacency, mask)
        edge_z, edge_mean, edge_log_std = self.adjacency_encoder(x, adjacency, mask)
        generated_x = self.node_decoder(node_z) * mask[..., None]
        adjacency_logits = edge_z @ edge_z.transpose(1, 2)
        # Keep continuous probabilities: classifier gradients reach edge_z.
        generated_adjacency = adjacency_logits.sigmoid() * _pair_mask(mask)
        return dict(
            node=generated_x,
            adjacency=generated_adjacency,
            adjacency_logits=adjacency_logits,
            posteriors=((node_mean, node_log_std), (edge_mean, edge_log_std)),
        )


class _Classifier(nn.Module):
    def __init__(self, layer_norm=True):
        super().__init__()
        self.backbone = _GIN(layer_norm)
        self.head = nn.Sequential(nn.Linear(48, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, adjacency, mask):
        nodes = self.backbone(x, adjacency, mask)
        graph = nodes.sum(dim=1)  # Global sum pooling over real nodes.
        return self.head(graph).squeeze(-1)


class SDMNAT(nn.Module):
    """Jointly trained generator/classifier; classifier-only inference, normal logit."""

    def __init__(self, discrepancy_weight=1., kl_weight=1., layer_norm=True):
        super().__init__()
        self.discrepancy_weight = discrepancy_weight
        self.kl_weight = kl_weight
        self.generator = _Generator(layer_norm)
        self.classifier = _Classifier(layer_norm)

    def forward(self, batch):
        """Generate graphs and classify both views of a PyG jet batch.

        B is the number of jets; N is the largest node count in this batch.
        Output fields:
          inputs: (x, adjacency, mask), shaped (B,N,1), (B,N,N), (B,N).
          node: generated features (B,N,1).
          adjacency / adjacency_logits: generated probabilities / logits (B,N,N).
          pair_mask: valid non-self node pairs (B,N,N), excluding padding.
          posteriors: ((node_mean, node_log_std), (edge_mean, edge_log_std));
                      each tensor is (B,N,10).
          real_logits / generated_logits: normal-class logits (B,) for each view.
        """
        x, adjacency, mask = _dense_inputs(batch)
        generated = self.generator(x, adjacency, mask)
        return dict(
            generated,
            inputs=(x, adjacency, mask),
            pair_mask=_pair_mask(mask),
            real_logits=self.classifier(x, adjacency, mask),
            generated_logits=self.classifier(generated['node'], generated['adjacency'], mask),
        )

    def classify(self, batch):
        """Return normal-class logits (B,); no generator, sampling or reconstruction."""
        return self.classifier(*_dense_inputs(batch))
