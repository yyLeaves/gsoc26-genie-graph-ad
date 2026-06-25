import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv
from torch_geometric.utils import scatter

BACKBONES = ("edgeconv",)


def mlp(dims: list, act_last: bool = True) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or act_last:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def mse_per_graph(pred: torch.Tensor, target: torch.Tensor,
                  index: torch.Tensor, n_graphs: int = None) -> torch.Tensor:
    """Mean per-row MSE pooled to graph level by `index` → (G,).

    `index` is the node→graph or edge→graph map. The shared anomaly score.
    """
    per_row = ((pred - target) ** 2).mean(dim=-1)
    return scatter(per_row, index, dim=0, reduce="mean", dim_size=n_graphs)


def make_conv(backbone: str, in_dim: int, out_dim: int) -> nn.Module:
    """Build the message-passing conv for a backbone (in_dim → out_dim)."""
    if backbone == "edgeconv":   # DGCNN: MLP([x_i || x_j-x_i]), max-pool
        return EdgeConv(nn=mlp([2 * in_dim, out_dim, out_dim]), aggr="max")
    raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")


class ConvBlock(nn.Module):
    """GNN conv + linear residual skip + optional BatchNorm, then ReLU."""

    def __init__(self, backbone: str, in_dim: int, out_dim: int,
                 use_bn: bool = True):
        super().__init__()
        self.conv = make_conv(backbone, in_dim, out_dim)
        self.skip = (nn.Linear(in_dim, out_dim)
                     if in_dim != out_dim else nn.Identity())
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()

    def forward(self, x, edge_index):
        return F.relu(self.bn(self.conv(x, edge_index) + self.skip(x)))


def encoder_layers(backbone: str, in_dim: int, hidden_dim: int,
                   use_bn: bool) -> nn.ModuleList:
    """Shared 3-block backbone stack (in_dim → hidden_dim → hidden_dim)."""
    return nn.ModuleList([
        ConvBlock(backbone, in_dim, hidden_dim, use_bn=use_bn),
        ConvBlock(backbone, hidden_dim, hidden_dim, use_bn=use_bn),
        ConvBlock(backbone, hidden_dim, hidden_dim, use_bn=use_bn),
    ])


def run_encoder(input_bn: nn.Module, layers: nn.ModuleList, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
    """Run input BatchNorm then the backbone stack → per-node hidden features."""
    h = input_bn(x)
    for layer in layers:
        h = layer(h, edge_index)
    return h
