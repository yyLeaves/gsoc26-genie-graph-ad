import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (EdgeConv, GATv2Conv, GCNConv, GINConv,
                                SAGEConv, TransformerConv)

BACKBONES = ("edgeconv", "gcn", "sage", "gatv2", "gin", "transformer")
ATTENTION_HEADS = 4


def mlp(dims: list[int], act_last: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or act_last:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def make_conv(backbone: str, in_dim: int, out_dim: int) -> nn.Module:
    """Build the message-passing conv for a backbone (in_dim → out_dim)."""
    if backbone == "edgeconv":
        # DGCNN: MLP([x_i || x_j - x_i]), max-pool
        return EdgeConv(nn=mlp([2 * in_dim, out_dim, out_dim]), aggr="max")
    if backbone == "gcn":
        return GCNConv(in_dim, out_dim)
    if backbone == "sage":
        return SAGEConv(in_dim, out_dim, aggr="mean")
    if backbone == "gatv2":
        return GATv2Conv(in_dim, out_dim, heads=ATTENTION_HEADS,
                         concat=False, dropout=0.0)
    if backbone == "gin":
        return GINConv(mlp([in_dim, out_dim, out_dim]), train_eps=True)
    if backbone == "transformer":
        return TransformerConv(in_dim, out_dim, heads=ATTENTION_HEADS,
                               concat=False, beta=True, dropout=0.0)
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
