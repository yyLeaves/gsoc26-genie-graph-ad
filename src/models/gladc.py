"""GLADC objectives on our unchanged EdgeConv autoencoder."""

import torch
from torch import nn
from torch.func import functional_call
from torch_geometric.nn import global_max_pool

from .edgeconv import EdgeConvAE


class GLADC(EdgeConvAE):
    """Add a shared 2->128->128 graph projection for cycle and contrast losses."""

    def __init__(self):
        super().__init__()
        self.contrastive_projection = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128))

    @torch.no_grad()
    def perturbed_graph(self, batch):
        """Add Gaussian noise to encoder weights; leave projection weights unchanged."""
        hidden = batch.x
        for block in self.encoder_blocks:
            parameters = {}
            for name, parameter in block.named_parameters():
                value = parameter.detach()
                std = value.std() if value.numel() > 1 else value.new_zeros(())
                parameters[name] = value + torch.randn_like(value) * std
            hidden = functional_call(block, parameters, (hidden, batch.edge_index, batch.edge_attr))
        return self.contrastive_projection(global_max_pool(hidden, batch.batch))
