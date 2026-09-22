"""Graph-local random masks with at least one eligible selection per graph."""

import torch
from torch_geometric.utils import scatter


def sample_graph_mask(graph, num_graphs, fraction, *, eligible=None, generator=None):
    """Sample eligible rows and retain the smallest draw in each nonempty group."""
    if fraction == 0.0:
        return torch.zeros_like(graph, dtype=torch.bool)
    random = torch.rand(graph.numel(), device=graph.device, generator=generator)
    if eligible is None:
        eligible = torch.ones_like(graph, dtype=torch.bool)
    candidates = random.masked_fill(~eligible, float("inf"))
    first = scatter(candidates, graph, dim=0, dim_size=num_graphs, reduce="min")
    return eligible & ((random < fraction) | (random == first[graph]))
