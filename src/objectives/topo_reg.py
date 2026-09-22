"""Adjacent-subjet regularization on input or latent graph edges."""
import math
import torch
import torch.nn.functional as F
from src.data.graph import unique_k_edges

_SPECS = {
    "unique_sum": ("unique_sum",),
    "hidden_unique_sum": ("hidden_unique_sum",),
    "graph_sum": ("graph_sum",),
    "hidden_graph_sum": ("hidden_graph_sum",),
    "graph_eb3_attraction": ("graph_sum",),
    "graph_eb2_attraction": ("hidden_graph_sum",),
    "graph_eb2_eb3_attraction": ("hidden_graph_sum", "graph_sum"),
}


def parse_topo_reg_spec(name):
    if name in (None, "", "none"):
        return []
    try:
        return list(_SPECS[name])
    except KeyError as error:
        raise ValueError(f"unknown topology regularizer: {name}") from error


def parse_lambda_list(value, count):
    if count == 0:
        return []
    weights = [float(part) for part in str(value).split(",")]
    if len(weights) == 1:
        weights *= count
    if len(weights) != count or any(
            weight < 0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("lambda weights must be finite and >= 0")
    return weights


def _unique_pairs(edge_index):
    source, target = edge_index
    lower = torch.minimum(source, target)
    upper = torch.maximum(source, target)
    pairs = torch.stack((lower, upper), dim=1)
    pairs = pairs[lower != upper]
    return torch.unique(pairs, dim=0).T.contiguous()


def encoder_hidden(model, node_target, edge_index, edge_target):
    hidden = node_target
    for block in model.encoder_blocks[:-1]:
        hidden = block(hidden, edge_index, edge_target)
    return hidden


def _latent_pairs(embedding, batch_index, num_graphs, k):
    pairs = []
    for graph in range(num_graphs):
        node_ids = torch.where(batch_index == graph)[0]
        if node_ids.numel() < 2:
            continue
        local = _unique_pairs(unique_k_edges(
            embedding[node_ids].detach().cpu().numpy(), k)).to(node_ids.device)
        pairs.append(node_ids[local])
    return (torch.cat(pairs, dim=1) if pairs else
            torch.empty((2, 0), dtype=torch.long, device=embedding.device))


def _edge_loss(embedding, batch_index, num_graphs, pairs):
    if pairs.numel() == 0:
        return embedding.new_zeros(())
    lengths = torch.linalg.vector_norm(
        embedding[pairs[0]] - embedding[pairs[1]], dim=1)
    edge_graph = batch_index[pairs[0]]
    length_sums = lengths.new_zeros(num_graphs).scatter_add(
        0, edge_graph, lengths)
    edge_counts = torch.bincount(edge_graph, minlength=num_graphs)
    node_counts = torch.bincount(batch_index, minlength=num_graphs).to(lengths)
    present = edge_counts > 0
    return (length_sums[present] / node_counts[present]).mean()


def compute_topo_regs(names, latent, batch, *, lambdas, unique_k=6,
                      normalize=False, model=None, node_target=None,
                      edge_target=None):
    hidden = (encoder_hidden(model, node_target, batch.edge_index, edge_target)
              if any(name.startswith("hidden_") for name in names) else None)
    graph_pairs = (_unique_pairs(batch.edge_index)
                   if any("unique" not in name for name in names) else None)
    terms = []
    for name in names:
        embedding = hidden if name.startswith("hidden_") else latent
        if normalize:
            embedding = F.normalize(embedding, p=2, dim=1, eps=1e-8)
        pairs = (_latent_pairs(embedding, batch.batch, batch.num_graphs, unique_k)
                 if "unique" in name else graph_pairs)
        terms.append(_edge_loss(
            embedding, batch.batch, batch.num_graphs, pairs))
    weighted = torch.stack([weight * term for weight, term in zip(lambdas, terms)])
    return weighted.sum(), torch.stack(terms).sum()
