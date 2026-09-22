"""GLADC-style graph contrast between clean and weight-perturbed encoders."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch_geometric.nn import global_max_pool


@dataclass(frozen=True, slots=True)
class PerturbedContrastTerms:
    """Contrastive objective and the graph representations it compares."""

    loss: torch.Tensor
    clean_graph: torch.Tensor
    perturbed_graph: torch.Tensor


def _perturbed_block(block, h, edge_index, edge_attr, scale: float):
    """Apply one encoder block with detached Gaussian-perturbed weights."""
    params = {}
    for name, param in block.named_parameters():
        detached = param.detach()
        # Match the official GLADC implementation's ``param.std()`` exactly.
        std = detached.std()
        noise = torch.randn_like(detached) * std * scale
        params[name] = detached + noise
    return functional_call(block, params, (h, edge_index, edge_attr))


def _encoder_blocks(model, source: str):
    if source == "latent":
        return model.encoder_blocks
    if source == "eb2":
        return model.encoder_blocks[:-1]
    raise ValueError("contrastive source must be 'latent' or 'eb2'")


def clean_encoder_representation(model, clean_latent, node_target, batch,
                                 edge_target, source: str) -> torch.Tensor:
    """Return the trainable node representation selected for contrast."""
    if source == "latent":
        return clean_latent
    h = node_target
    for block in _encoder_blocks(model, source):
        h = block(h, batch.edge_index, edge_target)
    return h


def perturbed_encoder_representation(model, node_target, batch, edge_target,
                                     scale: float,
                                     source: str = "latent") -> torch.Tensor:
    """Encode the graph to the selected layer with perturbed block weights."""
    h = node_target.detach()
    for block in _encoder_blocks(model, source):
        h = _perturbed_block(
            block, h, batch.edge_index, edge_target.detach(), scale)
    return h.detach()


def off_diagonal_contrastive_loss(anchors: torch.Tensor,
                                  candidates: torch.Tensor,
                                  temperature: float, *, center: bool = False) -> torch.Tensor:
    """GLADC objective: matched graph is positive; other graphs are negatives."""
    if anchors.ndim != 2 or candidates.shape != anchors.shape:
        raise ValueError("anchors and candidates must share shape (B, D)")
    if anchors.shape[0] < 2:
        raise ValueError("contrastive loss requires at least two graphs")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if center:
        anchors = anchors - anchors.mean(dim=0, keepdim=True)
        candidates = candidates - candidates.mean(dim=0, keepdim=True)
    logits = F.normalize(anchors, dim=-1) @ F.normalize(candidates, dim=-1).T
    logits = logits / temperature
    positive = logits.diagonal()
    mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    negative = torch.logsumexp(logits.masked_fill(mask, -torch.inf), dim=1)
    return (negative - positive).mean()


def perturbed_graph_contrast(model, clean_latent, node_target, batch,
                             edge_target, *, scale: float,
                             temperature: float,
                             source: str = "latent", center: bool = False) -> PerturbedContrastTerms:
    """Compute the paper-aligned clean/perturbed graph contrastive loss."""
    projection = getattr(model, "contrastive_eb2_projection" if source == "eb2"
                         else "contrastive_projection", None)
    if projection is None:
        raise ValueError("model has no contrastive projection head")

    clean_nodes = clean_encoder_representation(
        model, clean_latent, node_target, batch, edge_target, source)
    clean_graph = global_max_pool(clean_nodes, batch.batch)
    perturbed_nodes = perturbed_encoder_representation(
        model, node_target, batch, edge_target, scale, source)
    perturbed_graph = global_max_pool(perturbed_nodes, batch.batch)

    clean_projected = projection(clean_graph)
    with torch.no_grad():
        perturbed_projected = projection(perturbed_graph)
    # Official GLADC uses the perturbed branch as row-wise anchors and the
    # clean branch as candidates.  The target branch is detached, while the
    # clean candidate embeddings carry gradients to the trainable encoder.
    loss = off_diagonal_contrastive_loss(
        perturbed_projected, clean_projected, temperature, center=center)
    return PerturbedContrastTerms(
        loss=loss,
        clean_graph=clean_graph,
        perturbed_graph=perturbed_graph,
    )
