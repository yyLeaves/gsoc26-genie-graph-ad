import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from src.data.graph import edge_features, unique_k_edges
from src.models import EdgeGraphAE
from src.models.reconstruction import mean_loss, reconstruction_scores
from src.training import trainer
from src.objectives.perturbed_contrast import (
    clean_encoder_representation,
    off_diagonal_contrastive_loss,
    perturbed_encoder_representation,
    perturbed_graph_contrast,
)


def _batch():
    rng = np.random.default_rng(13)
    graphs = []
    for n_nodes in (8, 10, 12):
        pos = rng.normal(size=(n_nodes, 2)).astype("float32")
        pt = (rng.random(n_nodes) + 0.1).astype("float32")
        edges = unique_k_edges(pos, 3)
        graphs.append(Data(
            x=torch.tensor(np.log(pt)[:, None], dtype=torch.float),
            pt=torch.tensor(pt),
            pos=torch.tensor(pos),
            edge_index=edges,
            edge_attr=edge_features(pos, pt, edges, log=True),
        ))
    return Batch.from_data_list(graphs)


def test_eb2_contrast_and_projected_latent_cycle_have_separate_heads():
    from src.objectives.latent_cycle import latent_cycle_scores
    batch = _batch()
    model = EdgeGraphAE(in_dim=1, edge_dim=3, latent_dim=2,
                        contrastive_projection_dim=16, contrastive_source='eb2')
    output = model(batch.x, batch.edge_index, batch.edge_attr)
    cycle = latent_cycle_scores(model, output, batch, global_mode='projected').total.mean()
    contrast = perturbed_graph_contrast(model, output.latent, batch.x, batch,
                                       batch.edge_attr, scale=1., temperature=.2, source='eb2').loss
    cycle.backward(retain_graph=True)
    assert all(p.grad is None for p in model.contrastive_eb2_projection.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.contrastive_projection.parameters())
    model.zero_grad(set_to_none=True)
    contrast.backward()
    assert all(p.grad is None for p in model.contrastive_projection.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.contrastive_eb2_projection.parameters())


def test_zero_perturbation_matches_clean_encoder_and_preserves_graph():
    torch.manual_seed(7)
    batch = _batch()
    model = EdgeGraphAE(
        in_dim=1, edge_dim=3, contrastive_projection_dim=16)
    original_x = batch.x.clone()
    original_edges = batch.edge_index.clone()
    original_edge_attr = batch.edge_attr.clone()

    output = model(batch.x, batch.edge_index, batch.edge_attr)
    perturbed = perturbed_encoder_representation(
        model, batch.x, batch, batch.edge_attr, scale=0.0)

    assert torch.equal(perturbed, output.latent)
    assert torch.equal(batch.x, original_x)
    assert torch.equal(batch.edge_index, original_edges)
    assert torch.equal(batch.edge_attr, original_edge_attr)


@pytest.mark.parametrize("scale", [0.0, 1.0])
def test_scalar_encoder_parameters_have_finite_perturbations_and_gradients(scale):
    torch.manual_seed(42)
    batch = _batch()
    model = EdgeGraphAE(in_dim=1, edge_dim=3, latent_dim=1,
                        contrastive_projection_dim=16)
    output = model(batch.x, batch.edge_index, batch.edge_attr)
    perturbed = perturbed_encoder_representation(
        model, batch.x, batch, batch.edge_attr, scale=scale)
    assert torch.isfinite(perturbed).all()
    assert not perturbed.requires_grad
    if scale == 0:
        torch.testing.assert_close(perturbed, output.latent, rtol=0, atol=0)
    terms = perturbed_graph_contrast(
        model, output.latent, batch.x, batch, batch.edge_attr,
        scale=scale, temperature=0.2)
    assert torch.isfinite(terms.loss)
    terms.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_nonscalar_perturbation_keeps_the_upstream_std_and_rng_sequence():
    from torch.func import functional_call

    batch = _batch()
    model = EdgeGraphAE(in_dim=1, edge_dim=3, latent_dim=2)
    torch.manual_seed(43)
    expected = batch.x
    with torch.no_grad():
        for block in model.encoder_blocks:
            params = {}
            for name, param in block.named_parameters():
                assert param.numel() > 1
                params[name] = param + torch.randn_like(param) * param.std() * 0.3
            expected = functional_call(block, params,
                                       (expected, batch.edge_index, batch.edge_attr))
    torch.manual_seed(43)
    actual = perturbed_encoder_representation(
        model, batch.x, batch, batch.edge_attr, scale=0.3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_weight_perturbation_is_resampled_and_target_is_detached():
    torch.manual_seed(17)
    batch = _batch()
    model = EdgeGraphAE(
        in_dim=1, edge_dim=3, contrastive_projection_dim=16)
    output = model(batch.x, batch.edge_index, batch.edge_attr)
    first = perturbed_encoder_representation(
        model, batch.x, batch, batch.edge_attr, scale=1.0)
    second = perturbed_encoder_representation(
        model, batch.x, batch, batch.edge_attr, scale=1.0)

    assert not first.requires_grad
    assert not torch.equal(first, second)
    assert not torch.equal(first, output.latent)


def test_graph_contrast_is_finite_and_updates_clean_encoder_and_projection():
    torch.manual_seed(23)
    batch = _batch()
    model = EdgeGraphAE(
        in_dim=1, edge_dim=3, contrastive_projection_dim=16)
    output = model(batch.x, batch.edge_index, batch.edge_attr)
    terms = perturbed_graph_contrast(
        model, output.latent, batch.x, batch, batch.edge_attr,
        scale=1.0, temperature=0.2)

    assert terms.clean_graph.shape == (batch.num_graphs, model.latent_dim)
    assert terms.perturbed_graph.shape == terms.clean_graph.shape
    assert torch.isfinite(terms.loss)
    terms.loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.encoder_blocks.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.contrastive_projection.parameters()
    )


def test_eb2_source_uses_hidden_width_and_matches_clean_encoder_at_eta_zero():
    torch.manual_seed(27)
    batch = _batch()
    model = EdgeGraphAE(
        in_dim=1, edge_dim=3, hidden_dim=64, latent_dim=2,
        contrastive_projection_dim=16, contrastive_source="eb2")
    output = model(batch.x, batch.edge_index, batch.edge_attr)
    clean = clean_encoder_representation(
        model, output.latent, batch.x, batch, batch.edge_attr, "eb2")
    perturbed = perturbed_encoder_representation(
        model, batch.x, batch, batch.edge_attr, scale=0.0, source="eb2")

    assert clean.shape == (batch.num_nodes, model.hidden_dim)
    assert model.contrastive_eb2_projection[0].in_features == model.hidden_dim
    assert model.contrastive_projection[0].in_features == model.latent_dim
    assert torch.equal(clean, perturbed)


def test_eb2_contrast_updates_early_encoder_but_not_bottleneck_block():
    torch.manual_seed(28)
    batch = _batch()
    model = EdgeGraphAE(
        in_dim=1, edge_dim=3, hidden_dim=64, latent_dim=2,
        contrastive_projection_dim=16, contrastive_source="eb2")
    output = model(batch.x, batch.edge_index, batch.edge_attr)
    terms = perturbed_graph_contrast(
        model, output.latent, batch.x, batch, batch.edge_attr,
        scale=1.0, temperature=0.2, source="eb2")
    terms.loss.backward()

    assert any(p.grad is not None for p in model.encoder_blocks[0].parameters())
    assert any(p.grad is not None for p in model.encoder_blocks[1].parameters())
    assert all(p.grad is None for p in model.encoder_blocks[2].parameters())


def test_off_diagonal_loss_prefers_matched_pairs():
    clean = torch.eye(3)
    matched = clean.clone()
    mismatched = clean.roll(1, dims=0)
    assert off_diagonal_contrastive_loss(
        clean, matched, 0.2) < off_diagonal_contrastive_loss(
            clean, mismatched, 0.2)


def test_off_diagonal_loss_matches_official_gladc_formula():
    torch.manual_seed(29)
    anchors = torch.randn(5, 7)
    candidates = torch.randn(5, 7)
    temperature = 0.2
    similarity = torch.einsum("ik,jk->ij", anchors, candidates)
    similarity = similarity / torch.einsum(
        "i,j->ij", anchors.norm(dim=1), candidates.norm(dim=1))
    similarity = torch.exp(similarity / temperature)
    positive = similarity.diagonal()
    expected = -torch.log(
        positive / (similarity.sum(dim=1) - positive)).mean()

    actual = off_diagonal_contrastive_loss(
        anchors, candidates, temperature)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_zero_contrast_weight_matches_one_baseline_optimizer_step(monkeypatch, training_args):
    torch.manual_seed(31)
    batch = _batch()
    reference = EdgeGraphAE(in_dim=1, edge_dim=3)
    candidate = copy.deepcopy(reference)
    reference_opt = torch.optim.AdamW(
        reference.parameters(), lr=3e-3, weight_decay=0.01)
    candidate_opt = torch.optim.AdamW(
        candidate.parameters(), lr=3e-3, weight_decay=0.01)

    reference_opt.zero_grad()
    output, node_target, edge_target = reference._reconstruct(batch)
    reference_loss = mean_loss(reconstruction_scores(
        output, node_target, batch, edge_target=edge_target,
        edge_weight=reference.edge_weight)).total
    reference_loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    reference_opt.step()

    monkeypatch.setattr(
        trainer, "shard_iter", lambda *args, **kwargs: iter([batch]))
    monkeypatch.setattr(
        trainer, "prefetch", lambda iterable, depth=2: iterable)

    def unexpected_contrast_call(*args, **kwargs):
        raise AssertionError("alpha=0 must not execute the contrast branch")

    monkeypatch.setattr(
        trainer, "perturbed_graph_contrast", unexpected_contrast_call)

    class NoOpScheduler:
        def step(self):
            pass

    args = training_args
    splits = SimpleNamespace(train_idx=np.arange(batch.num_graphs))
    stats = trainer.train_one_epoch(
        candidate, object(), splits, candidate_opt, NoOpScheduler(), False,
        None, args, torch.device("cpu"), np.random.default_rng(3))

    assert stats.contrast == 0.0
    assert stats.total == reference_loss.item()
    assert all(
        torch.equal(candidate.state_dict()[name], value)
        for name, value in reference.state_dict().items()
    )
