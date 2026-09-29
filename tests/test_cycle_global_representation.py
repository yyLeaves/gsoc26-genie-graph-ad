import copy
import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_max_pool

from src.models import EdgeGraphAE
from src.objectives.latent_cycle import latent_cycle_scores
from src.objectives.perturbed_contrast import off_diagonal_contrastive_loss
from src.training.options import validate_objectives


def example():
    torch.manual_seed(42)
    graphs = []
    for n in (4, 5, 6):
        pairs = torch.combinations(torch.arange(n), r=2).T
        attr = torch.randn(pairs.shape[1], 3)
        graphs.append(Data(x=torch.randn(n, 1),
                           edge_index=torch.cat([pairs, pairs.flip(0)], dim=1),
                           edge_attr=torch.cat([attr, attr])))
    model = EdgeGraphAE(in_dim=1, edge_dim=3, hidden_dim=8,
                        contrastive_projection_dim=8)
    batch = Batch.from_data_list(graphs)
    return model, batch, model(batch.x, batch.edge_index, batch.edge_attr)


@pytest.mark.parametrize("mode", ["pooled", "projected", "unit_projected"])
def test_cycle_global_matches_explicit_definition(mode):
    model, batch, output = example()
    terms = latent_cycle_scores(model, output, batch, global_mode=mode)
    a = global_max_pool(output.latent, batch.batch)
    b = global_max_pool(terms.reencoded, batch.batch)
    if mode != "pooled":
        a, b = model.contrastive_projection(a), model.contrastive_projection(b)
    if mode == "unit_projected":
        a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    torch.testing.assert_close(terms.global_, (a-b).square().mean(dim=-1))
    original = latent_cycle_scores(model, output, batch)
    torch.testing.assert_close(terms.local, original.local, rtol=0, atol=0)
    if mode == "pooled":
        torch.testing.assert_close(terms.total, original.total, rtol=0, atol=0)
    else:
        terms.total.mean().backward()
        assert model.contrastive_projection[-1].weight.grad is not None
        assert model.edge_predictor.fc1.weight.grad is not None


def test_projection_rescaling_reduces_raw_cycle_but_not_cosine_contrast():
    model, batch, output = example()
    pooled = global_max_pool(output.latent, batch.batch).detach()
    other = pooled + torch.randn_like(pooled) * 0.1
    before = latent_cycle_scores(model, output, batch, global_mode="projected")
    unit_before = latent_cycle_scores(model, output, batch, global_mode="unit_projected")
    nce_before = off_diagonal_contrastive_loss(model.contrastive_projection(pooled),
        model.contrastive_projection(other), temperature=0.2)
    scaled = copy.deepcopy(model)
    with torch.no_grad():
        scaled.contrastive_projection[-1].weight.mul_(0.1)
        scaled.contrastive_projection[-1].bias.mul_(0.1)
    after = latent_cycle_scores(scaled, output, batch, global_mode="projected")
    unit_after = latent_cycle_scores(scaled, output, batch, global_mode="unit_projected")
    nce_after = off_diagonal_contrastive_loss(scaled.contrastive_projection(pooled),
        scaled.contrastive_projection(other), temperature=0.2)
    torch.testing.assert_close(after.global_, 0.01*before.global_, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(unit_after.global_, unit_before.global_, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(nce_after, nce_before, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(model.anomaly_score(batch), scaled.anomaly_score(batch),
                               rtol=0, atol=0)


@pytest.mark.parametrize("source", ["latent", "eb2"])
@pytest.mark.parametrize("contrast_weight", [0, 1])
def test_projected_cycle_retains_its_head_with_either_contrast_source(
        training_args, monkeypatch, source, contrast_weight):
    from src.training import trainer

    args = training_args
    args.cycle_global_mode = "projected"
    args.cycle_weight = 0.05
    args.contrast_weight = contrast_weight
    args.contrast_source = source
    validate_objectives(args)
    ds = SimpleNamespace(meta={
        "nodes": {"feature_dim": 5, "features": "log_phys"},
        "edges": {"feature_dim": 3, "features": "log", "pt_scale": "normalized"}})
    monkeypatch.setattr(trainer, "ensure_dataset_matches", lambda *a: None)
    model, spec, _, _ = trainer.build_model(
        args, ds, "cpu", logging.getLogger(__name__))
    assert spec.contrastive_projection_dim == args.contrast_projection_dim
    assert model.contrastive_projection[0].in_features == args.latent_dim
    if source == "eb2":
        assert model.contrastive_eb2_projection[0].in_features == args.hidden_dim
        assert model.contrastive_eb2_projection is not model.contrastive_projection
