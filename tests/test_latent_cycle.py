import copy
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from src.data.graph import edge_features, unique_k_edges
from src.models import EdgeGraphAE
from src.models.reconstruction import mean_loss, reconstruction_scores
from src.training import trainer
from src.objectives.latent_cycle import latent_cycle_scores, select_cycle_component


def _batch():
    rng = np.random.default_rng(7)
    graphs = []
    for n_nodes in (8, 11):
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


def test_latent_cycle_reencodes_nodes_and_edges_with_gradients():
    torch.manual_seed(5)
    batch = _batch()
    model = EdgeGraphAE(in_dim=1, edge_dim=3)
    output = model(batch.x, batch.edge_index, batch.edge_attr)
    terms = latent_cycle_scores(model, output, batch)

    assert terms.total.shape == (batch.num_graphs,)
    assert terms.local.shape == terms.total.shape
    assert terms.global_.shape == terms.total.shape
    assert terms.reencoded.shape == output.latent.shape
    assert torch.allclose(terms.total, terms.local + terms.global_)
    assert torch.isfinite(terms.total).all()

    terms.total.mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients)


def test_select_cycle_component():
    torch.manual_seed(11)
    batch = _batch()
    model = EdgeGraphAE(in_dim=1, edge_dim=3)
    output = model(batch.x, batch.edge_index, batch.edge_attr)
    terms = latent_cycle_scores(model, output, batch)

    assert select_cycle_component(terms, "total") is terms.total
    assert select_cycle_component(terms, "local") is terms.local
    assert select_cycle_component(terms, "global") is terms.global_

    try:
        select_cycle_component(terms, "bad")
    except ValueError as error:
        assert "unknown cycle component" in str(error)
    else:
        raise AssertionError("invalid cycle component must fail")


def test_zero_cycle_weight_can_skip_the_second_forward_exactly():
    """The baseline branch need not call latent_cycle_scores at beta=0."""
    torch.manual_seed(5)
    batch = _batch()
    model = EdgeGraphAE(in_dim=1, edge_dim=3)
    output = model(batch.x, batch.edge_index, batch.edge_attr)
    baseline = output.node.detach().clone()

    cycle_weight = 0.0
    if cycle_weight:
        latent_cycle_scores(model, output, batch)
    assert torch.equal(output.node, baseline)


def test_zero_cycle_weight_matches_one_baseline_optimizer_step(monkeypatch, training_args):
    torch.manual_seed(19)
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

    def unexpected_cycle_call(*args, **kwargs):
        raise AssertionError("beta=0 must not execute the cycle branch")

    monkeypatch.setattr(trainer, "latent_cycle_scores", unexpected_cycle_call)

    class NoOpScheduler:
        def step(self):
            pass

    args = training_args
    splits = SimpleNamespace(train_idx=np.arange(batch.num_graphs))
    stats = trainer.train_one_epoch(
        candidate, object(), splits, candidate_opt, NoOpScheduler(), False,
        None, args, torch.device("cpu"), np.random.default_rng(3))

    assert stats.cycle == 0.0
    assert stats.total == reference_loss.item()
    assert all(
        torch.equal(candidate.state_dict()[name], value)
        for name, value in reference.state_dict().items()
    )
