import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from src.models import EdgeGraphAE
from src.objectives.latent_cycle import latent_cycle_scores


def test_cycle_suite_reads_training_representation(tmp_path):
    from scripts.eval_cycle_suite import checkpoint_cycle_mode
    checkpoint = tmp_path/'last.pt'
    torch.save({'run_config': {'training': {'cycle_global_mode': 'projected'}}}, checkpoint)
    assert checkpoint_cycle_mode(checkpoint) == 'projected'
    torch.save({}, checkpoint)
    assert checkpoint_cycle_mode(checkpoint) == 'pooled'


@pytest.mark.parametrize('mode', ['pooled', 'projected'])
def test_cycle_component_evaluation_uses_requested_representation(monkeypatch, mode):
    from scripts import eval_cycle_components as script
    torch.manual_seed(123)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    graphs = [Data(x=torch.randn(3, 1), edge_index=edges,
                   edge_attr=torch.randn(4, 3), event_id=torch.tensor([0]),
                   y=torch.tensor([0])) for _ in range(2)]
    batch = Batch.from_data_list(graphs)
    model = EdgeGraphAE(in_dim=1, edge_dim=3, hidden_dim=8,
                        contrastive_projection_dim=8).eval()
    monkeypatch.setattr(script, 'shard_iter', lambda *args, **kwargs: iter([batch]))
    output, _, _ = model._reconstruct(batch)
    expected = latent_cycle_scores(model, output, batch, global_mode=mode).global_.sum()
    result = script.score_components(model, None, np.arange(2), 'cpu', 2, 'test', global_mode=mode)
    np.testing.assert_allclose(result['cycle_global'].scores, [expected.detach().item()], rtol=1e-6)
    from src.models.reconstruction import reconstruction_scores
    output, node, edge = model._reconstruct(batch)
    recon = reconstruction_scores(output, node, batch, edge_target=edge)
    np.testing.assert_allclose(result['node_mse'].scores, [recon.node.sum().item()], rtol=1e-6)
    np.testing.assert_allclose(result['edge_mse'].scores, [recon.edge.sum().item()], rtol=1e-6)
    np.testing.assert_allclose(result['reconstruction'].scores,
                               result['node_mse'].scores + result['edge_mse'].scores, rtol=1e-6)


def test_suite_saves_node_and_edge_metrics():
    from scripts.eval_cycle_suite import COMPONENTS, evaluate_parts
    from src.eval.scoring import EventScores
    assert 'node_mse' in COMPONENTS and 'edge_mse' in COMPONENTS
    def sample(label):
        return {name: EventScores(np.arange(10.) + label, np.full(10, label), np.arange(10))
                for name in COMPONENTS}
    metrics, labels, arrays = evaluate_parts(sample(0), sample(0), sample(1))
    for name in ('node_mse', 'edge_mse', 'reconstruction'):
        assert 'auc' in metrics[name] and 'max_sic' in metrics[name]
        assert len(arrays[name]) == len(labels) == 20
    np.testing.assert_allclose(
        arrays['reconstruction_plus_cycle'],
        arrays['reconstruction'] + arrays['cycle'])
