"""Reconstruction, cycle or classifier anomaly scores, summed over both event jets."""

import torch

from .calibration import jet_structure_features
from .losses import cycle_errors, netge_errors, reconstruction_errors
from .models import GLADC, NetGe, NetGeJet, SDMNAT


def scoring_options(model_name, *, variant='original'):
    """Default score and available alternatives for each model."""
    if model_name == 'sdm_nat':
        return 'classifier', ('total', 'classifier')
    if model_name in ('edgeconv', 'EdgeConvAE'):
        return 'total', ('total', 'node', 'edge')
    if model_name == 'netge' and variant in ('jet', 'fraction'):
        names = ('total', 'node', 'edge', 'reconstruction')
        if variant == 'jet':
            names += ('cycle', 'node_cycle', 'graph_cycle', 'reconstruction_cycle')
        return 'reconstruction', names
    names = ('total', 'node', 'cycle', 'node_cycle', 'graph_cycle')
    if model_name == 'netge':
        names += ('attribute_l2', 'structure')
    if model_name == 'gladc':
        names += ('edge', 'node_edge', 'node_edge_cycle')
    return ('cycle' if model_name == 'netge' else 'node_edge'), names


def _jet_scores(model, jets, edge_weight):
    if isinstance(model, NetGeJet):
        errors = model.scores(model.reconstruct(jets))
        return dict(errors, total=errors['reconstruction'])
    if isinstance(model, SDMNAT):
        # sigmoid(-logit) = 1 - sigmoid(logit), without subtraction cancellation.
        score = torch.sigmoid(-model.classify(jets))
        return dict(classifier=score, total=score)
    if isinstance(model, NetGe):
        output = model.reconstruct(jets)
        terms = netge_errors(output)
        x, _, _, mask = output['inputs']
        node = ((output['node'] - x).square().mean(-1) * mask).sum(1) / mask.sum(1)
        attribute_l2 = terms.pop('attribute')
        errors = dict(node=node, attribute_l2=attribute_l2, **terms)
    else:
        output = model(jets.x, jets.edge_index, jets.edge_attr)
        errors = reconstruction_errors(output, jets, edge_weight=edge_weight)
        if not isinstance(model, GLADC):
            return errors
        errors.update(cycle_errors(model, output, jets))
    errors['cycle'] = errors['node_cycle'] + errors['graph_cycle']
    if 'edge' in errors:
        errors['node_edge'] = errors['node'] + edge_weight * errors['edge']
        errors['node_edge_cycle'] = errors['node_edge'] + errors['cycle']
    errors['total'] = errors['cycle'] if isinstance(model, NetGe) else errors['node_edge']
    return errors


@torch.no_grad()
def score_events(model, batch, *, edge_weight=1.0, score=None):
    """Return event scores; total is the legacy alias for the selected anomaly score.

    It follows the model default when score is None, not the training objective.
    Call model.eval() first. Each score vector follows the event order in
    batch['jets'], matching batch['source'] and batch['row'].
    """
    errors = []
    for jets in batch['jets']:
        errors.append(_jet_scores(model, jets, edge_weight))
    leading, subleading = errors
    # Match the baseline's float64 event-sum accumulation.
    result = {name: leading[name].double() + subleading[name].double() for name in leading}
    if score is not None:
        result['total'] = result[score]
    return result


def score_loader(model, loader, device, *, edge_weight=1.0, score=None):
    """Return NumPy scores and event metadata in loader order, using eval mode."""
    model.eval()
    metadata = ('source', 'row', 'mjj', 'truth', 'weak_label')
    columns = {}
    for batch in loader:
        for jets in batch['jets']:
            jets.to(device)
        values = {name: batch[name] for name in metadata}
        values.update(score_events(model, batch, edge_weight=edge_weight, score=score))
        for name, value in values.items():
            columns.setdefault(name, []).append(value.cpu())
    return {name: torch.cat(values).numpy() for name, values in columns.items()}


@torch.no_grad()
def score_pair_loader(baseline, regularized, loader, device, *, baseline_edge_weight=1.0,
                      reg_edge_weight=1.0, baseline_score=None, reg_score=None):
    """Score two gate branches, allowing different architectures and scoring rules.

    Both models see the same events; keep both jets' structure features for gating.
    """
    baseline.eval()
    regularized.eval()
    metadata = ('source', 'row', 'mjj', 'truth', 'weak_label')
    columns = {name: [] for name in (*metadata, 'baseline', 'regularized', 'pr', 'radius')}
    for batch in loader:
        for jets in batch['jets']:
            jets.to(device)
        values = {name: batch[name] for name in metadata}
        values['baseline'] = score_events(baseline, batch, edge_weight=baseline_edge_weight,
                                          score=baseline_score)['total']
        values['regularized'] = score_events(regularized, batch, edge_weight=reg_edge_weight,
                                             score=reg_score)['total']
        features = [jet_structure_features(jets) for jets in batch['jets']]
        values['pr'] = torch.stack([part[0] for part in features], dim=1)
        values['radius'] = torch.stack([part[1] for part in features], dim=1)
        for name, value in values.items():
            columns[name].append(value.cpu())
    return {name: torch.cat(values).numpy() for name, values in columns.items()}
