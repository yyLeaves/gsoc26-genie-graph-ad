import torch

from src.data.iterate import prefetch, shard_iter
from src.eval.scoring import _event_score_arrays


def _graph_pt_sums(batch) -> torch.Tensor:
    """Raw Σ constituent pT per graph in a PyG Batch."""
    return torch.stack([
        batch.pt[batch.ptr[i]:batch.ptr[i + 1]].sum()
        for i in range(batch.num_graphs)
    ]).cpu()


@torch.no_grad()
def eval_loss_components(model, ds, indices, batch_size, device):
    """(total, c1, c2) averaged over indices via model.loss. (c1, c2) are the
    two components (node/edge for relae)."""
    model.eval()
    tot_total = tot_c1 = tot_c2 = n = 0.0
    for batch in prefetch(shard_iter(ds, indices, batch_size,
                                     shuffle_shards=False, shuffle_within=False,
                                     device=device)):
        total, c1, c2 = model.loss(batch)
        B = batch.num_graphs
        tot_total += total.item() * B
        tot_c1    += c1.item()    * B
        tot_c2    += c2.item()    * B
        n         += B
    return tot_total / n, tot_c1 / n, tot_c2 / n


@torch.no_grad()
def eval_scores(model, ds, indices, batch_size, device, aggregation="sum"):
    """(scores, labels) event-level numpy arrays via model.anomaly_score.

    Jet graph scores are summed by event_id, matching the reference repo's
    score(event) = score(jet0) + score(jet1).
    """
    if not getattr(ds, "has_event_ids", lambda: False)():
        raise ValueError(
            "Event-level eval requires event_ids in metadata. Rebuild point "
            "cloud and graph shards with the current preprocess/build_subset.")
    model.eval()
    scores_list, labels_list, event_ids_list, strength_list = [], [], [], []
    for batch in prefetch(shard_iter(ds, indices, batch_size,
                                     shuffle_shards=False, shuffle_within=False,
                                     device=device)):
        scores_list.append(model.anomaly_score(batch).cpu())
        labels_list.append(batch.y.squeeze(-1).cpu())
        event_ids_list.append(batch.event_id.squeeze(-1).cpu())
        if aggregation == "pt_weighted":
            strength_list.append(_graph_pt_sums(batch))
    strengths = torch.cat(strength_list).numpy() if strength_list else None
    scores, labels, _ = _event_score_arrays(
        torch.cat(scores_list).numpy(),
        torch.cat(event_ids_list).numpy(),
        torch.cat(labels_list).numpy(),
        strengths,
        aggregation)
    return scores, labels
