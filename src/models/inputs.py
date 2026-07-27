import torch


def normalize_feature_cols(
    feature_cols: tuple[int, ...] | None,
    expected_dim: int,
) -> tuple[int, ...] | None:
    """Validate a serialized node-feature view."""
    if feature_cols is None:
        return None
    feature_cols = tuple(int(index) for index in feature_cols)
    if len(feature_cols) != expected_dim:
        raise ValueError(
            f"feature_cols has {len(feature_cols)} columns, but in_dim is "
            f"{expected_dim}"
        )
    if any(index < 0 for index in feature_cols):
        raise ValueError("feature_cols must contain non-negative indices")
    if len(set(feature_cols)) != len(feature_cols):
        raise ValueError("feature_cols must contain unique indices")
    return feature_cols


def select_node_features(
    batch,
    expected_dim: int,
    feature_cols: tuple[int, ...] | None,
) -> torch.Tensor:
    """Return exactly the node-feature view declared by the model."""
    x = getattr(batch, "x", None)
    if x is None:
        raise ValueError("model input requires batch.x")
    if feature_cols is None:
        if x.size(-1) != expected_dim:
            raise ValueError(
                f"batch.x has {x.size(-1)} features; model requires exactly "
                f"{expected_dim} because feature_cols=None"
            )
        return x
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")
    if max(feature_cols) >= x.size(-1):
        raise ValueError(
            f"batch.x has {x.size(-1)} features, but feature_cols requests "
            f"column {max(feature_cols)}"
        )
    return x[:, list(feature_cols)]


def require_edge_features(batch, expected_dim: int) -> torch.Tensor:
    """Return edge features after enforcing the model's exact width."""
    edge_attr = getattr(batch, "edge_attr", None)
    if edge_attr is None:
        raise ValueError("model input requires batch.edge_attr")
    if edge_attr.size(-1) != expected_dim:
        raise ValueError(
            f"batch.edge_attr has {edge_attr.size(-1)} features; model "
            f"requires exactly {expected_dim}")
    return edge_attr
