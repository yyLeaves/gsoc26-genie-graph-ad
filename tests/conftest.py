import sys

import pytest


@pytest.fixture
def training_args(monkeypatch):
    """Use the same defaults as the training command, not a partial namespace."""
    from scripts.train_graph_ae import parse_args

    monkeypatch.setattr(sys, "argv", ["train", "--data_dir", "graphs",
                                    "--output", "runs", "--split_protocol", "ks_fixed"])
    return parse_args()
