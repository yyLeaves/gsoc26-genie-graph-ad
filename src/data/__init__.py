"""
Data package for the jet graph pipeline.

Pipeline steps (not imported here — they pull in fastjet/pandas that
training-time code doesn't need):
    python -m src.data.preprocess     step 1: HDF5 → point-cloud shards
    python -m src.data.build_graph    step 2: point-cloud → graph shards

The single-jet primitive src.data.graph.build_graph (function) is not
re-exported, to keep it distinct from the build_graph pipeline-step module.
"""

from .graph import (
    FEATURE_DIM,
    build_edges,
    compute_node_features,
    knn_edges,
    laman_edges,
    radius_edges,
)
from .dataset import JetDataset
from . import kinematics, shards
