"""
Data package for the jet graph pipeline.

Pipeline steps:
    python -m src.data.preprocess      HDF5 → selected-jet constituents
    python -m src.data.build_subjets   constituents → capped subjets
    python -m src.data.build_graph     point clouds → graph topology

"""

from .dataset import JetDataset
from . import shards

__all__ = ["JetDataset", "shards"]
