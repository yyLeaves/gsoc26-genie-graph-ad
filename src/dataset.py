"""Read split events as pairs of jet graphs for torch_geometric.loader.DataLoader.

Each item has two ``jets`` plus event-level ``source``, ``row``, ``truth``,
``weak_label`` and ``mjj`` (TeV, from the split). A batch contains two PyG
Batch objects: leading jets and subleading jets in the same event order.
"""

from pathlib import Path

import numpy as np
import tables
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


def _read_jet(root, index):
    start, stop = root.node_offsets[index:index+2]
    edge_start, edge_stop = root.edge_offsets[index:index+2]
    return Data(
        x=torch.from_numpy(root.x[start:stop]),
        pos=torch.from_numpy(root.pos[start:stop]),
        pt=torch.from_numpy(root.pt[start:stop]),
        edge_index=torch.from_numpy(root.edge_index[edge_start:edge_stop]).t().contiguous(),
        edge_attr=torch.from_numpy(root.edge_attr[edge_start:edge_stop]),
    )


class EventDataset(Dataset):
    """Load only split metadata up front; read graph slices as batches are requested."""

    def __init__(self, split_path, *, split='train', graph_dir='data/graphs'):
        with np.load(split_path) as file:
            self.events = file[split]
            self.graph_paths = [Path(graph_dir) / name for name in file['source_files']]

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        """Open each source once per batch; keep the requested order and duplicate indices.

        PyTorch calls this for batched loading. Handles stay local to the call,
        so DataLoader workers never share or pickle open HDF5 files.
        """
        events = self.events[indices]
        items = [None] * len(events)
        for source in np.unique(events['source']):
            path = self.graph_paths[source]
            with tables.open_file(path) as file:
                for index in np.flatnonzero(events['source'] == source):
                    event = events[index]
                    row = int(event['row'])
                    if not file.root.valid[row]:
                        raise ValueError(f'{path}: row {row} has fewer than two jets')
                    items[index] = {
                        'jets': (_read_jet(file.root, 2*row), _read_jet(file.root, 2*row+1)),
                        'source': int(source),
                        'row': row,
                        'truth': int(event['truth']),
                        'weak_label': int(event['weak_label']),
                        'mjj': float(event['mjj']),
                    }
        return items
